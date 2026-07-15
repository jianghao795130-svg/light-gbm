import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from core.model.factor_config import get_col_name
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger
from core.utils.meta_db import (
    META_DB_NAME,
    compute_file_hash,
    delete_factor_status_by_factor_name,
    init_meta_db,
    load_factor_status,
    load_meta_hashes,
    save_factor_status_batch,
    save_meta_hashes,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from core.model.backtest_config import BacktestConfig


def get_meta_db_path(factor_folder: Path) -> Path:
    """返回因子缓存 meta 数据库路径。"""
    return factor_folder / META_DB_NAME


def build_factor_col_to_name(hour_factor_params_dict: dict) -> Dict[str, str]:
    """构建 factor_col_name -> factor_name 的映射。"""
    col_to_factor_name: Dict[str, str] = {}
    for factor_name, params in hour_factor_params_dict.items():
        for param in params:
            col_name = get_col_name(factor_name, param)
            col_to_factor_name[col_name] = factor_name
    return col_to_factor_name


def sync_factor_cache(conn: sqlite3.Connection, factor_folder: Path, hour_factor_params_dict: dict) -> None:
    """检测因子代码变化 → 清理失效缓存 → 持久化最新 hash（原子步骤）。"""
    old_hashes = load_meta_hashes(conn)
    current_hashes: Dict[str, str] = {}
    changed_factor_names = set()
    changed_factor_hashes: List[Tuple[str, str, str]] = []
    for factor_name in hour_factor_params_dict:
        h = compute_file_hash(FactorHub.get_factor_file_path(factor_name))
        if h is not None:
            current_hashes[factor_name] = h
        old_hash = old_hashes.get(factor_name)
        new_hash = current_hashes.get(factor_name)
        if new_hash != old_hash:
            changed_factor_names.add(factor_name)
            # 首次运行或 hash 缺失时不提示，避免将初始化状态误解为“代码变化”
            if old_hash is not None and new_hash is not None:
                changed_factor_hashes.append((factor_name, old_hash, new_hash))

    if changed_factor_hashes:
        detail_lines = [
            f"  - {factor_name}: {old_hash[:8]} -> {new_hash[:8]}"
            for factor_name, old_hash, new_hash in sorted(changed_factor_hashes, key=lambda x: x[0])
        ]
        logger.info(
            "【因子缓存】检测到因子代码更新\n"
            f"- 变更数量: {len(changed_factor_hashes)}\n"
            "- 变更明细:\n" + "\n".join(detail_lines)
        )

    if changed_factor_names:
        col_to_factor_name = build_factor_col_to_name(hour_factor_params_dict)
        for factor_name in changed_factor_names:
            delete_factor_status_by_factor_name(conn, factor_name)
            for col_name, fn in col_to_factor_name.items():
                if fn == factor_name:
                    fp = factor_folder / f"factor_hour_v2_{col_name}.parquet"
                    fp.unlink(missing_ok=True)

    save_meta_hashes(conn, current_hashes)


def build_intraday_v2_cache_paths(
    factor_folder: Path, factor_col_name_list: List[str]
) -> List[Path]:
    """构建 intraday v2 缓存必需文件列表（kline + factor）。
    小时级因子全部是时序因子，不涉及截面因子。"""
    paths = [factor_folder / "all_hour_factors_kline_v2.parquet"]
    for factor_col_name in factor_col_name_list:
        paths.append(factor_folder / f"factor_hour_v2_{factor_col_name}.parquet")
    return paths


def _read_stock_parquet_max_time(file_path: Path) -> Optional[pd.Timestamp]:
    """读取单股票 parquet 的最新交易时间。

    优先从 row group metadata 的列统计信息中直接取 max，零行读取 O(1)。
    """
    try:
        meta = pq.read_metadata(file_path)
    except Exception:
        return None

    if meta.num_rows == 0:
        return None

    # 从所有 row group 的列统计中取全局最大值
    col_idx = None
    schema = pq.read_schema(file_path)
    for i in range(len(schema)):
        if schema.field(i).name == "交易日期":
            col_idx = i
            break
    if col_idx is None:
        return None

    global_max = None
    for rg in range(meta.num_row_groups):
        col_meta = meta.row_group(rg).column(col_idx)
        if not col_meta.is_stats_set:
            # 统计信息缺失，回退到读取整列
            return _read_stock_parquet_max_time_fallback(file_path)
        rg_max = col_meta.statistics.max
        if global_max is None or rg_max > global_max:
            global_max = rg_max

    if global_max is None:
        return None

    out_ts = pd.Timestamp(global_max)
    if out_ts.tzinfo is not None:
        out_ts = out_ts.tz_localize(None)
    return out_ts


def _read_stock_parquet_max_time_fallback(file_path: Path) -> Optional[pd.Timestamp]:
    """回退方案：读取完整列取 max（仅当 metadata 统计信息缺失时使用）。"""
    try:
        ts = pl.read_parquet(file_path, columns=["交易日期"]).select(pl.col("交易日期").max()).item()
    except Exception:
        return None

    if ts is None:
        return None

    out_ts = pd.Timestamp(ts)
    if out_ts.tzinfo is not None:
        out_ts = out_ts.tz_localize(None)
    return out_ts


def _resolve_parquet_max_time(
    file_path: Path,
    stock_factor_status: Dict[str, Tuple[str, Optional[float], Optional[int], Optional[str]]],
    plan_stats: Optional[Dict[str, int]],
) -> Optional[pd.Timestamp]:
    """获取源 parquet 的最新时间，优先用行内 mtime+size 的文件状态缓存避免 IO。

    从该股票所有因子行中取 data_max_time 最大的那行作为文件状态基准（最近一次计算时的文件状态）。
    返回值是源数据的 data_max_time（而非因子输出的 max_time）。
    """
    latest_file_state = None
    for _, file_mtime, file_size, data_max_time in stock_factor_status.values():
        if file_mtime is not None and data_max_time is not None and (
            latest_file_state is None
            or pd.Timestamp(data_max_time) > pd.Timestamp(latest_file_state[0])
        ):
            latest_file_state = (data_max_time, file_mtime, file_size)

    if latest_file_state is not None:
        try:
            st = file_path.stat()
            if st.st_mtime == latest_file_state[1] and st.st_size == latest_file_state[2]:
                ts = pd.Timestamp(latest_file_state[0])
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                if plan_stats is not None:
                    plan_stats["file_state_hit"] += 1
                return ts
        except OSError:
            pass

    if plan_stats is not None:
        plan_stats["file_state_miss"] += 1
    return _read_stock_parquet_max_time(file_path)


# 每个任务: (file_path, trim_before, lookback_bars, factor_col_names)
StockTask = Tuple[Path, Optional[pd.Timestamp], int, List[str]]


def plan_incremental_factors(
    file_list: list,
    factor_col_name_list: List[str],
    status: Dict[str, Dict[str, Tuple[str, Optional[float], Optional[int], Optional[str]]]],
    lookback_bars: int,
    missing_factor_cols: Optional[set] = None,
    plan_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, StockTask]:
    """制定增量计算计划（个股 × 时间 × 因子列 三维增量）。

    每个因子列独立分类：
      - v2 文件缺失 / meta 缺失 / meta 异常 → 全量 (trim_before=None)
      - 有 meta 且源文件有更新 → 增量 (trim_before=该因子的 max_time)
      - 源文件无更新 → 跳过
      - 源文件回退（data_max_time < meta.data_max_time）→ 全量重算

    每个 stock 最多一个任务，取最保守的 trim_before（min 或 None），
    parquet 只读一次，所有需计算因子一并完成。

    plan_stats 统计粒度为因子列级别。

    返回:
        Dict[stock_code, (file_path, effective_trim_before, lookback_bars, factor_cols_to_compute)]
    """
    missing_factor_cols = missing_factor_cols or set()
    stock_plan_map: Dict[str, StockTask] = {}

    # 按因子列级别追踪更新模式: 0=skip, 1=incremental, 2=full
    col_max_mode: Dict[str, int] = {col: 0 for col in factor_col_name_list}

    if plan_stats is not None:
        for key in ("full_missing_file", "full_missing_meta", "full_invalid_meta",
                    "full_data_rollback", "incremental", "latest_skip",
                    "file_state_hit", "file_state_miss"):
            plan_stats.setdefault(key, 0)

    for file_path in file_list:
        stock_code = file_path.stem
        stock_status = status.get(stock_code, {})

        # ---- 每个因子列独立分类 ----
        full_cols: List[str] = []                      # 需要全量重算
        incr_candidates: Dict[str, pd.Timestamp] = {}  # 因子列 → 该因子的 factor max_time
        data_max_times: Dict[str, Optional[str]] = {}  # 因子列 → 该因子的 data_max_time

        for col in factor_col_name_list:
            # 1) v2 parquet 文件不存在 → 全量
            if col in missing_factor_cols:
                full_cols.append(col)
                col_max_mode[col] = 2
                if plan_stats is not None:
                    plan_stats["full_missing_file"] += 1
                continue

            # 2) 该股票该因子无 meta 状态 → 全量
            col_status = stock_status.get(col)
            if col_status is None:
                full_cols.append(col)
                col_max_mode[col] = 2
                if plan_stats is not None:
                    plan_stats["full_missing_meta"] += 1
                continue

            # 3) max_time 为空或异常 → 全量
            max_time_str = col_status[0]
            if max_time_str is None:
                full_cols.append(col)
                col_max_mode[col] = 2
                if plan_stats is not None:
                    plan_stats["full_invalid_meta"] += 1
                continue

            ts = pd.Timestamp(max_time_str)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            incr_candidates[col] = ts
            data_max_times[col] = col_status[3]  # data_max_time

        # ---- 增量候选：逐因子检查源文件是否有更新 ----
        incr_cols: List[str] = []
        if incr_candidates:
            parquet_max_time = _resolve_parquet_max_time(file_path, stock_status, plan_stats)
            for col, ts in incr_candidates.items():
                if parquet_max_time is not None and parquet_max_time <= ts:
                    # 检测源数据回退：当前 parquet_max_time < 上次记录的 data_max_time
                    old_data_mt = data_max_times.get(col)
                    if old_data_mt is not None and parquet_max_time < pd.Timestamp(old_data_mt):
                        # 源数据被替换为更早的版本，强制全量重算
                        full_cols.append(col)
                        col_max_mode[col] = 2
                        if plan_stats is not None:
                            plan_stats["full_data_rollback"] += 1
                    else:
                        # 该因子已最新，跳过
                        if plan_stats is not None:
                            plan_stats["latest_skip"] += 1
                else:
                    incr_cols.append(col)
                    col_max_mode[col] = max(col_max_mode[col], 1)
                    if plan_stats is not None:
                        plan_stats["incremental"] += 1

        # ---- 合并为单一任务：取最保守的 trim_before，parquet 只读一次 ----
        compute_cols = full_cols + incr_cols
        if not compute_cols:
            continue  # 该股票所有因子均已最新

        # 有全量因子 → trim_before=None；否则取增量因子的 min(max_time)
        effective_trim = None if full_cols else min(incr_candidates[col] for col in incr_cols)
        stock_plan_map[stock_code] = (file_path, effective_trim, int(lookback_bars), compute_cols)

    # 汇总因子列级别的更新模式统计
    if plan_stats is not None:
        plan_stats["col_skip"] = sum(1 for v in col_max_mode.values() if v == 0)
        plan_stats["col_incremental"] = sum(1 for v in col_max_mode.values() if v == 1)
        plan_stats["col_full"] = sum(1 for v in col_max_mode.values() if v == 2)

    return stock_plan_map


def prepare_intraday_plan(
    conf: "BacktestConfig",
    file_list: list,
) -> Tuple[Dict[str, StockTask], Dict[str, int]]:
    """统一入口：同步缓存 + 刚性文件检查 + 增量计划。
    小时级因子全部是时序因子，不涉及截面因子。"""
    factor_folder = conf.get_factor_folder()
    hour_factor_params_dict = conf.hour_factor_params_dict
    factor_col_name_list = conf.hour_factor_col_name_list
    lookback_bars = conf.intraday_incremental_lookback

    conn = init_meta_db(get_meta_db_path(factor_folder))
    try:
        sync_factor_cache(conn, factor_folder, hour_factor_params_dict)
        status = load_factor_status(conn)
    finally:
        conn.close()
    required_v2_paths = build_intraday_v2_cache_paths(factor_folder, factor_col_name_list)
    missing_v2_paths = [p for p in required_v2_paths if not p.exists()]

    if missing_v2_paths:
        sample = ", ".join([p.name for p in missing_v2_paths[:3]])
        logger.warning(
            "【因子缓存】检测到 v2 缓存文件缺失\n"
            f"- 缺失数量: {len(missing_v2_paths)}/{len(required_v2_paths)}\n"
            f"- 示例文件: {sample}\n"
            "- 处理策略: 缺失因子全量重算，其余因子继续增量"
        )

    kline_path = factor_folder / "all_hour_factors_kline_v2.parquet"
    if not kline_path.exists():
        # kline 缺失意味着需要重建所有因子（kline 是所有因子共享的行情基础）
        missing_factor_cols = set(factor_col_name_list)
    else:
        missing_factor_cols = set()
        for p in missing_v2_paths:
            name = p.stem  # e.g. "factor_hour_v2_N日均价_28"
            prefix = "factor_hour_v2_"
            if name.startswith(prefix):
                missing_factor_cols.add(name[len(prefix):])

    plan_stats: Dict[str, int] = {}
    stock_plan_map = plan_incremental_factors(
        file_list,
        factor_col_name_list,
        status,
        lookback_bars,
        missing_factor_cols=missing_factor_cols,
        plan_stats=plan_stats,
    )
    return stock_plan_map, plan_stats


def save_intraday_factor_meta(
    conf: "BacktestConfig",
    worker_meta: list,
    stock_file_map: Optional[Dict[str, Path]] = None,
) -> None:
    """统一入口：写入 factor_status（含 mtime/size 文件状态缓存）。

    worker_meta 每个条目格式: (stock_code, data_max_time, factor_max_time, computed_cols)
    - data_max_time: 源 K 线 parquet 的原始最大日期（不受 end_date 裁切）
    - factor_max_time: 因子实际输出的最大日期（受 end_date 裁切）
    仅为实际计算的因子列写入状态，避免覆盖未参与计算的因子的增量起点。
    小时级因子全部是时序因子，不涉及截面因子。
    """
    factor_folder = conf.get_factor_folder()
    hour_factor_params_dict = conf.hour_factor_params_dict

    status_rows = []
    col_to_factor_name = build_factor_col_to_name(hour_factor_params_dict)
    for meta in worker_meta:
        if meta is None:
            continue
        stock_code, data_max_time, factor_max_time, computed_cols = meta
        if data_max_time is None:
            continue
        file_mtime, file_size = None, None
        if stock_file_map is not None:
            fp = stock_file_map.get(stock_code)
            if fp is not None:
                try:
                    st = fp.stat()
                    file_mtime, file_size = st.st_mtime, st.st_size
                except OSError:
                    pass
        # max_time 字段存储 factor_max_time（因子输出时间），data_max_time 字段存储源数据时间
        effective_max_time = factor_max_time or data_max_time
        for factor_col in computed_cols:
            fn = col_to_factor_name.get(factor_col, "")
            status_rows.append((stock_code, factor_col, effective_max_time, fn, file_mtime, file_size, data_max_time))
    conn = init_meta_db(get_meta_db_path(factor_folder))
    try:
        save_factor_status_batch(conn, status_rows)
    finally:
        conn.close()
