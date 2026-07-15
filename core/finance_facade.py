

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.model.backtest_config import BacktestConfig
from core.model.finance_manager_v5_fastX_H import create_factor_finance_manager_precomputed

# 模块级常量：允许的财务派生后缀白名单
_ALLOWED_DERIVED_SUFFIXES = frozenset({
    "_单季",
    "_单季环比",
    "_单季同比",
    "_ttm",
    "_ttm同比",
    "_累计同比",
    "_环比",  # B_ 截面字段专用
    "_同比",  # B_ 截面字段专用
})

# 流量型视图后缀（B_ 截面字段不支持）
_FLOW_VIEW_SUFFIXES = frozenset({
    "_单季",
    "_单季环比",
    "_单季同比",
    "_ttm",
    "_ttm同比",
    "_累计同比",
})

# 截面型视图后缀（R_/C_ 流量字段不支持）
_CROSS_VIEW_SUFFIXES = frozenset({
    "_环比",
    "_同比",
})


@lru_cache(maxsize=32)
def _normalize_base_fin_cols_cached(fin_cols: tuple) -> tuple:
    """
    将 fin_cols（允许包含派生后缀）规约为 V5 需要的基础列（xxx@xbx）。
    使用 lru_cache 缓存结果，同一进程内所有股票共享。
    """
    base_cols: List[str] = []
    seen: set = set()

    for col in fin_cols:
        if not isinstance(col, str) or not col:
            raise ValueError(f"财务字段名非法（必须为非空字符串）：{col!r}")

        at = col.find("@xbx")
        if at < 0:
            raise ValueError(f"财务字段名必须包含'@xbx'：{col}")

        base = col[:at] + "@xbx"
        suffix = col[at + len("@xbx"):]  # '' 或 '_单季' 等

        if suffix and suffix not in _ALLOWED_DERIVED_SUFFIXES:
            raise ValueError(
                f"不支持的财务派生后缀：{col}；允许的后缀为：{sorted(_ALLOWED_DERIVED_SUFFIXES)}"
            )

        if base not in seen:
            seen.add(base)
            base_cols.append(base)

    return tuple(base_cols)


def _normalize_base_fin_cols(fin_cols: List[str]) -> List[str]:
    """
    将 conf.fin_cols（允许包含派生后缀）规约为 V5 需要的基础列（xxx@xbx）。
    内部使用缓存版本，对外返回 list。
    """
    return list(_normalize_base_fin_cols_cached(tuple(fin_cols)))


def _ensure_trade_date_datetime(candle_df: pd.DataFrame) -> None:
    """
    确保 candle_df['交易日期'] 为 datetime64[ns]，并且不包含 NaT。
    这是 V5 对齐逻辑（searchsorted）正确性的必要前提。
    """
    if "交易日期" not in candle_df.columns:
        raise KeyError("candle_df 缺少必要列：'交易日期'")

    s = candle_df["交易日期"]
    # 使用 dtype.kind == 'M' 判断是否为 datetime 类型（比 pd.api.types 更快）
    if getattr(s.dtype, "kind", None) == "M":
        # 已是 datetime 类型，但仍需检查 NaT
        if not candle_df.empty and s.isna().any():
            raise ValueError("'交易日期' 存在 NaT 值，请先清洗K线数据。")
        return

    candle_df["交易日期"] = pd.to_datetime(s, errors="coerce")

    # 空表允许通过；非空表若存在 NaT，直接报错（避免静默错位）
    if not candle_df.empty and candle_df["交易日期"].isna().any():
        raise ValueError("'交易日期' 存在无法解析的值，请先清洗K线数据。")


def _split_base_and_suffix(fin_col: str) -> Tuple[str, str]:
    """拆分财务字段为基础列和后缀"""
    at = fin_col.find("@xbx")
    if at < 0:
        raise ValueError(f"财务字段名必须包含'@xbx'：{fin_col}")
    base = fin_col[:at] + "@xbx"
    suffix = fin_col[at + len("@xbx"):]  # '' 或 '_单季' 等
    return base, suffix


def _merge_finance_cols_to_df(conf, stock_df: pd.DataFrame, fin_obj) -> pd.DataFrame:
    """
    将 conf.fin_cols（含派生后缀）对应的财务列，使用 V5 API 计算并写入 stock_df。

    fin_obj 只需支持：fin_obj[base_col].raw()/quarter()/ttm() 以及这些返回值的 yoy()/qoq()
    （V5FinanceCompat 或 v5_mgr 均可）。
    """
    if not getattr(conf, "fin_cols", None):
        return stock_df

    # 缓存：避免同一 base_col 重复创建 proxy / 重复取视图
    proxy_cache: Dict[str, object] = {}
    view_cache: Dict[Tuple[str, str], pd.Series] = {}  # (base, view) -> Series

    for out_col in conf.fin_cols:
        if not isinstance(out_col, str) or not out_col:
            raise ValueError(f"财务字段名非法（必须为非空字符串）：{out_col!r}")

        base, suffix = _split_base_and_suffix(out_col)

        if suffix and suffix not in _ALLOWED_DERIVED_SUFFIXES:
            raise ValueError(f"不支持的财务派生后缀：{out_col}")

        # B_ 截面字段不支持单季/ttm视图
        if base.startswith("B_") and suffix in _FLOW_VIEW_SUFFIXES:
            raise ValueError(f"截面字段不支持该派生视图：{out_col}")

        # R_/C_ 流量字段不支持 _环比/_同比（应使用 _单季环比/_单季同比）
        if (base.startswith("R_") or base.startswith("C_")) and suffix in _CROSS_VIEW_SUFFIXES:
            raise ValueError(f"流量字段不支持该派生视图（请使用 _单季环比/_单季同比）：{out_col}")

        proxy = proxy_cache.get(base)
        if proxy is None:
            proxy = fin_obj[base]
            proxy_cache[base] = proxy

        if suffix == "":
            s = view_cache.get((base, "raw"))
            if s is None:
                s = proxy.raw()
                view_cache[(base, "raw")] = s

        elif suffix == "_累计同比":
            raw_s = view_cache.get((base, "raw"))
            if raw_s is None:
                raw_s = proxy.raw()
                view_cache[(base, "raw")] = raw_s
            s = raw_s.yoy()

        elif suffix == "_单季":
            s = view_cache.get((base, "quarter"))
            if s is None:
                s = proxy.quarter()
                view_cache[(base, "quarter")] = s

        elif suffix == "_单季环比":
            q = view_cache.get((base, "quarter"))
            if q is None:
                q = proxy.quarter()
                view_cache[(base, "quarter")] = q
            s = q.qoq()

        elif suffix == "_单季同比":
            q = view_cache.get((base, "quarter"))
            if q is None:
                q = proxy.quarter()
                view_cache[(base, "quarter")] = q
            s = q.yoy()

        elif suffix == "_ttm":
            s = view_cache.get((base, "ttm"))
            if s is None:
                s = proxy.ttm()
                view_cache[(base, "ttm")] = s

        elif suffix == "_ttm同比":
            t = view_cache.get((base, "ttm"))
            if t is None:
                t = proxy.ttm()
                view_cache[(base, "ttm")] = t
            s = t.yoy()

        elif suffix == "_环比":
            # B_ 截面字段专用：本季 / 上季 - 1
            raw_s = view_cache.get((base, "raw"))
            if raw_s is None:
                raw_s = proxy.raw()
                view_cache[(base, "raw")] = raw_s
            s = raw_s.qoq()

        elif suffix == "_同比":
            # B_ 截面字段专用：本季 / 去年同季 - 1
            raw_s = view_cache.get((base, "raw"))
            if raw_s is None:
                raw_s = proxy.raw()
                view_cache[(base, "raw")] = raw_s
            s = raw_s.yoy()

        else:
            raise RuntimeError(f"未覆盖的派生后缀分支：{out_col}")

        # 写入（长度不一致说明 trade_date_df/index 契约被破坏，应立即暴露）
        if len(s) != len(stock_df):
            raise ValueError(
                f"财务序列长度与行情不一致：col={out_col}, fin_len={len(s)}, df_len={len(stock_df)}"
            )
        stock_df[out_col] = s.to_numpy(copy=False)

    return stock_df


def _merge_finance_meta_to_df(candle_df: pd.DataFrame, v5_mgr) -> pd.DataFrame:
    """
    将财务元数据（report_date, publish_date）合并到 candle_df。

    语义：截至每个交易日，全局最新可见财报的 report_date 和 publish_date。
    - report_date: 最新可见财报的报告期截止日（如 2023-12-31）
    - publish_date: 对齐用发布日期（与 V5 其他 API 语义一致，可能包含理论法定截止日填充）
    """
    raw_map_df = v5_mgr._raw_map_df
    if raw_map_df.empty:
        candle_df['report_date'] = pd.NaT
        candle_df['publish_date'] = pd.NaT
        return candle_df

    # 过滤无效数据 + 同一 publish_date 保留最新季度（_raw_map_df 已按 publish_date, report_quarter 升序）
    meta_df = (
        raw_map_df
        .dropna(subset=['publish_date', 'report_date'])
        .drop_duplicates('publish_date', keep='last')
        [['publish_date', 'report_date']]
    )

    if meta_df.empty:
        candle_df['report_date'] = pd.NaT
        candle_df['publish_date'] = pd.NaT
        return candle_df

    # searchsorted 对齐：找 <= 交易日 的最近 publish_date
    trade_dates_int = candle_df['交易日期'].to_numpy(copy=False).view(np.int64)
    pub_dates_int = meta_df['publish_date'].to_numpy(copy=False).view(np.int64)

    indices = np.searchsorted(pub_dates_int, trade_dates_int, side='right') - 1

    # 边界处理：indices < 0 表示该交易日之前无任何财报发布
    valid_mask = indices >= 0

    n = len(candle_df)
    report_dates = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
    publish_dates = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')

    if valid_mask.any():
        valid_indices = indices[valid_mask]
        report_dates[valid_mask] = meta_df['report_date'].to_numpy(copy=False)[valid_indices]
        publish_dates[valid_mask] = meta_df['publish_date'].to_numpy(copy=False)[valid_indices]

    candle_df['report_date'] = report_dates
    candle_df['publish_date'] = publish_dates

    return candle_df


class V5FinanceCompat:
    """
    V5 管理器适配器：尽量复用 V5 计算能力，同时补齐旧因子可能访问的属性。

    兼容点：
    - 支持 fin_obj[col] -> 返回 V5 的 series 对象（含 raw/quarter/ttm/yoy/qoq 等API）
    - 提供 raw_fin_df / pivot_dict 属性（少量旧逻辑/调试代码可能会访问）
    """
    __slots__ = ("_mgr", "_pivot_dict_cache")

    def __init__(self, v5_fin_mgr):
        self._mgr = v5_fin_mgr
        self._pivot_dict_cache: Optional[Dict[str, pd.DataFrame]] = None

    @property
    def raw_fin_df(self) -> Optional[pd.DataFrame]:
        # V5 预计算对象里保存了清洗后的 raw_df；无财务数据时可能为空表
        precomputed = getattr(self._mgr, "precomputed", None)
        return getattr(precomputed, "raw_df", None)

    @property
    def pivot_dict(self) -> Dict[str, pd.DataFrame]:
        """
        以旧版 FinanceDataFrame 的契约形式提供 pivot_dict：
        - col -> base_pivot
        - f"{col}_is_na" -> is_na_mask
        """
        if self._pivot_dict_cache is not None:
            return self._pivot_dict_cache

        precomputed = getattr(self._mgr, "precomputed", None)
        pivots = getattr(precomputed, "pivots", {}) if precomputed is not None else {}
        is_na_pivots = getattr(precomputed, "is_na_pivots", {}) if precomputed is not None else {}
        fin_cols = getattr(precomputed, "fin_cols", []) if precomputed is not None else []

        d: Dict[str, pd.DataFrame] = {}
        for col in fin_cols:
            base_pivot = pivots.get(col)
            if base_pivot is not None:
                d[col] = base_pivot
                is_na = is_na_pivots.get(col)
                if is_na is not None:
                    d[f"{col}_is_na"] = is_na

        self._pivot_dict_cache = d
        return d

    def __getitem__(self, col: str):
        return self._mgr[col]

    def __getattr__(self, name: str):
        # 透传 V5 管理器的其它能力/属性（例如 raw_fin_cols / trade_date_df / stock_code 等）
        return getattr(self._mgr, name)


def prepare_finance_context(
    conf: BacktestConfig, stock_code: str, candle_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[Dict[str, object]]]:
    """
    统一财务数据入口（纯V5）：
    - 创建 V5 管理器
    - 使用 V5 API 将 conf.fin_cols 对应列合并进 candle_df（兼容旧因子直接读 df 列）
    - 返回 fin_data（兼容新因子走热路径）

    返回：
    - candle_df：已合并财务列
    - fin_data：{"财务数据对象": V5FinanceCompat, "v5_fin_mgr": v5_mgr}；无财务字段则返回 None
    """
    if not getattr(conf, "fin_cols", None):
        return candle_df, None

    if not isinstance(stock_code, str) or not stock_code:
        raise ValueError(f"stock_code 非法：{stock_code!r}")

    _ensure_trade_date_datetime(candle_df)
    base_cols = _normalize_base_fin_cols(list(conf.fin_cols))
    if not base_cols:
        raise ValueError("conf.fin_cols 非空但无法规约出任何基础财务字段，请检查配置。")

    trade_date_df = candle_df[["交易日期"]]
    v5_mgr = create_factor_finance_manager_precomputed(stock_code, base_cols, trade_date_df)

    fin_obj = V5FinanceCompat(v5_mgr)

    # 将所需财务列合并到 K 线（旧因子直接 df['xxx@xbx_单季同比'] 也能工作）
    candle_df = _merge_finance_cols_to_df(conf, candle_df, fin_obj)

    # 将财务元数据（report_date, publish_date）合并到 K 线
    candle_df = _merge_finance_meta_to_df(candle_df, v5_mgr)

    fin_data: Dict[str, object] = {"财务数据对象": fin_obj, "v5_fin_mgr": v5_mgr}
    return candle_df, fin_data
