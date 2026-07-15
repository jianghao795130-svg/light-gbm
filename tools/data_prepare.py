from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataColumns:
    trade_date: str = "交易日期"
    stock_code: str = "股票代码"
    stock_name: str = "股票名称"
    next_is_trade: str = "下日_是否交易"
    next_open_limit: str = "下日_开盘涨停"
    next_is_st: str = "下日_是否ST"
    next_is_delist: str = "下日_是否退市"
    next_one_word_limit: str = "下日_一字涨停"
    listed_days: str = "上市至今交易天数"
    row_id: str = "原始行号"


COLS = DataColumns()
INFO_COLS = [
    "股票名称",
    "复权因子",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "成交额",
    "是否交易",
    "流通市值",
    "总市值",
    "下日_是否交易",
    "下日_开盘涨停",
    "下日_是否ST",
    "下日_是否退市",
    "下日_是否涨停",
    "下日_一字涨停",
    "上市至今交易天数",
]


def find_data_dir(base_dir: Path) -> Path:
    """自动定位包含 all_factors_kline.pkl 的数据目录。"""
    base_dir = Path(base_dir)

    # 兼容旧结构：数据目录直接位于项目根目录下一层。
    direct_candidates = [
        candidate
        for candidate in base_dir.iterdir()
        if candidate.is_dir() and (candidate / "all_factors_kline.pkl").exists()
    ]
    if direct_candidates:
        return direct_candidates[0]

    # 兼容新结构：优先在 data 目录下递归查找。
    search_roots = [base_dir / "data", base_dir]
    all_candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for pkl_path in root.rglob("all_factors_kline.pkl"):
            parent = pkl_path.parent.resolve()
            if parent not in seen:
                seen.add(parent)
                all_candidates.append(parent)

    if not all_candidates:
        raise FileNotFoundError(
            f"没有找到包含 all_factors_kline.pkl 的数据目录。已检查: {base_dir} 和 {base_dir / 'data'}"
        )

    def _candidate_rank(path: Path) -> tuple[int, float]:
        try:
            mtime = (path / "all_factors_kline.pkl").stat().st_mtime
        except OSError:
            mtime = 0.0
        in_data_dir = 0 if (base_dir / "data") in path.parents else 1
        return (in_data_dir, -mtime)

    all_candidates.sort(key=_candidate_rank)
    return all_candidates[0]


def factor_name_from_path(path: Path) -> str:
    """把 factor_xxx.pkl 转成列名 xxx。"""
    return path.stem.replace("factor_", "", 1)


def load_kline(base_dir: Path | str) -> tuple[pd.DataFrame, Path]:
    """加载主表。"""
    base_dir = Path(base_dir)
    data_dir = find_data_dir(base_dir)
    kline = pd.read_pickle(data_dir / "all_factors_kline.pkl")
    return kline, data_dir


def build_pre_filter_mask(kline: pd.DataFrame) -> pd.Series:
    """构建官方风格的预过滤掩码。"""
    is_st_today = kline[COLS.stock_name].astype(str).str.contains("ST", case=False, na=False)
    listed_days_lt_250 = kline[COLS.listed_days].fillna(0) < 250
    next_not_trade = kline[COLS.next_is_trade].fillna(0).ne(1)
    cannot_buy_next_day = (
        kline[COLS.next_open_limit].fillna(0).eq(1)
        | kline[COLS.next_is_st].fillna(0).eq(1)
        | kline[COLS.next_is_delist].fillna(0).eq(1)
        | kline[COLS.next_one_word_limit].fillna(0).eq(1)
    )
    return ~(is_st_today | listed_days_lt_250 | next_not_trade | cannot_buy_next_day)


def summarize_pre_filter(kline: pd.DataFrame) -> pd.Series:
    """输出预过滤统计，便于 notebook 快速检查。"""
    is_st_today = kline[COLS.stock_name].astype(str).str.contains("ST", case=False, na=False)
    listed_days_lt_250 = kline[COLS.listed_days].fillna(0) < 250
    next_not_trade = kline[COLS.next_is_trade].fillna(0).ne(1)
    cannot_buy_next_day = (
        kline[COLS.next_open_limit].fillna(0).eq(1)
        | kline[COLS.next_is_st].fillna(0).eq(1)
        | kline[COLS.next_is_delist].fillna(0).eq(1)
        | kline[COLS.next_one_word_limit].fillna(0).eq(1)
    )
    keep_mask = ~(is_st_today | listed_days_lt_250 | next_not_trade | cannot_buy_next_day)
    return pd.Series(
        {
            "原始样本数": len(kline),
            "名称含ST样本数": int(is_st_today.sum()),
            "上市未满250天样本数": int(listed_days_lt_250.sum()),
            "次日不交易样本数": int(next_not_trade.sum()),
            "次日无法买入样本数": int(cannot_buy_next_day.sum()),
            "预过滤后保留样本数": int(keep_mask.sum()),
            "预过滤删除样本数": int((~keep_mask).sum()),
        }
    )


def pre_filter(kline: pd.DataFrame) -> pd.DataFrame:
    """执行预过滤，并保留原始行号用于和因子逐行对齐。"""
    keep_mask = build_pre_filter_mask(kline)
    keep_cols = [COLS.trade_date, COLS.stock_code]
    keep_cols += [col for col in INFO_COLS if col not in keep_cols]
    base_df = kline.loc[keep_mask, keep_cols].copy()
    base_df.insert(0, COLS.row_id, base_df.index)
    return base_df
