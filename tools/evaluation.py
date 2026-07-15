from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from tools.data_prepare import COLS
from tools.plot_config import setup_chinese_matplotlib


BASELINE_COL = "全市场等权"
setup_chinese_matplotlib()


def calc_rank_ic(g: pd.DataFrame) -> float:
    if g["ml_factor"].nunique(dropna=True) < 2 or g["raw_return"].nunique(dropna=True) < 2:
        return np.nan
    ic, _ = spearmanr(g["ml_factor"], g["raw_return"], nan_policy="omit")
    return ic


def calc_pearson_ic(g: pd.DataFrame) -> float:
    clean = g[["ml_factor", "raw_return"]].dropna()
    if len(clean) < 2:
        return np.nan
    ic, _ = pearsonr(clean["ml_factor"], clean["raw_return"])
    return ic


def evaluate_ic(pred_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    daily_rank_ic = pred_df.groupby(COLS.trade_date).apply(calc_rank_ic).dropna()
    daily_pearson_ic = pred_df.groupby(COLS.trade_date).apply(calc_pearson_ic).dropna()

    summary_df = pd.DataFrame(
        [
            {
                "指标": "Rank IC (Spearman)",
                "均值": daily_rank_ic.mean(),
                "标准差": daily_rank_ic.std(),
                "IR": daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else np.nan,
                "胜率": (daily_rank_ic > 0).mean(),
            },
            {
                "指标": "Normal IC (Pearson)",
                "均值": daily_pearson_ic.mean(),
                "标准差": daily_pearson_ic.std(),
                "IR": daily_pearson_ic.mean() / daily_pearson_ic.std() if daily_pearson_ic.std() > 0 else np.nan,
                "胜率": (daily_pearson_ic > 0).mean(),
            },
        ]
    )
    return daily_rank_ic, daily_pearson_ic, summary_df


def plot_ic_curves(
    daily_rank_ic: pd.Series,
    daily_pearson_ic: pd.Series,
    save_path: str | Path,
) -> Path:
    save_path = Path(save_path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    cum_rank_ic = daily_rank_ic.cumsum()
    cum_pearson_ic = daily_pearson_ic.cumsum()

    axes[0].plot(cum_rank_ic.index, cum_rank_ic.values, label="Rank IC", linewidth=1.5)
    axes[0].plot(cum_pearson_ic.index, cum_pearson_ic.values, label="Pearson IC", linewidth=1.5)
    axes[0].axhline(0, linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("累计 IC 曲线")
    axes[0].set_ylabel("累计IC")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in daily_rank_ic.fillna(0)]
    axes[1].bar(daily_rank_ic.index, daily_rank_ic.values, color=colors, width=1)
    axes[1].axhline(0, linestyle="--", color="gray", linewidth=1)
    axes[1].axhline(daily_rank_ic.mean(), linestyle="--", color="#2980b9", linewidth=1)
    axes[1].set_title("每日 RankIC")
    axes[1].set_ylabel("RankIC")
    axes[1].set_xlabel("日期")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def evaluate_top_n(pred_df: pd.DataFrame, top_n_list: list[int], holding_days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_returns: dict[str, pd.Series] = {}
    for n in top_n_list:
        daily_returns[f"Top{n}"] = pred_df.groupby(COLS.trade_date).apply(
            lambda g: g.nlargest(n, "ml_factor")["raw_return"].mean()
        )
    daily_returns["全市场等权"] = pred_df.groupby(COLS.trade_date)["raw_return"].mean()

    daily_returns_df = pd.DataFrame(daily_returns)
    rebalance_dates = daily_returns_df.index.sort_values()[::holding_days]
    rebalance_returns = daily_returns_df.loc[rebalance_dates]
    return daily_returns_df, rebalance_returns


def export_top_n_history(
    pred_df: pd.DataFrame,
    top_n_list: list[int],
    holding_days: int,
    save_path: str | Path,
) -> pd.DataFrame:
    daily_dates = pred_df[COLS.trade_date].drop_duplicates().sort_values()
    rebalance_dates = set(daily_dates.iloc[::holding_days].tolist())
    rebalance_df = pred_df[pred_df[COLS.trade_date].isin(rebalance_dates)].copy()

    records: list[dict] = []
    for date, group in rebalance_df.groupby(COLS.trade_date):
        sorted_group = group.sort_values("ml_factor", ascending=False)
        for n in top_n_list:
            top_df = sorted_group.head(n)
            for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
                records.append(
                    {
                        "交易日期": date,
                        "组合": f"Top{n}",
                        "排名": rank,
                        "股票代码": row[COLS.stock_code],
                        "未来n日回报": row["raw_return"],
                        "因子值": row["ml_factor"],
                    }
                )

    result = pd.DataFrame(records)
    save_path = Path(save_path)
    result.to_csv(save_path, index=False, encoding="utf-8-sig")
    return result


def build_top_n_performance_table(rebalance_returns: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    test_trading_days = len(rebalance_returns.index)
    rows = []
    for col in rebalance_returns.columns:
        ret_series = rebalance_returns[col].dropna()
        cum = (1 + ret_series).cumprod()
        total_ret = cum.iloc[-1] - 1 if len(cum) else np.nan
        ann_ret = (1 + total_ret) ** (250 / test_trading_days) - 1 if test_trading_days > 0 and pd.notna(total_ret) else np.nan
        ann_vol = ret_series.std() * np.sqrt(250 / holding_days) if len(ret_series) > 1 else np.nan
        sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
        max_dd = (cum / cum.cummax() - 1).min() if len(cum) else np.nan

        rows.append(
            {
                "组合": col,
                "累计收益": total_ret,
                "年化收益": ann_ret,
                "年化波动": ann_vol,
                "Sharpe": sharpe,
                "最大回撤": max_dd,
                "胜率": (ret_series > 0).mean() if len(ret_series) else np.nan,
                f"场均收益({holding_days}日)": ret_series.mean() if len(ret_series) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_top_n_curves(
    rebalance_returns: pd.DataFrame,
    holding_days: int,
    save_path: str | Path,
) -> Path:
    save_path = Path(save_path)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    for col in rebalance_returns.columns:
        cum = (1 + rebalance_returns[col].dropna()).cumprod()
        axes[0].plot(cum.index, cum.values, label=col, linewidth=1.8 if col == "全市场等权" else 1.4)
    axes[0].axhline(1, linestyle=":", color="gray")
    axes[0].set_title(f"TopN 与全市场累计净值对比 (每{holding_days}日换仓)")
    axes[0].set_ylabel("累计净值")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    benchmark = rebalance_returns["全市场等权"]
    for col in rebalance_returns.columns:
        if col == "全市场等权":
            continue
        excess = (rebalance_returns[col] - benchmark).dropna().cumsum()
        axes[1].plot(excess.index, excess.values, label=f"{col} 超额")
    axes[1].axhline(0, linestyle="--", color="gray")
    axes[1].set_title("相对全市场累计超额收益")
    axes[1].set_ylabel("累计超额")
    axes[1].set_xlabel("日期")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_monthly_return_table(rebalance_returns: pd.DataFrame) -> pd.DataFrame:
    monthly = rebalance_returns.copy()
    monthly.index = pd.to_datetime(monthly.index)
    return monthly.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def evaluate_relative_ic(pred_df: pd.DataFrame) -> pd.DataFrame:
    """评估相对全市场等权基准的日度 IC 情况。"""
    baseline = pred_df.groupby(COLS.trade_date)["raw_return"].mean().rename(BASELINE_COL)
    merged = pred_df.merge(baseline, on=COLS.trade_date, how="left")
    merged["excess_return"] = merged["raw_return"] - merged[BASELINE_COL]

    def _calc_excess_rank_ic(g: pd.DataFrame) -> float:
        if g["ml_factor"].nunique(dropna=True) < 2 or g["excess_return"].nunique(dropna=True) < 2:
            return np.nan
        ic, _ = spearmanr(g["ml_factor"], g["excess_return"], nan_policy="omit")
        return ic

    daily_excess_rank_ic = merged.groupby(COLS.trade_date).apply(_calc_excess_rank_ic).dropna()
    summary = pd.DataFrame(
        [
            {
                "指标": "Relative Rank IC",
                "均值": daily_excess_rank_ic.mean(),
                "标准差": daily_excess_rank_ic.std(),
                "IR": daily_excess_rank_ic.mean() / daily_excess_rank_ic.std() if daily_excess_rank_ic.std() > 0 else np.nan,
                "胜率": (daily_excess_rank_ic > 0).mean(),
            }
        ]
    )
    return daily_excess_rank_ic.to_frame("relative_rank_ic"), summary
