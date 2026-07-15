

import itertools
from datetime import timedelta

import numpy as np
import pandas as pd


# 计算策略评价指标
def strategy_evaluate(
    equity,
    net_col="净值",
    pct_col="涨跌幅",
    turnover_col="换手率",
    commission_col="券商佣金",
    asset_col="总资产",
    commission_rate=None,
    risk_free_rate=0.0,
):
    """
    回测评价函数
    :param equity: 资金曲线数据
    :param net_col: 资金曲线列名
    :param pct_col: 周期涨跌幅列名
    :param turnover_col: 周期换手率列名
    :param commission_col: 券商佣金列名，用于在没有换手率列时反推组合换手率
    :param asset_col: 账户总资产列名
    :param commission_rate: 券商佣金费率
    :param risk_free_rate: 年化无风险收益率
    :return:
    """
    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # 将数字转为百分数
    def num_to_pct(value):
        return "%.2f%%" % (value * 100)

    # ===计算累积净值
    results.loc[0, "累积净值"] = round(equity[net_col].iloc[-1], 2)

    # ===计算年化收益
    days = (equity["交易日期"].iloc[-1] - equity["交易日期"].iloc[0]) / timedelta(days=1)
    annual_return = (equity[net_col].iloc[-1]) ** (365 / days) - 1
    results.loc[0, "年化收益"] = num_to_pct(annual_return)

    # ===计算夏普比
    returns = equity[pct_col].dropna()
    periods_per_year = len(returns) / days * 365 if days > 0 else np.nan
    returns_std = returns.std()
    if len(returns) > 1 and pd.notna(returns_std) and returns_std > 0 and pd.notna(periods_per_year) and periods_per_year > 0:
        period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        sharpe_ratio = ((returns - period_rf).mean() / returns_std) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = np.nan
    results.loc[0, "夏普比"] = round(sharpe_ratio, 2) if pd.notna(sharpe_ratio) else np.nan

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity[f'{net_col.split("资金曲线")[0]}max2here'] = equity[net_col].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity[f'{net_col.split("资金曲线")[0]}dd2here'] = (
        equity[net_col] / equity[f'{net_col.split("资金曲线")[0]}max2here'] - 1
    )
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(
        equity.sort_values(by=[f'{net_col.split("资金曲线")[0]}dd2here']).iloc[0][
            ["交易日期", f'{net_col.split("资金曲线")[0]}dd2here']
        ]
    )
    # 计算最大回撤开始时间
    start_date = equity[equity["交易日期"] <= end_date].sort_values(by=net_col, ascending=False).iloc[0]["交易日期"]
    results.loc[0, "最大回撤"] = num_to_pct(max_draw_down)
    results.loc[0, "最大回撤开始时间"] = str(start_date)
    results.loc[0, "最大回撤结束时间"] = str(end_date)
    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, "年化收益/回撤比"] = round(annual_return / abs(max_draw_down), 2)
    mean_back_zf = 1 / (1 + equity[f'{net_col.split("资金曲线")[0]}dd2here']) - 1  # 回本涨幅
    mean_fix_zf = mean_back_zf.mean()  # 修复涨幅
    max_back_zf = 1 / (1 + max_draw_down) - 1  # 回本涨幅
    max_fix_zf = max_back_zf.mean()  # 修复涨幅
    results.loc[0, "修复涨幅（均/最大）"] = f"{num_to_pct(mean_fix_zf)} / {num_to_pct(max_fix_zf)}"
    results.loc[0, "修复时间（均/最大）"] = (
        f"{round(np.log10(1 + mean_fix_zf) / np.log10(1 + annual_return) * 365, 1)} / "
        f"{round(np.log10(1 + max_fix_zf) / np.log10(1 + annual_return) * 365, 1)}"
    )
    # ===统计每个周期
    results.loc[0, "盈利周期数"] = len(equity.loc[equity[pct_col] > 0])  # 盈利笔数
    results.loc[0, "亏损周期数"] = len(equity.loc[equity[pct_col] <= 0])  # 亏损笔数
    not_zero = len(equity.loc[equity[pct_col] != 0])
    results.loc[0, "胜率（含0/去0）"] = (
        f"{num_to_pct(results.loc[0, '盈利周期数'] / len(equity))} / "
        f"{num_to_pct(len(equity.loc[equity[pct_col] > 0]) / not_zero)}"
    )  # 胜率
    results.loc[0, "每周期平均收益"] = num_to_pct(equity[pct_col].mean())  # 每笔交易平均盈亏
    results.loc[0, "盈亏收益比"] = round(
        equity.loc[equity[pct_col] > 0][pct_col].mean() / equity.loc[equity[pct_col] <= 0][pct_col].mean() * (-1), 2
    )  # 盈亏比

    results.loc[0, "单周期最大盈利"] = num_to_pct(equity[pct_col].max())  # 单笔最大盈利
    results.loc[0, "单周期大亏损"] = num_to_pct(equity[pct_col].min())  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, "最大连续盈利周期数"] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] > 0, 1, np.nan))]
    )  # 最大连续盈利次数
    results.loc[0, "最大连续亏损周期数"] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] <= 0, 1, np.nan))]
    )  # 最大连续亏损次数

    # ===其他评价指标
    results.loc[0, "收益率标准差"] = num_to_pct(equity[pct_col].std())

    # ===统计每个周期的平均换手率
    if turnover_col in equity.columns:
        avg_turnover = equity[turnover_col].dropna().mean()
    elif (
        commission_rate
        and commission_rate > 0
        and commission_col in equity.columns
        and asset_col in equity.columns
    ):
        base_asset = equity[asset_col].shift(1).replace(0, np.nan)
        base_asset = base_asset.fillna(equity[asset_col].replace(0, np.nan))
        trade_amount = equity[commission_col] / commission_rate
        avg_turnover = (trade_amount / base_asset).replace([np.inf, -np.inf], np.nan).dropna().mean()
    else:
        avg_turnover = np.nan
    results.loc[0, "每周期平均换手率"] = num_to_pct(avg_turnover) if pd.notna(avg_turnover) else "N/A"

    # 空仓时，防止显示nan
    fillna_col = ["夏普比", "年化收益/回撤比", "盈亏收益比"]
    results[fillna_col] = results[fillna_col].fillna(0)

    # ===每年、每月收益率
    temp = equity.copy()
    temp.set_index("交易日期", inplace=True)

    year_return = temp[[pct_col]].resample(rule="YE").apply(lambda x: (1 + x).prod() - 1)
    month_return = temp[[pct_col]].resample(rule="ME").apply(lambda x: (1 + x).prod() - 1)
    quarter_return = temp[[pct_col]].resample(rule="QE").apply(lambda x: (1 + x).prod() - 1)

    def num2pct(x):
        if str(x) != "nan":
            return str(round(x * 100, 2)) + "%"
        else:
            return x

    year_return["涨跌幅"] = year_return[pct_col].apply(num2pct)
    month_return["涨跌幅"] = month_return[pct_col].apply(num2pct)
    quarter_return["涨跌幅"] = quarter_return[pct_col].apply(num2pct)

    return results.T, year_return, month_return, quarter_return
