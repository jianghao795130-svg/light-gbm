

import pandas as pd

from core.figure import draw_equity_curve_plotly, draw_config, merge_html, draw_table
from core.market_essentials import import_index_data, get_most_stock_by_year
from core.model.backtest_config import BacktestConfig
from core.utils.log_kit import logger


def show_performance_plot(conf: BacktestConfig, select_results, equity_df, rtn, year_return, title_prefix="", **kwargs):
    """
    绘制回测结果图表
    :param conf: 回测配置
    :param select_results: 选股结果
    :param equity_df: 回测结果净值数据
    :param rtn: 策略报告
    :param year_return: 分年收益率
    :param title_prefix: 标题前缀
    :param kwargs: 其他参数
    """
    # 添加指数数据
    for index_code, index_name in zip(["sh000300", "sh000852"], ["沪深300", "中证1000"]):
        index_path = (
            conf.index_hour_data_path / f"{index_code}.csv"
            if "小时" in title_prefix
            else conf.index_data_path / f"{index_code}.csv"
        )
        if not index_path.exists():
            logger.warning(f"{index_name}({index_code})指数数据不存在，无法添加指数数据")
            continue
        index_df = import_index_data(index_path, [conf.start_date, conf.end_date])
        equity_df = pd.merge(left=equity_df, right=index_df[["交易日期", "指数涨跌幅"]], on=["交易日期"], how="left")
        equity_df[index_name + "指数"] = (equity_df["指数涨跌幅"] + 1).cumprod()
        del equity_df["指数涨跌幅"]

    logger.debug(
        f"""📈 策略评价 --------------------------------
{rtn}

📊 分年收益率 --------------------------------
{year_return}"""
    )
    logger.debug(f'💰 总手续费: ￥{equity_df["手续费"].sum():,.2f}\n')

    logger.info("开始绘制资金曲线...")

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {"资金曲线": "净值", "沪深300指数": "沪深300指数", "中证1000指数": "中证1000指数"}

    right_axis = {"最大回撤": "净值dd2here"}

    if (pre_timing_equity := kwargs.get("pre_timing_equity")) is not None:
        equity_name = "再择时前资金曲线"
        equity_df[equity_name] = pre_timing_equity.reset_index(drop=True)
        data_dict.update({equity_name: equity_name})

    # 如果画资金曲线，同时也会画上回撤曲线
    date_start = equity_df["交易日期"].min().strftime("%Y/%m/%d")
    date_end = equity_df["交易日期"].max().strftime("%Y/%m/%d")
    ann_ret, max_dd, calmar = rtn.at["年化收益", 0], rtn.at["最大回撤", 0], rtn.at["年化收益/回撤比", 0]
    pic_title = f"年化收益:{ann_ret}  最大回撤:{max_dd}  收益回撤比:{calmar}  回测区间：{date_start} - {date_end}"
    pic_desc = ""
    # for stg in conf.strategy_list_raw:
    #     pic_desc += f'{stg["name"]}_{stg["hold_period"]}{stg["offset_list"]}_选{stg["select_num"]}_权{stg["cap_weight"]}_{stg["rebalance_time"]}+'

    fig_path = conf.get_result_folder() / f"{title_prefix}资金曲线.html"
    # 调用画图函数
    fig1 = draw_equity_curve_plotly(
        equity_df,
        data_dict=data_dict,
        date_col="交易日期",
        right_axis=right_axis,
        title=pic_title,
        desc=pic_desc[:-1],
        rtn_add=rtn,
        show_subplots=True,
    )

    # 获取每年选股最多的股票
    most_stock = (
        get_most_stock_by_year(select_results.loc[select_results["调仓类型"].eq("计划")])
        if not select_results.empty
        else pd.DataFrame()
    )
    fig2 = draw_table(most_stock)

    figs = [fig1, fig2]

    # 绘制子策略的资金曲线图
    sub_equity_df = equity_df[["交易日期"]].copy()
    sub_data_dict = {}
    for col_name, col_series in kwargs.get("extra_equities", {}).items():
        sub_equity_df[col_name] = col_series
        sub_data_dict[col_name] = col_name
    if not sub_equity_df.empty and sub_data_dict:
        fig3 = draw_equity_curve_plotly(
            sub_equity_df, data_dict=sub_data_dict, date_col="交易日期", title="子策略资金曲线", desc=""
        )
        figs.append(fig3)
    # 绘制config
    figs.append(draw_config(conf.strategy_list_raw))
    # 绘制图片
    merge_html(fig_path, figs)

    import webbrowser
    import os
    import platform

    # 获取操作系统类型
    system = platform.system().lower()
    abs_path = os.path.abspath(str(fig_path))
    if system == "darwin":  # macOS
        os.system(f'open "{abs_path}"')
    elif system == "windows":  # Windows
        os.system(f"start {abs_path}")
    else:  # Linux 或其他系统
        webbrowser.open("file://" + abs_path)
