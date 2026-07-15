

from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd

from core.model.backtest_config import BacktestConfig
from core.model.finance_manager import FinanceDataFrame
from core.utils.misc_kit import pd_concat

pd.set_option("expand_frame_repr", False)  # 当列太多时不换行


# region  财务数据处理
def mark_old_report(date_list):
    """
    标记当前研报期是否为废弃研报。
    例如，已经发布1季度报，又更新了去年的年报，则去年的年报就是废弃报告
    :param date_list:
    :return:1表示为旧研报，nan表示非旧研报
    """
    # 使用 list 相比 TSeries 在性能上要好很多，使用上保持一致
    date_list = date_list.tolist()
    res = []
    for index, date in enumerate(date_list):
        flag = 0  # 初始化返回结果，0表示为非废弃报告
        for i in sorted(range(index), reverse=True):
            # 如果之前已经有比now更加新的财报了，将now标记为1
            if date_list[i] > date:
                flag = 1
                break
        res.append(flag)
    return res


def get_last_quarter_and_year_index(date_list):
    """
    获取上季度、上年度、以及上一次年报的索引
    :param date_list: 财报日期数据
    :return: 上季度、上年度、以及上一次年报的索引
    """
    # 使用 list 相比 TSeries 在性能上要好很多，使用上保持一致
    date_list = date_list.tolist()
    # 申明输出变量
    last_q_index = []  # 上个季度的index
    last_4q_index = []  # 去年同期的index
    last_y_index = []  # 去年年报的index
    last_y_3q_index = []  # 去年三季度的index
    last_y_2q_index = []  # 去年二季度的index
    last_y_q_index = []  # 去年一季度的index

    no_meaning_index = len(date_list) - 1  # 无意义的索引值，（最后一行的索引）

    # 逐个日期循环
    for index, date in enumerate(date_list):
        # 首个日期时，添加空值
        if index == 0:
            last_q_index.append(no_meaning_index)
            last_4q_index.append(no_meaning_index)
            last_y_index.append(no_meaning_index)
            last_y_3q_index.append(no_meaning_index)
            last_y_2q_index.append(no_meaning_index)
            last_y_q_index.append(no_meaning_index)
            continue

        # 反向逐个遍历当前日期之前的日期
        q_finish = False
        _4q_finish = False
        y_finish = False
        _y_3q_index = False
        _y_2q_index = False
        _y_q_index = False
        for i in sorted(range(index), reverse=True):
            # 计算之前日期和当前日期相差的月份
            delta_month = (date - date_list[i]).days / 30
            delta_month = round(delta_month)
            # 如果相差3个月，并且尚未找到上个季度的值
            if delta_month == 3 and q_finish is False:
                last_q_index.append(i)
                q_finish = True  # 已经找到上个季度的值
            # 如果相差12个月，并且尚未找到去年同期的值
            if delta_month == 12 and _4q_finish is False:
                last_4q_index.append(i)
                _4q_finish = True  # 已经找到上个年度的值
            # 如果是去年4季度，并且尚未找到去年4季度的值
            if date.year - date_list[i].year == 1 and date_list[i].month == 3 and _y_q_index is False:
                last_y_q_index.append(i)
                _y_q_index = True
            # 如果是去年4季度，并且尚未找到去年4季度的值
            if date.year - date_list[i].year == 1 and date_list[i].month == 6 and _y_2q_index is False:
                last_y_2q_index.append(i)
                _y_2q_index = True
            # 如果是去年4季度，并且尚未找到去年4季度的值
            if date.year - date_list[i].year == 1 and date_list[i].month == 9 and _y_3q_index is False:
                last_y_3q_index.append(i)
                _y_3q_index = True
            # 如果是去年4季度，并且尚未找到去年4季度的值
            if date.year - date_list[i].year == 1 and date_list[i].month == 12 and y_finish is False:
                last_y_index.append(i)
                y_finish = True

            # 如果三个数据都找到了
            if q_finish and _4q_finish and y_finish and _y_q_index and _y_2q_index and _y_3q_index:
                break  # 退出寻找
        if q_finish is False:  # 全部遍历完之后，尚未找到上个季度的值
            last_q_index.append(no_meaning_index)
        if _4q_finish is False:  # 全部遍历完之后，尚未找到4个季度前的值
            last_4q_index.append(no_meaning_index)
        if y_finish is False:  # 全部遍历完之后，尚未找到去年4季度的值
            last_y_index.append(no_meaning_index)
        if _y_q_index is False:  # 全部遍历完之后，尚未找到去年4季度的值
            last_y_q_index.append(no_meaning_index)
        if _y_2q_index is False:  # 全部遍历完之后，尚未找到去年4季度的值
            last_y_2q_index.append(no_meaning_index)
        if _y_3q_index is False:  # 全部遍历完之后，尚未找到去年4季度的值
            last_y_3q_index.append(no_meaning_index)
    # 返回
    return last_q_index, last_4q_index, last_y_index, last_y_q_index, last_y_2q_index, last_y_3q_index


def get_index_data(data, index_list, col_list):
    """
    根据索引获取数据
    :param data: 输入的数据
    :param index_list: 索引值的list
    :param col_list: 需要获取的字段list
    :return:
    """
    # 因为 cal_fin_data 中只计算了必须的字段，此处做一个过滤
    col_list = [col for col in col_list if col in data.columns]
    df = data.loc[index_list, col_list].reset_index()
    df = df[df["index"] != df.shape[0] - 1]  # 删除没有意义的行
    return df


def cal_fin_data(data, flow_fin_list=(), cross_fin_list=(), discard=True) -> pd.DataFrame:
    """
    计算财务数据的各类指标
    :param data: 输入的财务数据
    :param flow_fin_list: 流量型财务指标：净利润之类的
    :param cross_fin_list: 截面型的财务指标：净资产
    :param discard: 是否废弃财报
    :return:计算好财务指标的数据
    """

    # 数据排序
    data.sort_values(["publish_date", "report_date"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 时间格式转换
    def time_change(x):
        try:
            return pd.to_datetime(x, format="%Y%m%d")
        except Exception as e:
            print(e)
            return pd.to_datetime(x)

    try:
        data["report_date"] = pd.to_datetime(data["report_date"], format="%Y%m%d")
    except Exception as exp:
        print(exp)
        data["report_date"] = data["report_date"].apply(time_change)

    # 获取上一季度、年度的索引、上年报索引
    last_q_index, last_4q_index, last_y_index, last_y_q_index, last_y_2q_index, last_y_3q_index = (
        get_last_quarter_and_year_index(data["report_date"])
    )

    # 计算单季度数据、ttm数据
    last_q_df = get_index_data(data, last_q_index, flow_fin_list)  # 获取上个季度的数据
    last_4q_df = get_index_data(data, last_4q_index, flow_fin_list)  # 获取去年同期的数据
    last_y_df = get_index_data(data, last_y_index, flow_fin_list)  # 获取去年4季度数据

    # 判断字段列表是否需要输出，需要输出的字段已经提前添加到 data 中了
    data_columns = data.columns

    def need_col(col_list: list) -> bool:
        for _col in col_list:
            if _col in data_columns:
                return True
        return False

    # ==========处理流量数据
    for col in flow_fin_list:
        # 计算当季度数据
        if need_col([col + "_单季", col + "_单季环比", col + "_单季同比"]):
            data[col + "_单季"] = data[col] - last_q_df[col]
            # 第一季度的单季值等于本身
            data.loc[data["report_date"].dt.month == 3, col + "_单季"] = data[col]
        # 计算累计同比数据
        if need_col([col + "_累计同比"]):
            data[col + "_累计同比"] = data[col] / last_4q_df[col] - 1
            minus_index = last_4q_df[last_4q_df[col] < 0].index
            data.loc[minus_index, col + "_累计同比"] = 1 - data[col] / last_4q_df[col]
        # 计算ttm数据
        if need_col([col + "_ttm", col + "_ttm同比"]):
            data[col + "_ttm"] = data[col] + last_y_df[col] - last_4q_df[col]
            # 第四季度的ttm等于本身
            data.loc[data["report_date"].dt.month == 12, col + "_ttm"] = data[col]

    # 单季度环比、同比，ttm同比
    last_q_df = get_index_data(data, last_q_index, [c + "_单季" for c in flow_fin_list])
    last_4q_df = get_index_data(
        data, last_4q_index, [c + "_单季" for c in flow_fin_list] + [c + "_ttm" for c in flow_fin_list]
    )
    for col in flow_fin_list:
        # 计算单季度环比、同比
        if need_col([col + "_单季环比"]):
            data[col + "_单季环比"] = data[col + "_单季"] / last_q_df[col + "_单季"] - 1  # 计算当季度环比
            minus_index = last_q_df[last_q_df[col + "_单季"] < 0].index
            data.loc[minus_index, col + "_单季环比"] = (
                1 - data[col + "_单季"] / last_q_df[col + "_单季"]
            )  # 计算当季度环比
        if need_col([col + "_单季同比"]):
            data[col + "_单季同比"] = data[col + "_单季"] / last_4q_df[col + "_单季"] - 1  # 计算当季度同比
            minus_index = last_4q_df[last_4q_df[col + "_单季"] < 0].index
            data.loc[minus_index, col + "_单季同比"] = (
                1 - data[col + "_单季"] / last_4q_df[col + "_单季"]
            )  # 计算当季度同比
        # ttm同比
        if need_col([col + "_ttm同比"]):
            data[col + "_ttm同比"] = data[col + "_ttm"] / last_4q_df[col + "_ttm"] - 1  # 计算ttm度同比
            minus_index = last_4q_df[last_4q_df[col + "_ttm"] < 0].index
            data.loc[minus_index, col + "_ttm同比"] = 1 - data[col + "_ttm"] / last_4q_df[col + "_ttm"]  # 计算ttm度同比

    # ==========处理截面数据
    last_q_df = get_index_data(data, last_q_index, cross_fin_list)  # 获取上个季度的数据
    last_4q_df = get_index_data(data, last_4q_index, cross_fin_list)  # 获取去年4季度数据
    for col in cross_fin_list:  # 处理截面型数据
        if need_col([col + "_环比"]):
            data[col + "_环比"] = data[col] / last_q_df[col] - 1
            minus_index = last_q_df[last_q_df[col] < 0].index
            data.loc[minus_index, col + "_环比"] = 1 - data[col] / last_q_df[col]
        if need_col([col + "_同比"]):
            data[col + "_同比"] = data[col] / last_4q_df[col] - 1
            minus_index = last_4q_df[last_4q_df[col] < 0].index
            data.loc[minus_index, col + "_同比"] = 1 - data[col] / last_4q_df[col]

    # 标记废弃报告：例如已经有了1季度再发去年4季度的报告，那么4季度报告就只用来计算，不最终合并。
    if discard:
        data["废弃报告"] = mark_old_report(data["report_date"])
        # 删除废弃的研报
        data = data[data["废弃报告"] != 1]
        # 删除不必要的行
        del data["废弃报告"]
    return data


def get_his_data(fin_df, data_cols, span="q"):
    """
    获取财务数据的历史数据值
    :param fin_df: 财务数据的dataframe
    :param data_cols:需要获取的列名
    :param span:事件间隔
    :return:
    """
    data = fin_df.copy()
    # 获取上一季度、年度的索引、上年报索引
    last_q_index, last_4q_index, last_y_index, last_y_q_index, last_y_2q_index, last_y_3q_index = (
        get_last_quarter_and_year_index(data["report_date"])
    )
    if span == "4q":  # 去年同期
        last_index = last_4q_index
        label = "去年同期"
    elif span == "y":  # 去年年报
        last_index = last_y_index
        label = "去年年报"
    elif span == "y_q":
        last_index = last_y_q_index
        label = "去年一季度"
    elif span == "y_2q":
        last_index = last_y_2q_index
        label = "去年二季度"
    elif span == "y_3q":
        last_index = last_y_3q_index
        label = "去年三季度"
    else:  # 默认使用上季度
        last_index = last_q_index
        label = "上季度"

    # 获取历史数据
    last_df = get_index_data(data, last_index, data_cols)
    del last_df["index"]
    # 合并数据
    data = pd.merge(left=data, right=last_df, left_index=True, right_index=True, how="left", suffixes=("", "_" + label))
    # 只输出历史数据
    new_cols = [col + "_" + label for col in data_cols]
    keep_col = ["publish_date", "report_date"] + new_cols
    data = data[keep_col].copy()

    return data, new_cols


def prepare_fin_cols(cols):
    # 划分流量型和截面型财务数据
    flow_fin_cols = list(
        set([col.split("@xbx")[0] + "@xbx" for col in cols if (col.startswith("R_")) or (col.startswith("C_"))])
    )  # 流量型
    cross_fin_cols = list(set([col.split("@xbx")[0] + "@xbx" for col in cols if col.startswith("B_")]))  # 截面型
    return flow_fin_cols, cross_fin_cols


# 计算财务预处理数据
def merge_with_finance_data(conf: BacktestConfig, stock_code, stock_df):
    """
    将财务数据合并到日线数据上
    :param conf: 回测配置
    :param stock_code: 股票代码
    :param stock_df: 日线数据
    """
    # 判断路径是否存在
    stock_fin_folder = conf.fin_data_path / stock_code
    fin_cols = deepcopy(conf.fin_cols)
    # 原始财务数据
    origin_fin_list = []

    if stock_fin_folder.exists():
        # 获取路径下的所有文件名
        # 划分流量型和截面型财务数据
        flow_fin_cols, cross_fin_cols = prepare_fin_cols(fin_cols)

        finance_dfs = []
        # 读取路径下的各个财务数据文件
        for file in stock_fin_folder.iterdir():
            # 读取财务数据
            finance_df = pd.read_csv(stock_fin_folder / file, parse_dates=["publish_date"], skiprows=1, encoding="gbk")
            origin_fin_list.append(finance_df)

            # 判断财务数据中是否包含我们需要的finance_cols
            for col in set(flow_fin_cols + cross_fin_cols + fin_cols):
                # 如果没有我们需要的，赋值nan到财务数据的dataframe中
                if col not in finance_df.columns:
                    finance_df[col] = np.nan

            necessary_cols = ["stock_code", "report_date", "publish_date"]  # 所必须的字段
            finance_df = finance_df[
                list(set(necessary_cols + flow_fin_cols + cross_fin_cols + fin_cols))
            ]  # 取需要的数据
            # 计算财务类因子
            finance_df = cal_fin_data(
                data=finance_df, flow_fin_list=flow_fin_cols, cross_fin_list=cross_fin_cols, discard=False
            )
            # 合并
            col = ["publish_date", "report_date"] + fin_cols
            finance_dfs.append(finance_df[col])

        # 对数据做合并和排序处理
        all_finance_df = pd_concat(finance_dfs, ignore_index=True, copy=False)
        all_finance_df.sort_values(by=["publish_date", "report_date"], inplace=True)
        all_finance_df_not_discord = all_finance_df.copy()

        all_finance_df["废弃报告"] = mark_old_report(all_finance_df["report_date"])  # 获取废弃报告
        # 删除废弃的研报
        all_finance_df = all_finance_df[all_finance_df["废弃报告"] != 1]
        # 删除不必要的行
        del all_finance_df["废弃报告"]

        all_finance_df.drop_duplicates(subset=["publish_date"], keep="last", inplace=True)  # 删除重复数据
        all_finance_df.reset_index(drop=True, inplace=True)  # 重置索引
        stock_df = pd.merge_asof(
            stock_df, all_finance_df, left_on="交易日期", right_on="publish_date", direction="backward"
        )  # 合并股票数据和财务数据
        origin_fin_df = pd_concat(origin_fin_list, copy=False).sort_values(
            by=["publish_date", "report_date"], ignore_index=True
        )
        # 演示merge_asof效果：右边的数据，会找左边最接近的日期去合并。backward往上找，forward往下找，nearest最近
    else:  # 如果本地没有财务数据，返回含有财务数据为nan的dataframe
        print(f"{stock_code}未找到财务数据，如果一直报这个错误，请检查财务数据路径是否正确。偶尔几个可以忽略。")
        all_finance_df = pd.DataFrame()
        for col in ["publish_date", "report_date"] + fin_cols:
            stock_df[f"{col}"] = np.nan
            all_finance_df[f"{col}"] = np.nan
        all_finance_df_not_discord = all_finance_df.copy()
        origin_fin_df = all_finance_df.copy()
    return stock_df, all_finance_df, all_finance_df_not_discord, origin_fin_df


def get_finance_data(conf: BacktestConfig, stock_code, stock_df):
    """
    将财务数据合并到日线数据上
    :param conf: 回测配置
    :param stock_code: 股票代码
    :param stock_df: 日线数据
    """
    # 判断路径是否存在
    stock_fin_folder = conf.fin_data_path / stock_code
    fin_cols = deepcopy(conf.fin_cols)
    flow_list, section_list = prepare_fin_cols(fin_cols)
    if stock_fin_folder.exists():
        # 读取路径下的各个财务数据文件
        origin_fin_list = [
            pd.read_csv(file, parse_dates=["publish_date"], skiprows=1, encoding="gbk")
            for file in stock_fin_folder.iterdir()
            # 排除隐藏文件、只读文件、非csv文件
            if (not file.stem.startswith(".")) and file.is_file() and (file.suffix == ".csv")
        ]
        origin_fin_df = pd_concat(origin_fin_list, copy=False).sort_values(
            by=["publish_date", "report_date"], ignore_index=True
        )
        # 判断财务数据中是否包含我们需要的finance_cols
        for col in set(flow_list + section_list + fin_cols):
            # 如果没有我们需要的，赋值nan到财务数据的dataframe中
            if col not in origin_fin_df.columns:
                origin_fin_df[col] = np.nan
    else:  # 如果本地没有财务数据，返回含有财务数据为nan的dataframe
        print(f"{stock_code}未找到财务数据，如果一直报这个错误，请检查财务数据路径是否正确。偶尔几个可以忽略。")
        all_finance_df = pd.DataFrame()
        # 此处要把flow_list和section_list加上，不然后面会报错
        for col in set(["publish_date", "report_date"] + fin_cols + flow_list + section_list):
            stock_df[f"{col}"] = np.nan
            all_finance_df[f"{col}"] = np.nan
        origin_fin_df = all_finance_df.copy()
    return stock_df, origin_fin_df


def merge_with_finance_data_new(
    conf: BacktestConfig, stock_df: pd.DataFrame, fin_data_ins: FinanceDataFrame
) -> pd.DataFrame:
    """
    原始数据：
    xxx
    xxx_累计同比：就是原始数据的同比
    单季数据：
    xxx_单季
    xxx_单季环比
    xxx_单季同比
    ttm数据：
    xxx_ttm
    xxx_ttm同比
    """
    for col in conf.fin_cols:
        # 新版财务因子实例才需要原始字段
        if is_raw_col := ("@xbx_" not in col):
            raw_col = col
        else:
            raw_col = col.split("@xbx_")[0] + "@xbx"
        if is_raw_col:
            # 原始数据
            df = fin_data_ins[raw_col].raw()
        elif col.endswith("_累计同比"):
            df = fin_data_ins[raw_col].raw().yoy()
        elif col.endswith("_单季"):
            df = fin_data_ins[raw_col].quarter()
        elif col.endswith("_单季环比"):
            df = fin_data_ins[raw_col].quarter().qoq()
        elif col.endswith("_单季同比"):
            df = fin_data_ins[raw_col].quarter().yoy()
        elif col.endswith("_ttm"):
            df = fin_data_ins[raw_col].ttm()
        elif col.endswith("_ttm同比"):
            df = fin_data_ins[raw_col].ttm().yoy()
        # ============截面型数据============
        elif col.startswith("B_") and col.endswith("_环比"):
            df = fin_data_ins[raw_col].raw().qoq()
        elif col.startswith("B_") and col.endswith("_同比"):
            df = fin_data_ins[raw_col].raw().yoy()
        else:
            continue
        # stock_df和df的长度，索引，一定是完全一致的
        stock_df[col] = df
    # 补上report_date和publish_date这两个字段
    stock_df[["report_date", "publish_date"]] = fin_data_ins.generate_finance_date()
    return stock_df


def merge_with_calc_fin_data(stock_df, no_discard_finance_df, calc_fin_cols, extra_agg_dict):
    """
    通过计算添加研报的同期数据
    :param stock_df: 元数据
    :param no_discard_finance_df: 未废除研报的完整数据
    :param calc_fin_cols: 计算的list
    :param extra_agg_dict: 时间转换的额外dict
    :return:
    """
    if len(calc_fin_cols) == 0:
        return stock_df

    for col_dict in calc_fin_cols:
        cols = col_dict.get("col")
        q = col_dict.get("quarter")
        if len(cols) == 0 or len(q) == 0:
            continue

        # 获取去年年报数据，注意用全量未删除的废弃的财报数据
        fin_df, new_cols = get_his_data(no_discard_finance_df, cols, q)
        # 刚刚上市的股票没有研报
        if fin_df.empty:
            for new_col in new_cols:
                stock_df[new_col] = np.nan
                extra_agg_dict[new_col] = "last"
            continue

        # 去年年报数据合并到df中
        stock_df = pd.merge_asof(
            left=stock_df,
            right=fin_df,
            left_on="交易日期",
            right_on="publish_date",
            direction="backward",
            suffixes=("", "_y"),
        )
        for new_col in new_cols:
            # stock_df[new_col].fillna(method='ffill', inplace=True)
            stock_df[new_col] = stock_df[new_col].ffill()
            extra_agg_dict[new_col] = "last"

    return stock_df


def generate_fin_pivot(origin_df: pd.DataFrame, candle_df: pd.DataFrame, cols: List):
    """
    根据原始财务数据，生成透析表，方便计算单季、TTM数据
    :param origin_df: 原始财务数据
    :param candle_df: 行情数据
    :param cols: 用到的列
    :return:
    """
    # 感谢小番茄老板提供的代码，https://bbs.quantclass.cn/thread/5948
    origin_df = origin_df.copy()
    origin_df["report_date"] = pd.to_datetime(origin_df["report_date"], format="%Y%m%d")
    # 如果存在publish_date, report_date两列都一致的情况下，保留最新的那一行
    fin_data = origin_df.sort_values(["publish_date", "report_date"]).drop_duplicates(
        ["publish_date", "report_date"], keep="last", ignore_index=True
    )
    if not fin_data.empty:
        min_year = min(fin_data["publish_date"].min(), fin_data["report_date"].min()).year
        min_year = min(candle_df["交易日期"].min().year, min_year)
        # 包含所有季报以及对应的publish截止日期
        fin_base_date = creat_fin_base_date(min_year)
        # 合并基准 publish_date
        merge_fin_data = fin_data.merge(
            fin_base_date[["report_date", "publish_date"]], on="report_date", how="left", suffixes=("", "_base")
        )
        # 找出每个 report_date 的“首次行”= 最早的 publish_date
        tmp_sorted = merge_fin_data.sort_values(["report_date", "publish_date"], kind="mergesort")
        first_idx = tmp_sorted.groupby("report_date", sort=False).head(1).index
        is_first = merge_fin_data.index.isin(first_idx)
        # 仅对“首次行”应用条件
        mask = (
            is_first
            & merge_fin_data["publish_date"].notna()
            & merge_fin_data["publish_date_base"].notna()
            & (merge_fin_data["publish_date"] > merge_fin_data["publish_date_base"])
        )
        # 需要插入的行：publish_date 改成基准日，@xbx相关字段 列置为 NaN
        deadline_rows = merge_fin_data.loc[mask].copy()
        deadline_rows["publish_date"] = deadline_rows["publish_date_base"]
        xbx_cols = [c for c in fin_data.columns if "@xbx" in c]
        if xbx_cols:
            deadline_rows.loc[:, xbx_cols] = np.nan
        deadline_rows = deadline_rows[fin_data.columns]
        # 找出fin_data中不存在于fin_base_date中的季报
        missing_rows = (
            pd.merge(fin_base_date, fin_data, how="left", on="report_date", indicator=True)
            .loc[lambda df: df["_merge"].eq("left_only")]
            .drop(columns="_merge")
            .rename(columns={"publish_date_x": "publish_date"})
        )
        # 合并最终的财务数据，注，不需要去重，否则会把deadline_rows给去除掉
        fin_data = pd.concat([fin_data, deadline_rows, missing_rows]).sort_values(
            ["report_date", "publish_date"], ignore_index=True
        )
    fin_data["report_quarter"] = pd.PeriodIndex.from_fields(
        year=fin_data["report_date"].dt.year, quarter=fin_data["report_date"].dt.quarter, freq="Q"
    )

    pivot_dict = {}
    # 对cols做清洗
    flow_list, section_list = prepare_fin_cols(cols)
    for col in flow_list + section_list:
        # 填充inf，为后面的stack做准备，生成pivot后，会把inf转换成nan，确保原始pivot数据是干净的
        new_fin_data = fin_data[["publish_date", "report_quarter", col]].sort_values(["report_quarter", "publish_date"])
        new_fin_data[col] = new_fin_data.groupby("report_quarter")[col].ffill()
        new_fin_data[col] = new_fin_data[col].fillna(np.inf)
        pivot_df = (
            pd.pivot(data=new_fin_data, columns="publish_date", index="report_quarter", values=col)
            .sort_index(axis=1)
            .ffill(axis=1)
        )
        pivot_dict[f"{col}_is_na"] = np.isinf(pivot_df)
        # 把inf转换成nan，确保原始pivot数据是干净的
        pivot_dict[col] = pivot_df.replace(np.inf, np.nan)
    return pivot_dict, fin_data


# ** 这部分代码由汪老板提供，感谢！ **
def creat_fin_base_date(min_year):
    """
    这个函数的目的是建立一个基准的，规范的财报时序数据表格，用于后续合并每个股票的财报数据
    得到两个基础日期字段，'report_date'，'publish_date'，
    其中发布日期是指理论上的最后发布日期，为了填充延迟发布财报的情况。
    """
    # 生成所有财报日期（2007年至今）
    dates = [
        pd.Timestamp(year=year, month=month, day=day)
        for year in range(min_year, pd.Timestamp.now().year + 1)
        for month, day in zip([3, 6, 9, 12], [31, 30, 30, 31])
    ]

    df = pd.DataFrame(
        {"report_date": [d for d in dates if d <= pd.Timestamp.now()]}
    )  # 只要过了财报日，基准日期就会生成对应的数据行
    # 测试 假如今天是20250705，理论上是没有中报的
    # df = pd.DataFrame({'report_date': [d for d in dates if d <= pd.Timestamp(year=2025, month=7, day=5)]})

    # 通过月份直接映射财报类型和截止日期
    month_map = {3: "04-30", 6: "08-31", 9: "10-31", 12: "04-30"}
    # 计算发布日期（规定的最后发布日期）
    df["publish_date"] = (
        # int + bool = int + (1, 0)
        (df["report_date"].dt.year + (df["report_date"].dt.month == 12)).astype(str)
        + "-"
        + df["report_date"].dt.month.map(month_map)
    )

    # 确保 publish_date report_date 是 datetime 类型
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    # 得用report_date，不然可能会漏最新一行的数据
    df = df[df["report_date"] <= pd.Timestamp.now()]

    df["原始数据标签"] = 0
    return df


# endregion
