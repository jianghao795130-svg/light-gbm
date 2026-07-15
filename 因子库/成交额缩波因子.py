
import pandas as pd

fin_cols = []  # 财务因子列


# noinspection PyUnusedLocal
def add_factor(df: pd.DataFrame, param, fin_data=None, **kwargs) -> (pd.DataFrame, dict):
    """
    合并数据后计算策略所需的因子。

    :param df: 输入的K线数据，包含各类市场指标。
    :param param: 策略参数，用于配置因子计算的具体细节。
    :param fin_data: 财务数据字典，格式为 {'财务数据': fin_df, '原始财务数据': raw_fin_df}，
                     其中raw_fin_df包含需要舍弃的原始报告数据。
    :param kwargs: 其他关键字参数，包括但不限于因子名称（'col_name'）。
    :return:
        tuple:
            pd.DataFrame: 包含计算后的因子数据，索引与输入的df一致。
            dict: 聚合字典，指定因子数据的聚合方式。
    """

    # 计算因子时需要用到的短参数
    short = param[0]
    # 计算因子时需要用到的长参数
    long = param[1]
    # 从额外参数中获取因子名称
    col_name = kwargs['col_name']

    # 计算短期的成交额标准差
    short_std = df['成交额'].rolling(short).std()
    # 计算长期的成交额标准差
    long_std = df['成交额'].rolling(long).std()
    # 计算成交额缩波因子
    df[col_name] = short_std / long_std

    # 创建包含指定因子的DataFrame

    # 定义因子聚合方式，这里选择获取最新的因子值


    return df[[col_name]]
