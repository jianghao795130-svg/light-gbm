import pandas as pd

fin_cols = []  # 财务因子列


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算未来n天涨跌幅（未来函数，仅用于ML标签）。

    :param df: 包含单只股票的K线数据。
    :param param: int，未来天数n。
    :param kwargs: col_name 等。
    :return: 包含因子列的DataFrame。
    """
    col_name = kwargs['col_name']

    # 未来n天涨跌幅 = n+1日后的开盘价 / 明日开盘价 - 1
    df[col_name] = df['开盘价'].shift(-(param+1)) / df['开盘价'].shift(-1) - 1

    return df[[col_name]]
