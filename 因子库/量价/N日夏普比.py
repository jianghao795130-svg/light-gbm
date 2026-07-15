import pandas as pd
import numpy as np

fin_cols = []  # 财务因子列


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    N日夏普比率因子（Rolling Sharpe Ratio）

    计算逻辑：
    Sharpe(N) = rolling_mean(ret, N) / rolling_std(ret, N)

    ret = 收盘价_复权的日收益率
    """

    col_name = kwargs['col_name']
    n = int(param)

    # 1. 计算收益率
    df['ret'] = df['收盘价_复权'].pct_change()

    # 2. rolling均值与标准差
    mean_ret = df['ret'].rolling(window=n).mean()
    std_ret = df['ret'].rolling(window=n).std()

    # 3. 夏普比（避免除0）
    df[col_name] = mean_ret / (std_ret + 1e-12)

    return df[[col_name]]