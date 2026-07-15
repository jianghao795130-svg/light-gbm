import pandas as pd

# 财务因子列：此列表用于存储财务因子相关的列名称
fin_cols = ['R_basic_eps@xbx']  # 财务因子列，配置后系统会自动加载对应的财务数据


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> (pd.DataFrame, dict):
    # ======================== 参数处理 ===========================
    # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称
    col_name = kwargs['col_name']

    # ======================== 计算因子 ===========================
    # 计算每股收益
    df[col_name] = df['R_basic_eps@xbx']

    # ======================== 聚合方式 ===========================

    # 返回新计算的因子列以及因子聚合方式
    return df[[col_name]]