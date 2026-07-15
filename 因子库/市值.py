
import pandas as pd

# 财务因子列：此列表用于存储财务因子相关的列名称
fin_cols = []  # 财务因子列，配置后系统会自动加载对应的财务数据


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:

    # ======================== 参数处理 ===========================
    # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称
    col_name = kwargs['col_name']

    # ======================== 计算因子 ===========================
    # 我们这里的市值因子使用总市值的数值
    df[col_name] = df['总市值']

    return df[[col_name]]
