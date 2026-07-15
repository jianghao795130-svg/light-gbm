
import pandas as pd

# 财务因子列：此列表用于存储财务因子相关的列名称
fin_cols = ['R_np_atoopc@xbx_单季', 'B_total_equity_atoopc@xbx', 'R_np_atoopc@xbx_ttm']  # 财务因子列，配置后系统会自动加载对应的财务数据


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    # ======================== 参数处理 ===========================
    # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称
    col_name = kwargs['col_name']

    # 净利润相关字段说明
    # - R_np_atoopc@xbx_ttm:利润表的归属于母公司所有者的净利润ttm
    # - R_np_atoopc@xbx_单季:利润表的归属于母公司所有者的净利润单季度
    profit_cols = {
        '全年': 'R_np_atoopc@xbx_ttm',
        '单季': 'R_np_atoopc@xbx_单季'
    }

    # 根据param选择相应的净利润字段
    if param not in profit_cols:
        raise ValueError(f"ROE因子不支持的参数值：{param}")
    else:
        profit_col = profit_cols[param]

    # ======================== 计算因子 ===========================
    # ROE：净资产收益率 = 净利润 / 净资产
    # - B_total_equity_atoopc@xbx:资产负债表_所有者权益的归属于母公司所有者权益合计
    df[col_name] = df[profit_col] / df['B_total_equity_atoopc@xbx']

    return df[[col_name]]
