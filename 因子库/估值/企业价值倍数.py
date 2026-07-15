
import pandas as pd
import numpy as np

fin_cols = ['B_st_borrow@xbx',  # 短期借款
            'B_noncurrent_liab_due_in1y@xbx',  # 一年内到期的长期借款
            'B_lt_loan@xbx',  # 长期借款
            'B_bond_payable@xbx',  # 应付债券
            'B_currency_fund@xbx', # 资产负债表_资产的货币资金_截面型
            'R_total_profit@xbx_单季',  # 利润总额
            'R_financing_expenses@xbx_单季',  # 财务费用
            'C_depreciation_etc@xbx_单季',  # 固定资产折旧、油气资产折耗、生产性生物资产折旧
            'C_intangible_assets_amortized@xbx_单季',  # 无形资产摊销
            'C_lt_deferred_expenses_amrtzt@xbx_单季',  # 长期待摊费用摊销
            'R_total_profit@xbx_ttm',  # 利润总额
            'R_financing_expenses@xbx_ttm',  # 财务费用
            'C_depreciation_etc@xbx_ttm',  # 固定资产折旧、油气资产折耗、生产性生物资产折旧
            'C_intangible_assets_amortized@xbx_ttm',  # 无形资产摊销
            'C_lt_deferred_expenses_amrtzt@xbx_ttm',  # 长期待摊费用摊销
            ]  # 财务因子列


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

    工作流程：
    1. 根据提供的参数计算股票的因子值。
    2. 将因子值添加到原始行情数据DataFrame中。

    :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
    :param param: 因子计算所需的参数，格式和含义根据因子类型的不同而有所不同。
    :param kwargs: 其他关键字参数，包括：
        - col_name: 新计算的因子列名。
        - fin_data: 财务数据字典，格式为 {'财务数据': fin_df, '原始财务数据': raw_fin_df}，其中fin_df为处理后的财务数据，raw_fin_df为原始数据，后者可用于某些因子的自定义计算。
        - 其他参数：根据具体需求传入的其他因子参数。
    :return:
        - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。

    注意事项：
    - 如果因子的计算涉及财务数据，可以通过`fin_data`参数提供相关数据。
    """

    """    
    ----->>>  配置方法  <<<-----
    配置：('企业价值倍数', is_sort_asc, param, arg)
    含义：EV(企业价值) = 公司市值 + 总债务 – 货币资金, EBITDA(息税折旧摊销前利润) = 净利润 + 利息 + 税收 + 折旧 + 摊销
        企业价值倍数 = EV / EBIT
    示例：'factor_list': [
                            ('企业价值倍数', True, 'ttm', 1),           # 企业价值倍数_ttm    
                            ('企业价值倍数', True, '单季', 1),           # 企业价值倍数_单季                         
                        ]
    """
    # 从kwargs中提取因子列的名称
    col_name = kwargs['col_name']
    # 如果参数不在可选范围：
    if param not in ['单季', 'ttm']:
        factor_name = __file__.replace('\\', '/').split('/')[-1].replace('.py', '')
        raise ValueError(f"{factor_name} 因子不支持的参数值：{param}")

    # 核心计算逻辑
    # 带息债务（一般企业）= 短期借款 + 一年内到期的长期借款 + 长期借款 + 应付债券
    df['带息债务'] = df[[
        'B_st_borrow@xbx',  # 短期借款
        'B_noncurrent_liab_due_in1y@xbx',  # 一年内到期的长期借款
        'B_lt_loan@xbx',  # 长期借款
        'B_bond_payable@xbx',  # 应付债券
        ]].sum(axis=1, skipna=True, min_count=0)
    df.loc[df['带息债务'] == 0, '带息债务'] = np.nan

    # EV(企业价值) = 公司市值 + 带息债务 – 货币资金
    df['EV'] = df['总市值'] + df['带息债务'].fillna(0) - df['B_currency_fund@xbx'].fillna(0)
    # EBITDA反推法模糊(利息用财务费用代替)

    df[f'EBITDA_{param}'] = df[[
        f'R_total_profit@xbx_{param}',  # 利润总额
        f'R_financing_expenses@xbx_{param}',  # 财务费用
        f'C_depreciation_etc@xbx_{param}',  # 固定资产折旧、油气资产折耗、生产性生物资产折旧
        f'C_intangible_assets_amortized@xbx_{param}',  # 无形资产摊销
        f'C_lt_deferred_expenses_amrtzt@xbx_{param}',  # 长期待摊费用摊销
    ]].sum(axis=1, skipna=True, min_count=0)

    df['企业价值倍数'] = df['EV'] / (df[f'EBITDA_{param}'] + 1e-8)
    factor_col = df['企业价值倍数']
    del df['EV'], df[f'EBITDA_{param}']
    # 创建包含指定因子的DataFrame
    factor_df = pd.DataFrame({col_name: factor_col}, index=df.index)

    return factor_df
