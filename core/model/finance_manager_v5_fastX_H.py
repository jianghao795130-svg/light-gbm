"""


核心特性:
- 严格PIT语义:财报修订版按发布日隔离,杜绝未来数据泄漏
- 四层inf防护:彻底清洗除零产生的无穷值
- 季度缺失处理:diff/shift/pct_change在缺失季度正确返回NaN
- 性能优化:NumPy快速路径,懒加载,骨架表复用(整体加速8.4%)

环境变量开关:
- FMV5_QDF_NUMPY_FASTPATH: NumPy快速路径(默认1开启,设0关闭)
- FMV5_FRAMEWORK_CLEANED_FASTPATH: 框架预清洗快速路径(默认1开启,设0关闭)
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
from core.utils.log_kit import logger

# =============================
# 全局缓存:conf.fin_cols(所有股票公用,全局只加载一次)
# =============================
_GLOBAL_CONF_FIN_COLS: Optional[List[str]] = None

# =============================
# 全局预计算缓存
# =============================
_PRECOMPUTED_CACHE: Dict[str, 'PrecomputedFinanceData'] = {}

# =============================
# 财务CSV表头缓存(目录级,按mtime失效)
# key: (目录路径, encoding) -> (目录mtime_ns, 样本文件名, 样本文件mtime_ns, columns)
# =============================
_FIN_CSV_HEADER_CACHE: Dict[tuple, tuple] = {}

# =============================
# 法定截止日基准表缓存(进程级,所有股票公用)
# =============================
_FIN_BASE_DATE_CACHE: Optional[pd.DataFrame] = None

# =============================
# QDF NumPy 快速路径开关
# =============================
# 默认开启;设置环境变量 FMV5_QDF_NUMPY_FASTPATH=0 可关闭(用于线上兜底/回归对比)
_USE_NUMPY_QDF_FASTPATH: bool = os.getenv('FMV5_QDF_NUMPY_FASTPATH', '1') != '0'

# =============================
# 框架预清洗快速路径开关
# =============================
# 默认开启;设置环境变量 FMV5_FRAMEWORK_CLEANED_FASTPATH=0 可关闭
_USE_FRAMEWORK_CLEANED_FASTPATH: bool = os.getenv('FMV5_FRAMEWORK_CLEANED_FASTPATH', '1') != '0'

# =============================
# 辅助函数
# =============================
def _ensure_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()  # 避免 SettingWithCopyWarning
    for c in cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def _add_report_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['report_quarter'] = pd.PeriodIndex(
        year=df['report_date'].dt.year,
        quarter=df['report_date'].dt.quarter,
        freq='Q'
    )
    return df

def _create_fin_base_date() -> pd.DataFrame:
    """创建法定截止日基准表(所有股票公用,进程级缓存)
    返回 .copy() 防止调用方修改缓存
    """
    global _FIN_BASE_DATE_CACHE
    if _FIN_BASE_DATE_CACHE is not None:
        return _FIN_BASE_DATE_CACHE.copy()

    dates: List[pd.Timestamp] = [
        pd.Timestamp(year=year, month=month, day=day)
        for year in range(2007, pd.Timestamp.now().year + 1)
        for month, day in zip([3, 6, 9, 12], [31, 30, 30, 31])
    ]
    base_df = pd.DataFrame({'report_date': [d for d in dates if d <= pd.Timestamp.now()]})
    month_map = {3: '04-30', 6: '08-31', 9: '10-31', 12: '04-30'}
    base_df['publish_date'] = base_df['report_date'].apply(
        lambda x: f"{x.year + 1 if x.month == 12 else x.year}-{month_map[x.month]}"
    )
    base_df['publish_date'] = pd.to_datetime(base_df['publish_date'], errors='coerce')
    base_df['report_date'] = pd.to_datetime(base_df['report_date'], errors='coerce')
    base_df['原始数据标签'] = 0

    _FIN_BASE_DATE_CACHE = base_df
    return base_df.copy()

def _clean_raw_fin_df_like_factorfun(
    raw_df: pd.DataFrame,
    fin_base_date: pd.DataFrame,
    raw_fin_cols: List[str],
    ipo_day: pd.Timestamp,
    delay_param: Optional[str] = None,
    keep_debug_columns: bool = False,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    missing_cols = [col for col in raw_fin_cols if col not in raw_df.columns]
    if missing_cols:
        # 一次性汇总告警,避免刷屏
        logger.warning(f"财务字段缺失:{missing_cols},已自动补齐为NaN")
        for col in missing_cols:
            raw_df[col] = np.nan

    df = raw_df[['publish_date', 'report_date'] + raw_fin_cols].copy()
    if not pd.api.types.is_datetime64_any_dtype(df['publish_date']):
        df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['report_date']):
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    df['原始数据标签'] = 1
    df['是否上市前财报'] = np.where(df['report_date'] <= ipo_day, 1, 0)
    month = df['report_date'].dt.month
    year = df['report_date'].dt.year
    month_map_type = {3: '一季报', 6: '中报', 9: '三季报', 12: '年报'}
    df['财报类型'] = month.map(month_map_type)
    month_map_deadline = {3: '04-30', 6: '08-31', 9: '10-31', 12: '04-30'}
    deadline_suffix = month.map(month_map_deadline)
    deadline_year = np.where(month == 12, year + 1, year)
    # 修复numpy字符串连接问题:使用pandas Series操作
    deadline_year_series = pd.Series(deadline_year).astype(str)
    df['财报截止日期'] = pd.to_datetime(
        deadline_year_series + '-' + deadline_suffix.astype(str),
        errors='coerce'
    )
    df['该财报提前发布天数'] = (df['财报截止日期'] - df['publish_date']).dt.days
    if delay_param == '截止日':
        df['publish_date'] = np.where(df['该财报提前发布天数'] > 0, df['财报截止日期'], df['publish_date'])
    if delay_param == '延时':
        month_map_delay = {3: '04-27', 6: '08-26', 9: '10-28', 12: '04-20'}
        delay_suffix = month.map(month_map_delay)
        delay_year = np.where(month == 12, year + 1, year)
        # 修复numpy字符串连接问题:使用pandas Series操作
        delay_year_series = pd.Series(delay_year).astype(str)
        df['财报延时收录日期'] = pd.to_datetime(
            delay_year_series + '-' + delay_suffix.astype(str),
            errors='coerce'
        )
        df['publish_date'] = np.where(df['该财报提前发布天数'] > 0, df['财报延时收录日期'], df['publish_date'])
    df = pd.concat([df, fin_base_date], axis=0, ignore_index=True)
    df.sort_values(by=['report_date', 'publish_date', '原始数据标签'], ascending=[True, True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    first_valid_idx = df['原始数据标签'].eq(1).idxmax()
    df = df.iloc[first_valid_idx:]
    # 前向填充:填补同一report_date下的缺失值
    df[raw_fin_cols] = df.groupby('report_date')[raw_fin_cols].transform('ffill')
    df = df.drop_duplicates(subset=['publish_date', 'report_date'], keep='last')
    if '财报截止日期' in df.columns and '该财报提前发布天数' in df.columns:
        try:
            df['首发财报提前发布天数'] = df.groupby('财报截止日期')['该财报提前发布天数'].transform('max')
        except Exception:
            df['首发财报提前发布天数'] = np.nan
    if not keep_debug_columns:
        temp_cols = ['原始数据标签', '财报截止日期', '该财报提前发布天数', '财报延时收录日期', '首发财报提前发布天数']
        df.drop(columns=[c for c in temp_cols if c in df.columns], inplace=True)
    return df

def _can_use_framework_cleaned_fastpath(raw_fin_df: pd.DataFrame) -> bool:
    """

    契约条件(全部满足才返回True):
    1. raw_fin_df非空且有columns属性
    2. 必须包含publish_date和report_date列
    3. 必须包含_merge列(框架merge indicator=True的产物)
    4. publish_date和report_date必须是datetime64类型且无NaT
    5. (publish_date, report_date)键必须唯一

    Returns:
        bool: True=可以走快速路径,False=需要完整清洗
    """
    try:
        if raw_fin_df is None or not hasattr(raw_fin_df, 'columns'):
            return False
        if getattr(raw_fin_df, 'empty', True):
            return False

        cols = raw_fin_df.columns
        if 'publish_date' not in cols or 'report_date' not in cols:
            return False

        # 框架generate_fin_pivot通过indicator=True生成_merge列
        if '_merge' not in cols:
            return False

        # 检查日期类型(必须是datetime64[ns])
        pub = raw_fin_df['publish_date'].to_numpy(copy=False)
        rep = raw_fin_df['report_date'].to_numpy(copy=False)
        if pub.dtype.kind != 'M' or rep.dtype.kind != 'M':
            return False

        # 检查无NaT
        if np.isnat(pub).any() or np.isnat(rep).any():
            return False

        # 验证_merge列值合法
        merge_series = raw_fin_df['_merge']
        if hasattr(merge_series, 'isna') and merge_series.isna().any():
            return False
        merge_vals = pd.unique(merge_series)
        allowed = {'left_only', 'both', 'right_only'}
        for v in merge_vals:
            if str(v) not in allowed:
                return False

        # 检查键唯一性
        n = len(raw_fin_df)
        if n > 1:
            pub_i8 = pub.view('i8')
            rep_i8 = rep.view('i8')
            key = np.empty(n, dtype=[('p', 'i8'), ('r', 'i8')])
            key['p'] = pub_i8
            key['r'] = rep_i8
            if np.unique(key).size != n:
                return False

        return True
    except Exception:
        return False

def _minimal_clean_raw_fin_df_from_framework(
    raw_fin_df: pd.DataFrame,
    raw_fin_cols: List[str],
    ipo_day: pd.Timestamp,
) -> pd.DataFrame:
    """

    前置条件:_can_use_framework_cleaned_fastpath(raw_fin_df) == True

    处理内容(仅做必要补齐,跳过所有清洗):
    1. 从_merge列生成原始数据标签(left_only=0基准行, 其他=1真实数据)
    2. 对迟披露报表注入"截止日锚点行"(使截止日后不回退旧季度,与fullpath一致)
    3. 补齐缺失的财务列为NaN
    4. 返回统一格式的DataFrame

    Args:
        raw_fin_df: 框架已清洗的数据(含_merge列)
        raw_fin_cols: 需要的财务字段列表
        ipo_day: IPO日期(预留给fast1扩展)

    Returns:
        pd.DataFrame: 格式与_clean_raw_fin_df_like_factorfun输出一致
    """
    if raw_fin_df is None or not hasattr(raw_fin_df, 'columns'):
        return pd.DataFrame(columns=['publish_date', 'report_date'] + list(raw_fin_cols))

    _ = ipo_day  # 预留给fast1扩展

    # 从_merge列生成原始数据标签:left_only=0(基准行),其他=1(真实数据)
    if '_merge' in raw_fin_df.columns:
        is_anchor = raw_fin_df['_merge'].eq('left_only').to_numpy(copy=False)
        raw_tag = np.where(is_anchor, 0, 1).astype(np.int8, copy=False)
    else:
        raw_tag = np.ones(len(raw_fin_df), dtype=np.int8)

    
    # 目的:使截止日后不回退旧季度,与fullpath一致
    # 条件:真实首次披露日 > 法定截止日
    anchor_df = None
    try:
        if (
            '_merge' in raw_fin_df.columns
            and 'publish_date_x' in raw_fin_df.columns
            and 'publish_date' in raw_fin_df.columns
            and 'report_date' in raw_fin_df.columns
        ):
            # 真实数据行:_merge != left_only
            real_mask = raw_fin_df['_merge'].ne('left_only')
            if real_mask.any():
                real_key = raw_fin_df.loc[real_mask, ['report_date', 'publish_date']].dropna(
                    subset=['report_date', 'publish_date']
                )
                if not real_key.empty:
                    # 每个report_date的首次真实披露日
                    first_real_pub = real_key.groupby('report_date', sort=False)['publish_date'].min()

                    # 每个report_date的法定截止日(来自框架merge生成的publish_date_x)
                    due_map = (
                        raw_fin_df[['report_date', 'publish_date_x']]
                        .dropna(subset=['report_date', 'publish_date_x'])
                        .drop_duplicates(subset=['report_date'], keep='last')
                        .set_index('report_date')['publish_date_x']
                    )

                    if not due_map.empty:
                        common_idx = first_real_pub.index.intersection(due_map.index)
                        if len(common_idx) > 0:
                            late_mask = (
                                first_real_pub.loc[common_idx].to_numpy(copy=False)
                                > due_map.loc[common_idx].to_numpy(copy=False)
                            )
                            late_idx = common_idx[late_mask]
                            if len(late_idx) > 0:
                                anchor_df = pd.DataFrame({
                                    'publish_date': due_map.loc[late_idx].to_numpy(copy=False),
                                    'report_date': late_idx.to_numpy(copy=False),
                                })
                                for c in raw_fin_cols:
                                    anchor_df[c] = np.nan
                                anchor_df['原始数据标签'] = np.int8(0)
    except Exception:
        anchor_df = None

    # 识别缺失列
    missing_cols = [col for col in raw_fin_cols if col not in raw_fin_df.columns]
    existing_fin_cols = [c for c in raw_fin_cols if c in raw_fin_df.columns]

    # 构建输出DataFrame
    clean_df = raw_fin_df[['publish_date', 'report_date', *existing_fin_cols]].copy()
    clean_df['原始数据标签'] = raw_tag

    # 补齐缺失列
    if missing_cols:
        logger.warning(f"财务字段缺失:{missing_cols},已自动补齐为NaN")
        for col in missing_cols:
            clean_df[col] = np.nan

    
    # 问题:sh603102在2021-04-30首发68452710.76,2021-05-14修订稿NaN
    #       fast0直接使用NaN,fullpath用ffill保留有效值
    # 修复:与fullpath一致,按report_date分组前向填充(排序确保首发在前)
    if existing_fin_cols:
        clean_df = clean_df.sort_values(['report_date', 'publish_date'])
        clean_df[existing_fin_cols] = clean_df.groupby('report_date', sort=False)[existing_fin_cols].transform('ffill')

    # 统一列顺序
    clean_df = clean_df[['publish_date', 'report_date', *raw_fin_cols, '原始数据标签']]

    # 拼接迟披露锚点行
    if anchor_df is not None and not anchor_df.empty:
        anchor_df = anchor_df[['publish_date', 'report_date', *raw_fin_cols, '原始数据标签']]
        clean_df = pd.concat([clean_df, anchor_df], axis=0)

    # 重置索引确保唯一性(与fullpath的reset_index行为一致)
    # precompute()的骨架表依赖index作为row_id回表取值
    clean_df = clean_df.reset_index(drop=True)

    return clean_df

def _prepare_fin_pivot_skeleton(raw_fin_df: pd.DataFrame) -> pd.DataFrame:
    """

    prepare-once模式:把与列无关的"键清洗+去重+report_quarter生成"抽成一次性操作,
    避免每列重复执行(~9ms/列 -> 只执行1次)

    处理步骤(完全复刻旧版去重逻辑):
    1. 只取键列(publish_date, report_date),dropna
    2. 保存原始行号_row_id(用于后续按列取值)
    3. (publish_date, report_date)排序+去重(keep='last')
    4. 生成report_quarter(PeriodIndex freq='Q')
    5. (publish_date, report_quarter)排序+去重(keep='last')

    Returns:
        pd.DataFrame: 骨架表,包含列[publish_date, report_quarter, _row_id]
            - _row_id: 原始raw_fin_df的index label,用于回表取值
    """
    # 1) 只取必要键列 + 丢弃键缺失(必须先做,避免dt/PeriodIndex出错)
    key_df = raw_fin_df[['publish_date', 'report_date']].dropna(
        subset=['publish_date', 'report_date']
    ).copy()

    # 2) 保存原始行号(用于回表取值)
    key_df['_row_id'] = key_df.index

    # 3) (publish_date, report_date)唯一化:排序+去重(与旧逻辑一致keep='last')
    key_df = key_df.sort_values(['publish_date', 'report_date'])
    key_df = key_df.drop_duplicates(['publish_date', 'report_date'], keep='last')

    # 4) 生成report_quarter(与旧逻辑一致,PeriodIndex freq='Q')
    key_df = _add_report_quarter(key_df)

    # 5) (publish_date, report_quarter)唯一化:排序+去重(与旧逻辑一致keep='last')
    key_df = key_df.sort_values(['publish_date', 'report_quarter'])
    key_df = key_df.drop_duplicates(['publish_date', 'report_quarter'], keep='last')

    skeleton = key_df[['publish_date', 'report_quarter', '_row_id']].copy()
    return skeleton

def _build_base_pivot_from_skeleton(
    raw_fin_df: pd.DataFrame,
    skeleton: pd.DataFrame,
    col: str
) -> tuple:
    """

    使用prepare-once模式的骨架表,避免重复排序去重,直接按_row_id取值构建pivot.
    语义与_build_base_pivot_optimized()完全一致.

    Args:
        raw_fin_df: 原始财务数据(清洗后)
        skeleton: 骨架表(由_prepare_fin_pivot_skeleton生成)
        col: 财务字段名

    Returns:
        tuple: (pivot_df, is_na_mask)
            - pivot_df: 累计值pivot(inf已替换为nan)
            - is_na_mask: 布尔DataFrame,标记原始数据为NaN的位置
    """
    # 1) 按骨架回表取值(顺序与skeleton行一致)
    row_ids = skeleton['_row_id'].values
    values = raw_fin_df.loc[row_ids, col]

    # 2) 复刻旧版:转数值+NaN/非法转inf(用于保留"数据不存在"语义)
    values = pd.to_numeric(values, errors='coerce').fillna(np.inf)

    # 3) 构造pivot输入(只包含三列)
    fin_for_pivot = pd.DataFrame({
        'publish_date': skeleton['publish_date'].values,
        'report_quarter': skeleton['report_quarter'].values,
        col: values.values,
    })

    # 4) pivot(与旧版一致,重复兜底pivot_table)
    try:
        pivot = pd.pivot(
            data=fin_for_pivot,
            index='report_quarter',
            columns='publish_date',
            values=col,
        )
    except ValueError:
        pivot = fin_for_pivot.pivot_table(
            index='report_quarter',
            columns='publish_date',
            values=col,
            aggfunc='last',
        )

    # 5) 条件排序+沿publish_date前向填充
    if not pivot.columns.is_monotonic_increasing:
        pivot = pivot.sort_index(axis=1)
    if not pivot.index.is_monotonic_increasing:
        pivot = pivot.sort_index(axis=0)
    pivot = pivot.ffill(axis=1)

    # 6) inf掩码+inf->nan(与旧版一致)
    is_na_mask = np.isinf(pivot)
    pivot = pivot.replace(np.inf, np.nan)

    return pivot, is_na_mask

def _build_base_pivot_optimized(raw_fin_df: pd.DataFrame, col: str) -> tuple:
    """
    构建base pivot(累计值)- 优化版本

    注意:此函数保留用于兼容性(单列场景,未使用骨架表时的回退路径)
    多列场景请使用 _prepare_fin_pivot_skeleton() + _build_base_pivot_from_skeleton()

    实现步骤:
    1. 清洗数据:过滤NaN,排序,去重
    2. 添加report_quarter列
    3. 用inf填充NaN(保留"数据不存在"的信息)
    4. 使用pivot构建二维表(包含所有季度行)
    5. 前向填充(ffill)
    6. 跟踪_is_na掩码,最后将inf替换回nan

    Returns:
        tuple: (pivot_df, is_na_mask)
            - pivot_df: 累计值pivot(inf已替换为nan)
            - is_na_mask: 布尔DataFrame,标记原始数据为NaN的位置

    注意:
        当某季度数据不存在时,对齐到交易日应返回NaN而非前向填充的上一季度值.
    """
    fin = raw_fin_df[['publish_date', 'report_date', col]].dropna(subset=['publish_date', 'report_date']).copy()

    # 排序并去重
    fin = fin.sort_values(['publish_date', 'report_date'])
    fin = fin.drop_duplicates(['publish_date', 'report_date'], keep='last')

    fin = _add_report_quarter(fin)

    # 确保 (publish_date, report_quarter) 唯一
    fin = fin.sort_values(['publish_date', 'report_quarter'])
    fin = fin.drop_duplicates(['publish_date', 'report_quarter'], keep='last')
    fin_for_pivot = fin[['publish_date', 'report_quarter', col]].copy()
    fin_for_pivot[col] = pd.to_numeric(fin_for_pivot[col], errors='coerce').fillna(np.inf)

    # 使用pivot构建二维表
    try:
        pivot = pd.pivot(
            data=fin_for_pivot,
            index='report_quarter',
            columns='publish_date',
            values=col
        )
    except ValueError:
        # 如果有重复,回退到pivot_table
        pivot = fin_for_pivot.pivot_table(
            index='report_quarter',
            columns='publish_date',
            values=col,
            aggfunc='last'
        )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1).ffill(axis=1)
    is_na_mask = np.isinf(pivot)

    # 把inf替换回nan,确保pivot数据干净
    pivot = pivot.replace(np.inf, np.nan)

    return pivot, is_na_mask

def _build_base_pivot(raw_fin_df: pd.DataFrame, col: str) -> tuple:
    """
    构建base pivot(累计值)

    将财务数据转换为二维表格式:
    - 行索引:report_quarter(报告期)
    - 列索引:publish_date(发布日期)
    - 值:累计财务数据

    Returns:
        tuple: (pivot_df, is_na_mask)
    """
    return _build_base_pivot_optimized(raw_fin_df, col)

def _calc_quarter_pivot_vectorized(base_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    计算季度pivot(单季值)- 按年分组差分

    实现逻辑:
    1. 按年份分组,找到每年的起始位置
    2. 在同一年内,用后一季度减去前一季度得到单季值
    3. Q1的单季值 = Q1累计值(因为前面没有季度)
    4. 如果某年只有一个季度,保持NaN(数据不足,无法计算)

    注意:使用NumPy向量化操作提升性能
    """
    # 处理空 DataFrame
    if base_pivot.empty:
        return base_pivot.copy()

    # 提取年份信息
    years = base_pivot.index.year.values

    # 找到年份变化点(每年的起始位置)
    year_changes = np.concatenate([[True], np.diff(years) != 0])
    year_starts = np.where(year_changes)[0]

    # 复制数组(避免修改原数据)
    arr = base_pivot.values.copy()
    result = np.full_like(arr, np.nan, dtype=float)

    # 按年份分段处理
    for i in range(len(year_starts)):
        start = year_starts[i]
        end = year_starts[i+1] if i+1 < len(year_starts) else len(arr)

        # 该年有多个季度时,进行diff计算
        if end - start > 1:
            result[start+1:end] = arr[start+1:end] - arr[start:end-1]

        # 如果某年只有1个季度,result[start]保持NaN(数据不足,无法计算)

    # 转回 DataFrame
    quarter_df = pd.DataFrame(
        result,
        index=base_pivot.index,
        columns=base_pivot.columns
    )

    # Q1的单季值 = Q1累计值
    q1_mask = quarter_df.index.quarter == 1
    quarter_df.loc[q1_mask] = quarter_df.loc[q1_mask].fillna(
        base_pivot.loc[q1_mask]
    )

    return quarter_df

def _calc_quarter_pivot(base_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    计算季度pivot(单季值)

    将累计值转换为单季值:
    - Q1单季 = Q1累计
    - Q2单季 = Q2累计 - Q1累计(同年内)
    - Q3单季 = Q3累计 - Q2累计(同年内)
    - Q4单季 = Q4累计 - Q3累计(同年内)
    """
    return _calc_quarter_pivot_vectorized(base_pivot)

def _calc_ttm_pivot(quarter_pivot: pd.DataFrame, base_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    计算TTM pivot(滚动4季度累计值)

    逻辑:
    1. 对单季值进行rolling(4).sum()计算TTM
    2. 当Q4的TTM为NaN时,用年报累计值填充(兼容数据不足4季度的情况)

    参数:
        quarter_pivot: 单季值pivot(由_calc_quarter_pivot计算)
        base_pivot: 原始累计值pivot(用于填充Q4的NaN)

    返回:
        pd.DataFrame: TTM pivot
    """
    # 1. 基础TTM计算(rolling 4季度)
    ttm_pivot = quarter_pivot.rolling(4, min_periods=4).sum()

    # 2. Q4填充逻辑(复刻旧版finance_manager.py的行为)
    # 当Q4的TTM为NaN时,说明该季度数据不足4个季度
    # 此时用年报累计值(base_pivot)填充,提高数据覆盖率
    # 财务原理:Q4年报 = Q1单季+Q2单季+Q3单季+Q4单季 = TTM(在Q4时点)
    mask = ttm_pivot.isna() & (base_pivot.index.quarter == 4)[:, np.newaxis]
    ttm_pivot[mask] = base_pivot[mask]

    return ttm_pivot

def _calc_annual_pivot(base_pivot: pd.DataFrame) -> pd.DataFrame:
    """提取年报数据(Q4)"""
    idx = base_pivot.index
    is_q4 = (idx.quarter == 4)
    annual = base_pivot[is_q4].copy()
    return annual

def _build_base_pivot_from_prepared(
    prepared_fin: pd.DataFrame,
    col: str
) -> pd.DataFrame:
    """
    从已准备好的数据构建base pivot

    参数:
        prepared_fin: 已经排序,添加了report_quarter,去重的数据
        col: 列名

    前置条件:
    - prepared_fin已经排序(by publish_date, report_date)
    - 已添加report_quarter列
    - 已去重(by publish_date, report_quarter)

    实现:直接pivot + ffill
    """
    # pivot(已经保证唯一性)
    try:
        pivot = prepared_fin.pivot(
            index='report_quarter',
            columns='publish_date',
            values=col
        )
    except ValueError as e:
        # 如果仍有重复,回退到 pivot_table
        import warnings
        warnings.warn(f"pivot失败,回退到pivot_table: {e}")
        pivot = prepared_fin.pivot_table(
            index='report_quarter',
            columns='publish_date',
            values=col,
            aggfunc='last'
        )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1).ffill(axis=1)

    return pivot

def _find_fin_folder(stock_code: str, base: Path) -> Optional[Path]:
    candidates = [stock_code]
    if len(stock_code) > 2:
        pref = stock_code[:2].lower()
        num = stock_code[2:]
        up = pref.upper()
        candidates += [
            num,
            f"{num}.{up}",
            f"{up}{num}",
        ]
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None

# =============================
# CSV读取辅助:编码探测 & 目录级表头缓存
# =============================
_FMV2_REPORT_DATE_CANDIDATES: List[str] = [
    'report_date', 'report_dt', 'reportday', 'reportdate', 'statement_report_date', '报表日期', '报告期'
]
_FMV2_PUBLISH_DATE_CANDIDATES: List[str] = [
    'publish_date', 'publish_dt', 'publish_de', 'pub_date', '公告日期', '披露日期', 'publish_time'
]

def _is_valid_utf8_bytes(data: bytes) -> bool:
    """零异常UTF-8合法性检测(仅用于区分UTF-8/GBK)"""
    i = 0
    n = len(data)
    while i < n:
        b0 = data[i]
        if b0 <= 0x7F:
            i += 1
            continue
        if 0xC2 <= b0 <= 0xDF:
            if i + 1 >= n:
                return True
            if (data[i + 1] & 0xC0) != 0x80:
                return False
            i += 2
            continue
        if 0xE0 <= b0 <= 0xEF:
            if i + 2 >= n:
                return True
            b1 = data[i + 1]
            b2 = data[i + 2]
            if (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80:
                return False
            if b0 == 0xE0 and b1 < 0xA0:
                return False
            if b0 == 0xED and b1 > 0x9F:
                return False
            i += 3
            continue
        if 0xF0 <= b0 <= 0xF4:
            if i + 3 >= n:
                return True
            b1 = data[i + 1]
            b2 = data[i + 2]
            b3 = data[i + 3]
            if (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80 or (b3 & 0xC0) != 0x80:
                return False
            if b0 == 0xF0 and b1 < 0x90:
                return False
            if b0 == 0xF4 and b1 > 0x8F:
                return False
            i += 4
            continue
        return False
    return True

def _detect_fin_csv_encoding(file: Path, sample_size: int = 65536) -> str:
    """零异常编码探测:读取首段字节判断UTF-8/GBK"""
    try:
        with open(file, 'rb') as f:
            head = f.read(sample_size)
    except Exception:
        return 'gbk'
    if head.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    return 'utf-8' if _is_valid_utf8_bytes(head) else 'gbk'

def _safe_mtime_ns(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except Exception:
        return -1

def _get_fin_csv_header_cached(fin_dir: Path, sample_file: Path, encoding: str) -> Optional[List[str]]:
    """目录级缓存CSV表头,避免重复nrows=0读盘(仅同名文件复用缓存)"""
    key = (str(fin_dir), encoding)
    dir_mtime_ns = _safe_mtime_ns(fin_dir)

    cached = _FIN_CSV_HEADER_CACHE.get(key)
    if cached is not None:
        cached_dir_mtime_ns, cached_sample_name, cached_sample_mtime_ns, cached_cols = cached
        # 方案A:仅当同名文件时才复用缓存,避免多CSV目录静默丢列
        if cached_sample_name == sample_file.name and cached_dir_mtime_ns == dir_mtime_ns:
            if _safe_mtime_ns(sample_file) == cached_sample_mtime_ns:
                return cached_cols

    try:
        header_df = pd.read_csv(sample_file, encoding=encoding, skiprows=1, nrows=0)
        cols = list(header_df.columns)
    except Exception:
        return None

    _FIN_CSV_HEADER_CACHE[key] = (dir_mtime_ns, sample_file.name, _safe_mtime_ns(sample_file), cols)
    return cols

def _normalize_date_columns_fmv2(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    report_candidates = _FMV2_REPORT_DATE_CANDIDATES
    publish_candidates = _FMV2_PUBLISH_DATE_CANDIDATES
    def pick(name_list):
        for n in name_list:
            if n in cols:
                return cols[n]
        return None
    rep_col = pick(report_candidates)
    pub_col = pick(publish_candidates)
    if rep_col is not None and rep_col != 'report_date':
        df['report_date'] = df[rep_col]
    elif 'report_date' not in df.columns:
        df['report_date'] = pd.NaT
    if pub_col is not None and pub_col != 'publish_date':
        df['publish_date'] = df[pub_col]
    elif 'publish_date' not in df.columns:
        df['publish_date'] = pd.NaT
    def _parse_ymd_like(s: pd.Series) -> pd.Series:
        try:
            if s.dtype.kind in 'iuf':
                v = s.copy()
                v = pd.to_numeric(v, errors='coerce').astype('Int64')
                strv = v.astype(str).str.replace('<NA>', '', regex=False).str.zfill(8)
                return pd.to_datetime(strv, format='%Y%m%d', errors='coerce')
            else:
                strv = s.astype(str).str.strip()
                mask8 = strv.str.fullmatch(r"\d{8}")
                out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
                if mask8.any():
                    out.loc[mask8] = pd.to_datetime(strv[mask8], format='%Y%m%d', errors='coerce')
                if (~mask8).any():
                    out.loc[~mask8] = pd.to_datetime(strv[~mask8], errors='coerce')
                return out
        except Exception:
            return pd.to_datetime(s, errors='coerce')
    df['report_date'] = _parse_ymd_like(df['report_date'])
    df['publish_date'] = _parse_ymd_like(df['publish_date'])
    return df

def _load_raw_fin_df(stock_code: str, fin_cols: List[str]) -> pd.DataFrame:
    """加载原始财务CSV数据(优化版:usecols callable + 零异常编码检测)"""
    try:
        import config as cfg
    except Exception:
        cfg = None
    out_cols = ['publish_date', 'report_date'] + list(fin_cols)
    if cfg is None or not hasattr(cfg, 'data_center_path'):
        return pd.DataFrame(columns=out_cols)
    fin_data_path = Path(cfg.data_center_path) / 'stock-fin-data-xbx'
    fin_folder = _find_fin_folder(stock_code, fin_data_path)
    if fin_folder is None:
        return pd.DataFrame(columns=out_cols)

    fin_cols_unique = list(dict.fromkeys(fin_cols))
    fin_cols_lower = {c.lower() for c in fin_cols_unique}
    # 日期候选列名(小写)
    date_cols_lower = {c.lower() for c in _FMV2_REPORT_DATE_CANDIDATES + _FMV2_PUBLISH_DATE_CANDIDATES}
    needed_lower = fin_cols_lower | date_cols_lower

    frames: List[pd.DataFrame] = []

    for file in fin_folder.glob('*.csv'):
        encoding = _detect_fin_csv_encoding(file)

        # usecols使用callable形式,无需预读表头
        def usecols_filter(col: str) -> bool:
            return col.lower() in needed_lower

        df = None
        try:
            df = pd.read_csv(file, encoding=encoding, skiprows=1, usecols=usecols_filter)
        except Exception:
            # 回退:不带usecols
            try:
                df = pd.read_csv(file, encoding=encoding, skiprows=1)
            except Exception:
                alt_enc = 'gbk' if encoding.startswith('utf-8') else 'utf-8'
                try:
                    df = pd.read_csv(file, encoding=alt_enc, skiprows=1)
                except Exception:
                    try:
                        df = pd.read_csv(file, encoding='latin1', skiprows=1)
                    except Exception:
                        df = None

        if df is None:
            continue

        df = _normalize_date_columns_fmv2(df)

        if fin_cols_unique:
            missing = [c for c in fin_cols_unique if c not in df.columns]
            for c in missing:
                df[c] = np.nan
            df[fin_cols_unique] = df[fin_cols_unique].apply(pd.to_numeric, errors='coerce')

        frames.append(df[['publish_date', 'report_date', *fin_cols]].copy())

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=out_cols)

class PrecomputedFinanceData:
    """预计算的财务数据容器 - 支持所有标签
    - 当前因子库6/6只用quarter(),ttm/annual完全未使用
    - 懒加载可节省约4.5ms/股 * 5726股 = 26秒/次全量回测
    """

    def __init__(self, stock_code: str, fin_cols: List[str]):
        self.stock_code = stock_code
        # 当前已经支持的财务字段列表(按需增量扩展)
        self.fin_cols = list(fin_cols)
        self.pivots = {}  # 累计/原始数据
        self.quarter_pivots = {}  # 单季数据
        self.ttm_pivots = {}  # TTM数据(
        self.annual_pivots = {}  # 年报数据(
        self.is_na_pivots = {}
        self.raw_df = None

    def precompute(self, raw_fin_df: pd.DataFrame, pivot_dict: Optional[Dict] = None):
        """
        预计算所有需要的pivot表

        计算流程:
        1. 基础pivot(累计值)
        2. 季度pivot(单季值)
        3. TTM pivot -
        4. 年报pivot -

        - 当 pivot_dict 存在时,直接复用框架已计算的pivot,跳过V5重复计算
        - 仅需计算 quarter_pivot(框架未提供)
        - 预估收益:25-40ms/股(避免skeleton构建+pivot计算)
        """
        self.ttm_pivots.clear()
        self.annual_pivots.clear()
        raw_fin_df = raw_fin_df.copy()
        self.raw_df = raw_fin_df

        # 容错:检查缺失字段并自动创建空列
        missing_cols = [col for col in self.fin_cols if col not in raw_fin_df.columns]
        if missing_cols:
            logger.warning(
                f"财务字段缺失:{missing_cols}(股票:{self.stock_code})- "
                f"已自动创建空列"
            )
            for col in missing_cols:
                raw_fin_df[col] = np.nan
        skeleton = None if pivot_dict is not None else _prepare_fin_pivot_skeleton(raw_fin_df)

        # 为每个列计算base和quarter pivot
        for col in self.fin_cols:
            if pivot_dict is not None and col in pivot_dict:
                base_pivot = pivot_dict[col]
                is_na_mask = pivot_dict.get(f"{col}_is_na")
                if is_na_mask is None:
                    is_na_mask = pd.DataFrame(False, index=base_pivot.index, columns=base_pivot.columns)
            else:
                # 原有逻辑:skeleton构建(确保skeleton已创建)
                if skeleton is None:
                    skeleton = _prepare_fin_pivot_skeleton(raw_fin_df)
                base_pivot, is_na_mask = _build_base_pivot_from_skeleton(raw_fin_df, skeleton, col)

            self.pivots[col] = base_pivot
            self.is_na_pivots[col] = is_na_mask

            # 季度pivot(单季值)
            quarter_pivot = _calc_quarter_pivot(base_pivot)
            self.quarter_pivots[col] = quarter_pivot

    def get_ttm_pivot(self, col: str) -> pd.DataFrame:
        """获取TTM pivot(懒加载)"""
        if col not in self.ttm_pivots:
            base_pivot = self.pivots.get(col)
            quarter_pivot = self.quarter_pivots.get(col)
            if base_pivot is None or quarter_pivot is None:
                return pd.DataFrame()
            self.ttm_pivots[col] = _calc_ttm_pivot(quarter_pivot, base_pivot)
        return self.ttm_pivots[col]

    def get_annual_pivot(self, col: str) -> pd.DataFrame:
        """获取年报pivot(懒加载)"""
        if col not in self.annual_pivots:
            base_pivot = self.pivots.get(col)
            if base_pivot is None:
                return pd.DataFrame()
            self.annual_pivots[col] = _calc_annual_pivot(base_pivot)
        return self.annual_pivots[col]

# 【方案B】已删除全局预计算缓存 - 在不改框架的场景下,全局缓存因碎片化和多进程隔离而无效

class FactorFinanceManagerV4Fast:
    """极速版财务管理器 - 使用预计算策略"""
    
    def __init__(
        self,
        precomputed_data: PrecomputedFinanceData,
        trade_date_df: pd.DataFrame,
        stock_code: str = '',
    ):
        self.precomputed = precomputed_data
        if trade_date_df is None:
            trade_date_df = pd.DataFrame({'交易日期': pd.Series([], dtype='datetime64[ns]')})
        self.trade_date_df = trade_date_df[['交易日期']].copy()
        self.stock_code = stock_code
        self.raw_fin_cols = precomputed_data.fin_cols
        # 跨实例共享的对齐缓存:键 (col, kind, shift) -> 对齐后的 Series
        self._aligned_cache: dict[tuple[str, str, int], pd.Series] = {}

        # ===================== 预构建索引映射表 =====================
        # 原始映射:publish_date * report_quarter -> report_date(唯一化,去重)
        raw_src = self.precomputed.raw_df
        if raw_src is None:
            # 兜底:空数据保护
            self._raw_map_df = pd.DataFrame(columns=['publish_date', 'report_quarter', 'report_date'])
            self._trade_dates_int64_sorted = np.array([], dtype=np.int64)
            self._sort_idx = None
        else:
            # 内存优化:避免不必要的copy
            raw = raw_src[['publish_date', 'report_date']]
            raw = _ensure_datetime(raw, ['publish_date', 'report_date']) if hasattr(raw, 'columns') else raw
            raw = _add_report_quarter(raw)
            need_cols = ['publish_date', 'report_quarter', 'report_date']
            self._raw_map_df = (
                raw[need_cols]
                .drop_duplicates(need_cols[:2], keep='last')
                .sort_values(['publish_date', 'report_quarter'])
            )
            trade_dates = self.trade_date_df['交易日期'].values

            # 检查是否已排序(P2#08:改为局部变量,后续不再使用)
            is_sorted = (
                len(trade_dates) <= 1 or
                np.all(trade_dates[:-1] <= trade_dates[1:])
            )

            if is_sorted:
                # 已排序,直接转int64缓存
                self._trade_dates_int64_sorted = trade_dates.astype('datetime64[ns]').astype(np.int64)
                self._sort_idx = None  # 无需排序映射
            else:
                # 未排序,构建排序索引映射
                self._sort_idx = np.argsort(trade_dates)
                sorted_dates = trade_dates[self._sort_idx]
                self._trade_dates_int64_sorted = sorted_dates.astype('datetime64[ns]').astype(np.int64)

    def __getitem__(self, col: str) -> 'FactorFinanceSeriesV4Fast':
        if col not in self.raw_fin_cols:
            raise KeyError(f"列 '{col}' 不存在")
        return FactorFinanceSeriesV4Fast(self, col)

    def _perform_stack_merge_dedup(
        self,
        pivot_df: pd.DataFrame,
        value_name: str,
        is_na_mask: pd.DataFrame = None
    ) -> pd.DataFrame:
        """执行stack+merge+dedup操作(提取为独立方法)

        这个方法封装了昂贵的数据转换流程:
        1. stack:pivot转long format
        2. merge:与_raw_map_df合并,过滤非法组合
        3. dedup:同一发布日保留最新季度
        - 在stack前用inf标记原NaN位置(保留"数据不存在"的信息)
        - 对齐后将inf替换回nan
        - 与原版finance_manager.py的__to_trade_time保持一致
        """
        # 列名标准化为datetime
        needs_copy = False
        if hasattr(pivot_df.columns, 'to_timestamp'):
            needs_copy = True
            new_cols = pivot_df.columns.to_timestamp()
        elif not pd.api.types.is_datetime64_any_dtype(pivot_df.columns):
            try:
                needs_copy = True
                new_cols = pd.to_datetime(pivot_df.columns)
            except Exception:
                needs_copy = False

        if needs_copy:
            pivot_df = pivot_df.copy()
            pivot_df.columns = new_cols
        if is_na_mask is not None:
            pivot_df = pivot_df.copy()
            # 同步is_na_mask的列名
            is_na_aligned = is_na_mask.copy()
            if hasattr(is_na_aligned.columns, 'to_timestamp'):
                is_na_aligned.columns = is_na_aligned.columns.to_timestamp()
            elif not pd.api.types.is_datetime64_any_dtype(is_na_aligned.columns):
                try:
                    is_na_aligned.columns = pd.to_datetime(is_na_aligned.columns)
                except Exception:
                    pass
            # 用inf标记原NaN位置
            pivot_df[is_na_aligned] = np.inf

        # Stack:pivot转long format
        stacked = pivot_df.stack(dropna=False).reset_index()
        stacked.columns = ['report_quarter', 'publish_date', value_name]

        if stacked.empty:
            return pd.DataFrame(columns=['publish_date', 'report_date', value_name])

        
        # 解决:已披露但字段为空时不再回退旧季度,而是返回NaN(保证跨字段同期)
        stacked = stacked.copy()
        if is_na_mask is not None:
            # 关键:在replace(inf->nan)之前计算_is_published
            # inf标记表示"已披露但字段为空",也应视为已披露
            stacked['_is_published'] = stacked[value_name].notna() | np.isinf(stacked[value_name])
            # 然后清理inf,避免泄漏
            stacked[value_name] = stacked[value_name].replace(np.inf, np.nan)
        else:
            # 无mask时降级:只能用notna判断
            stacked['_is_published'] = stacked[value_name].notna()

        stacked = stacked.sort_values(
            ['publish_date', '_is_published', 'report_quarter'],
            ascending=[True, False, False]
        )
        total_df = stacked.drop_duplicates('publish_date', keep='first').drop(columns=['_is_published'])

        # 从report_quarter映射report_date(每个季度对应唯一的report_date)
        quarter_to_date = self._raw_map_df.drop_duplicates(
            'report_quarter', keep='last'
        ).set_index('report_quarter')['report_date']
        total_df = total_df.copy()
        total_df['report_date'] = total_df['report_quarter'].map(quarter_to_date)

        total_df = total_df[['publish_date', 'report_date', value_name]]
        total_df = total_df.sort_values(['publish_date', 'report_date'])

        return total_df

    def _extract_latest_publish_values_numpy(
        self,
        pivot_df: pd.DataFrame,
        value_name: str,
        is_na_mask: pd.DataFrame = None,
    ) -> tuple:
        """
        NumPy快速路径:直接在pivot上计算每个publish_date对应的最新披露季度值

        性能优化:绕开_perform_stack_merge_dedup的stack/sort/dedup开销

        语义目标(与_perform_stack_merge_dedup完全一致):
        - 同一publish_date选择"已披露且最新"的季度(report_quarter最大)
        - "已披露"判定:value.notna() 或 is_na_mask=True
        - 若选中的单元来自is_na_mask(已披露但字段为空),则结果必须返回NaN

        Returns:
            (pub_dates_int64, pub_values): 发布日int64数组和对应值数组

        Raises:
            Exception: 任何异常都会触发上层回退到_perform_stack_merge_dedup
        """
        if pivot_df is None or pivot_df.empty:
            return np.array([], dtype=np.int64), np.array([], dtype=float)

        pivot = pivot_df

        
        if not isinstance(pivot.index, (pd.PeriodIndex, pd.DatetimeIndex)):
            raise ValueError("pivot.index must be PeriodIndex or DatetimeIndex for fast path")

        
        if not pivot.columns.is_unique:
            raise ValueError("pivot.columns must be unique for fast path")

        # 列名标准化为datetime(与_perform_stack_merge_dedup一致)
        if hasattr(pivot.columns, 'to_timestamp'):
            pivot = pivot.copy()
            pivot.columns = pivot.columns.to_timestamp()
        elif not pd.api.types.is_datetime64_any_dtype(pivot.columns):
            pivot = pivot.copy()
            pivot.columns = pd.to_datetime(pivot.columns, errors='coerce')

        # 确保行列有序:行是季度(升序,越靠后越新),列是发布日期(升序)
        if hasattr(pivot.index, 'is_monotonic_increasing') and not pivot.index.is_monotonic_increasing:
            pivot = pivot.sort_index(axis=0)
        if hasattr(pivot.columns, 'is_monotonic_increasing') and not pivot.columns.is_monotonic_increasing:
            pivot = pivot.sort_index(axis=1)

        # 对齐is_na_mask到pivot形状
        mask_arr = None
        if is_na_mask is not None:
            is_na_aligned = is_na_mask
            if hasattr(is_na_aligned.columns, 'to_timestamp'):
                is_na_aligned = is_na_aligned.copy()
                is_na_aligned.columns = is_na_aligned.columns.to_timestamp()
            elif not pd.api.types.is_datetime64_any_dtype(is_na_aligned.columns):
                is_na_aligned = is_na_aligned.copy()
                is_na_aligned.columns = pd.to_datetime(is_na_aligned.columns, errors='coerce')

            if hasattr(is_na_aligned.index, 'is_monotonic_increasing') and not is_na_aligned.index.is_monotonic_increasing:
                is_na_aligned = is_na_aligned.sort_index(axis=0)
            if hasattr(is_na_aligned.columns, 'is_monotonic_increasing') and not is_na_aligned.columns.is_monotonic_increasing:
                is_na_aligned = is_na_aligned.sort_index(axis=1)

            # 对齐到pivot的形状
            if is_na_aligned.index.equals(pivot.index) and is_na_aligned.columns.equals(pivot.columns):
                mask_arr = is_na_aligned.to_numpy(dtype=bool, copy=False)
            else:
                mask_arr = (
                    is_na_aligned.reindex(index=pivot.index, columns=pivot.columns)
                    .fillna(False)
                    .to_numpy(dtype=bool, copy=False)
                )

        values = pivot.to_numpy(dtype=float, copy=False)

        # 与旧逻辑一致:notna或mask=True都视为"已披露"
        published = ~np.isnan(values)
        if mask_arr is not None:
            published = published | mask_arr

        # 对每个publish_date(列)取"最新披露季度":从下往上找第一个True
        # 行是按季度升序排列,所以最下面的行是最新季度
        rev = published[::-1, :]  # 反转行顺序
        has_any = rev.any(axis=0)  # 每列是否有任何已披露数据
        idx_from_bottom = rev.argmax(axis=0)  # 从反转后的顶部找第一个True
        row_idx = values.shape[0] - 1 - idx_from_bottom  # 转换回原始行索引

        col_idx = np.arange(values.shape[1])
        selected = np.full(values.shape[1], np.nan, dtype=float)

        if has_any.any():
            # 只处理有数据的列
            valid_cols = has_any
            selected[valid_cols] = values[row_idx[valid_cols], col_idx[valid_cols]]

            
            if mask_arr is not None:
                # 检查选中的单元是否来自is_na_mask
                sel_mask = mask_arr[row_idx[valid_cols], col_idx[valid_cols]]
                if sel_mask.any():
                    # 这些位置已披露但字段为空,强制设为NaN
                    valid_positions = np.flatnonzero(valid_cols)
                    selected[valid_positions[sel_mask]] = np.nan

        # 构建发布日期int64数组
        pub_dates = pivot.columns.values.astype('datetime64[ns]')
        pub_dates_int64 = pub_dates.view(np.int64)

        # 清理NaT(等价于旧代码dropna(subset=['publish_date']))
        nat_val = np.datetime64('NaT').view(np.int64)
        nat_mask = pub_dates_int64 == nat_val
        if nat_mask.any():
            keep = ~nat_mask
            pub_dates_int64 = pub_dates_int64[keep]
            selected = selected[keep]

        # searchsorted要求升序(正常情况下已经是升序,这里做兜底)
        if len(pub_dates_int64) > 1 and np.any(pub_dates_int64[:-1] > pub_dates_int64[1:]):
            sorter = np.argsort(pub_dates_int64)
            pub_dates_int64 = pub_dates_int64[sorter]
            selected = selected[sorter]

        return pub_dates_int64, selected

    def _fast_align_numpy(
        self,
        pivot_df: pd.DataFrame,
        value_name: str,
        is_na_mask: pd.DataFrame = None
    ) -> pd.Series:
        """纯NumPy实现的快速对齐(

        """
        if pivot_df is None or pivot_df.empty:
            return pd.Series(index=self.trade_date_df.index, data=np.nan, name=value_name)
        try:
            pub_dates_int64, pub_values = self._extract_latest_publish_values_numpy(
                pivot_df, value_name, is_na_mask
            )
        except Exception:
            # 兼容兜底:保持旧语义(极端dtype/索引异常时)
            total_df = self._perform_stack_merge_dedup(pivot_df, value_name, is_na_mask)
            if total_df.empty:
                return pd.Series(index=self.trade_date_df.index, data=np.nan, name=value_name)
            temp = total_df[['publish_date', value_name]].dropna(subset=['publish_date'])
            pub_dates_int64 = temp['publish_date'].values.astype('datetime64[ns]').view(np.int64)
            pub_values = temp[value_name].values

        if pub_dates_int64.size == 0:
            return pd.Series(index=self.trade_date_df.index, data=np.nan, name=value_name)
        trade_dates_int64 = self._trade_dates_int64_sorted

        # 对每个交易日,找到 <= 该交易日的最近发布日
        indices = np.searchsorted(pub_dates_int64, trade_dates_int64, side='right') - 1

        # 构建结果数组(排序后的顺序)
        result_values = np.full(len(trade_dates_int64), np.nan)
        valid_mask = indices >= 0
        result_values[valid_mask] = pub_values[indices[valid_mask]]
        if self._sort_idx is not None:
            result_values_original_order = np.empty_like(result_values)
            result_values_original_order[self._sort_idx] = result_values
            result_values = result_values_original_order

        return pd.Series(result_values, index=self.trade_date_df.index, name=value_name)

    def _fast_align(
        self,
        pivot_df: pd.DataFrame,
        value_name: str,
        is_na_mask: pd.DataFrame = None
    ) -> pd.Series:
        """
        快速对齐方法(混合策略)

        使用NumPy searchsorted加速对齐操作,失败时回退到pandas版本

        """
        try:
            res = self._fast_align_numpy(pivot_df, value_name, is_na_mask)
        except Exception as e:
            import warnings
            warnings.warn(f"NumPy加速失败,回退到pandas: {e}", RuntimeWarning)
            res = self._fast_align_pandas(pivot_df, value_name, is_na_mask)
        res = res.replace([np.inf, -np.inf], np.nan)
        return res

    def _fast_align_pandas(self, pivot_df: pd.DataFrame, value_name: str, is_na_mask: pd.DataFrame = None) -> pd.Series:
        """
        使用pandas进行对齐(回退方案)

        直接在交易日上查找<=该日的最近发布日
        可以正确处理周末/假日发布的财报(在下一个交易日生效)
        """
        if pivot_df is None or pivot_df.empty:
            return pd.Series(index=self.trade_date_df.index, data=np.nan, name=value_name)
        original_index = self.trade_date_df.index.copy()

        # 列名标准化为datetime
        needs_copy = False
        if hasattr(pivot_df.columns, 'to_timestamp'):
            needs_copy = True
            new_cols = pivot_df.columns.to_timestamp()
        elif not pd.api.types.is_datetime64_any_dtype(pivot_df.columns):
            try:
                needs_copy = True
                new_cols = pd.to_datetime(pivot_df.columns)
            except Exception:
                needs_copy = False

        if needs_copy:
            pivot_df = pivot_df.copy()
            pivot_df.columns = new_cols
        if is_na_mask is not None:
            pivot_df = pivot_df.copy()
            # 同步is_na_mask的列名
            is_na_aligned = is_na_mask.copy()
            if hasattr(is_na_aligned.columns, 'to_timestamp'):
                is_na_aligned.columns = is_na_aligned.columns.to_timestamp()
            elif not pd.api.types.is_datetime64_any_dtype(is_na_aligned.columns):
                try:
                    is_na_aligned.columns = pd.to_datetime(is_na_aligned.columns)
                except Exception:
                    pass
            # 用inf标记原NaN位置
            pivot_df[is_na_aligned] = np.inf

        # 发布日 * 值
        stacked = pivot_df.stack(dropna=False).reset_index()
        stacked.columns = ['report_quarter', 'publish_date', value_name]

        if stacked.empty:
            return pd.Series(index=original_index, data=np.nan, name=value_name)

        
        # 与_perform_stack_merge_dedup保持一致
        stacked = stacked.copy()
        if is_na_mask is not None:
            # 关键:在replace(inf->nan)之前计算_is_published
            # inf标记表示"已披露但字段为空",也应视为已披露
            stacked['_is_published'] = stacked[value_name].notna() | np.isinf(stacked[value_name])
            # 然后清理inf,避免泄漏
            stacked[value_name] = stacked[value_name].replace(np.inf, np.nan)
        else:
            # 无mask时降级:只能用notna判断
            stacked['_is_published'] = stacked[value_name].notna()

        stacked = stacked.sort_values(
            ['publish_date', '_is_published', 'report_quarter'],
            ascending=[True, False, False]
        )
        total_df = stacked.drop_duplicates('publish_date', keep='first').drop(columns=['_is_published'])

        # 从report_quarter映射report_date(每个季度对应唯一的report_date)
        quarter_to_date = self._raw_map_df.drop_duplicates(
            'report_quarter', keep='last'
        ).set_index('report_quarter')['report_date']
        total_df = total_df.copy()
        total_df['report_date'] = total_df['report_quarter'].map(quarter_to_date)

        # 过滤列(去掉report_quarter)
        total_df = total_df[['publish_date', 'report_date', value_name]]
        total_df = total_df.sort_values(['publish_date', 'report_date'])
        if total_df.empty:
            return pd.Series(index=original_index, data=np.nan, name=value_name)
        trade_df_work = self.trade_date_df.copy()
        trade_df_work['_row_id'] = np.arange(len(trade_df_work))

        # 使用merge_asof对齐到交易日
        temp = total_df[['publish_date', value_name]].dropna(subset=['publish_date'])
        temp['publish_date'] = pd.to_datetime(temp['publish_date'])

        # 排序(merge_asof要求左表按key排序)
        trade_df_sorted = trade_df_work.sort_values('交易日期')

        aligned = pd.merge_asof(
            trade_df_sorted,
            temp.sort_values('publish_date'),
            left_on='交易日期', right_on='publish_date', direction='backward'
        )
        aligned = aligned.sort_values('_row_id')
        return pd.Series(
            aligned[value_name].values,
            index=original_index,
            name=value_name
        )

# =============================
# V5 Series子类实现
# =============================
class FactorFinanceSeriesV5Fast(pd.Series):
    """V5财务数据Series - 继承pd.Series,支持链式调用

    设计理念:
    - 既有pandas方法(.fillna(), .head()等)
    - 又有财务方法(.yoy(), .qoq()等)
    - 通过继承pd.Series实现完美兼容

    实现策略:
    - 数据存储:继承pd.Series的数据结构(已对齐到交易日)
    - 财务方法:委托给FactorFinanceSeriesV4Fast的计算逻辑
    """

    # pandas元数据传递机制(确保操作后保持子类类型)
    _metadata = ['manager', 'col', '_pivot_kind', '_v4_cache']

    @property
    def _constructor(self):
        """确保pandas操作后返回FactorFinanceSeriesV5Fast类型"""
        return FactorFinanceSeriesV5Fast

    def __init__(self, data=None, manager=None, col=None, **kwargs):
        """初始化Series子类

        Args:
            data: pd.Series或数组数据(已对齐到交易日)
            manager: FactorFinanceManagerV4Fast实例
            col: 财务字段名
            **kwargs: 传递给pd.Series的其他参数
        """
        # 初始化pd.Series
        if data is None:
            data = pd.Series(dtype=float)
        super().__init__(data, **kwargs)

        # 保存属性
        self.manager = manager
        self.col = col
        self._pivot_kind = None  # 'raw', 'quarter', 'ttm', 'annual'
        self._v4_cache = None

    # ===================== 财务方法委托 =====================
    # 通过创建临时FactorFinanceSeriesV4Fast对象调用原有计算逻辑

    def _create_temp_v4_object(self) -> 'FactorFinanceSeriesV4Fast':
        """创建或复用V4对象(P2优化：缓存复用)"""
        if self._v4_cache is None:
            self._v4_cache = FactorFinanceSeriesV4Fast(self.manager, self.col)
        # 每次调用更新_pivot_kind,确保使用正确的视图
        self._v4_cache._pivot_kind = self._pivot_kind
        return self._v4_cache

    def _wrap_result(self, result: pd.Series) -> 'FactorFinanceSeriesV5Fast':
        """包装计算结果为V5 Series"""
        new_obj = FactorFinanceSeriesV5Fast(
            result,
            manager=self.manager,
            col=self.col,
            index=result.index,
            name=self.col
        )
        new_obj._pivot_kind = self._pivot_kind
        return new_obj

    # ===================== 算术运算符兜底:清洗正负无穷 =====================

    @staticmethod
    def _clean_inf(series: pd.Series) -> pd.Series:
        """兜底清洗: 将运算产生的正负无穷替换为NaN

        适用场景:
        - 除零:1/0 = inf
        - 整除零:1//0 = inf
        - 无效幂:0**(-1) = inf
        - 取模零:1%0 = inf

        Returns:
            清洗后的Series(类型通过_constructor保持为V5)
        """
        return series.replace([np.inf, -np.inf], np.nan)

    def __truediv__(self, other):
        """除法兜底:防止分母为0产生inf"""
        return self._clean_inf(super().__truediv__(other))

    def __rtruediv__(self, other):
        """反向除法兜底:防止分母为0产生inf"""
        return self._clean_inf(super().__rtruediv__(other))

    def __floordiv__(self, other):
        """整除兜底:防止分母为0产生inf"""
        return self._clean_inf(super().__floordiv__(other))

    def __rfloordiv__(self, other):
        """反向整除兜底:防止分母为0产生inf"""
        return self._clean_inf(super().__rfloordiv__(other))

    def __pow__(self, other):
        """幂运算兜底:防止0的负幂产生inf"""
        return self._clean_inf(super().__pow__(other))

    def __rpow__(self, other):
        """反向幂运算兜底:防止0的负幂产生inf"""
        return self._clean_inf(super().__rpow__(other))

    def __mod__(self, other):
        """取模兜底:防止分母为0产生inf"""
        return self._clean_inf(super().__mod__(other))

    def __rmod__(self, other):
        """反向取模兜底:防止分母为0产生inf"""
        return self._clean_inf(super().__rmod__(other))

    # ===================== 财务方法委托(原有代码)=====================

    def yoy(self, exclude_negative: bool = False) -> 'FactorFinanceSeriesV5Fast':
        """同比增长(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.yoy(exclude_negative)
        return self._wrap_result(result)

    def qoq(self, exclude_negative: bool = False) -> 'FactorFinanceSeriesV5Fast':
        """环比增长(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.qoq(exclude_negative)
        return self._wrap_result(result)

    def yoy_accel(self, lag: int = 1) -> 'FactorFinanceSeriesV5Fast':
        """同比增速加速度(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.yoy_accel(lag)
        return self._wrap_result(result)

    def qoq_accel(self, lag: int = 1) -> 'FactorFinanceSeriesV5Fast':
        """环比增速加速度(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.qoq_accel(lag)
        return self._wrap_result(result)

    def y_diff(self) -> 'FactorFinanceSeriesV5Fast':
        """年度差额(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.y_diff()
        return self._wrap_result(result)

    def q_diff(self) -> 'FactorFinanceSeriesV5Fast':
        """季度差额(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.q_diff()
        return self._wrap_result(result)

    def last_y(self, n: int = 1) -> 'FactorFinanceSeriesV5Fast':
        """N年前同季数据(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.last_y(n)
        return self._wrap_result(result)

    def last_q(self, n: int = 1) -> 'FactorFinanceSeriesV5Fast':
        """N季前数据(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.last_q(n)
        return self._wrap_result(result)

    def last_q4(self, n: int = 1) -> 'FactorFinanceSeriesV5Fast':
        """N年前年报数据(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.last_q4(n)
        return self._wrap_result(result)

    def cagr(self, n: int, exclude_negative: bool = False) -> 'FactorFinanceSeriesV5Fast':
        """复合年化增长率(n为年数,与stability统一)"""
        temp = self._create_temp_v4_object()
        result = temp.cagr(n, exclude_negative)
        return self._wrap_result(result)

    def stability(self, periods: Union[int, List[int]]) -> 'FactorFinanceSeriesV5Fast':
        """稳定性指标(委托给V4计算逻辑)"""
        temp = self._create_temp_v4_object()
        result = temp.stability(periods)
        return self._wrap_result(result)

    def get_q(self, n: int) -> 'FactorFinanceSeriesV5Fast':
        """获取最近n个季度数据(委托给V4计算逻辑 - 优化版)"""
        temp = self._create_temp_v4_object()
        result = temp.get_q(n)
        return self._wrap_result(result)

    def quarter_df(self) -> QuarterDataFrame:
        """返回季度层DataFrame操作器(委托给V4)

        Returns:
            QuarterDataFrame: 季度数据操作器,支持链式调用和运算符

        Example:
            >>> # SUE因子(仅3行,代码减少90%)
            >>> qdf = fin_mgr['R_np@xbx'].quarter_df()
            >>> delta = qdf.diff()
            >>> df['SUE'] = ((qdf - qdf.shift(1) - delta.rolling(8).mean()) / delta.rolling(8).std()).to_series()
        """
        temp = self._create_temp_v4_object()
        return temp.quarter_df()

class FactorFinanceSeriesV4Fast:
    """极速版计算引擎 - 完全兼容彩虹版接口"""

    def __init__(self, manager: FactorFinanceManagerV4Fast, col: str):
        self.manager = manager
        self.col = col
        # 记录当前数据视图(raw/quarter/ttm)
        # None 表示未选择视图,仅支持 tag 接口
        self._pivot_kind: str | None = None
        # 实例不再持有缓存,改为走 manager 级共享缓存

    def _check_view_selected(self):
        """检查用户是否调用了视图方法(raw/quarter/ttm/annual)"""
        if self._pivot_kind is None:
            error_msg = (
                f"非规范用法:财务数据对象 '{self.col}' 必须先调用视图方法！\n"
                f"   正确示例:fin_mgr['{self.col}'].raw() 或 .quarter() 或 .ttm() 或 .annual()\n"
                f"   错误示例:fin_mgr['{self.col}'](缺少视图方法调用)\n"
                f"   详细说明请查看:因子库/新版财务因子示例_V5_FastX_Dem.py"
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def _is_stock_indicator_col(col: str) -> bool:
        """判断是否为存量指标(资产负债表 B_*).

        说明:
        - 按项目约定:R_*(利润表,流量),C_*(现金流,流量),B_*(资产负债表,存量)
        - 本方法仅基于命名约定做轻量判断,避免引入额外配置依赖
        """
        if not col:
            return False
        base = str(col).split('@', 1)[0]
        return base.upper().startswith('B_')

    def _assert_supports_flow_view(self, view_name: str) -> None:
        """断言当前字段支持流量视图(quarter/ttm).

        存量指标(资产负债表 B_*)是时点值,不存在"单季"与"TTM"的自然定义.
        若允许调用会产生静默的金融语义错误(例如对季度末资产做rolling(4).sum).
        """
        if not self._is_stock_indicator_col(self.col):
            return

        raise ValueError(
            f"字段 '{self.col}' 为资产负债表存量指标(B_*,时点值),不支持 {view_name}() 视图.\n"
            "原因:quarter()/ttm() 是面向流量指标(利润表 R_* / 现金流 C_*)的'累计->单季/TTM'转换.\n"
            "建议:\n"
            "  - 使用 raw() 获取季度末时点值;或使用 annual() 获取年末值.\n"
            "  - 如需季度变动额,可用 raw().q_diff() 或 raw().quarter_df().diff().to_series()."
        )

    # ===================== 算术运算符兜底:清洗正负无穷 =====================

    @staticmethod
    def _clean_inf(series: pd.Series) -> pd.Series:
        """兜底清洗: 将运算产生的正负无穷替换为NaN

        适用场景:
        - 除零:1/0 = inf
        - 整除零:1//0 = inf
        - 无效幂:0**(-1) = inf
        - 取模零:1%0 = inf

        Returns:
            清洗后的Series
        """
        return series.replace([np.inf, -np.inf], np.nan)

    # ===================== 运算符重载 - 实现彩虹版接口完全兼容 =====================
    def __truediv__(self, other):
        """除法: fin_data["col1"].quarter() / fin_data["col2"].quarter()"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return self._clean_inf(left / right)

    def __rtruediv__(self, other):
        """反向除法"""
        self._check_view_selected()
        right = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return self._clean_inf(other / right)

    def __mul__(self, other):
        """乘法"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left * right

    def __rmul__(self, other):
        """反向乘法"""
        return self.__mul__(other)

    def __add__(self, other):
        """加法"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left + right

    def __radd__(self, other):
        """反向加法"""
        return self.__add__(other)

    def __sub__(self, other):
        """减法"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left - right

    def __rsub__(self, other):
        """反向减法"""
        self._check_view_selected()
        right = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return other - right

    def __pow__(self, other):
        """幂运算"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return self._clean_inf(left ** right)

    def __rpow__(self, other):
        """反向幂运算"""
        self._check_view_selected()
        right = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return self._clean_inf(other ** right)

    def __mod__(self, other):
        """取模"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return self._clean_inf(left % right)

    def __rmod__(self, other):
        """反向取模"""
        self._check_view_selected()
        right = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return self._clean_inf(other % right)

    def __floordiv__(self, other):
        """整除"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return self._clean_inf(left // right)

    def __rfloordiv__(self, other):
        """反向整除"""
        self._check_view_selected()
        right = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return self._clean_inf(other // right)

    # 比较运算符
    def __lt__(self, other):
        """小于"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left < right

    def __le__(self, other):
        """小于等于"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left <= right

    def __gt__(self, other):
        """大于"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left > right

    def __ge__(self, other):
        """大于等于"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left >= right

    def __eq__(self, other):
        """等于"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left == right

    def __ne__(self, other):
        """不等于"""
        self._check_view_selected()
        if isinstance(other, FactorFinanceSeriesV4Fast):
            other._check_view_selected()
        left = self._get_aligned(self._pivot_kind or 'quarter', 0)
        right = other._get_aligned(other._pivot_kind or 'quarter', 0) if isinstance(other, FactorFinanceSeriesV4Fast) else other
        return left != right

    # 一元运算符
    def __neg__(self):
        """负号"""
        self._check_view_selected()
        data = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return -data

    def __pos__(self):
        """正号"""
        self._check_view_selected()
        data = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return +data

    def __abs__(self):
        """绝对值"""
        self._check_view_selected()
        data = self._get_aligned(self._pivot_kind or 'quarter', 0)
        return abs(data)

    # =========================
    # 兼容彩虹库的接口(参考 Hybrid 实现)
    # =========================
    def _get_pivot(self, kind: str) -> pd.DataFrame:
        """根据视图类型返回对应的pivot DataFrame
        """
        if kind == 'raw':
            return self.manager.precomputed.pivots.get(self.col, pd.DataFrame())
        if kind == 'quarter':
            return self.manager.precomputed.quarter_pivots.get(self.col, pd.DataFrame())
        if kind == 'ttm':
            return self.manager.precomputed.get_ttm_pivot(self.col)
        if kind == 'annual':
            return self.manager.precomputed.get_annual_pivot(self.col)
        return pd.DataFrame()

    def _get_aligned(self, kind: str, shift: int = 0) -> pd.Series:
        """获取对齐后的序列,支持在 pivot 维度的位移(不改变语义).
        """
        key = (self.col, kind, int(shift))
        cached = self.manager._aligned_cache.get(key)
        if cached is not None:
            return cached

        pivot_base = self._get_pivot(kind)
        # 空 pivot:缓存空值,避免重复 miss
        if pivot_base is None or pivot_base.empty:
            empty = pd.Series(index=self.manager.trade_date_df.index, data=np.nan, name=self.col)
            self.manager._aligned_cache[key] = empty
            return empty
        is_na_mask = self.manager.precomputed.is_na_pivots.get(self.col)

        # 按需计算并缓存
        pivot = pivot_base.shift(int(shift), axis=0) if int(shift) != 0 else pivot_base
        # 如果有shift,is_na_mask也需要同步shift
        if int(shift) != 0 and is_na_mask is not None:
            is_na_mask = is_na_mask.shift(int(shift), axis=0)

        # 修复：派生视图（ttm/quarter/annual）中"原始已发布但计算结果为NaN"的问题
        # 场景：2014Q1原始数据存在(base_pivot有值)，但TTM因缺少前置季度而为NaN
        # 期望：返回NaN，而非回退到2013Q4的TTM值
        # 关键：用base_pivot.notna()判断"原始数据是否存在"，而非is_na_mask
        effective_is_na_mask = is_na_mask
        if kind in ('ttm', 'quarter', 'annual') and is_na_mask is not None:
            # 获取原始累计值pivot，判断原始数据是否存在
            base_pivot = self.manager.precomputed.pivots.get(self.col)
            if base_pivot is not None:
                # 对齐shift（如果有）
                if int(shift) != 0:
                    base_pivot = base_pivot.shift(int(shift), axis=0)
                # 原始数据存在(base_pivot有值) 但 派生值=NaN → 也应视为"已披露"
                raw_data_exists = base_pivot.notna() & ~np.isinf(base_pivot)
                derived_na_but_raw_exists = pivot.isna() & raw_data_exists
                if derived_na_but_raw_exists.any().any():
                    effective_is_na_mask = is_na_mask | derived_na_but_raw_exists

        try:
            res = self.manager._fast_align(pivot, self.col, effective_is_na_mask)
        except Exception:
            res = pd.Series(index=self.manager.trade_date_df.index, data=np.nan, name=self.col)
        self.manager._aligned_cache[key] = res
        return res

    def _align_series(self, pivot_df: pd.DataFrame, is_na_mask: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        对齐到交易时间轴

        将pivot DataFrame对齐到交易日Series

        Args:
            pivot_df: 需要对齐的pivot DataFrame
            is_na_mask: 可选的NaN掩码,用于标记"已披露但为空/计算结果NaN"的位置,
                       避免对齐时回退到旧季度
        """
        val_col = self.col
        # 统一为DataFrame
        if isinstance(pivot_df, pd.Series):
            pivot_df = pivot_df.to_frame(name=val_col)
        # 空保护
        if pivot_df is None or pivot_df.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)

        # 兼容PeriodIndex与非时间列名
        if hasattr(pivot_df.columns, 'to_timestamp'):
            pivot_df = pivot_df.copy()
            pivot_df.columns = pivot_df.columns.to_timestamp()
        elif not pd.api.types.is_datetime64_any_dtype(pivot_df.columns):
            try:
                pivot_df = pivot_df.copy()
                pivot_df.columns = pd.to_datetime(pivot_df.columns)
            except Exception:
                pass

        # 使用快速对齐方法(
        return self.manager._fast_align(pivot_df, val_col, is_na_mask)

    def raw(self, raw: bool = False):
        """原始累计数据

        Args:
            raw: False返回对齐到交易日的Series,True返回pivot DataFrame

        Returns:
            pd.Series(对齐到交易日)或 DataFrame(pivot)

        替代标签:
            raw() -> tag("累计最新")
        """
        if raw:
            # 仅在raw=True时返回pivot DataFrame
            return self._get_pivot('raw').copy()

        # 返回对齐到交易日的Series
        aligned_series = self._get_aligned('raw', 0)
        result = FactorFinanceSeriesV5Fast(
            aligned_series,
            manager=self.manager,
            col=self.col,
            index=aligned_series.index,
            name=self.col
        )
        result._pivot_kind = 'raw'
        return result

    def quarter(self, raw: bool = False):
        """单季数据视图

        Args:
            raw: False返回对齐到交易日的Series,True返回pivot DataFrame

        Returns:
            pd.Series(对齐到交易日)或 DataFrame(pivot)

        替代标签:
            quarter() -> tag("单季最新")
        """
        self._assert_supports_flow_view('quarter')
        if raw:
            # 仅在raw=True时返回pivot DataFrame
            return self._get_pivot('quarter').copy()

        # 返回对齐到交易日的Series
        aligned_series = self._get_aligned('quarter', 0)
        result = FactorFinanceSeriesV5Fast(
            aligned_series,
            manager=self.manager,
            col=self.col,
            index=aligned_series.index,
            name=self.col
        )
        result._pivot_kind = 'quarter'
        return result

    def ttm(self, raw: bool = False):
        """TTM 数据视图

        Args:
            raw: False返回对齐到交易日的Series,True返回pivot DataFrame

        Returns:
            pd.Series(对齐到交易日)或 DataFrame(pivot)

        替代标签:
            ttm() -> tag("ttm最新")
        """
        self._assert_supports_flow_view('ttm')
        if raw:
            # 仅在raw=True时返回pivot DataFrame
            return self._get_pivot('ttm').copy()

        # 返回对齐到交易日的Series
        aligned_series = self._get_aligned('ttm', 0)
        result = FactorFinanceSeriesV5Fast(
            aligned_series,
            manager=self.manager,
            col=self.col,
            index=aligned_series.index,
            name=self.col
        )
        result._pivot_kind = 'ttm'
        return result

    def annual(self, raw: bool = False):
        """年报数据视图(仅Q4,一年一个值)

        Args:
            raw: False返回对齐到交易日的Series,True返回pivot DataFrame

        Returns:
            pd.Series(对齐到交易日)或 DataFrame(pivot)

        替代标签:
            annual() -> tag("年报最新")
        """
        if raw:
            # 仅在raw=True时返回pivot DataFrame
            return self._get_pivot('annual').copy()

        # 返回对齐到交易日的Series
        aligned_series = self._get_aligned('annual', 0)
        result = FactorFinanceSeriesV5Fast(
            aligned_series,
            manager=self.manager,
            col=self.col,
            index=aligned_series.index,
            name=self.col
        )
        result._pivot_kind = 'annual'
        return result

    def _require_kind(self) -> str:
        """若未选择视图,流量字段默认quarter,存量字段默认raw."""
        if self._pivot_kind is None and self._is_stock_indicator_col(self.col):
            return 'raw'
        return self._pivot_kind or 'quarter'

    def _get_cal_pivot(self) -> pd.DataFrame:
        kind = self._require_kind()
        return self._get_pivot(kind)

    def _get_preserve_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取需要保留的位置掩码,防止已发布但为空的季度被丢弃.

        问题场景:
        - 原本有值但计算后是NaN的情况已处理
        - 但"原始数据已发布但值为空"(如2014Q1财报发布但某字段为空)时,
          该季度的派生指标会被stack(dropna)丢弃,对齐时错误地沿用前一季度的值

        本方法返回的掩码标记了需要保留的位置:
        - data.notna(): 原本有值
        - is_na_mask: 原始数据已发布但值为空(应返回NaN,不应回退到旧季度)

        Returns:
            pd.DataFrame: 布尔掩码,True表示该位置需要保留(不应被丢弃)
        """
        # 获取原始数据的缺失掩码
        is_na_mask = self.manager.precomputed.is_na_pivots.get(self.col)

        if is_na_mask is not None:
            # 对齐到当前data的形状
            try:
                is_na_aligned = is_na_mask.reindex_like(data).fillna(False)
            except Exception as e:
                logger.warning(
                    f"掩码对齐失败(股票:{self.manager.stock_code},列:{self.col})\n"
                    f"  原因:{type(e).__name__}: {e}\n"
                    f"  掩码索引类型:{type(is_na_mask.index).__name__}\n"
                    f"  数据索引类型:{type(data.index).__name__}\n"
                    f"  掩码列类型:{type(is_na_mask.columns).__name__ if hasattr(is_na_mask, 'columns') else 'N/A'}\n"
                    f"  数据列类型:{type(data.columns).__name__ if hasattr(data, 'columns') else 'N/A'}\n"
                    f"  影响:已发布但为空的数据点可能被错误填充,建议检查数据"
                )
                is_na_aligned = pd.DataFrame(False, index=data.index, columns=data.columns)
        else:
            is_na_aligned = pd.DataFrame(False, index=data.index, columns=data.columns)

        # 合并:原本有值 或 原始数据标记为空(即发布过)
        return data.notna() | is_na_aligned

    def qoq(self, exclude_negative: bool = False) -> pd.Series:
        """环比增长(在 pivot 级计算后再对齐,与彩虹版完全一致)

        Args:
            exclude_negative: 是否剔除基期为负的情况(V4逻辑:基期<=0时返回NaN)

        替代标签:
            qoq() -> tag("单季/ttm环比增幅")
            qoq(exclude_negative=True) -> tag("单季/ttm环比增幅剔负")
        """
        # 使用当前视图的pivot(raw/quarter/ttm),与彩虹版保持一致
        data = self._get_cal_pivot()
        if data.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)

        
        
        preserve_mask = self._get_preserve_mask(data)

        previous = data.shift(1)
        qoq_ = data / previous - 1

        if exclude_negative:
            # V4剔负逻辑:基期>0时计算增长率,否则返回NaN
            qoq_ = qoq_.where(previous > 0, np.nan)
        else:
            # 非剔负:基期>0时正常计算,基期<=0时取反
            qoq_ = qoq_.mask(previous < 0, -qoq_)

        
        computed_nan = preserve_mask & qoq_.isna()

        result = self._align_series(qoq_, is_na_mask=computed_nan)

        return result

    def qoq_accel(self, lag: int = 1) -> pd.Series:
        """环比增速的加速度(特殊逻辑,复制V4)

        Args:
            lag: 时间间隔(季度数)
                 1 = 隔季增速
                 4 = 隔年增速

        替代标签:
            quarter().qoq_accel(1) -> tag("单季环比增幅隔季增速")
            quarter().qoq_accel(4) -> tag("单季环比增幅隔年增速")
            ttm().qoq_accel(1) -> tag("ttm环比增幅隔季增速")
            ttm().qoq_accel(4) -> tag("ttm环比增幅隔年增速")

        V4逻辑(特殊实现):
            lag=1: cur_g=Q0/Q2-1, old_g=Q1/Q5-1
            lag=4: cur_g=Q0/Q4-1, old_g=Q4/Q8-1 (实际上是同比增幅隔年增速)
        """
        data = self._get_cal_pivot()
        if data.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)
        preserve_mask = self._get_preserve_mask(data)

        def growth(cur: pd.Series, prev: pd.Series) -> pd.Series:
            """计算增长率(兼容负值)"""
            result = (cur / prev - 1).where(prev > 0, 1 - cur / prev)
            return result

        if lag == 1:
            # 隔季增速:cur=Q0/Q2, old=Q1/Q5
            cur = data
            prev_for_cur = data.shift(2)
            cur_g_pivot = growth(cur, prev_for_cur)

            old = data.shift(1)
            prev_for_old = data.shift(5)
            old_g_pivot = growth(old, prev_for_old)
            old_g_pivot = old_g_pivot.where(preserve_mask)
            cur_need_inf = preserve_mask & cur_g_pivot.isna()
            old_need_inf = preserve_mask & old_g_pivot.isna()
            cur_g_pivot[cur_need_inf] = -np.inf
            old_g_pivot[old_need_inf] = -np.inf

            # 对齐后相减
            cur_g = self._align_series(cur_g_pivot).replace(-np.inf, np.nan)
            old_g = self._align_series(old_g_pivot).replace(-np.inf, np.nan)
            return cur_g - old_g

        elif lag == 4:
            # 隔年增速:实际上是同比增幅的加速度
            col_0 = data
            col_1 = data.shift(4)
            col_2 = data.shift(8)
            new_g_pivot = growth(col_0, col_1)
            old_g_pivot = growth(col_1, col_2)
            old_g_pivot = old_g_pivot.where(preserve_mask)
            new_need_inf = preserve_mask & new_g_pivot.isna()
            old_need_inf = preserve_mask & old_g_pivot.isna()
            new_g_pivot[new_need_inf] = -np.inf
            old_g_pivot[old_need_inf] = -np.inf

            # 对齐后相减
            new_g = self._align_series(new_g_pivot).replace(-np.inf, np.nan)
            old_g = self._align_series(old_g_pivot).replace(-np.inf, np.nan)
            return new_g - old_g

        else:
            raise ValueError(f"qoq_accel不支持lag={lag},只支持1或4")

    def yoy(self, exclude_negative: bool = False) -> pd.Series:
        """同比增长(在 pivot 级计算后再对齐,与彩虹版完全一致)

        Args:
            exclude_negative: 是否剔除基期为负的情况(V4逻辑:基期<=0时返回NaN)

        替代标签:
            yoy() -> tag("单季/ttm/累计/年报同比增幅")
            yoy(exclude_negative=True) -> tag("单季/ttm/累计/年报同比增幅剔负")
        """
        # 使用当前视图的pivot(raw/quarter/ttm),与彩虹版保持一致
        data = self._get_cal_pivot()
        if data.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)

        
        
        preserve_mask = self._get_preserve_mask(data)

        # Annual视图特殊处理:每年只有1行,shift(1)表示上一年
        periods = 1 if self._pivot_kind == 'annual' else 4
        previous = data.shift(periods)
        yoy_ = data / previous - 1

        if exclude_negative:
            # V4剔负逻辑:基期>0时计算增长率,否则返回NaN
            yoy_ = yoy_.where(previous > 0, np.nan)
        else:
            # 非剔负:基期>0时正常计算,基期<=0时取反
            yoy_ = yoy_.mask(previous < 0, -yoy_)

        
        
        
        computed_nan = preserve_mask & yoy_.isna()

        result = self._align_series(yoy_, is_na_mask=computed_nan)

        return result

    def yoy_accel(self, lag: int = 1) -> pd.Series:
        """同比增速的加速度(增速的增速)

        Args:
            lag: 时间间隔(季度数)
                 1 = 隔季增速(本季增速 - 上季增速)
                 4 = 隔年增速(本季增速 - 去年同期增速)

        替代标签:
            quarter().yoy_accel(1) -> tag("单季同比增幅隔季增速")
            quarter().yoy_accel(4) -> tag("单季同比增幅隔年增速")
            ttm().yoy_accel(1) -> tag("ttm同比增幅隔季增速")
            ttm().yoy_accel(4) -> tag("ttm同比增幅隔年增速")

        实现逻辑:
            在pivot维度计算同比增速,然后shift(lag)个季度,
            分别对齐到交易日后相减,得到加速度
        """
        data = self._get_cal_pivot()
        if data.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)
        preserve_mask = self._get_preserve_mask(data)

        # 在pivot维度计算同比增速
        yoy_pivot = data.groupby(data.index.quarter).pct_change(fill_method=None)
        yoy_pivot = yoy_pivot.mask(data.shift(4) < 0, -yoy_pivot)
        need_inf = preserve_mask & yoy_pivot.isna()
        yoy_pivot[need_inf] = -np.inf

        # 当前的同比增速(对齐到交易日)
        yoy_cur = self._align_series(yoy_pivot).replace(-np.inf, np.nan)

        # lag个季度前的同比增速(先在pivot维度shift,再对齐)
        yoy_shifted = yoy_pivot.shift(lag)
        yoy_shifted = yoy_shifted.where(preserve_mask)
        need_inf_old = preserve_mask & yoy_shifted.isna()
        yoy_shifted[need_inf_old] = -np.inf
        yoy_old = self._align_series(yoy_shifted).replace(-np.inf, np.nan)

        # 加速度 = 当前增速 - lag个季度前的增速
        return yoy_cur - yoy_old

    def q_diff(self) -> pd.Series:
        """季度差额(在 pivot 级计算后再对齐,与彩虹版完全一致)"""
        # 使用当前视图的pivot(raw/quarter/ttm),与彩虹版保持一致
        data = self._get_cal_pivot()
        if data.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)

        
        preserve_mask = self._get_preserve_mask(data)

        diff_ = data.diff()

        
        computed_nan = preserve_mask & diff_.isna()
        diff_[computed_nan] = -np.inf

        result = self._align_series(diff_)
        result = result.replace(-np.inf, np.nan)
        return result

    def y_diff(self) -> pd.Series:
        """年度差额(同季同比差额:在 pivot 级计算后再对齐,与彩虹版完全一致)"""
        # 使用当前视图的pivot(raw/quarter/ttm),与彩虹版保持一致
        data = self._get_cal_pivot()
        if data.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=np.nan)

        
        preserve_mask = self._get_preserve_mask(data)

        diff_ = data.groupby(data.index.quarter).diff()

        
        computed_nan = preserve_mask & diff_.isna()
        diff_[computed_nan] = -np.inf

        result = self._align_series(diff_)
        result = result.replace(-np.inf, np.nan)
        return result

    def last_q(self, n: int = 1) -> pd.Series:
        """N 季前数据(与彩虹版完全一致的实现)
        """
        # 使用当前视图的pivot(raw/quarter/ttm),与彩虹版保持一致
        data = self._get_cal_pivot()
        q = data.shift(n)
        # 必须要把原来pivot数据中的nan值给复原,因为pivot中的nan值是有重要作用的,会影响最后日期的映射
        q[data.isna()] = np.nan
        is_na_mask = self.manager.precomputed.is_na_pivots.get(self.col)
        if is_na_mask is not None:
            try:
                is_na_mask = is_na_mask.reindex_like(data).fillna(False)
            except Exception:
                is_na_mask = None

        published_current = data.notna()
        if is_na_mask is not None:
            published_current = published_current | is_na_mask

        need_inf_mask = published_current & q.isna()
        return self._align_series(q, is_na_mask=need_inf_mask)

    def last_y(self, n: int = 1) -> pd.Series:
        """N 年前(同季)数据(与彩虹版完全一致的实现)
        """
        # 使用当前视图的pivot(raw/quarter/ttm),与彩虹版保持一致
        data = self._get_cal_pivot()
        y = data.groupby(data.index.quarter).shift(n)
        # 必须要把原来pivot数据中的nan值给复原,因为pivot中的nan值是有重要作用的,会影响最后日期的映射
        y[data.isna()] = np.nan
        is_na_mask = self.manager.precomputed.is_na_pivots.get(self.col)
        if is_na_mask is not None:
            try:
                is_na_mask = is_na_mask.reindex_like(data).fillna(False)
            except Exception:
                is_na_mask = None

        published_current = data.notna()
        if is_na_mask is not None:
            published_current = published_current | is_na_mask

        need_inf_mask = published_current & y.isna()
        return self._align_series(y, is_na_mask=need_inf_mask)

    def last_q4(self, n: int = 1) -> pd.Series:
        """N 年前年报数据(与彩虹版完全一致的实现)

        注意:在annual视图上调用时,由于annual pivot每年只有1行(Q4),
        需要特殊处理shift逻辑
        """
        # 使用当前视图的pivot(raw/quarter/ttm/annual),与彩虹版保持一致
        data = self._get_cal_pivot()

        # 检测当前视图类型
        if self._pivot_kind == 'annual':
            # Annual视图:每年只有1行(Q4),直接shift n行
            q4 = data.shift(n)
        else:
            # 其他视图(quarter/ttm/raw):每年4行,需要groupby.transform + shift
            q4 = data.groupby(data.index.year).transform("last").shift(n * 4)

        # 必须要把原来pivot数据中的nan值给复原,因为pivot中的nan值是有重要作用的,会影响最后日期的映射
        q4[data.isna()] = np.nan
        is_na_mask = self.manager.precomputed.is_na_pivots.get(self.col)
        if is_na_mask is not None:
            try:
                is_na_mask = is_na_mask.reindex_like(data).fillna(False)
            except Exception:
                is_na_mask = None

        published_current = data.notna()
        if is_na_mask is not None:
            published_current = published_current | is_na_mask

        need_inf_mask = published_current & q4.isna()
        return self._align_series(q4, is_na_mask=need_inf_mask)

    def get_q(self, n: int) -> pd.Series:
        """
        获取最近n个季度的数据

        实现策略:
        1. 使用numpy滑动窗口(sliding_window_view)在C层生成窗口(零拷贝)
        2. 中后段数据从预构造的窗口直接取值
        3. 头部数据用切片补齐到长度n
        4. 显式类型转换确保np.isnan安全

        Parameters
        ----------
        n : int
            窗口大小(季度数)

        Returns
        -------
        pd.Series
            每个交易日对应的最近n个季度的数据列表
        """
        from numpy.lib.stride_tricks import sliding_window_view

        pivot = self._get_cal_pivot()
        arr = pivot.to_numpy()
        rows, cols = arr.shape
        result = np.full((rows, cols), None, dtype=object)
        MAX_SAFE_INT = 2 ** 53  # float64精度安全范围

        # 按列处理
        for j in range(cols):
            col = arr[:, j]

            # 智能路径选择:兼顾性能与精度
            use_fast_path = False
            col_float = None

            if col.dtype in (np.float64, np.float32, np.float16):
                # 浮点类型:直接使用快速路径
                use_fast_path = True
                col_float = col
            elif col.dtype in (np.int64, np.int32, np.int16, np.int8,
                               np.uint64, np.uint32, np.uint16, np.uint8):
                # 整数类型:检查是否有超过安全范围的大整数
                valid_vals = col[~pd.isna(col)]
                if len(valid_vals) > 0:
                    max_abs = np.max(np.abs(valid_vals.astype(np.int64)))
                    if max_abs <= MAX_SAFE_INT:
                        # 安全范围内,使用快速路径
                        use_fast_path = True
                        col_float = col.astype(np.float64, copy=False)
                    # else: 大整数,走安全路径
                else:
                    # 全NaN,快速路径即可
                    use_fast_path = True
                    col_float = col.astype(np.float64, copy=False)
            # else: object或其他类型,走安全路径

            if use_fast_path:
                # ========== 快速路径(原逻辑,性能优先)==========
                valid_indices = np.where(~np.isnan(col_float))[0]

                if rows >= n:
                    windows = sliding_window_view(col_float, n)
                else:
                    windows = None

                for i in valid_indices:
                    if windows is None or i < n - 1:
                        values = col_float[:i+1].tolist()
                        if len(values) < n:
                            values = [np.nan] * (n - len(values)) + values
                    else:
                        window_idx = i - n + 1
                        values = windows[window_idx].tolist()
                    result[i, j] = values
            else:
                # ========== 安全路径(精度优先,处理大整数/混合类型)==========
                valid_mask = ~pd.isna(col)
                valid_indices = np.where(valid_mask)[0]

                for i in valid_indices:
                    # 切片方式(不使用sliding_window_view)
                    if i < n - 1:
                        values = col[:i+1].tolist()
                        values = [np.nan] * (n - len(values)) + values
                    else:
                        values = col[i-n+1:i+1].tolist()
                    result[i, j] = values

        # 转回 DataFrame
        result_df = pd.DataFrame(result, index=pivot.index, columns=pivot.columns)

        # 对齐到交易日
        col_name = f"{self.col}_近{n}"
        result_series = self.manager._fast_align(result_df, col_name)

        # 填充 NaN
        result_series = result_series.apply(
            lambda x: [np.nan] * n if not isinstance(x, list) and pd.isna(x) else x
        )

        return result_series

    def cagr(self, n: int, exclude_negative: bool = False) -> pd.Series:
        """复合年化增长率

        Args:
            n: 年数(与stability统一,都用年数)
            exclude_negative: 是否剔除负值(负值设为NaN)

        Examples:
            # 年报视图: 3年复合增长率(年报对年报)
            annual().cagr(3)   # shift(3)年, 如2020Q4 vs 2023Q4

            # 季度视图: 3年同期复合增长率(同季度对同季度)
            quarter().cagr(3)  # shift(12)季度, 如2020Q3 vs 2023Q3

            # TTM视图: 3年同期复合增长率(同期TTM对同期TTM)
            ttm().cagr(3)      # shift(12)季度

        替代标签:
            annual().cagr(3) -> tag("年报3年复合增长率")
            annual().cagr(3, exclude_negative=True) -> tag("年报3年复合增长率剔负")
            quarter().cagr(3) -> tag("单季3年同期复合增长率")
            ttm().cagr(3) -> tag("TTM3年同期复合增长率")
        """
        if n < 1:
            raise ValueError(f"CAGR的n参数必须>=1(至少1年),当前n={n}")

        data = self._get_cal_pivot()

        # 统一用年数,参考stability的设计
        if self._pivot_kind == 'annual':
            # annual视图: 每年1个数据点,直接shift(n)年
            shift_periods = n
        else:
            # 其他视图: 每年4个数据点,shift(n*4)季度
            shift_periods = n * 4

        current = data
        previous = data.shift(shift_periods)

        # CAGR公式: (current/previous)^(1/n) - 1
        def growth(cur: pd.Series, prev: pd.Series) -> pd.Series:
            if exclude_negative:
                return ((cur / prev) ** (1 / n) - 1).where(prev > 0, np.nan)
            else:
                return ((cur / prev) ** (1 / n) - 1).where(
                    prev > 0,
                    1 - (cur / prev) ** (1 / n)
                )

        cagr_ = growth(current, previous)

        # 剔除负值(如果需要)
        if exclude_negative:
            cagr_ = cagr_.where(cagr_ >= 0, np.nan)

        # 传递 is_na_mask：确保"已披露但计算为 NaN"时不回退旧季度值
        preserve_mask = self._get_preserve_mask(data)
        computed_nan = preserve_mask & cagr_.isna()
        return self._align_series(cagr_, is_na_mask=computed_nan)

    def stability(self, periods: Union[int, List[int]]) -> pd.Series:
        """计算多个时期的稳定性指标(均值/标准差)

        Args:
            periods: 支持两种格式
                - int n: n年同期稳定性(包含当前,共n+1个数据点)
                    annual视图: stability(3) = [0,1,2,3] = 3年同期(4个年报)
                    quarter视图: stability(3) = [0,4,8,12] = 3年同期(4个季度)
                - List[int]: 自定义时期列表
                    quarter().stability([0,4,8,12]) = 本期和过去3个同期(共4个数据点)

        Returns:
            稳定性指标Series(均值/标准差,值越大越稳定)
            注意:这是变异系数的倒数

        Examples:
            # 年报3年同期稳定性
            annual().stability(3)  # = [0,1,2,3] = 4个年报

            # 单季3年同期稳定性
            quarter().stability(3)  # = [0,4,8,12] = 4个季度

            # TTM 3年同期稳定性
            ttm().stability(3)  # = [0,4,8,12] = 4个TTM

        替代标签:
            quarter().stability(3) -> tag("单季三年同期稳定性")
        """
        # 支持整数参数:转换为n年同期列表
        if isinstance(periods, int):
            if self._pivot_kind == 'annual':
                # annual视图:每年1个数据点,n年同期 = [0,1,2,...,n]
                periods = list(range(periods + 1))
            else:
                # 其他视图:每年4个数据点,n年同期 = [0,4,8,...,4*n]
                periods = list(range(0, (periods + 1) * 4, 4))

        # 获取多个时期的数据
        kind = self._require_kind()
        data_list = []
        for shift in periods:
            shifted_data = self._get_aligned(kind, shift)
            data_list.append(shifted_data)

        # 将多个Series组合成DataFrame(每列是一个时期)
        df = pd.concat(data_list, axis=1)

        # 计算每行的标准差和均值
        std = df.std(axis=1)
        mean = df.mean(axis=1)

        # 计算稳定性指标(均值/标准差,与V4一致)
        stability = mean / std.replace(0, np.nan)

        # 替换inf为NaN(当标准差为0或除零时)
        stability = stability.replace([np.inf, -np.inf], np.nan)

        return stability

    def _cal_publish_date(self, col: str, quarter: int) -> pd.Series:
        """
        根据参数quarter,获取当前这一天对应的quarter季报的最早/最晚发布时间

        完全复制彩虹版实现逻辑,保证100%兼容

        Parameters
        ----------
        col : str
            列名,"latest_publish_date" 或 "first_publish_date"
        quarter : int
            季度,取值范围 [1, 2, 3, 4]

        Returns
        -------
        pd.Series
            每个交易日对应的该季度财报的首次/末次发布日期

        Notes
        -----
        比如今天是0531,quarter填1,那么就是获取今年的一季报,
        如果今年一季报还没发,那就是获取去年的一季报
        """
        # 验证N的范围
        if quarter not in [1, 2, 3, 4]:
            raise ValueError("参数quarter必须在[1,4]范围内")

        need_cols = ["publish_date", "report_date", col]
        df = self.manager.precomputed.raw_df[need_cols[:2]].copy()

        # 确保日期类型
        df = _ensure_datetime(df, ['publish_date', 'report_date'])

        # ====================================================================================================
        # 1. 根据quarter生成新的季报列,report_date2
        # ====================================================================================================
        # 季度末日期映射
        quarter_end_dates = {
            1: "-03-31",  # 第一季度:3月31日
            2: "-06-30",  # 第二季度:6月30日
            3: "-09-30",  # 第三季度:9月30日
            4: "-12-31",  # 第四季度:12月31日
        }
        # 获取年份
        year = df["report_date"].dt.year
        # 生成当年的季度结束日期
        current_year_quarter = pd.to_datetime(year.astype(str) + quarter_end_dates[quarter])
        # 生成前一年的季度结束日期
        prev_year_quarter = pd.to_datetime((year - 1).astype(str) + quarter_end_dates[quarter])
        # 使用np.where一次性选择:如果当年季度日期 > report_date,则用前一年的,否则用当年的
        df["report_date2"] = np.where(current_year_quarter > df["report_date"], prev_year_quarter, current_year_quarter)

        # ====================================================================================================
        # 2. 根据report_date2对应的publish_date,生成col列
        # ====================================================================================================
        # 创建report_date列值到索引的映射字典
        value_to_index = {val: idx for idx, val in enumerate(df["report_date"])}
        # 直接映射得到col列,省略target_index
        df[col] = df["report_date2"].map(value_to_index).map(df["publish_date"])

        # ====================================================================================================
        # 3. 对未来日期的情况进行修改
        # ====================================================================================================
        # 如果发现target_index对应的publish_date仍然超过当前行的publish_date,则用前一个publish_date
        rp_pb_map = df.groupby("report_date")["publish_date"].apply(list).to_dict()
        # 如果col列 > pb_date,就用report_date2对应的那几行report_date对应的publish_date,作为一个list,然后在这个list中,寻找满足条件的publish_date
        mask = df[col] > df["publish_date"]
        for idx in df.loc[mask].index:
            pb_date = df.at[idx, "publish_date"]
            rp_date2 = df.at[idx, "report_date2"]
            if col.startswith("latest_"):
                df.at[idx, col] = next((x for x in rp_pb_map[rp_date2][::-1] if x <= pb_date), np.nan)
            else:
                df.at[idx, col] = next((x for x in rp_pb_map[rp_date2] if x >= pb_date), np.nan)
        # 处理边界情况
        df[col].fillna(pd.to_datetime("1970-01-01"), inplace=True)

        # ====================================================================================================
        # 4. 映射到完整交易日期的df中
        # ====================================================================================================
        df = (
            # 关键提速,只保留需要的列
            df[need_cols]
            .drop_duplicates(subset=need_cols[0], keep="last")
            .sort_values(need_cols[:2], ignore_index=True)
        )
        if df.empty:
            total_df = self.manager.trade_date_df.copy()
            total_df[col] = np.nan
        else:
            total_df = pd.merge_asof(
                self.manager.trade_date_df, df, left_on=["交易日期"], right_on=["publish_date"], direction="backward"
            )
        return total_df[col]

    def latest_publish_date(self, quarter: int) -> pd.Series:
        """
        返回特定季度财报的最后发布日期(对齐彩虹版)

        Parameters
        ----------
        quarter : int
            季度,取值范围 [1, 2, 3, 4]

        Returns
        -------
        pd.Series
            每个交易日对应的该季度财报的最后发布日期

        Examples
        --------
        >>> # 获取每个交易日对应的Q1财报的最后发布日期
        >>> q1_latest = fin_mgr["R_np_atoopc@xbx"].latest_publish_date(quarter=1)
        """
        return self._cal_publish_date(col="latest_publish_date", quarter=quarter)

    def first_publish_date(self, quarter: int) -> pd.Series:
        """
        返回特定季度财报的首次发布日期(对齐彩虹版)

        Parameters
        ----------
        quarter : int
            季度,取值范围 [1, 2, 3, 4]

        Returns
        -------
        pd.Series
            每个交易日对应的该季度财报的首次发布日期

        Examples
        --------
        >>> # 获取每个交易日对应的Q1财报的首次发布日期
        >>> q1_first = fin_mgr["R_np_atoopc@xbx"].first_publish_date(quarter=1)
        """
        return self._cal_publish_date(col="first_publish_date", quarter=quarter)

    def initial_publish_date(self) -> pd.Series:
        """
        最初发布日期(按 report_date 分组的首次发布日期)

        对于每个交易日,返回该日之前最新财报的"首次发布日期".
        适用于需要使用首次发布数据(未修订版本)的场景.

        性能优化策略:
        - 零复制:使用布尔索引视图,避免不必要的 .copy()
        - 向量化:groupby.transform('first') 代替循环
        - 原地操作:inplace=True 减少内存分配
        - 最小列集:只提取必需的列

        Returns
        -------
        pd.Series
            索引为交易日索引,值为对应的最初发布日期(Timestamp类型)
            如果某个交易日之前没有发布过财报,则为 NaT

        Example
        -------
        >>> fin_data = create_factor_finance_manager_precomputed('sh600000', ['B_total_equity_atoopc@xbx'], trade_date_df)
        >>> fin_data['B_total_equity_atoopc@xbx'].initial_publish_date()
        0           NaT
        1           NaT
        2      2020-04-30
        3      2020-04-30
        ...

        Notes
        -----
        - 与 publish_date() 的区别:
          - initial_publish_date() 返回首次发布版本的日期
          - publish_date() 返回最新修订版本的日期
        - 与 first_publish_date(quarter) 的区别:
          - initial_publish_date() 无需指定季度,返回动态值
          - first_publish_date(quarter) 需要指定季度,返回特定季度的首次发布
        """
        raw_df = self.manager.precomputed.raw_df

        # 快速路径:该列无数据
        mask = raw_df[self.col].notna()
        if not mask.any():
            return pd.Series(
                index=self.manager.trade_date_df.index,
                data=pd.NaT,
                name=f'{self.col}_initial_publish_date'
            )

        # 提取必需列(最小内存占用)- 只复制需要的两列
        valid_df = raw_df.loc[mask, ['publish_date', 'report_date']].copy()

        # 确保日期类型(原地转换,避免额外内存分配)
        valid_df['publish_date'] = pd.to_datetime(valid_df['publish_date'])
        valid_df['report_date'] = pd.to_datetime(valid_df['report_date'])
        valid_df = valid_df.dropna(subset=['publish_date', 'report_date'])

        # 向量化分组:按 report_date 取首次 publish_date(高性能,零循环)
        # sort=False 避免不必要的排序开销
        valid_df['initial_pub'] = valid_df.groupby('report_date', sort=False)['publish_date'].transform('min')
        valid_df.drop_duplicates(subset=['report_date', 'publish_date'], keep='last', inplace=True)
        # 排序:先publish_date,再report_date(同日多报表时最新report_date排后面)
        valid_df.sort_values(['publish_date', 'report_date'], inplace=True)
        report_ord = valid_df['report_date'].to_numpy('datetime64[ns]').view('int64')
        frontier_mask = report_ord == np.maximum.accumulate(report_ord)
        valid_df = valid_df.loc[frontier_mask].copy()

        # 对齐到交易日(merge_asof 已是高度优化的算法)
        # 只取必需列,减少数据传输
        trade_idx = self.manager.trade_date_df.index
        trade_dates = self.manager.trade_date_df['交易日期'].sort_values()

        # 使用 searchsorted 进行高效对齐(NumPy级别性能,比 merge_asof 更快)
        pub_dates = valid_df['publish_date'].values
        initial_pubs = valid_df['initial_pub'].values

        # searchsorted: O(n log m) 复杂度,m 是 pub_dates 长度
        indices = np.searchsorted(pub_dates, trade_dates.values, side='right') - 1

        # 向量化构建结果(零循环)
        result_values = np.full(len(trade_dates), np.datetime64('NaT'), dtype='datetime64[ns]')
        valid_mask = indices >= 0
        result_values[valid_mask] = initial_pubs[indices[valid_mask]]

        # 恢复原索引顺序
        result = pd.Series(result_values, index=trade_dates.index, name=f'{self.col}_initial_publish_date')
        return result.reindex(trade_idx)

    def publish_date(self) -> pd.Series:
        """
        返回每个交易日对应的最新财报发布日期

        对于每个交易日,返回该日期之前最近的一次财报发布日期.
        只考虑该财务指标列有数据的发布日期.

        Returns:
        --------
        pd.Series
            索引为交易日索引,值为对应的最新财报发布日期(Timestamp类型)
            如果某个交易日之前没有发布过财报,则为 NaT

        Example:
        --------
        >>> fin_data = create_factor_finance_manager_precomputed('sh600000', ['B_total_equity_atoopc@xbx'], trade_date_df)
        >>> fin_data['B_total_equity_atoopc@xbx'].publish_date()
        0           NaT
        1           NaT
        2      2020-04-30
        3      2020-04-30
        4      2020-04-30
        5      2020-08-29
        ...
        Name: B_total_equity_atoopc@xbx_publish_date, dtype: datetime64[ns]

        Notes:
        ------
        - 与 latest_publish_date() 的区别:
          - publish_date() 返回最近的任意财报发布日期
          - latest_publish_date() 返回按 report_date 分组的最后发布日期
        - 与 initial_publish_date() 的区别:
          - publish_date() 返回最新修订版本的日期
          - initial_publish_date() 返回首次发布版本的日期
        """
        # 从 raw_fin_df 获取该列有数据的所有发布日期
        raw_df = self.manager.precomputed.raw_df

        # 过滤出该列有数据的行,并获取唯一的 publish_date
        mask = raw_df[self.col].notna()
        if '原始数据标签' in raw_df.columns:
            mask &= raw_df['原始数据标签'].eq(1)
        elif '财报类型' in raw_df.columns:
            mask &= raw_df['财报类型'].notna()
        elif '是否上市前财报' in raw_df.columns:
            mask &= raw_df['是否上市前财报'].notna()

        valid_df = raw_df.loc[mask, ['publish_date']].copy()
        if valid_df.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=pd.NaT, name=f'{self.col}_publish_date')

        valid_df = valid_df.drop_duplicates()
        valid_df['publish_date'] = pd.to_datetime(valid_df['publish_date'])
        valid_df = valid_df.dropna(subset=['publish_date'])
        if valid_df.empty:
            return pd.Series(index=self.manager.trade_date_df.index, data=pd.NaT, name=f'{self.col}_publish_date')
        valid_df = valid_df.sort_values('publish_date')

        # 创建辅助列:publish_date 本身就是要返回的值
        valid_df['pub_date_result'] = valid_df['publish_date']

        # 使用 merge_asof 对齐到交易日
        trade_df = self.manager.trade_date_df.sort_values('交易日期').reset_index()
        aligned = pd.merge_asof(
            trade_df,
            valid_df,
            left_on='交易日期',
            right_on='publish_date',
            direction='backward'
        )

        # 恢复原索引并返回结果
        result = aligned.set_index('index')['pub_date_result']
        result.name = f'{self.col}_publish_date'
        return result

    def quarter_df(self) -> QuarterDataFrame:
        """
        返回季度层DataFrame操作器,支持链式调用和运算符

        这是一个高级接口,提供极简的季度层面数据操作,将复杂的财务因子计算
        从30行代码减少到3行(代码减少90%).

        Returns:
            QuarterDataFrame: 季度数据操作器,支持:
                - 链式调用(diff, shift, pct_change, rolling等)
                - 运算符(+, -, *, /, **等)
                - 自动去重和对齐

        Example:
            >>> # SUE因子(盈余意外)- 仅需3行,代码减少90%
            >>> qdf = fin_mgr['R_np_atoopc@xbx'].quarter_df()
            >>> delta = qdf.diff()
            >>> df['SUE'] = ((qdf - qdf.shift(1) - delta.rolling(8).mean()) / delta.rolling(8).std()).to_series()

            >>> # 净利润稳定性 - 仅需1行
            >>> df['Stability'] = (qdf.rolling(8).std() / qdf.rolling(8).mean()).to_series()

            >>> # 支持所有视图
            >>> raw_qdf = fin_mgr['R_np@xbx'].raw().quarter_df()    # raw视图
            >>> ttm_qdf = fin_mgr['R_np@xbx'].ttm().quarter_df()    # ttm视图
            >>> annual_qdf = fin_mgr['R_np@xbx'].annual().quarter_df()  # annual视图

        Note:
            - 默认使用当前视图(raw/quarter/ttm/annual)的pivot
            - 通过_get_cal_pivot()自动选择正确的pivot
            - to_series()结果与原生方法(如quarter())100%一致
        """
        pivot = self._get_cal_pivot()  # 获取当前视图的pivot
        return QuarterDataFrame(self, pivot)

# =============================
# 全局缓存版本(兜底:脚本/Demo场景)
# =============================
def create_factor_finance_manager_precomputed(
    stock_code: str,
    fin_cols: List[str],
    trade_date_df: pd.DataFrame,
) -> FactorFinanceManagerV4Fast:
    """创建预计算的管理器(使用全局缓存,兜底场景)

    适用场景:
    - 独立脚本测试
    - Demo演示
    - 未接入回测主流程时

    缓存策略:
    - 使用 _PRECOMPUTED_CACHE 全局缓存
    -
    - 适用于:单批次回测,有限股票数,回测结束后进程退出
    - 不适用于:长时间运行的服务,无限循环的参数遍历(需改用LRU)
    """

    cache_key = f"{stock_code}_{','.join(sorted(fin_cols))}"

    # 检查缓存
    if cache_key not in _PRECOMPUTED_CACHE:
        raw_fin_df = _load_raw_fin_df(stock_code, fin_cols)

        # 清洗
        if trade_date_df is None or (hasattr(trade_date_df, 'empty') and trade_date_df.empty):
            ipo_day = pd.Timestamp('2007-01-01')
        else:
            ipo_day = trade_date_df['交易日期'].min()
        fin_base_date = _create_fin_base_date()
        clean_df = _clean_raw_fin_df_like_factorfun(
            raw_fin_df,
            fin_base_date,
            fin_cols,
            ipo_day,
            delay_param=None,
        )

        # 预计算
        precomputed = PrecomputedFinanceData(stock_code, fin_cols)
        precomputed.precompute(clean_df)

        _PRECOMPUTED_CACHE[cache_key] = precomputed

    return FactorFinanceManagerV4Fast(
        _PRECOMPUTED_CACHE[cache_key],
        trade_date_df,
        stock_code
    )

def create_factor_finance_manager_from_raw(
    stock_code: str,
    raw_fin_df: pd.DataFrame,
    fin_cols: List[str],
    trade_date_df: pd.DataFrame,
    pivot_dict: Optional[Dict] = None,
) -> FactorFinanceManagerV4Fast:
    """
    基于上游已经加载好的原始财务数据创建管理器(不使用全局缓存)

    典型场景:在 process_by_stock 中,get_finance_data 已经读取并合并了
    该股票的全部原始财报,这里直接复用 raw_fin_df,避免二次读盘.
    - 当 FMV5_FRAMEWORK_CLEANED_FASTPATH=1 且框架已完成清洗时,跳过V5重复清洗
    - 框架清洗标志:存在_merge列(generate_fin_pivot的indicator=True产物)
    - 契约检查失败时自动回退到完整清洗路径
    - 当 pivot_dict 存在时,直接复用框架已计算的pivot,跳过V5重复计算
    - pivot_dict 来自 generate_fin_pivot() 的返回值
    - 预估收益:25-40ms/股(避免skeleton构建+pivot计算)
    """
    if trade_date_df is None or (hasattr(trade_date_df, 'empty') and trade_date_df.empty):
        ipo_day = pd.Timestamp('2007-01-01')
    else:
        ipo_day = trade_date_df['交易日期'].min()
    if _USE_FRAMEWORK_CLEANED_FASTPATH and _can_use_framework_cleaned_fastpath(raw_fin_df):
        # 快速路径:框架已完成清洗,仅做最小处理
        clean_df = _minimal_clean_raw_fin_df_from_framework(raw_fin_df, fin_cols, ipo_day)
    else:
        # 完整路径:执行V5标准清洗
        fin_base_date = _create_fin_base_date()
        clean_df = _clean_raw_fin_df_like_factorfun(
            raw_fin_df,
            fin_base_date,
            fin_cols,
            ipo_day,
            delay_param=None,
        )

    precomputed = PrecomputedFinanceData(stock_code, fin_cols)
    precomputed.precompute(clean_df, pivot_dict=pivot_dict)

    return FactorFinanceManagerV4Fast(
        precomputed,
        trade_date_df,
        stock_code
    )

def get_v5_fin_mgr(
    stock_code: str,
    fin_cols: List[str] = None,
    trade_date_df: pd.DataFrame = None,
    fin_data: Optional[Dict[str, object]] = None,
) -> FactorFinanceManagerV4Fast:
    """
    【方案B】从框架传入的 fin_data 中获取财务管理器

    缓存策略(单股票缓存 + 用完即抛):
    1. 若框架已创建 v5_fin_mgr:直接复用(方案A,框架已修改)
    2. 若存在 fallback 缓存:复用(方案B首个因子已创建)
    3. 否则创建新管理器并缓存到 _v5_fin_mgr_fallback(方案B首次创建)

    同一只股票的多个因子会复用同一个管理器,处理完股票后fin_data被回收(用完即抛)

    字段选择策略(v5.4.1新增):
    - 若 fin_cols 未传入或为空:自动使用 pivot_dict.keys()(= conf.fin_cols)
    - 若 fin_cols 已传入:支持增量扩展(若缓存不足则合并列重建)
    """
    global _GLOBAL_CONF_FIN_COLS

    # 回测主流程:process_by_stock 已经构造好的 per-stock V5 管理器(方案A)
    if isinstance(fin_data, dict) and 'v5_fin_mgr' in fin_data:
        mgr = fin_data['v5_fin_mgr']
        existing_cols = set(mgr.raw_fin_cols) if hasattr(mgr, 'raw_fin_cols') else set()

        # 确定目标列集合
        if fin_cols is not None and len(fin_cols) > 0:
            target_cols = set(fin_cols)
        else:
            # 未指定,尝试使用全局配置列
            if _GLOBAL_CONF_FIN_COLS is not None and len(_GLOBAL_CONF_FIN_COLS) > 0:
                target_cols = set(_GLOBAL_CONF_FIN_COLS)
            else:
                # 保守策略:无法确定目标列时,不信任缓存,触发重建
                import warnings
                warnings.warn(
                    f"[get_v5_fin_mgr] 无法确定目标列(fin_cols未传且全局配置为空),"
                    f"缓存mgr可能缺列,将触发重建",
                    UserWarning
                )
                target_cols = None  # 标记需要重建

        # 检查是否缺列
        if target_cols is not None:
            missing_cols = target_cols - existing_cols
            if not missing_cols:
                return mgr
            # else: 缺列,继续往下走触发重建
        # else: target_cols为None,继续往下走触发重建

    # 【方案B改进】从旧版财务对象中获取数据,无则兜底到全局缓存
    if not isinstance(fin_data, dict) or '财务数据对象' not in fin_data:
        # 兜底:脚本 / Demo / 未接入回测主流程时,使用全局缓存版本
        # 注意:H版本fin_cols可选,需要处理None情况
        fallback_cols = fin_cols if fin_cols else []
        if not fallback_cols:
            # 尝试从全局配置读取
            if _GLOBAL_CONF_FIN_COLS is not None:
                fallback_cols = _GLOBAL_CONF_FIN_COLS
            else:
                try:
                    from core.model.backtest_config import load_config
                    conf = load_config()
                    if hasattr(conf, 'fin_cols') and conf.fin_cols:
                        fallback_cols = list(conf.fin_cols)
                        _GLOBAL_CONF_FIN_COLS = fallback_cols
                except Exception:
                    pass

        if not fallback_cols:
            raise ValueError(
                "【方案B】兜底场景必须指定 fin_cols！\n"
                "请在调用时传入 fin_cols 参数,或确保 conf.fin_cols 已配置"
            )

        return create_factor_finance_manager_precomputed(
            stock_code,
            fallback_cols,
            trade_date_df,
        )

    # ========================================================================
    # 【方案B】单股票缓存机制(fallback缓存)+ 智能字段选择
    # Author: Half open flowers
    # Date: 2025-11-22
    # Update: 2025-11-22 - 新增智能字段选择和增量扩展功能
    # ========================================================================

    old_fin_obj = fin_data['财务数据对象']
    raw_fin_df = old_fin_obj.raw_fin_df  # 这是 new_fin_df(清洗后的)
    pivot_dict = getattr(old_fin_obj, 'pivot_dict', None)

    # ========================================================================
    # 智能字段选择(全局缓存策略):
    # 1. 若 fin_cols 已传入 -> 使用传入的列
    # 2. 若 fin_cols 未传入 -> 依次尝试:
    #    a) 从全局缓存读取(所有股票公用,整个回测只加载一次)
    #    b) 从配置文件加载 conf.fin_cols(首次加载,然后全局缓存)
    #    c) 从 raw_fin_df.columns 中过滤非财务列(兜底)
    # ========================================================================
    # 注:global _GLOBAL_CONF_FIN_COLS 已在行2037声明,此处不再重复声明

    if fin_cols is not None and len(fin_cols) > 0:
        # 用户明确指定了字段
        target_cols = list(fin_cols)
    else:
        # 未指定字段,智能选择
        target_cols = None

        # 尝试1: 从全局缓存读取(整个回测只加载一次配置)
        if _GLOBAL_CONF_FIN_COLS is not None:
            target_cols = _GLOBAL_CONF_FIN_COLS
        else:
            # 尝试2: 从配置文件加载(首次加载,然后全局缓存)
            try:
                from core.model.backtest_config import load_config
                conf = load_config()
                if hasattr(conf, 'fin_cols') and conf.fin_cols:
                    target_cols = list(conf.fin_cols)
                    # 全局缓存,所有股票共享
                    _GLOBAL_CONF_FIN_COLS = target_cols
            except Exception:
                pass

        # 尝试3: 兜底方案 - 使用所有可用字段
        if target_cols is None or len(target_cols) == 0:
            target_cols = [col for col in raw_fin_df.columns
                          if col not in ['publish_date', 'report_date', '财报类型', '是否上市前财报']]

    # ========================================================================
    # 增量扩展逻辑:
    # - 若已有缓存且包含所有需要的列 -> 直接复用
    # - 若已有缓存但列不足 -> 合并列并重建
    # - 若无缓存 -> 首次创建
    # ========================================================================
    if '_v5_fin_mgr_fallback' in fin_data:
        existing_mgr = fin_data['_v5_fin_mgr_fallback']
        existing_cols = set(existing_mgr.raw_fin_cols)
        needed_cols = set(target_cols)

        if needed_cols.issubset(existing_cols):
            # 已有缓存包含所有需要的列,直接复用
            return existing_mgr

        # 需要扩展:合并现有列和新列
        all_cols = sorted(existing_cols | needed_cols)
        mgr = create_factor_finance_manager_from_raw(
            stock_code=stock_code,
            raw_fin_df=raw_fin_df,
            fin_cols=all_cols,
            trade_date_df=trade_date_df,
            pivot_dict=pivot_dict,
        )
        fin_data['_v5_fin_mgr_fallback'] = mgr
        return mgr

    # 首次创建管理器
    mgr = create_factor_finance_manager_from_raw(
        stock_code=stock_code,
        raw_fin_df=raw_fin_df,
        fin_cols=target_cols,
        trade_date_df=trade_date_df,
        pivot_dict=pivot_dict,
    )

    # 存入fallback缓存(供后续因子复用)
    fin_data['_v5_fin_mgr_fallback'] = mgr

    return mgr

# =============================
# QuarterDataFrame - 季度层DataFrame操作器
# =============================

class QuarterDataFrame:
    """季度层DataFrame操作器 - 支持运算符和链式调用

    设计模式:
    - Builder模式:链式调用构建计算流程
    - Immutable模式:每次操作返回新实例,不修改原数据
    - Facade模式:隐藏复杂的pivot转换和去重逻辑

    PIT语义契约:
    - **严格PIT原则**:diff/pct_change/shift/rolling 保证"修订版只影响发布日之后"
    - **场景**:同一季度有多次披露(业绩快报->正式报告->修订)
    - **保证**:每个发布日只能看到该日期之前披露的信息,修订版不会回填早期
    - **实现**:在pivot矩阵上按列计算(每列=一个发布日的独立视图)
    - **对齐**:to_series()时使用searchsorted找≤交易日的最近发布日,前向填充

    Example:
        >>> # SUE因子(盈余意外)- 仅需3行,代码减少77%
        >>> qdf = fin_mgr['R_np_atoopc@xbx'].quarter_df()
        >>> delta = qdf.diff()
        >>> df['SUE'] = ((qdf - qdf.shift(1) - delta.rolling(8).mean()) / delta.rolling(8).std()).to_series()
    """

    def __init__(self, finance_series: 'FactorFinanceSeriesV4Fast', pivot_df: pd.DataFrame):
        """
        Args:
            finance_series: 财务序列对象(用于对齐到交易日)
            pivot_df: 季度pivot表(index=季度PeriodIndex, columns=发布日期DatetimeIndex)
        """
        self._finance_series = finance_series
        self._pivot = pivot_df
        self._df = None  # 延迟计算的长格式DataFrame
        self._domain = 'pivot'  # 计算域:'pivot'(以_pivot为准)或'long'(以_df为准)
        self._align_is_na_mask = None

    # ========== 内部方法 ==========

    def _to_long_format(self) -> pd.DataFrame:
        """将pivot转成长格式DataFrame(延迟计算+缓存)

        Returns:
            pd.DataFrame,包含以下列:
                - report_quarter: pd.Period,季度索引(如2020Q1)
                - publish_date: pd.Timestamp,发布日期
                - value: float,财务数据值
            排序:按publish_date升序

        Note:
            使用缓存机制,只在首次调用时转换,后续直接返回缓存结果
        """
        if self._df is not None:
            return self._df
        if _USE_NUMPY_QDF_FASTPATH:
            try:
                self._df = self._to_long_format_numpy_fast()
                return self._df
            except Exception:
                # 任何异常都回退到pandas路径,保证兼容性与正确性
                pass

        # 使用stack转长格式(pandas高效方法)
        stacked = self._pivot.stack(dropna=False).reset_index()
        stacked.columns = ['report_quarter', 'publish_date', 'value']

        # 在操作前先过滤并去重(与传统方法一致)
        # 这样diff/rolling等操作看到的数据与传统方法相同
        self._df = self._apply_dedup(stacked)
        return self._df

    def _to_long_format_numpy_fast(self) -> pd.DataFrame:
        """NumPy快速路径:从pivot直接生成去重后的long格式(每个publish_date一行).
        Benchmark: 10.2ms -> 2.5ms (4.1x加速)

        语义要求(必须与 `_apply_dedup()` 完全一致):
            1) 严格PIT:pivot每一列(publish_date)是独立视图,互不影响
            2) 已披露优先:value非NaN 或 is_na_mask=True(已披露但字段为空)视为已披露
            3) 最新季度优先:同一publish_date选择 report_quarter 最大的那一行
            4) 若该publish_date列无任何"已披露",则退化为选择最新季度(value通常为NaN)

        Returns:
            pd.DataFrame: 列为 ['report_quarter', 'publish_date', 'value'],按publish_date升序.

        Raises:
            Exception: 任意异常将由上层捕获并回退到pandas实现(保证兼容性).
        """
        pivot = self._pivot
        if pivot is None or pivot.empty:
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])

        # 保护:列必须唯一;否则与pandas drop_duplicates('publish_date')存在歧义
        if not pivot.columns.is_unique:
            raise ValueError("pivot.columns contains duplicates")

        # 保护:行必须唯一;否则"最后一个True行"可能与pandas排序稳定性不同
        if not pivot.index.is_unique:
            raise ValueError("pivot.index contains duplicates")

        # 保护:QDF的report_quarter通常为PeriodIndex;非PeriodIndex时保守回退
        if not isinstance(pivot.index, pd.PeriodIndex):
            raise ValueError("pivot.index is not PeriodIndex")

        # 确保季度索引升序:这样"最后一个True行"= "report_quarter最大"(与_apply_dedup一致)
        pivot_sorted = pivot
        if not pivot_sorted.index.is_monotonic_increasing:
            pivot_sorted = pivot_sorted.sort_index(axis=0)

        values = pivot_sorted.to_numpy(copy=False)
        values = values.astype(np.float64, copy=False)

        # 已披露判定:value非NaN
        published = ~np.isnan(values)

        # 尝试从manager获取is_na_mask来识别"已披露但为空"
        manager = getattr(self._finance_series, 'manager', None)
        if manager is not None and hasattr(manager, 'precomputed') and manager.precomputed is not None:
            is_na_mask = manager.precomputed.is_na_pivots.get(self._finance_series.col)
            if is_na_mask is not None and not is_na_mask.empty:
                # 标准化列类型(与_apply_dedup一致)
                mask_for_reindex = is_na_mask
                if hasattr(mask_for_reindex.columns, 'to_timestamp'):
                    mask_for_reindex = mask_for_reindex.copy()
                    mask_for_reindex.columns = mask_for_reindex.columns.to_timestamp()
                elif not pd.api.types.is_datetime64_any_dtype(mask_for_reindex.columns):
                    try:
                        mask_for_reindex = mask_for_reindex.copy()
                        mask_for_reindex.columns = pd.to_datetime(mask_for_reindex.columns)
                    except Exception:
                        pass

                # 对齐到pivot(任何异常交由上层捕获回退,避免语义偏差)
                mask_aligned = mask_for_reindex.reindex_like(pivot_sorted).fillna(False)
                published = published | mask_aligned.to_numpy(dtype=bool, copy=False)

        # 对每个publish_date列选择"已披露且最新"的季度;若无已披露则选最新季度
        has_any = published.any(axis=0)
        last_from_bottom = published[::-1, :].argmax(axis=0)
        last_row = (published.shape[0] - 1) - last_from_bottom
        last_row = np.where(has_any, last_row, published.shape[0] - 1).astype(np.int64, copy=False)

        col_idx = np.arange(published.shape[1], dtype=np.int64)
        result_df = pd.DataFrame({
            'report_quarter': pivot_sorted.index.values[last_row],
            'publish_date': pivot_sorted.columns.values,
            'value': values[last_row, col_idx],
        })

        # 与_apply_dedup一致:按publish_date升序并重置索引
        result_df = result_df.sort_values('publish_date').reset_index(drop=True)
        return result_df

    def _apply_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用去重逻辑

        规则:同一发布日保留最新季度

        实现说明:
        - 移除inner merge,与_perform_stack_merge_dedup保持一致
        - 避免inner merge过滤掉ffill后的历史季度数据

        Args:
            df: 长格式DataFrame,包含['report_quarter', 'publish_date', 'value']

        Returns:
            去重后的DataFrame,按publish_date升序排列
        """
        
        # 与Manager的dedup逻辑保持一致
        df = df.copy()
        df['_is_published'] = df['value'].notna()

        # 尝试从manager获取is_na_mask来识别"已披露但为空"
        manager = getattr(self._finance_series, 'manager', None)
        if manager is not None and hasattr(manager, 'precomputed') and manager.precomputed is not None:
            is_na_mask = manager.precomputed.is_na_pivots.get(self._finance_series.col)

            if is_na_mask is not None and not is_na_mask.empty:
                # 将is_na_mask转为(report_quarter, publish_date)的集合
                # 标准化列类型
                mask_for_stack = is_na_mask.copy()
                if hasattr(mask_for_stack.columns, 'to_timestamp'):
                    mask_for_stack.columns = mask_for_stack.columns.to_timestamp()
                elif not pd.api.types.is_datetime64_any_dtype(mask_for_stack.columns):
                    try:
                        mask_for_stack.columns = pd.to_datetime(mask_for_stack.columns)
                    except Exception:
                        pass

                # stack得到True的(report_quarter, publish_date)索引
                mask_true = mask_for_stack.fillna(False).stack(dropna=False)
                mask_true = mask_true[mask_true]  # 只保留True的
                if len(mask_true) > 0:
                    mask_true_idx = mask_true.index
                    # 检查df中的key是否在mask中
                    df_keys = pd.MultiIndex.from_frame(df[['report_quarter', 'publish_date']])
                    df['_is_published'] = df['_is_published'] | df_keys.isin(mask_true_idx)

        df = df.sort_values(
            ['publish_date', '_is_published', 'report_quarter'],
            ascending=[True, False, False]
        )
        df = df.drop_duplicates('publish_date', keep='first').drop(columns=['_is_published'])

        # 提取需要的列并排序
        result_df = df[['report_quarter', 'publish_date', 'value']].copy()
        result_df = result_df.sort_values('publish_date').reset_index(drop=True)

        return result_df

    # ========== NumPy快速路径静态方法 ==========

    @staticmethod
    def _numpy_shift_2d(values: np.ndarray, periods: int) -> np.ndarray:
        """2D数组按行shift(NumPy实现)

        Args:
            values: 2D数组 (n_rows, n_cols)
            periods: 偏移量(正=向下,负=向上)

        Returns:
            偏移后的数组,边界填充NaN
        """
        periods = int(periods)
        out = np.full(values.shape, np.nan, dtype=np.float64)
        if periods == 0:
            out[:] = values
        elif periods > 0:
            if periods < values.shape[0]:
                out[periods:, :] = values[:-periods, :]
        else:
            p = -periods
            if p < values.shape[0]:
                out[:-p, :] = values[p:, :]
        return out

    @staticmethod
    def _numpy_diff_2d(values: np.ndarray, periods: int) -> np.ndarray:
        """2D数组按行diff(NumPy实现)

        Args:
            values: 2D数组 (n_rows, n_cols)
            periods: 差分间隔

        Returns:
            差分后的数组,边界填充NaN
        """
        periods = int(periods)
        out = np.full(values.shape, np.nan, dtype=np.float64)
        if periods == 0:
            return values - values
        if periods > 0:
            if periods < values.shape[0]:
                out[periods:, :] = values[periods:, :] - values[:-periods, :]
        else:
            p = -periods
            if p < values.shape[0]:
                out[:-p, :] = values[:-p, :] - values[p:, :]
        return out

    @staticmethod
    def _numpy_pct_change_2d(values: np.ndarray, periods: int) -> np.ndarray:
        """2D数组按行pct_change(NumPy实现)

        Args:
            values: 2D数组 (n_rows, n_cols)
            periods: 变化间隔

        Returns:
            百分比变化数组
        """
        shifted = QuarterDataFrame._numpy_shift_2d(values, periods)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (values / shifted) - 1.0

    @staticmethod
    def _numpy_rolling_sum_2d(values: np.ndarray, window: int) -> np.ndarray:
        """cumsum实现的rolling sum(min_periods=window语义)

        Args:
            values: 2D数组 (n_rows, n_cols)
            window: 窗口大小

        Returns:
            rolling sum数组,不足窗口的位置为NaN
        """
        window = int(window)
        n_rows, n_cols = values.shape
        out = np.full(values.shape, np.nan, dtype=np.float64)
        if window <= 0 or n_rows < window:
            return out

        mask = np.isnan(values)
        x = np.where(mask, 0.0, values)
        csum = np.cumsum(x, axis=0, dtype=np.float64)
        cnt = np.cumsum(~mask, axis=0, dtype=np.int32)

        # pad:便于做窗口差分
        csum = np.vstack([np.zeros((1, n_cols), dtype=np.float64), csum])
        cnt = np.vstack([np.zeros((1, n_cols), dtype=np.int32), cnt])

        win_sum = csum[window:] - csum[:-window]
        win_cnt = cnt[window:] - cnt[:-window]

        # 语义对齐pandas:min_periods=window -> 必须满窗口非NaN
        valid = win_cnt >= window
        tail = out[window - 1:, :]
        tail[valid] = win_sum[valid]
        return out

    @staticmethod
    def _numpy_rolling_mean_2d(values: np.ndarray, window: int) -> np.ndarray:
        """cumsum实现的rolling mean

        Args:
            values: 2D数组 (n_rows, n_cols)
            window: 窗口大小

        Returns:
            rolling mean数组
        """
        out = QuarterDataFrame._numpy_rolling_sum_2d(values, window)
        with np.errstate(invalid='ignore'):
            out[window - 1:, :] = out[window - 1:, :] / float(window)
        return out

    @staticmethod
    def _numpy_rolling_var_std_stable_2d(
        values: np.ndarray, window: int, ddof: int, is_std: bool
    ) -> np.ndarray:
        """两遍法实现的rolling var/std(数值稳定)

        Args:
            values: 2D数组 (n_rows, n_cols)
            window: 窗口大小
            ddof: 自由度
            is_std: True=返回std,False=返回var

        Returns:
            rolling var/std数组
        """
        window = int(window)
        ddof = int(ddof)
        n_rows, n_cols = values.shape
        out = np.full(values.shape, np.nan, dtype=np.float64)
        denom = window - ddof
        if window <= 0 or n_rows < window or denom <= 0:
            return out

        tmp = np.empty((window, n_cols), dtype=np.float64)
        for end in range(window - 1, n_rows):
            w = values[end - window + 1:end + 1, :]
            mean = w.mean(axis=0)  # 任意NaN会传播为NaN -> 等价min_periods=window
            np.subtract(w, mean, out=tmp)
            np.square(tmp, out=tmp)
            var = tmp.sum(axis=0) / float(denom)
            var = np.maximum(var, 0.0)  # 浮点误差兜底
            if is_std:
                var = np.sqrt(var)
            out[end, :] = var
        return out

    def _new_instance(self, df: pd.DataFrame) -> 'QuarterDataFrame':
        """创建新实例(Immutable模式)

        每次操作返回新实例,不修改原数据.这样做的好处:
        - 线程安全
        - 易于调试(可以检查中间结果)
        - 符合函数式编程理念

        Args:
            df: 新的长格式DataFrame

        Returns:
            新的QuarterDataFrame实例
        """
        new_qdf = QuarterDataFrame(self._finance_series, self._pivot)
        new_qdf._df = df
        new_qdf._domain = 'long'  # 算术运算结果落在long域,禁止再回到pivot季度算子
        return new_qdf

    def _assert_pivot_domain(self, op_name: str) -> None:
        """断言当前对象处于pivot域,避免链式调用静默错误.

        QuarterDataFrame 内部存在两种"计算域":
        - pivot 域:以 _pivot 为准,diff/shift/pct_change/rolling 等季度算子在此域上计算
        - long 域:以 _df 为准,算术运算在此域上计算;此时 _pivot 仍是原始值

        如果在 long 域对象上继续调用季度算子,会读取旧 _pivot 并静默忽略算术结果.
        """
        if getattr(self, '_domain', 'pivot') == 'pivot':
            return

        raise ValueError(
            f"QuarterDataFrame 链式调用顺序错误:{op_name}() 只能在 pivot 域对象上使用.\n"
            "你当前对象来自算术运算(long 域,数据在 _df),而季度算子在 pivot 域(_pivot)上计算.\n"
            "继续执行会静默忽略算术结果,得到错误结果.\n"
            "修正:把 diff/shift/pct_change/rolling 放在算术运算之前.\n"
            "注意:A.diff()/B.diff() ≠ (A/B).diff()(rolling/pct_change/shift 同理).\n"
            "如需严格对 (A/B) 结果做季度算子,需要先在 pivot 域构造 (A/B) 的 PIT 矩阵再调用季度算子(当前版本未提供公开API)."
        )

    def _reindex_pivot_for_quarter_ops(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """

        问题:Q1=100, Q2缺失, Q3=160 -> shift(1)让Q3获取Q1的100(错误)
        修复:用report_quarter reindex补齐Q2=NaN -> shift(1)让Q3获取Q2=NaN(正确)

        Args:
            pivot: 原始pivot (可能有缺失季度)

        Returns:
            补齐后的pivot (min~max之间无缺口)
        """
        if pivot.empty or len(pivot) == 0:
            return pivot

        idx = pivot.index
        if not isinstance(idx, pd.PeriodIndex):
            return pivot

        freq = idx.freq or idx.inferred_freq or 'Q'
        full_idx = pd.period_range(idx.min(), idx.max(), freq=freq)

        if len(full_idx) == len(idx):
            return pivot  # 已连续,无需reindex

        return pivot.reindex(full_idx)

    def _pivot_to_pit_long_fast(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """

        与_to_long_format的区别:
        - _to_long_format: stack+去重(会把同季度修订版压扁成一个值)
        - _pivot_to_pit_long: 保留pivot矩阵结构,每列独立(修订版只影响对应列及之后)

        Args:
            pivot: 计算后的pivot (每列=一个发布日视图)

        Returns:
            长格式DataFrame (report_quarter, publish_date, value)
            按publish_date升序,同publish_date内按report_quarter降序
        """
        if pivot.empty:
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])
        stacked = pivot.stack(dropna=False).reset_index()
        stacked.columns = ['report_quarter', 'publish_date', 'value']
        stacked = stacked.copy()
        stacked['_has_val'] = stacked['value'].notna()
        stacked = stacked.sort_values(
            ['publish_date', '_has_val', 'report_quarter'],
            ascending=[True, False, False]
        )
        stacked = stacked.drop_duplicates('publish_date', keep='first').drop(columns=['_has_val'])

        result = stacked[['report_quarter', 'publish_date', 'value']].copy()
        result = result.sort_values('publish_date').reset_index(drop=True)

        return result

    # ========== 基础操作 ==========

    def diff(self, periods: int = 1) -> 'QuarterDataFrame':
        """季度差分

        Args:
            periods: 差分间隔(默认1,即相邻季度)
                    1 = 环比差分(Q2 - Q1)
                    4 = 同比差分(2021Q1 - 2020Q1)

        Returns:
            新的QuarterDataFrame实例,包含差分后的值

        Example:
            >>> # 计算季度环比变化
            >>> qdf.diff()
            >>> # 计算同比变化
            >>> qdf.diff(periods=4)

        Note:
            - 问题:keep='last'+map() 让同季度修订版回填到早期发布日
            - 修复:在pivot矩阵按列计算diff,每列独立(修订版只影响该列及之后)
        """
        self._assert_pivot_domain('diff')
        pivot = self._pivot.copy()
        pivot = self._reindex_pivot_for_quarter_ops(pivot)  # 补齐缺失季度
        if _USE_NUMPY_QDF_FASTPATH:
            try:
                values = pivot.to_numpy(dtype=np.float64, copy=False)
                result_arr = self._numpy_diff_2d(values, periods)
                result_pivot = pd.DataFrame(result_arr, index=pivot.index, columns=pivot.columns)
            except Exception:
                result_pivot = pivot.diff(periods=periods)
        else:
            result_pivot = pivot.diff(periods=periods)

        # v3修复:保留运算后的pivot,以便to_dataframe()输出PIT矩阵
        new_qdf = QuarterDataFrame(self._finance_series, result_pivot)
        new_qdf._df = None  # 标记为未计算
        return new_qdf

    def pct_change(self, periods: int = 1) -> 'QuarterDataFrame':
        """百分比变化

        Args:
            periods: 变化间隔
                    1 = 环比增长率
                    4 = 同比增长率

        Returns:
            新的QuarterDataFrame实例,包含增长率

        Example:
            >>> # 环比增长率
            >>> qdf.pct_change()
            >>> # 同比增长率
            >>> qdf.pct_change(periods=4)

        Note:
            - 问题:keep='last'+map() 让同季度修订版回填到早期发布日
            - 修复:在pivot矩阵按列计算pct_change,每列独立(修订版只影响该列及之后)
        """
        self._assert_pivot_domain('pct_change')
        pivot = self._pivot.copy()
        pivot = self._reindex_pivot_for_quarter_ops(pivot)
        if _USE_NUMPY_QDF_FASTPATH:
            try:
                values = pivot.to_numpy(dtype=np.float64, copy=False)
                result_arr = self._numpy_pct_change_2d(values, periods)
                result_pivot = pd.DataFrame(result_arr, index=pivot.index, columns=pivot.columns)
            except Exception:
                result_pivot = pivot.pct_change(periods=periods, fill_method=None)
        else:
            result_pivot = pivot.pct_change(periods=periods, fill_method=None)

        # v3修复:保留运算后的pivot,以便to_dataframe()输出PIT矩阵
        new_qdf = QuarterDataFrame(self._finance_series, result_pivot)
        new_qdf._df = None  # 标记为未计算
        return new_qdf

    def shift(self, periods: int = 1) -> 'QuarterDataFrame':
        """季度偏移

        Args:
            periods: 偏移量
                    正数:向后偏移(获取历史值)
                    负数:向前偏移(获取未来值,谨慎使用)

        Returns:
            新的QuarterDataFrame实例,包含偏移后的值

        Example:
            >>> # 获取上一个季度的值
            >>> qdf.shift(1)
            >>> # 获取去年同期的值
            >>> qdf.shift(4)

        Note:
            - 问题:keep='last'+map() 让同季度修订版回填到早期发布日
            - 修复:在pivot矩阵按列计算shift,每列独立(修订版只影响该列及之后)
            - 问题A:shift会把值推到未来季度行,被dedup按publish_date选期抵消
            - 问题B:季度完全不存在时,shift结果为NaN会回退旧季度
            - 修复:result_pivot[pivot.isna()] = np.nan + 构造_align_is_na_mask
        """
        self._assert_pivot_domain('shift')
        pivot = self._pivot.copy()
        pivot = self._reindex_pivot_for_quarter_ops(pivot)
        if _USE_NUMPY_QDF_FASTPATH:
            try:
                values = pivot.to_numpy(dtype=np.float64, copy=False)
                result_arr = self._numpy_shift_2d(values, periods)
                result_pivot = pd.DataFrame(result_arr, index=pivot.index, columns=pivot.columns)
            except Exception:
                result_pivot = pivot.shift(periods=periods)
        else:
            result_pivot = pivot.shift(periods=periods)
        result_pivot[pivot.isna()] = np.nan
        need_inf_mask = None
        if int(periods) != 0:
            manager = getattr(self._finance_series, 'manager', None)
            col = getattr(self._finance_series, 'col', None)
            is_na_mask = None
            if manager is not None and hasattr(manager, 'precomputed') and manager.precomputed is not None and col is not None:
                is_na_mask = manager.precomputed.is_na_pivots.get(col)

            published_current = pivot.notna()
            if is_na_mask is not None:
                try:
                    is_na_aligned = is_na_mask.reindex_like(pivot).fillna(False)
                    published_current = published_current | is_na_aligned
                except Exception:
                    # 掩码对齐失败时保守降级:只基于pivot.notna()判断"已披露"
                    pass

            need_inf_mask = published_current & result_pivot.isna()

        # v3修复:保留运算后的pivot,以便to_dataframe()输出PIT矩阵
        new_qdf = QuarterDataFrame(self._finance_series, result_pivot)
        new_qdf._df = None  # 标记为未计算
        new_qdf._align_is_na_mask = need_inf_mask
        return new_qdf

    # ========== 滚动统计 ==========

    def rolling(self, window: int) -> 'RollingQuarterDataFrame':
        """
        创建滚动窗口对象(支持链式统计方法)

        Args:
            window: 窗口大小(季度数)

        Returns:
            RollingQuarterDataFrame: 支持链式统计方法的中间对象

        Example:
            >>> # 8季度滚动均值
            >>> qdf.rolling(8).mean()
            >>> # 8季度滚动标准差
            >>> qdf.rolling(8).std()
            >>> # 组合使用:变异系数
            >>> cv = qdf.rolling(8).std() / qdf.rolling(8).mean()
        """
        self._assert_pivot_domain('rolling')
        return RollingQuarterDataFrame(self, window)

    # ========== 对齐方法(终结操作)==========

    def to_series(self, name: str = None) -> pd.Series:
        """
        对齐到交易日(主接口,终结操作)

        这是终结操作,结束链式调用,返回已对齐到交易日的Series.
        内部调用财务库的_align_series高性能对齐方法,确保结果与
        原生方法(如fin_mgr[col].quarter())100%一致.

        Args:
            name: 可选,Series的名称(默认使用财务字段名)

        Returns:
            pd.Series,index=交易日期,value=财务数据(自动前向填充)

        Example:
            >>> # 对齐到交易日
            >>> result = qdf.diff().rolling(8).mean().to_series()
            >>> # 指定名称
            >>> result = qdf.diff().to_series(name='NP_Delta')

        Note:
            - 如果有操作(_df不为None),直接从long format对齐(避免unstack丢失结构)
            - 如果未操作,使用原始pivot
            - 自动前向填充到每个交易日
        """
        # 关键修复:如果经过操作,直接从long format对齐
        if self._df is not None:
            # 使用操作后的long format数据
            df = self._df.copy()
            manager = self._finance_series.manager
            if 'report_quarter' in df.columns:
                df = df.sort_values(['publish_date', 'report_quarter'], ascending=[True, False])
                df = df.drop_duplicates('publish_date', keep='first')

            # 只过滤publish_date的NaN,保留value的NaN(那是操作的正常结果,如diff的第一个值)
            temp = df[['publish_date', 'value']].dropna(subset=['publish_date'])
            temp = temp.sort_values('publish_date')

            # 使用searchsorted进行高效对齐
            pub_dates_int64 = temp['publish_date'].values.astype('datetime64[ns]').view(np.int64)
            pub_values = temp['value'].values
            if hasattr(manager, '_trade_dates_int64_sorted') and manager._trade_dates_int64_sorted is not None:
                trade_dates_int64 = manager._trade_dates_int64_sorted
                sort_idx = getattr(manager, '_sort_idx', None)
            else:
                # 回退:直接从trade_date_df计算
                trade_dates_int64 = manager.trade_date_df['交易日期'].values.astype('datetime64[ns]').view(np.int64)
                sort_idx = None

            # 对每个交易日,找到 <= 该交易日的最近发布日
            indices = np.searchsorted(pub_dates_int64, trade_dates_int64, side='right') - 1

            # 构建结果数组
            result_values = np.full(len(trade_dates_int64), np.nan)
            valid_mask = indices >= 0
            result_values[valid_mask] = pub_values[indices[valid_mask]]
            if sort_idx is not None:
                result_values_original_order = np.empty_like(result_values)
                result_values_original_order[sort_idx] = result_values
                result_values = result_values_original_order

            result = pd.Series(result_values, index=manager.trade_date_df.index, name=name or self._finance_series.col)
        else:
            # 未经操作,使用原始pivot
            pivot = self._pivot
            # 传递is_na_mask避免NaN被stack丢弃后前向填充
            manager = self._finance_series.manager
            is_na_mask = getattr(self, '_align_is_na_mask', None)
            if is_na_mask is None and hasattr(manager, 'precomputed') and manager.precomputed is not None:
                is_na_mask = manager.precomputed.is_na_pivots.get(self._finance_series.col)
            result = manager._fast_align(pivot, name or self._finance_series.col, is_na_mask)
            if name is not None:
                result.name = name
        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    def to_daily(self, name: str = None) -> pd.Series:
        """
        对齐到交易日(别名,等价于to_series)

        提供更直观的命名:转为日频数据

        Args:
            name: 可选,Series的名称

        Returns:
            pd.Series,index=交易日期

        Example:
            >>> # 语义更清晰:转为日频数据
            >>> daily_sue = sue_qdf.to_daily('SUE')
        """
        return self.to_series(name=name)

    def align(self, column: str = None) -> pd.Series:
        """
        对齐到交易日(向后兼容接口,不推荐使用)

        Args:
            column: 保留参数,无实际作用(QuarterDataFrame只有一列value)

        Returns:
            pd.Series,index=交易日期

        Note:
            新代码请使用 to_series() 或 to_daily()
            此方法仅为向后兼容保留
        """
        import warnings
        warnings.warn(
            "align()方法已弃用,请使用 to_series() 或 to_daily()",
            DeprecationWarning,
            stacklevel=2
        )
        return self.to_series()

    # ========== 调试方法 ==========

    def to_dataframe(self) -> pd.DataFrame:
        """
        导出长格式DataFrame(调试用,v3修复:输出PIT矩阵)

        Returns:
            pd.DataFrame,包含以下列:
                - report_quarter: pd.Period,季度索引(如2020Q1)
                - publish_date: pd.Timestamp,发布日期
                - value: float,财务数据值
            排序:按publish_date升序

        用途:
            - 调试:查看中间计算结果
            - 检查:验证PIT语义(同一季度可能有多个发布日)
            - 分析:理解修订版影响范围

        Note:
            v3修复:不再按publish_date去重,保留所有(report_quarter, publish_date)组合
            - 旧行为:同一季度只保留最新发布日 -> 无法观察修订版
            - 新行为:保留所有发布日 -> 可验证修订版只影响发布日之后

        Example:
            >>> # 查看中间计算结果(含修订版)
            >>> delta_df = qdf.diff().to_dataframe()
            >>> print(delta_df[delta_df['report_quarter'] == '2010Q4'])
        """
        # v3修复:直接从pivot生成long format,不去重
        if self._pivot.empty:
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])

        # stack(dropna=False)保留NaN
        stacked = self._pivot.stack(dropna=False).reset_index()
        stacked.columns = ['report_quarter', 'publish_date', 'value']

        # 只按publish_date排序,不去重(保留所有发布日)
        result = stacked.sort_values('publish_date').reset_index(drop=True)
        return result

    def to_long(self) -> pd.DataFrame:
        """
        导出长格式DataFrame(别名,等价于to_dataframe)

        Returns:
            pd.DataFrame
        """
        return self.to_dataframe()

    # ========== 内部辅助方法 ==========

    def _get_long_for_ops(self) -> pd.DataFrame:
        """获取用于运算的long format DataFrame"""
        if self._df is not None:
            return self._df.copy()
        else:
            # 从pivot转为long format
            return self._to_long_format()

    def _binary_op_numpy_fast(self, left: pd.DataFrame, right: pd.DataFrame, op) -> pd.DataFrame:
        """NumPy快速路径:在long域对齐两份QDF数据并执行二元运算(绕开pandas merge开销).
        Benchmark: 3.5ms -> 1.9ms (1.8x加速)

        语义目标(必须等价于):
            left.merge(right, on=['report_quarter','publish_date'], how='inner')

        对齐规则:
            1) publish_date 作为主键对齐(要求两边唯一;由_to_long_format去重保证)
            2) report_quarter 必须一致才参与运算(否则视为不匹配,等价inner join)

        失败策略:
            - 任意异常上抛,由上层统一捕获并回退到pandas merge,保证兼容性与正确性.
        """
        if left is None or right is None or left.empty or right.empty:
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])

        need_cols = {'report_quarter', 'publish_date', 'value'}
        if (not need_cols.issubset(left.columns)) or (not need_cols.issubset(right.columns)):
            raise ValueError("long格式缺少必要列:report_quarter/publish_date/value")

        # 保护:publish_date 必须唯一,否则np.intersect1d(return_indices)与merge语义不一致
        if (not left['publish_date'].is_unique) or (not right['publish_date'].is_unique):
            raise ValueError("publish_date not unique")

        # 保护:tz-aware时间在numpy/pandas间可能出现语义差异,保守回退
        if pd.api.types.is_datetime64tz_dtype(left['publish_date']) or pd.api.types.is_datetime64tz_dtype(right['publish_date']):
            raise ValueError("publish_date is tz-aware, fallback to pandas merge")

        # 转为numpy datetime64[ns]用于快速求交集
        left_dates = left['publish_date'].values.astype('datetime64[ns]')
        right_dates = right['publish_date'].values.astype('datetime64[ns]')

        # 交集对齐(publish_date唯一时:每个日期最多对应一个索引)
        common_dates, left_idx, right_idx = np.intersect1d(
            left_dates,
            right_dates,
            assume_unique=True,
            return_indices=True
        )
        if common_dates.size == 0:
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])

        # report_quarter 必须一致(等价merge on两个key)
        left_rq = left['report_quarter'].values[left_idx]
        right_rq = right['report_quarter'].values[right_idx]
        rq_match = (left_rq == right_rq)
        if not np.any(rq_match):
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])

        left_idx2 = left_idx[rq_match]
        right_idx2 = right_idx[rq_match]
        out_values = op(left['value'].values[left_idx2], right['value'].values[right_idx2])

        return pd.DataFrame({
            'report_quarter': left_rq[rq_match],
            'publish_date': common_dates[rq_match],
            'value': out_values,
        })

    def _binary_op_long(self, left: pd.DataFrame, right: pd.DataFrame, op) -> pd.DataFrame:
        """二元算术运算统一入口:优先NumPy对齐,失败自动回退到pandas merge."""
        if left is None or right is None or left.empty or right.empty:
            return pd.DataFrame(columns=['report_quarter', 'publish_date', 'value'])

        if _USE_NUMPY_QDF_FASTPATH:
            try:
                return self._binary_op_numpy_fast(left, right, op)
            except Exception:
                # 保守回退:任何异常都用pandas merge兜底,保证语义一致
                pass

        merged = left.merge(
            right,
            on=['report_quarter', 'publish_date'],
            how='inner',
            suffixes=('_left', '_right')
        )
        merged['value'] = op(
            merged['value_left'].to_numpy(copy=False),
            merged['value_right'].to_numpy(copy=False),
        )
        return merged[['report_quarter', 'publish_date', 'value']]

    # ========== 算术运算符 ==========

    def __add__(self, other) -> 'QuarterDataFrame':
        """加法运算符:qdf1 + qdf2 或 qdf + scalar"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            # QuarterDataFrame + QuarterDataFrame(
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.add)
        else:
            # QuarterDataFrame + scalar
            df['value'] = df['value'] + other
            result_df = df
        return self._new_instance(result_df)

    def __radd__(self, other) -> 'QuarterDataFrame':
        """反向加法:scalar + qdf"""
        return self.__add__(other)

    def __sub__(self, other) -> 'QuarterDataFrame':
        """减法运算符:qdf1 - qdf2 或 qdf - scalar"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.subtract)
        else:
            df['value'] = df['value'] - other
            result_df = df
        return self._new_instance(result_df)

    def __rsub__(self, other) -> 'QuarterDataFrame':
        """反向减法:scalar - qdf"""
        df = self._get_long_for_ops()
        df['value'] = other - df['value']
        return self._new_instance(df)

    def __mul__(self, other) -> 'QuarterDataFrame':
        """乘法运算符:qdf1 * qdf2 或 qdf * scalar"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.multiply)
        else:
            df['value'] = df['value'] * other
            result_df = df
        return self._new_instance(result_df)

    def __rmul__(self, other) -> 'QuarterDataFrame':
        """反向乘法:scalar * qdf"""
        return self.__mul__(other)

    def __truediv__(self, other) -> 'QuarterDataFrame':
        """除法运算符:qdf1 / qdf2 或 qdf / scalar"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.divide)
        else:
            df['value'] = df['value'] / other
            result_df = df
        return self._new_instance(result_df)

    def __rtruediv__(self, other) -> 'QuarterDataFrame':
        """反向除法:scalar / qdf"""
        df = self._get_long_for_ops()
        df['value'] = other / df['value']
        return self._new_instance(df)

    # ========== 高级运算符(

    def __pow__(self, other) -> 'QuarterDataFrame':
        """幂运算符:qdf ** n 或 qdf1 ** qdf2"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.power)
        else:
            df['value'] = df['value'] ** other
            result_df = df
        return self._new_instance(result_df)

    def __rpow__(self, other) -> 'QuarterDataFrame':
        """反向幂运算:scalar ** qdf(仅支持标量)"""
        if isinstance(other, QuarterDataFrame):
            # 理论上不会走到这里,但防御性处理
            return other.__pow__(self)
        df = self._get_long_for_ops()
        df['value'] = other ** df['value']
        return self._new_instance(df)

    def __mod__(self, other) -> 'QuarterDataFrame':
        """取模运算符:qdf % n 或 qdf1 % qdf2"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.mod)
        else:
            df['value'] = df['value'] % other
            result_df = df
        return self._new_instance(result_df)

    def __rmod__(self, other) -> 'QuarterDataFrame':
        """反向取模:scalar % qdf(仅支持标量)"""
        if isinstance(other, QuarterDataFrame):
            return other.__mod__(self)
        df = self._get_long_for_ops()
        df['value'] = other % df['value']
        return self._new_instance(df)

    def __floordiv__(self, other) -> 'QuarterDataFrame':
        """整除运算符:qdf // n 或 qdf1 // qdf2"""
        df = self._get_long_for_ops()
        if isinstance(other, QuarterDataFrame):
            other_df = other._get_long_for_ops()
            result_df = self._binary_op_long(df, other_df, np.floor_divide)
        else:
            df['value'] = df['value'] // other
            result_df = df
        return self._new_instance(result_df)

    def __rfloordiv__(self, other) -> 'QuarterDataFrame':
        """反向整除:scalar // qdf(仅支持标量)"""
        if isinstance(other, QuarterDataFrame):
            return other.__floordiv__(self)
        df = self._get_long_for_ops()
        df['value'] = other // df['value']
        return self._new_instance(df)

    # ========== 一元运算符 ==========

    def __neg__(self) -> 'QuarterDataFrame':
        """负号运算符:-qdf"""
        df = self._get_long_for_ops()
        df['value'] = -df['value']
        return self._new_instance(df)

    def __pos__(self) -> 'QuarterDataFrame':
        """正号运算符:+qdf"""
        df = self._get_long_for_ops()
        df['value'] = +df['value']
        return self._new_instance(df)

    def __abs__(self) -> 'QuarterDataFrame':
        """绝对值运算符:abs(qdf)"""
        df = self._get_long_for_ops()
        df['value'] = abs(df['value'])
        return self._new_instance(df)

class RollingQuarterDataFrame:
    """滚动窗口操作的中间对象 - 支持链式统计方法

    设计模式:
    - Proxy模式:代理rolling操作,支持链式调用

    Example:
        >>> # 8季度滚动均值
        >>> qdf.rolling(8).mean()
        >>> # 8季度滚动标准差
        >>> qdf.rolling(8).std()
        >>> # 组合使用
        >>> cv = qdf.rolling(8).std() / qdf.rolling(8).mean()
    """

    def __init__(self, qdf: QuarterDataFrame, window: int):
        """
        Args:
            qdf: QuarterDataFrame对象
            window: 窗口大小(季度数)
        """
        self._qdf = qdf
        self._window = window

    def _apply_rolling(self, func_name: str, **kwargs) -> QuarterDataFrame:
        """统一的rolling操作入口(

        Args:
            func_name: rolling方法名(如'mean', 'std', 'sum'等)
            **kwargs: 传递给rolling方法的额外参数(如ddof)

        Returns:
            QuarterDataFrame: 包含rolling结果的新实例

        Note:
            在pivot矩阵按列rolling,每列独立,避免同季度修订版被压扁
        """
        self._qdf._assert_pivot_domain(f'rolling.{func_name}')
        pivot = self._qdf._pivot.copy()
        pivot = self._qdf._reindex_pivot_for_quarter_ops(pivot)
        numpy_funcs = {'sum', 'mean', 'std', 'var'}
        use_numpy = _USE_NUMPY_QDF_FASTPATH and func_name in numpy_funcs
        result_pivot = None

        if use_numpy:
            try:
                values = pivot.to_numpy(dtype=np.float64, copy=False)
                if func_name == 'sum':
                    result_arr = QuarterDataFrame._numpy_rolling_sum_2d(values, self._window)
                elif func_name == 'mean':
                    result_arr = QuarterDataFrame._numpy_rolling_mean_2d(values, self._window)
                elif func_name == 'std':
                    ddof = kwargs.get('ddof', 1)
                    result_arr = QuarterDataFrame._numpy_rolling_var_std_stable_2d(
                        values, self._window, ddof, is_std=True
                    )
                elif func_name == 'var':
                    ddof = kwargs.get('ddof', 1)
                    result_arr = QuarterDataFrame._numpy_rolling_var_std_stable_2d(
                        values, self._window, ddof, is_std=False
                    )
                else:
                    result_arr = None

                if result_arr is not None:
                    result_pivot = pd.DataFrame(result_arr, index=pivot.index, columns=pivot.columns)
            except Exception:
                result_pivot = None  # 回退到pandas

        # Pandas回退路径
        if result_pivot is None:
            rolling_obj = pivot.rolling(self._window, min_periods=self._window)
            result_pivot = getattr(rolling_obj, func_name)(**kwargs)

        # v3修复:保留运算后的pivot,以便to_dataframe()输出PIT矩阵
        new_qdf = QuarterDataFrame(self._qdf._finance_series, result_pivot)
        new_qdf._df = None  # 标记为未计算
        return new_qdf

    def mean(self) -> QuarterDataFrame:
        """滚动均值

        Returns:
            QuarterDataFrame: 包含滚动均值的新实例
        """
        return self._apply_rolling('mean')

    def std(self, ddof: int = 1) -> QuarterDataFrame:
        """滚动标准差

        Args:
            ddof: 自由度(默认1,样本标准差)
                 0 = 总体标准差
                 1 = 样本标准差(默认,与pandas一致)

        Returns:
            QuarterDataFrame: 包含滚动标准差的新实例
        """
        return self._apply_rolling('std', ddof=ddof)

    def sum(self) -> QuarterDataFrame:
        """滚动求和

        Returns:
            QuarterDataFrame: 包含滚动求和的新实例
        """
        return self._apply_rolling('sum')

    def min(self) -> QuarterDataFrame:
        """滚动最小值

        Returns:
            QuarterDataFrame: 包含滚动最小值的新实例
        """
        return self._apply_rolling('min')

    def max(self) -> QuarterDataFrame:
        """滚动最大值

        Returns:
            QuarterDataFrame: 包含滚动最大值的新实例
        """
        return self._apply_rolling('max')

    def median(self) -> QuarterDataFrame:
        """滚动中位数

        Returns:
            QuarterDataFrame: 包含滚动中位数的新实例
        """
        return self._apply_rolling('median')

    def var(self, ddof: int = 1) -> QuarterDataFrame:
        """滚动方差

        Args:
            ddof: 自由度(默认1,样本方差)
                 0 = 总体方差
                 1 = 样本方差(默认,与pandas一致)

        Returns:
            QuarterDataFrame: 包含滚动方差的新实例
        """
        return self._apply_rolling('var', ddof=ddof)

__all__ = [
    'FactorFinanceManagerV4Fast',
    'create_factor_finance_manager_precomputed',
    'create_factor_finance_manager_from_raw',
    'get_v5_fin_mgr',
    'QuarterDataFrame',
    'RollingQuarterDataFrame',
]
