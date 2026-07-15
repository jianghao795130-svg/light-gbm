
import gc
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import config
from config import n_jobs, factor_col_limit
from core.factor_cache import build_intraday_v2_cache_paths, prepare_intraday_plan, save_intraday_factor_meta
from core.data_center import check_extra_data, merge_extra_data, load_candle_df_dict
from core.fin_essentials import generate_fin_pivot, get_finance_data, merge_with_finance_data_new
from core.finance_facade import prepare_finance_context
from core.model.backtest_config import BacktestConfig
from core.model.factor_config import get_col_name
from core.model.finance_manager import FinanceDataFrame
from core.model.strategy_config import StrategyConfig
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger, divider
from core.utils.misc_kit import save_csv_safely, pd_concat
from core.utils.path_kit import get_file_path

# fmt: off
# 后置过滤所需字段
FILTER_AFTER_COLS = ["下日_是否交易", "下日_开盘涨停", "下日_是否ST", "下日_是否退市", "下日_是否涨停", "下日_一字涨停",
                     "上市至今交易天数"]
# k线字段
KLINE_COLS = ["交易日期", "股票代码", "股票名称"]
# 因子计算之后，需要保存的行情数据
FACTOR_COLS = [
    *KLINE_COLS,
    "复权因子", "开盘价", "最高价", "最低价", "收盘价", "成交额", "是否交易", "流通市值", "总市值",
    *FILTER_AFTER_COLS,
]
# 小时因子计算之后，需要保存的行情数据
HOUR_FACTOR_COLS = [
    *KLINE_COLS,
    "复权因子", "开盘价", "最高价", "最低价", "收盘价", "成交额", "是否交易", "流通市值",
    "开盘价_复权", "最高价_复权", "最低价_复权", "收盘价_复权"
]
# 计算完选股之后，保留的字段
RES_COLS = [
    "选股日期", "股票代码", "股票名称", "策略", "持仓周期", "换仓时间", "目标资金占比",
    "择时信号", "分批进场仓位", "选股因子排名", "调仓类型"
]
# fmt: on


# region step2_计算因子.py
# ================================================================
# step2_计算因子.py
# ================================================================
def cal_strategy_factors(
    conf: BacktestConfig,
    stock_code,
    candle_df,
    fin_data: Dict[str, Union[pd.DataFrame, FinanceDataFrame]] = None,
    factor_col_name_list: List[str] = (),
):
    """
    计算指定股票的策略因子。

    参数:
    conf (BacktestConfig): 策略配置
    stock_code (str): 股票代码
    candle_df (DataFrame): 股票的K线数据，已经按照"交易日期"从小到大排序
    fin_data (dict): 财务数据

    返回:
    DataFrame: 包含计算因子的K线数据
    dict: 因子列的周期转换规则
    """
    factor_series_dict = {}
    before_len = len(candle_df)

    candle_df.sort_values(by="交易日期", inplace=True)  # 防止因子计算出错，计算之前，先进行排序
    factor_params_dict = conf.hour_factor_params_dict if conf.use_hour_data else conf.factor_params_dict
    for factor_name, param_list in factor_params_dict.items():
        factor_file = FactorHub.get_by_name(factor_name)
        # 如果是截面因子，跳过计算
        if factor_file.is_cross:
            continue
        minutes_tuple_set = conf.factor_minutes_dict.get(factor_name, set(()))  # 2025-03-20添加分钟数据的支持
        for minutes_tuple in minutes_tuple_set:
            for param in param_list:
                col_name = get_col_name(factor_name, param, minutes_tuple)
                if col_name in factor_col_name_list:
                    # 因子计算，factor_df是包含因子计算结果的DataFrame，必须是按照"交易日期"从小到大排序
                    # 使用 shallow copy 避免每个因子都做完整的 deep copy，大幅降低内存分配开销
                    factor_df = factor_file.add_factor(
                        candle_df.copy(deep=False),
                        param,
                        fin_data=fin_data,
                        col_name=col_name,
                        minutes=minutes_tuple,  # 2025-03-20添加分钟数据的支持
                    )

                    factor_series_dict[col_name] = factor_df[col_name].values
                    # 检查因子计算是否出错
                    if before_len != len(factor_series_dict[col_name]):
                        logger.error(
                            f"{stock_code}的{factor_name}因子({param}，{col_name})导致数据长度发生变化，请检查！"
                        )
                        raise Exception("因子计算出错，请避免在cal_factors中修改数据行数")

    kline_with_factor_dict = {
        **{col_name: candle_df[col_name] for col_name in (HOUR_FACTOR_COLS if conf.use_hour_data else FACTOR_COLS)},
        **factor_series_dict,
    }
    kline_with_factor_df = pd.DataFrame(kline_with_factor_dict)
    kline_with_factor_df.sort_values(by="交易日期", inplace=True)

    # 根据回测设置的时间区间进行裁切
    start_date = conf.start_date or kline_with_factor_df["交易日期"].min()
    end_date = conf.end_date or kline_with_factor_df["交易日期"].max()
    date_cut_condition = (kline_with_factor_df["交易日期"] >= start_date) & (
        kline_with_factor_df["交易日期"] <= end_date
    )

    return kline_with_factor_df[date_cut_condition].reset_index(drop=True)  # 返回计算完的因子数据


def process_by_stock(
    conf: BacktestConfig,
    candle_df: pd.DataFrame,
    factor_col_name_list: List[str],
    idx: int,
    trim_before: pd.Timestamp = None,
):
    """
    组装因子计算必要的数据结构，并且送入到因子计算函数中进行计算
    :param conf: 回测策略配置
    :param candle_df: 单只股票的K线数据
    :param factor_col_name_list: 需要计算的因子列名称列表
    :param idx: 股票索引
    :param trim_before: 若非 None，则只返回交易日期 > trim_before 的行（用于增量计算去除 lookback 部分）
    :return: idx, factor_df
    """
    stock_code = candle_df.iloc[-1]["股票代码"]

    if getattr(config, "official_finance", False):
        # 导入财务数据，将个股数据与财务数据合并，并计算财务指标的衍生指标
        if conf.fin_cols:  # 前面已经做了预检，这边只需要动态台南佳即可
            # 分别为：个股数据、财务数据、原始财务数据（不抛弃废弃的报告数据）
            candle_df, origin_fin_df = get_finance_data(conf, stock_code, candle_df)
            pivot_dict, new_fin_df = generate_fin_pivot(origin_fin_df, candle_df, conf.fin_cols)
            fin_data_ins = FinanceDataFrame(
                trade_date_df=candle_df[["交易日期"]], raw_fin_df=new_fin_df, pivot_dict=pivot_dict
            )
            # 将所需要计算的col，合并到candle_df中
            candle_df = merge_with_finance_data_new(conf, candle_df, fin_data_ins)
            fin_data = {"财务数据对象": fin_data_ins}
        else:
            fin_data = None
    else:
        candle_df, fin_data = prepare_finance_context(conf, stock_code, candle_df)

    if conf.extra_data:
        # 个股数据与其他数据合并
        for data_name in conf.extra_data.keys():
            candle_df = merge_extra_data(candle_df, data_name, conf.extra_data[data_name])

    factor_df = cal_strategy_factors(conf, stock_code, candle_df, fin_data, factor_col_name_list)

    # 增量模式：裁剪掉 lookback 部分，只保留新增数据
    if trim_before is not None:
        factor_df = factor_df[factor_df["交易日期"] > trim_before].reset_index(drop=True)

    return idx, factor_df


def calculate_factors(conf: BacktestConfig, boost: bool = True):
    """
    计算所有股票的因子，分为三步：
    1. 加载股票K线数据
    2. 计算每个股票的因子，并存储到列表
    3. 合并所有因子数据并存储

    参数:
    conf (BacktestConfig): 回测配置
    """
    logger.info("因子计算...")
    s_time = time.time()

    # 获取需要计算的因子名列表
    if conf.use_hour_data:
        factor_col_name_list = conf.hour_factor_col_name_list
        # 如果不需要重新计算，则直接退出
        if all(
            (conf.get_factor_folder() / f"factor_hour_{factor_col_name}.parquet").exists()
            for factor_col_name in factor_col_name_list
        ):
            logger.info("没有需要重新计算的小时因子")
            return
    else:
        factor_col_name_list = conf.factor_col_name_list

    # ====================================================================================================
    # 1. 加载股票K线数据
    # ====================================================================================================
    logger.debug("🛂 配置信息检查...")
    if len(conf.fin_cols) > 0 and not conf.has_fin_data:
        logger.warning(f"策略需要财务因子{conf.fin_cols}，但缺少财务数据路径")
        raise ValueError("请在 config.py 中配置财务数据路径")
    elif len(conf.fin_cols) > 0:
        logger.debug(f"ℹ️ 检测到财务因子：{conf.fin_cols}")
    else:
        logger.debug("ℹ️ 检测到没有财务因子")

    if len(conf.extra_data.keys()) > 0:
        logger.debug(f"🔍 检测到外部数据：{list(conf.extra_data.keys())}")
        for data_name in conf.extra_data.keys():
            is_ok, msg = check_extra_data(data_name)
            if not is_ok:
                logger.error(f"外部数据检测失败：{msg}")
                sys.exit(2)
    else:
        logger.debug("🔍 检测到没有外部数据")

    logger.debug(f"💿 读取{'小时' if conf.use_hour_data else ''}股票K线数据...")

    if conf.use_hour_data:
        exist_path = conf.stock_preprocess_data_path.is_dir() and any(conf.stock_preprocess_data_path.iterdir())
        if not exist_path:
            # script_path = get_file_path("program", "小时数据预处理.py")
            # if not script_path.exists():
            #     logger.error(f"小时数据预处理脚本不存在：{script_path}")
            #     sys.exit(2)
            # logger.warning(
            #     "未找到预处理小时数据，请先执行 program/小时数据预处理.py 脚本。\n不执行也没事，3秒后框架将自动执行该脚本，大概需要3分钟左右"
            # )
            # time.sleep(3)
            # execute_preprocess_script()
            logger.error(
                "请前往官网下载【股票1小时k线数据Pro】：https://www.quantclass.cn/data/stock/stock-1h-trading-data-pro\n"
                "把【stock-1h-trading-data-pro】放在【数据中心路径】下"
            )
            sys.exit(2)
        # 小时预处理数据通过数据中心获取
        candle_df_dict: Dict[str, pd.DataFrame] = load_candle_df_dict(conf.stock_preprocess_data_path, boost=boost)
    else:
        candle_df_dict: Dict[str, pd.DataFrame] = pd.read_pickle(conf.get_runtime_folder() / "股票预处理数据.pkl")

    # ====================================================================================================
    # 2. 计算因子并存储结果
    # ====================================================================================================
    factor_col_count = len(factor_col_name_list)
    shards = range(0, factor_col_count, factor_col_limit)

    logger.debug(
        f"""* 总共计算因子个数：{factor_col_count} 个
* 单次计算因子个数：{factor_col_limit} 个，(需分成{len(shards)}组计算)
* 需要计算币种数量：{len(candle_df_dict.keys())} 个"""
    )

    # 清理 cache 的缓存
    if conf.use_hour_data:
        all_kline_pkl = conf.get_factor_folder() / "all_hour_factors_kline.parquet"
        all_kline_pkl.unlink(missing_ok=True)
    else:
        all_kline_pkl = conf.get_runtime_folder() / "all_factors_kline.pkl"
        all_kline_pkl.unlink(missing_ok=True)

    # 所有截面因子的col_name
    section_col_name_list = [x.col_name for x in conf.section_factor_list]

    # ** 注意 **
    # `tqdm`是一个显示为进度条的，非常有用的工具
    # 目前是串行模式，比较适合debug和测试。
    logger.debug(f"🚀 多进程计算因子，进程数量：{n_jobs}" if boost else "🚲 单进程计算因子")
    for shard_index in shards:
        shard_num = int(shard_index / factor_col_limit) + 1
        logger.debug(f"🗂️ 因子分片计算中，进度：{shard_num}/{len(shards)}")
        factor_col_name_shard = factor_col_name_list[shard_index : shard_index + factor_col_limit]

        all_factor_df_list = [pd.DataFrame()] * len(candle_df_dict.keys())  # 计算结果会存储在这个列表
        if boost:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for candle_idx, candle_df in enumerate(candle_df_dict.values()):
                    futures.append(
                        executor.submit(process_by_stock, conf, candle_df, factor_col_name_shard, candle_idx)
                    )

                for future in tqdm(futures, desc="🧮 计算因子", total=len(futures), mininterval=2, file=sys.stdout):
                    idx, period_df = future.result()
                    # factor_col_info.update(agg_dict)  # 更新因子列的周期转换规则
                    all_factor_df_list[idx] = period_df
        else:
            for candle_idx, candle_df in tqdm(
                enumerate(candle_df_dict.values()),
                desc="🧮 计算因子",
                total=len(candle_df_dict.keys()),
                mininterval=2,
                file=sys.stdout,
            ):
                try:
                    idx, period_df = process_by_stock(conf, candle_df, factor_col_name_shard, candle_idx)
                except Exception as e:
                    logger.debug(traceback.format_exc())
                    logger.error(f"因子计算失败，{e}")
                    logger.error(f'股票代码：{candle_df.iloc[-1]["股票代码"]}')
                    logger.error(f"因子名称：{factor_col_name_shard}")
                    raise e
                # factor_col_info.update(agg_dict)  # 更新因子列的周期转换规则
                all_factor_df_list[idx] = period_df

        # ====================================================================================================
        # 3. 合并因子数据并存储
        # ====================================================================================================
        all_factors_df = pd_concat(all_factor_df_list, ignore_index=True, copy=False)
        logger.debug("📅 因子结果最晚日期：" + str(all_factors_df["交易日期"].max()))

        # 转化一下symbol的类型为category，可以加快因子计算速度，节省内存
        # 并且排序和整理index
        all_factors_df = (
            all_factors_df.assign(
                股票代码=all_factors_df["股票代码"].astype("category"),
                股票名称=all_factors_df["股票名称"].astype("category"),
            )
            .sort_values(by=["交易日期", "股票代码"])
            .reset_index(drop=True)
        )

        logger.debug("💾 存储因子数据...")

        logger.debug(f"- {all_kline_pkl}")
        logger.debug(f'最晚交易日期：{all_factors_df["交易日期"].max()}')

        # 选股需要的k线
        if not all_kline_pkl.exists():
            if conf.use_hour_data:
                all_kline_df = all_factors_df[HOUR_FACTOR_COLS].sort_values(by=["交易日期", "股票代码", "股票名称"])
                all_kline_df.to_parquet(all_kline_pkl)
            else:
                all_kline_df = all_factors_df[FACTOR_COLS].sort_values(by=["交易日期", "股票代码", "股票名称"])
                all_kline_df.to_pickle(all_kline_pkl)

        # 针对每一个因子进行存储
        for factor_col_name in factor_col_name_shard:
            # 因为截面因子没有计算，所以不包含在all_factors_df中
            if factor_col_name in section_col_name_list:
                continue
            if conf.use_hour_data:
                factor_pkl = conf.get_factor_folder() / f"factor_hour_{factor_col_name}.parquet"
                factor_pkl.unlink(missing_ok=True)  # 动态清理掉cache的缓存
                all_factors_df[factor_col_name].to_frame().to_parquet(factor_pkl)
            else:
                factor_pkl = conf.get_runtime_folder() / f"factor_{factor_col_name}.pkl"
                factor_pkl.unlink(missing_ok=True)  # 动态清理掉cache的缓存
                all_factors_df[factor_col_name].to_pickle(factor_pkl)

        gc.collect()

    logger.ok(f"因子计算完成，耗时：{time.time() - s_time:.2f}秒")


def process_factor_df(factor_col_name, runtime_path):
    # 准备所有时序因子数据
    factor_path = get_file_path(runtime_path, f"factor_{factor_col_name}.pkl")
    if not factor_path.exists():
        return factor_col_name, pd.DataFrame()

    return factor_col_name, pd.read_pickle(factor_path)


def load_all_factors(conf: BacktestConfig):
    all_kline_pkl = get_file_path(conf.get_runtime_folder(), "all_factors_kline.pkl")
    factor_df = pd.read_pickle(all_kline_pkl)

    # 准备所有时序因子数据
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(process_factor_df, factor.col_name, conf.get_runtime_folder())
            for factor in conf.factor_list_for_section
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(conf.factor_list_for_section),
            desc="📖 读取时序因子数据",
            mininterval=2,
            file=sys.stdout,
        ):
            factor_col_name, kline_with_factor_df = future.result()
            if not kline_with_factor_df.empty:
                factor_df[factor_col_name] = kline_with_factor_df

    return factor_df


def calc_cross_sections(conf: BacktestConfig):
    """
    截面因子计算，
    :param conf:    回测策略配置
    :return:
    """
    # 如果没有配置截面因子，那么直接跳过后续
    if not conf.section_factor_list:
        logger.info("未检查到截面因子配置，跳过计算截面因子步骤。")
        return
    logger.info("截面因子计算...")
    s_time = time.time()
    # 加载面板数据(包含了截面因子需要的时序因子)
    all_factor_df = load_all_factors(conf)

    before_len = len(all_factor_df)
    # 遍历截面因子，调用截面因子计算方法
    for section_factor in tqdm(
        conf.section_factor_list,
        desc="🧮 计算截面因子",
        total=len(conf.section_factor_list),
        mininterval=2,
        file=sys.stdout,
    ):
        factor_name = section_factor.name
        factor_file = FactorHub.get_by_name(factor_name)  # 获取因子信息
        col_name = section_factor.col_name
        param = section_factor.param
        # windows的文件名有最大长度限制，大概是256
        if len(col_name) > 200:
            err_col_name = col_name[:30] + "..."
            raise ValueError(
                f"截面因子{err_col_name}超出windows文件名的最大长度限制，请配置alias_name参数。示例："
                + """
                'cross_sections': [
                    {
                        'name': '高斯秩回归',
                        'is_sort_asc': False,
                        'factor_list': [
                            ('归母净利润', False, '单季', 1),
                            ('市值', True, None, 1),
                        ],
                        'params': ['normal', 'gauss_change'],
                        'args': 1,
                        'minutes': '0945',
                        'alias_name': '归母净利润+市值',  # 此处配置alias_name
                    }
                ]"""
            )
        # 因子计算，factor_df是包含因子计算结果的DataFrame，必须是按照"交易日期"从小到大排序
        factor_df = factor_file.add_factor(
            all_factor_df.copy(),
            param,
            # 暂时没传财务数据
            # fin_data=fin_data,
            col_name=col_name,
            minutes=section_factor.minutes,  # 2025-03-20添加分钟数据的支持
            section_factor=section_factor,
        )
        # 检查因子计算是否出错
        if before_len != len(factor_df[col_name]):
            logger.error(f"{factor_name}因子({param}，{col_name})导致数据长度发生变化，请检查！")
            raise Exception("截面因子计算出错，请避免在cal_factors中修改数据行数")

        factor_pkl = conf.get_runtime_folder() / f"factor_{col_name}.pkl"
        factor_pkl.unlink(missing_ok=True)  # 动态清理掉cache的缓存
        factor_df[col_name].to_pickle(factor_pkl)
        del factor_df
    del all_factor_df
    gc.collect()
    logger.ok(f"截面因子计算完成，耗时：{time.time() - s_time:.2f}秒")


# endregion


# region step3_选股.py
# ================================================================
# step3_选股.py
# ================================================================
def select_stocks(confs: BacktestConfig | List[BacktestConfig], boost=True):
    if isinstance(confs, BacktestConfig):
        # 如果是单例，就直接返回原来的结果
        return select_stock_by_conf(confs, boost=boost)

    # 否则就直接并行回测
    is_silent = True  # 减少输出
    if boost:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(select_stock_by_conf, conf, boost, is_silent) for conf in confs]
            for future in tqdm(as_completed(futures), total=len(confs), desc="选股", mininterval=2, file=sys.stdout):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(e)
                    sys.exit(1)
    else:
        for conf in tqdm(confs, total=len(confs), desc="选股", mininterval=2, file=sys.stdout):
            select_stock_by_conf(conf, boost, is_silent)
    import logging

    logger.setLevel(logging.DEBUG)  # 恢复日志模式


def select_stock_by_conf(conf: BacktestConfig, boost=True, silent=False):
    """
    选股流程：
    1. 初始化策略配置
    2. 加载并清洗选股数据
    3. 计算选股因子并进行筛选
    4. 缓存选股结果

    参数:
    conf (BacktestConfig): 回测配置
    返回:
    DataFrame: 选股结果
    """
    if silent:
        import logging

        logger.setLevel(logging.WARNING)  # 可以减少中间输出的log

    result_folder = conf.get_result_folder()  # 选股结果文件夹
    period_offset = conf.load_period_offset()  # 交易日期偏移
    factor_df_path = conf.get_runtime_folder() / "all_factors_kline.pkl"  # 在进程中，这个位置会无法区分实盘和回测

    logger.debug(f"🔍 因子文件：{factor_df_path}")

    if boost and len(conf.strategy_list) > 1:
        # 多进程模式
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(select_stocks_by_strategy, stg, factor_df_path, result_folder, period_offset)
                for stg in conf.strategy_list
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(e)
                    sys.exit(1)
    else:
        for strategy in conf.strategy_list:
            select_stocks_by_strategy(strategy, factor_df_path, result_folder, period_offset)


def select_stocks_by_strategy(
    strategy: StrategyConfig, factor_df_path: Path, result_folder: Path, period_offset: pd.DataFrame
):
    # ====================================================================================================
    # 1. 初始化策略配置
    # ====================================================================================================
    s_time = time.time()
    logger.debug(f"🎯 {strategy.name} 选股启动...")

    # ====================================================================================================
    # 2. 加载并清洗选股数据
    # - 2.1 准备选股数据
    # - 2.2 根据持仓周期裁切
    # - 2.3 过滤掉每一个周期中，没有交易的股票 & 针对选股日期进行筛选要选股的数据，
    # - 2.4 检查因子列都不为空的最大交易日期，方便检查一些策略可能在某些极端情况下，因子值丢失，方便debug
    # - 2.5 删除相同交易日期下，某个因子都为空的行
    # - 2.6 最后整理一下
    # ====================================================================================================
    # 2.1 准备选股数据
    runtime_folder = factor_df_path.parent
    factor_df = pd.read_pickle(factor_df_path)
    for factor_col_name in strategy.factor_columns:
        factor_df[factor_col_name] = pd.read_pickle(get_file_path(runtime_folder, f"factor_{factor_col_name}.pkl"))
    logger.debug(f'📦 [{strategy.name}] 选股数据加载完成，最晚日期：{factor_df["交易日期"].max()}')
    # 过滤板块
    if strategy.excluded_boards:
        temp_excluded_boards = "，".join(
            [
                f'`{x.replace("kcb", "科创板").replace("cyb", "创业板").replace("bj", "北交所")}`'
                for x in strategy.excluded_boards
            ]
        )
        logger.debug(f"🗑️ [{strategy.name}] 需要排除{temp_excluded_boards}")
    excluded_boards = [x.replace("kcb", "sh68").replace("cyb", "sz30") for x in strategy.excluded_boards]
    factor_df = factor_df[~factor_df["股票代码"].str.startswith(tuple(excluded_boards))]
    # 根据持仓周期裁切

    # 2.2 根据持仓周期裁切
    select_dates_dict = {}
    select_dates = []
    for hold_period_name in strategy.hold_period_name_list:
        select_dates_dict[hold_period_name] = period_offset.groupby(hold_period_name)["交易日期"].last().to_list()
        select_dates += select_dates_dict[hold_period_name]

    select_dates = list(set(select_dates))

    # 2.3 过滤掉每一个周期中，没有交易的股票 & 针对选股日期进行筛选要选股的数据，
    # ** 2025-03-30 为了保证数据的连续性，日期的筛选需要防在后面
    # ** 2025-06-21 取消drop NaN的因子数值，保留所有因子数据，方便后续因子计算
    factor_df = (
        factor_df[
            (factor_df["是否交易"] == 1)
            & (factor_df["交易日期"].between(min(select_dates), max(select_dates), inclusive="both"))
        ]
        # .dropna(subset=strategy.factor_columns)
        .copy()
    )
    factor_df.dropna(subset=["股票代码"], inplace=True)
    logger.debug(f"🧩 [{strategy.name}] 选股面板数据拼接完成，最晚日期：{factor_df['交易日期'].max()}")

    # 2.4 检查因子列都不为空的最大交易日期，方便检查一些策略可能在某些极端情况下，因子值丢失，方便debug
    max_date_with_factors = factor_df.dropna(subset=strategy.factor_columns)["交易日期"].max()
    if max_date_with_factors < factor_df["交易日期"].max():
        logger.warning(
            f'[{strategy.name}] 因子列都不为空的最大交易日期：{max_date_with_factors}，小于选股日期：{factor_df["交易日期"].max()}，可能存在因子数据丢失'
        )

    # 2.5 删除相同交易日期下，某个因子都为空的行
    # 优化版本：只针对最近5个交易日期进行空值检查，节省计算资源
    recent_5_dates = sorted(factor_df["交易日期"].unique())[-5:]  # 取最近5个交易日期

    # 只对最近5个日期进行空值检查
    recent_factor_df = factor_df[factor_df["交易日期"].isin(recent_5_dates)]
    # 得到某个交易日，所有股票都为na的因子列（输入5个因子列，返回5个因子列）
    date_factor_all_na = recent_factor_df.groupby("交易日期")[strategy.factor_columns].apply(lambda x: x.isna().all())
    # 找出某一个因子为空的日期
    empty_dates = date_factor_all_na.loc[date_factor_all_na.any(axis=1)].index.tolist()

    # 删除这些日期对应的所有行
    if empty_dates:
        factor_df = factor_df[~factor_df["交易日期"].isin(empty_dates)]
        logger.error(
            f"[{strategy.name}] 删除了最近5个日期中所有因子都为空的行，涉及日期：{empty_dates}，选股结果会存在问题，无法选出最新周期的股票"
        )

    # 2.6 最后整理一下
    factor_df.sort_values(by=["交易日期", "股票代码"], inplace=True)
    factor_df.reset_index(drop=True, inplace=True)

    # ====================================================================================================
    # 3. 因子计算和筛选流程
    # 3.1 前置筛选
    # 3.2 计算选股因子
    # 3.3 基于选股因子进行选股
    # ====================================================================================================

    # 3.1 前置筛选
    s = time.time()
    factor_df = strategy.filter_before_select(factor_df)
    factor_df = factor_df[list(dict.fromkeys(KLINE_COLS + strategy.factor_columns + FILTER_AFTER_COLS))]  # 裁切一下数据
    logger.debug(
        f"🚦 [{strategy.name}] 前置筛选耗时：{time.time() - s:.2f}s。" f'数据最晚日期：{factor_df["交易日期"].max()}'
    )

    # 3.2 计算选股因子
    s = time.time()
    factor_df = strategy.calc_select_factor(factor_df)
    logger.debug(
        f"🧮 [{strategy.name}] 选股复合因子计算耗时：{time.time() - s:.2f}s。"
        f'数据最晚日期：{factor_df["交易日期"].max()}'
    )

    # 3.3 计算择时信号（和夏普于2025-03-23 14:00确认，目前是在过滤后做的择时）
    s = time.time()
    if strategy.timing:
        signals = strategy.calc_signal(factor_df)
    else:
        signals = pd.DataFrame({"择时信号": 1}, index=sorted(factor_df["交易日期"].unique()))

    signals.index.name = "选股日期"  # 方便大家对答案理解，但是系统里会在最后一步重命名
    save_csv_safely(signals, result_folder / f"择时信号{strategy.name}.csv", index=True, with_pickle=True)
    logger.debug(f"🕑 [{strategy.name}] 择时：{time.time() - s:.2f}s")

    # 3.4 计算策略临时调仓信号
    s = time.time()
    if strategy.override:
        # 这个信号会在后续ratio计算时，被合并到ratio中
        override_signal_dict = strategy.calc_override_signal(factor_df)
        for period, override_signal in override_signal_dict.items():
            override_signal = override_signal.reset_index(names="选股日期")
            override_signal.to_pickle(result_folder / f"策略临时调仓信号{strategy.name}_{period}.pkl")
    else:
        override_signal_dict = {}
    logger.debug(f"🍃 [{strategy.name}] 策略临时调仓信号：{time.time() - s:.2f}s")

    # 3.5 进行选股
    s = time.time()
    # 先按照select_dates进行筛选
    factor_df = factor_df[factor_df["交易日期"].isin(select_dates)]
    # 开始筛选
    result_df = select_by_factor(factor_df, strategy.select_num, strategy.factor_name)
    logger.debug(
        f"💡 [{strategy.name}] 选股耗时：{time.time() - s:.2f}s。" f'数据最晚日期：{result_df["交易日期"].max()}'
    )

    # 3.6 选股后置过滤
    s = time.time()
    result_df = strategy.filter_after_select(result_df)
    logger.debug(
        f"🚦 [{strategy.name}] 后置筛选耗时：{time.time() - s:.2f}s。" f'数据最晚日期：{result_df["交易日期"].max()}'
    )

    result_path = result_folder / f"选股结果{strategy.name}.pkl"
    # 若无选股结果则直接返回
    if result_df.empty:
        pd.DataFrame(columns=[RES_COLS, strategy.factor_name]).to_pickle(result_path)
        return

    # 3.7 合并择时信号
    result_df = pd.merge(result_df, signals, left_on="交易日期", right_index=True, how="left")
    result_df = result_df.assign(择时信号=result_df["择时信号"].fillna(np.float64(1.0)))[
        [*KLINE_COLS, "目标资金占比", "择时信号", "选股因子排名", strategy.factor_name]
    ]

    # ====================================================================================================
    # 4. 处理多offset，并计算临时调仓信号（如有）
    # ====================================================================================================
    logger.debug(f"🧬 [{strategy.name}] 开始合成分批进场信号...")
    s = time.time()
    select_result_df = calc_scalein_pos(strategy, result_df, override_signal_dict, select_dates_dict)
    logger.debug(f"🧬 [{strategy.name}] 分批进场信号计算完成，耗时：{time.time() - s:.2f}s")

    # ====================================================================================================
    # 5. 缓存选股结果
    # ====================================================================================================
    select_result_df = select_result_df.assign(
        策略=strategy.name, 策略权重=np.float64(strategy.cap_weight), 换仓时间=strategy.rebalance_time
    ).rename(columns={"交易日期": "选股日期"})

    select_result_df = select_result_df.assign(
        策略=select_result_df["策略"].astype("category"),
        换仓时间=select_result_df["换仓时间"].astype("category"),
        持仓周期=select_result_df["持仓周期"].astype("category"),
        目标资金占比_原始=select_result_df["目标资金占比"],
        目标资金占比=(
            select_result_df["目标资金占比"]
            * select_result_df["策略权重"]
            * select_result_df["分批进场仓位"]  # offset间资金分配
        ).astype(
            np.float64
        ),  # 目标资金占比转为float64
    )

    # 缓存到本地文件
    select_result_df = select_result_df[RES_COLS]
    select_result_df.to_pickle(result_path)

    logger.debug(f"🏁 [{strategy.name}] 完成目标权重计算，选股整体耗时: {(time.time() - s_time):.2f}s")

    return select_result_df


def select_by_factor(period_df, select_num: float | int, factor_name):
    """
    基于因子选择目标股票并计算资金权重。

    参数:
    period_df (DataFrame): 筛选后的数据
    select_num (float | int): 选股数量或比例
    factor_name (str): 选股因子名称

    返回:
    DataFrame: 带目标资金占比的选股结果
    """
    period_df = calc_select_factor_rank(period_df, factor_column=factor_name, ascending=True)

    # 基于排名筛选股票
    if int(select_num) == 0:  # 选股数量是百分比
        period_df = period_df[period_df["选股因子排名"] <= period_df["总股数"] * select_num].copy()
    else:  # 选股数量是固定的数字
        period_df = period_df[period_df["选股因子排名"] <= select_num].copy()

    # 根据选股数量分配目标资金
    period_df["目标资金占比"] = 1 / period_df.groupby("交易日期")["股票代码"].transform("size")

    period_df.sort_values(by="交易日期", inplace=True)
    period_df.reset_index(drop=True, inplace=True)

    # 清理无关列
    period_df.drop(columns=["总股数"], inplace=True)

    return period_df


def calc_select_factor_rank(df, factor_column="因子", ascending=True):
    """
    计算因子排名。

    参数:
    df (DataFrame): 原始数据
    factor_column (str): 因子列名
    ascending (bool): 排序顺序，True为升序

    返回:
    DataFrame: 包含排名的原数据
    """
    # 计算因子的分组排名
    df["选股因子排名"] = df.groupby("交易日期")[factor_column].rank(method="min", ascending=ascending)
    # 根据时间和因子排名排序
    df.sort_values(by=["交易日期", "选股因子排名"], inplace=True)
    # 重新计算一下总股数
    df["总股数"] = df.groupby("交易日期")["股票代码"].transform("size")
    return df


def concat_select_results(conf: BacktestConfig) -> pd.DataFrame:
    """
    聚合策略选股结果，形成综合选股结果
    :param conf:
    :return:
    """
    # 如果是纯多头现货模式，那么就不转换合约数据，只下现货单
    all_select_df_list = []  # 存储每一个策略的选股结果
    result_folder = conf.get_result_folder()
    recent_select_df_list = []

    for strategy in conf.strategy_list:
        stg_select_result = result_folder / f"选股结果{strategy.name}.pkl"
        # 如果文件不存在，就跳过
        if not stg_select_result.exists():
            continue
        # 读入单策略选股结果
        stg_select = pd.read_pickle(stg_select_result)
        if not stg_select.empty:
            # 添加到最终选股结果
            all_select_df_list.append(stg_select)
            # 裁切最新选股结果
            logger.debug(f'🔍 计算`{strategy.name}`最新选股结果, 数据最晚选股日：{stg_select["选股日期"].max()}')
            recent_select_df_list.append(stg_select[stg_select["选股日期"] == stg_select["选股日期"].max()])

    # 合并最终选股结果
    if all_select_df_list:
        # 聚合选股结果
        all_select_df = pd_concat(all_select_df_list, ignore_index=True, copy=False)
    else:
        all_select_df = pd.DataFrame(columns=RES_COLS)
    # 合并最新选股结果
    if recent_select_df_list:
        recent_select_df = pd_concat(recent_select_df_list, ignore_index=True, copy=False)
    else:
        recent_select_df = pd.DataFrame(columns=RES_COLS)

    all_select_df = all_select_df.sort_values(by=["选股日期", "持仓周期", "选股因子排名"])[RES_COLS].reset_index(
        drop=True
    )
    # 同时保存pkl和csv，csv给你核对结果用😃
    save_csv_safely(all_select_df, conf.select_results_path.with_suffix(".csv"), with_pickle=True, index=False)
    # 再附赠一份最新选股结果
    recent_select_df = recent_select_df.sort_values(by=["选股日期", "持仓周期", "选股因子排名"])[RES_COLS]
    save_csv_safely(recent_select_df, result_folder / "最新选股结果.csv", index=False)

    return all_select_df


# ================================================================
# v1.7 新功能，分批进场相关函数，offset间仓位分配
# ================================================================
def calc_scalein_pos(strategy, result_df, override_pos_dict, select_dates_dict, aggressive=True) -> pd.DataFrame:
    """
    计算分批进场仓位。

    参数:
    :param strategy: 策略配置
    :param result_df: 原始数据
    :param override_pos_dict: 策略临时调仓信号
    :param select_dates_dict: 选股日期
    :param aggressive: 是否激进模式
    :return: 包含分批进场仓位的原始数据
    """
    # 存储最终的每个offset的选股结果 {offset: result_df}
    result_by_offset_dict = {}
    # 记录每个offset是否开仓信息的dict
    offset_signal_dict = {}
    for offset in strategy.hold_period_name_list:
        # 1. 裁切当前offset的选股结果
        result_by_offset = result_df[result_df["交易日期"].isin(select_dates_dict[offset])].copy()

        # 2. 插入临时调仓信号
        # 把临时调仓信号集成到原有选股结果中，包含新的目标权重
        result_by_offset = apply_override_pos(
            result_by_offset,
            override_pos_dict.get(offset, pd.DataFrame(columns=["目标仓位"], index=pd.Index([], name="交易日期"))),
        )
        result_by_offset["持仓周期"] = offset
        result_by_offset_dict[offset] = result_by_offset

        # 3. 结合择时/临时调整的信号，记录调仓时间节点
        offset_signal = result_by_offset.groupby("交易日期").agg({"择时信号": "last", "临时调仓信号": "last"})
        # 有同学写的因子不规范，如果这边报错请检查自己信号库计算的相关逻辑，
        # 空值逻辑自行处理，我不知道你空的时候想要0还是1
        offset_signal = offset_signal.assign(
            择时信号=offset_signal["择时信号"].astype(np.float64),
            临时调仓信号=offset_signal["临时调仓信号"].astype(np.float64),
        )

        # 整合择时信号和临时调仓信号
        offset_signal[offset] = offset_signal["临时调仓信号"] * offset_signal["择时信号"].round(9)
        # 重命名列名
        offset_signal.rename(
            columns={"择时信号": f"{offset}开仓信号", "临时调仓信号": f"{offset}离场信号"}, inplace=True
        )
        offset_signal_dict[offset] = offset_signal

    # 横向合并offset_entry_list，index不对齐时插入新行（相当于outer join）
    offset_signal_df = pd_concat(offset_signal_dict.values(), axis=1, join="outer")
    # 缓存开仓时间点
    offset_signal_df[[f"{offset_name}调仓点" for offset_name in strategy.hold_period_name_list]] = (
        offset_signal_df[strategy.hold_period_name_list].gt(0).astype("int8")
    )

    offset_signal_df["调仓offset数量"] = (
        offset_signal_df[strategy.hold_period_name_list].gt(0).sum(axis=1).astype("int8")
    )

    # 填充持仓的状态
    offset_signal_df = offset_signal_df.ffill().fillna(np.float64(0.0))
    # 完成ffill填充后，标记当前active的offset数量
    offset_signal_df["offset总数"] = (
        # 激进模式下，offset总数只要大于0，就认为有持仓
        offset_signal_df[strategy.hold_period_name_list].gt(0).sum(axis=1).astype("int8")
        if aggressive
        # 保守模式下，只有所有offset满仓信号，才认为有持仓，否则认为无持仓
        else offset_signal_df[strategy.hold_period_name_list].eq(1).sum(axis=1).astype("int8")
    )

    # 填充目标总仓位
    offset_signal_df["目标总仓位"] = offset_signal_df["offset总数"].apply(
        lambda x: strategy.scalein_targets[int(x - 1)] if x > 0 else 0
    )

    offset_scalein_pos = fill_scalein_targets(strategy, offset_signal_df)

    # 每个offset的选股结果，新增“分批进场仓位”列
    for offset in strategy.hold_period_name_list:
        offset_scalein_df = pd.DataFrame(offset_scalein_pos[offset]).fillna(np.float64(0.0))
        # 在追加模式中，可能会存在重复的数值
        offset_scalein_df.drop_duplicates(subset=["交易日期"], keep="last", inplace=True)

        if offset_scalein_df.empty:
            result_with_scalein = result_by_offset_dict[offset].assign(分批进场仓位=np.float64(0.0))
        else:
            result_with_scalein = pd.merge(
                result_by_offset_dict[offset], offset_scalein_df, left_on="交易日期", right_on="交易日期", how="left"
            )
        result_with_scalein["调仓类型"] = np.where(
            result_with_scalein["交易日期"].isin(select_dates_dict[offset]), "计划", "临时"
        )
        result_by_offset_dict[offset] = result_with_scalein

    # 合成最终选股结果
    return pd_concat(result_by_offset_dict.values(), ignore_index=True, copy=False)


def apply_override_pos(result_df, override_pos, use_cum=True):
    """
    应用策略临时调仓信号。

    参数:
    :param result_df: 原始数据
    :param override_pos: 策略临时调仓信号
    :param use_cum: 是否使用累计减仓信号
    :return: 包含临时调仓信号的原始数据
    """
    # 默认临时调仓信号是1（比如0表示临时清仓）
    result_df["临时调仓信号"] = np.float64(1.0)

    override_pos = override_pos.dropna(subset=["目标仓位"])
    if override_pos.empty:
        return result_df

    # 逐行插入临时调整的目标权重
    for trade_date, row in override_pos.iterrows():
        if trade_date in result_df["交易日期"].values:
            # 如果交易日期已经存在，则跳过
            continue
        override_pos_val = row["目标仓位"]
        # 找到result_df中交易日期不大于trade_date的最大值的所有行
        trade_date0 = result_df[result_df["交易日期"] <= trade_date]["交易日期"].max()
        # 找到需要复制的选股结果的所有行
        if pd.notna(trade_date0):
            rows_to_insert = result_df[result_df["交易日期"] == trade_date0].copy()
        else:
            rows_to_insert = pd.DataFrame(columns=result_df.columns)

        rows_to_insert["临时调仓信号"] = (
            (rows_to_insert["临时调仓信号"] * override_pos_val) if use_cum else override_pos_val
        )
        # 默认择时是开，可能会和临时调仓信号冲突，但是无所谓，临时调仓信号优先级更高
        rows_to_insert["交易日期"] = trade_date

        # 将matched_rows追加到result_df中，
        # 真实最后使用的目标资金占比是：rows_to_insert["临时调仓信号"] * rows_to_insert["目标资金占比"] * rows_to_insert["择时信号"]
        # 但是后续的“分批进场仓位”会结合考虑“择时信号”和“临时调仓信号”，因此整合的时候不用再乘以这俩兄弟了
        result_df = pd_concat([result_df, rows_to_insert], ignore_index=True)

    return result_df.sort_values(by="交易日期", ignore_index=True)


def fill_scalein_targets(strategy, offset_signal_df):
    """
    填充分批进场仓位。
    重要注意事项，我们记录就当前仓位的状态的时候，是记录“择时信号”为1的状态，来表示充分的占用，而只启用部分资金。
    落实到实际的仓位填充的时候，会需要在比例上动态乘以开仓的杠杆。
    具体影响：
    1. 调仓真实仓位 = 当前仓位 * 开仓信号（杠杆）
    2. 开仓真实仓位 = 当前仓位 * 开仓信号（杠杆）
    3. 持仓真实仓位 = 当前仓位 * 开仓信号（杠杆）

    参数:
    :param strategy: 策略配置
    :param offset_signal_df: 分批进场仓位信号
    :return: 包含分批进场仓位的原始数据
    """
    # ==== 初始化 ====
    current_pos = dict()
    offset_scalein_pos = dict()
    offset_list = strategy.hold_period_name_list
    for offset in offset_list:
        # 缓存当前仓位
        current_pos[offset] = np.float64(0.0)
        # 记录逐个offset对应的时间点和分批仓位
        offset_scalein_pos[offset] = []
        # 追加空列，避免索引问题
        offset_signal_df[f"{offset}仓位"] = np.float64(0.0)

    # 当且仅当“调仓”时候才会使用，那时候prev row肯定不为空了
    prev_row = None

    # ==== 循环填充 ====
    for trade_date, row in offset_signal_df.iterrows():
        # 先卖出，找到非零持仓并且今日信号为0
        sell_offsets = [offset for offset in offset_list if (current_pos[offset] > 1e-9) and (row[f"{offset}"] < 1e-9)]
        for offset in sell_offsets:
            current_pos[offset] = np.float64(0.0)

        # 1. 需要新开仓位的offset
        # - 调仓点为1，表示需要调仓
        # - 开仓信号大于0，表示需要开仓
        new_offsets = [
            offset
            for offset in offset_list
            if (row[f"{offset}调仓点"] == 1) and (row[f"{offset}开仓信号"] > 1e-9) and (row[f"{offset}离场信号"] == 1)
            # 这里离场也会触发，所以需要判断进行不是离场
        ]
        # 2. 需要调整仓位的offset
        # - 调仓点为1，表示需要调仓
        # - 离场信号大于0，表示需要调整仓位
        reb_offsets = [
            offset for offset in offset_list if (row[f"{offset}调仓点"] == 1) and (1e-9 < row[f"{offset}离场信号"] < 1)
        ]
        # 其余持仓offset
        hold_offsets = [
            offset
            for offset in offset_list
            if (current_pos[offset] > 1e-9) and (offset not in reb_offsets) and (offset not in new_offsets)
        ]

        # 处理调仓
        for offset in sorted(reb_offsets):
            # 获取相比于上次信号的调整比值
            ratio = row[f"{offset}"] / prev_row[f"{offset}"] if prev_row[f"{offset}"] > 1e-9 else row[f"{offset}"]
            # 等比缩小/放大
            current_pos[offset] = current_pos[offset] * ratio
            # 实际仓位 = 当前仓位 * 开仓信号，我们会有开仓信号部分开的情况
            real_pos = current_pos[offset] * row[f"{offset}开仓信号"]
            offset_signal_df.loc[trade_date, f"{offset}仓位"] = real_pos
            offset_scalein_pos[offset].append({"交易日期": trade_date, "分批进场仓位": real_pos})

        # 不需要换仓的既有仓位，只需要添加非“新开”的情况
        prev_pos = sum([pos for offset, pos in current_pos.items() if offset not in new_offsets])
        # 如果此时2个仓位同时开仓，采用均分策略
        new_pos = (row["目标总仓位"] - prev_pos) / len(new_offsets) if len(new_offsets) > 0 else np.float64(0.0)
        new_pos = np.float64(round(new_pos, 9))

        # 处理开仓
        for offset in sorted(new_offsets):
            if new_pos < 0:
                # 如果遇到当前仓位，比目标仓位要重的时候，新开仓位作废
                continue
            current_pos[offset] = new_pos
            # 实际仓位 = 当前仓位 * 开仓信号，我们会有开仓信号部分开的情况
            real_pos = current_pos[offset] * row[f"{offset}开仓信号"]
            offset_signal_df.loc[trade_date, f"{offset}仓位"] = real_pos
            offset_scalein_pos[offset].append({"交易日期": trade_date, "分批进场仓位": real_pos})

        # 处理持仓
        for offset in sorted(hold_offsets):
            if row[f"{offset}"] < 1e-9:
                # 防御式处理，逻辑上是不会走入到这里
                current_pos[offset] = np.float64(0.0)
            # 如果值没有变化，直接设置仓位为当前仓位
            real_pos = current_pos[offset] * row[f"{offset}开仓信号"]
            offset_signal_df.loc[trade_date, f"{offset}仓位"] = real_pos
        prev_row = row

    offset_signal_df["实际仓位"] = offset_signal_df[[f"{offset}仓位" for offset in offset_list]].sum(axis=1).round(8)
    offset_signal_df = offset_signal_df.rename_axis("选股日期")

    save_csv_safely(offset_signal_df, strategy.result_folder / f"分批进场仓位{strategy.name}.csv", index=True)

    return offset_scalein_pos


# endregion


# region step3点5_个股择时.py
# ================================================================
# step3点5_个股择时.py
# ================================================================

def _merge_parquet_append_mode(
    parquet_path: Path,
    new_df: Union[pd.DataFrame, pl.DataFrame],
    sort_cols: List[str],
    dedup_cols: Optional[List[str]] = None,
) -> None:
    """统一 parquet 增量合并：append -> dedup -> sort -> write。

    使用 Polars 进行 concat/dedup/sort/write，性能比 pandas 提升约 2-5x。
    接受 pd.DataFrame 或 pl.DataFrame，内部统一转为 Polars 处理。
    如果旧文件存在但损坏（上次写入被中断等），则丢弃旧文件并降级为全量写入，
    同时在日志中记录警告以便排查。
    """
    dedup_cols = dedup_cols or ["交易日期", "股票代码"]
    new_pl = new_df if isinstance(new_df, pl.DataFrame) else pl.from_pandas(new_df)

    if parquet_path.exists():
        try:
            old_pl = pl.read_parquet(parquet_path)
        except Exception:
            logger.warning(f"⚠️ 旧缓存文件损坏，将丢弃并全量重写: {parquet_path.name}")
            old_pl = None
        if old_pl is not None:
            merged = pl.concat([old_pl, new_pl], how="diagonal_relaxed")
        else:
            merged = new_pl
    else:
        merged = new_pl

    merged = merged.unique(subset=dedup_cols, keep="last").sort(sort_cols)
    merged.write_parquet(parquet_path)


def calculate_factors_intraday(conf: BacktestConfig, boost: bool = True, stock_codes: Optional[set] = None):
    """
    计算所有股票的 intraday 因子，支持两种模式：
    - 增量模式 (use_intraday_cache=True): 基于 per-stock 增量缓存计算
    - 兼容模式 (use_intraday_cache=False): 保持原有的全量 parquet 文件缓存逻辑

    参数:
    conf (BacktestConfig): 回测配置
    boost (bool): 是否启用因子计算加速
    stock_codes: 若非 None，则只计算这些股票代码对应的因子（用于个股择时场景）
    """
    use_cache = conf.is_intraday_cache_enabled()

    if use_cache:
        _calculate_factors_intraday(conf, boost, stock_codes=stock_codes)
    else:
        calculate_factors(conf, boost)


def _check_intraday_prerequisites(conf: BacktestConfig):
    """检查 intraday 因子计算的前置条件（数据、配置等）"""
    logger.debug("🛂 配置信息检查...")
    if len(conf.fin_cols) > 0 and not conf.has_fin_data:
        logger.warning(f"策略需要财务因子{conf.fin_cols}，但缺少财务数据路径")
        raise ValueError("请在 config.py 中配置财务数据路径")
    elif len(conf.fin_cols) > 0:
        logger.debug(f"ℹ️ 检测到财务因子：{conf.fin_cols}")
    else:
        logger.debug("ℹ️ 检测到没有财务因子")

    if len(conf.extra_data.keys()) > 0:
        logger.debug(f"🔍 检测到外部数据：{list(conf.extra_data.keys())}")
        for data_name in conf.extra_data.keys():
            is_ok, msg = check_extra_data(data_name)
            if not is_ok:
                logger.error(f"外部数据检测失败：{msg}")
                sys.exit(2)
    else:
        logger.debug("🔍 检测到没有外部数据")

    exist_path = conf.stock_preprocess_data_path.is_dir() and any(conf.stock_preprocess_data_path.iterdir())
    if not exist_path:
        logger.error(
            "请前往官网下载【股票1小时k线数据Pro】：https://www.quantclass.cn/data/stock/stock-1h-trading-data-pro\n"
            f"把【stock-1h-trading-data-pro】放在【数据中心路径】下\n{conf.stock_preprocess_data_path}"
        )
        sys.exit(2)


def _process_by_stock_from_file(
    conf: BacktestConfig,
    file_path: Path,
    factor_col_name_list: List[str],
    stock_ctx: Tuple[int, Optional[pd.Timestamp], int],
):
    """Worker: 从 parquet 文件加载 K 线 → 输入裁切 → 因子计算 → 输出裁切。

    返回 (idx, factor_df, (stock_code, data_max_time, factor_max_time))
    - data_max_time 取自原始 K 线最大日期（非裁切后），用于更新 meta 状态
    - factor_max_time 取自因子计算输出的最大日期（受 end_date 裁切），用于增量 trim
    - factor_df 可能为空（源数据为空或裁切后无新增）
    """
    idx, trim_before, lookback_bars = stock_ctx
    # 使用 Polars 读取 parquet 再转 pandas，Rust 原生解析比 PyArrow 快 2-4x
    candle_df = pl.read_parquet(file_path).to_pandas()
    stock_code = file_path.stem

    if "k线结束时间" in candle_df.columns:
        candle_df = candle_df.rename(columns={"k线结束时间": "交易日期"})

    # 空文件防护：跳过计算，返回空结果
    if len(candle_df) == 0:
        return idx, pd.DataFrame(), (stock_code, None, None)

    data_max_time = str(candle_df["交易日期"].max())

    # trim_before 为空时保持全量路径；否则按 bar lookback 计算输入裁切起点
    if trim_before is not None:
        trim_before_ts = pd.Timestamp(trim_before)
        max_time_ts = pd.Timestamp(data_max_time)
        if trim_before_ts.tzinfo is not None:
            trim_before_ts = trim_before_ts.tz_localize(None)
        if max_time_ts.tzinfo is not None:
            max_time_ts = max_time_ts.tz_localize(None)

        # 兜底短路：已经计算到最新，避免无意义的 lookback 裁切和因子计算
        # 返回 None 作为 max_time，明确告知上游该股票未实际计算，不应写入 meta
        if max_time_ts <= trim_before_ts:
            return idx, pd.DataFrame(), (stock_code, None, None)

        lookback_bars = max(int(lookback_bars or 0), 0)
        trade_date = pd.to_datetime(candle_df["交易日期"], errors="coerce")
        hit_pos = np.flatnonzero(trade_date <= trim_before_ts)
        if len(hit_pos) > 0:
            end_pos = int(hit_pos[-1])
            start_pos = max(0, end_pos - lookback_bars)
            trim_start = candle_df.iloc[start_pos]["交易日期"]
            candle_df = candle_df[candle_df["交易日期"] >= trim_start]

    idx, factor_df = process_by_stock(conf, candle_df, factor_col_name_list, idx, trim_before)

    # factor_max_time: 因子实际输出的最大日期（受 end_date 裁切影响）
    factor_max_time = str(factor_df["交易日期"].max()) if not factor_df.empty else None

    return idx, factor_df, (stock_code, data_max_time, factor_max_time)


def _calculate_factors_intraday(conf: BacktestConfig, boost: bool = True, stock_codes: Optional[set] = None):
    """基于 per-stock 增量的 intraday 因子计算。

    安全设计要点：
    1. 文件使用 v2 后缀命名，与旧版全量缓存隔离
    2. parquet 写入 → meta 更新的顺序保证崩溃恢复幂等
    3. 旧 parquet 损坏时自动降级为全量写入
    4. 空数据/空裁切结果安全跳过

    参数:
    conf: 回测配置
    boost: 是否启用加速模式（默认启用）
    stock_codes: 若非 None，则只计算这些股票（文件名 stem 匹配）
    """
    logger.info("【日内因子】开始增量计算（per-stock）")
    s_time = time.time()

    factor_col_name_list = conf.hour_factor_col_name_list
    # 小时级因子全部是时序因子，不涉及截面因子，无需过滤

    # 前置检查
    _check_intraday_prerequisites(conf)

    # ====================================================================================================
    # 1. 扫描 K 线文件列表
    # ====================================================================================================
    parquet_dir = conf.stock_preprocess_data_path
    file_list = sorted(parquet_dir.glob("*.parquet"))

    # 按选股结果过滤：只计算选中股票的因子
    if stock_codes:
        file_list = [f for f in file_list if f.stem in stock_codes]
        logger.info(f"【日内因子】按选股结果过滤后待计算股票数: {len(file_list)}")

    if not file_list:
        logger.warning(f"【日内因子】未发现可用 K 线文件，跳过计算: {parquet_dir}")
        return

    # ====================================================================================================
    # 2. 制定增量计划（per-stock 粒度）
    # ====================================================================================================
    factor_folder = conf.get_factor_folder()

    # 缓存同步 + 刚性文件检查 + 增量计划
    stock_plan_map, plan_stats = prepare_intraday_plan(conf=conf, file_list=file_list)
    # 每个 stock 最多一个任务，取最保守的 trim_before，parquet 只读一次
    dispatch_items = list(stock_plan_map.values())
    dispatch_count = len(dispatch_items)
    skip_count = len(file_list) - dispatch_count
    logger.info(
        "【日内因子】增量计划概览\n"
        f"- 股票任务: 计算 {dispatch_count} 只，跳过 {skip_count} 只\n"
        "- 因子列级明细:\n"
        f"  - 全量(文件缺失): {plan_stats.get('full_missing_file', 0)}\n"
        f"  - 全量(无状态): {plan_stats.get('full_missing_meta', 0)}\n"
        f"  - 全量(状态异常): {plan_stats.get('full_invalid_meta', 0)}\n"
        f"  - 全量(数据回退): {plan_stats.get('full_data_rollback', 0)}\n"
        f"  - 增量: {plan_stats.get('incremental', 0)}\n"
        f"  - 已跳过(已最新): {plan_stats.get('latest_skip', 0)}\n"
        f"  - 文件状态命中: {plan_stats.get('file_state_hit', 0)}\n"
        f"  - 文件状态回退: {plan_stats.get('file_state_miss', 0)}"
    )

    if not dispatch_items:
        logger.info("【日内因子】本轮无任务，所有股票均已最新")
        return

    # ====================================================================================================
    # 3. 计算因子（每个 stock 一个任务，携带该股需要计算的因子子集）
    # ====================================================================================================
    factor_col_count = len(factor_col_name_list)
    col_skip = plan_stats.get("col_skip", 0)
    col_incr = plan_stats.get("col_incremental", 0)
    col_full = plan_stats.get("col_full", 0)

    logger.debug(
        "【日内因子】任务规模\n"
        f"- 因子列总数: {factor_col_count}\n"
        f"- 不需要更新: {col_skip}\n"
        f"- 增量更新: {col_incr}\n"
        f"- 全量更新: {col_full}\n"
        f"- 需计算股票: {dispatch_count}"
    )

    # v2 文件名：区别于旧版全量缓存，增量模式使用独立文件避免混淆
    all_kline_pkl = factor_folder / "all_hour_factors_kline_v2.parquet"

    logger.debug(f"【日内因子】计算模式: 多进程（worker={n_jobs}）" if boost else "【日内因子】计算模式: 单进程")

    # 收集 worker 返回的 (stock_code, max_time, computed_cols)；边收边拆分 kline / factor 部分
    worker_meta = [None] * dispatch_count
    kline_parts: List[pd.DataFrame] = []
    factor_parts: Dict[str, List[pd.DataFrame]] = {col: [] for col in factor_col_name_list}

    def _split_period_df(
        _period_df: pd.DataFrame,
        _kline_parts: List[pd.DataFrame] = kline_parts,
        _factor_parts: Dict[str, List[pd.DataFrame]] = factor_parts,
    ) -> None:
        """将单只股票的因子结果拆分到 kline_parts 和 factor_parts。
        收集阶段只存 pandas 引用（O(1)），避免逐股票 pl.from_pandas 的转换开销。
        支持部分因子列：只拆分 period_df 中实际存在的因子列。"""
        if _period_df.empty:
            return
        _kline_parts.append(_period_df[HOUR_FACTOR_COLS])
        for col, parts_list in _factor_parts.items():
            if col in _period_df.columns:
                parts_list.append(_period_df[["交易日期", "股票代码", col]])

    if boost:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for candle_idx, plan_ctx in enumerate(dispatch_items):
                file_path, trim_before, lookback_bars, task_factor_cols = plan_ctx
                stock_ctx = (candle_idx, trim_before, lookback_bars)
                futures.append(
                    executor.submit(_process_by_stock_from_file, conf, file_path, task_factor_cols, stock_ctx)
                )
            for future in tqdm(futures, desc="🧮 计算因子", total=len(futures), mininterval=2, file=sys.stdout):
                idx, period_df, meta = future.result()
                _split_period_df(period_df)
                # 空结果（短路/无新数据）时不写入 meta，避免污染增量状态
                if not period_df.empty:
                    # meta = (stock_code, data_max_time, factor_max_time)
                    worker_meta[idx] = (meta[0], meta[1], meta[2], dispatch_items[idx][3])
    else:
        for candle_idx, plan_ctx in tqdm(
            enumerate(dispatch_items), desc="🧮 计算因子", total=dispatch_count, mininterval=2, file=sys.stdout
        ):
            file_path, trim_before, lookback_bars, task_factor_cols = plan_ctx
            stock_ctx = (candle_idx, trim_before, lookback_bars)

            try:
                idx, period_df, meta = _process_by_stock_from_file(conf, file_path, task_factor_cols, stock_ctx)
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.error(
                    "【日内因子】单股票因子计算失败\n"
                    f"- 股票文件: {file_path.name}\n"
                    f"- 因子列: {task_factor_cols}\n"
                    f"- 错误: {e}"
                )
                raise e
            _split_period_df(period_df)
            # 空结果（短路/无新数据）时不写入 meta，避免污染增量状态
            if not period_df.empty:
                # meta = (stock_code, data_max_time, factor_max_time)
                worker_meta[idx] = (meta[0], meta[1], meta[2], task_factor_cols)

    # ====================================================================================================
    # 4. 合并拆分结果，增量写入 parquet（统一 append + dedup + sort）
    # ====================================================================================================
    # 如果所有 worker 均返回空结果（数据无更新），跳过写入，仅更新 meta
    if not kline_parts:
        logger.info("【日内因子】本轮无新增数据，跳过 parquet 写入")
    else:
        logger.debug("【日内因子】开始写入 parquet 缓存")

        # 一次性 pd.concat → 一次 pl.from_pandas，避免逐股票转换的开销
        new_kline_pd = pd.concat(kline_parts, ignore_index=True, copy=False)
        new_kline_pl = pl.from_pandas(new_kline_pd)
        del new_kline_pd

        if len(new_kline_pl) > 0:
            # kline 长表（v2 格式：按 交易日期+股票代码 排序去重，与 factor 统一保证行对齐）
            _merge_parquet_append_mode(
                all_kline_pkl, new_kline_pl, sort_cols=["交易日期", "股票代码"], dedup_cols=["交易日期", "股票代码"]
            )
        del new_kline_pl

        # 逐因子长表：使用线程池并行写入（I/O bound，各因子写入独立文件，互不依赖）
        def _write_single_factor(args):
            factor_col_name, parts_list = args
            factor_path = factor_folder / f"factor_hour_v2_{factor_col_name}.parquet"
            if not parts_list:
                return
            new_factor_pl = pl.from_pandas(pd.concat(parts_list, ignore_index=True, copy=False))
            _merge_parquet_append_mode(
                factor_path, new_factor_pl, sort_cols=["交易日期", "股票代码"], dedup_cols=["交易日期", "股票代码"]
            )

        factor_write_items = list(factor_parts.items())
        if len(factor_write_items) > 1:
            with ThreadPoolExecutor(max_workers=min(len(factor_write_items), 8)) as write_pool:
                write_futures = [write_pool.submit(_write_single_factor, item) for item in factor_write_items]
                for fut in write_futures:
                    fut.result()  # 收集异常
        elif factor_write_items:
            _write_single_factor(factor_write_items[0])

    del kline_parts, factor_parts
    gc.collect()

    # ====================================================================================================
    # 5. 更新 factor_status（含 mtime/size 文件状态缓存，每行自描述计算时的源数据状态）
    # 写入顺序保证：parquet 先于 meta 更新。若中途崩溃，meta 仍为旧值，
    # 下次运行重新计算并 merge（dedup keep="last"），结果幂等。
    # ====================================================================================================
    stock_file_map = {ctx[0].stem: ctx[0] for ctx in dispatch_items}
    save_intraday_factor_meta(conf=conf, worker_meta=worker_meta, stock_file_map=stock_file_map)

    logger.ok(f"【日内因子】增量计算完成，耗时: {time.time() - s_time:.2f}s")


def _load_factor_data_for_stocks(conf: BacktestConfig, factor_columns: List[str], stock_codes: set) -> pd.DataFrame:
    """加载指定股票的小时因子数据（kline + factor columns）。"""
    factor_folder = conf.get_factor_folder()
    use_cache = conf.is_intraday_cache_enabled()

    if use_cache:
        required_v2_paths = build_intraday_v2_cache_paths(factor_folder, factor_columns)
        missing_v2_paths = [p for p in required_v2_paths if not p.exists()]
        if missing_v2_paths:
            sample = ", ".join([p.name for p in missing_v2_paths[:3]])
            raise FileNotFoundError(
                f"增量缓存文件缺失({len(missing_v2_paths)}/{len(required_v2_paths)}): {sample}，"
                "请先执行因子计算或执行 rebuild 后重试"
            )
        factor_df = pd.read_parquet(factor_folder / "all_hour_factors_kline_v2.parquet")
        factor_df = factor_df.loc[factor_df["股票代码"].isin(stock_codes)].reset_index(drop=True)

        for col_name in factor_columns:
            factor_path = factor_folder / f"factor_hour_v2_{col_name}.parquet"
            fac = pd.read_parquet(factor_path, columns=["交易日期", "股票代码", col_name])
            fac = fac.loc[fac["股票代码"].isin(stock_codes)].reset_index(drop=True)
            if len(fac) != len(factor_df):
                kline_keys = set(zip(factor_df["交易日期"], factor_df["股票代码"]))
                fac_keys = set(zip(fac["交易日期"], fac["股票代码"]))
                only_kline = len(kline_keys - fac_keys)
                only_fac = len(fac_keys - kline_keys)
                raise AssertionError(
                    f"因子 {col_name} 行数({len(fac)})与 kline 行数({len(factor_df)})不一致，"
                    f"kline独有{only_kline}行, factor独有{only_fac}行，"
                    f"数据可能未对齐，请执行 rebuild 后重试"
                )
            factor_df[col_name] = fac[col_name].values
            del fac
    else:
        factor_df = pd.read_parquet(factor_folder / "all_hour_factors_kline.parquet")
        factor_df = factor_df.loc[factor_df["股票代码"].isin(stock_codes)]
        for col_name in factor_columns:
            factor_df[col_name] = pd.read_parquet(factor_folder / f"factor_hour_{col_name}.parquet")

    return factor_df


def calc_stock_timing(conf: BacktestConfig, boost=True):
    stock_codes = conf.load_selected_stock_codes(check_stock_timing=True)
    if stock_codes is None:
        return

    # ====================================================================================================
    # 1. 小时因子计算
    # ====================================================================================================
    conf.use_hour_data = True  # 因为个股择时需要小时因子，所以强制开启小时数据计算

    divider(f"{conf.name}@小时因子计算", "-")
    calculate_factors_intraday(conf, boost=boost, stock_codes=stock_codes)

    # ====================================================================================================
    # 2. 计算个股择时原始信号（去重计算，按 unique signal config）
    # ====================================================================================================
    calculate_stock_signal(conf, boost=boost)

    conf.use_hour_data = False


def calculate_stock_signal(conf: BacktestConfig, boost: bool = True):
    """计算个股择时原始信号。

    按 unique (name, params, factor_list) 去重后只算一次。
    每个信号配置单独落盘为 signal_{key}.parquet，策略级结果输出为 CSV。
    """
    s_time = time.time()
    logger.info("个股择时信号计算...")

    select_results = pd.read_pickle(conf.select_results_path)
    if select_results.empty:
        logger.warning("选股结果为空，跳过个股择时")
        return

    # ====================================================================================================
    # 1. 收集去重的信号配置
    # ====================================================================================================
    plan = conf.collect_signal_plan(select_results)
    del select_results

    if not plan:
        logger.warning("没有策略配置个股择时信号，跳过")
        return

    logger.debug(f"📊 去重后信号配置数: {len(plan)} 个")
    for key, (signal, stocks) in plan.signals.items():
        logger.debug(f"  - {key}: {signal.name}(params={signal.params}), 股票数={len(stocks)}")

    # ====================================================================================================
    # 2. 加载因子数据 + 准备时间映射
    # ====================================================================================================
    factor_df = _load_factor_data_for_stocks(conf, list(plan.all_factor_cols), plan.all_stock_codes)
    time_mapping = conf.build_time_mapping(factor_df["交易日期"].max())
    stock_groups = {code: df for code, df in factor_df.groupby("股票代码", observed=True)}
    del factor_df
    gc.collect()

    # ====================================================================================================
    # 3. 逐信号配置计算 + 单独落盘
    # ====================================================================================================
    runtime_folder = conf.get_runtime_folder()
    signal_cache_dict = {}
    for hash_key, (signal, stock_set) in tqdm(
        plan.signals.items(), desc="🧮 计算个股择时信号", total=len(plan), mininterval=2, file=sys.stdout
    ):
        signal_parts = []
        for code in sorted(stock_set):
            if code not in stock_groups:
                continue
            stock_df = stock_groups[code].copy()
            raw_signal = signal.get_stock_signal(stock_df)
            signal_parts.append(
                pd.DataFrame({"交易日期": stock_df["交易日期"].values, "股票代码": code, "signal": raw_signal.values})
            )
        if not signal_parts:
            continue

        sig_df = pd.concat(signal_parts, ignore_index=True)
        sig_df.sort_values(by=["交易日期", "股票代码"], inplace=True)
        sig_df.reset_index(drop=True, inplace=True)
        sig_df["生效时间"] = sig_df["交易日期"].map(time_mapping)
        sig_df.to_parquet(runtime_folder / f"signal_{hash_key}.parquet", index=False)
        signal_cache_dict[hash_key] = sig_df

    del stock_groups
    gc.collect()

    if not signal_cache_dict:
        logger.warning("所有个股择时信号计算结果为空")
        return

    # ====================================================================================================
    # 4. 策略级加权汇总 + 持久化
    # ====================================================================================================
    plan.aggregate_signals(signal_cache_dict, runtime_folder, conf.get_result_folder())

    del signal_cache_dict
    logger.ok(f"个股择时信号计算完成，共 {len(plan)} 个信号配置，耗时：{time.time() - s_time:.2f}秒")


# endregion


# region step4_实盘模拟.py
# ================================================================
# step4_实盘模拟.py
# ================================================================
def agg_ratios_by_period(conf: BacktestConfig, select_results: pd.DataFrame):
    s_time = time.time()

    logger.debug("🔀 持仓周期权重聚合...")
    symbols = sorted(select_results["股票代码"].unique())
    period_ratio_df = {}
    for (strategy, period, reb_time), grp_df in select_results.groupby(["策略", "持仓周期", "换仓时间"], observed=True):
        # 按策略的话，就不需要pivot-table了，直接pivot
        pivot_table_df = grp_df.pivot(index="选股日期", columns="股票代码", values="目标资金占比")
        period_ratio_df[(strategy, period, reb_time)] = pivot_table_df

    logger.debug(f"👌 权重聚合完成，耗时：{time.time() - s_time:.3f}秒")

    # 防御性编程
    if len(period_ratio_df) == 0:
        logger.critical("权重聚合结果为空，请检查选股结果")
        logger.debug("⏏️ 退出试盘模拟，因为选股结果为空")
        sys.exit()

    # ====================================================================================================
    # 2. 对数据进行处理
    # ====================================================================================================
    min_ratio_dt = min(ratio_df.index.min() for ratio_df in period_ratio_df.values()).date()
    min_ratio_date_str = min_ratio_dt.strftime("%Y-%m-%d")

    max_ratio_dt = max(ratio_df.index.max() for ratio_df in period_ratio_df.values()).date()
    max_ratio_date_str = max_ratio_dt.strftime("%Y-%m-%d")

    # 确定回测区间
    conf.start_date = max(conf.start_date, min_ratio_date_str)
    conf.end_date = min(conf.end_date or max_ratio_date_str, max_ratio_date_str)
    logger.debug(f"🗓️ 回测区间:{conf.start_date}~{conf.end_date}")

    period_offset = conf.load_period_offset()

    # 对于交易日可能为空的周期进行重新填充
    for (strategy, period, reb_time), df_stock_ratio in period_ratio_df.items():
        rebalance_dates = period_offset.groupby(period)["交易日期"].last()
        # 对于交易日可能为空的周期进行重新填充，不存在的 symbol 填充 ratio 为 0
        period_ratio_df[(strategy, period, reb_time)] = df_stock_ratio.reindex(
            index=rebalance_dates, columns=symbols, fill_value=0
        ).sort_index()

    pd.to_pickle(period_ratio_df, conf.get_result_folder() / "period_ratio_df.pkl")
    backtest_days = (max_ratio_dt - min_ratio_dt).days
    logger.info(f"需要回溯 {backtest_days:,} 天...")
    return period_ratio_df


# endregion
