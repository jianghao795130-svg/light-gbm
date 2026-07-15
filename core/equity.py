

import time
from typing import Dict, Tuple

import numba as nb
import numpy as np
import pandas as pd
from numba.typed import List

import config
from core.evaluate import strategy_evaluate
from core.figure import save_performance
from core.perf import show_performance_plot
from core.model.backtest_config import BacktestConfig
from core.model.timing_signal import EquityTiming
from core.model.type_def import SimuParams, StockMarketData, get_symbol_type, AdjustRatios
from core.model.type_def import price_array
from core.rebalance import RebAlways
from core.simulator import Simulator
from core.utils.log_kit import logger

pd.set_option("display.max_rows", 1000)
pd.set_option("expand_frame_repr", False)  # 当列太多时不换行


def parse_rebalance_time(reb_time) -> tuple[int, int]:
    match reb_time:
        case "open":
            return 0, 0
        case "close":
            return -1, -1
        case "close-open":
            return -1, 0
        case _:
            sell_time, buy_time = reb_time.split("-")
            return price_array.index(sell_time), price_array.index(buy_time)


def get_stock_market(pivot_dict_stock, trading_dates, symbols, symbol_types) -> StockMarketData:
    df_open: pd.DataFrame = pivot_dict_stock["open"].loc[trading_dates, symbols]
    df_close: pd.DataFrame = pivot_dict_stock["close"].loc[trading_dates, symbols]
    df_preclose: pd.DataFrame = pivot_dict_stock["preclose"].loc[trading_dates, symbols]
    df_dieting: pd.DataFrame = pivot_dict_stock["dieting"].loc[trading_dates, symbols]
    # Not sure if necessary
    should_copy = True

    hour_prices = []
    # PLUMSOFT 于 2025-09-23 优化，可以有效减少内存占用
    nan_arr = np.full(df_open.shape, np.nan)
    for hour in sorted(price_array):
        if hour in ["open", "close", "preclose", "dieting"]:
            continue
        if hour in pivot_dict_stock.keys():
            hour_prices.append(pivot_dict_stock[hour].loc[trading_dates, symbols].to_numpy(copy=should_copy))
        else:
            hour_prices.append(nan_arr)

    data = StockMarketData(
        candle_begin_ts=(trading_dates.astype(np.int64) // 1000000000).to_numpy(copy=should_copy),
        op=df_open.to_numpy(copy=should_copy),
        cl=df_close.to_numpy(copy=should_copy),
        pre_cl=df_preclose.to_numpy(copy=should_copy),
        dieting=df_dieting.to_numpy(copy=should_copy),
        types=np.array(symbol_types, dtype=np.int16),
        hour_prices=hour_prices,
    )

    return data


def get_adjust_ratios(
    df_stock_ratio: pd.DataFrame, start_date, end_date, symbols, reb_time, offset_dt=0
) -> AdjustRatios:
    df_stock_ratio = df_stock_ratio.loc[start_date:end_date].reindex(columns=symbols, fill_value=0)

    adj_dts = (df_stock_ratio.index + pd.Timedelta(hours=offset_dt)).to_numpy().astype(np.int64) // 1000000000
    ratios = df_stock_ratio.to_numpy(dtype=np.float64)

    return AdjustRatios(adj_dts=adj_dts, ratios=ratios, reb_time=parse_rebalance_time(reb_time))


def build_hour_adj_ratios(
    conf: BacktestConfig, select_results: pd.DataFrame, symbols: list[str]
) -> Tuple[List, Dict[tuple, pd.DataFrame]]:
    """从策略级宽表 parquet + 选股结果构建小时级别的 adj_ratios。

    步骤：
    1. 按 (策略, 持仓周期, 换仓时间) 分组 → pivot 构建初始 period_ratio_hour_df
    2. 读取策略级信号宽表 parquet（上游已 pivot，此处直接读取）
       按策略收集所有 offset 的 union 时间范围，信号 reindex 每策略只做一次
    3. 每个 offset: reindex+ffill → 从预 reindex 信号中 .loc 切片乘算 → shift(-1h)
    4. get_adjust_ratios() → numba 结构

    返回:
        hour_adj_ratios: numba.typed.List[AdjustRatios]
        period_ratio_hour_df: Dict[tuple, pd.DataFrame]（用于小时评价）
    """
    t_total = time.perf_counter()
    period_ratio_hour_df = {}
    # ====================================================================================================
    # 1. 选股日期 → 交易日期(小时)，构建初始 pivot
    # ====================================================================================================
    period_offset = conf.load_period_offset()
    trading_arr = period_offset["交易日期"].sort_values().values

    # 全量小时轴（有序数组），用 searchsorted 做 O(log n) 区间切片
    # 直接用 numpy 向量化构建，避免 expand_daily_to_hourly 的 DataFrame 创建开销
    unique_dates = (
        period_offset["交易日期"]
        .drop_duplicates()
        .sort_values()
        .values.astype("datetime64[D]")
        .astype("datetime64[ns]")
    )
    _hour_offsets = np.array(
        [
            np.timedelta64(10 * 3600 + 30 * 60, "s"),  # 10:30
            np.timedelta64(11 * 3600 + 30 * 60, "s"),  # 11:30
            np.timedelta64(14 * 3600, "s"),  # 14:00
            np.timedelta64(15 * 3600, "s"),  # 15:00
        ]
    )
    hourly_arr = (unique_dates[:, None] + _hour_offsets[None, :]).ravel()

    for (strategy, period, reb_time), grp_df in select_results.groupby(["策略", "持仓周期", "换仓时间"], observed=True):
        pivot_table_df = grp_df.pivot_table(
            index="选股日期", columns="股票代码", values="目标资金占比", aggfunc="sum", fill_value=0, observed=True
        )
        # 选股日期 -> 交易日期
        pos = np.searchsorted(trading_arr, pivot_table_df.index, side="right")
        # 越界防护：选股日期 >= max(交易日期) 时 clip 到最后一个交易日
        overflow_mask = pos >= len(trading_arr)
        if np.any(overflow_mask):
            logger.warning(f"选股日期超出交易日期范围，已自动clip: {pivot_table_df.index[overflow_mask].tolist()}")
            pos = np.clip(pos, 0, len(trading_arr) - 1)
        pivot_table_df.index = trading_arr[pos]
        # 换仓日改成小时
        pivot_table_df.index = pivot_table_df.index.normalize() + pd.DateOffset(hour=10, minute=30)
        period_ratio_hour_df[(strategy, period, reb_time)] = pivot_table_df

    # ====================================================================================================
    # 2. 自动发现策略级信号宽表 parquet（上游已 pivot，此处直接读取）
    # ====================================================================================================
    strategy_signals = {}
    runtime_folder = conf.get_runtime_folder()
    for stg_parquet in runtime_folder.glob("个股择时_*.parquet"):
        stg_name = stg_parquet.stem.removeprefix("个股择时_")
        strategy_signals[stg_name] = pd.read_parquet(stg_parquet)

    # 按策略收集所有 offset 的时间范围，信号 reindex 每策略只做一次
    # 同时预存每个 key 的 (left, right) 区间，Step 3 直接复用，消除重复 searchsorted
    stg_trade_ranges: dict[str, tuple] = {}  # stg_name -> (global_left, global_right)
    key_ranges: dict[tuple, tuple] = {}  # key -> (left, right)
    target_end_dt = pd.Timestamp(conf.end_date) if conf.end_date is not None else pd.Timestamp(hourly_arr[-1])
    if conf.end_date is not None and target_end_dt == target_end_dt.normalize():
        target_end_dt = target_end_dt + pd.Timedelta(hours=15)
    target_right = np.searchsorted(hourly_arr, target_end_dt.to_datetime64(), side="right")
    for key, ratio_df in period_ratio_hour_df.items():
        stg_name = key[0]
        idx_min = ratio_df.index.min().to_datetime64()
        idx_max = ratio_df.index.max().to_datetime64()
        left = np.searchsorted(hourly_arr, idx_min)
        right = max(np.searchsorted(hourly_arr, idx_max, side="right"), target_right)
        key_ranges[key] = (left, right)
        if stg_name in stg_trade_ranges:
            prev_l, prev_r = stg_trade_ranges[stg_name]
            stg_trade_ranges[stg_name] = (min(prev_l, left), max(prev_r, right))
        else:
            stg_trade_ranges[stg_name] = (left, right)

    # 每策略信号 reindex 一次到 union 时间范围
    stg_signal_reindexed = {}
    for stg_name, sig_wide in strategy_signals.items():
        if stg_name in stg_trade_ranges:
            gl, gr = stg_trade_ranges[stg_name]
            union_dates = pd.DatetimeIndex(hourly_arr[gl:gr])
            stg_signal_reindexed[stg_name] = sig_wide.reindex(index=union_dates).fillna(1.0)

    # ====================================================================================================
    # 3. reindex + 信号宽表逐元素乘算 + shift(-1h)
    #    信号数据与小时轴严格对齐，缺失 = 无择时（1.0），无需 ffill
    # ====================================================================================================
    for key, ratio_df in period_ratio_hour_df.items():
        stg_name = key[0]
        # 复用 Step 2 预存的区间，避免重复 searchsorted
        left, right = key_ranges[key]
        trade_dates = pd.DatetimeIndex(hourly_arr[left:right])

        base_full = ratio_df.reindex(trade_dates)
        base_full.ffill(inplace=True)
        base_full.fillna(0, inplace=True)

        if stg_name in stg_signal_reindexed:
            sig_reindexed = stg_signal_reindexed[stg_name]
            common_cols = base_full.columns.intersection(sig_reindexed.columns)
            if len(common_cols) > 0:
                # 用 .values 跳过 pandas 索引对齐开销（trade_dates 已是 sig_reindexed.index 的子集）
                base_full.loc[:, common_cols] *= sig_reindexed.loc[trade_dates, common_cols].values

        base_full.index -= pd.Timedelta(hours=1)
        period_ratio_hour_df[key] = base_full

    # ====================================================================================================
    # 4. 转换为 numba AdjustRatios
    # ====================================================================================================
    hour_adj_ratios = List()
    for keys, df_stock_ratio in period_ratio_hour_df.items():
        adj_ratio = get_adjust_ratios(df_stock_ratio, conf.start_date, conf.end_date, symbols, keys[2], offset_dt=-8)
        hour_adj_ratios.append(adj_ratio)

    logger.debug(f"⏱️ build_hour_adj_ratios 总耗时: {time.perf_counter() - t_total:.3f}秒")

    return hour_adj_ratios, period_ratio_hour_df


def create_account_dataframes(
    trading_dates: pd.Series,
    cashes: np.ndarray,
    pos_values: np.ndarray,
    stamp_taxes: np.ndarray,
    commissions: np.ndarray,
    intraday: np.ndarray,
    leverages: np.ndarray,
    initial_cash: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成日级别和小时级别的账户DataFrame

    Parameters:
    -----------
    trading_dates : list
        交易日期列表，长度为N
    cashes : np.ndarray
        日级别账户可用资金，shape=(N,)
    pos_values : np.ndarray
        日级别持仓市值，shape=(N,)
    stamp_taxes : np.ndarray
        日级别印花税，shape=(N,)
    commissions : np.ndarray
        日级别券商佣金，shape=(N,)
    intraday : np.ndarray
        小时级别数据，shape=(4*N, 4)，列顺序为[cashes, pos_values, stamp_taxes, commissions]
    leverages : np.ndarray
        日级别杠杆，shape=(N,)
    initial_cash : float
        初始资金

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (日级别account_df, 小时级别account_df)
    """

    # ==================== 日级别 DataFrame ====================
    daily_df = pd.DataFrame(
        {
            "交易日期": trading_dates,
            "账户可用资金": cashes,
            "持仓市值": pos_values,
            "印花税": stamp_taxes,
            "券商佣金": commissions,
        }
    ).reset_index(drop=True)

    daily_df["总资产"] = daily_df["账户可用资金"] + daily_df["持仓市值"]
    daily_df["净值"] = daily_df["总资产"] / initial_cash
    daily_df = daily_df.assign(
        手续费=daily_df["印花税"] + daily_df["券商佣金"],
        涨跌幅=daily_df["净值"].pct_change(),
        杠杆=leverages,
        实际杠杆=daily_df["持仓市值"] / daily_df["总资产"],
    )

    # ==================== 小时级别 DataFrame ====================
    # 生成小时级别的时间标签
    # 生成小时级别的交易日期时间
    # 生成小时级别的交易日期时间
    hour_labels = ["10:30", "11:30", "14:00", "15:00"]

    # 使用列表推导式生成小时级别日期时间
    hourly_datetimes = [f"{pd.Timestamp(date).date()} {hour}" for date in trading_dates.values for hour in hour_labels]

    # 从intraday数组提取数据
    hourly_cashes = intraday[:, 0]
    hourly_pos_values = intraday[:, 1]
    hourly_stamp_taxes = intraday[:, 2]
    hourly_commissions = intraday[:, 3]

    # 生成小时级别杠杆
    hourly_leverages = np.repeat(leverages, len(hour_labels))

    hourly_df = pd.DataFrame(
        {
            "交易日期": hourly_datetimes,
            "账户可用资金": hourly_cashes,
            "持仓市值": hourly_pos_values,
            "印花税": hourly_stamp_taxes,
            "券商佣金": hourly_commissions,
        }
    ).reset_index(drop=True)
    hourly_df["交易日期"] = pd.to_datetime(hourly_df["交易日期"])

    hourly_df["总资产"] = hourly_df["账户可用资金"] + hourly_df["持仓市值"]
    hourly_df["净值"] = hourly_df["总资产"] / initial_cash
    hourly_df = hourly_df.assign(
        手续费=hourly_df["印花税"] + hourly_df["券商佣金"],
        涨跌幅=hourly_df["净值"].pct_change(),
        杠杆=hourly_leverages,
        实际杠杆=hourly_df["持仓市值"] / hourly_df["总资产"],
    )

    return daily_df, hourly_df


def calc_equity(
    conf: BacktestConfig,
    pivot_dict_stock: dict,
    period_ratio_df: dict[tuple, pd.DataFrame],
    symbols: list[str],
    leverage: float | pd.Series = None,
    select_results: pd.DataFrame = None,
):
    """
    模拟投资组合的表现，生成资金曲线以跟踪组合收益变化。
    :param conf: 回测配置
    :param pivot_dict_stock: 原始数据
    :param period_ratio_df: 持仓周期权重
    :param symbols: 股票代码
    :param leverage: 杠杆
    :param select_results: 选股结果（个股择时模式下需要传入，用于构建 hour_adj_ratios）
    :return: (daily_account_df, hourly_account_df, period_ratio_hour_df or None)
    """
    symbol_types = [get_symbol_type(sym) for sym in symbols]
    # if any(x == BSE_MAIN for x in symbol_types):
    #     raise ValueError(f'BSE not supported')  # No Beijing stocks

    # 确定回测区间
    start_date = pd.to_datetime(conf.start_date)
    trading_dates = conf.read_trading_dates(start_date, conf.end_date)
    market_dates = pivot_dict_stock["close"].index
    # 如果交易日期的最新一天 > 行情数据的最新一天 说明是昨天跑的回测，今天又跑了step4
    if trading_dates.iloc[-1] > market_dates[-1]:
        # warning要放在裁剪时间之前，不然trading_dates.iloc[-1]拿到的是裁切后的数据
        logger.warning(
            f"交易日期和行情日期不匹配：疑似用之前运行的回测数据，今天再次运行step4。已自动裁剪时间\n"
            f"最新交易日：{trading_dates.iloc[-1]} 最新行情日：{market_dates[-1]}"
        )
        trading_dates = trading_dates.loc[trading_dates <= market_dates[-1]]

    # 读取行情
    market = get_stock_market(pivot_dict_stock, trading_dates, symbols, symbol_types)

    if leverage is None:
        leverage = conf.total_cap_usage

    if isinstance(leverage, pd.Series):
        leverages = leverage.to_numpy(dtype=np.float64)
    else:
        leverages = np.full(len(market.candle_begin_ts), leverage, dtype=np.float64)

    # 小时框架实际交易时偏移的offset  （注意，此处又做了一次转换，因为我们要用的是索引，而不是分钟值，且我们使用的是5m数据，所以除5）
    stock_timing_order_price = np.full(
        len(market.candle_begin_ts),
        getattr(config, "stock_timing_order_price", 5) // 5 if conf.use_stock_timing else 0,
        dtype=np.int8,
    )

    stay_real = np.full(len(market.candle_begin_ts), int(getattr(config, "stay_real", True)), dtype=np.int8)

    # 开始回测
    params = SimuParams(
        init_cash=conf.initial_cash,  # 初始资金
        stamp_tax_rate=conf.t_rate,  # 印花税率
        commission_rate=conf.c_rate,  # 券商佣金费率
    )
    logger.debug(
        f"ℹ️ 实际模拟资金:{params.init_cash:,.2f}(整体使用率:{conf.total_cap_usage * 100:.2f}%), "
        f"印花税率:{params.stamp_tax_rate * 100 :.2f}%, "
        f"券商佣金费率:{params.commission_rate * 100 :.2f}%"
    )

    adj_ratios = List()
    for keys, df_stock_ratio in period_ratio_df.items():
        # strategy, period, reb_time = keys
        adj_ratio = get_adjust_ratios(df_stock_ratio, conf.start_date, conf.end_date, symbols, keys[2])
        adj_ratios.append(adj_ratio)

    period_ratio_hour_df = None
    if conf.use_stock_timing and select_results is not None:
        hour_adj_ratios, period_ratio_hour_df = build_hour_adj_ratios(conf, select_results, symbols)
    elif conf.use_stock_timing:
        logger.warning("use_stock_timing=True 但未传入 select_results，跳过小时级别择时")
        hour_adj_ratios = List.empty_list(AdjustRatios.class_type.instance_type)  # type: ignore
    else:
        # njit模式下，必须要传一个empty list过去，此时hour_adj_ratios的长度仍然为0
        hour_adj_ratios = List.empty_list(AdjustRatios.class_type.instance_type)  # type: ignore

    pos_calc = RebAlways(market.types)

    s_time = time.perf_counter()
    logger.debug("🎯 开始模拟交易...")
    if len(adj_ratios) > 0:
        cashes, pos_values, stamp_taxes, commissions, intraday = start_simulation(
            market, params, adj_ratios, leverages, pos_calc, stay_real, hour_adj_ratios, stock_timing_order_price
        )
    else:
        # 无交易时的默认值
        n_days = len(trading_dates)
        cashes = np.full(n_days, params.init_cash)
        pos_values = np.zeros(n_days)
        stamp_taxes = np.zeros(n_days)
        commissions = np.zeros(n_days)
        # 小时级别默认值: shape=(4*n_days, 4)
        n_hours = 4 * n_days
        intraday = np.column_stack(
            [
                np.full(n_hours, params.init_cash),  # cashes
                np.zeros(n_hours),  # pos_values
                np.zeros(n_hours),  # stamp_taxes
                np.zeros(n_hours),  # commissions
            ]
        )

    logger.ok(f"完成模拟交易，花费时间: {time.perf_counter() - s_time:.3f}秒")

    # 计算收益
    daily_account_df, hourly_account_df = create_account_dataframes(
        trading_dates=trading_dates,
        cashes=cashes,
        pos_values=pos_values,
        stamp_taxes=stamp_taxes,
        commissions=commissions,
        intraday=intraday,
        leverages=leverages,
        initial_cash=conf.initial_cash,
    )

    return daily_account_df, hourly_account_df, period_ratio_hour_df


@nb.njit
def find_hour_idx_in_adj_dts(ratios, daily_ts: int, idx_price: int) -> Tuple[int, np.ndarray, int, np.ndarray]:
    """
    在adj_dts中查找对应的小时时间戳，同时返回当前和前一个索引及复权因子

    Args:
        ratios: 包含 adj_dts 和 ratios 属性的对象
        daily_ts: 日级别时间戳
        idx_price: 价格索引

    Returns:
        Tuple[int, np.ndarray, int, np.ndarray]:
            (当前索引, 当前复权因子, 前一个索引, 前一个复权因子)
            - 索引为 -1 时，返回 ratios[0] 作为默认值

    Note:
        idx_price=25 时，虽然价格数据来自 13:05，
        但会查找并返回 adj_dts 中 13:00 的索引位置
    """
    # 获取目标时间戳（idx_price=25 时返回 13:00）
    # ========== 空值保护 ==========
    default_ratios = np.ones(1, dtype=np.float64)

    if ratios is None:
        return -1, default_ratios, -1, default_ratios

    if len(ratios.adj_dts) == 0 or len(ratios.ratios) == 0:
        return -1, default_ratios, -1, default_ratios

    idx_price_map = {
        0: (9 * 3600 + 30 * 60, 9 * 3600 + 30 * 60),  # 09:30 数据 -> 09:30 位置
        12: (10 * 3600 + 30 * 60, 10 * 3600 + 30 * 60),  # 10:30 数据 -> 10:30 位置
        25: (13 * 3600 + 5 * 60, 13 * 3600),  # 13:05 数据 -> 13:00 位置 (特殊)
        36: (14 * 3600, 14 * 3600),  # 14:00 数据 -> 14:00 位置
        48: (15 * 3600, 15 * 3600),  # 15:00 数据 -> 15:00 位置
    }

    if idx_price not in idx_price_map:
        return -1, ratios.ratios[0], -1, ratios.ratios[0]

    _, target_offset = idx_price_map[idx_price]
    utf8_offset = 8 * 3600  # 8小时 = 28800秒
    target_ts = daily_ts + target_offset - utf8_offset

    # 二分查找
    idx = np.searchsorted(ratios.adj_dts, target_ts, side="left")

    # 当前索引：精确匹配
    hour_idx = int(idx) if idx < len(ratios.adj_dts) and ratios.adj_dts[idx] == target_ts else -1
    hour_ratios = ratios.ratios[0] if hour_idx == -1 else ratios.ratios[hour_idx]

    # 前一个索引：严格小于 target_ts 的最后一个位置
    prev_hour_idx = int(idx - 1) if idx > 0 else -1
    prev_hour_ratios = ratios.ratios[0] if prev_hour_idx == -1 else ratios.ratios[prev_hour_idx]

    return prev_hour_idx, prev_hour_ratios, hour_idx, hour_ratios


@nb.njit(boundscheck=True)
def start_simulation(
    market, simu_params, adj_ratios, leverages, pos_calc, stay_real, hour_adj_ratios, stock_timing_order_price
):
    """
    模拟股票交易的函数，逐 K 线模拟交易过程，计算账户资金、仓位价值、印花税和佣金等。

    参数:
    - market: StockMarketData 类型，包含市场数据（如 K 线时间戳、价格等）。
    - simu_params: SimuParams 类型，包含模拟参数（如初始资金、佣金率、印花税率等）。
    - adj_ratios: AdjustRatios 类型，包含策略调仓信息（如调仓日期、目标权重、买卖价格索引等）。
    - leverages: np.array 类型，包含动态杠杆
    - pos_calc: 仓位计算函数，用于计算目标买入仓位。

    返回:
    - cashes: 每根 K 线收盘时的账户可用资金。
    - pos_values: 每根 K 线收盘时的仓位价值。
    - stamp_taxes: 每根 K 线产生的印花税。
    - commissions: 每根 K 线产生的券商佣金。
    """
    # K 线数量
    n_bars = len(market.candle_begin_ts)

    # 股票品种数量
    n_syms = len(market.types)

    # 策略数量
    n_ratios = len(adj_ratios)

    # 账户可用资金 = 初始资金
    available_cash = simu_params.init_cash

    # 记录每根 K 线收盘时的仓位价值
    pos_values = np.zeros(n_bars, dtype=np.float64)

    # 记录每根 K 线收盘时的账户可用资金
    cashes = np.zeros(n_bars, dtype=np.float64)

    # 日内换仓的记录
    intraday = np.zeros((n_bars * 4, 4), dtype=np.float64)
    idx_hour = 0

    # 是否使用了小时ratios
    use_hour_data = len(hour_adj_ratios) > 0

    # 记录每根 K 线产生的印花税
    stamp_taxes = np.zeros(n_bars, dtype=np.float64)

    # 记录每根 K 线产生的券商佣金
    commissions = np.zeros(n_bars, dtype=np.float64)

    # 为每个策略创建模拟器
    sims = List()
    for i in range(n_ratios + 1):
        sim = Simulator(0, simu_params.commission_rate, simu_params.stamp_tax_rate, np.zeros(n_syms, dtype=np.float64))
        sims.append(sim)
    # 跌停模拟器，独立于【策略模拟器】之外，用于处理跌停还卖出成功的BUG
    dieting_sim = sims[-1]

    """
    下面这些一维/二维变量用于在主循环中记录每个策略在“本根 K 线”下的待执行状态
    与目标权重。它们会在当天不同价格点/收盘后被读取或更新。
    """
    # 策略的调仓周期索引，用于跟踪每个策略的调仓日期。所以没有日期的概念，只有调仓周期的概念
    adj_dt_idxes = np.zeros(n_ratios, dtype=np.int64)

    # 策略的调仓日期索引：(当日信号，次日信号)
    # - sell_dt_idxes: 卖出调仓日期索引，0 表示不调仓，1 表示调仓。
    sell_dt_idxes = np.full((n_ratios, 2), 0, dtype=np.int8)
    # - buy_dt_idxes: 买入调仓日期索引，0 表示不调仓，1 表示调仓。
    buy_dt_idxes = np.full((n_ratios, 2), 0, dtype=np.int8)

    # 可卖金额量（全局，所有sim共用一个。不是手数，而是金额）
    sellable_values = np.zeros(n_syms, dtype=np.float64)

    # 策略的调仓价格索引：
    # - sell_price_idxes: 卖出价格索引，与 market.prices 对应。
    sell_price_idxes = np.zeros(n_ratios, dtype=np.int8)
    # - buy_price_idxes: 买入价格索引，与 market.prices 对应。
    buy_price_idxes = np.zeros(n_ratios, dtype=np.int8)

    # 策略的买入权重矩阵，形状为: 策略数 * 股票品种数
    buy_ratios = np.zeros((n_ratios, n_syms), dtype=np.float64)  # 当前调仓的买入权重
    next_buy_ratios = np.zeros((n_ratios, n_syms), dtype=np.float64)  # 下一个调仓的买入权重

    # 缓存 find_hour_idx_in_adj_dts 的结果，避免同一 (idx_bar, idx_price) 下被调用 3 次
    cached_hour_idx = np.full(n_ratios, -1, dtype=np.int64)
    cached_prev_hour_idx = np.full(n_ratios, -1, dtype=np.int64)

    # 主循环：逐 K 线（交易日）模拟整段回测。
    # 当天流程：
    # 1) 开盘前：用前收盘价刷新各策略持仓价格（作为 T+1 的持仓价值基线）。
    # 2) 设置调仓：若某策略的调仓日等于今天，记录其卖/买的执行“日偏移”(T+0/T+1) 与执行“价格点索引”。
    # 3) 连续竞价：按价格点顺序执行——先“只卖不买”释放现金，再按目标权重“买/换仓”，并在每个价格点结算持仓价值与最新价。
    # 4) 收盘：将 T+1 任务的日偏移从 1 递减为 0，并记录当日的资金/仓位/费用。
    for idx_bar in range(n_bars):
        # 初始化本周期印花税和券商佣金
        stamp_tax = commission = hour_stamp_tax = hour_commission = 0.0

        # K 线开盘前操作：用前收盘价更新模拟器的持仓价格
        for sim in sims:
            sim.fill_last_prices(market.pre_cl[idx_bar])

        # 遍历所有策略：若到了该策略的调仓日期，则设置本次卖/买的执行计划
        # - idx_simu：策略索引；adj_dt_idx：该策略下一次调仓日期在 adj_dts 中的下标；adj_info：调仓信息
        for idx_simu, (adj_dt_idx, adj_info) in enumerate(zip(adj_dt_idxes, adj_ratios)):
            if adj_dt_idx < len(adj_info.adj_dts) and adj_info.adj_dts[adj_dt_idx] == market.candle_begin_ts[idx_bar]:
                # 配置卖出：sp_idx（sell price index）决定卖出的价格点
                if adj_info.sp_idx < 0:  # 负数表示 T+0 当日卖出（如 -1=当日收盘价）
                    sell_dt_idxes[idx_simu, 0] = 1  # 当日卖出
                    # 负索引从prices数组末尾倒数，如-1对应收盘价
                    sell_price_idxes[idx_simu] = len(market.prices) + adj_info.sp_idx
                else:  # 非负表示 T+1 次日卖出（0=次日开盘，1=次日09:30，…）
                    sell_dt_idxes[idx_simu, 1] = 1  # 次日卖出
                    sell_price_idxes[idx_simu] = adj_info.sp_idx

                # 配置买入：bp_idx（buy price index）决定买入的价格点
                if adj_info.bp_idx < 0:  # 负数表示 T+0 当日买入（如 -1=当日收盘价）
                    buy_dt_idxes[idx_simu, 0] = 1  # 当日买入
                    # 同一个模拟器内，这个暂时是不会发生变化的，因为换仓时间点是固定的
                    buy_price_idxes[idx_simu] = len(market.prices) + adj_info.bp_idx
                    buy_ratios[idx_simu, :] = adj_info.ratios[adj_dt_idx]
                else:  # 非负表示 T+1 次日买入
                    buy_dt_idxes[idx_simu, 1] = 1  # 次日买入
                    # 同一个模拟器内，这个暂时是不会发生变化的，因为换仓时间点是固定的
                    buy_price_idxes[idx_simu] = adj_info.bp_idx
                    next_buy_ratios[idx_simu, :] = adj_info.ratios[adj_dt_idx]

                # 设置本次调仓的目标权重分配
                # 当前这次调仓下，各标的的目标资金占比

                # 调仓周期索引递增，指向下一个调仓日期
                adj_dt_idxes[idx_simu] += 1

        # 连续竞价阶段：逐价格点模拟交易。从 open -> 0930 -> 0935 -> ... -> 1300 -> ... -> 1455-> close 逐个价格点模拟交易
        for idx_price, last_price in enumerate(market.prices):
            offset_idx = idx_price + stock_timing_order_price[0]
            offset_idx = max(0, min(offset_idx, len(market.prices) - 1))
            # 如果是1305，需要把offset_idx减1。因为发生偏移时，1130偏移30分钟，用的是1330的价格。而非偏移时，1130用的是1305的价格，相当于自动偏移了5分钟。
            if stock_timing_order_price[0] > 0 and idx_price == 25:
                offset_idx -= 1
            last_price = market.prices[offset_idx]

            # 更新每个模拟器的持仓价值和最新价格
            for i, sim in enumerate(sims):
                sim.settle_pos_values(last_price[idx_bar])
                # 可卖量只需要更新一次，反复更新会导致可卖量错误
                if i == 0:
                    sellable_values = sim.settle_sellable_values(last_price[idx_bar], sellable_values)
                sim.fill_last_prices(last_price[idx_bar])

            # PLUMSOFT 于 2025-09-23 优化，可以有效减少循环次数
            if not np.all(np.isnan(last_price[idx_bar])):
                # ==================跌停模拟器，每天的换仓价都尝试卖出==================
                if stay_real[idx_bar] == 1:
                    # 将跌停模拟器可用资金转回账户总可用资金
                    dieting_sim_stamp_tax, dieting_sim_commission = dieting_sim.dieting_sell_all(
                        last_price[idx_bar], dieting_sim.is_pos_and_dieting(market.dieting[idx_bar])
                    )
                    stamp_tax += dieting_sim_stamp_tax
                    commission += dieting_sim_commission

                    hour_stamp_tax += dieting_sim_stamp_tax
                    hour_commission += dieting_sim_commission

                    sim_cash = dieting_sim.withdraw_all()
                    available_cash += sim_cash
                # ================================================================

                # 判断需要卖出的策略：当日执行且以当前价格换仓
                need_sell = np.logical_and(sell_dt_idxes[:, 0] == 1, sell_price_idxes == idx_price)

                # 判断需要买入的策略：当日执行且以当前价格换仓
                need_buy = np.logical_and(buy_dt_idxes[:, 0] == 1, buy_price_idxes == idx_price)

                # ===== 缓存 find_hour_idx_in_adj_dts，避免同一 (idx_bar, idx_price) 重复调用 3 次 =====
                # 仅在小时时刻 {0,12,25,36,48} 才可能匹配到有效的 hour_idx，其余 ~90% 的 idx_price 直接跳过
                _is_hour_price = (
                    idx_price == 0 or idx_price == 12 or idx_price == 25 or idx_price == 36 or idx_price == 48
                )
                if use_hour_data and _is_hour_price:
                    for _i in range(n_ratios):
                        cached_prev_hour_idx[_i], _, cached_hour_idx[_i], _ = find_hour_idx_in_adj_dts(
                            hour_adj_ratios[_i], market.candle_begin_ts[idx_bar], idx_price=idx_price
                        )
                else:
                    for _i in range(n_ratios):
                        cached_prev_hour_idx[_i] = -1
                        cached_hour_idx[_i] = -1

                # 先处理“只卖不买”的策略：释放现金，避免买入受限  注，最后一个模拟器是跌停模拟器，所以要过滤掉
                for idx_simu, sim in enumerate(sims[:-1]):
                    use_hour_ratios = cached_hour_idx[idx_simu] != -1
                    if stay_real[idx_bar] == 1 and (need_sell[idx_simu] or need_buy[idx_simu] or use_hour_ratios):
                        # 获取【有持仓】且【最新价跌停】的股票
                        # 把要转入到跌停模拟器的金额给保存下来
                        # 此处只需要考虑纯卖出，没有轧差的情况
                        # 通过transfer，将sim中跌停的pos_values给减掉，那么sim.sell_all中，得到的delta_values, target_values就是两个0
                        # 当delta_values为0时，就不会产生手续费
                        # 当sell_all执行完毕，跌停的股票产生的手续费为0，且pos_values已经通过transfer转移到了跌停模拟器中
                        has_pos_and_dieting = sim.is_pos_and_dieting(market.dieting[idx_bar])
                        # 将资金转入到跌停模拟器中
                        dieting_sim.transfer(sim, has_pos_and_dieting)
                    if need_sell[idx_simu] and not need_buy[idx_simu]:
                        # 卖出全部股票，并计算印花税和佣金
                        sim_stamp_tax, sim_commission = sim.sell_all(last_price[idx_bar])
                        stamp_tax += sim_stamp_tax
                        commission += sim_commission

                        hour_stamp_tax += sim_stamp_tax
                        hour_commission += sim_commission

                        # 将模拟器可用资金转回账户总可用资金
                        sim_cash = sim.withdraw_all()
                        available_cash += sim_cash

                # 计算账户总权益（可用资金 + 所有模拟器的仓位价值），并应用当日杠杆
                total_equity = available_cash + sum([sim.get_pos_value() for sim in sims])
                total_equity *= leverages[idx_bar]

                # 禁止卖出的股票，也就是T+0限制了的股票
                disable_sell_mask_arr = np.full((n_ratios, n_syms), False, dtype=np.bool_)
                # 判断当前时刻和前一时刻的ratios是否完全一致（把原来的always rebalance模式给改了）
                eq_hour_ratio_arr = np.full(n_ratios, False, dtype=np.bool_)

                # ====================================================================================================
                # 注：此段for循环仅仅是为了拿到disable_sell_mask_arr。不能和后面的for循环写一起，不然会导致可用资金不对
                if use_hour_data:
                    for idx_simu, (sim, adj_dt_idx, ratios) in enumerate(zip(sims, adj_dt_idxes, buy_ratios)):
                        # 使用缓存的索引，避免重复调用 find_hour_idx_in_adj_dts
                        hour_idx = cached_hour_idx[idx_simu]
                        prev_hour_idx = cached_prev_hour_idx[idx_simu]
                        use_hour_ratios = hour_idx != -1
                        # 从缓存索引反查 ratios，hour_idx==-1 时回退到 ratios[0]
                        # 与原始 find_hour_idx_in_adj_dts 行为一致：空数组时返回 ones 默认值
                        _hr = hour_adj_ratios[idx_simu].ratios
                        if len(_hr) == 0:
                            _fallback = np.ones(n_syms, dtype=np.float64)
                            hour_ratios = _fallback
                            prev_hour_ratios = _fallback
                        else:
                            hour_ratios = _hr[hour_idx] if hour_idx != -1 else _hr[0]
                            prev_hour_ratios = _hr[prev_hour_idx] if prev_hour_idx != -1 else _hr[0]
                        if need_buy[idx_simu] or use_hour_ratios:
                            # 用是否完全相等的方式来判断前后两个ratio，而不是单纯用sum判断！会有坑！！
                            eq_hour_ratio_arr[idx_simu] = np.array_equal(hour_ratios, prev_hour_ratios)
                            # 如果是换仓时间，那必须固定换仓？是的，必须固定换仓，实盘也这样，所以会轧差，继而发生补买/卖。
                            if not need_buy[idx_simu] and eq_hour_ratio_arr[idx_simu]:
                                continue
                            # 目标建仓权益 = 当日总权益 × 该策略本次调仓的资金占比之和
                            ratio_sum = np.sum(hour_ratios) if use_hour_ratios else np.sum(ratios)
                            target_equity = total_equity * ratio_sum

                            # 最大可达权益 = 策略仓位价值 + 总可用资金
                            max_possible_equity = sim.get_pos_value() + available_cash

                            # 若即使转入全部可用现金也达不到目标，则下调为最大可达权益
                            if max_possible_equity < target_equity:
                                target_equity = max_possible_equity

                            # 归一化持仓权重（若占比和≈0，则不下单）
                            if abs(ratio_sum) < 1e-8:
                                ratios_norm = np.zeros(n_syms, dtype=np.float64)
                            else:
                                ratios_norm = (hour_ratios if use_hour_ratios else ratios) / ratio_sum

                            # 基于目标建仓权益与归一化权重，计算各标的目标持仓
                            target_pos = pos_calc.calc_lots(target_equity, last_price[idx_bar], ratios_norm)

                            # 之前做跌停模拟器的时候把逻辑拆开了，后来发现又不用拆了，暂时先这样
                            # 注意！目前在处理跌停的时候，不会考虑轧差的情况
                            # 比如买2手，卖3手，理论上轧差后需要补卖1手。但目前是买2手，卖的3手由于卖不出去，所以会交给跌停模拟器
                            delta_values, target_values = sim.calc_delta_values(last_price[idx_bar], target_pos)

                            # 卖出股票mask
                            sold_mask = delta_values < 0
                            # 不允许卖出的股票，卖出量 > 可卖量，卖出量：delta_values中小于0的部分就是卖出量，转正即可
                            disable_sell_mask = np.logical_and(sold_mask, -delta_values > sellable_values)

                            # 需要卖，且能卖的股票
                            legal_sold_mask = np.logical_and(sold_mask, np.logical_not(disable_sell_mask))
                            # 需要修改每个股票的可卖量，减去卖掉的金额即可（减去一个负数就是加上一个正数）
                            sellable_values[legal_sold_mask] += delta_values[legal_sold_mask]

                            # fmt: off
                            # T+0顺延的核心逻辑，把当前值用上一个值替代   (之所以放在前面，是因为可用资金那边有点问题)
                            hour_adj_ratios[idx_simu].ratios[hour_idx][disable_sell_mask] = hour_adj_ratios[idx_simu].ratios[prev_hour_idx][disable_sell_mask]
                            # fmt: on
                            disable_sell_mask_arr[idx_simu] = disable_sell_mask
                # ====================================================================================================

                # 再处理需要买入/换仓的策略
                for idx_simu, (sim, adj_dt_idx, ratios) in enumerate(zip(sims, adj_dt_idxes, buy_ratios)):
                    # 使用缓存的索引，避免重复调用 find_hour_idx_in_adj_dts
                    hour_idx = cached_hour_idx[idx_simu]
                    use_hour_ratios = hour_idx != -1
                    if use_hour_ratios:
                        # 注意: loop 2 可能已修改 ratios[hour_idx]，此处读取更新后的值
                        hour_ratios = hour_adj_ratios[idx_simu].ratios[hour_idx]
                    if need_buy[idx_simu] or use_hour_ratios:
                        # 如果是换仓时间，那必须固定换仓？是的，必须固定换仓，实盘也这样，所以会轧差，继而发生补买/卖。
                        # 如果所有的股票差值累加和为0，那么说明所有的股票都不需要换仓，直接跳过即可
                        if not need_buy[idx_simu] and eq_hour_ratio_arr[idx_simu]:
                            continue

                        # 目标建仓权益 = 当日总权益 × 该策略本次调仓的资金占比之和
                        ratio_sum = np.sum(hour_ratios) if use_hour_ratios else np.sum(ratios)
                        target_equity = total_equity * ratio_sum

                        # 最大可达权益 = 策略仓位价值 + 总可用资金
                        max_possible_equity = sim.get_pos_value() + available_cash

                        # 若即使转入全部可用现金也达不到目标，则下调为最大可达权益
                        if max_possible_equity < target_equity:
                            target_equity = max_possible_equity

                        # 需要转入资金 = max(目标建仓权益 - 当前仓位价值, 0)
                        if target_equity > sim.get_pos_value():
                            required_cash = target_equity - sim.get_pos_value()
                        else:
                            required_cash = 0

                        # 将建仓所需资金存入策略模拟器
                        available_cash -= required_cash
                        sim.deposit(required_cash)

                        # 归一化持仓权重（若占比和≈0，则不下单）
                        if abs(ratio_sum) < 1e-8:
                            ratios_norm = np.zeros(n_syms, dtype=np.float64)
                        else:
                            ratios_norm = (hour_ratios if use_hour_ratios else ratios) / ratio_sum

                        # 基于目标建仓权益与归一化权重，计算各标的目标持仓
                        target_pos = pos_calc.calc_lots(target_equity, last_price[idx_bar], ratios_norm)

                        # 之前做跌停模拟器的时候把逻辑拆开了，后来发现又不用拆了，暂时先这样
                        # 注意！目前在处理跌停的时候，不会考虑轧差的情况
                        # 比如买2手，卖3手，理论上轧差后需要补卖1手。但目前是买2手，卖的3手由于卖不出去，所以会交给跌停模拟器
                        delta_values, target_values = sim.calc_delta_values(last_price[idx_bar], target_pos)

                        # 获取禁止卖出的股票
                        disable_sell_mask = disable_sell_mask_arr[idx_simu]

                        # 当出现禁止卖出的情况，能卖多少卖多少
                        delta_values[disable_sell_mask] = -sellable_values[disable_sell_mask]
                        target_values[disable_sell_mask] = (
                            sim.pos_values[disable_sell_mask] - sellable_values[disable_sell_mask]
                        )

                        # ==================巨坑代码==================
                        # numba模式下，对 jitclass 对象的数组属性使用布尔索引进行原地修改不生效
                        # 这两行代码完全失效，也不会报错。必须要要在sim对象中修改才有用！！！
                        # dieting_sim.pos_values[re_sell] += sim.pos_values[re_sell]
                        # sim.pos_values[re_sell] -= sim.pos_values[re_sell]
                        # ===========================================
                        # 调整仓位并统计费用
                        sim_stamp_tax, sim_commission = sim.adjust_positions(
                            last_price[idx_bar], delta_values, target_values
                        )

                        # 调仓过后，需要修改每个股票的可卖量，减去卖掉的金额即可（减去一个负数就是加上一个正数）
                        # 注意，此处只需要更新【禁止卖出的股票】即可，disable_sell_mask为False的情况已经在另一个for循环更新了
                        sellable_values[disable_sell_mask] += delta_values[disable_sell_mask]

                        stamp_tax += sim_stamp_tax
                        commission += sim_commission

                        hour_stamp_tax += sim_stamp_tax
                        hour_commission += sim_commission

                        # 将模拟器可用资金转回账户总可用资金
                        sim_cash = sim.withdraw_all()
                        available_cash += sim_cash
            # 如果idx_price属于小时时刻[1030 1130 1400 1500]，那么就需要统计一次，注意在统计之前要更新没个模拟器的市值
            if idx_price in np.array([12, 24, 36, 48]):
                intraday[idx_hour] = np.array(
                    [available_cash, sum([sim.get_pos_value() for sim in sims]), hour_stamp_tax, hour_commission]
                )
                idx_hour += 1
                hour_stamp_tax = hour_commission = 0.0

        # 更新调仓任务的“日偏移”：把 T+1（值为1）的任务在收盘后置为 0，表示“次日将变为当日任务”
        # 把 T+1（值为1）的任务在收盘后置为 0，表示“次日将变为当日任务”
        buy_dt_idxes[:, 0] = buy_dt_idxes[:, 1]
        sell_dt_idxes[:, 0] = sell_dt_idxes[:, 1]
        sell_dt_idxes[:, 1] = 0
        buy_dt_idxes[:, 1] = 0

        # 更新下一个调仓的买入权重
        buy_ratios = next_buy_ratios
        next_buy_ratios = np.zeros((n_ratios, n_syms), dtype=np.float64)

        # 更新可卖量  注：nb模式下不支持np.sum(axis=0)
        sellable_values = sims[0].pos_values.copy()
        for i in range(1, len(sims)):
            sellable_values += sims[i].pos_values

        # 记录本周期数据
        stamp_taxes[idx_bar] = stamp_tax
        commissions[idx_bar] = commission
        pos_values[idx_bar] = sum([sim.get_pos_value() for sim in sims])
        cashes[idx_bar] = available_cash

    return cashes, pos_values, stamp_taxes, commissions, intraday


# ====================================================================================================
# 动态杠杆再择时模拟
# 1. 生成动态杠杆
# 2. 进行动态杠杆再择时的回测模拟
# 3. 保存结果
# ====================================================================================================
def simu_equity_timing(
    conf: BacktestConfig, pivot_dict_stock: dict, period_ratio_df: Dict[tuple, pd.DataFrame], symbols: List[str]
):
    """
    动态杠杆再择时模拟
    :param conf: 回测配置
    :param pivot_dict_stock: 全部行情数据
    :param period_ratio_df: 股票目标资金占比
    :param symbols: 股票代码列表
    :return: 资金曲线，策略收益，年化收益
    """
    logger.info(f"资金曲线再择时，生成动态杠杆")

    # 记录开始时间，用于计算耗时
    s_time = time.time()

    # 读取资金曲线数据，作为动态杠杆计算的基础
    account_df = pd.read_csv(conf.get_result_folder() / "资金曲线.csv", index_col=0, encoding="utf-8-sig")

    # 生成动态杠杆，根据资金曲线的权益变化进行杠杆调整
    equity_signal = conf.equity_timing.get_equity_signal(account_df)
    logger.ok(f"完成生成动态杠杆，花费时间： {time.time() - s_time:.3f}秒")

    # 将equity_signals的index设置为交易日期
    equity_signal.index = pd.to_datetime(account_df["交易日期"])

    # 对每个换仓日期，找到对应的动态杠杆值并相乘
    for (strategy, period, reb_time), df_stock_ratio in period_ratio_df.items():
        period_ratio_df[(strategy, period, reb_time)] = df_stock_ratio.mul(
            equity_signal.reindex(df_stock_ratio.index), axis=0
        )

    # 记录时间，用于后续动态杠杆再择时的耗时统计
    s_time = time.time()
    logger.info(f"开始动态杠杆再择时模拟交易，累计回溯{len(account_df):,} 天...")

    # 进行资金曲线的再择时回测模拟
    # - 使用动态杠杆调整后的持仓计算资金曲线
    # - 包括现货和合约的比例数据
    # - 计算回测的总体收益、年度收益、季度收益和月度收益
    daily_account_df, hourly_account_df, _ = calc_equity(conf, pivot_dict_stock, period_ratio_df, symbols)

    rtn, year_return, month_return, quarter_return = _evaluate_and_save_performance(
        conf, daily_account_df, period_name="日线", suffix="_再择时"
    )
    # if conf.use_stock_timing:
    #     _evaluate_and_save_performance(conf, hourly_account_df, period_name="小时", suffix="_再择时")

    logger.ok(f"完成动态杠杆再择时模拟交易，花费时间：{time.time() - s_time:.3f}秒")

    # 返回再择时后的资金曲线和收益结果，用于后续分析或评估
    return daily_account_df, rtn, year_return


# ================================================================
# step4_实盘模拟.py
# ================================================================
def simulate_performance(conf: BacktestConfig, show_plot=True, extra_equities=None):
    """
    模拟投资组合的表现，生成资金曲线以跟踪组合收益变化。

    参数:
    conf (BacktestConfig): 回测配置
    select_results (DataFrame): 选股结果数据
    show_plot (bool): 是否显示回测结果图表

    返回:
    None
    """
    # ====================================================================================================
    # 1. 聚合选股结果中的权重
    # ====================================================================================================
    s_time = time.time()
    select_results = pd.read_pickle(conf.select_results_path)

    logger.debug("🔀 持仓周期权重聚合...")
    symbols = sorted(select_results["股票代码"].unique())
    period_ratio_df = {}
    for (strategy, period, reb_time), grp_df in select_results.groupby(["策略", "持仓周期", "换仓时间"], observed=True):
        pivot_table_df = grp_df.pivot_table(
            index="选股日期", columns="股票代码", values="目标资金占比", aggfunc="sum", fill_value=0, observed=False
        )
        period_ratio_df[(strategy, period, reb_time)] = pivot_table_df

    logger.debug(f"👌 权重聚合完成，耗时：{time.time() - s_time:.3f}秒")

    # ====================================================================================================
    # 2. 对数据进行处理
    # ====================================================================================================
    is_hour = conf.use_stock_timing
    max_dt = conf.load_index_hour_data()["交易日期"].max() if is_hour else conf.load_index_data()["交易日期"].max()
    max_dt_str = max_dt.strftime("%Y-%m-%d %H:%M:%S") if is_hour else max_dt.strftime("%Y-%m-%d")
    # 防御性编程
    if len(period_ratio_df) == 0:
        logger.warning("权重聚合结果为空，请检查选股结果")
        min_ratio_date_str = conf.start_date
        max_ratio_date_str = conf.end_date or max_dt_str
    else:
        min_ratio_dt = min(ratio_df.index.min() for ratio_df in period_ratio_df.values())
        max_ratio_dt = max(ratio_df.index.max() for ratio_df in period_ratio_df.values())
        if is_hour:
            min_ratio_date_str = min_ratio_dt.strftime("%Y-%m-%d %H:%M:%S")
            max_ratio_date_str = max_ratio_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            min_ratio_date_str = min_ratio_dt.strftime("%Y-%m-%d")
            max_ratio_date_str = max_ratio_dt.strftime("%Y-%m-%d")

    # 确定回测区间
    conf.start_date = max(conf.start_date, min_ratio_date_str)
    conf.end_date = conf.end_date or max_dt_str  # 如果没有设置结束日期，就默认到指数最新的交易日
    logger.debug(
        f"🗓️ 回测模拟区间:{conf.start_date}~{conf.end_date}，" f"选股结果区间:{min_ratio_date_str}~{max_ratio_date_str}"
    )

    period_offset = conf.load_period_offset()

    # 对于交易日可能为空的周期进行重新填充
    for (strategy, period, reb_time), df_stock_ratio in period_ratio_df.items():
        rebalance_dates = pd.concat(
            [period_offset.groupby(period)["交易日期"].last(), pd.Series(df_stock_ratio.index)]
        ).unique()

        # 对于交易日可能为空的周期进行重新填充，不存在的 symbol 填充 ratio 为 0
        new_df_stock_ratio = df_stock_ratio.reindex(index=rebalance_dates, columns=symbols, fill_value=0).sort_index()

        period_ratio_df[(strategy, period, reb_time)] = new_df_stock_ratio

    # ====================================================================================================
    # 3. 计算资金曲线
    # ====================================================================================================
    pivot_dict_stock = pd.read_pickle(conf.get_runtime_folder() / "全部股票行情pivot.pkl")
    logger.info("开始模拟日线交易...")

    # 计算资金曲线及收益数据
    daily_account_df, hourly_account_df, period_ratio_hour_df = calc_equity(
        conf, pivot_dict_stock, period_ratio_df, symbols, select_results=select_results
    )

    # 日线评价
    daily_result = evaluate_and_save_performance(
        conf=conf,
        account_df=daily_account_df,
        period_ratio_df=period_ratio_df,
        pivot_dict_stock=pivot_dict_stock,
        symbols=symbols,
        select_results=select_results,
        period_name="日线",
        show_plot=show_plot,
        extra_equities=extra_equities,
    )

    # 小时线评价
    if conf.use_stock_timing and period_ratio_hour_df is not None:
        hourly_result = evaluate_and_save_performance(
            conf=conf,
            account_df=hourly_account_df,
            period_ratio_df=period_ratio_hour_df,
            pivot_dict_stock=pivot_dict_stock,
            symbols=symbols,
            select_results=select_results,
            period_name="小时",
            show_plot=show_plot,
            extra_equities=extra_equities,
        )

    return conf.report


def _evaluate_and_save_performance(conf: BacktestConfig, account_df: pd.DataFrame, period_name="日线", suffix=""):
    rtn, year_return, month_return, quarter_return = strategy_evaluate(
        account_df,
        net_col="净值",
        pct_col="涨跌幅",
        commission_rate=conf.c_rate,
    )

    # 保存性能数据
    if period_name == "日线":
        conf.set_report(rtn.T)
        prefix = ""
    else:
        prefix = f"{period_name}_"

    save_performance(
        conf,
        **{
            f"{prefix}资金曲线{suffix}": account_df,
            f"{prefix}策略评价{suffix}": rtn,
            f"{prefix}年度账户收益{suffix}": year_return,
            f"{prefix}季度账户收益{suffix}": quarter_return,
            f"{prefix}月度账户收益{suffix}": month_return,
        },
    )
    return rtn, year_return, month_return, quarter_return


def evaluate_and_save_performance(
    conf: BacktestConfig,
    account_df: pd.DataFrame,
    period_ratio_df: Dict[tuple, pd.DataFrame],
    pivot_dict_stock: Dict[str, pd.DataFrame],
    symbols,
    select_results,
    period_name="日线",
    show_plot=False,
    extra_equities=None,
):
    """
    对资金曲线进行策略评价、保存性能数据，并根据配置执行再择时。
    """
    if account_df is None or account_df.empty:
        logger.warning(f"{period_name} 账户数据为空，跳过评价")
        return None

    rtn, year_return, month_return, quarter_return = _evaluate_and_save_performance(
        conf, account_df, period_name=period_name
    )

    # 构建返回结果
    result = {
        "account_df": account_df,
        "rtn": rtn,
        "year_return": year_return,
        "month_return": month_return,
        "quarter_return": quarter_return,
    }

    # 检查是否启用择时信号
    has_equity_signal = isinstance(conf.equity_timing, EquityTiming)

    if has_equity_signal:
        logger.info(f"开始计算 {period_name} 资金曲线再择时...")
        account_df2, rtn2, year_return2 = simu_equity_timing(conf, pivot_dict_stock, period_ratio_df, symbols)

        result["timing"] = {"account_df": account_df2, "rtn": rtn2, "year_return": year_return2}

        if show_plot:
            show_performance_plot(
                conf,
                select_results,
                account_df2,
                rtn2,
                year_return2,
                title_prefix=f"{period_name}-再择时-" if period_name != "日线" else "再择时-",
                pre_timing_equity=account_df["净值"],
                extra_equities=extra_equities or {},
            )
    elif show_plot:
        show_performance_plot(
            conf,
            select_results,
            account_df,
            rtn,
            year_return,
            title_prefix=f"{period_name}-" if period_name != "日线" else "",
            extra_equities=extra_equities or {},
        )

    return result
