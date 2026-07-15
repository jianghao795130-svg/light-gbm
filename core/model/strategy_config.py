

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional, Union

import numpy as np
import pandas as pd

import config
from config import days_listed, runtime_data_path
from core.model.factor_config import FilterFactorConfig, FactorConfig, CrossSectionConfig
from core.model.timing_signal import OverrideSignal, TimingSignal, StockTiming
from core.utils.log_kit import logger
from core.utils.misc_kit import pd_concat
from core.utils.signal_hub import get_signal_by_name


# fmt: off
ALLOWED_OFFSETS = [
    "2_0", "2_1", "3_0", "3_1", "3_2", "4_0", "4_1", "4_2", "4_3", "5_0", "5_1", "5_2", "5_3", "5_4", "10_0", "10_1", "10_2", "10_3", "10_4", "10_5", "10_6", "10_7", "10_8", "10_9",
    "W_0", "W_1", "W_2", "W_3", "W_4", "2W_0", "2W_1", "3W_0", "3W_1", "3W_2", "4W_0", "4W_1", "4W_2", "4W_3", "5W_0", "5W_1", "5W_2", "5W_3", "5W_4", "6W_0", "6W_1", "6W_2", "6W_3", "6W_4", "6W_5",
    "M_0", "M_-5", "W53_0",
]
# fmt: on


def calc_factor_common(df, factor_list: List[Union[FactorConfig, CrossSectionConfig]]):
    factor_val = np.zeros(df.shape[0])
    for factor_config in factor_list:
        # 计算单个因子的排名
        _rank = df.groupby("交易日期")[factor_config.col_name].rank(ascending=factor_config.is_sort_asc, method="min")
        # 将因子按照权重累加
        factor_val += _rank * factor_config.weight
    return factor_val


def filter_series_by_range(series, range_str):
    # 提取运算符和数值
    operator = range_str[:2] if range_str[:2] in [">=", "<=", "==", "!="] else range_str[0]
    value = float(range_str[len(operator) :])

    match operator:
        case ">=":
            return series >= value
        case "<=":
            return series <= value
        case "==":
            return series == value
        case "!=":
            return series != value
        case ">":
            return series > value
        case "<":
            return series < value
        case _:
            raise ValueError(f"Unsupported operator: {operator}")


def filter_common(df, filter_list):
    condition = pd.Series(True, index=df.index)

    for filter_config in filter_list:
        col_name = filter_config.col_name
        match filter_config.method.how:
            case "rank":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=False)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "pct":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=True)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "val":
                condition = condition & filter_series_by_range(df[col_name], filter_config.method.range)
            case _:
                raise ValueError(f"不支持的过滤方式：{filter_config.method.how}")

    return condition


@dataclass
class StrategyConfig:
    name: str = "Strategy"

    # 持仓周期。
    hold_period: str = "W"

    # 持仓周期的参数，比如offset
    offset_list: Tuple[int] = (0,)

    # 分批进场目标持仓占比（最大为1），默认为空，在初始化时候加载
    scalein_targets: Tuple[float] = ()
    # 默认分批进场目标持仓占比（最大为1），使用offset信息自动生成
    scalein_targets_default: Tuple[float] = ()

    # 策略权重
    cap_weight: float = 1.0

    # 原始数据的周期。
    candle_period: str = "D"

    # 选股数量。1 表示一个股票; 0.1 表示做多10%的股票
    select_num: Union[int, float] = 0.1

    # ** 换仓时间 **
    # 选股日换仓的时候，我们可以自定义换仓的时间点
    # - 'close-open'：选股日收盘前卖出，交易日开盘后买入（隔日换仓）；
    # - 'open'：交易日开盘后先卖出，交易日开盘后再买入（日内早盘）；
    # - 'close'：选股日收盘前卖出，选股日收盘前再买入（日内尾盘）；
    # 默认是 'close-open'，表示收盘买，下个开盘买，即隔日换仓
    rebalance_time: str = "close-open"

    # 选股过程中最终用于股票排名的因子名
    factor_name: str = "复合因子"

    # 因子名（和因子库目录中的文件同名），排序方式，参数，权重。
    factor_list: List[FactorConfig] = field(default_factory=list)

    # 前置过滤列表
    filter_list: List[FilterFactorConfig] = field(default_factory=list)

    # 后置过滤列表
    filter_list_post: List[FilterFactorConfig] = field(default_factory=list)

    # 截面因子列表
    cross_sections: List[CrossSectionConfig] = field(default_factory=list)

    # 策略函数
    funcs: Dict[str, Callable] = field(default_factory=dict)

    # 择时信号
    timing: Optional[TimingSignal] = None

    # 策略临时调仓信号，比如提前平仓
    override: Optional[OverrideSignal] = None

    # 个股择时信号
    stock_timing_list: List[StockTiming] = field(default_factory=list)

    # 运行过程中的文件夹，依赖于 backtest 在初始化时传入
    runtime_folder: Path = field(default_factory=Path)

    # 选股结果文件夹，依赖于 backtest 在初始化时传入
    result_folder: Path = field(default_factory=Path)

    # 需要排除的板块
    excluded_boards: List[str] = field(default_factory=list)

    @cached_property
    def period_type(self) -> str:
        return self.hold_period[-1]

    @cached_property
    def period_num(self) -> int:
        num_str = self.hold_period[:-1]

        if num_str.isnumeric():
            return int(num_str)
        else:
            return 1

    @cached_property
    def hold_period_name_list(self) -> List[str]:
        match self.period_type:
            case "D":
                period_prefix = f"{self.period_num}_"
            case "M":
                period_prefix = f"M_"
            case _:
                period_prefix = f"{self.hold_period}_"

        if period_prefix.startswith("1W_"):
            period_prefix = period_prefix.replace("1W_", "W_")
        return [f"{period_prefix}{offset}" for offset in self.offset_list]

    @cached_property
    def hold_period_name(self) -> str:
        return ",".join(self.hold_period_name_list)

    @cached_property
    def factor_columns(self) -> List[str]:
        """获取所有因子的factor_columns"""
        factor_columns = set()  # 去重

        # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
        for factor_config in self.factor_list:
            # 策略因子最终在df中的列名
            factor_columns.add(factor_config.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局
        for filter_factor in self.filter_list + self.filter_list_post:
            # 策略过滤因子最终在df中的列名
            factor_columns.add(filter_factor.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的择时所需因子信息，整理之后的列名信息，并且缓存到全局
        for timing_factor in self.timing.factor_list if self.timing is not None else ():
            # 策略择时所需因子最终在df中的列名
            factor_columns.add(timing_factor.col_name)

        # 针对当前策略的策略临时调仓信号所需因子信息，整理之后的列名信息，并且缓存到全局
        for override_factor in self.override.factor_list if self.override is not None else ():
            # 策略策略临时调仓信号所需因子最终在df中的列名
            factor_columns.add(override_factor.col_name)

        # 针对当前策略的截面所需因子信息，整理之后的列名信息，并且缓存到全局
        for cross_section in self.cross_sections:
            for section_require_factor in cross_section.factor_list:
                # 策略截面所需因子最终在df中的列名
                factor_columns.add(section_require_factor.col_name)

        # 针对当前策略的截面因子信息，整理之后的列名信息，并且缓存到全局
        for section_factor in self.cross_sections:
            # 策略截面因子最终在df中的列名
            factor_columns.add(section_factor.col_name)  # 添加到当前策略缓存信息中
        return list(factor_columns)

    @cached_property
    def hour_factor_columns(self) -> List[str]:
        """获取所有因子的factor_columns"""
        factor_columns = set()  # 去重

        # 针对当前策略的截面所需因子信息，整理之后的列名信息，并且缓存到全局
        for stock_timing in self.stock_timing_list:
            for stock_timing_factor in stock_timing.factor_list:
                # 策略截面所需因子最终在df中的列名
                factor_columns.add(stock_timing_factor.col_name)

        return list(factor_columns)

    @cached_property
    def all_factors(self) -> set[Union[FactorConfig, FilterFactorConfig, TimingSignal, CrossSectionConfig]]:
        all_factors = set()
        # 普通(时序)因子
        for factor_config in self.factor_list:
            all_factors.add(factor_config)
        # 过滤因子
        for filter_factor in self.filter_list + self.filter_list_post:
            all_factors.add(filter_factor)
        # 择时所需因子
        for timing_factor in self.timing.factor_list if self.timing else []:
            all_factors.add(timing_factor)
        # 策略临时调仓信号所需因子
        for override_factor in self.override.factor_list if self.override else []:
            all_factors.add(override_factor)
        # 截面所需因子
        for cross_section in self.cross_sections:
            for section_require_factor in cross_section.factor_list:
                all_factors.add(section_require_factor)
        # 截面因子
        for section_factor in self.cross_sections:
            all_factors.add(section_factor)
        return all_factors

    @cached_property
    def all_hour_factors(self) -> set[Union[FactorConfig, FilterFactorConfig]]:
        all_factors = set()

        # 个股择时所需因子
        for stock_timing in self.stock_timing_list:
            for stock_timing_factor in stock_timing.factor_list:
                all_factors.add(stock_timing_factor)

        return all_factors

    @classmethod
    def init(cls, index: int, **config):
        is_custom_select = "calc_select_factor" in config["funcs"]
        config["factor_list"] = FactorConfig.parse_list(config.get("factor_list", []), is_custom_select)
        config["filter_list"] = [
            FilterFactorConfig.init(filter_config) for filter_config in config.get("filter_list", [])
        ]
        config["filter_list_post"] = [
            FilterFactorConfig.init(filter_config) for filter_config in config.get("filter_list_post", [])
        ]
        # 根据offset生成默认 scalein_targets 按照分批进场，每一批的目标仓位为 (i+1)/n，且保留9位有效小数，且不超过1
        n = len(config["offset_list"])
        scalein_targets_default = []
        for i in range(n):
            val = (i + 1) / n
            # 保留9位有效小数
            val = math.floor(val * 1e9) / 1e9
            val = min(val, 1.0)
            scalein_targets_default.append(val)
        config["scalein_targets_default"] = tuple(scalein_targets_default)

        # 分批进场目标值
        scalein_targets = config.get("scalein_targets", [])
        if len(scalein_targets) == 0:
            scalein_targets = scalein_targets_default
        config["scalein_targets"] = tuple(scalein_targets)

        # 择时信号
        if timing_config := config.get("timing", {}):
            timing_config["funcs"] = get_signal_by_name(timing_config["name"])
            config["timing"] = TimingSignal.init(**timing_config)
            # 检查择时信号是否包含signal函数
            if "signal" not in timing_config["funcs"]:
                raise ValueError(
                    f"择时信号{timing_config['name']}，没有检测到`signal`函数，不支持开仓择时，请检查信号库文件"
                )

        # 策略临时调仓信号
        if override_config := config.get("override", {}):
            override_config["funcs"] = get_signal_by_name(override_config["name"])
            config["override"] = OverrideSignal.init(**override_config)
            # 检查策略临时调仓信号是否包含signal_override函数
            if "signal_override" not in override_config["funcs"]:
                raise ValueError(
                    f"策略临时调仓信号{override_config['name']}，没有检测到`signal_override`函数，不支持临时调仓，请检查信号库文件"
                )

        # 个股则时信号，not_weight按默认的False就行，因为个股择时是作用于选股之后
        config["stock_timing_list"] = StockTiming.parse_list(config.get("stock_timing_list", []))

        # 截面因子
        config["cross_sections"] = CrossSectionConfig.parse_list(config.get("cross_sections", []), is_custom_select)
        stg_conf = cls(**config)
        stg_conf.name = f"#{index}.{stg_conf.name}"

        # 检查分批进场目标值，如果不合法，则抛出异常
        stg_conf.check_scalein_targets()

        return stg_conf

    def __repr__(self):
        return (
            f"{self.cap_weight * 100:.2f}%{self.name}，周期{self.hold_period_name}，{self.select_num}个，"
            f"因子{self.factor_list}，前滤{self.filter_list}，后滤{self.filter_list_post}，截面{self.cross_sections}，{self.trade_mode_name()}。"
            f"{self.timing if self.timing else '无择时'}，{self.override if self.override else '无临时调仓'}，个股择时{self.stock_timing_list}，"
            f"{self.get_scalein_targets_str()}"
        )

    def trade_mode_name(self):
        match self.rebalance_time:
            case "close-open":
                return "隔日换仓"
            case "close":
                return "日内尾盘"
            case "open":
                return "日内早盘"
            case _:
                sell_time, buy_time = self.rebalance_time.split("-")
                return "自定义换仓({}卖{}买)".format(sell_time, buy_time)

    def max_int_param(self) -> int:
        max_int = 0
        for factor_config in self.all_factors:
            if isinstance(factor_config.param, int):
                max_int = max(max_int, factor_config.param)
        return max_int

    @staticmethod
    def filter_after_condition(period_df):
        cond6 = period_df["下日_是否交易"] == 1
        # 真实模式下，开盘涨停不代表之后不涨停，仍有交易成功的机会，所以需要用一字涨停替代。
        cond7 = (
            period_df["下日_一字涨停"] != 1 if getattr(config, "stay_real", True) else period_df["下日_开盘涨停"] != 1
        )
        cond8 = period_df["下日_是否ST"] != 1
        cond9 = period_df["下日_是否退市"] != 1
        return cond6 & cond7 & cond8 & cond9

    def filter_before_select(self, period_df):
        if "filter_stock" in self.funcs:
            return self.funcs["filter_stock"](period_df, self)

        # 通用的filter筛选
        # =删除不能交易的周期数
        # 删除月末为st状态的周期数
        cond1 = ~period_df["股票名称"].str.contains("ST", regex=False)
        # 删除月末为s状态的周期数
        cond2 = ~period_df["股票名称"].str.contains("S", regex=False)
        # 删除月末有退市风险的周期数
        cond3 = ~period_df["股票名称"].str.contains("*", regex=False)
        cond4 = ~period_df["股票名称"].str.contains("退", regex=False)
        # 删除交易天数过少的周期数
        # cond5 = period_df['交易天数'] / period_df['市场交易天数'] >= 0.8
        cond10 = period_df["上市至今交易天数"] > days_listed
        common_filter = cond1 & cond2 & cond3 & cond4 & cond10
        if not getattr(config, "stay_real", True):
            common_filter &= self.filter_after_condition(period_df)
        period_df = period_df[common_filter]
        # 只有配置了method的截面因子才是过滤截面因子
        filter_list = self.filter_list + [x for x in self.cross_sections if x.method]
        filter_condition = filter_common(period_df, filter_list)

        return period_df[filter_condition]

    def filter_after_select(self, period_df):
        filter_list = self.filter_list_post
        filter_condition = filter_common(period_df, filter_list)

        if getattr(config, "stay_real", True):
            common_filter = self.filter_after_condition(period_df)
            period_df = period_df[common_filter]
        return period_df[filter_condition]

    def calc_select_factor(self, period_df):
        if "calc_select_factor" in self.funcs:
            return self.funcs["calc_select_factor"](period_df, self)
        logger.warning(f"[{self.name}] 不在策略库中，默认使用因子加权算法")
        period_df[self.factor_name] = self.calc_select_factor_default(period_df)
        return period_df

    def calc_select_factor_default(self, period_df):
        # 需要计算权重的因子列表
        # 注：截面因子比较特殊，当args!=0时，哪怕他是过滤因子，也允许他参与权重的计算
        factor_list = self.factor_list + [
            x for x in self.cross_sections if (isinstance(x.args, float) or isinstance(x.args, int)) and x.args != 0
        ]
        return calc_factor_common(period_df, factor_list)

    def calc_stock_timing(self, stock_df) -> pd.DataFrame:
        # 对stock_timing_list进行加权平均
        stock_signal_val = np.zeros(stock_df.shape[0])
        for stock_timing in self.stock_timing_list:
            stock_signal = stock_timing.get_stock_signal(stock_df)
            stock_signal_val += stock_signal * stock_timing.weight
        stock_df["个股择时信号"] = stock_signal_val
        return stock_df

    # ================================ 计算择时信号 ================================
    def calc_signal(self, factor_df: pd.DataFrame, mode="backtest") -> pd.DataFrame:
        """
        目前是：前置过滤后的界面DataFrame
        :param factor_df: 前置过滤后的截面DataFrame
        :param mode: 运行模式
        :return: 择时信号DataFrame
        """
        # ======================== 处理选股范围 ===========================
        if self.timing.limit > 0:
            # 是否是百分比
            pct = self.timing.limit < 1
            factor_rank = factor_df.groupby("交易日期")[self.factor_name].rank(method="min", ascending=True, pct=pct)
            # 选取排名靠前的股票
            df_after_limit = factor_df[factor_rank <= self.timing.limit]
        else:  # 全部股票，stock_range小于0时，表示全部股票
            df_after_limit = factor_df

        # 如果有缓存的话拼接一下历史数据
        hist_df_path = self.get_trade_info_path().parent / f"{self.name}_择时行情数据.pkl"
        if (mode != "backtest") and hist_df_path.exists():
            # 读入历史数据
            hist_df = pd.read_pickle(hist_df_path)
            # 取出需要拼接的列
            limited_cols = [col for col in hist_df.columns if col in df_after_limit.columns]

            # 拼接历史数据和最新数据，并且保持排序（会复制一份，避免污染）
            df_after_limit = pd_concat(
                [hist_df, df_after_limit[limited_cols].copy()], ignore_index=True, sort=True, copy=False
            )

            # 按照日期、股票代码排序，自动填充factor需要的非择时期间计算的数据
            df_after_limit.sort_values(["交易日期", "股票代码"], inplace=True)
            df_after_limit.ffill(inplace=True)

            df_after_limit.drop_duplicates(["交易日期", "股票代码"], keep="last", inplace=True)
        df_after_limit.to_pickle(hist_df_path)

        signals = self.timing.funcs["signal"](self, df_after_limit)

        # signals 回测模式下，最后一行用 fallback_position 填充，如果fallback_position小于0，则不填充
        if not signals.empty and (mode == "backtest") and self.timing.fallback_position >= 0:
            signals.iloc[-1, signals.columns.get_loc("择时信号")] = self.timing.fallback_position

        # 保存实盘需要的交易信息
        self.save_time_and_stock(self.timing, df_after_limit, "择时开仓")

        return signals

    def get_today_signal_path(self, root=runtime_data_path) -> Path:
        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        if not isinstance(root, Path):
            root = Path(root)
        folder = root / "实盘信息" / today_str
        folder.mkdir(exist_ok=True, parents=True)
        return folder / f"{self.name}_信号.pkl"

    def save_today_signal(self, signal: pd.DataFrame):
        signal.to_pickle(self.get_today_signal_path())

    def get_trade_info_path(self):
        path = runtime_data_path / "实盘信息" / f"{self.name}.pkl"
        path.parent.mkdir(exist_ok=True, parents=True)  # 创建文件夹
        return path

    def save_trade_info(self, key, value):
        # 读取实盘信息
        save_path = self.get_trade_info_path()
        trade_info = self.read_trade_info()

        # 存储实盘信息
        trade_info[key] = value
        pd.to_pickle(trade_info, save_path)

    def save_time_and_stock(self, signal: Union[TimingSignal, OverrideSignal], df_after_limit: pd.DataFrame, key: str):
        """
        保存实盘需要的交易信息
        :param signal: 择时信号或策略临时调仓信号
        :param df_after_limit: 前置过滤后的截面DataFrame
        :param key: 交易信息键，"择时开仓" | "择时调仓" | "早盘择时"
        """
        stock_list = df_after_limit[df_after_limit["交易日期"] == df_after_limit["交易日期"].max()][
            "股票代码"
        ].to_list()  # 选取最后一个交易日的股票代码
        # 选取最大的分钟数据
        time_str = max(signal.min_list) if signal.min_list else "close"
        self.save_trade_info(key, [time_str, stock_list])

        # 合并早盘择时信息
        trade_info = self.read_trade_info()
        time_str1, stock_list1 = trade_info.get("择时开仓", ["0000", []])
        time_str2, stock_list2 = trade_info.get("择时调仓", ["0000", []])

        time_str = max(time_str1, time_str2)
        stock_list = list(set(stock_list1 + stock_list2))
        self.save_trade_info("早盘择时", [time_str, stock_list])

        # for key, value in self.read_trade_info().items():
        #     print(key, value)

    def read_trade_info(self, key=None):
        save_path = self.get_trade_info_path()
        trade_info = pd.read_pickle(save_path) if save_path.exists() else {}
        if key:
            return trade_info.get(key, None)
        else:
            return trade_info

    # ================================ 计算策略临时调仓信号 ================================
    def calc_override_signal(self, factor_df: pd.DataFrame, mode="backtest") -> Dict[str, pd.DataFrame]:
        """
        TODO: 计算策略临时调仓信号。需要结合策略进行参数和输入输出的调整
        :param factor_df: 前置过滤后的截面DataFrame
        :param mode: 运行模式
        :return: 策略临时调仓信号DataFrame
        """
        # ======================== 处理选股范围 ===========================
        if self.override.limit > 0:
            # 是否是百分比
            pct = self.override.limit < 1
            factor_rank = factor_df.groupby("交易日期")[self.factor_name].rank(method="min", ascending=True, pct=pct)
            # 选取排名靠前的股票
            df_after_limit = factor_df[factor_rank <= self.override.limit]
        else:  # 全部股票，stock_range小于0时，表示全部股票
            df_after_limit = factor_df

        # 如果有缓存的话拼接一下历史数据
        hist_df_path = self.get_trade_info_path().parent / f"{self.name}_策略临时调仓行情数据.pkl"
        if (mode != "backtest") and hist_df_path.exists():
            # 读入历史数据
            hist_df = pd.read_pickle(hist_df_path)
            # 取出需要拼接的列
            limited_cols = [col for col in hist_df.columns if col in df_after_limit.columns]

            # 拼接历史数据和最新数据，并且保持排序（会复制一份，避免污染）
            df_after_limit = pd_concat(
                [hist_df, df_after_limit[limited_cols].copy()], ignore_index=True, sort=True, copy=False
            )
        df_after_limit.to_pickle(hist_df_path)

        # 保存实盘需要的交易信息
        self.save_time_and_stock(self.override, df_after_limit, "择时调仓")

        # 计算策略临时调仓信号
        signal_dict = {}
        for period in self.hold_period_name_list:
            signal_df = self.override.funcs["signal_override"](self, df_after_limit)
            signal_dict[period] = signal_df
        return signal_dict

    def get_scalein_targets_str(self):
        return (
            f'自定义分批进场0%->{"->".join([f"{x*100:.0f}%" for x in self.scalein_targets])}'
            if self.scalein_targets != self.scalein_targets_default
            else "按offset数量分批"
        )

    def check_scalein_targets(self):
        """
        检查分批进场目标值是否合法，如果不合法，则抛出异常
        1. 长度必须和offset list保持一致
        2. targets后续的数值必须 >= 前序数值
        3. 最大值不能超过1
        :return:
        """
        # 1. 长度必须和offset list保持一致
        if len(self.scalein_targets) != len(self.offset_list):
            err_msg = f"分批进场目标值和持仓周期数量不一致，{self.name}，{self.offset_list}，{self.scalein_targets}"
            logger.critical(err_msg)
            raise ValueError(err_msg)

        # 2. targets后续的数值必须 >= 前序数值
        for i in range(1, len(self.scalein_targets)):
            if self.scalein_targets[i] < self.scalein_targets[i - 1]:
                err_msg = f"分批进场目标值必须递增，{self.name}，{self.scalein_targets}"
                logger.critical(err_msg)
                raise ValueError(err_msg)

        # 3. 最大值不能超过1
        if max(self.scalein_targets) > 1:
            err_msg = f"分批进场目标值最大值不能超过1，{self.name}，{self.scalein_targets}"
            logger.critical(err_msg)
            raise ValueError(err_msg)
