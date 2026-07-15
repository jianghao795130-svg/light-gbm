

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Callable, Union, Tuple, Set

import numpy as np
import pandas as pd

from core.model.factor_config import FactorConfig, parse_param, FilterMethod, FilterFactorConfig
from core.utils.signal_hub import get_signal_by_name


@dataclass
class StockTimingPlan:
    """聚合去重后的个股择时信号配置 + 策略级映射关系。

    Attributes:
        signals:  signal_key → (StockTiming, 需要计算该信号的股票集合)
        timing_weights:  策略名 → [(signal_key, weight), ...]
        strategy_stocks:  策略名 → 该策略选股结果中的股票集合
    """

    # key 是 StockTiming 的 signal_key，value 是 (StockTiming对象, 需要计算该信号的股票集合)。
    # 通过 signal_key 可以唯一标识一个个股择时信号配置，避免重复计算。
    signals: Dict[str, tuple]  # signal_key -> (StockTiming, Set[str])

    # 策略级的信号加权配置，key 是策略名，value 是该策略使用的信号列表，
    # 每个元素是 (signal_key, weight)，表示该信号在策略中的权重。
    # 通过这个配置可以将多个个股择时信号按权重组合成一个策略级的择时信号。
    timing_weights: Dict[str, list]  # strategy_name -> [(signal_key, weight)]

    # 策略级的股票集合，key 是策略名，value 是该策略选股结果中的股票集合。
    # 这个集合用于在生成策略级择时信号时过滤股票，确保最终的 parquet 只包含该策略选中的股票，
    # 避免不必要的计算和存储。
    strategy_stocks: Dict[str, Set[str]]  # strategy_name -> set of stock_codes

    @property
    def all_factor_cols(self) -> Set[str]:
        """所有信号配置需要的因子列（去重）。"""
        cols: Set[str] = set()
        for st, _ in self.signals.values():
            for f in st.factor_list:
                cols.add(f.col_name)
        return cols

    @property
    def all_stock_codes(self) -> Set[str]:
        """所有信号配置涉及的股票并集。"""
        codes: Set[str] = set()
        for _, stocks in self.signals.values():
            codes |= stocks
        return codes

    def aggregate_signals(self, signal_cache: dict, runtime_folder: Path, result_folder: Path) -> None:
        """按策略加权求和各信号。

        - 同步写 parquet 到 runtime_folder（供 build_hour_adj_ratios 直接读取）
        - 异步写 CSV 到 result_folder（仅供人工查看）
        """
        import threading
        from core.utils.log_kit import logger
        from core.utils.misc_kit import save_csv_safely

        csv_tasks = []  # (stg_df, path)

        for stg_name, key_weights in self.timing_weights.items():
            if not key_weights:
                continue

            parts = []
            for key, weight in key_weights:
                if key not in signal_cache:
                    continue
                df = signal_cache[key][["交易日期", "股票代码", "生效时间", "signal"]].rename(columns={"signal": key})
                parts.append((key, weight, df))

            if not parts:
                continue

            stg_df = parts[0][2]
            for _, _, df in parts[1:]:
                stg_df = pd.merge(stg_df, df, on=["交易日期", "股票代码", "生效时间"], how="outer")

            combined = np.zeros(len(stg_df))
            for key, weight, _ in parts:
                if key in stg_df.columns:
                    combined += stg_df[key].fillna(0).values * weight
            stg_df["个股择时信号"] = combined

            # 按策略过滤股票，确保 parquet 只含本策略选中的股票
            stg_stock_set = self.strategy_stocks.get(stg_name)
            if stg_stock_set:
                stg_df = stg_df.loc[stg_df["股票代码"].isin(stg_stock_set)].reset_index(drop=True)

            # 同步写宽表 parquet 到 runtime_folder（供下游 build_hour_adj_ratios 直接读取，免 pivot）
            sig_wide = stg_df.pivot_table(index="生效时间", columns="股票代码", values="个股择时信号", aggfunc="first")
            sig_wide.to_parquet(runtime_folder / f"个股择时_{stg_name}.parquet")

            csv_tasks.append((stg_df, result_folder / f"个股择时_{stg_name}.csv"))

        if not csv_tasks:
            return

        def _write_csvs():
            for df, path in csv_tasks:
                save_csv_safely(df, path, index=False)
                logger.debug(f"📄 策略级择时信号已保存: {path.name}")

        t = threading.Thread(target=_write_csvs, daemon=True)
        t.start()
        logger.debug(f"📝 {len(csv_tasks)} 个策略级 CSV 已提交后台写入")

    def __bool__(self):
        return bool(self.signals)

    def __len__(self):
        return len(self.signals)


@dataclass
class TimingSignal:
    # 信号名称
    name: str = "Signal"
    # 因子计算的股票范围 例如 100 表示复合因子前50个股票择时，0.5 表示前50%的股票择时，0 表示全部股票择时（不建议）；
    limit: Union[int, float] = 100
    # 信号因子
    factor_list: List[FactorConfig] = field(default_factory=list)
    # 信号参数
    params: Union[tuple, float, int, str] = ()
    # 信号时间
    signal_time: str = "close"
    # 回溯多久的历史数据，因子rolling越大，参数越大，速度也会越慢
    recall_days: int = 256

    # **fallback仓位**，当择时信号因为各种原因在换仓前无法执行的时候，比如计算超时，会使用这个仓位。
    # 1 表示到了换仓时间，没有算出来就全部出击。也可以是 0 表示不出击。也可以选 0.5 表示出击一半仓位。
    # 默认是 -1 表示按照因子计算和择时逻辑走，不使用 fallback_position（目前夏普提供的策略大部分是出击）
    fallback_position: Union[int, float] = -1

    # 策略函数
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def init(cls, **config):
        config["factor_list"] = FactorConfig.parse_list(config.get("factor_list", []), False)
        config["params"] = parse_param(config.get("params", ()))
        timing_signal = cls(**config)

        if timing_signal.min_list:  # 有分钟数据的因子的话，会自动获取最大值，否则默认为close
            timing_signal.signal_time = max(timing_signal.min_list)

        return timing_signal

    @property
    def min_list(self):
        _min_list = [m for f in self.factor_list for m in f.minutes if str(m).isdigit()]
        return tuple(sorted(set(_min_list)))

    def __repr__(self) -> str:
        _str = f"{self.name}_{self.signal_time}，择时范围{self.limit}，因子{self.factor_list}，参数{self.params}"
        if self.fallback_position >= 0:
            _str += f"，fallback仓位{self.fallback_position}"
        return _str


@dataclass
class OverrideSignal(TimingSignal):
    @classmethod
    def init(cls, **config):
        config["factor_list"] = FactorConfig.parse_list(config.get("factor_list", []), False)
        config["params"] = parse_param(config.get("params", ()))
        override_signal = cls(**config)

        return override_signal

    def __repr__(self) -> str:
        _str = f"{self.name}_{self.signal_time}，择时范围{self.limit}，因子{self.factor_list}，参数{self.params}"
        if self.fallback_position >= 0:
            _str += f"，fallback仓位{self.fallback_position}"
        return _str


@dataclass
class EquityTiming:
    name: str
    params: Union[tuple, list] = ()

    # 策略函数
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def init(cls, **config) -> "EquityTiming":
        config["params"] = parse_param(config.get("params", ()))
        config["funcs"] = get_signal_by_name(config["name"])
        leverage_signal = cls(**config)

        return leverage_signal

    def get_equity_signal(self, equity_df: pd.DataFrame) -> pd.Series:
        return self.funcs["equity_signal"](equity_df, *self.params)

    def __repr__(self) -> str:
        _str = f"{self.name}，参数{self.params}"
        return _str


@dataclass
class StockTiming:
    name: str
    factor_list: Tuple[FactorConfig] = field(default_factory=tuple)  # 个股择时需要的时序因子
    params: Union[tuple, float, int, str] = ()
    weight: Union[int, float] = 1
    period: str = ""

    # 策略函数
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def parse_list(cls, stock_timing_list: list, not_weight=False) -> List:
        args_list = [factor.get("weight", 1) for factor in stock_timing_list]
        if not not_weight and not all([isinstance(x, (float, int)) for x in args_list]):
            raise ValueError("因子权重必须是float或int类型")
        all_long_factor_weight = 0 if not_weight else max(sum(args_list), 1)  # 小于1的时候不做归一化

        parsed_stock_timing_list = []
        for stock_timing_dict in stock_timing_list:
            # 将个股择时需要的过滤因子转成FilterFactorConfig对象
            factor_list = FactorConfig.parse_list(stock_timing_dict.get("factor_list", []), True)
            # param的类型需要转换为hashable的状态
            p_param = parse_param(stock_timing_dict.get("params"))
            # weight的相关处理，作为因子权重
            weight = stock_timing_dict.get("weight", 1)
            if not_weight:
                p_weight = parse_param(weight)
            else:
                p_weight = weight / all_long_factor_weight
            # 周期
            period = stock_timing_dict.get("period", "")

            _stock_timing = cls(
                name=stock_timing_dict["name"],
                factor_list=tuple(factor_list),
                params=p_param,
                weight=p_weight,
                period=period,
                funcs=get_signal_by_name(stock_timing_dict["name"]),
            )
            parsed_stock_timing_list.append(_stock_timing)
        return parsed_stock_timing_list

    @property
    def signal_key(self) -> str:
        """去重 key：name + params + factor_list 的哈希摘要。"""
        raw = f"{self.name}|{self.params}|{tuple((f.col_name, f.param) for f in self.factor_list)}"
        digest = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{self.name}_{digest}"

    def get_stock_signal(self, df: pd.DataFrame) -> pd.Series:
        return self.funcs["stock_signal"](self, df)

    def __repr__(self) -> str:
        _str = f"{self.name}，因子{self.factor_list}，参数{self.params}，权重{self.weight}，周期{self.period}"
        return _str
