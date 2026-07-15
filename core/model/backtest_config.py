

import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from functools import cached_property
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Optional, List, Union, Dict

import pandas as pd

import config
from config import runtime_data_path
from core.data_center import check_extra_data, expand_daily_to_hourly
from core.market_essentials import import_index_data, check_period_offset, download_period_offset
from core.model.factor_config import CrossSectionConfig
from core.model.rebalance_mode import RebalanceMode
from core.model.strategy_config import StrategyConfig, FactorConfig
from core.model.timing_signal import EquityTiming, StockTimingPlan
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger
from core.utils.path_kit import get_folder_path
from core.utils.serializable_kit import object_to_json
from core.utils.strategy_hub import get_strategy_by_name


class BacktestConfig:
    def __init__(self, **config_dict: dict):
        # 账户名称，建议用英文，不要带有特殊符号
        self.name: str = config_dict.get("backtest_name", "默认策略回测")
        # 回测开始时间
        self.start_date: Optional[str] = config_dict.get("start_date", None)
        # 日期，为None时，代表使用到最新的数据，也可以指定日期，例如'2022-11-01'，但是指定日期
        self.end_date: Optional[str] = config_dict.get("end_date", None)
        # 是否使用小时数据
        self.use_hour_data: Optional[bool] = False

        # 策略列表，包含每个策略的详细配置
        self.strategy_list: List[StrategyConfig] = []
        self.strategy_name_list: List[str] = []
        self.strategy_list_raw: List[dict] = []
        # 初始资金默认100万
        self.initial_cash: float = config_dict.get("initial_cash", 100_0000)
        # 手续费，默认为0.002，表示万分之二
        self.c_rate: float = config_dict.get("c_rate", 1.2 / 10000)
        self.t_rate: float = config_dict.get("t_rate", 1 / 1000)  # 印花税，默认为0.001

        # 根据输入，进行一下重要中间变量的处理
        self.data_center_path: Path = Path(str(config_dict["data_center_path"]))
        # Rebalance 模式
        self.rebalance_mode: RebalanceMode = RebalanceMode.init(config_dict.get("rebalance_mode", None))
        # 整体资金使用率
        self.total_cap_usage: float = config_dict.get("total_cap_usage", 1)
        self.result_folder_name: str = config_dict.get("result_folder_name", "回测结果")

        # 如果你要diy的话，在这里设置你的数据中心路径
        # 股票日线数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-trading-data
        self.stock_data_path: Path = self.data_center_path / "stock-trading-data-pro"
        # 股票小时数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-1h-trading-data
        self.stock_hour_data_path: Path = self.data_center_path / "stock-1h-trading-data-pro"
        # 指数数据路径，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-main-index-data
        self.index_data_path: Path = self.data_center_path / "stock-main-index-data"
        # 指数小时数据路径，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-1h-index-data
        self.index_hour_data_path: Path = self.data_center_path / "stock-1h-index-data"
        # 其他的数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-fin-data-xbx
        self.fin_data_path: Path = self.data_center_path / "stock-fin-data-xbx"
        # 小时预处理数据
        self.stock_preprocess_data_path: Path = self.data_center_path / "stock-1h-trading-data-pro-2026-04-17"

        self.has_fin_data: bool = self.fin_data_path.exists()  # 是否使用财务数据

        self.factor_params_dict: dict = {}  # 缓存因子参数，用于后续的因子聚合
        self.hour_factor_params_dict: dict = {}  # 缓存因子参数，用于后续的因子聚合
        # 缓存分钟级因子参数，用于后续的因子聚合。2025-03-20添加分钟数据的支持 2025-04-17 修改格式
        self.factor_minutes_dict: dict[str, set[tuple]] = {}
        self.factor_col_name_list: List[str] = []
        self.hour_factor_col_name_list: List[str] = []
        # 截面因子信息
        self.section_factor_list: List[CrossSectionConfig] = []
        self.hold_period_name_list: List[str] = []  # 持仓周期列表
        # 缓存分钟级数据的列表，包含换仓的分钟以及因子中包含的分钟节点
        self.min_data_list = []

        self.fin_cols: list = []  # 缓存财务因子列
        self.ov_cols: list = []  # 缓存全息数据的额外字段
        self.extra_data: dict = {}  # 缓存额外数据

        # 资金曲线再择时配置，会在load_strategy中初始化
        self.equity_timing: Optional[EquityTiming] = None

        # 缓存被排除的板块
        self.excluded_boards: list = config_dict.get("excluded_boards", [])
        self.rebalance_time_list = []  # 需要用到分钟级rebalance_time时间的列表
        # 需要加载的分钟数据的级别，5分钟或者15分钟，默认为15分钟
        self.min_data_level = "1d"
        # intraday 增量因子计算时的额外 lookback 窗口（bar 数量）
        # 0: 从 trim_before 起算，不额外回看；负数按 0 处理
        self.intraday_incremental_lookback: int = int(getattr(config, "intraday_incremental_lookback", 31) or 0)

        self.info = {}  # 用于缓存数据状态
        self.report: pd.DataFrame = pd.DataFrame()  # 回测报告

        # 遍历标记，用于遍历参数的时候，标记当前是第几个遍历
        # 遍历的INDEX，0表示非遍历场景，从1、2、3、4、...开始表示是第几个循环，当然也可以赋值为具体名称
        self.iter_round: Union[int, str] = 0
        # 遍历场景下，需要在原来文件路径基础上套一层回测名，在寻找最优参数.py中设置
        self.factory_backtest_name = ""

        self.period_offset_path = self.data_center_path / "period_offset.csv"

        if self.period_offset_path.exists():
            check_period_offset(self.period_offset_path)
        else:
            download_period_offset(self.period_offset_path)

        if all((self.stock_data_path.exists(), self.fin_data_path.exists(), self.index_data_path.exists())):
            pass  # 数据检查通过
        else:
            logger.critical(
                f"""必要数据有缺失，请检查:
1. {"🟢" if self.stock_data_path.exists() else "🔴"} {self.stock_data_path}
2. {"🟢" if self.fin_data_path.exists() else "🔴"} {self.fin_data_path}
3. {"🟢" if self.index_data_path.exists() else "🔴"} {self.index_data_path}
3. {"🟢" if self.period_offset_path.exists() else "🔴"} {self.period_offset_path}"""
            )
            sys.exit()

    @property
    def factor_list_for_section(self) -> List[FactorConfig]:
        """获取截面因子所需要的时序因子名"""
        factor_set = set()
        for section_factor in self.section_factor_list:
            for section_require_factor in section_factor.factor_list:
                factor_set.add(section_require_factor)
        return list(factor_set)

    @property
    def factor_minutes_list(self) -> set[str]:
        """
        所有因子的分钟数据（不包含调仓分钟数据）
        :return: set[str]
        示例：{"0945", "0955", "1015"}
        """
        result = set()
        for value in self.factor_minutes_dict.values():
            for item in value:
                if isinstance(item, tuple):
                    result.update(item)
                else:
                    result.add(item)
        # 如果数据量太大可以改成生成式加快效率（目前只有5分钟数据，全部用上也就48个，即便算上1分钟，也才240+48，数据量太小了）
        # result = set(
        #     sub_item
        #     for value in self.factor_minutes_dict.values()
        #     for item in value
        #     for sub_item in (item if isinstance(item, tuple) else [item])
        # )
        return result

    @cached_property
    def use_stock_timing(self):
        return any(x.stock_timing_list for x in self.strategy_list)

    def desc(self):
        info_list = [
            "=" * 82,
            f"""🔵 {self.name}
→ 回测周期：{self.start_date} -> {self.end_date}
→ 初始资金：￥{self.initial_cash:,.2f}
→ 费率设置：手续费{self.c_rate * 10000:,.1f}‱, 印花税{self.t_rate * 1000:,.1f}‰
→ 数据设置:
  - 财务数据: {self.fin_cols if self.fin_cols else '∅ 否'}
  - 全息数据: {self.ov_cols if self.ov_cols else '∅ 否'}
  - 分钟数据: {self.min_data_list if self.min_data_list else '∅ 否'}，换仓时间：{self.rebalance_time_list if self.rebalance_time_list else '∅ 否'}
  - 外部数据: {list(self.extra_data.keys()) if self.extra_data else '∅ 否'}
→ 数据中心路径："{self.data_center_path}"
→ 结果路径："{self.get_result_folder()}"
→ 板块过滤：{self.excluded_boards}
→ 包含子策略：{'、'.join(self.strategy_name_list)}
→ 额外信息：{self.info}""",
        ]

        for strategy in self.strategy_list:
            info_list.append(f"  {strategy}")

        info_list.append("=" * 82 + "\n")

        return "\n".join(info_list)

    def save(self):
        # 保存成json
        try:
            input_file = get_folder_path() / "config.py"
            field_mapping = {"name": "backtest_name"}  # obj中叫backtest_title，config中叫backtest_name
            object_to_json(
                self,
                input_file=input_file,
                field_mapping=field_mapping,
                output_file=self.get_result_folder() / "config.json",
            )
            shutil.copyfile(input_file, self.get_result_folder() / "config.py")
        except:
            logger.warning("保存config失败")
        # 保存成py   保存成py就是为了方便的copy，但是想要转成可以copy的格式，实在是太麻烦了，综上，还是采用shutil直接copy。仅仅是不兼容参数遍历。
        # object_to_python_config(
        #     self,
        #     config_module="config",
        #     field_mapping=field_mapping,
        #     float_precision=4,
        #     output_file=self.get_result_folder() / "config.py",
        # )
        # 保存成pkl
        pd.to_pickle(self, self.get_result_folder() / "config.pkl")

    @staticmethod
    def _apply_time_offset(base_times: list, offset_minutes: int) -> list:
        """
        对时间列表应用分钟偏移（考虑股票交易时间，中午休市11:30-13:00）

        Args:
            base_times: 基础时间列表，格式为 ["HHMM", ...]
            offset_minutes: 偏移分钟数

        Returns:
            偏移后的时间列表
        """
        result = []
        for t in base_times:
            hour = int(t[:2])
            minute = int(t[2:]) + offset_minutes

            # 处理分钟进位
            while minute >= 60:
                hour += 1
                minute -= 60
            while minute < 0:
                hour -= 1
                minute += 60

            # 处理中午休市：如果时间落在11:30-13:00之间，跳转到下午
            if hour == 11 and minute > 30:
                # 超出11:30的部分，加到13:00上
                overflow = minute - 30
                hour = 13
                minute = overflow
            elif hour == 12:
                # 12点整个小时都在休市，跳到13点
                minute = minute  # 保持分钟数
                hour = 13

            result.append(f"{hour:02d}{minute:02d}")
        return result

    # noinspection PyUnusedLocal
    def load_strategies(self, strategy_list: Union[list, tuple], equity_timing=None):
        self.strategy_list_raw = strategy_list
        # 所有策略中的权重，当且仅当超过1的时候，才会做归一化处理
        all_cap_weight = max(sum(item.get("cap_weight", 1) for item in strategy_list), 1)
        merged_dict = defaultdict(list)  # 合并额外数据引用

        for index, stg_dict in enumerate(strategy_list):
            strategy_name = stg_dict["name"]
            strategy_info = stg_dict.pop("info", {})
            stg_dict["funcs"] = get_strategy_by_name(strategy_name)
            stg_dict["runtime_folder"] = self.get_runtime_folder()  # 运行过程中的文件夹
            stg_dict["result_folder"] = self.get_result_folder()  # 选股结果文件夹
            # 需要过滤的板块，目前是写死，所有策略统一过滤conf中的板块，之后这里可以改成不同策略过滤不同板块
            stg_dict["excluded_boards"] = self.excluded_boards
            strategy = StrategyConfig.init(index, **stg_dict)
            if strategy.cap_weight < 1e-9:
                continue
            strategy.cap_weight = strategy.cap_weight / all_cap_weight  # 加权平均策略权重

            # 缓存持仓周期的事情
            self.hold_period_name_list += strategy.hold_period_name_list

            # 判断是否有额外的调仓时间
            self.rebalance_time_list += [
                reb_time for reb_time in strategy.rebalance_time.split("-") if reb_time not in ["open", "close"]
            ]

            self.strategy_list.append(strategy)
            self.strategy_name_list.append(strategy_name)
            self.factor_col_name_list += strategy.factor_columns
            self.hour_factor_col_name_list += strategy.hour_factor_columns
            self.section_factor_list.extend(strategy.cross_sections)

            self.info = strategy_info  # 缓存策略的状态信息，主要是应对单策略配置的模式

            # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
            for _factor in strategy.all_factors | strategy.all_hour_factors:
                # 日级别factor
                if _factor in strategy.all_factors:
                    # 添加到并行计算的缓存中
                    self.factor_params_dict.setdefault(_factor.name, set()).add(_factor.param)
                # 小时级factor
                else:
                    # 添加到并行计算的缓存中
                    self.hour_factor_params_dict.setdefault(_factor.name, set()).add(_factor.param)

                # 2025-03-20添加分钟数据的支持 # 2025-04-17 修改格式
                self.factor_minutes_dict.setdefault(_factor.name, (set())).add(_factor.minutes)

                factor_ins = FactorHub.get_by_name(_factor.name)

                # 1. 合并财务因子
                self.fin_cols += getattr(factor_ins, "fin_cols", [])
                # 2. 合并全息数据的额外字段
                self.ov_cols += getattr(factor_ins, "ov_cols", [])
                # 3. 合并额外数据
                for k, v in getattr(factor_ins, "extra_data", {}).items():
                    merged_dict[k].extend(v)

        if len(self.strategy_list) == 0:
            logger.critical(f"没有读取到包含权重的策略，请检查策略配置")
            sys.exit(1)

        offset_time_list = []
        if self.use_stock_timing:
            # 加入1130是为了，资金曲线在统计的时候，统计的时间点是[1030,1130,1400,1500]，需要加入一个1130的价格。
            base_times = ["0930", "1030", "1130", "1300", "1400"]
            # fmt:off
            # base_times = [
            #     '0930', '0945', '1000', '1015', '1030', '1045', '1100', '1115', '1130',
            #     '1300', '1315', '1330', '1345', '1400', '1415', '1430', '1445'
            # ]
            # fmt:on
            offset_time_list = self._apply_time_offset(base_times, getattr(config, "stock_timing_order_price", 5))
            try:
                time_1300_index = offset_time_list.index("1300")
            except ValueError:
                time_1300_index = None
            if time_1300_index is not None:
                offset_time_list[time_1300_index] = "1305"
            self.rebalance_time_list += offset_time_list

        # 对列名进行去重
        self.fin_cols = list(sorted(set(self.fin_cols)))
        self.ov_cols = list(sorted(set(self.ov_cols)))
        self.extra_data = {key: list(set(value)) for key, value in sorted(merged_dict.items())}
        self.hold_period_name_list = list(sorted(set(self.hold_period_name_list)))
        self.factor_col_name_list = list(sorted(set(self.factor_col_name_list)))
        self.hour_factor_col_name_list = list(sorted(set(self.hour_factor_col_name_list)))
        self.min_data_list = list(sorted(self.factor_minutes_list.union(self.rebalance_time_list)))
        self.rebalance_time_list = list(sorted(set(self.rebalance_time_list)))
        self.section_factor_list = list(sorted(set(self.section_factor_list), key=lambda x: x.name))

        # 资金曲线再则时
        if equity_timing is not None:
            self.equity_timing = EquityTiming.init(**equity_timing)

        # 判断要用到什么级别的分钟数据
        if self.min_data_list:
            is_all_15min = all(minute[-2:] in ["45", "00", "15", "30"] for minute in self.min_data_list)
            temp_min_level = "15m" if is_all_15min else "5m"
            # 增加数据源检查，解决"如果设置的0945，但是没订阅15分钟数据，只订阅了5分钟数据，会报错"的问题
            if temp_min_level == "15m":
                if not check_extra_data(f"{temp_min_level}in_close")[0]:
                    temp_min_level = "5m"
            self.min_data_level = temp_min_level

            self.extra_data[f"{self.min_data_level}in_close"] = list(
                set(self.min_data_list + self.extra_data.get(f"{self.min_data_level}in_close", []))
            )

            # 使用到分钟数据，回测时间需要从2010-01-01开始
            if self.start_date < "2010-01-01":
                logger.warning(
                    f"回测使用到分钟数据，应当从2010年开始，已经自动将回测起始时间从：{self.start_date}修改为2010-01-01"
                )
                self.start_date = "2010-01-01"

            # if timing_config:
        #     self.timing = TimingSignal(**timing_config)
        # 缓存交易日偏移，按照策略自动裁切

    def load_period_offset(self, auto_cols=True) -> pd.DataFrame:
        if self.hold_period_name_list and auto_cols:
            return pd.read_csv(
                self.period_offset_path,
                encoding="gbk",
                parse_dates=["交易日期"],
                skiprows=1,
                usecols=["交易日期"] + self.hold_period_name_list,
            )
        else:
            return pd.read_csv(self.period_offset_path, encoding="gbk", parse_dates=["交易日期"], skiprows=1)

    def _load_index_data(self, index_data_path, use_range=False):
        """
        加载指数数据
        index_data (DataFrame): 合并后的指数数据
        """
        if use_range:
            return import_index_data(index_data_path / "sh000001.csv", [self.start_date, self.end_date])
        else:
            # 2025-03-25 10:48:09和夏普确认，我们回测研究时候，历史指数数据从2007年开始
            return import_index_data(index_data_path / "sh000001.csv", ["2007-01-01", None])

    def load_index_data(self, use_range=False):
        """
        加载指数数据
        index_data (DataFrame): 合并后的指数数据
        """
        return self._load_index_data(self.index_data_path, use_range=use_range)

    def load_index_hour_data(self, use_range=False):
        """
        加载小时指数数据
        index_data (DataFrame): 合并后的指数数据
        """
        return self._load_index_data(self.index_hour_data_path, use_range=use_range)

    def read_trading_dates(self, first_date, last_date):
        period_offset = self.load_period_offset()
        trading_dates = period_offset["交易日期"]

        # 支持一下开、闭区间的设定
        if first_date:
            trading_dates = trading_dates[trading_dates >= first_date]
        if last_date:
            trading_dates = trading_dates[trading_dates <= last_date]
        # trading_dates = trading_dates[(trading_dates >= first_date) & (trading_dates <= last_date)]
        return trading_dates

    def get_result_folder(self) -> Path:
        config_name = self.name
        if self.iter_round == 0:
            sub_folder = [self.result_folder_name]
        else:
            # 就算self.factory_backtest_name为""，也不影响。在非参数遍历场景下，会直接在"仓管子策略"后拼接"config_name"。在参数遍历场景下，factory_backtest_name必定有值
            sub_folder = ["遍历结果", self.factory_backtest_name]
            config_name = f"策略组_{self.iter_round}" if isinstance(self.iter_round, int) else self.iter_round
            if self.name.startswith(f"S{self.iter_round}"):
                config_name = self.name
        return get_folder_path(runtime_data_path, *tuple(sub_folder), config_name)

    def get_cache_folder(self, folder_name):
        name = self.name if self.iter_round == 0 else self.factory_backtest_name
        return get_folder_path(runtime_data_path, folder_name, name)

    def get_runtime_folder(self):
        return self.get_cache_folder("运行缓存")

    @staticmethod
    def is_intraday_cache_enabled() -> bool:
        if hasattr(config, "use_intraday_cache"):
            return bool(getattr(config, "use_intraday_cache"))
        return bool(getattr(config, "use_intraday_incremental_cache", False))

    def _get_factor_cache_timestamp(self) -> str:
        """根据预处理数据版本生成因子缓存时间戳。"""
        path = self.stock_preprocess_data_path / "timestamp.txt"
        if path.exists():
            with open(path, "r") as f:
                timestamp = f.read()

            parts = timestamp.split(",")
            try:
                date2 = datetime.strptime(parts[1], "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                date2 = datetime.strptime(parts[1], "%Y-%m-%d %H:%M:%S")
            return date2.strftime("%Y-%m-%d_%H-%M-%S")

        parquet_file = self.stock_preprocess_data_path / "sh600000.parquet"
        if parquet_file.exists():
            mtime = parquet_file.stat().st_mtime
            file_datetime = datetime.fromtimestamp(mtime)
            return file_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d_%H-%M-%S")

    def get_factor_folder(self) -> Path:
        if self.is_intraday_cache_enabled():
            return get_folder_path(runtime_data_path, "因子缓存")
        else:
            timestamp_ = self._get_factor_cache_timestamp()
            return get_folder_path(runtime_data_path, "因子缓存", timestamp_)    

    def get_intraday_factor_db_path(self) -> Path:
        """获取 intraday 因子 DuckDB 缓存数据库路径 (策略绑定，跟随数据版本)"""
        return self.get_factor_folder() / "intraday_factor_cache.duckdb"

    def get_kline_db_path(self) -> Path:
        """获取全局 K-line DuckDB 缓存路径 (跨策略共享)"""
        return self.data_center_path / "kline_cache.duckdb"

    @staticmethod
    def get_analysis_folder() -> Path:
        return get_folder_path(runtime_data_path, "分析结果")

    def get_fullname(self, as_folder_name=False):
        fullname_list = [self.name]
        for stg in self.strategy_list:
            fullname_list.append(str(stg))

        fullname = " ".join(fullname_list) + f"，初始资金￥{self.initial_cash * self.total_cap_usage:,.2f}"
        if self.equity_timing is not None:
            fullname += f"，再择时：{self.equity_timing.name}"
        return f"{self.name}" if as_folder_name else fullname

    def set_report(self, report: pd.DataFrame):
        report["param"] = self.get_fullname()
        self.report = report

    def get_strategy_config_sheet(self, with_factors=True, sep_filter=False) -> dict:
        factor_dict = {"持仓周期": [], "选股数量": []}
        for stg in self.strategy_list:
            factor_dict["持仓周期"].append(stg.hold_period_name_list)
            factor_dict["选股数量"].append(stg.select_num)

            for factor_config in stg.all_factors:
                if sep_filter:
                    factor_type = "因子" if isinstance(factor_config, FactorConfig) else "过滤"
                    _name = f"#{factor_type}-{factor_config.name}"
                else:
                    _name = f"#因子-{factor_config.name}"
                _val = factor_config.param
                if _name not in factor_dict:
                    factor_dict[_name] = []
                factor_dict[_name].append(_val)
        ret = {"策略": self.name, "策略详情": self.get_fullname()}
        if with_factors:
            ret.update(**{k: "，".join(map(str, v)) for k, v in factor_dict.items()})

        # if self.timing:
        #     ret['再择时'] = str(self.timing)
        return ret

    def get_final_equity_path(self):
        # has_timing_signal = isinstance(self.timing, TimingSignal)
        # if has_timing_signal:
        #     filename = '资金曲线_再择时.csv'
        # else:
        filename = "资金曲线.csv"
        final_equity_path = self.get_result_folder() / filename
        return final_equity_path

    def get_period_weights(self) -> Dict[str, float]:
        weight = {hold_period: 0 for hold_period in self.hold_period_name_list}
        for strategy in self.strategy_list:
            for hold_period in strategy.hold_period_name_list:
                weight[hold_period] += strategy.cap_weight / len(strategy.offset_list)
        return weight

    @classmethod
    def init_from_config(cls, backtest_name=None, load_strategy_list=True, real_trading=False) -> "BacktestConfig":
        import config

        # 提取自定义变量
        config_dict = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("__") and not isinstance(value, ModuleType)
        }
        if backtest_name:
            config_dict["backtest_name"] = backtest_name
        conf = cls(**config_dict)

        if not real_trading:
            # Rebalance 模式，实盘中禁用
            conf.rebalance_mode = RebalanceMode.init(config_dict.get("rebalance_mode", None))
        if load_strategy_list:
            # 是否自动加载策略，默认会初始化策略列表
            conf.load_strategies(config.strategy_list, getattr(config, "re_timing", None))
        return conf

    @classmethod
    def init_with_stg_config(cls, stg_config: dict, backtest_name=None, factory_info=None) -> "BacktestConfig":
        """
        通过输入的config初始化，并且自动加载对应的策略列表
        :param stg_config:
        :param backtest_name:
        :param factory_info:
        :return:
        """
        # 针对单策略模式做充分兼容
        if "strategy_list" not in stg_config:
            stg_config = dict(name=stg_config["name"], strategy_list=[stg_config])
        backtest_config = BacktestConfig.init_from_config(backtest_name, load_strategy_list=False, real_trading=False)

        # 如果factory_info不为None，则设置为iter_round。需要前置判断，避免load_strategies中文件夹路径错误
        if factory_info:
            backtest_config.iter_round = factory_info["iter_round"]
            backtest_config.factory_backtest_name = factory_info["backtest_name"]

        # 加载策略
        backtest_config.load_strategies(stg_config["strategy_list"], stg_config.get("re_timing", None))

        return backtest_config

    @property
    def select_results_path(self):
        filename = "选股结果.pkl"
        return self.get_result_folder() / filename

    def collect_signal_plan(self, select_results: pd.DataFrame) -> StockTimingPlan:
        """从所有策略中提取去重的 StockTiming 配置，以及每个配置需要计算的股票并集。"""
        signals = {}  # signal_key -> (StockTiming, set of stock_codes)
        timing_weights = {}  # strategy.name -> [(signal_key, weight), ...]
        strategy_stocks = {}  # strategy.name -> set of stock_codes

        plan_results = select_results.loc[select_results["调仓类型"].eq("计划")]

        for strategy in self.strategy_list:
            if not strategy.stock_timing_list:
                continue
            stg_stocks = set(plan_results.loc[plan_results["策略"] == strategy.name, "股票代码"].dropna())
            strategy_stocks[strategy.name] = stg_stocks
            stg_keys = []
            for stock_timing in strategy.stock_timing_list:
                key = stock_timing.signal_key
                if key in signals:
                    signals[key] = (signals[key][0], signals[key][1] | stg_stocks)
                else:
                    signals[key] = (stock_timing, stg_stocks.copy())
                stg_keys.append((key, stock_timing.weight))
            timing_weights[strategy.name] = stg_keys

        return StockTimingPlan(signals=signals, timing_weights=timing_weights, strategy_stocks=strategy_stocks)

    def build_time_mapping(self, max_date) -> dict:
        """构建小时时间戳 → 生效时间（下一小时）的映射。"""
        period_offset = self.load_period_offset()
        base_time = period_offset["交易日期"].to_frame()
        end = self.end_date if self.end_date is not None else max_date
        base_time = base_time.loc[base_time["交易日期"].gt(self.start_date) & base_time["交易日期"].le(end)]
        base_time = expand_daily_to_hourly(base_time)
        base_time["生效时间"] = base_time["交易日期"].shift(-1)
        return dict(zip(base_time["交易日期"], base_time["生效时间"]))

    def load_selected_stock_codes(self, check_stock_timing=False) -> Optional[set]:
        """检查个股择时前置条件并提取股票代码集合。

        返回 None 表示应跳过个股择时（未配置 / 文件不存在 / 结果为空）。
        """
        if not self.use_stock_timing and check_stock_timing:
            logger.info("没有策略配置个股择时，跳过")
            return None
        if not self.select_results_path.exists():
            logger.warning("选股结果文件不存在，跳过个股择时")
            return None
        select_results = pd.read_pickle(self.select_results_path)
        if select_results.empty:
            logger.warning("选股结果为空，跳过个股择时")
            return None
        return set(select_results["股票代码"].dropna().unique())


class BacktestConfigFactory:
    """
    遍历参数的时候，动态生成配置
    """

    def __init__(self, **conf):
        # ====================================================================================================
        # ** 参数遍历配置 **
        # 可以指定因子遍历的参数范围
        # ====================================================================================================
        # 存储生成好的config list和strategy list
        self.config_list: List[BacktestConfig] = []
        self.backtest_name = conf.get("backtest_name")

        if not self.backtest_name:
            self.backtest_name = f'默认策略-{datetime.now().strftime("%Y%m%dT%H%M%S")}'

        # 缓存全局配置
        self.is_use_spot = conf.get("is_use_spot", False)
        self.black_list = conf.get("black_list", set())

        # 存储生成好的config list和strategy list
        self.strategy_list: List[StrategyConfig] = []

    @property
    def result_folder(self) -> Path:
        return get_folder_path(runtime_data_path, "遍历结果", self.backtest_name)

    def generate_all_factor_config(self):
        """
        产生一个conf，拥有所有策略的因子，用于因子加速并行计算
        """
        backtest_config = BacktestConfig.init_from_config(
            self.backtest_name, load_strategy_list=False, real_trading=False
        )
        strategy_list = []
        for conf in self.config_list:
            strategy_list.extend(conf.strategy_list_raw)
        backtest_config.load_strategies(strategy_list)

        return backtest_config

    def get_name_params_sheet(self) -> pd.DataFrame:
        rows = []
        for config in self.config_list:
            rows.append(config.get_strategy_config_sheet())

        sheet = pd.DataFrame(rows)
        sheet.to_excel(self.config_list[-1].get_result_folder().parent / "策略回测参数总表.xlsx", index=False)
        return sheet

    def generate_configs_by_strategies(self, strategies: List[list], timing_strategies=None) -> List[BacktestConfig]:
        config_list = []
        iter_round = 0

        if not timing_strategies:
            timing_strategies = [None]

        self.backtest_name = self.backtest_name or "默认参数遍历"

        for strategy_list, timing_config in product(strategies, timing_strategies):
            iter_round += 1

            backtest_name = f"S{iter_round}-{self.backtest_name}"
            backtest_config = BacktestConfig.init_with_stg_config(
                # 传入的strategy_list是list of list，需要转换为list of dict，选股策略框架中，名称就是配置中的backtest_name
                {"strategy_list": strategy_list, "re_timing": timing_config, "name": self.backtest_name},
                backtest_name,
                factory_info={"iter_round": iter_round, "backtest_name": self.backtest_name},
            )

            config_list.append(backtest_config)

        self.config_list = config_list

        return config_list


def load_config(real_trading=False) -> BacktestConfig:
    if os.getenv("FUEL_CLIENT_CONFIG_PATH"):
        real_trading = True
    return BacktestConfig.init_from_config(real_trading=real_trading)


def create_factory(strategies, backtest_name=None):
    if backtest_name is None:
        from config import backtest_name
    factory = BacktestConfigFactory(backtest_name=backtest_name)
    factory.generate_configs_by_strategies(strategies)

    return factory
