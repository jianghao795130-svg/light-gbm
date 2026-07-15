from collections import defaultdict
from copy import deepcopy
from typing import Union, Dict

import numpy as np
import pandas as pd


class FinanceDataFrame(pd.DataFrame):
    """财务数据Dataframe版，暂无实际作用，仅__getitem__"""

    def __init__(
        self,
        trade_date_df: pd.DataFrame = None,
        raw_fin_df: pd.DataFrame = None,
        pivot_dict: Dict[str, pd.DataFrame] = None,
    ):
        """
        初始化财务数据容器
        :param trade_date_df: 带有全部交易日期的dataframe
        :param raw_fin_df: 原始财务数据
        :param pivot_dict: key是col，val是财务数据pivot
        """
        super().__init__()
        # 带有完整交易日期索引的df
        object.__setattr__(self, "_dt_df", trade_date_df)
        # 原始财务数据
        object.__setattr__(self, "raw_fin_df", raw_fin_df)
        # pivot财务数据
        object.__setattr__(self, "pivot_dict", pivot_dict)
        # 数据缓存，第一层是col，第二层是区分是否为原始数据
        object.__setattr__(
            self,
            "data_cache",
            defaultdict(lambda: defaultdict(pd.Series))
        )

    def generate_finance_date(self):
        """生成report_date，publish_date"""
        raw_fin_df = self.raw_fin_df.copy()
        raw_fin_df = raw_fin_df.drop_duplicates(subset="publish_date", keep="last").sort_values(
            ["publish_date", "report_quarter"], ignore_index=True
        )
        cols = ["report_date", "publish_date"]
        if raw_fin_df.empty:
            total_df = self._dt_df.copy()
            total_df[cols] = np.nan
        else:
            total_df = pd.merge_asof(
                self._dt_df, raw_fin_df, left_on=["交易日期"], right_on=["publish_date"], direction="backward"
            )
        return total_df[cols]

    def __getitem__(self, col) -> "FinanceDataSeries":
        # 如果cols属于截面数据，直接返回raw=False的全部交易日期
        if is_mul := not isinstance(col, str):
            raise TypeError("暂时不支持多列")
        if is_mul:
            # 暂时不支持多列操作，该代码无实际意义
            obj = FinanceDataFrame(**self.__dict__)
        else:
            obj = FinanceDataSeries(
                _dt_df=self._dt_df,
                _raw_fin_df=self.raw_fin_df,
                _pivot_dict=self.pivot_dict,
                data_cache=self.data_cache,
                col=col,
            )
        return obj


class FinanceDataSeries(pd.Series):

    def __init__(self, **kwargs):
        # 必须要先调用一次__init__，不然报错
        super().__init__()
        self.suffix: str = kwargs.get("suffix", "")
        self.is_pivot: bool = kwargs.get("is_pivot", False)
        # 数据缓存，第一层是col，第二层是区分是否为原始数据
        self.data_cache: Dict[str, Dict[bool, Union[pd.Series, pd.DataFrame]]] = kwargs.get(
            "data_cache", defaultdict(lambda: defaultdict(pd.Series))
        )
        """==========================计算单季、ttm所需数据=========================="""
        # 带有完整交易日期的dataframe（合并交易日期依赖的数据）
        self._dt_df: pd.DataFrame = kwargs["_dt_df"]
        # 原始财务数据（合并交易日期依赖的数据）
        self._raw_fin_df: pd.DataFrame = kwargs["_raw_fin_df"]
        # key是col，val是财务数据pivot（计算单季依赖的数据）
        self._pivot_dict: Dict[str, pd.DataFrame] = kwargs["_pivot_dict"]
        """======================================================================"""
        self.col: str = kwargs["col"]
        # 初始化缓存，把原始数据存到缓存里
        self.__init_cache()

        # 在初始化的时候就自动计算单季数值，并保存到原始self.pivot_dict中
        rs = self.__cal_fin_data(self.suffix, self.is_pivot)
        super().__init__(rs)

    def __get_in_args(self, suffix, is_pivot):
        in_args = deepcopy(self.__dict__)
        in_args["suffix"] = suffix
        in_args["is_pivot"] = is_pivot
        return in_args

    def __init_cache(self):
        """初始化缓存，把原始数据存到缓存中"""
        for col, pivot in self._pivot_dict.items():
            self.data_cache[col][True] = pivot

    def __set_cache(self, col: str, value: Union[pd.DataFrame, pd.Series], is_pivot: bool = None):
        if is_pivot is None:
            is_pivot = self.is_pivot
        self.data_cache[col][is_pivot] = value

    def __get_cache(
        self, col: str, is_pivot: bool = None, is_copy=False, is_auto=False
    ) -> Union[pd.DataFrame, pd.Series]:
        if is_pivot is None:
            is_pivot = self.is_pivot
        if (rs := self.data_cache[col][is_pivot]).empty and is_auto:
            rs = self.__cal_fin_data(col.replace(self.col, ""), is_pivot)
        if is_copy:
            return rs.copy()
        return rs

    def _get_cal_pivot(self) -> pd.Series:
        """获取计算需要的数据"""
        return self.__get_cache(self.suffix_col, True, is_copy=True)

    @staticmethod
    def __cal_quarter(pivot_df) -> pd.DataFrame:
        """
        1. 以年为单位，进行diff
        2. 每年的第一个季度，单季值==原始值
        """
        quarter_df = pivot_df.diff()
        quarter_df[quarter_df.index.quarter == 1] = pivot_df
        return quarter_df

    def __cal_fin_data(self, suffix, is_pivot) -> Union[pd.Series, pd.DataFrame]:
        # 关于为什么suffix要拆开，而不是用self.suffix：
        # 是因为当raw=True时，需要返回一个df，而如果直接套FinanceDataSeries，得到的是series，会报错
        # 所以必须要直接调用__cal_fin_data函数，而直接调用该函数，不能用self.suffix，因为默认值是""，那么就会导致不管是quarter、ttm，结果都是raw的值
        suffix_col = self.col + suffix
        # 如果缓存里没有有suffix_col数据，才需要计算
        if self.__get_cache(suffix_col, True).empty:
            # 拿到原始的pivot_dict去计算单季、ttm数据
            pivot_df = self._pivot_dict[self.col].copy()
            if suffix == "_单季":
                quarter = self.__cal_quarter(pivot_df)
                fin_data = quarter
            elif suffix == "_ttm":
                """
                1. 以近4个季度为单位(不是以年为单位)，进行rolling sum
                2. 如果sum后得到的结果，Q4为nan，那就用原始Q4填充，Q1~03不管
                """
                quarter = self.__cal_quarter(pivot_df)
                fin_data = quarter.rolling(4).sum()
                # pivot_df.index.quarter == 4得到的是单列series，isna()得到的是多维，合并会报错
                mask = fin_data.isna() & (pivot_df.index.quarter == 4)[:, np.newaxis]
                fin_data[mask] = pivot_df
            else:
                fin_data = pivot_df
            self.__set_cache(suffix_col, fin_data, True)
        # 如果需要原始pivot，则直接返回缓存中的数据
        if is_pivot:
            return self.__get_cache(suffix_col, is_pivot, is_copy=True)
        # 如果需要带交易日期的df，则丢给加工函数处理并返回
        else:
            return self.__to_trade_time(self.__get_cache(suffix_col, True), is_cache=True)

    def __to_trade_time(self, pivot_df: Union[pd.DataFrame, pd.Series], col=None, is_cache=False) -> pd.Series:
        if col is None:
            col = self.suffix_col
        # 正常情况下不会是empty，只有当"未找到财务数据"时，才会为empty
        if pivot_df.empty:
            total_df = self._dt_df.copy()
            total_df[col] = np.nan
        else:
            new_pivot = pivot_df.copy()
            # 1.为了防止stack的时候把补全季报的nan值也过滤了，所以用inf替代一下
            inf_cond1 = self._pivot_dict[f"{self.col}_is_na"]
            # 2.为了防止当计算了单季、TTM后，原始值变成了nan值，导致那一个季度的单季/TTM数据在stack后丢失
            # 所以此处把 计算后的数据为na，但是原始值不为na 的值，给填充inf
            inf_cond2 = new_pivot.isna() & self.__get_cache(self.col, True).notna()
            new_pivot[inf_cond1 | inf_cond2] = np.inf
            stacked = new_pivot.stack().reset_index()
            stacked.columns = ["report_quarter", "publish_date", col]  # 确保列名匹配
            total_df = stacked.drop_duplicates(subset="publish_date", keep="last").sort_values(
                ["publish_date", "report_quarter"], ignore_index=True
            )
            # 把临时用的inf给替换回nan
            total_df[col] = total_df[col].replace(np.inf, np.nan)
            total_df = pd.merge_asof(
                self._dt_df, total_df, left_on=["交易日期"], right_on=["publish_date"], direction="backward"
            )
        if is_cache:
            self.__set_cache(col, total_df[col])
            return self.__get_cache(col, is_copy=True)
        return total_df[col]

    @property
    def df(self) -> pd.Series:
        return pd.Series(self)

    @property
    def suffix_col(self) -> str:
        return self.col + self.suffix

    def raw(self, raw=False) -> Union["FinanceDataSeries", pd.DataFrame]:
        suffix = ""
        if raw:
            # 不能用self.suffix和is_pivot，当fin_data["xxx@xbx"]时，此时self.suffix默认是""，self.is_pivot默认是False
            return self.__cal_fin_data(suffix, True)
        return FinanceDataSeries(**self.__get_in_args(suffix, raw))

    def quarter(self, raw=False) -> Union["FinanceDataSeries", pd.DataFrame]:
        suffix = "_单季"
        if raw:
            return self.__cal_fin_data(suffix, True)
        return FinanceDataSeries(**self.__get_in_args(suffix, raw))

    def ttm(self, raw=False) -> Union["FinanceDataSeries", pd.DataFrame]:
        suffix = "_ttm"
        if raw:
            return self.__cal_fin_data(suffix, True)
        return FinanceDataSeries(**self.__get_in_args(suffix, raw))

    def qoq(self) -> pd.Series:
        data = self._get_cal_pivot()
        qoq_ = data.pct_change(fill_method=None)
        qoq_ = qoq_.mask(data.shift() < 0, -qoq_)
        return self.__to_trade_time(qoq_)

    def yoy(self) -> pd.Series:
        data = self._get_cal_pivot()
        yoy_ = data.groupby(data.index.quarter).pct_change(fill_method=None)
        yoy_ = yoy_.mask(data.shift(4) < 0, -yoy_)
        return self.__to_trade_time(yoy_)

    def q_diff(self) -> pd.Series:
        data = self._get_cal_pivot()
        q_diff_ = data.diff()
        return self.__to_trade_time(q_diff_)

    def y_diff(self) -> pd.Series:
        data = self._get_cal_pivot()
        y_diff_ = data.groupby(data.index.quarter).diff()
        return self.__to_trade_time(y_diff_)

    def last_q(self, n=1) -> pd.Series:
        data = self._get_cal_pivot()
        q = data.shift(n)
        # 必须要把原来pivot数据中的nan值给复原，因为pivot中的nan值是有重要作用的，会影响最后日期的映射
        q[data.isna()] = np.nan
        return self.__to_trade_time(q)

    def last_y(self, n=1) -> pd.Series:
        data = self._get_cal_pivot()
        y = data.groupby(data.index.quarter).shift(n)
        # 必须要把原来pivot数据中的nan值给复原，因为pivot中的nan值是有重要作用的，会影响最后日期的映射
        y[data.isna()] = np.nan
        return self.__to_trade_time(y)

    def last_q4(self, n=1) -> pd.Series:
        data = self._get_cal_pivot()
        q4 = data.groupby(data.index.year).transform("last").shift(n * 4)
        # 必须要把原来pivot数据中的nan值给复原，因为pivot中的nan值是有重要作用的，会影响最后日期的映射
        q4[data.isna()] = np.nan
        return self.__to_trade_time(q4)

    def get_q(self, n) -> pd.DataFrame:
        data = self._get_cal_pivot()
        # 创建与原data相同结构的DataFrame
        result_df = pd.DataFrame(index=data.index, columns=data.columns)

        # 遍历每个位置，创建包含前n个数据的list
        for i, row_idx in enumerate(data.index):
            for j, col in enumerate(data.columns):
                # 获取当前位置前n个数据（包括当前位置）
                start_idx = max(0, i + 1 - n)  # 确保不越界
                end_idx = i + 1
                # 这个判断必须要，不然rs结果有误，大概原因是stack函数默认dropna导致的，如果此处把原本是nan的值改成了非nan，那么dropna就会有问题。
                if np.isnan(data.iat[i, j]):
                    continue
                # 提取前n个数据
                values_list = data.iloc[start_idx:end_idx, j].tolist()

                # 如果长度不够n，在前面补充np.nan
                while len(values_list) < n:
                    values_list.insert(0, np.nan)
                result_df.at[row_idx, col] = values_list
        col = f"{self.suffix_col}_近{n}"
        rs = self.__to_trade_time(result_df, col)
        # rs[col] = rs.fillna([np.nan] * n)  # 报错，用apply替代
        rs = rs.apply(lambda x: [np.nan] * n if not isinstance(x, list) and pd.isna(x) else x)
        return rs

    def cagr(self, n) -> pd.Series:
        data = self._get_cal_pivot()

        def cal_cagr(x):
            if len(x) < 2:  # 窗口不足时返回NaN
                return np.nan
            if x[0] == 0:  # 避免除零错误
                return np.nan
            if np.isnan(x[0]) or np.isnan(x[-1]):  # 如果窗口内有任何NaN值，返回NaN
                return np.nan
            # 减1是因为当n传入2时，实际只算了一个季度的值
            return (x[-1] / x[0]) ** (4 / (n - 1)) - 1

        cagr_ = data.rolling(n).apply(cal_cagr, raw=True)
        return self.__to_trade_time(cagr_)

    def _cal_publish_date(self, col, quarter):
        """
        根据参数quarter，获取当前这一天对应的quarter季报的最早/最晚发布时间
        比如今天是0531，quarter填1，那么就是获取今年的一季报，如果今年一季报还没发，那就是获取去年的一季报
        """
        # 验证N的范围
        if quarter not in [1, 2, 3, 4]:
            raise ValueError(
                "因子使用错误！latest_publish_date/first_publish_date的参数quarter必须在[1,4]范围内，请检查因子文件"
            )

        # 判断是 最新/首次发布日期
        is_latest = col.startswith("latest_")
        need_cols = ["publish_date", "report_date", col]
        df = self._raw_fin_df.copy()[need_cols[:2]]

        # ====================================================================================================
        # 1. 根据quarter生成新的季报列，report_date2
        # ====================================================================================================
        # 季度末日期映射
        quarter_end_dates = {
            1: "-03-31",  # 第一季度：3月31日
            2: "-06-30",  # 第二季度：6月30日
            3: "-09-30",  # 第三季度：9月30日
            4: "-12-31",  # 第四季度：12月31日
        }
        # 获取年份
        year = df["report_date"].dt.year
        # 生成当年的季度结束日期
        current_year_quarter = pd.to_datetime(year.astype(str) + quarter_end_dates[quarter])
        # 生成前一年的季度结束日期
        prev_year_quarter = pd.to_datetime((year - 1).astype(str) + quarter_end_dates[quarter])
        # 使用np.where一次性选择：如果当年季度日期 > report_date，则用前一年的，否则用当年的
        df["report_date2"] = np.where(current_year_quarter > df["report_date"], prev_year_quarter, current_year_quarter)

        # ====================================================================================================
        # 2. 根据report_date2对应的publish_date，生成col列
        # ====================================================================================================
        # 创建report_date列值到索引的映射字典
        value_to_index = {val: idx for idx, val in enumerate(df["report_date"])}
        # 直接映射得到col列，省略target_index
        df[col] = df["report_date2"].map(value_to_index).map(df["publish_date"])

        # ====================================================================================================
        # 3. 对未来日期的情况进行修改
        # ====================================================================================================
        # 如果发现target_index对应的publish_date仍然超过当前行的publish_date，则用前一个publish_date
        # rp_pb_map：是一个字典，key为report_date，value为对应的list(publish_date)。
        # 也就是说，如果某个季度，发布了多次，那么该list中就有多个publish_date的值
        rp_pb_map = df.groupby("report_date")["publish_date"].apply(list).to_dict()
        # 如果col列 > pb_date，就用report_date2对应的那几行report_date对应的publish_date，作为一个list，然后在这个list中，寻找满足条件的publish_date
        mask = df[col] > df["publish_date"]
        for idx in df.loc[mask].index:
            pb_date = df.at[idx, "publish_date"]
            rp_date2 = df.at[idx, "report_date2"]
            # 注意，如果是first模式，那么可以直接用rp_pb_map[rp_date2][0]，但是如果是latest模式，就要注意不能超过发布日期，此处为了方便，统一采用循环的方式，但是用next，效率不会低多少。
            lst = reversed(rp_pb_map[rp_date2]) if is_latest else rp_pb_map[rp_date2]
            df.at[idx, col] = next((x for x in lst if x <= pb_date), np.nan)
        # 处理边界情况
        df[col].fillna(pd.to_datetime("1970-01-01"), inplace=True)

        # ====================================================================================================
        # 4. 映射到完整交易日期的df中
        # ====================================================================================================
        df = (
            # 关键提速，只保留需要的列
            df[need_cols]
            .drop_duplicates(subset=need_cols[0], keep="last")
            .sort_values(need_cols[:2], ignore_index=True)
        )
        if df.empty:
            total_df = self._dt_df.copy()
            total_df[col] = np.nan
        else:
            total_df = pd.merge_asof(
                self._dt_df, df, left_on=["交易日期"], right_on=["publish_date"], direction="backward"
            )
        return total_df[col]

    def latest_publish_date(self, quarter: int) -> pd.Series:
        return self._cal_publish_date(col="latest_publish_date", quarter=quarter)

    def first_publish_date(self, quarter: int) -> pd.Series:
        return self._cal_publish_date(col="first_publish_date", quarter=quarter)
