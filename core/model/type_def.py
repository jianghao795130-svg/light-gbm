

import numba as nb
from numba.experimental import jitclass

# 定义股票交易所类型的常量
# 北交所(理应拉黑) bjxxxxxx
BSE_MAIN = 0

# 上交所主板 sh60xxxx
SSE_MAIN = 1

# 上交所科创板 sh68xxxx
SSE_STAR = 2

# 深交所主板 sz00xxxx
SZSE_MAIN = 3

# 深交所创业板 sz30xxxx
SZSE_CHINEXT = 4

# 价格序列数据
price_array = [
    "open",
    "0935",
    "0940",
    "0945",
    "0950",
    "0955",
    "1000",
    "1005",
    "1010",
    "1015",
    "1020",
    "1025",
    "1030",
    "1035",
    "1040",
    "1045",
    "1050",
    "1055",
    "1100",
    "1105",
    "1110",
    "1115",
    "1120",
    "1125",
    "1130",
    "1305",
    "1310",
    "1315",
    "1320",
    "1325",
    "1330",
    "1335",
    "1340",
    "1345",
    "1350",
    "1355",
    "1400",
    "1405",
    "1410",
    "1415",
    "1420",
    "1425",
    "1430",
    "1435",
    "1440",
    "1445",
    "1450",
    "1455",
    "close",
]


@jitclass
class StockMarketData:
    """
    股票市场数据类，用于存储和管理股票市场的历史数据
    """

    # 交易日零点时间戳，单位秒
    candle_begin_ts: nb.int64[:]

    # 前收盘价数据，二维数组，第一维表示股票，第二维表示时间
    pre_cl: nb.float64[:, :]

    # 跌停价数据，二维数组，第一维表示股票，第二维表示时间
    dieting: nb.float64[:, :]

    # 价格数据，包含开盘价、日内不同时间点价格和收盘价
    prices: nb.types.UniTuple(nb.float64[:, :], 49)

    # 股票所属交易所类型数组，表示每只股票对应的交易所类型（如 BSE_MAIN, SSE_MAIN 等）
    types: nb.int16[:]

    def __init__(self, candle_begin_ts, op, cl, pre_cl, dieting, types, hour_prices=()):
        """
        初始化股票市场数据

        :param candle_begin_ts: 交易日零点时间戳数组，单位秒
        :param op: 开盘价数据，二维数组，第一维表示股票，第二维表示时间
        :param cl: 收盘价数据，二维数组，第一维表示股票，第二维表示时间
        :param pre_cl: 前收盘价数据，二维数组，第一维表示股票，第二维表示时间
        :param dieting: 跌停价，二维数组，第一维表示股票，第二维表示时间
        :param types: 股票所属交易所类型数组，表示每只股票对应的交易所类型
        :param hour_prices: 日内不同时间点价格数据，包含 3 个二维数组的元组，分别表示不同时间点的价格
        """
        # 交易日零点时间戳
        self.candle_begin_ts = candle_begin_ts
        self.prices = (
            op,
            hour_prices[0],
            hour_prices[1],
            hour_prices[2],
            hour_prices[3],
            hour_prices[4],
            hour_prices[5],
            hour_prices[6],
            hour_prices[7],
            hour_prices[8],
            hour_prices[9],
            hour_prices[10],
            hour_prices[11],
            hour_prices[12],
            hour_prices[13],
            hour_prices[14],
            hour_prices[15],
            hour_prices[16],
            hour_prices[17],
            hour_prices[18],
            hour_prices[19],
            hour_prices[20],
            hour_prices[21],
            hour_prices[22],
            hour_prices[23],
            hour_prices[24],
            hour_prices[25],
            hour_prices[26],
            hour_prices[27],
            hour_prices[28],
            hour_prices[29],
            hour_prices[30],
            hour_prices[31],
            hour_prices[32],
            hour_prices[33],
            hour_prices[34],
            hour_prices[35],
            hour_prices[36],
            hour_prices[37],
            hour_prices[38],
            hour_prices[39],
            hour_prices[40],
            hour_prices[41],
            hour_prices[42],
            hour_prices[43],
            hour_prices[44],
            hour_prices[45],
            hour_prices[46],
            cl,
        )

        # 前收盘价数据
        self.pre_cl = pre_cl

        # 跌停价格数据
        self.dieting = dieting

        # 股票所属交易所类型
        self.types = types


@jitclass
class SimuParams:
    """
    模拟参数类，用于定义模拟交易中的初始资金、交易佣金和印花税等参数
    """

    # 初始资金，单位人民币元
    init_cash: float

    # 券商佣金费率，表示每次交易（买入或卖出）时按交易金额收取的佣金比例
    commission_rate: float

    # 印花税率，表示卖出股票时按卖出金额收取的税率
    stamp_tax_rate: float

    def __init__(self, init_cash, commission_rate, stamp_tax_rate):
        """
        初始化模拟参数

        :param init_cash: 初始资金，单位人民币元
        :param commission_rate: 券商佣金费率，表示每次交易（买入或卖出）时按交易金额收取的佣金比例
        :param stamp_tax_rate: 印花税率，表示卖出股票时按卖出金额收取的税率
        """
        # 设置初始资金
        self.init_cash = init_cash

        # 设置券商佣金费率
        self.commission_rate = commission_rate

        # 设置印花税率
        self.stamp_tax_rate = stamp_tax_rate


@jitclass
class AdjustRatios:
    """
    调仓参数类，用于定义调仓操作的日期、目标权重以及买卖价格索引
    """

    # 调仓日期数组，存储每次调仓的时间戳（单位：秒）
    adj_dts: nb.int64[:]

    # 目标权重矩阵，二维数组，第一维表示调仓日期，第二维表示每个股票的目标权重
    ratios: nb.float64[:, :]

    # 卖出价格索引，表示在价格数据中使用哪个时间点的价格作为卖出价格
    sp_idx: nb.int8

    # 买入价格索引，表示在价格数据中使用哪个时间点的价格作为买入价格
    bp_idx: nb.int8

    def __init__(self, adj_dts, ratios, reb_time):
        """
        初始化调仓参数

        :param adj_dts: 调仓日期数组，存储每次调仓的时间戳（单位：秒）
        :param ratios: 目标权重矩阵，二维数组，第一维表示调仓日期，第二维表示每个股票的目标权重
        :param reb_time: 买卖价格索引元组，格式为 (卖出价格索引, 买入价格索引)
        """
        # 设置调仓日期
        self.adj_dts = adj_dts

        # 设置目标权重矩阵
        self.ratios = ratios

        # 设置卖出价格索引和买入价格索引
        self.sp_idx, self.bp_idx = reb_time


def get_symbol_type(symbol: str) -> int:
    """
    根据股票代码判断其所属的交易所类型

    :param symbol: 股票代码，格式为交易所代码 + 股票编号(例如 sh600000, sz000001, bj430090)
    :return: 交易所类型常量（如 BSE_MAIN, SSE_MAIN, SSE_STAR, SZSE_MAIN, SZSE_CHINEXT)
    :raises ValueError: 如果股票代码不符合已知的交易所代码规则，抛出 ValueError
    """
    # 判断是否为北交所股票（代码以 'bj' 开头）
    if symbol.startswith("bj"):
        return BSE_MAIN  # 北交所

    # 判断是否为上交所股票（代码以 'sh' 开头）
    if symbol.startswith("sh"):
        # 判断是否为科创板股票（代码以 'sh68' 开头）
        if symbol.startswith("sh68"):
            return SSE_STAR  # 科创板
        else:
            return SSE_MAIN  # 上交所主板

    # 判断是否为深交所股票（代码以 'sz' 开头）
    if symbol.startswith("sz"):
        # 判断是否为深交所主板股票（代码以 'sz0' 开头）
        if symbol.startswith("sz0"):
            return SZSE_MAIN  # 深交所主板
        else:
            return SZSE_CHINEXT  # 深交所创业板

    # 如果股票代码不符合已知规则，抛出 ValueError
    raise ValueError(f"Unknown stock {symbol}")
