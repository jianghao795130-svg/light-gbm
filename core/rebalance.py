
import numba as nb
import numpy as np
from numba.experimental import jitclass

from core.model.type_def import SSE_STAR

LONG_ONLY_EQUITY_RATIO = 0.97


@nb.njit
def calc_target_lots_by_ratio(equity, prices, ratios, types):
    """
    根据目标持仓比例，计算目标持仓手数
    """
    n_syms = len(prices)

    # 初始化目标持仓
    target_positions = np.zeros(n_syms, dtype=np.int64)

    # 分配目标持仓资金
    target_equities = equity * ratios

    for idx_sym, (pr, eq, ty) in enumerate(zip(prices, target_equities, types)):
        # 分配资金小于 1 分钱，或价格无效，则不分配仓位
        if eq < 0.01 or np.isnan(pr):
            target_positions[idx_sym] = 0
            continue

        pos = int(eq / pr)

        # 科创板必须买入至少 200 股
        if ty == SSE_STAR:
            if pos >= 200:
                target_positions[idx_sym] = pos
            else:
                target_positions[idx_sym] = 0
        else:
            # 其他板块必须按 100 的整数倍
            target_positions[idx_sym] = pos - pos % 100

    return target_positions


@jitclass
class RebAlways:
    types: nb.int16[:]

    def __init__(self, types):
        self.types = types

    def calc_lots(self, equity, prices, ratios):
        """
        计算每个股票的目标手数
        :param equity: 总权益
        :param prices: 股票最新价格
        :param ratios: 股票的资金比例
        :return: tuple[股票目标仓位]
        """

        equity *= LONG_ONLY_EQUITY_RATIO  # 留一部分的资金作为缓冲

        # 直接计算股票目标持仓手数
        target_pos = calc_target_lots_by_ratio(equity, prices, ratios, self.types)

        return target_pos


# Only for test purpose, lots are not considered
@jitclass
class RebAlwaysSimple:
    types: nb.int16[:]

    def __init__(self, types):
        self.types = types

    # noinspection PyMethodMayBeStatic
    def calc_lots(self, equity, prices, ratios):
        """
        计算每个股票的目标手数
        :param equity: 总权益
        :param prices: 股票最新价格
        :param ratios: 股票的资金比例
        :return: tuple[股票目标仓位]
        """

        n_syms = len(prices)

        # 初始化目标持仓
        target_positions = np.zeros(n_syms, dtype=np.int64)

        # 分配目标持仓资金
        target_equities = equity * ratios

        mask = target_equities > 0.01

        target_positions[mask] = (target_equities[mask] / prices[mask]).astype(np.int64)

        return target_positions
