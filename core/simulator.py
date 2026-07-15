

import numba as nb
import numpy as np
from numba.experimental import jitclass


@jitclass
class Simulator:
    cash: float  # 账户现金余额, 单位人民币元
    pos_values: nb.float64[:]  # 仓位价值，单位人民币元

    commission_rate: float  # 交易佣金费率，表示每次买入或卖出股票时，按交易金额收取的佣金比例
    stamp_tax_rate: float  # 印花税率，仅在卖出股票时收取，按卖出金额的比例计算

    last_prices: nb.float64[:]  # 最新价格

    def __init__(self, init_capital, commission_rate, stamp_tax_rate, init_pos_values):
        """
        初始化模拟器

        :param init_capital: 初始资金，单位人民币元
        :param commission_rate: 交易佣金费率，表示每次买入或卖出股票时，按交易金额收取的佣金比例（例如 0.0003 表示万分之三）
        :param stamp_tax_rate: 印花税率，仅在卖出股票时收取，按卖出金额的比例计算（例如 0.001 表示千分之一）
        :param init_pos_values: 初始仓位价值，表示每个股票的初始持仓价值，单位人民币元
        """
        self.cash = init_capital  # 初始化现金余额为初始资金
        self.commission_rate = commission_rate  # 设置交易佣金费率
        self.stamp_tax_rate = stamp_tax_rate  # 设置印花税率

        n = len(init_pos_values)  # 获取初始仓位价值的数量

        # 初始化仓位价值数组，长度为n，数据类型为float64
        self.pos_values = np.zeros(n, dtype=np.float64)
        self.pos_values[:] = init_pos_values  # 将初始仓位价值赋值给仓位价值数组

        # 初始化最新价格数组，长度为n，数据类型为float64
        self.last_prices = np.zeros(n, dtype=np.float64)

    def deposit(self, cash):
        """
        模拟向账户中存入资金

        :param cash: 入金金额，单位人民币元。必须为非负数。
        """
        # 检查入金金额是否为负数，如果是负数则直接返回，不进行任何操作
        if cash < 0:
            return
        # 将入金金额加到当前现金余额中
        self.cash += cash

    def withdraw(self, cash):
        """
        模拟从账户中提取资金

        :param cash: 出金金额，单位人民币元。必须为非负数。
        :return: 实际提取的金额。如果请求的金额大于当前现金余额，则返回当前全部可用现金。
        """
        # 检查出金金额是否为负数，如果是负数则返回 0，表示未提取任何资金
        if cash < 0:
            return 0

        # 如果请求的出金金额大于当前现金余额，则将出金金额调整为当前现金余额
        if cash > self.cash:
            cash = self.cash

        # 从当前现金余额中扣除出金金额
        self.cash -= cash
        # 返回实际提取的金额
        return cash

    def withdraw_all(self):
        """
        提取账户中全部可用现金

        :return: 账户中全部可用现金的金额，单位人民币元。
        """
        # 调用 withdraw 方法，提取当前全部现金余额
        return self.withdraw(self.cash)

    def fill_last_prices(self, prices):
        """
        更新最新价格数组 `last_prices`，将非 NaN 的价格填充到对应位置

        :param prices: 当前价格数组，可能包含 NaN 值
        """
        # 创建一个布尔掩码，标记 prices 数组中非 NaN 的位置
        mask = np.logical_not(np.isnan(prices))
        # 将 prices 数组中非 NaN 的值更新到 last_prices 数组的对应位置
        self.last_prices[mask] = prices[mask]

    def settle_pos_values(self, prices):
        """
        根据当前价格计算并更新仓位价值 `pos_values`

        :param prices: 当前价格数组，可能包含 NaN 值
        """
        # 创建一个布尔掩码，标记满足以下两个条件的位置：
        # 1. pos_values 大于 1e-6（避免极小值的影响）
        # 2. prices 数组中对应位置的值不是 NaN
        mask = np.logical_and(self.pos_values > 1e-6, np.logical_not(np.isnan(prices)))

        # 根据当前价格和上一次价格的比率，更新仓位价值
        # 公式：pos_values = pos_values * (当前价格 / 上一次价格)
        self.pos_values[mask] *= prices[mask] / self.last_prices[mask]

    def settle_sellable_values(self, prices, sellable_values):
        """
        根据当前价格计算并更新仓位价值 `sellable_values`

        :param prices: 当前价格数组，可能包含 NaN 值
        :param sellable_values: 可卖量数组，可能包含 NaN 值
        """
        mask = np.logical_and(sellable_values > 1e-6, np.logical_not(np.isnan(prices)))
        sellable_values[mask] *= prices[mask] / self.last_prices[mask]
        return sellable_values

    def get_pos_value(self):
        """
        计算并返回当前所有仓位的总价值

        :return: 所有仓位的总价值，单位人民币元
        """
        # 使用 numpy 的 sum 函数对 pos_values 数组求和，得到所有仓位的总价值
        return np.sum(self.pos_values)

    def sub_pos_values(self, val, mask=None):
        self.calc_pos_values(val, mask=mask, opt="sub")

    def add_pos_values(self, val, mask=None):
        self.calc_pos_values(val, mask=mask, opt="add")

    def calc_pos_values(self, val, mask=None, opt="add"):
        """
        Numba 的一个已知限制：对 jitclass 对象的数组属性使用布尔索引进行原地修改可能不生效。
        需要在 jitclass 对象函数中修改才行
        """
        if mask is None:
            mask = np.empty(len(val), dtype=np.bool_)
            mask[:] = True
        match opt:
            case "add":
                self.pos_values[mask] += val[mask]
            case "sub":
                self.pos_values[mask] -= val[mask]

    def is_pos_and_dieting(self, dieting_prices):
        # 跌停价不能为nan
        valid_dieting = np.logical_not(np.isnan(dieting_prices))
        is_dieting = np.logical_and(valid_dieting, self.last_prices <= dieting_prices)
        return np.logical_and(self.pos_values > 1e-6, is_dieting)

    def dieting_sell_all(self, exec_prices, has_pos_and_dieting):
        """
        卖出所有持仓，将当前仓位调整为 0，并返回交易印花税和佣金

        :param exec_prices: 卖出价格数组，表示每个股票的卖出价格
        :param has_pos_and_dieting: 跌停价数组，表示每个股票的跌停价
        :return: 印花税和佣金，单位人民币元
        """
        # 创建一个全零数组，表示目标仓位为 0（即清空所有持仓）
        target_values = np.zeros(len(self.pos_values), dtype=np.float64)

        # 根据调仓价格和上一次的最新价格（开盘价），结算当前仓位价值   貌似可以不要，但是保险起见还是加上了
        self.settle_pos_values(exec_prices)
        # 计算仓位价值的变化量：目标仓位价值 - 当前仓位价值
        delta_values = target_values - self.pos_values
        # 获取没跌停的股票索引
        # 跌停的股票：
        #   target_values改成self.pos_values，防止adjust_positions函数中把self.pos_values清0。
        #   delta_values改成0，这样就不会交易（产生卖出成交额）
        # 没跌停的股票：
        #   target_values改成0，代表仓位清0。
        #   delta_values用dieting_sim.pos_values
        #   然而这两个值都已经算好了，所以不需要做任何修改
        target_values[has_pos_and_dieting] += self.pos_values[has_pos_and_dieting]
        delta_values[has_pos_and_dieting] = 0
        # 调用 adjust_positions 方法，将仓位调整为 0，并返回印花税和佣金
        stamp_tax, commission = self.adjust_positions(exec_prices, delta_values, target_values)

        # 返回印花税和佣金
        return stamp_tax, commission

    def sell_all(self, exec_prices):
        """
        卖出所有持仓，将当前仓位调整为 0，并返回交易印花税和佣金

        :param exec_prices: 卖出价格数组，表示每个股票的卖出价格
        :param has_pos_and_down: 跌停价数组，表示每个股票的跌停价
        :return: 印花税和佣金，单位人民币元
        """
        # 创建一个全零数组，表示目标仓位为 0（即清空所有持仓）
        target_pos = np.zeros(len(self.pos_values), dtype=np.float64)
        # 计算仓位价值的变化量、目标仓位价值数组
        delta_values, target_values = self.calc_delta_values(exec_prices, target_pos)
        # 调用 adjust_positions 方法，将仓位调整为 0，并返回印花税和佣金
        stamp_tax, commission = self.adjust_positions(exec_prices, delta_values, target_values)

        # 返回印花税和佣金
        return stamp_tax, commission

    def calc_delta_values(self, exec_prices, target_pos):
        """
        计算仓位价值的变化量、目标仓位价值数组

        :param exec_prices: 卖出价格数组，表示每个股票的卖出价格
        :param target_pos: 目标仓位数组，表示每个股票的目标持仓数量
        :return: 印花税和佣金，单位人民币元
        """
        # 根据调仓价格和上一次的最新价格（开盘价），结算当前仓位价值
        self.settle_pos_values(exec_prices)

        # 初始化目标仓位价值数组，长度与 pos_values 相同，数据类型为 float64
        target_values = np.zeros(len(self.pos_values), dtype=np.float64)

        # 创建一个布尔掩码，标记 目标仓位数组大于0 的位置
        mask = target_pos > 0
        # 计算目标仓位价值：目标仓位价值 = 调仓价格 * 目标持仓数量
        target_values[mask] = exec_prices[mask] * target_pos[mask]

        # 计算仓位价值的变化量：目标仓位价值 - 当前仓位价值
        delta_values = target_values - self.pos_values
        return delta_values.copy(), target_values.copy()

    def adjust_positions(self, exec_prices, delta_values, target_values):
        """
        模拟调仓操作，根据目标仓位和调仓价格调整当前仓位，并计算交易成本和更新账户状态
        1.如果上次买的股票不在本次买入范围，则会通过target_values的初始值，直接归0，代表卖出
        2.如果上次买的股票又在本次买入范围，则会通过mask，重新更改pos_values，相当于卖了又买进来

        :param exec_prices: 调仓价格数组，表示每个股票的调仓价格
        :param delta_values: 仓位价值变化量数组，表示每个股票的仓位价值变化量
        :param target_values: 目标仓位价值数组，表示每个股票的目标持仓价值
        :return: 交易佣金，单位人民币元
        """
        # 计算买入成交额：所有仓位价值变化量为正的部分之和
        buy_turnover = np.sum(delta_values[delta_values > 0])

        # 计算卖出成交额：所有仓位价值变化量为负的部分之和（取绝对值）
        sell_turnover = -np.sum(delta_values[delta_values < 0])

        # 将当前仓位价值更新为目标仓位价值
        self.pos_values[:] = target_values

        # 计算券商佣金：佣金 = (买入成交额 + 卖出成交额) * 佣金费率
        commission = (buy_turnover + sell_turnover) * self.commission_rate

        # 计算印花税：印花税 = 卖出成交额 * 印花税率
        stamp_tax = sell_turnover * self.stamp_tax_rate

        # 更新账户现金余额：
        # 现金变化 = 卖出成交额 - 买入成交额 - 佣金 - 印花税
        self.cash += sell_turnover - buy_turnover - commission - stamp_tax

        # 更新最新价格为调仓价格
        self.fill_last_prices(exec_prices)

        # 返回交易佣金
        return stamp_tax, commission

    def transfer(self, sim: "Simulator", mask):
        """
        将pos_values从一个模拟器划转到跌停模拟器
        :param sim: 需要划转的模拟器
        :param mask:
        :return:
        """
        # 跌停模拟器划转进来
        self.pos_values[mask] += sim.pos_values[mask]
        # 原模拟器划转出去
        sim.sub_pos_values(sim.pos_values, mask)
