import os
from pathlib import Path
from core.utils.path_kit import get_folder_path

# ====================================================================================================
# 1️⃣ 回测配置
# ====================================================================================================
# 回测数据的起始时间。如果因子使用滚动计算方法，在回测初期因子值可能为 NaN，实际的首次交易日期可能晚于这个起始时间。
start_date = "2015-01-01"
# 回测数据的结束时间。可以设为 None，表示使用最新数据；也可以指定具体日期，例如 '2024-11-01'。
end_date = None
# 性能模式，BAL表示均衡，MAX表示快速，ECO表示节能
performance_mode = "BAL"
# Performance Mode同时会修改 `n_jobs` 和 `factor_col_limit` 的值，不过需要的话，你依旧可以在下面修改她们。
# - ECO: ♻️节能模式。n_jobs = CPU_COUNT / 4，factor_col_limit = 6
# - BAL: ⚖️均衡模式，适合大部分情况。n_jobs = CPU_COUNT / 2，factor_col_limit = 8
# - MAX: ⚡️性能模式。n_jobs = CPU_COUNT - 1，factor_col_limit = 12
# 注意，不管你怎么修改，n_jobs 最小是4，并且Windows系统下，最大是61。

# ====================================================================================================
# 2️⃣ 数据配置
# ====================================================================================================
data_center_path = r"E:\stock_line_data"  # 数据中心的文件夹
runtime_data_path = get_folder_path("data")  # 回测结果存放的的文件夹，默认为项目文件夹下的 data 文件夹，可以自定义
clean_result_folder = False  # 清理【回测结果/遍历结果】整个文件夹

# ====================================================================================================
# 3️⃣ 策略配置
# ====================================================================================================
backtest_name = "小市值facor_dig"

# 策略明细
strategy_list = [
    {
        "name": "123",
        "hold_period": "20D",
        "offset_list": [0],
        "select_num": 10,
        "cap_weight": 1,
        "rebalance_time": "open",
        "factor_list": [
            #未来收益标签
            ("未来n日涨跌", False, 20, 1),
            # 行业列
            ("一级行业", False, None, 1),
            # 市值列
            ("市值", True, None, 1),
            #波动因子
            ("收盘价STD", True, 20, 1),
            ("成交额缩波因子", True, (3, 20), 1),
            ("当前回撤", True, 20, 1),
            ("量价.N日夏普比", False, 20, 1),
            # 流动性因子
            ("短期大户净买入", False, 20, 1),
            ("换手率", True, 20, 1),
            ("成交额Mean", False, 20, 1),
            # 动量因子
            ("近期涨跌幅", False, 20, 1),
            # 规模因子
            ("规模.成交额Std", True, 5,1),
            ("规模.Alpha95V2", True, 10, 1),
            ("规模.G144", True, 20, 1),
            ("规模.资金流买入占比", False, "非机构", 1),
            #反转
            ("长期反转.Mm", True, 20, 1),
            ("短期反转.CoppAtr", True, 20, 1),
            ("长期反转.Wc", True, 20, 1),
            ("短期反转.MakV2", True,20, 1),
            ('长期反转.Po', True, 20, 1),
            ("factor_dig", False, 20, 1),
            #估值因子
            ("估值.EP", False, "单季", 1),
            ("估值.HML因子", False, "单季", 1),
            ("估值.SP", False, "单季", 1),
            ("估值.企业价值倍数", False, "单季", 1),
            ("估值.捡烟蒂因子", False, "单季", 1),
            #成长因子
            ('成长.EPS相关因子', True, '单季环比', 1),
            ('成长.营业收入单季同比增速', True, '单季', 1),
            ('每股收益', False, '单季', 1),
            ('成长.毛利率季度增加',True, '单季', 1),
            ('成长.归母净利润同比增速', False, 60, 1),
            ("ROE", False, "单季", 1),
            ("规模.G144", False, "单季", 1),

        ]
    }
]

# 排除板块，比如 cyb 表示创业板，kcb 表示科创板，bj 表示北交所
excluded_boards = ["bj"]
# excluded_boards = ["cyb", "kcb", "bj"]  # 同时过滤创业板和科创板和北交所

# 上市至今交易天数
days_listed = 250
# 整体资金使用率，也就是用于模拟的资金比例
total_cap_usage = 100 / 100  # 100%表示用全部的资金买入，如果是0.5就是使用一半的资金来模拟交易

# ====================================================================================================
# 4️⃣ 模拟交易配置
# 以下参数几乎不需要改动
# ====================================================================================================
initial_cash = 1_0000_0000  # 初始资金10w
# initial_cash = 1_0000_0000  # 初始资金10w
# 手续费
c_rate = 1 / 10000
# 印花税
t_rate = 1 / 1000

# ====================================================================================================
# 5️⃣ 其他配置
# 以下参数几乎不需要改动
# ====================================================================================================
match performance_mode:
    case "BAL" | "EQUAL": # 均衡
        n_jobs = int(os.cpu_count() / 2)
        factor_col_limit = 8
    case "MAX" | "PERFORMANCE": # 快速
        n_jobs = int(os.cpu_count() - 1)
        factor_col_limit = 12
    case "ECO" | "ECONOMY": # 节能
        n_jobs = int(os.cpu_count() / 4)
        factor_col_limit = 6
    case _:
        raise ValueError(f"不支持的性能模式：{performance_mode}")

# 限制进程数量范围是4->61，限制最少位4，不然要地久天长了
n_jobs = max(n_jobs, 4)
# windows系统下，最大进程数量是61
if os.name == "nt":
    n_jobs = min(n_jobs, 61)

# ⚠️ 友情提示：
# 如果你对performance_mode的设置不满意，可以手动修改n_jobs和factor_col_limit的值。
# 如果你手动修改了n_jobs和factor_col_limit的值，请注意，performance_mode的设置会失效。
# n_jobs = 4
# ==== factor_col_limit 介绍 ====
# factor_col_limit = 8  # 内存优化选项，一次性计算多少列因子。8 是16G电脑的推荐配置
# - 数字越大，计算速度越快，但同时内存占用也会增加。
# - 该数字是在 "因子数量 * 参数数量" 的基础上进行优化的。
#   - 例如，当你遍历 200 个因子，每个因子有 10 个参数，总共生成 2000 列因子。
#   - 如果 `factor_col_limit` 设置为 64，则计算会拆分为 ceil(2000 / 64) = 32 个批次，每次最多处理 64 列因子。
# - 以上数据仅供参考，具体值会根据机器配置、策略复杂性、回测周期等有所不同。建议大家根据实际情况，逐步测试自己机器的性能极限，找到适合的最优值。

# =====参数预检查=====
runtime_folder = get_folder_path(runtime_data_path, "运行缓存")
if not Path(data_center_path).exists():
    print(f"数据中心路径不存在：{data_center_path}，请检查配置或联系助教，程序退出")
    exit()

# 强制转换为 Path 对象
data_center_path = Path(data_center_path)
runtime_data_path = Path(runtime_data_path)
