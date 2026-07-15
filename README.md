# light-gbm

A 股截面多因子与 LightGBM 机器学习选股研究框架。项目覆盖从行情预处理、因子生成、标签构造、截面标准化、滚动样本外训练，到 IC 检验、TopN 组合回测、交易成本评估和特征重要性分析的完整研究链路。

本仓库定位为**量化研究与策略原型验证**，重点关注研究流程的可复现性、样本外检验、交易约束和风控口径，而不是简单展示一次性回测收益。



## 研究框架

本项目采用典型的截面选股研究范式：

```text
原始行情/财务数据
    -> 可交易样本过滤
    -> 因子计算
    -> 未来收益标签构造
    -> 截面去极值 / 中性化 / 标准化
    -> 滚动窗口训练 LightGBM
    -> 样本外预测 ml_factor
    -> IC / TopN / 换手 / 成本 / 重要性评估
```

核心研究原则：

- **样本外验证**：使用滚动窗口训练、验证和测试，避免在同一时期内训练和评估。
- **截面一致性**：标签和特征均按交易日截面处理，降低极端值和规模/行业暴露干扰。
- **可交易约束**：过滤 ST、上市天数不足、次日不可交易、次日涨停或退市等无法正常建仓样本。
- **交易成本意识**：TopN 评估显式计算买入换手、卖出换手、手续费、印花税和滑点。
- **模型稳定性检查**：使用多随机种子集成、RMSE 曲线、Valid/Train RMSE 比例和特征重要性诊断。

## 项目结构

```text
.
├── config.py                          # 回测、数据路径、策略、因子和标签参数配置
├── requirements.txt                   # Python 依赖清单
├── README.md
├── 研究流程说明.md
├── 机器学习LGBM.ipynb                  # LightGBM 完整研究 Notebook
│
├── program/                           # 可直接执行的流程入口
│   ├── step1_整理数据.py               # 行情预处理与运行缓存生成
│   ├── step2_计算因子.py               # 因子与截面因子计算
│   └── 计算因子主程序.py               # 串行执行 step1 + step2，并覆盖同名缓存
│
├── core/                              # 回测与选股框架核心
│   ├── backtest.py                    # 回测主流程调度
│   ├── data_center.py                 # 数据预处理、缓存与数据装载
│   ├── select_stock.py                # 因子计算、截面处理、选股逻辑
│   ├── equity.py                      # 资金曲线与账户模拟
│   ├── rebalance.py                   # 调仓执行逻辑
│   ├── simulator.py                   # 模拟撮合/持仓更新
│   ├── market_essentials.py           # 行情基础函数、指数与市场辅助
│   ├── fin_essentials.py              # 财务数据处理
│   ├── finance_facade.py              # 财务因子上下文封装
│   ├── factor_cache.py                # 因子缓存与增量缓存逻辑
│   ├── figure.py / perf.py            # 回测结果可视化与绩效展示
│   ├── evaluate.py                    # 评价指标计算
│   ├── model/                         # 配置对象、数据结构、信号与类型定义
│   │   ├── backtest_config.py
│   │   ├── strategy_config.py
│   │   ├── factor_config.py
│   │   ├── timing_signal.py
│   │   ├── rebalance_mode.py
│   │   ├── type_def.py
│   │   └── finance_manager*.py
│   └── utils/                         # 日志、路径、序列化、Hub、辅助函数
│       ├── log_kit.py
│       ├── path_kit.py
│       ├── serializable_kit.py
│       ├── factor_hub.py
│       ├── strategy_hub.py
│       ├── signal_hub.py
│       ├── data_hub.py
│       ├── meta_db.py
│       └── misc_kit.py
│
├── tools/                             # 机器学习研究与 Notebook 工具层
│   ├── data_prepare.py                # ML 主表加载、样本过滤、基础字段定义
│   ├── feature_engineering.py         # 标签拼接、去极值、中性化、标准化
│   ├── training_pipeline.py           # 滚动训练流程封装
│   ├── training_monitor.py            # RMSE、过拟合监控与训练摘要
│   ├── tuning.py                      # Optuna 调参
│   ├── evaluation.py                  # IC、TopN、月度收益与评估函数
│   ├── importance_analysis.py         # Gain / Split / SHAP 重要性分析
│   ├── notebook_checkpoints.py        # Notebook 中间结果 checkpoint
│   └── plot_config.py                 # 中文绘图与 Matplotlib 配置
│
├── 因子库/                             # 因子定义层
│   ├── 一级行业.py / 市值.py           # 行业与规模基准字段
│   ├── 未来n日涨跌.py                  # 监督学习标签
│   ├── factor_dig.py / ROE.py / ...   # 单因子实现
│   ├── 估值/                           # 估值因子
│   ├── 成长/                           # 成长因子
│   ├── 规模/                           # 规模与流动性因子
│   ├── 量价/                           # 量价类因子
│   ├── 短期反转/                        # 短周期反转因子
│   └── 长期反转/                        # 长周期反转因子
│
├── 策略库/                             # 策略扩展入口，预留自定义策略
│   └── __init__.py
│
├── data/                              # 本地运行缓存与回测结果，Git 忽略
├── artifacts/                         # 模型文件、训练日志、图形产物，Git 忽略
├── research_outputs/                  # Notebook 研究输出与 checkpoint，Git 忽略
├── logs/                              # 日志输出，Git 忽略
├── training_logs/                     # 训练日志，Git 忽略
└── training_logs_ascii/               # ASCII 日志镜像，Git 忽略
```


## 环境依赖

建议使用 Python 3.10+。项目依赖见 [requirements.txt](/E:/机器学习lgbm备份/requirements.txt)。

```bash
pip install -r requirements.txt
```

其中包含因子计算、回测框架、LightGBM 训练、Optuna 调参、Plotly 图表和 SHAP 解释所需的核心第三方包。

## 数据与缓存口径

数据来源入口：

```text
https://www.quantclass.cn/data/stock/category
```

本项目依赖本地数据中心中的股票行情、指数和财务数据。数据文件不随仓库分发，需要用户自行从数据源获取并放置到本地数据中心目录。

在 `config.py` 中配置本地数据中心：

```python
data_center_path = r"E:\stock_line_data"
```

运行缓存默认写入：

```text
data/运行缓存/{backtest_name}/
```

其中 `backtest_name` 是非常重要的实验标识。同名运行会覆盖同名缓存；如果需要保留多个实验版本，建议修改 `backtest_name` 或单独备份输出。

## 因子计算流程

推荐使用主程序一次性刷新数据和因子：

```powershell
python .\program\计算因子主程序.py
```

该脚本会先删除当前 `backtest_name` 对应的旧运行缓存，再执行：

```text
整理数据 -> 计算因子 -> 计算截面因子
```

也可以分步运行：

```powershell
python .\program\step1_整理数据.py
python .\program\step2_计算因子.py
```

当前标签因子为：

```text
未来n日涨跌_20
```

其计算口径为：

```text
n+1 日后的开盘价 / 明日开盘价 - 1
```

该标签仅用于机器学习监督训练和样本外评估，不应作为实盘当期可见因子参与选股。

## 机器学习研究流程

核心 Notebook：

```text
机器学习LGBM.ipynb
```

主要步骤：

1. 读取 `all_factors_kline.pkl`，构建基础样本池。
2. 执行可交易过滤：剔除 ST、上市不足 250 天、次日不可交易、次日开盘涨停、次日 ST、次日退市、一字涨停等样本。
3. 读取 `factor_未来n日涨跌_20.pkl` 作为原始收益标签，其他 `factor_*.pkl` 作为候选特征。
4. 对标签和特征按交易日截面做 MAD 去极值。
5. 对截面值做市值 + 行业中性化。
6. 对中性化残差做标准化，生成模型训练标签和特征。
7. 使用 6 年训练、12 个月验证、3 个月测试、3 个月步长的滚动窗口。
8. 每个窗口训练多个随机种子的 LightGBM 模型，并对测试集预测取平均。
9. 生成样本外 `ml_factor`，评估 RankIC、Pearson IC、TopN 收益、换手率和交易成本。
10. 输出 Gain / Split / SHAP 综合重要性，并识别低重要性噪声特征。

### 关键代码示例

以下示例对应当前项目中 Notebook 的核心研究链路，便于快速理解“标签构造 -> 特征预处理 -> 滚动训练 -> 样本外评估”的主流程。

**1. 标签与特征预处理**

该步骤对应截面去极值、中性化、标准化以及训练样本表构建。

```python
from tools.feature_engineering import (
    build_step1_dataset,
    prepare_training_dataset,
)

step1_df, meta = build_step1_dataset(
    base_df=base_df,
    data_dir=data_dir,
    mad_n=5.0,
)

train_df, train_meta = prepare_training_dataset(
    df=step1_df,
    feature_cols=meta["feature_processed_cols"],
    label_col="未来收益_预处理",
    raw_return_col="raw_return",
    original_return_col=meta["label_name"],
)

features = train_meta["feature_cols"]
```

**2. 滚动窗口训练与样本外预测**

该步骤使用时间滚动窗完成训练集、验证集、测试集切分，并输出每个测试窗口的样本外预测值。

```python
from tools.training_pipeline import run_rolling_training

prediction_df, windows, monitor = run_rolling_training(
    df=train_df,
    features=features,
    label_col="未来收益_预处理",
    train_window_months=72,
    val_window_months=12,
    test_window_months=3,
    stride_months=3,
    test_start="2021-01-01",
)
```

**3. IC 与 TopN 组合评估**

该步骤将模型输出转化为研究指标，分别检查截面预测能力与组合可实现性。

```python
from tools.evaluation import (
    evaluate_ic,
    evaluate_top_n,
    build_top_n_performance_table,
)

daily_rank_ic, daily_pearson_ic, ic_summary = evaluate_ic(prediction_df)

daily_returns_df, rebalance_returns = evaluate_top_n(
    prediction_df,
    top_n_list=[20, 50, 100],
    holding_days=20,
)

perf_df = build_top_n_performance_table(
    rebalance_returns,
    holding_days=20,
)
```

在实际研究中，建议将上述三步与 `training_monitor.py` 的 RMSE 监控、`importance_analysis.py` 的特征重要性分析以及 TopN / TopM 保留池机制联合使用，而不是仅凭单一收益曲线判断模型优劣。

## 验证与评估指标

研究评估覆盖三层：

```text
因子预测能力：RankIC、Pearson IC、ICIR、胜率
组合表现：TopN 累计收益、年化收益、年化波动、Sharpe、最大回撤
交易可实现性：买入换手、卖出换手、单期成本、扣费后净收益
```

其中 TopN 组合评估会显式区分：

- 税前收益
- 净收益
- 买入换手比例
- 卖出换手比例
- 单期交易成本

这有助于避免只看毛收益而忽略交易摩擦。

## TopN / TopM 保留池机制

为降低机器学习截面排序带来的高换手，Notebook 第九步支持双阈值组合构建：

```text
新买入：进入 TopN
继续持有：老持仓没有跌出 TopM
```

参数示例：

```python
use_topm_buffer = True
topm_multiplier_grid = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
min_net_return_retention = None
```

统一公式：

```text
TopM = ceil(TopN * multiplier)
```

寻参目标默认是：

```text
每个平均持仓周期降低的换手率
```

结果导出到：

```text
research_outputs/topm_buffer_search.csv
```

实务上不建议只按降低换手最大化来选择参数；应结合收益保留率、最大回撤、换手成本和策略容量综合判断。可通过 `min_net_return_retention` 加入收益保留约束。

## Checkpoint 机制

Notebook 已加入中间结果 checkpoint。每个关键步骤运行结束后，会保存必要变量到：

```text
research_outputs/notebook_checkpoints/
```

第一次需要按顺序运行生成 checkpoint；之后重新打开 Notebook 时，可以从某一步直接运行，前置变量会自动恢复。

这解决了 Notebook 常见的“重启内核后只能从头跑”的问题，也便于长流程研究分阶段迭代。

## 严谨性与风险提示

- **标签对齐**：当前标签为 `未来n日涨跌_20`，TopN 评估中的 `holding_days=20` 应与标签周期保持一致。
- **换仓日口径**：Notebook 中 TopN 评估默认按固定交易日间隔换仓；若要严格模拟月度换仓，应进一步按交易日历定义月初/月末换仓点。
- **未来函数边界**：`未来n日涨跌` 是监督学习标签，不应作为当期可用特征进入实盘模型。
- **样本池偏差**：过滤规则会影响样本分布，所有 IC 和组合指标都应在相同样本池口径下比较。
- **交易成本敏感性**：高换手模型对手续费、滑点和冲击成本敏感，实盘前应做成本压力测试。
- **容量约束**：TopN 结果不等于可实盘容量，需结合成交额、流动性、停牌/涨跌停和下单冲击进一步验证。
- **数据不可入库**：行情、缓存、模型和研究输出未上传 GitHub，换机器复现需重新准备数据中心并生成缓存。

## GitHub 复现说明

从 GitHub 克隆后，需要自行准备：

1. 本地行情/指数/财务数据中心。
2. `config.py` 中的数据路径。
3. Python 依赖环境。
4. 重新运行因子计算和 Notebook。

本仓库提供的是研究框架和流程代码，不包含商业数据源和已生成的训练产物。
