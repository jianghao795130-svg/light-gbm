# light-gbm

A 股多因子 + LightGBM 研究项目。项目包含股票日线数据预处理、因子计算、机器学习训练、滚动窗口预测、TopN 组合评估、换手成本评估和特征重要性分析流程。

> 注意：本仓库只保存代码、Notebook 和因子定义；行情数据、运行缓存、模型产物和研究输出不会提交到 Git。

## 项目结构

```text
.
├── config.py                  # 回测、数据路径、策略和因子配置
├── 机器学习LGBM.ipynb          # LightGBM 完整研究 Notebook
├── program/
│   ├── step1_整理数据.py       # 整理并缓存预处理行情数据
│   ├── step2_计算因子.py       # 计算因子和截面因子
│   └── 计算因子主程序.py       # 串行执行 step1 + step2，并覆盖同名运行缓存
├── core/                      # 回测框架核心逻辑
├── tools/                     # 机器学习、评估、调参、Notebook checkpoint 工具
├── 因子库/                    # 因子实现
├── 策略库/                    # 策略扩展入口
└── 研究流程说明.md             # 研究流程备忘
```

以下目录是本地运行产物，已在 `.gitignore` 中排除：

```text
data/
artifacts/
logs/
research_outputs/
training_logs/
training_logs_ascii/
```

## 环境依赖

建议使用 Python 3.10+。常用依赖包括：

```bash
pip install pandas numpy scipy matplotlib lightgbm shap optuna tqdm openpyxl
```

如果只运行因子计算和基础数据处理，通常不需要 `shap`、`optuna`；如果运行完整 Notebook，建议安装完整依赖。

## 数据路径

在 `config.py` 中配置数据中心路径：

```python
data_center_path = r"E:\stock_line_data"
```

项目默认从数据中心读取股票行情、指数和财务数据，并把运行缓存写入：

```text
data/运行缓存/{backtest_name}/
```

## 因子计算

推荐使用主程序一次性整理数据并计算因子：

```powershell
python .\program\计算因子主程序.py
```

这个脚本会先删除当前 `backtest_name` 对应的旧运行缓存，再执行：

```text
整理数据 -> 计算因子 -> 计算截面因子
```

也可以分步运行：

```powershell
python .\program\step1_整理数据.py
python .\program\step2_计算因子.py
```

## 机器学习研究流程

核心 Notebook：

```text
机器学习LGBM.ipynb
```

主要步骤：

1. 读取 `all_factors_kline.pkl`，并过滤 ST、上市天数不足、次日不可买入等样本。
2. 读取 `factor_未来n日涨跌_20.pkl` 作为收益标签，其他 `factor_*.pkl` 作为候选特征。
3. 对标签和特征按交易日做 MAD 去极值、市值 + 行业中性化、标准化。
4. 使用滚动窗口训练 LightGBM。
5. 多随机种子模型预测取平均，生成 `ml_factor`。
6. 评估 RankIC、Pearson IC、TopN 组合收益、换手率和交易成本。
7. 输出 Gain / Split / SHAP 重要性，并识别低重要性特征。

Notebook 已加入 checkpoint 机制。每个关键步骤结束后会保存中间结果到：

```text
research_outputs/notebook_checkpoints/
```

第一次需要按顺序运行生成 checkpoint；之后重新打开 Notebook 时，可以从某一步直接运行，前置变量会自动从 checkpoint 恢复。

## TopN / TopM 保留池机制

第九步 TopN 组合评估支持双阈值机制：

```text
新买入：进入 TopN
继续持有：老持仓没有跌出 TopM
```

Notebook 中可通过参数控制：

```python
use_topm_buffer = True
topm_multiplier_grid = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
min_net_return_retention = None
```

统一公式：

```text
TopM = ceil(TopN * multiplier)
```

寻参结果会导出到：

```text
research_outputs/topm_buffer_search.csv
```

## 重要提醒

- `config.py` 中的 `backtest_name` 决定运行缓存目录；同名运行会覆盖同名缓存。
- 当前标签参数是 `未来n日涨跌_20`，需要和 `holding_days=20` 的评估口径保持一致。
- 大体积数据、模型和输出没有上传到 GitHub，换机器运行时需要重新准备数据中心和重新生成缓存。
- 若要严格复现月度换仓，应确认 Notebook 中 TopN 评估的换仓日期逻辑是否和策略配置 `hold_period` 对齐。
