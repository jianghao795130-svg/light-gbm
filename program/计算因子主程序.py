import warnings
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from core.data_center import prepare_data
from core.model.backtest_config import load_config
from core.select_stock import calculate_factors, calc_cross_sections
from core.utils.log_kit import divider
from core.version import version_prompt

# ====================================================================================================
# ** 计算因子主程序 **
# 依次执行 program/step1_整理数据.py 和 program/step2_计算因子.py 的核心流程：
# 1. 整理数据
# 2. 计算因子
# 3. 计算截面因子
# ====================================================================================================
warnings.filterwarnings("ignore")
pd.set_option("expand_frame_repr", False)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)


if __name__ == "__main__":
    version_prompt()
    conf = load_config()
    print(conf.desc())

    runtime_folder = conf.get_runtime_folder()
    divider(f"{conf.name}@清理旧运行缓存", "-")
    print(f"删除运行缓存：{runtime_folder}")
    shutil.rmtree(runtime_folder, ignore_errors=True)
    runtime_folder.mkdir(parents=True, exist_ok=True)

    divider(f"{conf.name}@整理数据", "-")
    prepare_data(conf, boost=True)

    divider(f"{conf.name}@计算因子", "-")
    calculate_factors(conf, boost=True)

    divider(f"{conf.name}@计算截面因子", "-")
    calc_cross_sections(conf)
