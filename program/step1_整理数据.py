
import warnings
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from core.data_center import prepare_data
from core.model.backtest_config import load_config
from core.version import version_prompt

# ====================================================================================================
# ** 配置与初始化 **
# 设置必要的显示选项及忽略警告，以优化代码输出的阅读体验
# ====================================================================================================
warnings.filterwarnings('ignore')  # 忽略不必要的警告
pd.set_option('expand_frame_repr', False)  # 使数据框在控制台显示不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

if __name__ == '__main__':
    version_prompt()
    conf = load_config()
    print(conf.desc())

    prepare_data(conf, boost=True)
