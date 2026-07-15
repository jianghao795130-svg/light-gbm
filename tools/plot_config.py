from __future__ import annotations

from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib import font_manager


@lru_cache(maxsize=1)
def setup_chinese_matplotlib() -> str:
    """统一配置 matplotlib 中文字体和负号显示。"""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
    ]
    selected_font = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")

    plt.rcParams["font.sans-serif"] = [selected_font] + [
        font for font in preferred_fonts if font != selected_font and font in available_fonts
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    return selected_font
