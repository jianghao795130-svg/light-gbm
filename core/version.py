
import config
from core.utils.log_kit import logger, divider

sys_version = "2.3.0"
sys_name = "select-stock-pro"
build_version = f"v{sys_version}.20260327"


def version_prompt():
    divider("[SYSTEM INFO]", "#", with_timestamp=False)
    logger.debug(f"# VERSION: {sys_name}({sys_version})")
    logger.debug(f"# BUILD VERSION: {build_version}")
    divider("[SYSTEM INFO]", "#", with_timestamp=False)

    match getattr(config, "performance_mode", "MAX"):
        case "BAL" | "EQUAL":  # 均衡
            print("⚖️ 性能模式：均衡")
        case "MAX" | "PERFORMANCE":  # 快速
            print("⚡️ 性能模式：快速")
        case "ECO" | "ECONOMY":  # 节能
            print("♻️ 性能模式：节能")
