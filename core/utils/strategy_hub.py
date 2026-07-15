

import importlib


def get_strategy_by_name(name) -> dict:
    module = "策略库"
    full_module_name = f"{module}.{name}"

    try:
        # 动态导入模块
        strategy_module = importlib.import_module(full_module_name)

        # 创建一个包含模块变量和函数的字典
        strategy_content = {
            name: getattr(strategy_module, name)
            for name in dir(strategy_module)
            if not name.startswith("__") and callable(getattr(strategy_module, name))
        }

        return strategy_content
    except ModuleNotFoundError as e:
        if e.name not in (full_module_name, name):
            raise ValueError(f"因子【{name}】存在于【{module}】中, 但是该因子缺少依赖 '{e.name}'") from e
        return {}
        # raise ValueError(f"Strategy {strategy_name} not found.")
    except AttributeError:
        raise ValueError(f"Error accessing strategy content in module {name}.")
