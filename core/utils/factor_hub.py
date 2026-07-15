

import importlib
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


class FactorInterface:
    """
    ！！！！抽象因子对象，仅用于代码提示！！！！
    """

    # 财务因子列：此列表用于存储财务因子相关的列名称
    fin_cols = []  # 财务因子列，配置后系统会自动加载对应的财务数据
    extra_data = {}  # 额外数据
    is_cross = False  # 是否为截面因子

    @staticmethod
    def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
        """
        计算并将新的因子列添加到股票行情数据中，并返回包含计算因子的DataFrame及其聚合方式。

        工作流程：
        1. 根据提供的参数计算股票的因子值。
        2. 将因子值添加到原始行情数据DataFrame中。
        3. 定义因子的聚合方式，用于周期转换时的数据聚合。

        :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
        :param param: 因子计算所需的参数，格式和含义根据因子类型的不同而有所不同。
        :param kwargs: 其他关键字参数，包括：
            - col_name: 新计算的因子列名。
            - fin_data: 财务数据字典，格式为 {'财务数据': fin_df, '原始财务数据': raw_fin_df}，其中fin_df为处理后的财务数据，raw_fin_df为原始数据，后者可用于某些因子的自定义计算。
            - 其他参数：根据具体需求传入的其他因子参数。
        :return: tuple
            - pd.DataFrame: 包含新计算的因子列，与输入的df具有相同的索引。
            - dict: 聚合方式字典，定义因子在周期转换时如何聚合（例如保留最新值、计算均值等）。

        注意事项：
        - 如果因子的计算涉及财务数据，可以通过`fin_data`参数提供相关数据。
        - 聚合方式可以根据实际需求进行调整，例如使用'last'保留最新值，或使用'mean'、'max'、'sum'等方法。
        """

        # ======================== 参数处理 ===========================
        # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称
        col_name = kwargs["col_name"]
        print(param)  # 实际使用中，因子文件需要自己解析输入参数的具体含义，比如周期长度，比如一些枚举类型等等

        # ======================== 计算因子 ===========================
        """
        [abstract]
        目前这个接口中并没有实现任何的计算逻辑，只是提供一个接口，用于提示
        需要在这个位置实现计算逻辑，并且在 `df` 中添加一个新的因子列，列名为 col_name
        """

        # 我们只返回因子的列信息，以及周期转换时候因子列的聚合方式
        return df[[col_name]]

    def add_factors(self, df: pd.DataFrame, params=(), **kwargs) -> Tuple[pd.DataFrame, dict]:
        """
        批量计算多个参数下的因子数值
        """
        raise NotImplementedError


class FactorHub:
    _factor_cache = {}
    _factor_dirs = ["因子库", "截面因子库"]

    @staticmethod
    def get_factor_file_path(factor_name: str) -> Optional[Path]:
        """根据因子名查找对应的 .py 文件路径，依次搜索 因子库 和 截面因子库"""
        for module_dir in FactorHub._factor_dirs:
            py_file = Path(module_dir) / f"{factor_name}.py"
            if py_file.exists():
                return py_file
        return None

    # noinspection PyTypeChecker
    @staticmethod
    def get_by_name(factor_name) -> FactorInterface:
        """根据因子名获取因子文件对象"""
        if factor_name in FactorHub._factor_cache:
            return FactorHub._factor_cache[factor_name]

        modules_to_try = FactorHub._factor_dirs

        for module in modules_to_try:
            full_module_name = f"{module}.{factor_name}"
            try:
                factor_module = importlib.import_module(full_module_name)

                # 创建一个包含模块变量和函数的字典
                factor_content = {
                    name: getattr(factor_module, name) for name in dir(factor_module) if not name.startswith("__")
                }

                if "fin_cols" not in factor_content:
                    factor_content["fin_cols"] = []

                factor_content["is_cross"] = module == "截面因子库"

                # 创建一个包含这些变量和函数的对象
                factor_instance = type(factor_name, (), factor_content)

                # 缓存策略对象
                FactorHub._factor_cache[factor_name] = factor_instance

                return factor_instance

            except ModuleNotFoundError as e:
                # 关键判断：检查是哪个模块没找到
                # 情况1: 目标模块本身不存在 -> 继续尝试下一个路径
                if e.name in (full_module_name, factor_name):
                    continue
                # 情况2: 因子库/截面因子库不存在
                if e.name == module:
                    raise ValueError(f"【{e.name}】不存在") from e
                # 情况3: 目标模块存在，但其内部依赖的模块不存在 -> 直接报错
                raise ValueError(f"因子【{factor_name}】存在于【{module}】中, 但是该因子缺少依赖 '{e.name}'") from e

            except AttributeError as e:
                raise ValueError(f"Error accessing factor content in module {factor_name}: {e}") from e

        # 所有路径都尝试完毕，模块确实不存在
        raise ValueError(f"因子【{factor_name}】不在因子库/截面因子库中")
