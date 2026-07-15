import json
import importlib
import inspect
import re
import ast
from pathlib import Path

from core.utils.log_kit import logger
from core.utils.path_kit import get_folder_path


class SmartJSONEncoder(json.JSONEncoder):
    """智能JSON编码器，自动清理无用数据并自定义格式化"""

    def __init__(self, config_module="config", clean_funcs=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_vars = self._load_config(config_module)
        self.clean_funcs = clean_funcs
        self.visited = set()

    def _load_config(self, config_module):
        """加载配置模块中的变量"""
        try:
            module = importlib.import_module(config_module)
            variables = set()
            for name in dir(module):
                if not name.startswith("_"):
                    attr = getattr(module, name)
                    if not inspect.isfunction(attr) and not inspect.isclass(attr) and not inspect.ismodule(attr):
                        variables.add(name)
            return variables
        except ImportError:
            return set()

    def default(self, obj):
        """处理无法序列化的对象"""
        obj_id = id(obj)
        if obj_id in self.visited:
            return "<CircularReference>"

        self.visited.add(obj_id)

        try:
            if hasattr(obj, "__dict__"):
                return self._clean_dict(obj.__dict__)
            elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                return list(obj)
            else:
                return str(obj)
        finally:
            self.visited.discard(obj_id)

    def _clean_dict(self, data_dict):
        """清理字典数据"""
        cleaned = {}
        for key, value in data_dict.items():
            # 跳过无用字段
            if self.clean_funcs and self._should_skip_field(key, value):
                continue

            cleaned[key] = value
        return cleaned

    def _should_skip_field(self, key, value):
        """判断是否应该跳过某个字段"""
        # 跳过私有字段
        if str(key).startswith("_"):
            return True

        # 跳过特定的无用字段
        if key in ["funcs", "lock"]:
            return True

        # 跳过包含锁对象的字符串
        if isinstance(value, str) and "RLock object" in value:
            return True

        # 跳过空的函数字典
        if isinstance(value, dict) and len(value) == 0 and key == "funcs":
            return True

        return False


# 创建一个特殊的列表类，用来标记应该转换为tuple的列表
class TupleList(list):
    """标记应该在Python代码中显示为tuple的列表"""

    pass


class ConfigBasedConverter:
    """基于config.py文件的对象转换器"""

    def __init__(self, input_file: Path, field_mapping=None, exclude_fields=None):
        self.input_file = input_file
        self.field_mapping = field_mapping or {}  # 字段名映射 {obj字段名: config字段名}
        self.exclude_fields = exclude_fields or []  # 排除的字段列表
        self.float_precision = 6  # 浮点数精度

        # 解析config.py文件
        self.config_variables = self._parse_config_file()

    def _parse_config_file(self):
        """解析config.py文件，提取变量定义及其默认值"""

        if not self.input_file.exists():
            logger.warning(f"配置文件 {self.input_file} 不存在")
            return []

        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)

            variables = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # 只处理简单的变量赋值 (var = value)
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):

                        var_name = node.targets[0].id

                        # 跳过私有变量和排除的字段
                        if not var_name.startswith("_") and var_name not in self.exclude_fields:

                            # 尝试获取默认值（用于了解数据类型）
                            try:
                                default_value = ast.literal_eval(node.value)
                            except (ValueError, TypeError):
                                default_value = None

                            variables.append({"name": var_name, "default_value": default_value, "lineno": node.lineno})

            # 按行号排序，保持原文件中的顺序
            variables.sort(key=lambda x: x["lineno"])

            return variables

        except Exception as e:
            logger.warning(f"解析配置文件失败: {e}")
            return []

    def _clean_float_precision(self, value, precision=None):
        """清理浮点数精度误差"""
        if precision is None:
            precision = self.float_precision

        if isinstance(value, float):
            return round(value, precision)
        elif isinstance(value, (list, tuple)):
            return [self._clean_float_precision(item, precision) for item in value]
        elif isinstance(value, dict):
            return {k: self._clean_float_precision(v, precision) for k, v in value.items()}
        else:
            return value

    def _clean_strategy_name(self, name):
        """清理策略名称中的前缀，如 #0. #1. 等"""
        if isinstance(name, str):
            cleaned_name = re.sub(r"^#\d+\.", "", name)
            return cleaned_name
        return name

    def _convert_lists_to_tuples_recursive(self, value, for_python=False):
        """递归地将嵌套列表转换为TupleList（用于Python输出）或保持为列表（用于JSON输出）"""
        if isinstance(value, list):
            converted_items = [self._convert_lists_to_tuples_recursive(item, for_python) for item in value]

            if for_python:
                return TupleList(converted_items)
            else:
                return converted_items
        elif isinstance(value, tuple):
            converted_items = [self._convert_lists_to_tuples_recursive(item, for_python) for item in value]
            return tuple(converted_items)
        elif isinstance(value, dict):
            return {k: self._convert_lists_to_tuples_recursive(v, for_python) for k, v in value.items()}
        else:
            return value

    def _obj_to_dict(self, obj):
        """将对象转换为字典，处理嵌套对象"""
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._obj_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._obj_to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return {k: self._obj_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        else:
            return str(obj)

    def _get_obj_value(self, obj, config_var_name):
        """从对象中获取对应config变量的值"""
        obj_dict = obj.__dict__ if hasattr(obj, "__dict__") else {}

        # 创建反向映射 {config字段名: obj字段名}
        reverse_mapping = {v: k for k, v in self.field_mapping.items()}

        # 确定要查找的obj字段名
        obj_field_name = reverse_mapping.get(config_var_name, config_var_name)

        if obj_field_name in obj_dict:
            value = obj_dict[obj_field_name]

            # 特殊处理
            if config_var_name in ["name"] and hasattr(self, "_clean_strategy_name"):
                value = self._clean_strategy_name(value)

            # 处理浮点数精度
            value = self._clean_float_precision(value)

            return value
        else:
            return None

    def _format_factor_tuple(self, factor_data, for_python=False):
        """格式化因子元组为config格式，保持None值，并处理嵌套列表转tuple"""
        if isinstance(factor_data, (list, tuple)):
            if len(factor_data) >= 4:
                result = [
                    factor_data[0],  # 因子名称
                    factor_data[1],  # 是否升序
                    factor_data[2] if factor_data[2] is not None else None,  # 参数，保持None
                    self._convert_lists_to_tuples_recursive(factor_data[3], for_python),  # 权重（可能包含嵌套列表）
                ]
                if len(factor_data) >= 5:  # 如果有第5个元素（时间）
                    result.append(self._convert_lists_to_tuples_recursive(factor_data[4], for_python))

                return result
        elif isinstance(factor_data, dict):
            name = factor_data.get("name", "")
            is_asc = factor_data.get("is_sort_asc", True)
            param = factor_data.get("param", None)
            weight = factor_data.get("args", 1)

            weight = self._convert_lists_to_tuples_recursive(weight, for_python)

            result = [name, is_asc, param, weight]
            return result

        return factor_data

    def _format_condition_dict(self, condition):
        """将条件字典转换为字符串格式"""
        if isinstance(condition, dict):
            how = condition.get("how", "")
            range_val = condition.get("range", "")
            if how and range_val:
                return f"{how}:{range_val}"
            elif range_val:
                return str(range_val)
            elif how:
                return str(how)
            else:
                return str(condition)
        else:
            return condition

    def _format_filter_tuple(self, filter_data, for_python=False):
        """格式化过滤器元组为config格式"""
        if isinstance(filter_data, (list, tuple)):
            if len(filter_data) >= 3:
                condition = filter_data[2]
                formatted_condition = self._format_condition_dict(condition)

                result = [
                    filter_data[0],  # 过滤器名称
                    self._convert_lists_to_tuples_recursive(filter_data[1], for_python),  # 参数（可能包含嵌套列表）
                    formatted_condition,  # 条件（转换后的字符串）
                    filter_data[3] if len(filter_data) > 3 else True,  # 布尔值
                ]
                return result
        elif isinstance(filter_data, dict):
            name = filter_data.get("name", "")
            param = filter_data.get("param", "")
            method = filter_data.get("method", "")
            enabled = filter_data.get("enabled", True)

            param = self._convert_lists_to_tuples_recursive(param, for_python)
            formatted_method = self._format_condition_dict(method)

            return [name, param, formatted_method, enabled]

        return filter_data

    def _format_strategy_list(self, strategy_list, for_python=False):
        """根据config.py格式化strategy_list"""
        if not strategy_list:
            return []

        formatted_strategies = []

        for strategy in strategy_list:
            strategy_dict = self._obj_to_dict(strategy)
            is_section = "alias_name" in strategy_dict
            formatted_strategy = {}

            # 基本字段
            basic_fields = (
                ["name", "is_sort_asc", "args", "minutes", "method"]
                if is_section
                else ["name", "hold_period", "offset_list", "select_num", "cap_weight", "rebalance_time"]
            )

            for field in basic_fields:
                if field in strategy_dict:
                    value = strategy_dict[field]
                    if field == "method" and value:
                        value = f"{value['how']}:{value['range']}"
                    else:
                        if field == "name":
                            value = self._clean_strategy_name(value)
                        value = self._clean_float_precision(value)
                    formatted_strategy[field] = value

            # 处理factor_list
            formatted_factors = []
            for factor in strategy_dict.get("factor_list", []):
                formatted_factor = self._format_factor_tuple(factor, for_python)
                formatted_factor = self._clean_float_precision(formatted_factor)
                if for_python and isinstance(formatted_factor, list):
                    formatted_factor = TupleList(formatted_factor)
                formatted_factors.append(formatted_factor)
            formatted_strategy["factor_list"] = formatted_factors

            # 处理filter_list
            formatted_filters = []
            for filter_item in strategy_dict.get("filter_list", []):
                formatted_filter = self._format_filter_tuple(filter_item, for_python)
                formatted_filter = self._clean_float_precision(formatted_filter)
                if for_python and isinstance(formatted_filter, list):
                    formatted_filter = TupleList(formatted_filter)
                formatted_filters.append(formatted_filter)
            if not is_section:
                formatted_strategy["filter_list"] = formatted_filters

            # 处理timing
            if (timing := strategy_dict.get("timing", {})) and not is_section:
                timing_dict = self._obj_to_dict(timing)
                formatted_timing = {}

                timing_basic_fields = ["name", "limit"]
                for field in timing_basic_fields:
                    if field in timing_dict:
                        value = self._clean_float_precision(timing_dict[field])
                        formatted_timing[field] = value

                timing_factors = []
                for factor in timing_dict.get("factor_list", []):
                    timing_factor = self._format_factor_tuple(factor, for_python)
                    timing_factor = self._clean_float_precision(timing_factor)
                    if for_python and isinstance(timing_factor, list):
                        timing_factor = TupleList(timing_factor)
                    timing_factors.append(timing_factor)
                formatted_timing["factor_list"] = timing_factors

                if "params" in timing_dict and timing_dict["params"]:
                    params = timing_dict["params"]
                    if isinstance(params, (list, tuple)):
                        params = self._clean_float_precision(list(params))
                        params = self._convert_lists_to_tuples_recursive(params, for_python)
                        formatted_timing["params"] = params
                    else:
                        formatted_timing["params"] = self._clean_float_precision(params)

                formatted_strategy["timing"] = formatted_timing

            # 处理截面因子
            if cross_sections := strategy_dict.get("cross_sections", []):
                formatted_strategy["cross_sections"] = self._format_strategy_list(cross_sections)

            if is_section:
                params = strategy_dict["param"]
                if isinstance(params, (list, tuple)):
                    params = self._clean_float_precision(list(params))
                    params = self._convert_lists_to_tuples_recursive(params, for_python)
                    formatted_strategy["params"] = params
                else:
                    formatted_strategy["params"] = self._clean_float_precision(params)

            formatted_strategies.append(formatted_strategy)

        return formatted_strategies

    def convert_to_json(self, obj, output_file=None, indent=2):
        """转换对象为JSON"""
        data = {}

        # 按照config.py中的变量顺序逐一处理
        for var_info in self.config_variables:
            var_name = var_info["name"]
            value = self._get_obj_value(obj, var_name)

            if value is not None:
                # 特殊处理strategy_list
                if var_name == "strategy_list":
                    formatted_value = self._format_strategy_list(value, for_python=False)
                else:
                    formatted_value = value

                data[var_name] = formatted_value

        # 转换为JSON
        json_str = json.dumps(
            data, cls=SmartJSONEncoder, config_module=self.input_file.stem, indent=indent, ensure_ascii=False
        )

        # 输出处理
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.ok(f"JSON已保存到 {output_file}")

        return json_str


def object_to_json(
    obj, input_file: Path, output_file=None, indent=2, field_mapping=None, exclude_fields=None, float_precision=6
):
    """
    基于config.py文件结构，将对象转换为JSON

    Args:
        obj: 要转换的对象
        input_file: 输入文件路径
        output_file: 输出文件路径
        indent: JSON缩进
        field_mapping: 字段映射字典 {obj字段名: config字段名}
        exclude_fields: 排除的字段列表
        float_precision: 浮点数保留的小数位数

    Returns:
        JSON字符串
    """
    converter = ConfigBasedConverter(input_file, field_mapping, exclude_fields)
    converter.float_precision = float_precision
    return converter.convert_to_json(obj, output_file, indent)


if __name__ == "__main__":
    from core.model.backtest_config import load_config

    conf = load_config()
    # 把conf对象转为json和py文件
    field_mapping = {"name": "backtest_name"}  # obj中叫name，config中叫backtest_name
    object_to_json(
        conf,
        input_file=get_folder_path() / "config.py",
        field_mapping=field_mapping,
        # exclude_fields=exclude_fields,
        output_file=conf.get_result_folder() / "config.json",
    )
    print("转换完成！")
