import os
import warnings
from pathlib import Path
import subprocess
import sys
import pandas as pd

from core.utils.log_kit import logger
from core.utils.path_kit import get_file_path


def check_runtime_result(cur_cfg):
    """
    检查运行缓存能否直接copy，方便单步调试
    为了解决这么一种场景：
    当我改了backtest_name（前提条件），或者个别参数（不影响数据和因子），不需要重新运行step1和step2
    此时如果直接运行step3或者step4，由于【运行缓存】中是以backtest_name为子文件的，所以backtest_name修改后，就没有对应的缓存了，进而报错
    人工要解决这个问题，其实复制一下运行缓存就行了。此函数就是为了省略人工复制这一步骤而专门写的。
    :param cur_cfg:
    :return:
    """

    def get_cfg_str(cfg):
        return "_".join(
            [
                # stg中的这几个参数不变，就不需要重新运行step1和step2
                f"{stg.rebalance_time, stg.factor_list, stg.filter_list, stg.filter_list_post, stg.timing, stg.cross_sections}"
                for stg in cfg.strategy_list
            ]
        )

    # 运行缓存里的conf列表
    for file in cur_cfg.get_result_folder().parent.iterdir():
        if (cfg_path := file / "config.pkl").exists() and (cur_cfg.get_runtime_folder().parent / file.stem).exists():
            src_cfg = pd.read_pickle(cfg_path)
            # 如果配置名不同，且当前配置的运行缓存为空文件夹（get_runtime_folder函数会自动创建），就说明有copy的前提条件
            if src_cfg.name != cur_cfg.name and not any(cur_cfg.get_runtime_folder().iterdir()):
                if get_cfg_str(src_cfg) != get_cfg_str(cur_cfg):
                    logger.warning(
                        f"【{cur_cfg.name}】和【{src_cfg.name}】的关键参数不相同，不能直接复制运行缓存，请重新运行step1和step2"
                    )
                    continue
                logger.ok(f"【{cur_cfg.name}】和【{src_cfg.name}】的关键参数相同，可以直接复制运行缓存")
                import shutil

                # 复制运行缓存
                shutil.copytree(src_cfg.get_runtime_folder(), cur_cfg.get_runtime_folder(), dirs_exist_ok=True)
                break


def save_csv_safely(df: pd.DataFrame, path: Path, with_pickle=False, with_parquet=False, encoding="utf-8-sig", index=False):
    # 检查目标文件路径是否可以写入，不行的话就提示报错跳过
    try:
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        # 检查是否有写入权限
        test_file = path.parent / f".test_write_permission_{path.stem}"
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
    except Exception as e:
        logger.error(f"无法写入到目标路径: {path}，跳过保存。错误信息: {e}")
        return
    # 如果需要，先保存pkl文件，避免csv保存失败
    if with_pickle:
        df.to_pickle(path.with_suffix(".pkl"))
    if with_parquet:
        df.to_parquet(path.with_suffix(".parquet"), index=index)
    try:
        df.to_csv(path, encoding=encoding, index=index)
        logger.info(f"成功保存CSV到: {path}")
    except Exception as e:
        logger.error(f"保存CSV失败: {path}，可能你用Excel打开了当前的csv。错误信息: {e}")


def pd_concat(dfs, default_cols=None, **kwargs) -> pd.DataFrame:
    """
    # 在新版 pandas（2.2.x 及以后），当参与 concat 的 DataFrame 中存在 “空表” 或 “全是 NA” 的列时，pandas 将改变 dtype 推断逻辑。这意味着未来版本可能报错或结果类型不同。
    安全地拼接多个 DataFrame，自动过滤空表或全NA表，
    并屏蔽未来 pandas 版本中 concat 的无关 FutureWarning。

    参数
    ----
    dfs : list[pd.DataFrame]
        需要拼接的 DataFrame 列表。
    kwargs : 传给 pd.concat 的其他参数
        例如 ignore_index=True, axis=0, etc.

    返回
    ----
    pd.DataFrame
        拼接后的 DataFrame。如果输入全为空，返回空 DataFrame。
    """
    # 过滤掉 None、空表、全 NA 表
    valid_dfs = []
    for df in dfs:
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        if df.empty:
            continue
        if df.isna().all().all():
            continue
        valid_dfs.append(df)

    if not valid_dfs:
        # 返回一个空 DataFrame，确保结构统一
        return pd.DataFrame(columns=default_cols or [])

    # 安静地执行 concat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        result = pd.concat(valid_dfs, **kwargs)

    return result


def execute_preprocess_script() -> None:
    """执行预处理脚本（兼容脚本内部使用多进程）"""

    script_path = get_file_path("program", "小时数据预处理.py")

    if not script_path.exists():
        logger.error(f"预处理脚本不存在：{script_path}")
        sys.exit(2)

    logger.info(f"开始执行预处理脚本：{script_path}")

    # 设置环境变量，确保子进程的输出不缓冲
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=script_path.parent,
            env=env,
            # macOS 多进程兼容：确保正确处理信号
            start_new_session=False,
        )

        # 实时读取输出
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                logger.info(line)

        process.wait()

        if process.returncode != 0:
            logger.error(f"预处理脚本执行失败，返回码：{process.returncode}")
            sys.exit(process.returncode)

        logger.info("预处理脚本执行成功")

    except FileNotFoundError:
        logger.error(f"Python 解释器或脚本未找到")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        process.terminate()
        process.wait(timeout=5)
        sys.exit(130)

    except Exception as e:
        logger.error(f"执行预处理脚本时发生未知错误：{e}")
        sys.exit(1)
