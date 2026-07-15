from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_DIRNAME = "notebook_checkpoints"


def get_project_dir() -> Path:
    return Path.cwd()


def get_output_dir(project_dir: Path | None = None) -> Path:
    project_dir = project_dir or get_project_dir()
    output_dir = project_dir / "research_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_checkpoint_dir(project_dir: Path | None = None) -> Path:
    checkpoint_dir = get_output_dir(project_dir) / DEFAULT_CHECKPOINT_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def checkpoint_path(name: str, project_dir: Path | None = None) -> Path:
    return get_checkpoint_dir(project_dir) / f"{name}.pkl"


def save_checkpoint(name: str, project_dir: Path | None = None, **objects: Any) -> Path:
    path = checkpoint_path(name, project_dir)
    with path.open("wb") as f:
        pickle.dump(objects, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(name: str, project_dir: Path | None = None) -> dict[str, Any]:
    path = checkpoint_path(name, project_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 checkpoint: {path}\n"
            f"请先运行生成该 checkpoint 的前置步骤，或从头运行 Notebook 到对应步骤。"
        )
    with path.open("rb") as f:
        objects = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return objects


def restore_vars(scope: dict[str, Any], checkpoint_name: str, names: list[str], project_dir: Path | None = None) -> None:
    missing = [name for name in names if name not in scope]
    if not missing:
        return
    objects = load_checkpoint(checkpoint_name, project_dir)
    still_missing = [name for name in missing if name not in objects]
    if still_missing:
        raise KeyError(f"checkpoint {checkpoint_name} 中缺少变量: {still_missing}")
    scope.update({name: objects[name] for name in missing})


def bootstrap_paths(scope: dict[str, Any], mad_n: float = 5.0) -> None:
    project_dir = get_project_dir()
    output_dir = get_output_dir(project_dir)
    checkpoint_dir = get_checkpoint_dir(project_dir)
    scope.setdefault("project_dir", project_dir)
    scope.setdefault("output_dir", output_dir)
    scope.setdefault("checkpoint_dir", checkpoint_dir)
    scope.setdefault("mad_n", mad_n)
