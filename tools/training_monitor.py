from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tools.plot_config import setup_chinese_matplotlib


setup_chinese_matplotlib()


@dataclass
class TrainingRecord:
    window_id: int
    seed: int
    best_iteration: int
    train_rmse: float
    val_rmse: float
    overfit_ratio: float


class TrainingMonitor:
    """记录每个窗口、每个随机种子的训练结果，并输出过拟合检查。"""

    def __init__(self, output_dir: str | Path = "training_logs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir = self.output_dir / "metric_curves"
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[TrainingRecord] = []

    def add_record(
        self,
        window_id: int,
        seed: int,
        best_iteration: int,
        train_rmse: float,
        val_rmse: float,
    ) -> None:
        overfit_ratio = val_rmse / train_rmse if train_rmse > 0 else float("nan")
        self.records.append(
            TrainingRecord(
                window_id=window_id,
                seed=seed,
                best_iteration=best_iteration,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
                overfit_ratio=overfit_ratio,
            )
        )

    def plot_rmse_curve(
        self,
        window_id: int,
        seed: int,
        evals_result: dict,
    ) -> Path:
        train_curve = evals_result["train"]["rmse"]
        valid_curve = evals_result["valid"]["rmse"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_curve, label="训练集 RMSE", linewidth=1.5)
        ax.plot(valid_curve, label="验证集 RMSE", linewidth=1.5)
        ax.set_title(f"窗口 {window_id} - 种子 {seed} 训练曲线")
        ax.set_xlabel("迭代轮数")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        save_path = self.curves_dir / f"window_{window_id:02d}_seed_{seed}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    def save_summary(self) -> tuple[pd.DataFrame, Path]:
        summary_df = self.to_frame()
        save_path = self.output_dir / "training_summary.csv"
        summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return summary_df, save_path

    def build_overfit_summary(self) -> pd.DataFrame:
        summary_df = self.to_frame()
        if summary_df.empty:
            return summary_df

        grouped = summary_df.groupby("window_id").agg(
            model_count=("seed", "count"),
            best_iteration_mean=("best_iteration", "mean"),
            train_rmse_mean=("train_rmse", "mean"),
            val_rmse_mean=("val_rmse", "mean"),
            overfit_ratio_mean=("overfit_ratio", "mean"),
            overfit_ratio_max=("overfit_ratio", "max"),
        )
        return grouped.reset_index()

    def plot_overfit_summary(self) -> Path | None:
        summary_df = self.build_overfit_summary()
        if summary_df.empty:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(summary_df["window_id"], summary_df["train_rmse_mean"], marker="o", label="训练集 RMSE")
        axes[0].plot(summary_df["window_id"], summary_df["val_rmse_mean"], marker="o", label="验证集 RMSE")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("滚动窗口过拟合检查")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(summary_df["window_id"], summary_df["overfit_ratio_mean"], marker="o", color="#c0392b")
        axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axes[1].set_xlabel("窗口编号")
        axes[1].set_ylabel("验证集 / 训练集 RMSE")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = self.output_dir / "overfit_summary.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
