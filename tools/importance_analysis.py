from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tools.plot_config import setup_chinese_matplotlib


setup_chinese_matplotlib()


def load_models(model_dir: str | Path) -> tuple[list[lgb.Booster], list[str]]:
    model_dir = Path(model_dir)
    model_files = sorted(model_dir.glob("*.txt"))
    models = [lgb.Booster(model_file=str(path)) for path in model_files]
    names = [path.stem for path in model_files]
    return models, names


def get_importance_df(model: lgb.Booster) -> pd.DataFrame:
    feat_names = model.feature_name()
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": feat_names, "gain": gain, "split": split})
    df["gain_pct"] = df["gain"] / df["gain"].sum() * 100 if df["gain"].sum() else 0.0
    df["split_pct"] = df["split"] / df["split"].sum() * 100 if df["split"].sum() else 0.0
    return df


def aggregate_importance(models: list[lgb.Booster], model_names: list[str]) -> pd.DataFrame:
    frames = []
    for model, name in zip(models, model_names):
        df = get_importance_df(model)[["feature", "gain_pct", "split_pct"]].copy()
        df = df.rename(columns={"gain_pct": f"gain_{name}", "split_pct": f"split_{name}"})
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="feature", how="outer")

    gain_cols = [c for c in merged.columns if c.startswith("gain_")]
    split_cols = [c for c in merged.columns if c.startswith("split_")]
    merged["gain_mean"] = merged[gain_cols].mean(axis=1)
    merged["split_mean"] = merged[split_cols].mean(axis=1)
    return merged.sort_values("gain_mean", ascending=False)


def restore_feature_names(df: pd.DataFrame, reverse_feature_rename_map: dict[str, str]) -> pd.DataFrame:
    restored = df.copy()
    if "feature" in restored.columns:
        restored["feature"] = restored["feature"].map(lambda x: reverse_feature_rename_map.get(x, x))
    return restored


def plot_importance_summary(agg_df: pd.DataFrame, save_path: str | Path, top_n: int = 20) -> Path:
    save_path = Path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.35)))

    by_gain = agg_df.sort_values("gain_mean", ascending=False).head(top_n).sort_values("gain_mean")
    by_split = agg_df.sort_values("split_mean", ascending=False).head(top_n).sort_values("split_mean")

    axes[0].barh(by_gain["feature"], by_gain["gain_mean"], color="#3498db")
    axes[0].set_title("Gain 重要性 TopN")
    axes[0].set_xlabel("平均贡献占比(%)")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(by_split["feature"], by_split["split_mean"], color="#e67e22")
    axes[1].set_title("Split 重要性 TopN")
    axes[1].set_xlabel("平均分裂占比(%)")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def aggregate_shap_importance(
    models: list[lgb.Booster],
    x_sample: pd.DataFrame,
    model_names: list[str],
    max_samples: int = 2000,
) -> pd.DataFrame:
    if len(x_sample) > max_samples:
        x_sample = x_sample.sample(n=max_samples, random_state=42)

    frames = []
    for model, name in zip(models, model_names):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
        importance = np.abs(shap_values).mean(axis=0)
        frames.append(pd.DataFrame({"feature": x_sample.columns, f"shap_{name}": importance}))

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="feature", how="outer")

    shap_cols = [c for c in merged.columns if c.startswith("shap_")]
    merged["shap_mean"] = merged[shap_cols].mean(axis=1)
    return merged.sort_values("shap_mean", ascending=False)


def plot_shap_summary(agg_shap_df: pd.DataFrame, save_path: str | Path, top_n: int = 20) -> Path:
    save_path = Path(save_path)
    top_df = agg_shap_df.head(top_n).sort_values("shap_mean")
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
    ax.barh(top_df["feature"], top_df["shap_mean"], color="#9b59b6")
    ax.set_title("SHAP 重要性 TopN")
    ax.set_xlabel("平均 |SHAP|")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_low_importance_table(
    agg_df: pd.DataFrame,
    agg_shap_df: pd.DataFrame,
    quantile: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = agg_df[["feature", "gain_mean", "split_mean"]].merge(
        agg_shap_df[["feature", "shap_mean"]],
        on="feature",
        how="outer",
    )
    summary["gain_rank"] = summary["gain_mean"].rank(ascending=False)
    summary["split_rank"] = summary["split_mean"].rank(ascending=False)
    summary["shap_rank"] = summary["shap_mean"].rank(ascending=False)
    summary["avg_rank"] = summary[["gain_rank", "split_rank", "shap_rank"]].mean(axis=1)
    summary = summary.sort_values("avg_rank")
    threshold = len(summary) * quantile
    low_importance = summary[summary["avg_rank"] > threshold].copy()
    return summary, low_importance
