from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.data_prepare import COLS
from tools.feature_engineering import RAW_RETURN_COL


@dataclass(frozen=True)
class OptunaTuningConfig:
    n_trials: int = 30
    study_name: str = "lgbm_optuna_tuning"
    storage: str | None = None
    load_if_exists: bool = True
    metric: str = "rank_ic_mean"
    seed: int = 716
    train_window_months: int = 72
    val_window_months: int = 12
    test_window_months: int = 3
    stride_months: int = 3
    test_start: str | pd.Timestamp = "2024-01-01"
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    sample_window_limit: int | None = None
    verbose_eval: bool = False


@dataclass(frozen=True)
class RollingWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def calc_rank_ic(g: pd.DataFrame) -> float:
    clean = g[["ml_factor", RAW_RETURN_COL]].dropna()
    if len(clean) < 2 or clean["ml_factor"].nunique() < 2 or clean[RAW_RETURN_COL].nunique() < 2:
        return np.nan
    return clean["ml_factor"].corr(clean[RAW_RETURN_COL], method="spearman")


def build_rolling_windows(
    df: pd.DataFrame,
    train_window_months: int = 72,
    val_window_months: int = 12,
    test_window_months: int = 3,
    stride_months: int = 3,
    test_start: str | pd.Timestamp = "2024-01-01",
) -> list[RollingWindow]:
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(df[COLS.trade_date].max())

    windows: list[RollingWindow] = []
    current_test_start = test_start
    while current_test_start < test_end:
        current_test_end = min(current_test_start + pd.DateOffset(months=test_window_months), test_end)
        train_end = current_test_start
        train_start = train_end - pd.DateOffset(months=train_window_months)
        val_start = train_end - pd.DateOffset(months=val_window_months)
        windows.append(
            RollingWindow(
                window_id=len(windows),
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=train_end,
                test_start=current_test_start,
                test_end=current_test_end,
            )
        )
        current_test_start += pd.DateOffset(months=stride_months)

    return windows


def get_default_lgb_params(seed: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.3,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": -1,
    }


def _require_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "Optuna is not installed in the current Python environment. "
            "Please install it with `pip install optuna` in the environment "
            "where you run the notebook."
        ) from exc
    return optuna


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "LightGBM is not installed in the current Python environment. "
            "Please install it with `pip install lightgbm` in the environment "
            "where you run the notebook."
        ) from exc
    return lgb


def suggest_lgb_params(trial: Any, base_seed: int = 716) -> dict[str, Any]:
    params = get_default_lgb_params(seed=base_seed).copy()
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, step=0.01),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63, step=4),
            "max_bin": trial.suggest_categorical("max_bin", [31, 63, 127]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, step=50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0, step=0.1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0, step=0.1),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0, step=0.01),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0, step=0.01),
        }
    )
    return params


def _build_validation_predictions(
    model: Any,
    val_df: pd.DataFrame,
    features: list[str],
    window_id: int,
) -> pd.DataFrame:
    pred_df = val_df[[COLS.trade_date, COLS.stock_code, RAW_RETURN_COL]].copy()
    pred_df["ml_factor"] = model.predict(val_df[features], num_iteration=model.best_iteration)
    pred_df["window_id"] = window_id
    return pred_df


def _score_validation_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {
            "rank_ic_mean": np.nan,
            "rank_ic_ir": np.nan,
            "pearson_proxy_rmse": np.nan,
        }

    daily_rank_ic = pred_df.groupby(COLS.trade_date).apply(calc_rank_ic).dropna()
    rank_ic_mean = float(daily_rank_ic.mean()) if len(daily_rank_ic) else np.nan
    rank_ic_std = float(daily_rank_ic.std()) if len(daily_rank_ic) else np.nan
    rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std and rank_ic_std > 0 else np.nan
    return {
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": rank_ic_ir,
    }


def evaluate_params_on_validation(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    params: dict[str, Any],
    *,
    config: OptunaTuningConfig | None = None,
) -> dict[str, Any]:
    lgb = _require_lightgbm()
    config = config or OptunaTuningConfig()

    windows = build_rolling_windows(
        df=df,
        train_window_months=config.train_window_months,
        val_window_months=config.val_window_months,
        test_window_months=config.test_window_months,
        stride_months=config.stride_months,
        test_start=config.test_start,
    )
    if config.sample_window_limit is not None:
        windows = windows[: config.sample_window_limit]

    all_pred_parts: list[pd.DataFrame] = []
    rmse_rows: list[dict[str, float]] = []

    for window in windows:
        train_mask = (df[COLS.trade_date] >= window.train_start) & (df[COLS.trade_date] < window.val_start)
        val_mask = (df[COLS.trade_date] >= window.val_start) & (df[COLS.trade_date] < window.val_end)
        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()
        if train_df.empty or val_df.empty:
            continue

        train_data = lgb.Dataset(train_df[features], label=train_df[label_col], free_raw_data=False)
        val_data = lgb.Dataset(val_df[features], label=val_df[label_col], reference=train_data, free_raw_data=False)

        evals_result: dict[str, dict[str, list[float]]] = {}
        callbacks: list[Any] = [
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
        ]
        if config.verbose_eval:
            callbacks.append(lgb.log_evaluation(100))

        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=config.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        train_rmse = float(evals_result["train"]["rmse"][-1])
        val_rmse = float(evals_result["valid"]["rmse"][-1])
        rmse_rows.append(
            {
                "window_id": float(window.window_id),
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "best_iteration": float(model.best_iteration),
            }
        )
        all_pred_parts.append(_build_validation_predictions(model, val_df, features, window.window_id))

    pred_df = pd.concat(all_pred_parts, axis=0, ignore_index=True) if all_pred_parts else pd.DataFrame()
    rmse_df = pd.DataFrame(rmse_rows)
    ic_metrics = _score_validation_predictions(pred_df)

    if rmse_df.empty:
        rmse_summary = {
            "train_rmse_mean": np.nan,
            "val_rmse_mean": np.nan,
            "overfit_ratio_mean": np.nan,
            "best_iteration_mean": np.nan,
        }
    else:
        train_rmse_mean = float(rmse_df["train_rmse"].mean())
        val_rmse_mean = float(rmse_df["val_rmse"].mean())
        rmse_summary = {
            "train_rmse_mean": train_rmse_mean,
            "val_rmse_mean": val_rmse_mean,
            "overfit_ratio_mean": val_rmse_mean / train_rmse_mean if train_rmse_mean > 0 else np.nan,
            "best_iteration_mean": float(rmse_df["best_iteration"].mean()),
        }

    return {
        "params": params,
        "window_count": 0 if rmse_df.empty else int(rmse_df["window_id"].nunique()),
        **ic_metrics,
        **rmse_summary,
    }


def _get_study_direction(metric: str) -> str:
    maximize_metrics = {"rank_ic_mean", "rank_ic_ir"}
    return "maximize" if metric in maximize_metrics else "minimize"


def run_optuna_tuning(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    *,
    config: OptunaTuningConfig | None = None,
    output_dir: str | Path = "research_outputs",
) -> tuple[Any, pd.DataFrame, Path]:
    optuna = _require_optuna()
    _require_lightgbm()
    config = config or OptunaTuningConfig()
    direction = _get_study_direction(config.metric)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: Any) -> float:
        params = suggest_lgb_params(trial, base_seed=config.seed)
        summary = evaluate_params_on_validation(
            df=df,
            features=features,
            label_col=label_col,
            params=params,
            config=config,
        )
        for key, value in summary.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                trial.set_user_attr(key, float(value))
        score = summary.get(config.metric, np.nan)
        if pd.isna(score):
            return -1e9 if direction == "maximize" else 1e9
        return float(score)

    study = optuna.create_study(
        direction=direction,
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=config.load_if_exists,
    )
    study.optimize(objective, n_trials=config.n_trials)

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "objective": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        rows.append(row)
    trials_df = pd.DataFrame(rows).sort_values("objective", ascending=(direction == "minimize"))
    save_path = output_dir / "optuna_tuning_results.csv"
    trials_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return study, trials_df, save_path


def describe_best_trial(study: Any) -> pd.Series:
    best = {
        "best_value": study.best_value,
        **study.best_params,
        **study.best_trial.user_attrs,
    }
    return pd.Series(best)


def config_to_dict(config: OptunaTuningConfig) -> dict[str, Any]:
    return asdict(config)
