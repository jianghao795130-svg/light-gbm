from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tools.data_prepare import COLS
from tools.feature_engineering import RAW_RETURN_COL
from tools.training_monitor import TrainingMonitor


DEFAULT_SEEDS = [716, 666, 168, 999, 618]


@dataclass(frozen=True)
class RollingWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_rolling_windows(
    df: pd.DataFrame,
    train_window_months: int = 72,
    val_window_months: int = 12,
    test_window_months: int = 3,
    stride_months: int = 3,
    test_start: str | pd.Timestamp = "2024-01-01",
) -> list[RollingWindow]:
    """构建滚动训练窗口。"""
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


def get_default_lgb_params(seed: int) -> dict:
    """LightGBM 回归参数，参数上方会在 notebook 里配合注释说明。"""
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


def run_rolling_training(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    seeds: list[int] | None = None,
    train_window_months: int = 72,
    val_window_months: int = 12,
    test_window_months: int = 3,
    stride_months: int = 3,
    test_start: str | pd.Timestamp = "2024-01-01",
    model_dir: str | Path = "models",
    training_log_dir: str | Path = "training_logs",
) -> tuple[pd.DataFrame, list[RollingWindow], TrainingMonitor]:
    """滚动训练 5 个种子模型，并对测试集做集成预测。"""
    seeds = seeds or DEFAULT_SEEDS
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    windows = build_rolling_windows(
        df=df,
        train_window_months=train_window_months,
        val_window_months=val_window_months,
        test_window_months=test_window_months,
        stride_months=stride_months,
        test_start=test_start,
    )

    monitor = TrainingMonitor(output_dir=training_log_dir)
    prediction_parts: list[pd.DataFrame] = []

    for window in windows:
        train_mask = (df[COLS.trade_date] >= window.train_start) & (df[COLS.trade_date] < window.val_start)
        val_mask = (df[COLS.trade_date] >= window.val_start) & (df[COLS.trade_date] < window.val_end)
        test_mask = (df[COLS.trade_date] >= window.test_start) & (df[COLS.trade_date] < window.test_end)

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()
        test_df = df.loc[test_mask].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            continue

        x_train = train_df[features]
        y_train = train_df[label_col]
        x_val = val_df[features]
        y_val = val_df[label_col]
        x_test = test_df[features]

        test_preds = []

        for seed in seeds:
            params = get_default_lgb_params(seed=seed)
            train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
            val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, free_raw_data=False)

            evals_result: dict = {}
            model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=2000,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(100),
                ],
            )

            train_rmse = evals_result["train"]["rmse"][-1]
            val_rmse = evals_result["valid"]["rmse"][-1]
            monitor.add_record(
                window_id=window.window_id,
                seed=seed,
                best_iteration=model.best_iteration,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
            )
            monitor.plot_rmse_curve(window.window_id, seed, evals_result)

            model_path = model_dir / f"window_{window.window_id:02d}_seed_{seed}.txt"
            model.save_model(str(model_path.resolve().as_posix()))
            test_preds.append(model.predict(x_test))

        ensemble_pred = np.mean(test_preds, axis=0)
        pred_df = test_df[[COLS.trade_date, COLS.stock_code, RAW_RETURN_COL]].copy()
        pred_df["ml_factor"] = ensemble_pred
        pred_df["window_id"] = window.window_id
        prediction_parts.append(pred_df)

    prediction_df = pd.concat(prediction_parts, axis=0, ignore_index=True)
    return prediction_df, windows, monitor
