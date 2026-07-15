from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tools.data_prepare import COLS, factor_name_from_path


INDUSTRY_COL = "一级行业"
MARKET_CAP_FACTOR_COL = "市值"
NEUTRALIZED_TARGET_COL = "未来收益_预处理"
RAW_RETURN_COL = "raw_return"
PROCESSED_SUFFIX = "_预处理"


def list_factor_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("factor_*.pkl"))


def split_label_and_features(
    factor_files: list[Path],
    label_keyword: str = "未来n日涨跌",
) -> tuple[Path, list[Path]]:
    label_file = next(path for path in factor_files if label_keyword in path.stem)
    feature_files = [path for path in factor_files if path != label_file]
    return label_file, feature_files


def attach_label_and_features(
    base_df: pd.DataFrame,
    label_file: Path,
    feature_files: list[Path],
) -> tuple[pd.DataFrame, str, list[str]]:
    df = base_df.copy()
    row_ids = df[COLS.row_id]

    label_name = factor_name_from_path(label_file)
    label_series = pd.read_pickle(label_file)
    # Reindex to the base row ids so a partially missing factor file becomes NaN
    # instead of breaking the whole pipeline with a KeyError.
    df[label_name] = label_series.reindex(row_ids).to_numpy()

    feature_names: list[str] = []
    for path in feature_files:
        factor_name = factor_name_from_path(path)
        factor_series = pd.read_pickle(path)
        df[factor_name] = factor_series.reindex(row_ids).to_numpy()
        feature_names.append(factor_name)

    return df, label_name, feature_names


def mad_clip_series(series: pd.Series, n: float = 5.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = x.median()
    mad = (x - median).abs().median()

    if pd.isna(median) or not np.isfinite(median):
        return x

    if pd.isna(mad) or mad == 0 or not np.isfinite(mad):
        return x

    scaled_mad = 1.4826 * mad
    if not np.isfinite(scaled_mad):
        return x

    lower = median - n * scaled_mad
    upper = median + n * scaled_mad

    if not np.isfinite(lower) or not np.isfinite(upper):
        return x

    return x.clip(lower=lower, upper=upper)


def zscore_series(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = x.notna()

    if valid.sum() < 2:
        return result

    mean = x[valid].mean()
    std = x[valid].std(ddof=0)
    if pd.isna(mean) or pd.isna(std) or std == 0 or not np.isfinite(std):
        return result

    result.loc[valid] = ((x[valid] - mean) / std).to_numpy(dtype=float)
    return result


def neutralize_one_cross_section(
    group: pd.DataFrame,
    value_col: str,
    market_cap_col: str = MARKET_CAP_FACTOR_COL,
    industry_col: str = INDUSTRY_COL,
) -> pd.Series:
    y = pd.to_numeric(group[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    mv = pd.to_numeric(group[market_cap_col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    ln_mv = np.full(len(group), np.nan, dtype=float)
    valid_mv = np.isfinite(mv) & (mv > 0)
    ln_mv[valid_mv] = np.log(mv[valid_mv])
    industry = group[industry_col]

    valid = np.isfinite(y) & np.isfinite(ln_mv) & industry.notna().to_numpy()
    if valid.sum() < 2:
        return pd.Series(np.nan, index=group.index, dtype=float)

    x_parts = [ln_mv[valid].reshape(-1, 1)]
    industry_dummies = pd.get_dummies(industry[valid], dtype=float, drop_first=True)
    if industry_dummies.shape[1] > 0:
        x_parts.append(industry_dummies.to_numpy())

    x_valid = np.column_stack(x_parts)
    x_valid = np.column_stack([np.ones(len(x_valid)), x_valid])
    y_valid = y[valid]

    try:
        beta, _, _, _ = np.linalg.lstsq(x_valid, y_valid, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=group.index, dtype=float)

    x_full_parts = [ln_mv.reshape(-1, 1)]
    full_dummies = pd.get_dummies(industry, dtype=float, drop_first=True).reindex(
        columns=industry_dummies.columns,
        fill_value=0.0,
    )
    if full_dummies.shape[1] > 0:
        x_full_parts.append(full_dummies.to_numpy())

    x_full = np.column_stack(x_full_parts)
    x_full = np.column_stack([np.ones(len(x_full)), x_full])

    pred = x_full @ beta
    residual = np.full(len(group), np.nan, dtype=float)
    residual[valid] = y[valid] - pred[valid]
    return pd.Series(residual, index=group.index, dtype=float)


def process_cross_sectional_column(
    df: pd.DataFrame,
    source_col: str,
    date_col: str = COLS.trade_date,
    market_cap_col: str = MARKET_CAP_FACTOR_COL,
    industry_col: str = INDUSTRY_COL,
    mad_n: float = 5.0,
) -> pd.Series:
    def _process_group(group: pd.DataFrame) -> pd.Series:
        clipped = mad_clip_series(group[source_col], n=mad_n)
        temp_group = group.copy()
        temp_group["_clipped_value"] = clipped
        neutralized = neutralize_one_cross_section(
            temp_group,
            value_col="_clipped_value",
            market_cap_col=market_cap_col,
            industry_col=industry_col,
        )
        return zscore_series(neutralized)

    pieces: list[pd.Series] = []
    for _, group in df.groupby(date_col, sort=False):
        pieces.append(_process_group(group))
    processed = pd.concat(pieces).sort_index()
    return processed.reindex(df.index)


def build_step1_dataset(
    base_df: pd.DataFrame,
    data_dir: Path,
    label_keyword: str = "未来n日涨跌",
    mad_n: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    factor_files = list_factor_files(data_dir)
    label_file, feature_files = split_label_and_features(factor_files, label_keyword=label_keyword)

    merged_df, label_name, feature_names = attach_label_and_features(
        base_df=base_df,
        label_file=label_file,
        feature_files=feature_files,
    )

    label_null_count = int(merged_df[label_name].isna().sum())
    merged_df = merged_df.loc[merged_df[label_name].notna()].copy()

    # 行业和市值用于截面中性化，不再作为模型输入特征。
    excluded_from_model_features = {INDUSTRY_COL, MARKET_CAP_FACTOR_COL}
    feature_process_names = [name for name in feature_names if name not in excluded_from_model_features]
    label_processed_col = f"{label_name}{PROCESSED_SUFFIX}"
    feature_processed_cols = [f"{name}{PROCESSED_SUFFIX}" for name in feature_process_names]

    merged_df[label_processed_col] = process_cross_sectional_column(
        merged_df,
        source_col=label_name,
        mad_n=mad_n,
    )

    for feature_name, processed_col in zip(feature_process_names, feature_processed_cols):
        merged_df[processed_col] = process_cross_sectional_column(
            merged_df,
            source_col=feature_name,
            mad_n=mad_n,
        )

    info_cols = [
        col
        for col in merged_df.columns
        if col in base_df.columns and col not in [COLS.row_id, COLS.trade_date, COLS.stock_code]
    ]

    meta = {
        "label_file": label_file.name,
        "feature_files": [path.name for path in feature_files],
        "label_name": label_name,
        "feature_names": feature_names,
        "feature_process_names": feature_process_names,
        "label_null_count": label_null_count,
        "processed_suffix": PROCESSED_SUFFIX,
        "label_processed_col": label_processed_col,
        "feature_processed_cols": feature_processed_cols,
        "info_cols": info_cols,
        "mad_n": mad_n,
    }
    return merged_df.reset_index(drop=True), meta


def prepare_training_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = NEUTRALIZED_TARGET_COL,
    raw_return_col: str = RAW_RETURN_COL,
    original_return_col: str | None = None,
    drop_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    train_df = df.copy()

    if original_return_col is None:
        raise ValueError("original_return_col 不能为空，必须保留原始收益用于后续 IC 评估。")

    train_df[raw_return_col] = train_df[original_return_col]

    drop_feature_cols = set(drop_feature_cols or [])
    # 市值已经参与中性化，默认不再进入模型；这里保留兜底过滤，
    # 防止外部流程手动传入了市值预处理列。
    drop_feature_cols.add(f"{MARKET_CAP_FACTOR_COL}{PROCESSED_SUFFIX}")
    final_feature_cols = [col for col in feature_cols if col not in drop_feature_cols]

    before_rows = len(train_df)
    train_df = train_df.loc[train_df[label_col].notna()].copy()
    after_rows = len(train_df)
    dropped_rows = before_rows - after_rows
    dropped_ratio = dropped_rows / before_rows if before_rows else 0.0

    meta = {
        "label_col": label_col,
        "raw_return_col": raw_return_col,
        "feature_cols": final_feature_cols,
        "dropped_label_na_rows": dropped_rows,
        "dropped_label_na_ratio": dropped_ratio,
        "before_rows": before_rows,
        "after_rows": after_rows,
    }
    return train_df, meta


def sanitize_feature_names(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, str]]:
    rename_map = {col: re.sub(r'[\(\)\{\}\[\],":]', "_", col) for col in feature_cols}
    sanitized_df = df.rename(columns=rename_map).copy()
    sanitized_features = [rename_map[col] for col in feature_cols]
    reverse_map = {new: old for old, new in rename_map.items()}
    return sanitized_df, sanitized_features, rename_map, reverse_map
