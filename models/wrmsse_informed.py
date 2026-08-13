"""Training-only coefficient construction for the WRMSSE-informed objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.cache import get_cache_path, is_cache_valid, resolve_stores
from utils.paths import get_dataset_dir
from utils.wrmsse import (
    compute_rmsse_scale,
    economic_weight_numerators,
    normalize_economic_weight,
)


@dataclass(frozen=True)
class WRMSSEInformedCoefficients:
    by_series: dict[str, float]
    audit: dict[str, Any]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _mass_share(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(values) * fraction)))
    total = float(values.sum())
    if total <= 0:
        return 0.0
    return float(np.sort(values)[-count:].sum() / total)


def _window_count(
    first_time: int,
    last_time: int,
    encoder_length: int,
    prediction_length: int,
    stride: int,
) -> int:
    first_prediction = first_time + encoder_length
    last_prediction = last_time - prediction_length + 1
    if last_prediction < first_prediction:
        return 0
    starts = np.arange(first_prediction, last_prediction + 1, dtype=np.int64)
    return int(np.count_nonzero(starts % stride == 0)) if stride > 1 else len(starts)


def build_wrmsse_informed_coefficients(
    cfg, objective_config=None
) -> WRMSSEInformedCoefficients:
    """Compute fixed bottom-level coefficients using rows at or before train_end only.

    ``objective_config`` may be the teacher or student configuration section.
    The coefficient formula is deliberately shared: it uses the same training
    boundary, economic-weight window, RMSSE stabilization, and window-frequency
    normalization whichever point model consumes the coefficients.
    """
    objective_config = objective_config if objective_config is not None else cfg.teacher
    train_end = int(cfg.dataset.splits.train.end)
    train_start = int(cfg.dataset.splits.train.start)
    weight_days = int(getattr(objective_config, "wrmsse_weight_days", 28))
    floor_quantile = float(getattr(objective_config, "wrmsse_scale_floor_quantile", 0.01))
    epsilon = float(getattr(objective_config, "wrmsse_epsilon", 1e-8))
    stride = int(getattr(cfg.dataset, "window_stride", 1))
    encoder_length = int(cfg.dataset.lookback_window)
    prediction_length = int(cfg.dataset.prediction_window)
    dataset_dir = get_dataset_dir(cfg)

    if not 0.0 <= floor_quantile <= 1.0:
        raise ValueError("wrmsse_scale_floor_quantile must be in [0, 1]")
    if epsilon <= 0:
        raise ValueError("wrmsse_epsilon must be positive")

    records: list[pd.DataFrame] = []
    max_time_seen = -1
    for store in resolve_stores(cfg.environment.store_filter):
        if not is_cache_valid(dataset_dir, store):
            raise FileNotFoundError(f"Missing or stale cache for {store}")
        frame = pd.read_parquet(
            get_cache_path(dataset_dir, store),
            engine="pyarrow",
            columns=["id", "time_idx", "sales", "sell_price"],
            filters=[("time_idx", ">=", train_start), ("time_idx", "<=", train_end)],
        )
        frame["id"] = frame["id"].astype(str)
        max_time_seen = max(max_time_seen, int(frame["time_idx"].max()))

        weight_frame = frame[frame["time_idx"] > train_end - weight_days]
        numerators = economic_weight_numerators(weight_frame, ["id"]).rename(
            columns={"dollar_value": "economic_numerator"}
        )

        series_rows = []
        for series_id, group in frame.groupby("id", sort=False, observed=True):
            ordered = group.sort_values("time_idx")
            first_time = int(ordered["time_idx"].iloc[0])
            last_time = int(ordered["time_idx"].iloc[-1])
            full_range = np.arange(train_start, train_end + 1)
            sales = (
                ordered.set_index("time_idx")["sales"]
                .reindex(full_range, fill_value=0)
                .to_numpy()
            )
            scale, reason = compute_rmsse_scale(sales)
            series_rows.append(
                {
                    "id": str(series_id),
                    "scale": scale,
                    "scale_reason": reason,
                    "history_observations": int(ordered["time_idx"].nunique()),
                    "first_time": first_time,
                    "last_time": last_time,
                    "window_count": _window_count(
                        train_start,
                        last_time,
                        encoder_length,
                        prediction_length,
                        stride,
                    ),
                }
            )
        store_stats = pd.DataFrame(series_rows).merge(numerators, on="id", how="left")
        store_stats["economic_numerator"] = store_stats["economic_numerator"].fillna(0.0)
        records.append(store_stats)
        del frame

    stats = pd.concat(records, ignore_index=True)
    if stats["id"].duplicated().any():
        raise AssertionError("Each bottom-level series must appear exactly once")
    if max_time_seen > train_end:
        raise AssertionError("Post-training rows entered coefficient construction")

    valid_scales = stats.loc[stats["scale_reason"] == "valid", "scale"].to_numpy(float)
    if len(valid_scales) == 0:
        raise ValueError("No valid training-history RMSSE scales were found")
    scale_floor = float(np.quantile(valid_scales, floor_quantile))

    total_economic_value = float(stats["economic_numerator"].sum())
    stats["economic_weight"] = stats["economic_numerator"].map(
        lambda value: normalize_economic_weight(value, total_economic_value)
    )
    stats["stabilized_scale"] = np.maximum(stats["scale"].to_numpy(float), scale_floor)
    stats["raw_coefficient"] = stats["economic_weight"] / (
        stats["stabilized_scale"] + epsilon
    )

    window_counts = stats["window_count"].to_numpy(np.int64)
    raw = stats["raw_coefficient"].to_numpy(float)
    total_windows = int(window_counts.sum())
    if total_windows <= 0:
        raise ValueError("No training windows were found")
    normalization = float(np.dot(raw, window_counts) / total_windows)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(f"Invalid coefficient normalization constant: {normalization}")
    stats["normalized_coefficient"] = raw / normalization

    normalized = stats["normalized_coefficient"].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise ValueError("Non-finite WRMSSE-informed coefficient")

    reason_counts = stats["scale_reason"].value_counts().to_dict()
    raw_sum = float(raw.sum())
    effective_sample_size = float(raw_sum**2 / np.square(raw).sum())
    equal_windows = bool(np.all(window_counts == window_counts[0]))
    unique_mean = float(raw.mean())
    pathological_reasons = []
    top_1_share = _mass_share(raw, 0.01)
    if top_1_share > 0.50:
        pathological_reasons.append("top 1% holds more than 50% of coefficient mass")
    if effective_sample_size < 0.01 * len(stats):
        pathological_reasons.append("effective sample size is below 1% of series count")
    if float(normalized.max()) > 1000.0:
        pathological_reasons.append("maximum normalized coefficient exceeds 1000")

    audit = {
        "provenance": {
            "maximum_time_idx_used": max_time_seen,
            "train_end": train_end,
            "economic_weight_window_start_exclusive": train_end - weight_days,
            "economic_weight_window_end_inclusive": train_end,
            "held_out_targets_used": False,
            "validation_targets_used": False,
        },
        "series_count": int(len(stats)),
        "scale_floor_rule": f"training-valid scale quantile q={floor_quantile:g}",
        "scale_floor": scale_floor,
        "epsilon": epsilon,
        "rmsse_scales": {
            **_quantiles(stats["scale"].to_numpy(float)),
            "valid": int(reason_counts.get("valid", 0)),
            "all_zero": int(reason_counts.get("all_zero", 0)),
            "zero_variance": int(reason_counts.get("zero_variance", 0)),
            "insufficient_history": int(reason_counts.get("insufficient_length", 0)),
            "fallback": int((stats["scale_reason"] != "valid").sum()),
        },
        "raw_coefficients": {
            **_quantiles(raw),
            "top_1_percent_mass_share": top_1_share,
            "top_5_percent_mass_share": _mass_share(raw, 0.05),
            "top_10_percent_mass_share": _mass_share(raw, 0.10),
            "effective_sample_size": effective_sample_size,
        },
        "normalization_constant": normalization,
        "training_window_count": total_windows,
        "windows_per_series_min": int(window_counts.min()),
        "windows_per_series_max": int(window_counts.max()),
        "window_counts_equal": equal_windows,
        "unique_series_raw_mean": unique_mean,
        "window_mean_equals_unique_mean": bool(equal_windows and np.isclose(normalization, unique_mean)),
        "normalized_window_mean": float(np.dot(normalized, window_counts) / total_windows),
        "pathological": bool(pathological_reasons),
        "pathological_reasons": pathological_reasons,
    }
    return WRMSSEInformedCoefficients(
        by_series=dict(zip(stats["id"], stats["normalized_coefficient"].astype(float))),
        audit=audit,
    )
