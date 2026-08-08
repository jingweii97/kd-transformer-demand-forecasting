"""Shared authoritative WRMSSE primitives.

Both the exact hierarchy evaluator and the bottom-level supervised surrogate use
these functions.  Keeping the first-difference scale and economic-weight
numerator here prevents the two paths from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RMSSE_SCALE_FALLBACK = 1e-4


def compute_rmsse_scale(series: np.ndarray) -> tuple[float, str]:
    """Return the M5 mean-squared first-difference denominator and audit reason."""
    values = np.asarray(series)
    if len(values) < 2:
        return RMSSE_SCALE_FALLBACK, "insufficient_length"

    first_nonzero_idx = int(np.argmax(values > 0))
    if values[first_nonzero_idx] == 0:
        return RMSSE_SCALE_FALLBACK, "all_zero"

    trimmed = values[first_nonzero_idx:]
    if len(trimmed) < 2:
        return RMSSE_SCALE_FALLBACK, "insufficient_length"

    scale = float(np.mean(np.diff(trimmed.astype(float)) ** 2))
    if scale <= 0:
        return RMSSE_SCALE_FALLBACK, "zero_variance"
    return scale, "valid"


def economic_weight_numerators(
    frame: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Aggregate training-period dollar sales used by M5 economic weights."""
    required = {*group_cols, "sales", "sell_price"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing economic-weight columns: {sorted(missing)}")

    work = frame.loc[:, [*group_cols, "sales", "sell_price"]].copy()
    work["dollar_value"] = work["sales"] * work["sell_price"]
    # Deliberately retain pandas' default observed=False behavior because this
    # helper is also used by the established exact hierarchy evaluator.
    return work.groupby(group_cols)["dollar_value"].sum().reset_index()


def normalize_economic_weight(numerator: float, total: float) -> float:
    """Normalize one dollar-sales numerator using the hierarchy-level total."""
    return float(numerator / total) if total > 0 else 0.0
