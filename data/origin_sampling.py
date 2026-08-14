"""Deterministic forecast-start schedules used by controlled experiments."""

from __future__ import annotations

import json
from pathlib import Path


def load_training_origins(config_value, repo_root: Path) -> list[int] | None:
    """Return an explicit, validated training-start list or ``None``.

    ``None`` preserves the legacy modulo-stride sampler.  A configured list is
    intentionally global: every store and every bottom-level series receives
    the identical set of calendar forecast starts.
    """
    if config_value is None:
        return None
    if getattr(config_value, "mode", None) != "explicit_list":
        raise ValueError("training_origin_sampling.mode must be 'explicit_list'")
    path_value = getattr(config_value, "origins_file", None)
    if not path_value:
        raise ValueError("explicit_list sampling requires origins_file")
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    origins = payload.get("origins")
    if not isinstance(origins, list) or not all(isinstance(value, int) for value in origins):
        raise ValueError(f"Invalid origins list in {path}")
    if origins != sorted(set(origins)):
        raise ValueError(f"Origins must be strictly increasing and unique: {path}")
    return origins
