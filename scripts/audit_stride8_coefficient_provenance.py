"""Compare legacy-stride and explicit-origin WRMSSE coefficient provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wrmsse_informed import build_wrmsse_informed_coefficients
from utils.config import Config, load_config
from utils.paths import resolve_path


def _legacy_stride_config(cfg: Config) -> Config:
    """Return an in-memory copy that reproduces legacy modulo-stride auditing."""
    payload = cfg.to_dict()
    payload["dataset"].pop("training_origin_sampling", None)
    return Config(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--experiment", default="phase_stride8_wis_v2")
    parser.add_argument(
        "--output",
        default="outputs/student/no_kd/phase_stride8_wis_v2/wrmsse_coefficient_audit_stride8_corrected.json",
    )
    args = parser.parse_args()

    output = resolve_path(args.output)
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite provenance sidecar: {output}")
    cfg = load_config(env_name=args.env, experiment_name=args.experiment)

    # Recreate the historical calculation only for comparison, then compute the
    # corrected explicit-origin variant used by queued stride-8 training.
    legacy = build_wrmsse_informed_coefficients(
        _legacy_stride_config(cfg), objective_config=cfg.student
    )
    corrected = build_wrmsse_informed_coefficients(cfg, objective_config=cfg.student)
    if legacy.by_series.keys() != corrected.by_series.keys():
        raise AssertionError("Legacy and corrected coefficient series sets differ")

    ids = sorted(legacy.by_series)
    before = np.asarray([legacy.by_series[key] for key in ids], dtype=np.float64)
    after = np.asarray([corrected.by_series[key] for key in ids], dtype=np.float64)
    absolute = np.abs(after - before)
    relative = absolute / np.maximum(np.abs(before), 1e-30)
    comparison = {
        "series_count": len(ids),
        "max_absolute_difference": float(absolute.max()),
        "max_relative_difference": float(relative.max()),
        "allclose_rtol_1e-12_atol_1e-12": bool(
            np.allclose(before, after, rtol=1e-12, atol=1e-12)
        ),
        "exact_equal": bool(np.array_equal(before, after)),
    }
    if not comparison["allclose_rtol_1e-12_atol_1e-12"]:
        raise AssertionError("Corrected stride-8 coefficients differ from legacy values")

    report = {
        "audit": "stride8_wrmsse_coefficient_provenance_correction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparison": comparison,
        "legacy_recomputed_audit": legacy.audit,
        "corrected_explicit_origin_audit": corrected.audit,
        "note": (
            "The prior WIS metadata is intentionally preserved. This sidecar records "
            "the corrected exact-origin provenance and a read-only numerical comparison."
        ),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
