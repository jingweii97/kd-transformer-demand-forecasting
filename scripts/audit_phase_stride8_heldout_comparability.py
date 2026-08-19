"""Read-only integrity audit for the final stride-8 v2 held-out evaluation.

This deliberately does not load any model checkpoint or make predictions.  It
checks the completed evaluation artifact, reconstructs the source forecast
keys/actuals for every held-out origin, and records the evaluation code-path
that supplied common WRMSSE weights and scales to all three learned models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_from_cache, resolve_stores
from data.origin_sampling import load_training_origins
from utils.config import load_config
from utils.paths import get_dataset_dir, resolve_path


SERIES_COUNT = 30_490
HORIZON = 28
EXPECTED_ROWS = SERIES_COUNT * HORIZON
EVAL_EXPERIMENT = "phase_stride8_v2_validation_selected_heldout"
MODEL_LABELS = {
    "TFT Teacher",
    "Student Without KD",
    "Student-WIKD stride-8 v2",
}
ALL_MODEL_LABELS = MODEL_LABELS | {"Seasonal Naive"}
HORIZON_BANDS = {"Overall (1-28)", "Short (1-7)", "Medium (8-14)", "Long (15-28)"}


def source_hashes(path: str) -> dict[str, str]:
    """Provide raw and LF-canonical hashes so a read-only Windows mirror works."""
    with open(path, "rb") as handle:
        raw = handle.read()
    return {
        "raw": hashlib.sha256(raw).hexdigest(),
        "lf_canonical": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def stable_frame_hash(frame: pd.DataFrame) -> str:
    """Hash sorted key/actual rows without retaining forecast-sized arrays."""
    values = pd.util.hash_pandas_object(frame, index=False, categorize=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def expected_windows() -> list[tuple[str, int]]:
    windows = [
        ("ID Reference", 1554),
        ("Event-Intensive OOD", 1819),
        ("Extended-Gap OOD", 1914),
    ]
    windows.extend((f"Overall Test Stream (Origin {origin})", origin) for origin in range(1554, 1912, 7))
    assert len(windows) == 55
    return windows


def audit_results_csv(csv_path: str) -> dict:
    results = pd.read_csv(csv_path)
    expected = dict(expected_windows())
    expected_pairs = {(model, band) for model in ALL_MODEL_LABELS for band in HORIZON_BANDS}
    observed_names = set(results["Window"])
    if observed_names != set(expected):
        raise AssertionError("Held-out CSV window names do not exactly match the fixed protocol")
    if len(results) != len(expected) * len(expected_pairs):
        raise AssertionError(f"Expected {len(expected) * len(expected_pairs)} result rows, got {len(results)}")

    numeric = ["WRMSSE", "MAE", "RMSE", "MASE", "WAPE", "Inference_Time_Sec", "Inference_Time_Per_1k_Sec"]
    window_rows = {}
    for name, origin in expected.items():
        group = results.loc[results["Window"] == name]
        pairs = set(zip(group["Model"], group["Horizon"]))
        if len(group) != len(expected_pairs) or pairs != expected_pairs:
            raise AssertionError(f"Incomplete or duplicate model/horizon records for {name}")
        if not np.isfinite(group[numeric].to_numpy(dtype=float)).all():
            raise AssertionError(f"Non-finite completed metric in {name}")
        window_rows[name] = {"origin": origin, "result_rows": int(len(group)), "all_metrics_finite": True}

    return {
        "csv": os.path.basename(csv_path),
        "result_row_count": int(len(results)),
        "window_count": len(expected),
        "expected_rows_per_window": len(expected_pairs),
        "every_window_has_exactly_one_record_per_model_horizon_pair": True,
        "all_reported_metrics_finite": True,
        "windows": window_rows,
    }


def audit_source_keys_and_actuals(dataset_dir: str, store_filter: str, origins: list[int]) -> dict:
    """Reconstruct source ground truth only; no models or prediction datasets."""
    aggregate = {
        origin: {"rows": 0, "ids": set(), "key_hashes": [], "target_hashes": []}
        for origin in origins
    }
    stores = resolve_stores(store_filter)
    for store in stores:
        frame = load_from_cache(artifacts_dir=dataset_dir, store_filter=store)
        if frame is None:
            raise AssertionError(f"Missing cached partition for {store}")
        frame = frame[["id", "time_idx", "sales"]].copy()
        frame["id"] = frame["id"].astype(str)
        for origin in origins:
            part = frame.loc[
                frame["time_idx"].between(origin, origin + HORIZON - 1), ["id", "time_idx", "sales"]
            ].sort_values(["id", "time_idx"], kind="stable").reset_index(drop=True)
            keys = part[["id", "time_idx"]]
            if keys.duplicated().any():
                raise AssertionError(f"d{origin}: duplicate bottom-level forecast key within store {store}")
            counts = part.groupby("id", sort=False)["time_idx"].size()
            if not (counts == HORIZON).all():
                raise AssertionError(f"d{origin}: store {store} has incomplete series targets")
            if not np.isfinite(part["sales"].to_numpy(dtype=float)).all():
                raise AssertionError(f"d{origin}: non-finite actual target in store {store}")
            state = aggregate[origin]
            ids = set(part["id"].unique())
            if state["ids"].intersection(ids):
                raise AssertionError(f"d{origin}: same bottom-level ID occurs in multiple store partitions")
            state["rows"] += int(len(part))
            state["ids"].update(ids)
            state["key_hashes"].append((store, stable_frame_hash(keys)))
            state["target_hashes"].append((store, stable_frame_hash(part)))

    report = {}
    for origin, state in aggregate.items():
        if state["rows"] != EXPECTED_ROWS:
            raise AssertionError(f"d{origin}: expected {EXPECTED_ROWS} target rows, got {state['rows']}")
        if len(state["ids"]) != SERIES_COUNT:
            raise AssertionError(f"d{origin}: expected {SERIES_COUNT} distinct IDs")
        key_hash = hashlib.sha256(json.dumps(state["key_hashes"], sort_keys=True).encode()).hexdigest()
        target_hash = hashlib.sha256(json.dumps(state["target_hashes"], sort_keys=True).encode()).hexdigest()
        report[f"d{origin}"] = {
            "origin": origin,
            "target_days": [origin, origin + HORIZON - 1],
            "bottom_level_rows": state["rows"],
            "series_count": len(state["ids"]),
            "unique_key_count": state["rows"],
            "keys_unique": True,
            "exactly_28_targets_per_series": True,
            "actuals_finite": True,
            "forecast_key_sha256": key_hash,
            "actual_target_sha256": target_hash,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--experiment", default="phase_stride8_wikd_v2")
    parser.add_argument("--evaluation-dir", default=f"outputs/evaluation/{EVAL_EXPERIMENT}")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    evaluation_dir = resolve_path(args.evaluation_dir)
    output = resolve_path(args.output or os.path.join(args.evaluation_dir, "comparability_provenance_audit.json"))
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite existing provenance sidecar: {output}")

    run_state_path = os.path.join(evaluation_dir, "run_state.json")
    results_path = os.path.join(evaluation_dir, "evaluation_results_incremental.csv")
    with open(run_state_path, encoding="utf-8") as handle:
        run_state = json.load(handle)

    eval_script = resolve_path("scripts/evaluate_models.py")
    live_script_hashes = source_hashes(eval_script)
    if run_state["script_hash"] not in set(live_script_hashes.values()):
        raise AssertionError("Current evaluator source does not match the completed run_state script hash")
    if int(run_state["series_count"]) != SERIES_COUNT or int(run_state["horizon"]) != HORIZON:
        raise AssertionError("run_state does not describe the final full-M5, 28-day protocol")

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    training_origins = load_training_origins(cfg.dataset.training_origin_sampling, os.getcwd())
    if training_origins != list(range(91, 1332, 8)):
        raise AssertionError("Final model configuration does not contain the exact d(91 + 8*k) stride-8 schedule")
    if int(cfg.dataset.prediction_window) != HORIZON:
        raise AssertionError("Unexpected prediction horizon in final stride-8 configuration")

    results_audit = audit_results_csv(results_path)
    unique_origins = sorted({origin for _, origin in expected_windows()})
    target_audit = audit_source_keys_and_actuals(
        get_dataset_dir(cfg), getattr(cfg.environment, "store_filter", ""), unique_origins
    )
    report = {
        "audit": "phase_stride8_v2_heldout_comparability",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "inference_rerun": False,
        "evaluation_artifact": {
            "directory": args.evaluation_dir,
            "run_state_sha256_matches_live_evaluator": True,
            "evaluator_script_sha256": run_state["script_hash"],
            "evaluator_script_hash_verification": {
                "matched": True,
                "raw": live_script_hashes["raw"],
                "lf_canonical": live_script_hashes["lf_canonical"],
                "note": "LF canonicalization is only relevant when auditing a Windows mirror of the cluster source.",
            },
            "checkpoint_hashes": {
                "teacher": run_state["teacher_hash"],
                "wis": run_state["student_nokd_hash"],
                "wikd": run_state["student_kd_hash"],
            },
        },
        "forecast_key_and_actual_target_audit": {
            "complete_evaluation_windows": 55,
            "unique_origins": len(unique_origins),
            "expected_bottom_level_rows_per_origin": EXPECTED_ROWS,
            "same_keys_and_actuals_for_teacher_wis_wikd": True,
            "basis": "The verified evaluator creates one decoded-index/actuals array per store-origin, then passes its identically ordered part_loader to Teacher, WIS, and WIKD; its completed run asserted shape equality and finite predictions for each model.",
            "source_target_hashes": target_audit,
            "d1554_aliases": ["ID Reference", "Overall Test Stream (Origin 1554)"],
        },
        "prediction_finiteness": {
            "teacher_wis_wikd_runtime_assertions_completed": True,
            "basis": "The hash-matched evaluator asserts np.isfinite(preds).all() for each learned model/store/window before persisting each completed result window. Raw prediction tensors are intentionally not persisted, so this audit does not regenerate them.",
        },
        "wrmsse_provenance": {
            "same_scales_and_weights_for_teacher_wis_wikd": True,
            "basis": "The hash-matched evaluator computes weights_dict and scales_dict once from the common training frame through d1359, before the held-out window loop, and passes those same objects to compute_hierarchical_wrmsse for every model and horizon band.",
            "training_end_day": int(run_state["train_end"]),
        },
        "stride_clarification": {
            "run_state_stride": int(run_state["stride"]),
            "meaning": "The seven-day stride of the 52-origin Overall Test Stream (d1554, d1561, ..., d1911), not the final model-training origin schedule.",
            "final_model_training_origin_rule": "origin_k = d(91 + 8*k), k=0..155",
            "final_model_training_stride": 8,
            "training_origin_count_per_series": len(training_origins),
            "training_origin_first_last": [training_origins[0], training_origins[-1]],
        },
        "completed_results_csv_audit": results_audit,
    }

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({"status": "pass", "output": output, "unique_origins": len(unique_origins)}, indent=2))


if __name__ == "__main__":
    main()
