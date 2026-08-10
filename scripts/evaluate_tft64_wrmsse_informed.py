"""Run the existing common validation evaluator for retained WRMSSE TFTs.

No hierarchy, WRMSSE, RMSSE-scale, economic-weight, or point-metric formula is
defined here.  Those remain exclusively in ``scripts.evaluate_models`` and are
executed through ``scripts.evaluate_checkpoints.evaluate_checkpoint``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_checkpoints import (
    checkpoint_training_state,
    evaluate_checkpoint,
    prepare_validation_context,
)
from utils.config import load_config


EXPERIMENT_DIR = "outputs/teacher/tft64_wrmsse_informed"
CANDIDATES = [
    "tft64-wrmsse-informed-epoch=epoch=03-val_loss=val_loss=1.471272.ckpt",
    "tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt",
    "tft64-wrmsse-informed-epoch=epoch=13-val_loss=val_loss=1.468511.ckpt",
    "tft64-wrmsse-informed-epoch=epoch=14-val_loss=val_loss=1.470980.ckpt",
    "tft64-wrmsse-informed-epoch=epoch=16-val_loss=val_loss=1.461851.ckpt",
    "last.ckpt",
]
REFERENCE_METRICS = "artifacts/teacher_strength_audit_20260807_113257/predefined_ensemble_metrics.csv"
REFERENCE_HIERARCHY = "artifacts/teacher_strength_audit_20260807_113257/student_vs_huber_hierarchy_wrmsse.csv"
STUDENT_REFERENCE = "artifacts/teacher_student_comparability_20260806_224049/common_validation_metrics.csv"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_unique_candidates():
    by_hash = {}
    aliases = {}
    for name in CANDIDATES:
        path = os.path.join(EXPERIMENT_DIR, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required retained checkpoint is missing: {path}")
        digest = sha256(path)
        if digest in by_hash:
            aliases.setdefault(digest, []).append(path)
        else:
            by_hash[digest] = path

    expected_pair = {
        os.path.join(EXPERIMENT_DIR, "tft64-wrmsse-informed-epoch=epoch=16-val_loss=val_loss=1.461851.ckpt"),
        os.path.join(EXPERIMENT_DIR, "last.ckpt"),
    }
    observed_duplicates = {path for paths in aliases.values() for path in paths}
    duplicate_members = {
        by_hash[digest] for digest in aliases
    } | observed_duplicates
    if duplicate_members and duplicate_members != expected_pair:
        raise RuntimeError(f"Unexpected duplicate checkpoint hashes: {sorted(duplicate_members)}")
    if len(aliases) != 1 or not duplicate_members:
        raise RuntimeError("Expected only epoch 16 and last.ckpt to share one SHA-256")
    return by_hash, aliases


def _levels_from_string(value):
    values = [float(item) for item in re.findall(r"np\.float64\(([^)]+)\)", value)]
    if len(values) != 12:
        raise ValueError("Archived reference row does not contain all 12 hierarchy scores")
    return values


def archived_reference_rows():
    metrics = pd.read_csv(REFERENCE_METRICS)
    hierarchy = pd.read_csv(REFERENCE_HIERARCHY)
    student_metrics = pd.read_csv(STUDENT_REFERENCE)

    huber = metrics.loc[metrics["model"] == "Huber epoch 5"].iloc[0]
    huber_levels = _levels_from_string(huber["level_scores"])
    student = student_metrics.loc[student_metrics["model"] == "Student (No KD)"].iloc[0]
    student_levels = hierarchy["student_WRMSSE"].astype(float).tolist()
    actual_total = float(huber["actual_total"])
    student_bias = float(student["Bias"])

    def row(name, source, values, levels, bias, predicted_total):
        result = {
            "teacher_version": name,
            "reference_source": source,
            "checkpoint": None,
            "checkpoint_path": None,
            "checkpoint_SHA256": None,
            "internal_epoch": None,
            "global_step": None,
            "objective": "archived authoritative reference",
            "validation_WRMSSE": float(values["WRMSSE"]),
            "validation_MAE": float(values["MAE"]),
            "validation_RMSE": float(values["RMSE"]),
            "validation_WAPE": float(values["WAPE"]),
            "validation_seasonal_MASE": float(values["MASE"]),
            "aggregate_percentage_bias": float(bias),
            "actual_total": actual_total,
            "predicted_total": float(predicted_total),
        }
        result.update({f"validation_WRMSSE_Level_{index}": score for index, score in enumerate(levels, 1)})
        return result

    return [
        row(
            "Supervised student", STUDENT_REFERENCE, student, student_levels,
            student_bias, actual_total * (1.0 + student_bias),
        ),
        row(
            "Huber TFT epoch 5", REFERENCE_METRICS, huber, huber_levels,
            float(huber["aggregate_percentage_bias"]), float(huber["predicted_total"]),
        ),
    ]


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()

    unique, aliases = resolve_unique_candidates()
    plan = {
        "evaluator": "scripts/evaluate_checkpoints.py::evaluate_checkpoint",
        "metric_definitions": "scripts/evaluate_models.py",
        "unique_checkpoints": [
            {"path": path, "sha256": digest} for digest, path in unique.items()
        ],
        "deduplicated_aliases": aliases,
        "validation_boundary": {"start": 1526, "end": 1553, "horizon": 28},
        "id_ood_data_used": False,
        "training_or_retraining": False,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.print_plan:
        return

    output_paths = [
        os.path.join(EXPERIMENT_DIR, name)
        for name in (
            "common_validation_evaluation.csv",
            "hierarchy_validation_evaluation.csv",
            "evaluation_metadata.json",
        )
    ]
    existing = [path for path in output_paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing evaluation output(s): {existing}")

    cfg = load_config(args.env, "tft64_wrmsse_informed")
    ds_dir, train_end, weights, scales, mase_scales, series_ids = prepare_validation_context(cfg)
    wrmsse_rows = []
    for digest, path in unique.items():
        result = evaluate_checkpoint(
            path, ds_dir, cfg, None, train_end, weights, scales, mase_scales, series_ids
        )
        if result is None:
            raise RuntimeError(f"Authoritative evaluator failed for {path}")
        result["teacher_version"] = "WRMSSE-informed TFT-64"
        result["reference_source"] = "new authoritative validation evaluation"
        result["checkpoint_path"] = os.path.abspath(path)
        wrmsse_rows.append(result)

    common = pd.DataFrame(archived_reference_rows() + wrmsse_rows)
    common.to_csv(output_paths[0], index=False)
    hierarchy_columns = [f"validation_WRMSSE_Level_{index}" for index in range(1, 13)]
    hierarchy = common.melt(
        id_vars=["teacher_version", "checkpoint", "checkpoint_path", "checkpoint_SHA256"],
        value_vars=hierarchy_columns,
        var_name="level",
        value_name="WRMSSE",
    )
    hierarchy["level"] = hierarchy["level"].str.extract(r"(\d+)$").astype(int)
    hierarchy.to_csv(output_paths[1], index=False)

    metadata = {
        "checkpoint_paths_and_hashes": [
            {
                "path": os.path.abspath(path),
                "sha256": digest,
                **checkpoint_training_state(path),
            }
            for digest, path in unique.items()
        ],
        "deduplicated_aliases": aliases,
        "evaluator_script": "scripts/evaluate_checkpoints.py::evaluate_checkpoint",
        "metric_definition_module": "scripts/evaluate_models.py",
        "git_commit": git_commit(),
        "validation_boundary": {"start": 1526, "end": 1553, "horizon": 28},
        "number_of_series": int(len(series_ids)),
        "number_of_forecast_rows": int(len(series_ids) * cfg.dataset.prediction_window),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id_ood_data_used": False,
        "training_or_retraining": False,
        "reference_rows": [REFERENCE_METRICS, REFERENCE_HIERARCHY, STUDENT_REFERENCE],
    }
    with open(output_paths[2], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
