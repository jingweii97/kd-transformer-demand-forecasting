"""Repair only objective/training-state fields in a completed common evaluation.

This utility never loads a model for prediction and never calls a metric
function. It verifies checkpoint hashes, replaces the erroneous inference-time
metadata, and asserts that all saved numeric evaluation values are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_checkpoints import checkpoint_training_state, objective_label


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint_path(reported_path, experiment_dir):
    """Accept the persisted cluster path or its same-file local counterpart."""
    if os.path.isfile(reported_path):
        return reported_path
    local_path = os.path.join(experiment_dir, os.path.basename(reported_path))
    if os.path.isfile(local_path):
        return local_path
    raise FileNotFoundError(f"Checkpoint missing: {reported_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="outputs/teacher/tft64_wrmsse_informed")
    args = parser.parse_args()

    common_path = os.path.join(args.experiment_dir, "common_validation_evaluation.csv")
    metadata_path = os.path.join(args.experiment_dir, "evaluation_metadata.json")
    hierarchy_path = os.path.join(args.experiment_dir, "hierarchy_validation_evaluation.csv")
    for path in (common_path, metadata_path, hierarchy_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    common = pd.read_csv(common_path)
    metric_columns = [
        column for column in common.columns
        if column.startswith("validation_") or column in {
            "aggregate_percentage_bias", "actual_total", "predicted_total",
            "parameter_count", "checkpoint_size_bytes",
        }
    ]
    before_metrics = common[metric_columns].copy(deep=True)
    hierarchy_hash_before = sha256(hierarchy_path)
    wi_rows = common["teacher_version"].eq("WRMSSE-informed TFT-64")
    if int(wi_rows.sum()) != 5:
        raise AssertionError("Expected exactly five unique WRMSSE-informed rows")

    repaired = []
    for index, row in common.loc[wi_rows].iterrows():
        reported_path = row["checkpoint_path"]
        path = resolve_checkpoint_path(reported_path, args.experiment_dir)
        digest = sha256(path)
        if digest != row["checkpoint_SHA256"]:
            raise AssertionError(f"CSV hash mismatch for {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        label = objective_label(checkpoint["hyper_parameters"]["loss"])
        state = checkpoint_training_state(path)
        common.at[index, "objective"] = label
        common.at[index, "internal_epoch"] = state["internal_epoch"]
        common.at[index, "global_step"] = state["global_step"]
        repaired.append({
            "path": reported_path, "sha256": digest, "objective": label, **state
        })

    if not common[metric_columns].equals(before_metrics):
        raise AssertionError("A numeric evaluation metric changed during metadata repair")
    common.to_csv(common_path, index=False)

    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    state_by_path = {entry["path"]: entry for entry in repaired}
    for entry in metadata["checkpoint_paths_and_hashes"]:
        repaired_entry = state_by_path.get(entry["path"])
        if repaired_entry is None:
            raise AssertionError(f"Metadata checkpoint not in evaluation rows: {entry['path']}")
        if entry["sha256"] != repaired_entry["sha256"]:
            raise AssertionError(f"Metadata hash mismatch: {entry['path']}")
        entry.update({
            "internal_epoch": repaired_entry["internal_epoch"],
            "global_step": repaired_entry["global_step"],
            "objective": repaired_entry["objective"],
        })
    metadata["metadata_reporting_correction"] = {
        "prediction_generation_invoked": False,
        "metric_computation_invoked": False,
        "numeric_metrics_unchanged": True,
        "hierarchy_file_sha256_unchanged": sha256(hierarchy_path) == hierarchy_hash_before,
        "repaired_checkpoints": repaired,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata["metadata_reporting_correction"], indent=2))


if __name__ == "__main__":
    main()
