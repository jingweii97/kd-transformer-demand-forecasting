"""Evaluate retained student checkpoints with the existing common evaluator.

The metric implementation remains in ``scripts.audit_comparability``.  This
script only discovers retained checkpoint files, deduplicates them by SHA-256,
calls that evaluator once for all unique candidates, and persists the exact
validation-WRMSSE selection manifest for the downstream held-out launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.paths import resolve_path


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "internal_epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }


def unique_checkpoints(run_dir: str) -> tuple[list[dict], dict[str, list[str]]]:
    paths = sorted(
        (os.path.join(run_dir, name) for name in os.listdir(run_dir) if name.endswith(".ckpt")),
        key=lambda path: (os.path.basename(path) == "last.ckpt", os.path.basename(path)),
    )
    if not paths:
        raise FileNotFoundError(f"No retained .ckpt files exist in: {run_dir}")

    by_sha: dict[str, dict] = {}
    aliases: dict[str, list[str]] = {}
    for path in paths:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"Checkpoint is missing or empty: {path}")
        digest = sha256(path)
        absolute = os.path.abspath(path)
        if digest in by_sha:
            aliases[digest].append(absolute)
            continue
        by_sha[digest] = {
            "checkpoint_path": absolute,
            "checkpoint_sha256": digest,
            "aliases": [absolute],
            **checkpoint_state(path),
        }
        aliases[digest] = by_sha[digest]["aliases"]
    return list(by_sha.values()), aliases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--experiment", default="full")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant-label", required=True)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Student run directory does not exist: {run_dir}")
    output_dir = os.path.join(run_dir, "common_validation_evaluation")
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite existing validation selection output: {output_dir}")

    candidates, aliases = unique_checkpoints(run_dir)
    command = [
        sys.executable,
        "scripts/audit_comparability.py",
        "--env", args.env,
        "--experiment", args.experiment,
        "--output-dir", output_dir,
    ]
    labels = []
    for index, candidate in enumerate(candidates, 1):
        label = f"{args.variant_label} candidate {index}"
        candidate["model_label"] = label
        labels.append(label)
        command.extend(["--model", "student", label, candidate["checkpoint_path"]])

    print("COMMON VALIDATION EVALUATOR COMMAND:")
    print(" ".join(command))
    print(json.dumps({"candidates": candidates, "deduplicated_aliases": aliases}, indent=2))
    subprocess.run(command, check=True)

    metrics_path = os.path.join(output_dir, "common_validation_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    if set(metrics["model"]) != set(labels):
        raise AssertionError("Common evaluator did not return exactly one metrics row per unique candidate")
    if not pd.api.types.is_numeric_dtype(metrics["WRMSSE"]) or not metrics["WRMSSE"].notna().all():
        raise AssertionError("Common evaluator returned missing/non-numeric WRMSSE")

    by_label = {candidate["model_label"]: candidate for candidate in candidates}
    rows = []
    for _, metric in metrics.iterrows():
        candidate = by_label[metric["model"]]
        rows.append({**candidate, "validation_WRMSSE": float(metric["WRMSSE"])})
    ranked = sorted(rows, key=lambda row: (row["validation_WRMSSE"], row["checkpoint_sha256"]))
    selected = ranked[0]

    pd.DataFrame(ranked).to_csv(os.path.join(output_dir, "checkpoint_validation_ranking.csv"), index=False)
    manifest = {
        "variant_label": args.variant_label,
        "selection_criterion": "lowest exact common-validation full-hierarchy WRMSSE",
        "evaluator": "scripts/audit_comparability.py",
        "evaluator_metrics": os.path.abspath(metrics_path),
        "validation_boundary": {"start": 1526, "end": 1553, "horizon": 28},
        "retained_checkpoint_file_count": sum(len(paths) for paths in aliases.values()),
        "unique_checkpoint_count": len(candidates),
        "deduplicated_aliases": aliases,
        "selected_checkpoint": selected,
        "ranked_candidates": ranked,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(output_dir, "selected_checkpoint.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
