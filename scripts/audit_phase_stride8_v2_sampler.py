"""Deterministically audit the final constant-stride-8 v2 origin protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.origin_sampling import load_training_origins
from utils.config import load_config
from utils.paths import resolve_path


EXPERIMENTS = (
    "phase_stride8_teacher_v2",
    "phase_stride8_wis_v2",
    "phase_stride8_wikd_v2",
)
EXPECTED_PHASE_COUNTS = [23, 23, 22, 22, 22, 22, 22]
EXPECTED_SERIES = 30_490
EXPECTED_ORIGINS = list(range(91, 1332, 8))
EXPECTED_VALIDATION = list(range(1520, 1527))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = resolve_path(args.output)
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite sampler audit: {output}")

    configs = {name: load_config(env_name=args.env, experiment_name=name) for name in EXPERIMENTS}
    schedules = {
        name: load_training_origins(cfg.dataset.training_origin_sampling, os.getcwd())
        for name, cfg in configs.items()
    }
    reference = schedules[EXPERIMENTS[0]]
    if any(schedule != reference for schedule in schedules.values()):
        raise AssertionError("Teacher/WIS/WIKD stride-8 training-origin schedules differ")
    if reference != EXPECTED_ORIGINS:
        raise AssertionError("Training schedule is not exactly d_(91 + 8*k), k=0..155")

    cfg = configs[EXPERIMENTS[0]]
    horizon = int(cfg.dataset.prediction_window)
    train_end = int(cfg.dataset.splits.train.end)
    validation_start = int(cfg.dataset.splits.validation.start)
    test_start = int(cfg.dataset.splits.test_stream.start)
    phase_counts = [sum(origin % 7 == phase for origin in reference) for phase in range(7)]
    assert len(reference) == 156
    assert len(set(reference)) == 156
    assert reference[0] == 91 and reference[-1] == 1331
    assert set(origin % 7 for origin in reference) == set(range(7))
    assert phase_counts == EXPECTED_PHASE_COUNTS
    assert max(reference) + horizon - 1 <= train_end
    assert max(reference) < validation_start < test_start

    validation_schedules = {
        name: load_training_origins(cfg.dataset.validation_origin_sampling, os.getcwd())
        for name, cfg in configs.items()
    }
    validation_reference = validation_schedules[EXPERIMENTS[0]]
    if any(schedule != validation_reference for schedule in validation_schedules.values()):
        raise AssertionError("Teacher/WIS/WIKD stride-8 validation schedules differ")
    if validation_reference != EXPECTED_VALIDATION:
        raise AssertionError("Validation schedule must be exactly d1520..d1526")
    assert validation_reference[-1] + horizon - 1 == int(cfg.dataset.splits.validation.end)

    series_count = int(
        pd.read_csv(
            os.path.join(resolve_path(cfg.environment.input_dir), "sales_train_evaluation.csv"),
            usecols=["id"],
        ).shape[0]
    )
    assert series_count == EXPECTED_SERIES

    report = {
        "audit": "final_phase_stride8_v2_sampler",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series_count": series_count,
        "origins_per_series": len(reference),
        "origins": reference,
        "phase_counts_by_modulo_7": {str(i): phase_counts[i] for i in range(7)},
        "all_seven_phases_per_series": True,
        "no_duplicate_origins": True,
        "max_origin": max(reference),
        "max_target_day": max(reference) + horizon - 1,
        "training_cutoff_day": train_end,
        "no_validation_or_test_origin_in_training": True,
        "same_training_schedule": {name: schedules[name] == reference for name in EXPERIMENTS},
        "validation_origins": validation_reference,
        "same_validation_schedule": {
            name: validation_schedules[name] == validation_reference for name in EXPERIMENTS
        },
        "origin_schedule_source": cfg.dataset.training_origin_sampling.origins_file,
        "horizon": horizon,
        "lookback": int(cfg.dataset.lookback_window),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
