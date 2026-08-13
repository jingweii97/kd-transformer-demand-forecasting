"""Read-only preflight checks for the Student-WIS and Student-WIKD objectives.

This reconstructs one deterministic first streamed batch for each new student
variant.  It proves the shared coefficient formula, the PyTorch Forecasting
elementwise WRMSSE loss semantics, the WIKD verified-cache namespace, and the
alpha arithmetic before any training job is submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import StorePartitionManager, build_timeseries_dataset
from models.losses import WRMSSEInformedLossMetric
from models.student import M5TransformerStudent
from models.wrmsse_informed import build_wrmsse_informed_coefficients
from utils.config import load_config
from utils.paths import resolve_path
from utils.seed import set_seed


EXPECTED_TEACHER_SHA256 = "43841d3db586b7acad384f5068a23a609773ec1ad55e13bf004dc364f2d9bdf2"
STORES = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_tensor(y):
    return y[0] if isinstance(y, (tuple, list)) else y


def target_weights(y):
    if not isinstance(y, (tuple, list)) or len(y) < 2 or y[1] is None:
        raise AssertionError("WRMSSE preflight batch is missing PyTorch Forecasting target weights")
    return y[1]


def valid_mask(predictions: torch.Tensor, decoder_lengths: torch.Tensor | None) -> torch.Tensor:
    if decoder_lengths is None:
        return torch.ones_like(predictions, dtype=torch.bool)
    return torch.arange(
        predictions.shape[1], device=predictions.device
    ).unsqueeze(0) < decoder_lengths.unsqueeze(1)


def manually_reduce(weighted_elementwise: torch.Tensor, decoder_lengths: torch.Tensor | None) -> torch.Tensor:
    mask = valid_mask(weighted_elementwise, decoder_lengths).to(weighted_elementwise.dtype)
    return (weighted_elementwise * mask).sum() / mask.sum()


def make_model(training_data, cfg, alpha: float) -> M5TransformerStudent:
    return M5TransformerStudent(
        training_dataset=training_data,
        d_model=cfg.student.d_model,
        nhead=cfg.student.nhead,
        num_layers=cfg.student.layers,
        dim_feedforward=cfg.student.dim_feedforward,
        dropout=cfg.student.dropout,
        lr=cfg.student.lr,
        alpha=alpha,
        lookback_window=cfg.dataset.lookback_window,
        prediction_window=cfg.dataset.prediction_window,
        supervised_loss="wrmsse_informed",
    ).eval()


def first_weighted_batch(cfg, experiment_name: str, coefficients):
    training_data = build_timeseries_dataset(None, cfg, is_train=True)
    manager = StorePartitionManager(
        training_data,
        cfg,
        exp_name=experiment_name,
        series_coefficients=coefficients,
    )
    return training_data, next(iter(manager.train_dataloader(cfg.student.batch_size)))


def assert_teacher_loss_semantics(model, x, y, require_soft_targets: bool) -> dict:
    target = target_tensor(y)
    weights = target_weights(y)
    coefficients = x.get("wrmsse_informed_coefficient")
    if coefficients is None:
        raise AssertionError("WRMSSE preflight batch is missing per-series coefficients")
    if weights.shape != target.shape:
        raise AssertionError(f"Target-weight shape mismatch: {weights.shape} vs {target.shape}")
    if not torch.allclose(weights, coefficients.unsqueeze(1).expand_as(weights)):
        raise AssertionError("Dataset target weights do not equal the fixed per-series coefficients")

    decoder_lengths = x.get("decoder_lengths")
    with torch.no_grad():
        predictions = model(x)
    teacher_metric = WRMSSEInformedLossMetric()
    teacher_sup_elementwise = teacher_metric.loss(predictions, target)
    student_sup = model._point_loss(predictions, target, coefficients, decoder_lengths)
    manual_sup = manually_reduce(teacher_sup_elementwise * weights, decoder_lengths)
    if not torch.allclose(student_sup, manual_sup, rtol=1e-6, atol=1e-7):
        raise AssertionError(
            f"Student supervised loss differs from teacher semantics: {student_sup} != {manual_sup}"
        )

    result = {
        "batch_size": int(predictions.shape[0]),
        "horizon": int(predictions.shape[1]),
        "decoder_lengths": sorted(set(decoder_lengths.tolist())) if decoder_lengths is not None else [int(predictions.shape[1])],
        "supervised_loss": float(student_sup),
        "teacher_elementwise_matches_squared_error": bool(torch.allclose(
            teacher_sup_elementwise, torch.square(predictions - target)
        )),
    }
    if not require_soft_targets:
        return result

    teacher_targets = x.get("soft_targets")
    if teacher_targets is None or not torch.isfinite(teacher_targets).all():
        raise AssertionError("WIKD batch does not contain finite verified soft targets")
    teacher_dist_elementwise = teacher_metric.loss(predictions, teacher_targets)
    student_dist = model._point_loss(predictions, teacher_targets, coefficients, decoder_lengths)
    manual_dist = manually_reduce(teacher_dist_elementwise * weights, decoder_lengths)
    if not torch.allclose(student_dist, manual_dist, rtol=1e-6, atol=1e-7):
        raise AssertionError("Student distillation loss differs from weighted teacher-loss semantics")
    combined = 0.5 * student_sup + 0.5 * student_dist
    manual_combined = 0.5 * manual_sup + 0.5 * manual_dist
    if not torch.allclose(combined, manual_combined, rtol=1e-6, atol=1e-7):
        raise AssertionError("WIKD alpha arithmetic check failed")
    result.update({
        "distillation_loss": float(student_dist),
        "combined_loss": float(combined),
        "alpha_formula_verified": True,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--soft-targets-dir", default="artifacts/soft_targets")
    parser.add_argument("--soft-targets-exp-name", default="wi_e09_43841d3d_verified")
    parser.add_argument(
        "--teacher-checkpoint",
        default="outputs/teacher/tft64_wrmsse_informed/tft64-wrmsse-informed-epoch=epoch=09-val_loss=val_loss=1.466434.ckpt",
    )
    args = parser.parse_args()

    set_seed(42)
    targets_dir = resolve_path(args.soft_targets_dir)
    checkpoint = resolve_path(args.teacher_checkpoint)
    if sha256(checkpoint) != EXPECTED_TEACHER_SHA256:
        raise AssertionError("WI epoch-9 teacher checkpoint SHA-256 mismatch")
    for store in STORES:
        sidecar = os.path.join(targets_dir, f"{args.soft_targets_exp_name}_{store}.json")
        tensor = os.path.join(targets_dir, f"{args.soft_targets_exp_name}_{store}.pt")
        if not os.path.isfile(tensor) or not os.path.isfile(sidecar):
            raise FileNotFoundError(f"Missing verified soft-target artifact for {store}")
        with open(sidecar, encoding="utf-8") as handle:
            if json.load(handle).get("checkpoint_sha256") != EXPECTED_TEACHER_SHA256:
                raise AssertionError(f"Soft-target provenance SHA mismatch for {store}")

    wis_cfg = load_config(args.env, "student_wis")
    wikd_cfg = load_config(args.env, "student_wikd")
    wikd_cfg.student.soft_targets_path = targets_dir
    wikd_cfg.student.soft_targets_exp_name = args.soft_targets_exp_name
    if wis_cfg.student.kd or not wikd_cfg.student.kd or float(wikd_cfg.student.alpha) != 0.5:
        raise AssertionError("Student-WIS/WIKD configuration does not match the requested experiment design")

    wis_bundle = build_wrmsse_informed_coefficients(wis_cfg, objective_config=wis_cfg.student)
    wikd_bundle = build_wrmsse_informed_coefficients(wikd_cfg, objective_config=wikd_cfg.student)
    if wis_bundle.by_series != wikd_bundle.by_series:
        raise AssertionError("Student-WIS and Student-WIKD coefficients differ")
    if wis_bundle.audit["provenance"]["validation_targets_used"] or wis_bundle.audit["provenance"]["held_out_targets_used"]:
        raise AssertionError("WRMSSE coefficient construction used non-training targets")

    wis_data, wis_batch = first_weighted_batch(wis_cfg, "student_wis", wis_bundle.by_series)
    wis_result = assert_teacher_loss_semantics(make_model(wis_data, wis_cfg, 1.0), *wis_batch, False)
    del wis_data, wis_batch

    wikd_data, wikd_batch = first_weighted_batch(
        wikd_cfg, "student_wikd_wi_e09_verified", wikd_bundle.by_series
    )
    wikd_result = assert_teacher_loss_semantics(make_model(wikd_data, wikd_cfg, 0.5), *wikd_batch, True)

    print(json.dumps({
        "status": "PASS",
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "soft_target_cache_prefix": args.soft_targets_exp_name,
        "coefficient_provenance": wis_bundle.audit["provenance"],
        "coefficient_normalization": wis_bundle.audit["normalization_constant"],
        "Student-WIS": wis_result,
        "Student-WIKD": wikd_result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
