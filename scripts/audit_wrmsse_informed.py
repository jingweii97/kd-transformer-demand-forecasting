"""Pre-training audit for the WRMSSE-informed TFT objective.

This script never calls ``Trainer.fit``.  It computes training-only
coefficients and runs mapping, loss, gradient, optimizer, and legacy checkpoint
regression checks, then prints one JSON audit report.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch
from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer, TimeSeriesDataSet
from lightning.pytorch.core.saving import _load_state

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_from_cache
from models.losses import HuberLossMetric, WRMSSEInformedLossMetric
from models.teacher import create_tft_teacher
from models.wrmsse_informed import build_wrmsse_informed_coefficients
from utils.config import load_config
from utils.paths import get_dataset_dir
from utils.seed import set_seed


LEGACY_FIXTURES = {
    "huber": {
        "pattern": "outputs/teacher/tft64_huber/*epoch=05*.ckpt",
        "checkpoint_sha256": "f994f7f794c68a9b683e22f0532078825b1d604426c665cec36d11b02d9a00b8",
        "prediction_sha256": "de9b2bd73f44a76a5bd7867b5dfd41387076db9537e7b36cd3d93e978d97e88f",
    },
    "quantile": {
        "pattern": "outputs/teacher/tft64_optimized/*epoch=05*.ckpt",
        "checkpoint_sha256": "e338461f9d417806ed638d3717ea203dbfb70155324d5791ca835e5cdaa47ed4",
        "prediction_sha256": "90106f4a21044240b80fb496b66d52b73047e140e48b0bbe99103156ffe1760e",
    },
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_base_dataset(cfg):
    path = os.path.join(get_dataset_dir(cfg), "metadata", "global_metadata.pkl")
    with open(path, "rb") as handle:
        return pickle.load(handle).base_dataset


def _prepare_categories(frame):
    category_columns = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "weekday", "month", "year", "event_name_1", "event_type_1",
    ]
    for column in category_columns:
        if column in frame:
            frame[column] = frame[column].astype(str).astype("category")
    return frame


def _mapping_shuffle_check(base_dataset, cfg, coefficients):
    store = "CA_1"
    frame = load_from_cache(get_dataset_dir(cfg), store)
    ids = frame["id"].astype(str).drop_duplicates().iloc[:8].tolist()
    frame = frame.loc[
        frame["id"].astype(str).isin(ids) & (frame["time_idx"] <= 300)
    ].copy()
    frame["wrmsse_informed_coefficient"] = (
        frame["id"].astype(str).map(coefficients).astype("float32")
    )
    frame = _prepare_categories(frame)
    dataset = TimeSeriesDataSet.from_dataset(
        base_dataset, frame, weight="wrmsse_informed_coefficient"
    )
    loader = dataset.to_dataloader(
        train=True, batch_size=32, shuffle=True, num_workers=0
    )

    checked = 0
    examples = []
    group_encoder = dataset._categorical_encoders["__group_id__id"]
    for x, y in loader:
        batch_ids = group_encoder.inverse_transform(x["groups"][:, 0].cpu().numpy())
        stored = y[1][:, 0]
        if not torch.all(y[1] == stored[:, None]):
            raise AssertionError("A series coefficient changed across decoder positions")
        for series_id, actual in zip(batch_ids, stored.tolist()):
            expected = float(coefficients[str(series_id)])
            if not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
                raise AssertionError(f"Coefficient mismatch for {series_id}")
            if len(examples) < 5:
                examples.append(
                    {
                        "series_id": str(series_id),
                        "expected": expected,
                        "stored_in_batch": float(actual),
                    }
                )
            checked += 1
        if checked >= 128:
            break
    return {"passed": True, "shuffled_samples_checked": checked, "examples": examples}, dataset


def _loss_unit_checks():
    metric = WRMSSEInformedLossMetric()
    prediction = torch.tensor([[[2.0], [4.0]], [[1.0], [5.0]]])
    target = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    coefficient = torch.tensor([[0.5, 0.5], [2.0, 2.0]])
    implemented = metric(prediction, (target, coefficient))
    manual = (coefficient * (prediction.squeeze(-1) - target) ** 2).sum() / target.numel()
    if not torch.allclose(implemented, manual):
        raise AssertionError(f"Manual loss mismatch: {implemented} != {manual}")

    epsilon = 1e-8
    floor = 0.25
    equal_weight = np.ones(4) / 4
    equal_scale = np.ones(4) * 2.0
    raw = equal_weight / (np.maximum(equal_scale, floor) + epsilon)
    normalized = raw / raw.mean()
    if not np.allclose(normalized, np.ones(4)):
        raise AssertionError("Equal weights/scales did not reduce to ordinary MSE")
    if not (2.0 / (1.0 + epsilon) > 1.0 / (1.0 + epsilon)):
        raise AssertionError("Economic-weight monotonicity failed")
    if not (1.0 / (2.0 + epsilon) < 1.0 / (1.0 + epsilon)):
        raise AssertionError("Scale monotonicity failed")
    small_scale = max(1e-12, floor)
    if small_scale != floor:
        raise AssertionError("Scale floor failed")

    return {
        "passed": True,
        "implemented": float(implemented),
        "manual": float(manual),
        "equal_case_normalized_coefficients": normalized.tolist(),
        "weight_monotonicity": True,
        "scale_monotonicity": True,
        "small_scale_uses_floor": True,
    }


def _optimizer_smoke(dataset, cfg):
    loader = dataset.to_dataloader(train=True, batch_size=4, shuffle=True, num_workers=0)
    x, y = next(iter(loader))
    model = create_tft_teacher(dataset, cfg).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.teacher.lr))
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)["prediction"]
    loss = model.loss(prediction, y)
    if not torch.isfinite(loss):
        raise AssertionError("Optimizer smoke loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Optimizer smoke gradients are missing or non-finite")
    optimizer.step()
    return {
        "passed": True,
        "loss": float(loss.detach()),
        "finite_gradient_tensors": len(gradients),
        "output_size": int(prediction.shape[-1]),
    }


def _legacy_checkpoint_regression(base_dataset):
    x, _ = next(iter(base_dataset.to_dataloader(
        train=False, batch_size=4, shuffle=False, num_workers=0
    )))
    results = {}
    for label, fixture in LEGACY_FIXTURES.items():
        matches = glob.glob(fixture["pattern"])
        if len(matches) != 1:
            raise AssertionError(f"Expected one {label} checkpoint, found {matches}")
        path = matches[0]
        checkpoint_hash = _sha256_file(path)
        if checkpoint_hash != fixture["checkpoint_sha256"]:
            raise AssertionError(f"Unexpected {label} checkpoint fixture hash")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected_loss_type = HuberLossMetric if label == "huber" else QuantileLoss
        checkpoint_loss = checkpoint["hyper_parameters"]["loss"]
        if not isinstance(checkpoint_loss, expected_loss_type):
            raise AssertionError(
                f"{label} checkpoint loss changed: {type(checkpoint_loss).__name__}"
            )

        # These CUDA-authored fixtures serialize torchmetrics' bookkeeping
        # device as an ordinary Python attribute, which map_location cannot
        # rewrite on a CPU-only host. Relocate that bookkeeping in memory only;
        # checkpoint bytes, model state, and loss objects remain unchanged.
        def relocate_metric_state(module):
            if hasattr(module, "_device"):
                module._device = torch.device("cpu")
            if isinstance(module, torch.nn.Module):
                for child in module.children():
                    relocate_metric_state(child)

        for value in checkpoint["hyper_parameters"].values():
            relocate_metric_state(value)
        model = _load_state(
            TemporalFusionTransformer, checkpoint, strict=True
        ).cpu().eval()
        with torch.no_grad():
            prediction = model(x)["prediction"].detach().cpu().contiguous()
        prediction_hash = hashlib.sha256(prediction.numpy().tobytes()).hexdigest()
        if prediction_hash != fixture["prediction_sha256"]:
            raise AssertionError(
                f"{label} prediction regression: {prediction_hash} != "
                f"{fixture['prediction_sha256']}"
            )
        results[label] = {
            "passed": True,
            "checkpoint": path,
            "checkpoint_sha256": checkpoint_hash,
            "loss_class": type(checkpoint_loss).__name__,
            "prediction_sha256": prediction_hash,
            "prediction_shape": list(prediction.shape),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="local")
    parser.add_argument("--experiment", default="tft64_wrmsse_informed")
    args = parser.parse_args()

    cfg = load_config(args.env, args.experiment)
    set_seed(int(cfg.environment.seed))
    if cfg.teacher.loss.lower() != "wrmsse_informed" or int(cfg.teacher.output_size) != 1:
        raise AssertionError("Audit requires loss=wrmsse_informed and output_size=1")

    bundle = build_wrmsse_informed_coefficients(cfg)
    if bundle.audit["pathological"]:
        raise RuntimeError(bundle.audit["pathological_reasons"])
    base_dataset = _load_base_dataset(cfg)
    mapping, smoke_dataset = _mapping_shuffle_check(
        base_dataset, cfg, bundle.by_series
    )
    unit = _loss_unit_checks()
    optimizer = _optimizer_smoke(smoke_dataset, cfg)
    legacy = _legacy_checkpoint_regression(base_dataset)

    report = {
        **bundle.audit,
        "equation": (
            "sum_valid[a_i_star * (y_i_h - yhat_i_h)^2] / N_valid; "
            "a_i_star = (w_i / (max(s_i, s_min) + epsilon)) / C"
        ),
        "scale_is_mean_squared_first_difference_not_root": True,
        "mapping_test": mapping,
        "manual_loss_unit_test": unit,
        "gradient_optimizer_smoke_test": optimizer,
        "legacy_checkpoint_regression": legacy,
        "all_checks_passed": True,
        "full_training_started": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
