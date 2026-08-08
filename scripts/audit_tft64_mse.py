"""Pre-training smoke test for the plain-MSE TFT-64 control.

This script deliberately never calls ``Trainer.fit``.  It uses one genuine
unweighted TFT training batch, verifies the MSE reduction and gradient path,
then performs lightweight construction/load regressions for the three legacy
objectives.  It is intended to run on the same environment as training.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_from_cache
from data.dataset import build_timeseries_dataset
from models.losses import HuberLossMetric, MSELossMetric, WRMSSEInformedLossMetric
from models.teacher import create_tft_teacher
from utils.config import load_config
from utils.paths import get_dataset_dir
from utils.seed import set_seed


HUBER_CHECKPOINT = "outputs/teacher/tft64_huber/*epoch=05*.ckpt"
QUANTILE_CHECKPOINT = "outputs/teacher/tft64_optimized/*epoch=05*.ckpt"


def _target_from_batch(batch_y):
    return batch_y[0] if isinstance(batch_y, (tuple, list)) else batch_y


def _valid_mask(batch_x, target):
    lengths = batch_x.get("decoder_lengths")
    if lengths is None:
        return torch.ones_like(target, dtype=torch.bool)
    positions = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
    return positions < lengths.to(target.device).unsqueeze(1)


def _load_legacy_checkpoint(path):
    """Load CPU checkpoints, including legacy metric bookkeeping if needed."""
    try:
        return TemporalFusionTransformer.load_from_checkpoint(path, map_location="cpu")
    except Exception:
        # Some historical CUDA-authored fixtures preserve a metric bookkeeping
        # device. Relocate only that in-memory bookkeeping, as the existing
        # WRMSSE audit does, then use Lightning's strict loader.
        from lightning.pytorch.core.saving import _load_state

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        def relocate_metric_device(value):
            if hasattr(value, "_device"):
                value._device = torch.device("cpu")
            if isinstance(value, torch.nn.Module):
                for child in value.children():
                    relocate_metric_device(child)

        for value in checkpoint["hyper_parameters"].values():
            relocate_metric_device(value)
        return _load_state(TemporalFusionTransformer, checkpoint, strict=True).cpu()


def _legacy_load_checks():
    expected = {
        "huber": (HUBER_CHECKPOINT, HuberLossMetric),
        "quantile": (QUANTILE_CHECKPOINT, QuantileLoss),
    }
    results = {}
    for label, (pattern, loss_type) in expected.items():
        matches = glob.glob(pattern)
        if len(matches) != 1:
            raise AssertionError(f"Expected exactly one {label} fixture, found {matches}")
        model = _load_legacy_checkpoint(matches[0])
        if not isinstance(model.loss, loss_type):
            raise AssertionError(f"{label} loss is {type(model.loss).__name__}, not {loss_type.__name__}")
        item = {"passed": True, "checkpoint": matches[0], "loss_class": type(model.loss).__name__}
        if label == "huber":
            if float(model.loss.delta) != 1.0:
                raise AssertionError(f"Huber delta changed to {model.loss.delta}")
            item["delta"] = float(model.loss.delta)
        else:
            item["quantiles"] = [float(q) for q in model.loss.quantiles]
        results[label] = item
    return results


def _raw_sales_domain_check(training_dataset, cfg, batch_x, target):
    """Confirm this MSE batch uses the same raw-sales tensor path as Huber."""
    group_encoder = training_dataset._categorical_encoders["__group_id__id"]
    series_ids = group_encoder.inverse_transform(batch_x["groups"][:, 0].cpu().numpy())
    stores = {"_".join(str(series_id).split("_")[-3:-1]) for series_id in series_ids}
    if len(stores) != 1:
        raise AssertionError(f"A streamed store batch must contain one store, got {sorted(stores)}")
    raw_frame = load_from_cache(get_dataset_dir(cfg), stores.pop())
    lookup = raw_frame.assign(id=raw_frame["id"].astype(str)).set_index(
        ["id", "time_idx"]
    )["sales"]
    starts = batch_x["decoder_time_idx"][:, 0].detach().cpu().numpy().astype(int)
    raw_rows = []
    for series_id, start in zip(series_ids, starts):
        raw_rows.append([
            lookup[(str(series_id), time_idx)]
            for time_idx in range(start, start + target.shape[1])
        ])
    raw_target = torch.as_tensor(raw_rows, dtype=target.dtype)
    max_abs_gap = float(torch.max(torch.abs(raw_target - target.detach().cpu())))
    if max_abs_gap > 1e-5:
        raise AssertionError(f"MSE batch target is not raw sales; max abs gap={max_abs_gap}")
    return {
        "passed": True,
        "raw_sales_vs_dataset_target_max_abs_gap": max_abs_gap,
        "same_legacy_unweighted_path_as_huber": True,
    }


def _variable_decoder_mask_check():
    """Exercise MultiHorizonMetric's packed-target decoder masking path."""
    prediction = torch.tensor([[[2.0], [4.0], [8.0], [16.0]], [[1.0], [3.0], [99.0], [99.0]]])
    target = torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.0, 1.0, 0.0, 0.0]])
    lengths = torch.tensor([4, 2])
    packed_target = torch.nn.utils.rnn.pack_padded_sequence(
        target, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    implemented = MSELossMetric()(prediction, packed_target)
    positions = torch.arange(target.shape[1]).unsqueeze(0)
    valid_mask = positions < lengths.unsqueeze(1)
    manual = torch.square(prediction.squeeze(-1) - target)[valid_mask].mean()
    if not torch.allclose(implemented, manual):
        raise AssertionError(f"Variable decoder masking mismatch: {manual} != {implemented}")
    return {
        "passed": True,
        "manual_mse": float(manual),
        "implemented_mse": float(implemented),
        "valid_elements": int(valid_mask.sum()),
        "padded_elements_excluded": int((~valid_mask).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="local")
    parser.add_argument("--experiment", default="tft64_mse")
    parser.add_argument(
        "--smoke-batch-size",
        type=int,
        default=8,
        help="Small real-batch size for this no-training CPU/GPU smoke test.",
    )
    args = parser.parse_args()

    cfg = load_config(args.env, args.experiment)
    set_seed(int(cfg.environment.seed))
    if cfg.teacher.loss.lower() != "mse":
        raise AssertionError("MSE audit requires teacher.loss=mse")
    if int(cfg.teacher.output_size) != 1:
        raise AssertionError("MSE audit requires teacher.output_size=1")

    training_dataset = build_timeseries_dataset(None, cfg, is_train=True)
    # The cached base dataset is the exact unweighted TimeSeriesDataSet schema
    # used to construct each streamed Huber/MSE store partition. It gives a
    # real TFT batch without loading an entire store merely for this smoke gate.
    loader = training_dataset.to_dataloader(
        train=True,
        batch_size=int(args.smoke_batch_size),
        shuffle=False,
        num_workers=0,
    )
    batch_x, batch_y = next(iter(loader))
    if not isinstance(batch_y, (tuple, list)) or len(batch_y) < 2 or batch_y[1] is not None:
        raise AssertionError("MSE batch must retain legacy unweighted target tuple (target, None)")
    if "wrmsse_informed_coefficient" in batch_x:
        raise AssertionError("MSE batch unexpectedly contains a WRMSSE-informed coefficient")

    model = create_tft_teacher(training_dataset, cfg).train()
    if not isinstance(model.loss, MSELossMetric):
        raise AssertionError(f"Expected MSELossMetric, got {type(model.loss).__name__}")

    output = model(batch_x)
    prediction = output["prediction"]
    target = _target_from_batch(batch_y)
    point_prediction = prediction.squeeze(-1) if prediction.ndim > target.ndim else prediction
    if tuple(point_prediction.shape) != tuple(target.shape):
        raise AssertionError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if not torch.allclose(model.to_prediction(output), point_prediction):
        raise AssertionError("TFT prediction transformation was not applied exactly once")

    elementwise = model.loss.loss(prediction, target)
    expected_elementwise = torch.square(point_prediction - target)
    if not torch.allclose(elementwise, expected_elementwise):
        raise AssertionError("MSE elementwise equation is not (y - yhat)^2")
    valid_mask = _valid_mask(batch_x, target)
    manual_mse = expected_elementwise[valid_mask].mean()
    implemented_mse = model.loss(prediction, batch_y)
    if not torch.allclose(implemented_mse, manual_mse, rtol=1e-6, atol=1e-6):
        raise AssertionError(f"Manual MSE mismatch: {manual_mse} != {implemented_mse}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.teacher.lr))
    optimizer.zero_grad(set_to_none=True)
    implemented_mse.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise AssertionError("MSE gradients are missing or non-finite")
    optimizer.step()

    wrmsse_model = create_tft_teacher(
        training_dataset, load_config(args.env, "tft64_wrmsse_informed")
    )
    if not isinstance(wrmsse_model.loss, WRMSSEInformedLossMetric):
        raise AssertionError("WRMSSE-informed teacher construction regressed")

    legacy = _legacy_load_checks()
    raw_domain = _raw_sales_domain_check(training_dataset, cfg, batch_x, target)
    variable_decoder_mask = _variable_decoder_mask_check()
    report = {
        "all_checks_passed": True,
        "full_training_started": False,
        "equation": "mean_valid((y - yhat)^2)",
        "loss_class": type(model.loss).__name__,
        "output_size": int(model.hparams.output_size),
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "manual_mse": float(manual_mse.detach()),
        "implemented_mse": float(implemented_mse.detach()),
        "manual_vs_implemented_abs_gap": float(torch.abs(manual_mse - implemented_mse).detach()),
        "valid_decoder_elements": int(valid_mask.sum()),
        "total_decoder_elements": int(valid_mask.numel()),
        "variable_decoder_masking": variable_decoder_mask,
        "target_weight": None,
        "wrmsse_coefficient_in_batch": False,
        "economic_weighting": False,
        "rmsse_scaling": False,
        "hierarchy_aggregation": False,
        "finite_loss": bool(torch.isfinite(implemented_mse)),
        "finite_gradients": True,
        "finite_gradient_tensors": len(gradients),
        "optimizer_step": "succeeded",
        "raw_sales_domain": raw_domain,
        "prediction_transformation": "exactly_once",
        "legacy_checkpoint_loads": legacy,
        "wrmsse_informed_construction": {"passed": True, "output_size": int(wrmsse_model.hparams.output_size)},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
