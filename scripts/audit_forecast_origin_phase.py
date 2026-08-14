"""Read-only daily forecast-origin sensitivity audit for frozen checkpoints.

This tool does not train, alter checkpoints, or overwrite existing evaluation
outputs.  It emits a new CSV of derived metrics only after all requested
origins have completed successfully.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_dataset_from_cache, load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from models.losses import WRMSSEInformedLossMetric
from models.student import M5TransformerStudent
from scripts.evaluate_models import (
    FULL_M5_SERIES_COUNT,
    compute_hierarchical_wrmsse,
    compute_point_metrics,
    compute_wrmsse_weights_and_scales,
)
from utils.config import load_config
from utils.paths import get_dataset_dir, resolve_path


CATEGORY_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "month", "year", "event_name_1", "event_type_1",
]
BANDS = [("Overall (1-28)", 0, 28), ("Short (1-7)", 0, 7),
         ("Medium (8-14)", 7, 14), ("Long (15-28)", 14, 28)]


def _direct_predictions(model, loader, is_teacher, device):
    """Generate point predictions without a trainer or any filesystem writes."""
    values = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in x.items()
            }
            if is_teacher:
                prediction = model.to_prediction(model(x))
            else:
                prediction = model(x)
            values.append(prediction.detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def _load_teacher_cpu(checkpoint_path):
    """Load the frozen TFT on CPU without mutating its CUDA-trained checkpoint.

    Lightning's normal loader restores a metric object serialized on CUDA and
    attempts to move it back to CUDA even when ``map_location='cpu'``.  Build
    the identical network from checkpoint hyperparameters, replace only that
    non-inference metric with its CPU equivalent, and load the frozen weights
    strictly in memory.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint["hyper_parameters"])
    hparams["loss"] = WRMSSEInformedLossMetric()
    hparams.pop("logging_metrics", None)
    model = TemporalFusionTransformer(**hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def _prepare_partition(df_part, start_day, lookback, training_data, batch_size):
    sliced = df_part[
        (df_part["time_idx"] >= start_day - lookback)
        & (df_part["time_idx"] <= start_day + 27)
    ].copy()
    for column in CATEGORY_COLUMNS:
        if column in sliced:
            sliced[column] = sliced[column].astype(str).astype("category")
    dataset = TimeSeriesDataSet.from_dataset(
        training_data, sliced, predict=True, stop_randomization=True
    )
    decoded = dataset.decoded_index
    assert decoded["time_idx_first_prediction"].nunique() == 1
    assert int(decoded["time_idx_first_prediction"].iloc[0]) == start_day
    loader = dataset.to_dataloader(
        train=False, batch_size=batch_size, shuffle=False, num_workers=0
    )
    actuals = []
    for _, y in loader:
        target = y[0] if isinstance(y, (tuple, list)) else y
        actuals.append(target.cpu().numpy())
    return dataset, loader, decoded, np.concatenate(actuals, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--experiment", default="full")
    parser.add_argument("--origins", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--wis-checkpoint", required=True)
    parser.add_argument("--wikd-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output = resolve_path(args.output)
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    lookback = int(cfg.dataset.lookback_window)
    horizon = int(cfg.dataset.prediction_window)
    if horizon != 28:
        raise ValueError("This audit expects the frozen 28-day protocol")
    dataset_dir = get_dataset_dir(cfg)
    df = load_dataset_from_cache(dataset_dir, cfg.environment.store_filter)
    training_data = build_timeseries_dataset(df, cfg, is_train=True)
    train = df[df["time_idx"] <= cfg.dataset.splits.train.end].copy()
    weights, scales, _ = compute_wrmsse_weights_and_scales(
        train, cfg.dataset.splits.train.end
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        teacher = TemporalFusionTransformer.load_from_checkpoint(
            resolve_path(args.teacher_checkpoint)
        ).to(device).eval()
    else:
        teacher = _load_teacher_cpu(resolve_path(args.teacher_checkpoint))
    wis = M5TransformerStudent.load_from_checkpoint(
        resolve_path(args.wis_checkpoint), training_dataset=training_data,
        map_location="cpu", strict=True,
    ).to(device).eval()
    wikd = M5TransformerStudent.load_from_checkpoint(
        resolve_path(args.wikd_checkpoint), training_dataset=training_data,
        map_location="cpu", strict=True,
    ).to(device).eval()
    models = [("TFT Teacher", teacher, True), ("Student-WIS", wis, False),
              ("Student-WIKD", wikd, False)]
    calendar = pd.read_csv(resolve_path("input/calendar.csv"), usecols=["d", "date", "weekday"])
    calendar["time_idx"] = calendar["d"].str.split("_").str[-1].astype(int)
    calendar = calendar.set_index("time_idx")
    stores = resolve_stores(cfg.environment.store_filter)
    rows = []

    for start_day in args.origins:
        print(f"Auditing daily origin {start_day}", flush=True)
        gt = df[(df["time_idx"] >= start_day) & (df["time_idx"] < start_day + horizon)].copy()
        gt["id"] = gt["id"].astype(str)
        gt = gt.sort_values(["id", "time_idx"]).reset_index(drop=True)
        parts, decoded_parts = [], []
        predictions = {name: [] for name, _, _ in models}
        for store in stores:
            part = load_from_cache(dataset_dir, store)
            dataset, loader, decoded, actual = _prepare_partition(
                part, start_day, lookback, training_data, args.batch_size
            )
            parts.append(actual)
            decoded_parts.append(decoded)
            for name, model, is_teacher in models:
                pred = _direct_predictions(model, loader, is_teacher, device)
                assert pred.shape == actual.shape == (len(decoded), horizon)
                predictions[name].append(pred)

        decoded = pd.concat(decoded_parts, ignore_index=True)
        decoded["id"] = decoded["id"].astype(str)
        order = decoded.assign(_row=np.arange(len(decoded))).sort_values("id")["_row"].to_numpy()
        actuals = np.concatenate(parts, axis=0)[order]
        ids = decoded.sort_values("id")["id"].to_numpy()
        assert np.array_equal(ids, gt["id"].drop_duplicates().to_numpy())
        assert actuals.shape == (FULL_M5_SERIES_COUNT, horizon)

        day_meta = calendar.loc[start_day]
        for name, _, _ in models:
            forecast = np.concatenate(predictions[name], axis=0)[order]
            for band, lo, hi in BANDS:
                gt_slice = gt[(gt["time_idx"] >= start_day + lo) & (gt["time_idx"] < start_day + hi)].copy()
                pred_slice = gt_slice.copy()
                pred_slice["sales"] = forecast[:, lo:hi].reshape(-1)
                wrmsse, levels = compute_hierarchical_wrmsse(gt_slice, pred_slice, weights, scales)
                mae, rmse, wape = compute_point_metrics(
                    actuals[:, lo:hi].reshape(-1), forecast[:, lo:hi].reshape(-1)
                )
                row = {
                    "origin": start_day, "date": day_meta["date"],
                    "weekday": day_meta["weekday"], "model": name, "band": band,
                    "wrmsse": wrmsse, "mae": mae, "rmse": rmse, "wape": wape,
                }
                row.update({f"L{i + 1}": value for i, value in enumerate(levels)})
                rows.append(row)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote derived audit metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
