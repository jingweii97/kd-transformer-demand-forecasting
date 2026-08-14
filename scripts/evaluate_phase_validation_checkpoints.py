"""Select retained phase-v2 checkpoints on a common 7-phase validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_dataset_from_cache, load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from models.student import M5TransformerStudent
from scripts.evaluate_models import (
    FULL_M5_SERIES_COUNT,
    compute_hierarchical_wrmsse,
    compute_point_metrics,
    compute_wrmsse_weights_and_scales,
    get_predictions,
)
from utils.config import load_config
from utils.paths import get_dataset_dir, resolve_path


CATEGORY_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "month", "year", "event_name_1", "event_type_1",
]
BANDS = [("Overall (1-28)", 0, 28), ("Short (1-7)", 0, 7),
         ("Medium (8-14)", 7, 14), ("Long (15-28)", 14, 28)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates(run_dir):
    paths = sorted(
        os.path.join(run_dir, name) for name in os.listdir(run_dir) if name.endswith(".ckpt")
    )
    if not paths or not os.path.isfile(os.path.join(run_dir, "last.ckpt")):
        raise FileNotFoundError("Expected retained top-k checkpoints and last.ckpt")
    by_hash = {}
    for path in paths:
        digest = sha256(path)
        by_hash.setdefault(digest, {"checkpoint_path": os.path.abspath(path), "checkpoint_sha256": digest, "aliases": []})
        by_hash[digest]["aliases"].append(os.path.abspath(path))
    return list(by_hash.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dicc")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--model-type", choices=["student", "teacher"], required=True)
    parser.add_argument("--origins-file", required=True)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    output_dir = os.path.join(run_dir, "phase_validation_evaluation")
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite selection output: {output_dir}")
    with open(resolve_path(args.origins_file), "r", encoding="utf-8") as handle:
        origin_payload = json.load(handle)
    origins = origin_payload["origins"]
    reference = origin_payload["reference_friday_origin"]
    if len(origins) != 7 or sorted({origin % 7 for origin in origins}) != list(range(7)):
        raise ValueError("Validation schedule must contain exactly one of all seven modulo-7 phases")
    if reference not in origins:
        raise ValueError("Reference Friday origin must be part of the validation schedule")

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    ds_dir = get_dataset_dir(cfg)
    df = load_dataset_from_cache(ds_dir, cfg.environment.store_filter)
    train_ds = build_timeseries_dataset(None, cfg, is_train=True)
    train = df[df["time_idx"] <= cfg.dataset.splits.train.end].copy()
    weights, scales, _ = compute_wrmsse_weights_and_scales(train, cfg.dataset.splits.train.end)
    del train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retained = candidates(run_dir)
    loaded = []
    for index, candidate in enumerate(retained, 1):
        label = f"{args.variant_label} candidate {index}"
        if args.model_type == "teacher":
            model = TemporalFusionTransformer.load_from_checkpoint(
                candidate["checkpoint_path"], map_location="cpu"
            ).to(device).eval()
        else:
            model = M5TransformerStudent.load_from_checkpoint(
                candidate["checkpoint_path"], training_dataset=train_ds, map_location="cpu", strict=True
            ).to(device).eval()
        loaded.append((label, candidate, model))

    calendar = pd.read_csv(resolve_path("input/calendar.csv"), usecols=["d", "date", "weekday"])
    calendar["origin"] = calendar["d"].str.split("_").str[-1].astype(int)
    calendar = calendar.set_index("origin")
    rows = []
    for origin in origins:
        print(f"Phase-validation origin {origin}", flush=True)
        gt = df[(df.time_idx >= origin) & (df.time_idx < origin + 28)].copy()
        gt["id"] = gt["id"].astype(str)
        gt = gt.sort_values(["id", "time_idx"]).reset_index(drop=True)
        actual_parts, decoded_parts = [], []
        prediction_parts = {label: [] for label, _, _ in loaded}
        for store in resolve_stores(cfg.environment.store_filter):
            part = load_from_cache(ds_dir, store)
            subset = part[(part.time_idx >= origin - cfg.dataset.lookback_window) & (part.time_idx < origin + 28)].copy()
            for column in CATEGORY_COLUMNS:
                if column in subset:
                    subset[column] = subset[column].astype(str).astype("category")
            dataset = TimeSeriesDataSet.from_dataset(train_ds, subset, predict=True, stop_randomization=True)
            decoded = dataset.decoded_index
            assert decoded["time_idx_first_prediction"].nunique() == 1
            assert int(decoded["time_idx_first_prediction"].iloc[0]) == origin
            loader = dataset.to_dataloader(train=False, batch_size=256, shuffle=False, num_workers=cfg.environment.num_workers)
            actual_parts.append(np.concatenate([(y[0] if isinstance(y, (tuple, list)) else y).cpu().numpy() for _, y in loader]))
            decoded_parts.append(decoded)
            for label, _, model in loaded:
                prediction_parts[label].append(get_predictions(model, loader))

        decoded = pd.concat(decoded_parts, ignore_index=True)
        decoded["id"] = decoded["id"].astype(str)
        order = decoded.assign(_row=np.arange(len(decoded))).sort_values("id")["_row"].to_numpy()
        actuals = np.concatenate(actual_parts)[order]
        assert actuals.shape == (FULL_M5_SERIES_COUNT, 28)
        assert np.array_equal(decoded.sort_values("id")["id"].to_numpy(), gt["id"].drop_duplicates().to_numpy())
        meta = calendar.loc[origin]
        for label, candidate, _ in loaded:
            forecast = np.concatenate(prediction_parts[label])[order]
            for band, lo, hi in BANDS:
                gt_band = gt[(gt.time_idx >= origin + lo) & (gt.time_idx < origin + hi)].copy()
                pred_band = gt_band.copy()
                pred_band["sales"] = forecast[:, lo:hi].reshape(-1)
                wrmsse, levels = compute_hierarchical_wrmsse(gt_band, pred_band, weights, scales)
                mae, rmse, wape = compute_point_metrics(actuals[:, lo:hi].reshape(-1), forecast[:, lo:hi].reshape(-1))
                row = {"origin": origin, "date": meta["date"], "weekday": meta["weekday"], "phase_mod_7": origin % 7,
                       "model": label, **candidate, "band": band, "WRMSSE": wrmsse, "MAE": mae, "RMSE": rmse, "WAPE": wape}
                row.update({f"L{i + 1}": value for i, value in enumerate(levels)})
                rows.append(row)

    os.makedirs(output_dir)
    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(output_dir, "phase_validation_by_origin.csv"), index=False)
    overall = detail[detail.band == "Overall (1-28)"]
    keys = ["model", "checkpoint_path", "checkpoint_sha256"]
    mean_scores = overall.groupby(keys, as_index=False)["WRMSSE"].mean().rename(
        columns={"WRMSSE": "mean_phase_validation_WRMSSE"}
    )
    friday_scores = overall[overall.origin == reference][keys + ["WRMSSE"]].rename(
        columns={"WRMSSE": "friday_reference_WRMSSE"}
    )
    ranking = mean_scores.merge(friday_scores, on=keys, validate="one_to_one").sort_values(
        ["mean_phase_validation_WRMSSE", "checkpoint_sha256"]
    )
    ranking.to_csv(os.path.join(output_dir, "phase_validation_ranking.csv"), index=False)
    selected = ranking.iloc[0].to_dict()
    # Keep the legacy resolver contract while making the new mean criterion
    # explicit in the ranking and manifest.
    selected["validation_WRMSSE"] = selected["mean_phase_validation_WRMSSE"]
    manifest = {"variant_label": args.variant_label, "model_type": args.model_type,
                "selection_criterion": "lowest mean exact full-hierarchy WRMSSE across predefined 7-phase validation origins",
                "origins": origins, "reference_friday_origin": reference, "selected_checkpoint": selected,
                "detail_csv": os.path.abspath(os.path.join(output_dir, "phase_validation_by_origin.csv")),
                "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(output_dir, "selected_checkpoint.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
