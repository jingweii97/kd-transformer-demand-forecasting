"""Read-only reproducibility verification for store-partitioned soft targets."""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import load_from_cache, resolve_stores
from data.dataset import build_timeseries_dataset
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from scripts.generate_soft_targets import _assert_dataset_state_matches_checkpoint, _sha256_file
from utils.config import load_config
from utils.paths import get_dataset_dir, resolve_path


CAT_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "month", "year", "event_name_1", "event_type_1",
]


def _sample_positions(decoded_index, cache_tensor, local_group, count):
    """Choose reproducible, finite forecast rows spanning the available origins."""
    origins = decoded_index["time_idx_first_prediction"].to_numpy(dtype=np.int64)
    valid = np.asarray(
        [torch.isfinite(cache_tensor[local_group, int(origin)]).all().item() for origin in origins],
        dtype=bool,
    )
    positions = np.flatnonzero(valid)
    if len(positions) == 0:
        return np.asarray([], dtype=np.int64)
    return positions[np.linspace(0, len(positions) - 1, num=min(count, len(positions)), dtype=int)]


def main():
    parser = argparse.ArgumentParser(description="Verify fresh TFT forecasts against a soft-target cache")
    parser.add_argument("--env", default="local")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--exp-name", required=True, help="Soft-target experiment prefix")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--soft-targets-dir", default="artifacts/soft_targets")
    parser.add_argument("--samples-per-store", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    cfg.environment.num_workers = 0
    checkpoint_path = resolve_path(args.checkpoint_path)
    targets_dir = resolve_path(args.soft_targets_dir)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    training_data = build_timeseries_dataset(None, cfg, is_train=True)
    teacher = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path).eval()
    _assert_dataset_state_matches_checkpoint(training_data, teacher)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = teacher.to(device)
    group_encoder = training_data._categorical_encoders["id"]
    dataset_dir = get_dataset_dir(cfg)

    results = []
    for store in resolve_stores(cfg.environment.store_filter):
        cache_path = os.path.join(targets_dir, f"{args.exp_name}_{store}.pt")
        sidecar_path = cache_path.replace(".pt", ".json")
        if not os.path.isfile(cache_path) or not os.path.isfile(sidecar_path):
            raise FileNotFoundError(f"Missing cache or provenance sidecar for {store}")
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        if provenance.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {store}: sidecar has "
                f"{provenance.get('checkpoint_sha256')}, expected {checkpoint_sha256}"
            )

        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_tensor = cached["tensor"]
        local_by_global = {int(code): index for index, code in enumerate(cached["unique_groups"])}
        max_day = int(provenance["max_day"])

        frame = load_from_cache(dataset_dir, store)
        frame = frame[(frame["time_idx"] >= 1) & (frame["time_idx"] <= max_day)].copy()
        for column in CAT_COLUMNS:
            if column in frame:
                frame[column] = frame[column].astype(str).astype("category")

        items = sorted(frame["item_id"].astype(str).unique())
        item_positions = np.linspace(0, len(items) - 1, num=min(args.samples_per_store, len(items)), dtype=int)
        for item_position in item_positions:
            item = items[item_position]
            item_frame = frame[frame["item_id"].astype(str) == item].copy()
            item_dataset = TimeSeriesDataSet.from_dataset(
                training_data, item_frame, predict=False, stop_randomization=True
            )
            decoded = item_dataset.decoded_index.reset_index(drop=True)
            series_id = str(decoded.iloc[0]["id"])
            global_code = int(group_encoder.transform([series_id])[0])
            if global_code not in local_by_global:
                raise KeyError(f"Cache has no row for {series_id} in {store}")
            local_group = local_by_global[global_code]
            positions = _sample_positions(decoded, cache_tensor, local_group, 1)
            if len(positions) == 0:
                raise ValueError(f"Cache has no finite target for {series_id} in {store}")

            base_loader = item_dataset.to_dataloader(train=False, batch_size=1, shuffle=False, num_workers=0)
            loader = DataLoader(
                Subset(item_dataset, positions.tolist()),
                batch_size=len(positions),
                shuffle=False,
                num_workers=0,
                collate_fn=base_loader.collate_fn,
            )
            x, _ = next(iter(loader))
            x_device = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in x.items()}
            with torch.no_grad():
                fresh = teacher.to_prediction(teacher(x_device)).detach().cpu()
            if fresh.ndim == 3 and fresh.shape[-1] == 1:
                fresh = fresh[..., 0]

            origin = int(decoded.iloc[int(positions[0])]["time_idx_first_prediction"])
            target = cache_tensor[local_group, origin]
            difference = (fresh[0] - target).abs()
            result = {
                "store": store,
                "series_id": series_id,
                "global_code": global_code,
                "cache_local_row": local_group,
                "origin": origin,
                "horizon": int(target.shape[-1]),
                "mean_abs_diff": float(difference.mean()),
                "max_abs_diff": float(difference.max()),
                "allclose": bool(torch.allclose(fresh[0], target, rtol=args.rtol, atol=args.atol)),
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True))

    failed = [result for result in results if not result["allclose"]]
    summary = {
        "checkpoint_sha256": checkpoint_sha256,
        "samples": len(results),
        "failed": len(failed),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
