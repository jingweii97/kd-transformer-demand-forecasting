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


def _generation_context_batch(
    training_data,
    frame,
    generation_items,
    target_item,
    cache_tensor,
    local_by_global,
    group_encoder,
    chunk_size,
    batch_size,
):
    """Recreate the exact chunk and contiguous inference batch used by generation.

    The generator emits predictions from batches of 256 windows.  Evaluating a
    window by itself can select a different GPU kernel/reduction path, yielding
    tiny but real floating-point differences.  This function preserves the
    generator's item ordering, chunk membership, batch ordering and batch size.
    """
    item_index = int(np.flatnonzero(generation_items == target_item)[0])
    chunk_start = (item_index // chunk_size) * chunk_size
    chunk_items = generation_items[chunk_start: chunk_start + chunk_size]
    chunk_frame = frame[frame["item_id"].astype(str).isin(chunk_items)].copy()
    chunk_dataset = TimeSeriesDataSet.from_dataset(
        training_data, chunk_frame, predict=False, stop_randomization=True
    )
    decoded = chunk_dataset.decoded_index.reset_index(drop=True)

    target_series = str(frame.loc[frame["item_id"].astype(str) == target_item, "id"].iloc[0])
    global_code = int(group_encoder.transform([target_series])[0])
    if global_code not in local_by_global:
        raise KeyError(f"Cache has no row for {target_series}")
    local_group = local_by_global[global_code]
    candidate_positions = np.flatnonzero(decoded["id"].astype(str).to_numpy() == target_series)
    if len(candidate_positions) == 0:
        raise ValueError(f"Generation chunk has no rows for {target_series}")

    for position in candidate_positions:
        origin = int(decoded.iloc[int(position)]["time_idx_first_prediction"])
        if torch.isfinite(cache_tensor[local_group, origin]).all():
            target_position = int(position)
            break
    else:
        raise ValueError(f"Cache has no finite target for {target_series}")

    batch_start = (target_position // batch_size) * batch_size
    batch_indices = list(range(batch_start, min(batch_start + batch_size, len(chunk_dataset))))
    base_loader = chunk_dataset.to_dataloader(
        train=False, batch_size=batch_size, shuffle=False, num_workers=0
    )
    context_loader = DataLoader(
        Subset(chunk_dataset, batch_indices),
        batch_size=len(batch_indices),
        shuffle=False,
        num_workers=0,
        collate_fn=base_loader.collate_fn,
    )
    x, _ = next(iter(context_loader))
    return {
        "x": x,
        "series_id": target_series,
        "global_code": global_code,
        "cache_local_row": local_group,
        "origin": int(decoded.iloc[target_position]["time_idx_first_prediction"]),
        "batch_offset": target_position - batch_start,
        "batch_size": len(batch_indices),
        "chunk_index": chunk_start // chunk_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify fresh TFT forecasts against a soft-target cache")
    parser.add_argument("--env", default="local")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--exp-name", required=True, help="Soft-target experiment prefix")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--soft-targets-dir", default="artifacts/soft_targets")
    parser.add_argument("--samples-per-store", type=int, default=3)
    parser.add_argument("--generation-chunk-size", type=int, default=500)
    parser.add_argument("--origins-file", default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    cfg = load_config(env_name=args.env, experiment_name=args.experiment)
    cfg.environment.num_workers = 0
    checkpoint_path = resolve_path(args.checkpoint_path)
    targets_dir = resolve_path(args.soft_targets_dir)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    requested_origins = None
    if args.origins_file:
        with open(resolve_path(args.origins_file), "r", encoding="utf-8") as handle:
            requested_origins = json.load(handle).get("origins")
        if not isinstance(requested_origins, list) or not requested_origins:
            raise ValueError("--origins-file must contain a non-empty 'origins' list")

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
        if requested_origins is not None:
            if provenance.get("requested_origins") != requested_origins:
                raise ValueError(f"{store}: cache provenance does not match the requested origin schedule")
            required = torch.tensor(requested_origins, dtype=torch.long)
            if not torch.isfinite(cache_tensor[:, required]).all():
                raise ValueError(f"{store}: cache has missing/non-finite predictions for requested origins")

        frame = load_from_cache(dataset_dir, store)
        frame = frame[(frame["time_idx"] >= 1) & (frame["time_idx"] <= max_day)].copy()
        for column in CAT_COLUMNS:
            if column in frame:
                frame[column] = frame[column].astype(str).astype("category")

        # Preserve the generator's original order. It chunks this array directly.
        items = frame["item_id"].astype(str).unique()
        item_positions = np.linspace(0, len(items) - 1, num=min(args.samples_per_store, len(items)), dtype=int)
        generation_batch_size = int(provenance["batch_size"])
        generation_chunk_size = int(provenance.get("chunk_size", args.generation_chunk_size))
        for item_position in item_positions:
            item = items[item_position]
            context = _generation_context_batch(
                training_data=training_data,
                frame=frame,
                generation_items=items,
                target_item=item,
                cache_tensor=cache_tensor,
                local_by_global=local_by_global,
                group_encoder=group_encoder,
                chunk_size=generation_chunk_size,
                batch_size=generation_batch_size,
            )
            x_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in context["x"].items()
            }
            with torch.no_grad():
                fresh = teacher.to_prediction(teacher(x_device)).detach().cpu()
            if fresh.ndim == 3 and fresh.shape[-1] == 1:
                fresh = fresh[..., 0]

            target = cache_tensor[context["cache_local_row"], context["origin"]]
            prediction = fresh[context["batch_offset"]]
            difference = (prediction - target).abs()
            result = {
                "store": store,
                "series_id": context["series_id"],
                "global_code": context["global_code"],
                "cache_local_row": context["cache_local_row"],
                "origin": context["origin"],
                "horizon": int(target.shape[-1]),
                "generation_chunk_index": context["chunk_index"],
                "generation_batch_size": context["batch_size"],
                "generation_batch_offset": context["batch_offset"],
                "mean_abs_diff": float(difference.mean()),
                "max_abs_diff": float(difference.max()),
                "allclose": bool(torch.allclose(prediction, target, rtol=args.rtol, atol=args.atol)),
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
