"""Read and validate a checkpoint-selection manifest for a held-out launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    manifest_path = os.path.abspath(args.manifest)
    run_dir = os.path.realpath(args.run_dir)
    with open(manifest_path, encoding="utf-8") as handle:
        selected = json.load(handle)["selected_checkpoint"]
    path = os.path.realpath(selected["checkpoint_path"])
    if os.path.commonpath([path, run_dir]) != run_dir:
        raise ValueError("Selected checkpoint is outside the requested run directory")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Selected checkpoint is missing or empty: {path}")
    actual_sha = sha256(path)
    if actual_sha != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint SHA-256 differs from the validation manifest")
    print(path)
    print(actual_sha)
    print(selected["validation_WRMSSE"])


if __name__ == "__main__":
    main()
