"""Read-only regression checks for common-evaluator checkpoint metadata."""

from __future__ import annotations

import glob
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_checkpoints import checkpoint_training_state, objective_label


FIXTURES = {
    "Quantile": ("outputs/teacher/tft64_optimized/*epoch=05*.ckpt", None, None),
    "Huber": ("outputs/teacher/tft64_huber/*epoch=05*.ckpt", None, None),
    "WRMSSE-informed": (
        "outputs/teacher/tft64_wrmsse_informed/*epoch=09*.ckpt", 9, 212000
    ),
}


def main():
    for expected_label, (pattern, expected_epoch, expected_step) in FIXTURES.items():
        matches = glob.glob(pattern)
        if len(matches) != 1:
            raise AssertionError(f"Expected one {expected_label} fixture, found {matches}")
        path = matches[0]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        label = objective_label(checkpoint["hyper_parameters"]["loss"])
        if label != expected_label:
            raise AssertionError(f"{os.path.basename(path)}: {label} != {expected_label}")
        state = checkpoint_training_state(path)
        if expected_epoch is not None and state["internal_epoch"] != expected_epoch:
            raise AssertionError(f"Unexpected epoch: {state}")
        if expected_step is not None and state["global_step"] != expected_step:
            raise AssertionError(f"Unexpected global step: {state}")
        print({"checkpoint": path, "objective": label, **state})


if __name__ == "__main__":
    main()
