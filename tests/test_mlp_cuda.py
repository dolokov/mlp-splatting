"""Opt-in: real NVIDIA device checks, separate from CPU regression tests."""
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("DRONE_SCENE_RUN_CUDA_TESTS") != "1",
                    reason="Set DRONE_SCENE_RUN_CUDA_TESTS=1 on the configured NVIDIA workstation")
def test_real_mlp_forward_backward_update_and_native_checkpoint(tmp_path):
    from drone_scene.config import load_config
    from drone_scene.mlp_smoke import run

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/mlp-pilot.json")
    report = run(cfg, tmp_path)
    assert report["passed"]
    assert report["final_loss"] < report["initial_loss"]
    assert report["checkpoint_max_rgb_error"] < .006
