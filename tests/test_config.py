import json
import inspect
import os
from pathlib import Path

import pytest

from mip360.config import parse_args_with_config
from mip360.cumodel import _tile_ext_build_config
from mip360.eval_rgb_psnr import build_parser as build_eval_parser
from mip360.run import build_parser as build_train_parser, parse_training_args

ROOT = Path(__file__).resolve().parents[1]
SCENES = {"room0": (70, 4402, ""), "room1": (69, 5134, "rgb_009.png"),
          "office3": (70, 2257, ""), "office4": (63, 1503, "rgb_102.png,rgb_108.png")}


@pytest.mark.parametrize("scene", SCENES)
def test_replica_recipe_train_eval_identity(scene):
    views, primitives, exclusions = SCENES[scene]
    path = ROOT / "configs" / "replica" / f"{scene}_lseg.json"
    train = parse_training_args(["--config", str(path), "--data", scene, "--output", "run"])
    assert train.dataset == "replica"
    assert train.max_points_per_leaf == 4
    assert train.num_epochs == 60000
    assert train.expected_views == views
    assert train.expected_primitives == primitives
    assert train.exclude_train_images == exclusions
    assert (train.training_backend, train.csr_forward, train.csr_precision) == ("csr", "wmma_fp16", "float16")
    assert (train.csr_max_hits, train.tile_candidate_capacity) == (512, 1536)
    assert (train.tile_order, train.tile_size, train.tile_mosaic_k_views) == ("tilecenter", 8, 4)
    assert (train.feature_type, train.feature_slots, train.feature_lowdim_divisor) == ("lseg", 5, 1)
    assert train.preload_features and train.dataloader_num_workers == 0

    evaluate = parse_args_with_config(build_eval_parser(),
        ["--config", str(path), "--data", scene, "--checkpoint", "scene.pt"], section="evaluation")
    assert evaluate.dataset_loader == "replica"
    assert evaluate.expected_views == 10
    assert evaluate.max_points_per_leaf == train.max_points_per_leaf
    assert evaluate.split_test_offset == train.split_test_offset == 2
    assert evaluate.tile_candidate_capacity == train.tile_candidate_capacity
    assert evaluate.score_threshold == pytest.approx(.005)
    assert evaluate.limit_frames == 0


@pytest.mark.parametrize("scene", SCENES)
def test_replica_defaults_and_training_exclusions(scene):
    args = parse_training_args(["--dataset", "replica", "--data", scene])
    assert args.max_points_per_leaf == 4
    assert args.exclude_train_images == SCENES[scene][2]
    assert args.tile_candidate_capacity == 1536


def test_other_dataset_default_is_preserved():
    assert parse_training_args(["--data", "scene"]).max_points_per_leaf == 10


def test_cli_overrides_config():
    path = ROOT / "configs" / "replica" / "room1_lseg.json"
    args = parse_training_args(["--config", str(path), "--data", "room1",
        "--iterations", "7", "--max-points-per-leaf", "6", "--exclude-train-images", ""])
    assert args.num_epochs == 7
    assert args.max_points_per_leaf == 6
    assert args.exclude_train_images == ""


@pytest.mark.parametrize("training", [
    {"not_a_real_option": 1}, {"tile_order": "random"},
    {"tile_mosaic": "yes"}, {"data_dir": "/machine/specific/path"},
])
def test_invalid_config_fails_early(tmp_path, training):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "training": training}), encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args_with_config(build_train_parser(), ["--config", str(path), "--data", "scene"], section="training")


def test_default_capacity_selects_full_kernel_build(monkeypatch):
    monkeypatch.delenv("MLPSP_TILE_CANDIDATE_CAPACITY", raising=False)
    monkeypatch.setenv("MLPSP_TRAINING_BACKEND", "csr")
    monkeypatch.setenv("MLPSP_CSR_MAX_HITS", "512")
    monkeypatch.delenv("MLPSP_NVCC_MAXRREGCOUNT", raising=False)
    _, name, flags = _tile_ext_build_config(8)
    assert name == "np_tile_kernels_csr_h512_c1536"
    assert "-DMLPSP_MAX_TILE_PRIMS=1536" in flags
    assert "-DMLPSP_TILECENTER_MAX_HITS=512" in flags


def test_training_entrypoint_forwards_replica_recipe(tmp_path, monkeypatch):
    from mip360 import run

    signature = inspect.signature(run.train_scene_llff)
    captured = {}

    def capture(*args, **kwargs):
        captured.update(signature.bind(*args, **kwargs).arguments)

    for key in ("MLPSP_EXCLUDE_TRAIN_IMAGES", "MLPSP_TILE_CANDIDATE_CAPACITY",
                "MLPSP_TRAINING_BACKEND", "MLPSP_CSR_FORWARD_BACKEND",
                "MLPSP_CSR_MLP_PRECISION", "MLPSP_CSR_MAX_HITS"):
        monkeypatch.setenv(key, "")
    # Parser defaults read the backend environment before applying the config.
    monkeypatch.setenv("MLPSP_TRAINING_BACKEND", "legacy")
    monkeypatch.setenv("MLPSP_CSR_MAX_HITS", "128")
    monkeypatch.setattr(run, "train_scene_llff", capture)
    monkeypatch.setattr(run, "set_seed", lambda seed: None)
    output = tmp_path / "output"
    run.main(["--config", str(ROOT / "configs/replica/room1_lseg.json"),
              "--data", "room1", "--output", str(output), "--tag", "example"])
    assert captured["output_root"] == str(output)
    assert captured["run_tag"] == "example"
    assert captured["min_points"] == 4
    assert captured["num_epochs"] == 60000
    assert captured["renorm_quats_after_step"] is True
    assert captured["preload_features"] is True
    assert captured["feature_type"] == "lseg"
    assert captured["feature_slots"] == 5
    assert captured["lr"] == pytest.approx(.001)
    assert captured["log_scale_lr"] == pytest.approx(.01)
    assert os.environ["MLPSP_EXCLUDE_TRAIN_IMAGES"] == "rgb_009.png"
    assert os.environ["MLPSP_TILE_CANDIDATE_CAPACITY"] == "1536"
    assert os.environ["MLPSP_CSR_MAX_HITS"] == "512"
    resolved = json.loads((output / "example/resolved_config.json").read_text())
    assert Path(resolved["output_dir"]) == Path(captured["output_root"]) / captured["run_tag"]
    assert resolved["max_points_per_leaf"] == 4
    assert resolved["exclude_train_images"] == "rgb_009.png"
