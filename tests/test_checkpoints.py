from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scannet.checkpoints import (
    convert_checkpoint,
    convert_feature_slots,
    infer_scene_spec,
    make_inference_checkpoint,
    normalize_scene_state_dict,
    restore_scene_state_dict,
    split_checkpoint,
)


def make_state(*, n: int = 3, hidden: int = 8, pe: int = 6, slots: int = 5, dim: int = 7):
    encoded = 12 * pe
    return {
        "positions": torch.randn(n, 3),
        "log_scales": torch.randn(n, 3),
        "quats": torch.randn(n, 4),
        "W1": torch.randn(n, hidden, encoded),
        "b1": torch.randn(n, hidden),
        "W2": torch.randn(n, hidden, hidden),
        "b2": torch.randn(n, hidden),
        "W3": torch.randn(n, 4, hidden),
        "b3": torch.randn(n, 4),
        "feat_attr": torch.randn(n, slots, dim),
    }


def directional_value(feature: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    if feature.shape[1] == 5:
        return feature[:, 0] + (gates[..., None] * feature[:, 1:]).sum(dim=1)
    return (gates[..., None] * feature).sum(dim=1)


def test_infer_dimensions_and_validate_shapes():
    state = make_state(n=11, hidden=32, pe=10, slots=5, dim=512)
    spec = infer_scene_spec(state)
    assert spec.num_primitives == 11
    assert spec.hidden_dim == 32
    assert spec.encoded_dim == 120
    assert spec.pe_frequencies == 10
    assert spec.feature_slots == 5
    assert spec.feature_dim == 512

    state["W3"] = torch.empty(11, 32, 4)
    with pytest.raises(ValueError, match="W3 has shape"):
        infer_scene_spec(state)

    state = make_state()
    del state["positions"]
    with pytest.raises(KeyError, match="positions"):
        infer_scene_spec(state)


def test_module_prefix_and_legacy_mlp_remap():
    current = make_state()
    reverse = {"W1": "mlps.lin1.weight", "b1": "mlps.lin1.bias", "W2": "mlps.lin2.weight", "b2": "mlps.lin2.bias", "W3": "mlps.lin3.weight", "b3": "mlps.lin3.bias"}
    legacy = {f"module._orig_mod.{reverse.get(k, k)}": v for k, v in current.items()}
    normalized = normalize_scene_state_dict(legacy)
    assert all(key in normalized for key in ("W1", "b1", "W2", "b2", "W3", "b3"))
    assert not any(key.startswith(("module.", "_orig_mod.", "mlps.")) for key in normalized)
    assert infer_scene_spec(normalized) == infer_scene_spec(current)


def test_five_to_four_is_algebraically_equivalent():
    generator = torch.Generator().manual_seed(17)
    feature5 = torch.randn(9, 5, 13, generator=generator, dtype=torch.float64)
    gates = torch.softmax(torch.randn(9, 4, generator=generator, dtype=torch.float64), dim=1)
    feature4 = convert_feature_slots(feature5, 4)
    torch.testing.assert_close(
        directional_value(feature5, gates), directional_value(feature4, gates), rtol=1e-13, atol=1e-13
    )


def test_four_to_five_round_trip_is_exact():
    feature4 = torch.randn(5, 4, 19, dtype=torch.float64)
    feature5 = convert_feature_slots(feature4, 5)
    torch.testing.assert_close(convert_feature_slots(feature5, 4), feature4, rtol=0, atol=0)


def test_wrapper_metadata_is_preserved_and_shape_metadata_updated():
    raw = {
        "format_version": 1,
        "epoch": 60_000,
        "engine": "tile_cuda",
        "feature_slots": 5,
        "feature_dim_low": 7,
        "optimizer_state_dict": {"state": {123: {"step": torch.tensor(8)}}},
        "metadata": {
            "feature_slots": 5,
            "feature_dim": 7,
            "feature_slot_layout": "legacy_base_plus_four_deltas",
            "keep": "yes",
        },
        "custom_metadata": {"keep": [1, 2, 3]},
        "scene_state_dict": make_state(),
    }
    converted, before, after = convert_checkpoint(raw)
    assert split_checkpoint(converted).wrapper_key == "scene_state_dict"
    assert converted["epoch"] == 60_000
    assert "optimizer_state_dict" not in converted
    assert converted["custom_metadata"] is raw["custom_metadata"]
    assert converted["feature_slots"] == 4
    assert converted["feature_dim_low"] == 7
    assert converted["metadata"] == {
        "feature_slots": 4,
        "feature_dim": 7,
        "feature_slot_layout": "direct_tetrahedral",
        "keep": "yes",
    }
    assert before.feature_slots == 5 and after.feature_slots == 4


def test_bare_state_dict_stays_bare():
    converted, _, after = convert_checkpoint(make_state())
    assert "scene_state_dict" not in converted
    assert converted["feat_attr"].shape == (3, 4, 7)
    assert after.feature_slots == 4


def test_common_state_dict_wrapper_is_supported():
    raw = {"state_dict": make_state(), "epoch": 12, "keep": "metadata"}
    view = split_checkpoint(raw)
    assert view.wrapper_key == "state_dict"

    converted, _, after = convert_checkpoint(raw)
    assert converted["epoch"] == 12
    assert converted["keep"] == "metadata"
    assert converted["state_dict"]["feat_attr"].shape == (3, 4, 7)
    assert after.feature_slots == 4


def test_cli_writes_new_file_and_refuses_overwrite(tmp_path: Path):
    source = tmp_path / "scene.pt"
    torch.save({"scene_state_dict": make_state(), "epoch": 4}, source)
    script = Path(__file__).parents[1] / "tools" / "convert_legacy_checkpoint.py"
    command = [sys.executable, str(script), str(source)]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "scene.slots4.pt"
    assert output.is_file()

    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 4
    assert saved["scene_state_dict"]["feat_attr"].shape[1] == 4

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


@pytest.mark.parametrize("slots", [0, 4, 5])
def test_inference_storage_round_trip_preserves_geometry_and_inputs(slots):
    state = make_state(slots=max(slots, 1))
    if slots == 0:
        state.pop("feat_attr")
    state["stat_sum_w"] = torch.randn(3)
    state["stat_hits"] = torch.arange(3, dtype=torch.int32)
    original = {k: v.clone() for k, v in state.items()}
    raw = {"scene_state_dict": state, "epoch": 60000,
           "metadata": {"camera_mode": "colmap-native", "camera_intrinsics": {"fx": 123}},
           "optimizer_state_dict": {"state": {1: {"exp_avg": torch.ones(3)}}},
           "feature_decoder_state_dict": {"weight": torch.ones(2, 2)}}
    exported = make_inference_checkpoint(raw)
    stored = split_checkpoint(exported).scene_state_dict
    restored = restore_scene_state_dict(stored)
    assert exported["metadata"]["inference_only"] is True
    assert exported["metadata"]["camera_intrinsics"] == {"fx": 123}
    assert exported["epoch"] == 60000
    assert "optimizer_state_dict" not in exported and "optimizer_state_dict" in raw
    assert "inference_only" not in raw["metadata"]
    assert exported["feature_decoder_state_dict"]["weight"].dtype == torch.float32
    canonical = split_checkpoint(make_inference_checkpoint(raw, storage_precision="fp32")).scene_state_dict
    for k, v in state.items():
        assert torch.equal(v, original[k]) and v.dtype == original[k].dtype
        assert stored[k].data_ptr() != v.data_ptr()
        if k in {"W1", "b1", "W2", "b2", "W3", "b3", "feat_attr"}:
            assert stored[k].dtype == torch.float16
            torch.testing.assert_close(restored[k], canonical[k].half().float(), rtol=0, atol=0)
        else:
            assert stored[k].dtype == v.dtype
            assert torch.equal(stored[k].view(torch.uint8), v.view(torch.uint8))
    assert infer_scene_spec(stored).feature_slots == (4 if slots == 5 else slots)


@pytest.mark.parametrize("key,value", [("W1", float("nan")), ("positions", float("inf")), ("b1", 70000.)])
def test_inference_export_rejects_nonfinite_and_half_overflow(key, value):
    state = make_state()
    state[key].flatten()[0] = value
    with pytest.raises(ValueError, match="non-finite|overflow"):
        make_inference_checkpoint(state)


def test_restore_rejects_half_geometry_and_unknown_precision():
    state = make_state()
    state["positions"] = state["positions"].half()
    with pytest.raises(TypeError, match="positions"):
        restore_scene_state_dict(state)
    state = make_state()
    state["W1"] = state["W1"].bfloat16()
    with pytest.raises(TypeError, match="unsupported"):
        restore_scene_state_dict(state)
    with pytest.raises(ValueError, match="storage_precision"):
        make_inference_checkpoint(make_state(), storage_precision="unknown")


def test_export_rejects_unsupported_layout_and_geometry_precision():
    state = make_state()
    state["feat_attr"] = state["feat_attr"][:, 0]
    with pytest.raises(ValueError, match="4 or 5 slots"):
        make_inference_checkpoint(state)
    state = make_state()
    state["positions"] = state["positions"].double()
    with pytest.raises(TypeError, match="FP32 geometry"):
        make_inference_checkpoint(state)
    with pytest.raises(ValueError, match="target_feature_slots"):
        make_inference_checkpoint(make_state(), target_feature_slots=1)


def test_export_respects_explicit_five_slot_layout():
    state = make_state()
    exported = make_inference_checkpoint(state, target_feature_slots=5)
    assert exported["metadata"]["feature_slots"] == 5
    assert torch.equal(exported["scene_state_dict"]["feat_attr"], state["feat_attr"].half())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_export_from_cuda_does_not_mutate_live_state():
    state = {k: v.cuda().requires_grad_() for k, v in make_state(hidden=32, dim=512).items()}
    before = {k: v.detach().cpu().clone() for k, v in state.items()}
    exported = make_inference_checkpoint(state)
    expected = make_inference_checkpoint(before)
    for key, value in exported["scene_state_dict"].items():
        assert value.device.type == "cpu" and not value.requires_grad
        assert torch.equal(value, expected["scene_state_dict"][key])
        assert state[key].device.type == "cuda" and state[key].dtype == torch.float32
        assert torch.equal(state[key].detach().cpu(), before[key])


def test_mixed_export_is_idempotent_and_bare_input_gets_metadata():
    once = make_inference_checkpoint(make_state())
    twice = make_inference_checkpoint(once)
    assert split_checkpoint(once).wrapper_key == "scene_state_dict"
    for key, value in once["scene_state_dict"].items():
        assert torch.equal(value, twice["scene_state_dict"][key])
    assert once["metadata"] == twice["metadata"]


def test_save_export_does_not_quantize_training_or_adam_snapshot(tmp_path):
    from scannet.train import _save_scene_with_optional_decoder, _save_train_state, _load_resume_state
    scene = torch.nn.Module()
    for name, tensor in make_state().items():
        scene.register_parameter(name, torch.nn.Parameter(tensor))
    scene.checkpoint_metadata = {"feature_slots": 5}
    optimizer = torch.optim.Adam(scene.parameters())
    sum(p.square().sum() for p in scene.parameters()).backward()
    optimizer.step()
    before = {k: v.clone() for k, v in scene.state_dict().items()}
    inference = tmp_path / "scene.pt"
    training = tmp_path / "train_state_latest.pt"
    _save_scene_with_optional_decoder(str(inference), scene, None, epoch=9)
    _save_train_state(str(training), scene, optimizer, None, False, 7, 5, 7, 9, "tile_cuda", "tilecenter", str(tmp_path))
    saved = torch.load(training, weights_only=True)
    assert saved["scene_state_dict"]["feat_attr"].shape[1] == 5
    assert not saved["metadata"].get("inference_only", False)
    for key, value in scene.state_dict().items():
        assert value.dtype == torch.float32 and torch.equal(value, before[key])
        assert torch.equal(saved["scene_state_dict"][key], before[key])
    for moments in saved["optimizer_state_dict"]["state"].values():
        assert moments["exp_avg"].dtype == torch.float32
        assert moments["exp_avg_sq"].dtype == torch.float32
    with pytest.raises(ValueError, match="train_state_latest"):
        _load_resume_state(str(inference), scene, optimizer, None, False, "cpu")
    epoch, resumed = _load_resume_state(str(training), scene, optimizer, None, False, "cpu")
    assert (epoch, resumed) == (9, True)


@pytest.mark.parametrize("loader", ["rgb", "semantic"])
def test_both_evaluation_loaders_restore_float32(tmp_path, loader):
    from scannet import eval_rgb_psnr, eval_lseg_table2
    path = tmp_path / "scene.pt"
    torch.save(make_inference_checkpoint(make_state(hidden=32, dim=512)), path)
    if loader == "rgb":
        scene = eval_rgb_psnr._load_scene_tile_cuda(str(path), torch.device("cpu"), 4, 8)
    else:
        scene, _ = eval_lseg_table2._load_scene_tile_cuda(str(path), torch.device("cpu"), 4, 8)
    assert scene.feature_slots == 4
    assert all(t.dtype == torch.float32 for t in scene.state_dict().values() if t.is_floating_point())
