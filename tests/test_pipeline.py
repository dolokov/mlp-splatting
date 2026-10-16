import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from drone_scene import backends
from drone_scene.cli import main
from drone_scene.colmap_model import read_images
from drone_scene.common import PipelineError, Runner, read_json, stage, write_json
from drone_scene.config import load_config
from drone_scene.reconstruct import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_all_shipped_configs_validate():
    for config in (ROOT / "configs").glob("*.json"):
        load_config(config)


def test_config_rejects_unknown_root_key(tmp_path):
    cfg = read_json(ROOT / "configs/titan.json")
    cfg["typo"] = True
    path = tmp_path / "invalid.json"
    write_json(path, cfg)
    with pytest.raises(PipelineError):
        load_config(path)


@pytest.mark.parametrize("section,key,value", [("sfm", "global_stride", 2.5),
                                               ("render", "max_frames", 1),
                                               ("render", "port", 99999),
                                               ("extraction", "fps", float("nan"))])
def test_invalid_config_fails_before_expensive_work(tmp_path, section, key, value):
    cfg = read_json(ROOT / "configs/titan.json")
    cfg[section][key] = value
    path = tmp_path / "invalid.json"
    write_json(path, cfg)
    with pytest.raises(PipelineError):
        load_config(path)


def test_cache_skips_only_complete_unchanged_stage(tmp_path):
    runner = Runner(tmp_path)
    output = tmp_path / "output.txt"
    calls = []

    def action():
        calls.append(1)
        output.write_text("ok")

    stage(runner, "example", {"input": 1}, [output], action)
    stage(runner, "example", {"input": 1}, [output], action)
    assert calls == [1]
    with pytest.raises(PipelineError, match="changed"):
        stage(runner, "example", {"input": 2}, [output], action)
    output.unlink()
    with pytest.raises(PipelineError, match="missing"):
        stage(runner, "example", {"input": 1}, [output], action)


def test_failed_stage_is_not_cached(tmp_path):
    with pytest.raises(PipelineError):
        stage(Runner(tmp_path), "broken", {}, [tmp_path / "missing"], lambda: None)
    assert not (tmp_path / "state/broken.json").exists()


def test_dry_run_is_read_only_and_needs_no_gpu(tmp_path, capsys):
    source = tmp_path / "inputs" / "a clip.MP4"
    source.parent.mkdir()
    source.write_bytes(b"dry run does not decode")
    work = tmp_path / "workspace"
    code = main(["run", "--config", str(ROOT / "configs/titan.json"),
                 "--videos", str(source), "--workspace", str(work), "--dry-run"])
    assert code == 0
    assert not work.exists()
    output = capsys.readouterr().out
    assert "matches_importer" in output and "simple_trainer.py" in output
    assert "render_backend" in output


def test_workspace_cannot_reingest_its_rendered_videos(tmp_path):
    (tmp_path / "input.mp4").write_bytes(b"video")
    code = main(["run", "--videos", str(tmp_path), "--workspace", str(tmp_path / "work"), "--dry-run"])
    assert code == 2


def test_native_mlp_rejects_turing_before_compilation(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (7, 5),
                           get_device_name=lambda: "NVIDIA TITAN RTX")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(PipelineError, match="Ampere"):
        backends.gpu_check("mlp")


def test_gsplat_checks_actual_toolkit_not_nvidia_smi(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (7, 5),
                           get_device_name=lambda: "NVIDIA TITAN RTX", mem_get_info=lambda: (20 * 2**30, 24 * 2**30))
    torch = SimpleNamespace(cuda=cuda, __version__="2.7.1+cu126", version=SimpleNamespace(cuda="12.6"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(backends.shutil, "which", lambda _: "/conda/bin/nvcc")
    monkeypatch.setattr(backends, "checked_output", lambda _: "Cuda compilation tools, release 13.4, V13.4")
    with pytest.raises(PipelineError, match="mismatch"):
        backends.gpu_check("gsplat")
    monkeypatch.setattr(backends, "checked_output", lambda _: "Cuda compilation tools, release 12.6, V12.6")
    assert backends.gpu_check("gsplat") == {"TORCH_CUDA_ARCH_LIST": "7.5"}


def test_prepare_preserves_model_geometry_and_matches_image_sizes(tmp_path, model_factory):
    names = [f"v{clip}/000000{frame:02d}.jpg" for clip in range(2) for frame in range(1, 9)]
    source_model, original = model_factory("original", names)
    write_json(tmp_path / "selected-model.json", {"model": str(source_model)})

    class FakeUndistorter(Runner):
        def run(self, command, **kwargs):
            assert command[:2] == ["colmap", "image_undistorter"]
            output = self.work / "undistorted"
            shutil.copytree(source_model, output / "sparse")
            for name in names:
                target = output / "images" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), (12, 30, 80)).save(target)

    cfg = read_json(ROOT / "configs/titan.json")
    prepare(FakeUndistorter(tmp_path), cfg)
    result = read_images(tmp_path / "scene/sparse/0/images.bin")
    assert len(result) == 16
    assert {im.image_id for im in result} == {im.image_id for im in original}
    assert [im.observations for im in result] == [im.observations for im in original]
    assert (tmp_path / "scene/sparse/0/points3D.bin").read_bytes() == (source_model / "points3D.bin").read_bytes()
    assert all((tmp_path / "scene/images" / im.name).exists() for im in result)
    assert read_json(tmp_path / "reports/dataset.json")["shared_calibration"]


def test_commands_choose_native_formats_and_do_not_confuse_resume(tmp_path):
    cfg = load_config(ROOT / "configs/titan.json")
    command, _ = backends.gsplat_command(cfg, tmp_path, verify=False)
    assert "--save-ply" in command and "--disable-viewer" in command
    assert "--ckpt" not in command
    viewer, _ = backends.gsplat_command(cfg, tmp_path, action="view", verify=False)
    assert "--ckpt" in viewer and "--disable-viewer" not in viewer
    native = copy.deepcopy(cfg)
    native["backend"] = "mlp"
    command, _ = backends.mlp_command(native, tmp_path, verify=False)
    assert "--camera-mode" in command and "colmap-native" in command
    assert "--save-ply" not in command
    assert backends.checkpoint(tmp_path, native).name == "scene_ckpt.pt"


def test_render_without_training_does_not_claim_success(tmp_path):
    code = main(["render", "--workspace", str(tmp_path / "empty")])
    assert code == 2


def test_repository_pin_file_is_valid():
    lock = json.loads((ROOT / "upstreams.lock.json").read_text())
    assert set(lock) == {"checked", "mlp", "gsplat"}
    assert all(len(lock[name]["commit"]) == 40 for name in ("mlp", "gsplat"))
