"""Exercise benchmark camera preflight with the real loader, without a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scannet import scannet_loader
from scannet.camera import distort_xy
from scannet.data_validation import parse_supported_camera
from tools import benchmark_inference as benchmark


class _SceneLoadReached(Exception):
    """Stop immediately before scene creation and GPU rendering."""


@pytest.fixture
def benchmark_case(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images = {}
    for index in range(3):
        name = f"rgb_{index:03d}.png"
        (image_dir / name).touch()
        images[index] = SimpleNamespace(
            name=name, camera_id=1, qvec=np.array([1., 0., 0., 0.]),
            tvec=np.array([float(index), 0., 0.]),
        )
    camera = SimpleNamespace(
        model="PINHOLE", width=6, height=4, params=[10., 10., 3., 2.],
    )
    monkeypatch.setattr(scannet_loader, "read_images_binary", lambda _: images)
    monkeypatch.setattr(scannet_loader, "read_cameras_binary", lambda _: {1: camera})
    monkeypatch.setattr(scannet_loader, "read_points3D_binary",
                        lambda _: {1: SimpleNamespace(xyz=np.array([0., 0., 1.]))})
    monkeypatch.setattr(scannet_loader.imageio, "imread",
                        lambda _: np.zeros((4, 6, 3), dtype=np.uint8))

    # Only CUDA availability/device selection is mocked. Dataset construction,
    # restricted checkpoint loading, calibration checks and rays are real.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setenv("MLPSP_TILE_ORDER", "tilecenter")
    reached = {}
    build_dataset = benchmark.evaluation._build_dataset

    def capture_dataset(cls, args):
        reached["dataset"] = build_dataset(cls, args)
        return reached["dataset"]

    def stop_at_scene(*args):
        reached["scene"] = True
        raise _SceneLoadReached

    monkeypatch.setattr(benchmark.evaluation, "_build_dataset", capture_dataset)
    monkeypatch.setattr(benchmark.evaluation, "_load_scene_tile_cuda", stop_at_scene)
    checkpoint = tmp_path / "scene.pt"
    argv = ["--data", str(tmp_path), "--checkpoint", str(checkpoint),
            "--output", str(tmp_path / "benchmark.json"), "--dataset-loader", "scannet"]
    return camera, checkpoint, argv, reached


def _native_checkpoint(camera):
    return {"metadata": {
        "camera_mode": "colmap-native", "camera_pixel_center": .5,
        "camera_culling": "raycone-v1",
        "camera_intrinsics": parse_supported_camera(
            camera, image_name="rgb_000.png", camera_mode="colmap-native",
        ),
    }}


@pytest.mark.parametrize("model,params", [
    ("PINHOLE", [10., 10., 3., 2.]),
    ("PINHOLE", [10., 12., 1.5, .5]),
    ("OPENCV", [10., 12., 1.5, .5, .1, -.02, .01, -.01]),
])
def test_benchmark_uses_native_calibration_and_pixel_centers(benchmark_case, model, params):
    camera, checkpoint, argv, reached = benchmark_case
    camera.model, camera.params = model, params
    torch.save(_native_checkpoint(camera), checkpoint)

    with pytest.raises(_SceneLoadReached):
        benchmark.main(argv)

    dataset = reached["dataset"]
    assert dataset.camera_mode == "colmap-native"
    _, rays = dataset._get_rays((4, 6), torch.eye(4)[:3])
    xy = rays[..., :2].double()
    if model == "OPENCV":
        xy = distort_xy(xy, params[4:])
    projected = xy * torch.tensor(params[:2]) + torch.tensor(params[2:4])
    yy, xx = torch.meshgrid(torch.arange(4) + .5, torch.arange(6) + .5, indexing="ij")
    torch.testing.assert_close(projected, torch.stack((xx, yy), -1).double(),
                               rtol=0, atol=1e-6)


def test_benchmark_preserves_legacy_checkpoint_rays(benchmark_case):
    _, checkpoint, argv, reached = benchmark_case
    torch.save({}, checkpoint)
    with pytest.raises(_SceneLoadReached):
        benchmark.main(argv)
    dataset = reached["dataset"]
    assert dataset.camera_mode == "legacy-centered"
    _, rays = dataset._get_rays((4, 6), torch.eye(4)[:3])
    torch.testing.assert_close(rays[0, 0], torch.tensor([-.3, -.2, 1.]))


def test_benchmark_rejects_calibration_mismatch_before_scene_load(benchmark_case):
    camera, checkpoint, argv, reached = benchmark_case
    payload = _native_checkpoint(camera)
    payload["metadata"]["camera_intrinsics"]["fx"] += 1.
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="calibration differ at fx"):
        benchmark.main(argv)
    assert "scene" not in reached


def test_benchmark_rejects_native_checkpoint_with_replica_loader(benchmark_case):
    camera, checkpoint, argv, reached = benchmark_case
    torch.save(_native_checkpoint(camera), checkpoint)
    argv[-1] = "replica"
    with pytest.raises(ValueError, match="requires the scannet COLMAP loader"):
        benchmark.main(argv)
    assert not reached


def test_benchmark_rejects_incompatible_native_metadata_before_dataset_load(benchmark_case):
    camera, checkpoint, argv, reached = benchmark_case
    payload = _native_checkpoint(camera)
    payload["metadata"]["camera_pixel_center"] = 0.
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="incompatible pixel-center/culling"):
        benchmark.main(argv)
    assert not reached
