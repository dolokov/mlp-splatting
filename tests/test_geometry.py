import re
import struct
from dataclasses import replace

import numpy as np
import pytest

from drone_scene.colmap_model import (
    assess_components, count_points, flattened, hybrid_pairs, read_cameras, read_images,
    safe_name, shared_calibration, write_images,
)
from drone_scene.common import PipelineError
from drone_scene.trajectory import interpolate, select_clip


def test_binary_renaming_preserves_ids_calibration_and_observations(model_factory, tmp_path):
    model, original = model_factory("source", ["flightB/00000001.jpg", "flightA/00000001.jpg"])
    transformed = flattened(original)
    out = tmp_path / "renamed.bin"
    write_images(out, transformed)
    actual = read_images(out)
    assert [replace(im, name=old.name) for im, old in zip(actual, original)] == original
    assert len({im.name for im in actual}) == 2
    assert all("/" not in im.name for im in actual)
    cameras = read_cameras(model / "cameras.bin")
    assert cameras[1] == {"model": "PINHOLE", "width": 32, "height": 24,
                          "params": [28., 29., 16., 12.]}
    assert shared_calibration(actual, cameras)
    assert count_points(model / "points3D.bin") == 123


def test_both_backend_sort_orders_keep_the_same_holdout(model_factory):
    names = [f"v{clip:03d}/000000{frame:02d}.jpg" for frame in range(1, 14) for clip in range(2)]
    _, images = model_factory("sort", names)
    names = [im.name for im in flattened(images)]
    lexical = sorted(names)
    numeric = sorted(names, key=lambda name: (int(re.search(r"(\d+)\.", name)[1]), name))
    assert lexical == numeric
    assert lexical[::8] == numeric[::8]


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "a/../../escape.jpg", "a\\b.jpg"])
def test_model_rejects_paths_outside_image_root(name):
    with pytest.raises(PipelineError):
        safe_name(name)


def test_truncated_model_is_a_clear_error(tmp_path):
    path = tmp_path / "images.bin"
    path.write_bytes(struct.pack("<Q", 1) + b"short")
    with pytest.raises(PipelineError, match="Truncated"):
        read_images(path)


def test_differing_calibration_is_detected(model_factory):
    model, images = model_factory("multi", ["a/1.jpg", "b/2.jpg"], [1, 2])
    cameras = read_cameras(model / "cameras.bin")
    assert shared_calibration(images, cameras)
    cameras[2]["params"][0] = 100.
    assert not shared_calibration(images, cameras)


def test_hybrid_pairs_connect_clips_and_cover_temporal_neighbors():
    retained = {"a": [f"a/{i:04d}.jpg" for i in range(20)],
                "b": [f"b/{i:04d}.jpg" for i in range(20)]}
    pairs = hybrid_pairs(retained, window=3, stride=5)
    assert ("a/0001.jpg", "a/0004.jpg") in pairs
    assert ("a/0000.jpg", "b/0005.jpg") in pairs
    assert ("a/0000.jpg", "a/0019.jpg") in pairs
    assert len(pairs) == len(set(pairs))
    assert all(a != b and a < b for a, b in pairs)


def test_component_checks_do_not_accept_only_one_video(model_factory):
    retained = {"a": [f"a/{i}.jpg" for i in range(20)], "b": [f"b/{i}.jpg" for i in range(4)]}
    first, _ = model_factory("components/0", retained["a"])
    second, _ = model_factory("components/1", retained["b"])
    best, report = assess_components([first, second], retained, .8, .6)
    assert best["registered"] == 20
    assert report["registered_fraction"] > .8
    assert any("clip b" in failure for failure in report["failures"])


def test_component_checks_report_missing_fraction(model_factory):
    retained = {"a": [f"a/{i}.jpg" for i in range(10)], "b": [f"b/{i}.jpg" for i in range(10)]}
    model, _ = model_factory("partial", retained["a"][:6] + retained["b"][:6])
    _, report = assess_components([model], retained, .85, .6)
    assert report["registered_fraction"] == .6
    assert report["failures"] == ["only 12/20 frames registered"]


def test_clip_selection_avoids_cross_flight_interpolation():
    names = ["a__00001.jpg", "b__00001.jpg", "b__00002.jpg", "b__00003.jpg"]
    assert select_clip(names) == ("b", [1, 2, 3])
    with pytest.raises(PipelineError):
        select_clip(names, "a")


def test_slerp_retains_endpoints_and_valid_rotations():
    pytest.importorskip("scipy")
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, :3, :3] = np.diag([-1., -1., 1.])
    poses[1, :3, 3] = [4., 2., 0.]
    result = interpolate(poses, steps=4, max_frames=10)
    assert result.shape == (5, 4, 4)
    np.testing.assert_allclose(result[[0, -1]], poses, atol=1e-6)
    np.testing.assert_allclose(result[2, :3, 3], [2., 1., 0.])
    for rotation in result[:, :3, :3]:
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-6)
