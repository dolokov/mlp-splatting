"""Training exclusions must never shift or remove held-out test views."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from replica import replica_loader


@pytest.fixture
def replica_scene(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images = {}
    for index in range(80):
        name = f"rgb_{index:03d}.png"
        (image_dir / name).touch()
        images[index] = SimpleNamespace(
            name=name, camera_id=1, qvec=np.array([1., 0., 0., 0.]),
            tvec=np.array([float(index), 0., 0.]),
        )
    camera = SimpleNamespace(id=1, model="PINHOLE", width=6, height=4,
                             params=np.array([10., 10., 3., 2.]))
    monkeypatch.setattr(replica_loader, "read_images_binary", lambda _: images)
    monkeypatch.setattr(replica_loader, "read_cameras_binary", lambda _: {1: camera})
    monkeypatch.setattr(replica_loader, "read_points3D_binary",
                        lambda _: {1: SimpleNamespace(xyz=np.array([0., 0., 1.]))})
    monkeypatch.setattr(replica_loader.imageio, "imread",
                        lambda _: np.zeros((4, 6, 3), dtype=np.uint8))
    return tmp_path


@pytest.mark.parametrize("excluded", ["rgb_009.png", "rgb_001.png,rgb_009.png"])
def test_exclusions_preserve_test_identity(replica_scene, monkeypatch, excluded):
    monkeypatch.setenv("MLPSP_EXCLUDE_TRAIN_IMAGES", excluded)
    kwargs = dict(basedir=str(replica_scene), factor=1, llffhold=8,
                  split_mode="replica_feature3dgs", split_test_offset=2)
    train = replica_loader.SceneDataset(**kwargs, split="train")
    test = replica_loader.SceneDataset(**kwargs, split="test")
    excluded_names = set(excluded.split(","))
    assert test.select_ids == list(range(2, 80, 8))
    assert len(test) == 10
    assert len(train) == 70 - len(excluded_names)
    assert excluded_names.isdisjoint(Path(p).name for p in train.image_paths_selected)
    assert set(train.select_ids).isdisjoint(test.select_ids)
    assert len(list((replica_scene / "images").iterdir())) == 80


@pytest.mark.parametrize("excluded", ["rgb_999.png", "rgb_002.png"])
def test_invalid_training_exclusion_fails(replica_scene, monkeypatch, excluded):
    monkeypatch.setenv("MLPSP_EXCLUDE_TRAIN_IMAGES", excluded)
    with pytest.raises(KeyError, match="not in the train split"):
        replica_loader.SceneDataset(
            str(replica_scene), factor=1, split="train",
            split_mode="replica_feature3dgs", split_test_offset=2,
        )
