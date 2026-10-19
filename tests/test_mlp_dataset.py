from types import SimpleNamespace

import numpy as np
import pytest

from drone_scene.common import PipelineError
from drone_scene.mlp_dataset import selected_image_names
from drone_scene.trajectory import interpolate, select_clip


def dataset_for(names):
    selected = [i for i in range(len(names)) if i % 8 != 0]
    poses = np.tile(np.eye(4), (len(names), 1, 1))
    poses[:, 0, 3] = np.arange(len(names))
    return SimpleNamespace(image_paths=names,
                           image_paths_selected=[names[i] for i in selected],
                           poses=poses[selected])


def test_24_images_render_with_21_training_poses():
    dataset = dataset_for([f"/scene/images/flight__{i:04d}.jpg" for i in range(24)])
    names = selected_image_names(dataset)
    clip, indices = select_clip(names)
    assert clip == "flight"
    assert len(names) == len(indices) == 21
    assert names[0] == "flight__0001.jpg"
    assert names[-1] == "flight__0023.jpg"
    poses = interpolate(dataset.poses[indices], 5, 300)
    assert poses.shape == (101, 4, 4)
    np.testing.assert_array_equal(poses[[0, -1], 0, 3], [1, 23])


def test_multiple_clips_keep_names_aligned_with_pose_centers():
    dataset = dataset_for([f"/scene/{'first' if i < 12 else 'second'}__{i:04d}.jpg"
                           for i in range(24)])
    names = selected_image_names(dataset)
    clip, indices = select_clip(names, "second")
    assert clip == "second"
    assert len(indices) == 11
    np.testing.assert_array_equal(dataset.poses[indices, 0, 3],
                                  [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23])
    for name, pose in zip(names, dataset.poses):
        assert int(name.split("__")[1].split(".")[0]) == pose[0, 3]


@pytest.mark.parametrize("paths", [None, [], ["one.jpg", "two.jpg", "three.jpg"]])
def test_missing_or_misaligned_selected_paths_fail_clearly(paths):
    dataset = SimpleNamespace(poses=np.zeros((2, 3, 4)), image_paths=["unfiltered.jpg"])
    if paths is not None:
        dataset.image_paths_selected = paths
    with pytest.raises(PipelineError, match="selected image paths.*dataset poses"):
        selected_image_names(dataset)
