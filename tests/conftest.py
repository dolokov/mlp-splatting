import struct

import pytest

from drone_scene.colmap_model import ImageRecord, write_images


@pytest.fixture
def model_factory(tmp_path):
    def create(name, image_names, camera_ids=None):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        camera_ids = camera_ids or [1] * len(image_names)
        used = sorted(set(camera_ids))
        with (folder / "cameras.bin").open("wb") as handle:
            handle.write(struct.pack("<Q", len(used)))
            for camera_id in used:
                handle.write(struct.pack("<iiQQdddd", camera_id, 1, 32, 24, 28., 29., 16., 12.))
        images = [ImageRecord(i + 1, (1., 0., 0., 0.), (float(i), 0., 1.), camera_id, name,
                               struct.pack("<Qddq", 1, 5.5, 6.25, -1))
                  for i, (name, camera_id) in enumerate(zip(image_names, camera_ids))]
        write_images(folder / "images.bin", images)
        (folder / "points3D.bin").write_bytes(struct.pack("<Q", 123) + b"synthetic-point-payload")
        return folder, images
    return create
