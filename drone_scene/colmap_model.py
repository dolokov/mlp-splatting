"""Small COLMAP binary reader; image observations/tracks are preserved verbatim.

Format: https://colmap.github.io/format.html#binary-file-format
No pycolmap import here: gsplat's example uses a different, older package of that name.
"""
from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy as np

from .common import PipelineError

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise PipelineError("Truncated COLMAP model")
    return data


def unpack(handle, fmt):
    return struct.unpack("<" + fmt, exact(handle, struct.calcsize("<" + fmt)))


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise PipelineError(f"Unsafe COLMAP image name: {name!r}")
    return path


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    qvec: tuple
    tvec: tuple
    camera_id: int
    name: str
    observations: bytes


def read_images(path):
    images = []
    with Path(path).open("rb") as handle:
        (count,) = unpack(handle, "Q")
        if count > 10_000_000:
            raise PipelineError("Invalid image count")
        for _ in range(count):
            values = unpack(handle, "idddddddi")
            name = bytearray()
            while (char := exact(handle, 1)) != b"\0":
                name.extend(char)
                if len(name) > 8192:
                    raise PipelineError("Invalid image filename")
            name = name.decode("utf-8")
            safe_name(name)
            (npoints,) = unpack(handle, "Q")
            observations = struct.pack("<Q", npoints) + exact(handle, 24 * npoints)
            images.append(ImageRecord(values[0], values[1:5], values[5:8], values[8],
                                      name, observations))
    if len({x.name for x in images}) != len(images):
        raise PipelineError("Duplicated image names in COLMAP model")
    return images


def write_images(path, images):
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for im in images:
            safe_name(im.name)
            handle.write(struct.pack("<idddddddi", im.image_id, *im.qvec, *im.tvec, im.camera_id))
            handle.write(im.name.encode("utf-8") + b"\0")
            handle.write(im.observations)


def read_cameras(path):
    cameras = {}
    with Path(path).open("rb") as handle:
        (count,) = unpack(handle, "Q")
        for _ in range(count):
            camera_id, model, width, height = unpack(handle, "iiQQ")
            if model not in CAMERA_MODELS:
                raise PipelineError(f"Unsupported COLMAP camera model id {model}")
            model_name, nparams = CAMERA_MODELS[model]
            params = unpack(handle, "d" * nparams)
            cameras[camera_id] = {"model": model_name, "width": width,
                                  "height": height, "params": list(params)}
    return cameras


def count_points(path):
    with Path(path).open("rb") as handle:
        return unpack(handle, "Q")[0]


def shared_calibration(images, cameras):
    used = [cameras[im.camera_id] for im in images]
    if not used:
        return False
    first = used[0]
    return all(
        c["model"] == first["model"] and c["width"] == first["width"]
        and c["height"] == first["height"]
        and np.allclose(c["params"], first["params"], rtol=1e-6, atol=1e-6)
        for c in used
    )


def flattened(images):
    original_names = ["__".join(safe_name(im.name).parts) for im in images]
    if len(set(original_names)) != len(images):
        raise PipelineError("Flattening image paths would collide")
    # MLP sorts numeric suffixes; gsplat sorts full names. A global suffix keeps
    # their periodic holdout order identical, including across multiple clips.
    order = {name: index for index, name in enumerate(sorted(im.name for im in images))}
    result = []
    for im, name in zip(images, original_names):
        path = Path(name)
        result.append(replace(im, name=f"{path.stem}_g{order[im.name]:08d}{path.suffix}"))
    return result


def assess_components(models, retained, min_fraction, per_video_fraction):
    """A best component is accepted only if it covers the capture and every input clip."""
    counts = {clip: len(names) for clip, names in retained.items()}
    summaries = []
    for model in models:
        images = read_images(model / "images.bin")
        covered = {clip: 0 for clip in counts}
        for im in images:
            clip = safe_name(im.name).parts[0]
            if clip in covered:
                covered[clip] += 1
        summaries.append({"model": str(model), "registered": len(images), "per_video": covered})
    if not summaries:
        raise PipelineError("COLMAP did not produce a model. Add overlap, texture, and sharper frames.")
    summaries.sort(key=lambda s: s["registered"], reverse=True)
    best = summaries[0]
    total = sum(counts.values())
    failures = []
    if best["registered"] / total < min_fraction:
        failures.append(f"only {best['registered']}/{total} frames registered")
    for clip, count in counts.items():
        if best["per_video"][clip] / count < per_video_fraction:
            failures.append(f"clip {clip}: {best['per_video'][clip]}/{count} registered")
    return best, {"components": summaries, "total_frames": total,
                  "registered_fraction": best["registered"] / total, "failures": failures}


def hybrid_pairs(retained, window, stride):
    """Temporal neighbors plus globally compared keyframes (including cross-flight pairs)."""
    pairs = set()
    for names in retained.values():
        for i, a in enumerate(names):
            for b in names[i + 1:i + 1 + window]:
                pairs.add(tuple(sorted((a, b))))
    keys = []
    for names in retained.values():
        keys.extend(names[::stride])
        if names[-1] not in keys:
            keys.append(names[-1])
    pairs.update(tuple(sorted(pair)) for pair in itertools.combinations(keys, 2))
    return sorted(pairs)
