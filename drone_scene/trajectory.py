from __future__ import annotations

from collections import Counter

import numpy as np

from .common import PipelineError


def clip_name(name):
    return name.split("__", 1)[0]


def select_clip(names, requested=None):
    counts = Counter(clip_name(name) for name in names)
    selected = requested if requested else counts.most_common(1)[0][0]
    if selected not in counts:
        raise PipelineError(f"Render clip {selected!r} not found. Choices: {sorted(counts)}")
    indices = [i for i, name in enumerate(names) if clip_name(name) == selected]
    if len(indices) < 2:
        raise PipelineError("Need at least two poses in the selected clip")
    return selected, indices


def interpolate(poses, steps, max_frames):
    """Linear camera centers and SO(3) slerp, never elementwise rotation interpolation."""
    from scipy.spatial.transform import Rotation, Slerp

    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2 or steps < 1 or max_frames < 2:
        raise PipelineError("Interpolation needs two poses, steps>=1 and max_frames>=2")
    n = min((len(poses) - 1) * steps + 1, max_frames)
    keys = np.arange(len(poses))
    times = np.linspace(0, len(poses) - 1, n)
    rotations = Slerp(keys, Rotation.from_matrix(poses[:, :3, :3]))(times).as_matrix()
    centers = np.stack([np.interp(times, keys, poses[:, axis, 3]) for axis in range(3)], axis=-1)
    result = np.tile(np.eye(4), (n, 1, 1))
    result[:, :3, :3] = rotations
    result[:, :3, 3] = centers
    return result.astype(np.float32)
