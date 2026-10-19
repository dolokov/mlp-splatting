"""Camera metadata aligned with the native MLP loader's train/test split."""
from pathlib import Path

from .common import PipelineError


def selected_image_names(dataset):
    """Return names aligned with poses, never the loader's unfiltered image_paths."""
    paths = getattr(dataset, "image_paths_selected", None)
    if paths is None or len(paths) != len(dataset.poses):
        raise PipelineError("MLP selected image paths must match the number of dataset poses")
    return [Path(path).name for path in paths]
