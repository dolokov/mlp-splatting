from pathlib import Path

import numpy as np
import pytest

from drone_scene import extract as extraction
from drone_scene.common import PipelineError, read_json, write_json
from drone_scene.config import load_config


def select(tmp_path, monkeypatch, scores, window=1, values=None, threshold=0, dedup=0):
    images = [tmp_path / f"{i + 1:08d}.jpg" for i in range(len(scores))]
    for image in images:
        image.touch()
    values = values if values is not None else list(range(len(scores)))
    quality = {image: (score, np.full((2, 2), value, dtype=np.float32), (32, 24))
               for image, score, value in zip(images, scores, values)}
    monkeypatch.setattr(extraction, "image_quality", quality.__getitem__)
    result = extraction.select_frames(images, {
        "fps": 4, "selection_window_seconds": window,
        "min_sharpness": threshold, "dedup_mae": dedup,
    })
    return result, [p.name for p in images if p.exists()]


def test_sharpest_per_window_and_partial_tail(tmp_path, monkeypatch):
    (selected, dropped, sizes), remaining = select(
        tmp_path, monkeypatch, [1, 4, 2, 3, 8, 5, 6, 7, 2])
    assert [s["name"] for s in selected] == remaining == [
        "00000002.jpg", "00000005.jpg", "00000009.jpg"]
    assert [s["timestamp_seconds"] for s in selected] == [0.25, 1, 2]
    assert dropped == {"blur": 0, "duplicate": 0, "window": 6}
    assert sizes == {(32, 24)}


def test_ties_choose_earliest(tmp_path, monkeypatch):
    (selected, _, _), _ = select(tmp_path, monkeypatch, [3, 3, 1])
    assert selected[0]["name"] == "00000001.jpg"


def test_duplicate_winner_uses_runner_up(tmp_path, monkeypatch):
    (selected, dropped, _), remaining = select(
        tmp_path, monkeypatch, [9, 1, 1, 1, 10, 8],
        values=[0, 0, 0, 0, 0, 10], dedup=1)
    assert [s["name"] for s in selected] == remaining == ["00000001.jpg", "00000006.jpg"]
    assert dropped["duplicate"] == 1


def test_all_blurry_windows_leave_no_files(tmp_path, monkeypatch):
    (selected, dropped, sizes), remaining = select(
        tmp_path, monkeypatch, [1] * 9, threshold=2)
    assert selected == remaining == []
    assert sizes == set()
    assert dropped["blur"] == 9


def test_disabled_windows_preserve_fixed_sampling(tmp_path, monkeypatch):
    (selected, dropped, _), remaining = select(
        tmp_path, monkeypatch, [1, 9, 2], window=0)
    assert len(selected) == len(remaining) == 3
    assert dropped == {"blur": 0, "duplicate": 0}


@pytest.mark.parametrize("key,value", [
    ("selection_window_seconds", -1), ("selection_window_seconds", True),
    ("selection_window_seconds", float("nan")), ("selection_window_seconds", 0.01),
    ("max_candidates", 0), ("max_candidates", 1.5), ("max_candidates", True),
])
def test_invalid_selection_config(tmp_path, key, value):
    cfg = read_json(Path(__file__).resolve().parents[1] / "configs/pilot.json")
    cfg["extraction"][key] = value
    path = tmp_path / "config.json"
    write_json(path, cfg)
    with pytest.raises(PipelineError):
        load_config(path)