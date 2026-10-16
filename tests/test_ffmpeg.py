import json
import shutil
import subprocess

import numpy as np
import pytest

from drone_scene.common import PipelineError, Runner, read_json
from drone_scene.extract import discover, extract
from drone_scene.render_backend import write_video

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                reason="FFmpeg/ffprobe are not installed")


def make_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=6:duration=6",
                    "-c:v", "mpeg4", "-q:v", "2", "-threads", "1", str(path)], check=True)


def test_real_multiple_videos_are_sampled_without_filename_collisions(tmp_path):
    first = tmp_path / "inputs/flight one/take.mp4"
    second = tmp_path / "inputs/flight two/take.MP4"
    make_video(first)
    make_video(second)
    videos = discover([str(tmp_path / "inputs")])
    work = tmp_path / "work"
    cfg = {"fps": 2., "max_size": 128, "max_frames": 30, "min_sharpness": 0., "dedup_mae": 0.}
    extract(Runner(work), videos, cfg)
    report = read_json(work / "reports/extraction.json")
    assert len(report["videos"]) == 2
    assert [v["kept"] for v in report["videos"]] == [12, 12]
    retained = [name for names in report["retained"].values() for name in names]
    assert len(retained) == len(set(retained)) == 24
    assert all((work / "raw" / name).exists() for name in retained)
    assert all(v["output_size"] == [128, 96] for v in report["videos"])


def test_frame_budget_aborts_before_decoding_or_silently_truncating(tmp_path):
    video = tmp_path / "input.mp4"
    make_video(video)
    work = tmp_path / "work"
    cfg = {"fps": 2., "max_size": 128, "max_frames": 5, "min_sharpness": 0., "dedup_mae": 0.}
    with pytest.raises(PipelineError, match="No clips have been silently shortened"):
        extract(Runner(work), [video], cfg)
    assert not (work / "raw").exists()
    assert not (work / "raw.partial").exists()


def test_rgb_video_encoder_handles_odd_dimensions_and_requested_fps(tmp_path):
    output = tmp_path / "render/scene.mp4"
    frames = [np.full((15, 17, 3), i / 4, dtype=np.float32) for i in range(4)]
    assert write_video(iter(frames), output, 17, 15, 24, save_frames=True) == 4
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
        "-of", "json", str(output),
    ], text=True))["streams"][0]
    assert info == {"width": 18, "height": 16, "r_frame_rate": "24/1", "nb_frames": "4"}
    assert len(list((output.parent / "frames").glob("*.png"))) == 4


def test_empty_renderer_does_not_leave_a_success_artifact(tmp_path):
    output = tmp_path / "empty.mp4"
    with pytest.raises(PipelineError):
        write_video(iter([]), output, 16, 16, 30)
    assert not output.exists()
