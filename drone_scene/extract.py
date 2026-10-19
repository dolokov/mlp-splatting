from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import PipelineError, checked_output, write_json


def discover(inputs):
    videos = []
    for text in inputs:
        path = Path(text).expanduser().resolve()
        if path.is_dir():
            videos.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")
        elif path.is_file() and path.suffix.lower() == ".mp4":
            videos.append(path)
        else:
            raise PipelineError(f"Expected an MP4 file or directory: {path}")
    videos = sorted(set(videos))
    if not videos:
        raise PipelineError("No MP4 videos found")
    if any(any(ch.isspace() for ch in p.name) for p in videos):
        # Source paths may contain spaces; generated COLMAP paths never do.
        print("Source video paths contain spaces; subprocess arguments handle them safely.")
    return videos


def clip_id(path, index):
    short = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    return f"v{index:03d}_{short}"


def source_identity(videos):
    return [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in videos]


def probe(path):
    payload = json.loads(checked_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,duration:format=duration", "-of", "json", str(path),
    ]))
    if not payload.get("streams"):
        raise PipelineError(f"No video stream in {path}")
    stream = payload["streams"][0]
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration)
    except (ValueError, TypeError):
        raise PipelineError(f"Cannot determine duration for {path}") from None
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError(f"Invalid duration for {path}")
    return {"width": stream["width"], "height": stream["height"], "duration": duration}


def image_quality(path):
    with Image.open(path) as image:
        size = image.size
        gray = np.asarray(ImageOps.grayscale(image).resize((256, 144)), dtype=np.float32)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return float(lap.var()), gray, size


def select_frames(sampled, cfg, runners_up=None):
    """Select chronologically, ranking within windows; only buffer one window's thumbnails."""
    window = cfg.get("selection_window_seconds", 0)
    dropped = {"blur": 0, "duplicate": 0}
    if window:
        dropped["window"] = 0
    selected, dimensions = [], set()
    previous = None

    def choose(candidates):
        nonlocal previous
        winner = None
        for image, score, thumbnail, size, timestamp in sorted(
                candidates, key=lambda item: (-item[1], item[4])):
            if winner is not None:
                dropped["window"] += 1
                reason = "window"
            elif (previous is not None
                  and float(np.abs(thumbnail - previous).mean()) < cfg["dedup_mae"]):
                dropped["duplicate"] += 1
                reason = "duplicate"
            else:
                winner = image
                previous = thumbnail
                dimensions.add(size)
                selected.append({"name": image.name, "timestamp_seconds": timestamp,
                                 "sharpness": score})
                continue
            if runners_up is not None and len(runners_up) < 200:
                runners_up.append({"name": image.name, "timestamp_seconds": timestamp,
                                   "sharpness": score, "reason": reason})
            image.unlink()

    bucket, pending = None, []
    for index, image in enumerate(sampled):
        timestamp = index / cfg["fps"]
        current = math.floor(timestamp / window) if window else index
        if bucket is not None and current != bucket:
            choose(pending)
            pending = []
        bucket = current
        score, thumbnail, size = image_quality(image)
        if score < cfg["min_sharpness"]:
            dropped["blur"] += 1
            image.unlink()
        else:
            pending.append((image, score, thumbnail, size, timestamp))
    choose(pending)
    return selected, dropped, dimensions


def extract(runner, videos, cfg):
    raw = runner.work / "raw"
    if runner.dry_run:
        print(f"Sample {len(videos)} clips at {cfg['fps']} FPS; cap {cfg['max_frames']} retained frames; "
              f"selection window {cfg.get('selection_window_seconds', 0)} seconds; "
              f"candidate cap {cfg.get('max_candidates', cfg['max_frames'])}.")
        return
    metadata = [probe(p) for p in videos]
    # Guard against accidentally decoding hours at full rate; reject rather than truncate clips.
    estimated = sum(math.ceil(m["duration"] * cfg["fps"]) for m in metadata)
    candidate_limit = cfg.get("max_candidates", cfg["max_frames"])
    if estimated > candidate_limit:
        raise PipelineError(
            f"Sampling would create about {estimated} frames (candidate limit {candidate_limit}). "
            "Lower extraction.fps, trim a pilot, or explicitly raise extraction.max_candidates "
            "(defaults to extraction.max_frames). "
            "No clips have been silently shortened."
        )
    tmp = runner.work / "raw.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    report = {"videos": [], "retained": {}}
    total_kept = 0
    for i, (video, meta) in enumerate(zip(videos, metadata)):
        clip = clip_id(video, i)
        folder = tmp / clip
        folder.mkdir()
        max_size = int(cfg["max_size"])
        vf = (f"fps={cfg['fps']},scale=w='min({max_size},iw)':"
              f"h='min({max_size},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2")
        runner.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-i", video,
                    "-map", "0:v:0", "-vf", vf, "-q:v", "2", "-threads", "2",
                    "-fps_mode", "vfr", folder / "%08d.jpg"])
        sampled = sorted(folder.glob("*.jpg"))
        if sum(v["sampled"] for v in report["videos"]) + len(sampled) > candidate_limit:
            raise PipelineError("Actual decoded candidate count exceeds extraction.max_candidates")
        runners_up = []
        selected, dropped, dimensions = select_frames(sampled, cfg, runners_up)
        kept = [f"{clip}/{frame['name']}" for frame in selected]
        total_kept += len(kept)
        if total_kept > cfg["max_frames"]:
            raise PipelineError("Selected frames exceed extraction.max_frames; increase the selection "
                                "window, lower fps, or raise the retained-frame limit. No clips have "
                                "been silently shortened.")
        if len(kept) < 8:
            raise PipelineError(f"Only {len(kept)} usable frames in {video}; need >=8. Check quality settings.")
        if len(dimensions) != 1:
            raise PipelineError(f"Frame dimensions change inside {video}; split it into fixed-camera clips.")
        report["retained"][clip] = kept
        report["videos"].append({"source": str(video), "clip": clip, **meta,
                                  "sampled": len(sampled), "kept": len(kept),
                                   "selected": selected,
                                   "runners_up": runners_up,
                                   "selection_window_seconds": cfg.get("selection_window_seconds", 0),
                                   "max_gap_seconds": max((b["timestamp_seconds"] - a["timestamp_seconds"]
                                                           for a, b in zip(selected, selected[1:])),
                                                          default=0),
                                  "dropped": dropped, "output_size": list(dimensions.pop())})
    if raw.exists():
        shutil.rmtree(raw)
    tmp.replace(raw)
    write_json(runner.work / "reports" / "extraction.json", report)
