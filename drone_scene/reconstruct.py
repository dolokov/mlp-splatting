from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image

from .colmap_model import (
    assess_components, count_points, flattened, hybrid_pairs, read_cameras, read_images,
    shared_calibration, write_images,
)
from .common import PipelineError, checked_output, read_json, write_json


def gpu_flag(command, old, new, runner):
    if runner.dry_run:
        return old
    help_text = checked_output(["colmap", command, "-h"], include_stderr=True)
    if new in help_text:
        return new
    if old in help_text:
        return old
    raise PipelineError(
        f"COLMAP {command} help exposes neither {old} nor {new}. "
        f"Check `which colmap` and `colmap {command} -h` in the active environment."
    )


def sfm(runner, cfg):
    raw = runner.work / "raw"
    db = runner.work / "database.db"
    sparse = runner.work / "sparse"
    if not runner.dry_run:
        report = read_json(runner.work / "reports" / "extraction.json")
        sizes = {tuple(v["output_size"]) for v in report["videos"]}
        if cfg["camera_mode"] == "shared" and len(sizes) != 1:
            raise PipelineError("Shared calibration requires equal frame dimensions. Use per-video + gsplat.")
        # Restart mapping on the same extracted features/matches after interruption.
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir(parents=True)
    extraction_gpu = gpu_flag("feature_extractor", "--SiftExtraction.use_gpu",
                              "--FeatureExtraction.use_gpu", runner)
    camera_flag = ("--ImageReader.single_camera" if cfg["camera_mode"] == "shared"
                   else "--ImageReader.single_camera_per_folder")
    runner.run(["colmap", "feature_extractor", "--database_path", db, "--image_path", raw,
                "--ImageReader.camera_model", "OPENCV", camera_flag, "1",
                extraction_gpu, str(int(cfg["use_gpu"]))])
    matcher = cfg["matcher"]
    cmd = {"hybrid": "matches_importer", "exhaustive": "exhaustive_matcher",
           "vocab-tree": "vocab_tree_matcher"}[matcher]
    matching_gpu = gpu_flag(cmd, "--SiftMatching.use_gpu", "--FeatureMatching.use_gpu", runner)
    command = ["colmap", cmd, "--database_path", db, matching_gpu, str(int(cfg["use_gpu"]))]
    if matcher == "hybrid":
        pairs_path = runner.work / "pairs.txt"
        if not runner.dry_run:
            pairs = hybrid_pairs(report["retained"], cfg["temporal_window"], cfg["global_stride"])
            if len(pairs) > cfg["max_pairs"]:
                raise PipelineError(f"{len(pairs)} candidate pairs exceed sfm.max_pairs; increase stride.")
            pairs_path.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
            write_json(runner.work / "reports" / "matching.json", {"pairs": len(pairs), **cfg})
        command += ["--match_type", "pairs", "--match_list_path", pairs_path]
    elif matcher == "exhaustive":
        if not runner.dry_run:
            n = sum(len(v) for v in report["retained"].values())
            if n * (n - 1) // 2 > cfg["max_pairs"]:
                raise PipelineError("Exhaustive matching exceeds sfm.max_pairs; use hybrid or a smaller pilot.")
    else:
        vocab = Path(cfg.get("vocab_tree", "")).expanduser()
        if not runner.dry_run and not vocab.is_file():
            raise PipelineError("Set sfm.vocab_tree to an existing COLMAP vocabulary-tree file.")
        command += ["--VocabTreeMatching.vocab_tree_path", vocab]
    runner.run(command)
    runner.run(["colmap", "mapper", "--database_path", db, "--image_path", raw,
                "--output_path", sparse])
    if runner.dry_run:
        print("Validate all components and per-video registration before choosing a model.")
        return
    models = [p for p in sparse.iterdir() if p.is_dir() and (p / "images.bin").is_file()]
    best, summary = assess_components(models, report["retained"],
                                       cfg["min_registered_fraction"], cfg["min_per_video_fraction"])
    write_json(runner.work / "reports" / "reconstruction.json", summary)
    if summary["failures"]:
        raise PipelineError(
            "Reconstruction coverage failed: " + "; ".join(summary["failures"])
            + ". See reports/reconstruction.json. Try lower global_stride or exhaustive matching, "
              "or add a bridging flight with shared landmarks. Independent models cannot be concatenated."
        )
    if count_points(Path(best["model"]) / "points3D.bin") < 100:
        raise PipelineError("Fewer than 100 triangulated points; fix capture/matching before training.")
    write_json(runner.work / "selected-model.json", best)


def prepare(runner, cfg):
    dataset = runner.work / "scene"
    if runner.dry_run:
        runner.run(["colmap", "image_undistorter", "--image_path", runner.work / "raw",
                    "--input_path", "<selected-connected-model>", "--output_path",
                    runner.work / "undistorted", "--output_type", "COLMAP",
                    "--max_image_size", cfg["extraction"]["max_size"]])
        print("Flatten unique image names, preserve camera IDs/observations, validate calibration.")
        return
    selected = Path(read_json(runner.work / "selected-model.json")["model"])
    undistorted = runner.work / "undistorted"
    if undistorted.exists():
        shutil.rmtree(undistorted)
    runner.run(["colmap", "image_undistorter", "--image_path", runner.work / "raw",
                "--input_path", selected, "--output_path", undistorted, "--output_type", "COLMAP",
                "--max_image_size", cfg["extraction"]["max_size"]])
    source_model = undistorted / "sparse"
    if not (source_model / "images.bin").exists():
        source_model = source_model / "0"
    images = read_images(source_model / "images.bin")
    cameras = read_cameras(source_model / "cameras.bin")
    is_shared = shared_calibration(images, cameras)
    tmp = runner.work / "scene.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images").mkdir(parents=True)
    (tmp / "sparse" / "0").mkdir(parents=True)
    flat_images = flattened(images)
    for original, flat in zip(images, flat_images):
        source = undistorted / "images" / original.name
        target = tmp / "images" / flat.name
        if not source.is_file():
            raise PipelineError(f"Undistorted image is missing: {source}")
        with Image.open(source) as image:
            cam = cameras[original.camera_id]
            if image.size != (cam["width"], cam["height"]):
                raise PipelineError(f"Camera/image dimensions disagree for {original.name}")
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    for name in ("cameras.bin", "points3D.bin"):
        shutil.copy2(source_model / name, tmp / "sparse" / "0" / name)
    write_images(tmp / "sparse" / "0" / "images.bin", flat_images)
    if dataset.exists():
        shutil.rmtree(dataset)
    tmp.replace(dataset)
    summary = {"images": len(images), "points": count_points(source_model / "points3D.bin"),
               "shared_calibration": is_shared, "cameras": cameras,
               "name_map": {a.name: b.name for a, b in zip(images, flat_images)}}
    write_json(runner.work / "reports" / "dataset.json", summary)
    if len(images) < 16:
        raise PipelineError("Fewer than 16 registered views; use more overlapping frames for a scene.")
