from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import backends
from .common import PipelineError, Runner, digest, read_json, stage, write_json
from .config import load_config
from .extract import discover, extract, source_identity
from .reconstruct import prepare, sfm

STAGES = ("extract", "sfm", "prepare", "train", "render")


def workspace(path, dry_run):
    path = path.expanduser().resolve()
    marker = path / ".drone-scene.json"
    if not dry_run:
        if path.exists() and not marker.exists() and any(path.iterdir()):
            raise PipelineError("Workspace is not empty and has no pipeline marker. Choose a new directory.")
        path.mkdir(parents=True, exist_ok=True)
        if marker.exists() and read_json(marker).get("pipeline") != "drone-scene-v1":
            raise PipelineError("Unrecognized workspace marker")
        write_json(marker, {"pipeline": "drone-scene-v1"})
    return path


def require_training(work, cfg):
    stamp = work / "state" / f"train-{cfg['backend']}.json"
    if not stamp.exists() or not backends.checkpoint(work, cfg).is_file():
        raise PipelineError("No completed training stage. Run the pipeline through --until train first.")
    saved = read_json(stamp)["signature"]
    if saved["training"] != cfg["training"] or saved["backend"] != cfg["backend"]:
        raise PipelineError("Training settings differ from the completed checkpoint. Use its original config.")


def render_stage(runner, cfg):
    # Different render paths/settings get separate outputs without retraining.
    token = digest({"backend": cfg["backend"], "render": cfg["render"]})[:12]
    output = runner.work / "renders" / token / "scene.mp4"
    stage(runner, f"render-{token}", {"backend": cfg["backend"], "training": cfg["training"],
                                      "render": cfg["render"], "checkpoint": str(backends.checkpoint(runner.work, cfg))},
          [output, output.with_suffix(".json")], lambda: backends.render(runner, cfg, output))
    if not runner.dry_run:
        write_json(runner.work / "latest-render.json", {"video": str(output),
                                                       "metadata": str(output.with_suffix('.json'))})
    print(f"Rendered video: {output}")


def run_pipeline(args, cfg, runner):
    if not args.videos:
        raise PipelineError("run requires --videos /path/to/clips or --videos a.mp4 b.mp4")
    videos = discover(args.videos)
    for source in args.videos:
        source_path = Path(source).expanduser().resolve()
        if source_path.is_dir() and runner.work.is_relative_to(source_path):
            raise PipelineError("Keep --workspace outside the input video directory to avoid ingesting rendered videos.")
    if any(video.is_relative_to(runner.work) for video in videos):
        raise PipelineError("Input videos must be outside the generated workspace.")
    last = STAGES.index(args.until)
    if not runner.dry_run:
        backends.doctor(cfg["backend"], preprocessing_only=last < 3,
                         mlp_kernel_backend=backends.mlp_requested(cfg))
    lock = read_json(backends.ROOT / "upstreams.lock.json")
    extraction_sig = {"inputs": source_identity(videos), "settings": cfg["extraction"]}
    sfm_sig = {"extraction": digest(extraction_sig), "settings": cfg["sfm"]}
    preparation_sig = {"sfm": digest(sfm_sig), "max_size": cfg["extraction"]["max_size"]}
    training_sig = {"scene": digest(preparation_sig), "backend": cfg["backend"],
                    "training": cfg["training"], "upstream": lock[cfg["backend"]]}
    actions = [
        ("extract", extraction_sig, [runner.work / "raw", runner.work / "reports" / "extraction.json"],
         lambda: extract(runner, videos, cfg["extraction"])),
        ("sfm", sfm_sig, [runner.work / "selected-model.json", runner.work / "reports" / "reconstruction.json"],
         lambda: sfm(runner, cfg["sfm"])),
        ("prepare", preparation_sig, [runner.work / "scene" / "sparse" / "0" / "images.bin",
                                      runner.work / "reports" / "dataset.json"],
         lambda: prepare(runner, cfg)),
        (f"train-{cfg['backend']}", training_sig, [backends.checkpoint(runner.work, cfg)],
         lambda: backends.train(runner, cfg)),
    ]
    if cfg["backend"] == "gsplat":
        actions[3][2].append(backends.model_dir(runner.work, "gsplat") / "ply" /
                             f"point_cloud_{cfg['training']['iterations'] - 1}.ply")
    for name, signature, outputs, action in actions[:last + 1]:
        stage(runner, name, signature, outputs, action)
    if last >= 3 and not runner.dry_run:
        write_json(runner.work / "resolved-config.json", cfg)
        print(f"Final checkpoint: {backends.checkpoint(runner.work, cfg)}")
    if last == 4:
        render_stage(runner, cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multiple MP4s → one connected scene → native splats")
    parser.add_argument("command", choices=("doctor", "run", "render", "view", "smoke"))
    parser.add_argument("--config", type=Path, default=backends.ROOT / "configs" / "titan.json")
    parser.add_argument("--workspace", type=Path, default=Path("work/site"))
    parser.add_argument("--videos", nargs="+")
    parser.add_argument("--until", choices=STAGES, default="render")
    parser.add_argument("--dry-run", action="store_true", help="Print a plan without creating files or running tools")
    parser.add_argument("--preprocessing-only", action="store_true", help="doctor: skip training hardware checks")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config.expanduser().resolve())
        if args.command == "doctor":
            backends.doctor(cfg["backend"], args.preprocessing_only,
                             mlp_kernel_backend=backends.mlp_requested(cfg))
            return 0
        work = workspace(args.workspace, args.dry_run)
        runner = Runner(work, dry_run=args.dry_run)
        if args.command == "smoke":
            if cfg["backend"] != "mlp":
                raise PipelineError("smoke requires --config configs/mlp-pilot.json (or another MLP config)")
            if args.dry_run:
                print("Plan: select MLP CUDA build; compile, render, backpropagate, update, save and reload a toy scene.")
            else:
                from .mlp_smoke import run as run_smoke
                run_smoke(cfg, work)
        elif args.command == "run":
            run_pipeline(args, cfg, runner)
        else:
            if not args.dry_run:
                require_training(work, cfg)
            if args.command == "render":
                render_stage(runner, cfg)
            else:
                backends.view(runner, cfg)
        return 0
    except (PipelineError, OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped. Completed stages remain cached; see README for training restart behavior.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
