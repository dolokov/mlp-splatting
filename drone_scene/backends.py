from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from .colmap_model import read_cameras, read_images, shared_calibration
from .common import PipelineError, checked_output, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def upstream(backend, *, verify=True):
    lock = read_json(ROOT / "upstreams.lock.json")[backend]
    directory = ROOT / "third_party" / backend
    if verify:
        if not (directory / ".git").exists():
            raise PipelineError(f"Install backend first: bash scripts/install_backend.sh {backend}")
        actual = checked_output(["git", "-C", str(directory), "rev-parse", "HEAD"]).strip()
        if actual != lock["commit"]:
            raise PipelineError(f"{backend} source differs from upstreams.lock.json; reinstall the backend.")
        if checked_output(["git", "-C", str(directory), "status", "--porcelain",
                           "--untracked-files=no"]).strip():
            raise PipelineError(f"{backend} source has local edits; use a clean pinned checkout.")
        modules = checked_output(["git", "-C", str(directory), "submodule", "status", "--recursive"])
        if any(line.startswith(("-", "+", "U")) for line in modules.splitlines()):
            raise PipelineError(f"{backend} submodules are missing or differ from the pinned checkout; reinstall.")
    return directory


def gpu_check(backend):
    try:
        import torch
    except ImportError:
        raise PipelineError("PyTorch is missing. Activate the conda environment and install a backend.") from None
    if not torch.cuda.is_available():
        raise PipelineError("PyTorch cannot see an NVIDIA GPU. Check the driver and active environment.")
    cc = torch.cuda.get_device_capability()
    floor = (8, 0) if backend == "mlp" else (7, 5)
    if cc < floor:
        raise PipelineError(
            f"{torch.cuda.get_device_name()} is sm_{cc[0]}{cc[1]}; this {backend} route requires "
            f"sm_{floor[0]}{floor[1]}+. TITAN RTX users: select configs/titan.json (gsplat). "
            "For native MLP training use an Ampere-or-newer GPU. A newer driver does not change compute capability."
        )
    required_torch = (2, 10) if backend == "mlp" else (2, 7)
    torch_version = tuple(int(n) for n in torch.__version__.split("+")[0].split(".")[:2])
    if torch_version != required_torch:
        raise PipelineError(f"Use the pinned {backend} environment: expected torch {required_torch}, got {torch.__version__}.")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise PipelineError("nvcc is missing. Install the conda CUDA toolkit, not only a CUDA runtime wheel.")
    match = re.search(r"release (\d+\.\d+)", checked_output([nvcc, "--version"]))
    if not match or match[1] != torch.version.cuda:
        raise PipelineError(f"nvcc/PyTorch CUDA mismatch: nvcc={match[1] if match else 'unknown'}, torch={torch.version.cuda}.")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name()} | sm_{cc[0]}{cc[1]} | "
          f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free | torch {torch.__version__} | CUDA {torch.version.cuda}")
    return {"TORCH_CUDA_ARCH_LIST": f"{cc[0]}.{cc[1]}"}


def doctor(backend, preprocessing_only=False):
    for command in ("ffmpeg", "ffprobe", "colmap", "git"):
        if not shutil.which(command):
            raise PipelineError(f"{command} missing. Activate the environment created from environment.yml.")
    checked_output(["ffmpeg", "-version"])
    colmap_version = checked_output(["colmap", "-h"]).splitlines()
    print("COLMAP:", next((line for line in colmap_version if "COLMAP" in line), "found"))
    if preprocessing_only:
        print("Preprocessing tools found; training hardware check skipped.")
        return
    gpu_check(backend)
    upstream(backend)
    if backend == "gsplat":
        try:
            import gsplat
        except ImportError:
            raise PipelineError("gsplat is missing; run scripts/install_backend.sh gsplat") from None
        print(f"gsplat: {gsplat.__version__}")
    print("Checks passed. CUDA extensions still need to compile on first use.")


def model_dir(work, backend):
    return work / "model" / backend


def checkpoint(work, cfg):
    folder = model_dir(work, cfg["backend"])
    steps = cfg["training"]["iterations"]
    if cfg["backend"] == "gsplat":
        return folder / "ckpts" / f"ckpt_{steps - 1}_rank0.pt"
    return folder / "scene_ckpt.pt"


def gsplat_command(cfg, work, action="train", verify=True):
    repo = upstream("gsplat", verify=verify)
    t = cfg["training"]
    g = t["gsplat"]
    steps = t["iterations"]
    command = [sys.executable, str(repo / "examples" / "simple_trainer.py"), "mcmc",
               "--data-dir", str(work / "scene"), "--data-factor", "1",
               "--result-dir", str(model_dir(work, "gsplat")), "--test-every", str(t["test_every"]),
               "--max-steps", str(steps), "--steps-scaler", "1.0",
               "--eval-steps", str(steps), "--save-steps", str(steps), "--ply-steps", str(steps),
               "--sh-degree", str(g["sh_degree"]), "--strategy.cap-max", str(g["cap_max"]),
               "--strategy.refine-stop-iter", str(max(1, int(steps * 0.83))),
               "--packed", "--disable-video"]
    if action == "train":
        command += ["--disable-viewer", "--save-ply"]
    elif action == "view":
        command += ["--ckpt", str(checkpoint(work, cfg)), "--port", str(cfg["render"]["port"])]
    return command, repo


def mlp_command(cfg, work, verify=True):
    repo = upstream("mlp", verify=verify)
    t = cfg["training"]
    m = t["mlp"]
    command = [sys.executable, str(repo / "train.py"), "--data", str(work / "scene"),
               "--output", str(model_dir(work, "mlp")), "--dataset", "scannet", "--factor", "1",
               "--iterations", str(t["iterations"]), "--camera-mode", "colmap-native",
               "--feature-type", "none", "--split-mode", "llff", "--split-test-offset", "0",
               "--llffhold", str(t["test_every"]), "--max-points-per-leaf", str(m["points_per_leaf"]),
               "--tile-size", "8", "--tile-order", "tilecenter", "--training-backend", "legacy",
               "--ray-cache-size", "2", "--workers", "0", "--no-persistent-workers",
               "--no-tile-mosaic", "--seed", "2025"]
    return command, repo


def train(runner, cfg):
    backend = cfg["backend"]
    if not runner.dry_run:
        env = gpu_check(backend)
        images = read_images(runner.work / "scene" / "sparse" / "0" / "images.bin")
        cameras = read_cameras(runner.work / "scene" / "sparse" / "0" / "cameras.bin")
        if backend == "mlp":
            if not shared_calibration(images, cameras):
                raise PipelineError("Native MLP requires shared calibration for every image. Use shared + fixed optics, or gsplat.")
            bytes_images = sum(cameras[im.camera_id]["width"] * cameras[im.camera_id]["height"] * 12
                               for im in images)
            # Upstream stacks all RGBs and copies the selected split: budget 2.5x, before workers/caches.
            print(f"MLP eager image storage estimate: {bytes_images / 2**30:.1f} GiB; peak budget >= {2.5 * bytes_images / 2**30:.1f} GiB RAM.")
            if hasattr(os, "sysconf"):
                available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                if bytes_images * 2.5 > available * 0.8:
                    raise PipelineError("Insufficient available host RAM for the MLP eager loader. Reduce frame count/resolution.")
        else:
            from .colmap_model import count_points
            points = count_points(runner.work / "scene" / "sparse" / "0" / "points3D.bin")
            if points > cfg["training"]["gsplat"]["cap_max"]:
                raise PipelineError("SfM initialization exceeds cap_max. Increase cap_max or reduce pilot frames; MCMC caps growth, not initialization.")
    else:
        env = {}
    command, repo = (gsplat_command(cfg, runner.work, verify=not runner.dry_run) if backend == "gsplat"
                     else mlp_command(cfg, runner.work, verify=not runner.dry_run))
    env.update({"MAX_JOBS": "2", "TORCH_EXTENSIONS_DIR": str(runner.work / "torch-extensions"),
                "CUDA_VISIBLE_DEVICES": "0"})
    runner.run(command, cwd=repo, env=env)


def render(runner, cfg, output):
    if not runner.dry_run:
        gpu_check(cfg["backend"])
        upstream(cfg["backend"])
    config_path = output.parent / "config.json"
    if not runner.dry_run:
        write_json(config_path, cfg)
    runner.run([sys.executable, "-m", "drone_scene.render_backend", "--config", config_path,
                "--workspace", runner.work, "--output", output], cwd=ROOT,
               env={"MAX_JOBS": "2", "TORCH_EXTENSIONS_DIR": str(runner.work / "torch-extensions"),
                    "CUDA_VISIBLE_DEVICES": "0"})


def view(runner, cfg):
    if cfg["backend"] != "gsplat":
        raise PipelineError("Native MLP has no packaged free-fly viewer here. Use render; see docs/rendering.md for custom viewer options.")
    if not checkpoint(runner.work, cfg).is_file() and not runner.dry_run:
        raise PipelineError("No final checkpoint; finish training first.")
    if not runner.dry_run:
        gpu_check("gsplat")
    command, repo = gsplat_command(cfg, runner.work, "view", verify=not runner.dry_run)
    runner.run(command, cwd=repo, env={"MAX_JOBS": "2", "TORCH_EXTENSIONS_DIR": str(runner.work / "torch-extensions"),
                                       "CUDA_VISIBLE_DEVICES": "0"})
