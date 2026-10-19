"""Offline RGB renderer for the pinned native representations (no checkpoint conversion)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .backends import checkpoint, gpu_check, mlp_requested, upstream
from .common import PipelineError, read_json, write_json
from .mlp_dataset import selected_image_names
from .trajectory import interpolate, select_clip


def write_video(frames, output, width, height, fps, save_frames=False):
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    # Pipe exact RGB dimensions; pad a single odd pixel for H.264 rather than rescale calibration.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    try:
        for count, rgb in enumerate(frames, 1):
            array = np.ascontiguousarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
            if array.shape != (height, width, 3):
                raise PipelineError(f"Unexpected rendered frame shape: {array.shape}")
            proc.stdin.write(array.tobytes())
            if save_frames:
                folder = output.parent / "frames"
                folder.mkdir(exist_ok=True)
                Image.fromarray(array).save(folder / f"{count:06d}.png")
            if count % 30 == 0:
                print(f"Rendered {count} frames", flush=True)
        proc.stdin.close()
        if proc.wait() or not count:
            raise PipelineError("FFmpeg encoding failed or the renderer produced no frames")
    except BaseException:
        if not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        output.unlink(missing_ok=True)
        raise
    return count


def gsplat_frames(cfg, work):
    import torch

    repo = upstream("gsplat")
    sys.path.insert(0, str(repo / "examples"))
    from datasets.colmap import Parser
    from datasets.traj import generate_ellipse_path_z, generate_spiral_path
    from gsplat import rasterization

    parser = Parser(str(work / "scene"), factor=1, normalize=True,
                    test_every=cfg["training"]["test_every"])
    r = cfg["render"]
    clip, indices = select_clip(parser.image_names, r.get("clip"))
    captured = parser.camtoworlds[indices]
    if r["trajectory"] == "interp":
        poses = interpolate(captured, r["interpolation_steps"], r["max_frames"])
    else:
        if len(captured) < 4:
            raise PipelineError("Ellipse/spiral needs at least four poses")
        if r["trajectory"] == "ellipse":
            poses = generate_ellipse_path_z(captured, height=captured[:, 2, 3].mean())
        else:
            poses = generate_spiral_path(captured, bounds=parser.bounds * parser.scene_scale * 1.1,
                                         spiral_scale_r=parser.extconf["spiral_radius_scale"])
        poses = poses[np.linspace(0, len(poses) - 1, min(len(poses), r["max_frames"]), dtype=int)]
        if poses.shape[1] == 3:
            poses = np.concatenate([poses, np.tile([[[0, 0, 0, 1]]], (len(poses), 1, 1))], axis=1)
    camera_id = parser.camera_ids[indices[0]]
    K = torch.as_tensor(parser.Ks_dict[camera_id], device="cuda", dtype=torch.float32)[None]
    width, height = parser.imsize_dict[camera_id]
    state = torch.load(checkpoint(work, cfg), map_location="cuda", weights_only=True)["splats"]
    colors = torch.cat([state["sh0"], state["shN"]], dim=1)

    def generate():
        with torch.inference_mode():
            for pose in poses:
                c2w = torch.as_tensor(pose, dtype=torch.float32, device="cuda")
                rgb, _, _ = rasterization(
                    means=state["means"], quats=state["quats"], scales=state["scales"].exp(),
                    opacities=state["opacities"].sigmoid(), colors=colors,
                    viewmats=torch.linalg.inv(c2w)[None], Ks=K, width=width, height=height,
                    sh_degree=cfg["training"]["gsplat"]["sh_degree"], packed=True,
                    render_mode="RGB", near_plane=0.01, far_plane=1e10,
                )
                yield rgb[0].clamp(0, 1).cpu().numpy()
    return generate(), width, height, {"clip": clip, "trajectory": r["trajectory"]}


def mlp_frames(cfg, work):
    import torch

    # Also dispatch correctly when this renderer is invoked directly for a manual resume.
    os.environ.update(gpu_check("mlp", mlp_kernel_backend=mlp_requested(cfg)))
    repo = upstream("mlp")
    sys.path.insert(0, str(repo))
    from scannet.camera import validate_checkpoint_calibration
    from scannet.eval_rgb_psnr import (
        _build_projection_from_focal, _c2w_to_view_matrix, _call_scene_rgb, _load_scene_tile_cuda,
    )
    from scannet.safe_io import load_checkpoint
    from scannet.scannet_loader import SceneDataset

    if cfg["render"]["trajectory"] != "interp":
        raise PipelineError("MLP render currently supports trajectory=interp; use gsplat for ellipse/spiral.")
    os.environ["MLPSP_TILE_ORDER"] = "tilecenter"
    os.environ["MLPSP_TILE_CANDIDATE_CAPACITY"] = "1536"
    dataset = SceneDataset(str(work / "scene"), factor=1, llffhold=cfg["training"]["test_every"],
                           split="train", feature_type="none", ray_cache_size=1,
                           split_mode="llff", split_test_offset=0, camera_mode="colmap-native")
    ckpt = checkpoint(work, cfg)
    validate_checkpoint_calibration(load_checkpoint(str(ckpt)), dataset)
    scene = _load_scene_tile_cuda(str(ckpt), torch.device("cuda:0"),
                                  cfg["training"]["mlp"]["points_per_leaf"], tile_size=8)
    clip, indices = select_clip(selected_image_names(dataset), cfg["render"].get("clip"))
    poses = interpolate(dataset.poses[indices].numpy(), cfg["render"]["interpolation_steps"],
                        cfg["render"]["max_frames"])
    height, width = dataset.images.shape[-2:]
    # Keep normalized poses/ray calibration from the SAME loader as training.
    scene.set_image_size(width, height)
    projection = _build_projection_from_focal(dataset.focal, width, height, device="cuda")

    def generate():
        with torch.inference_mode():
            for pose in poses:
                pose_cpu = torch.from_numpy(pose[:3, :4])
                origins, directions = dataset._get_rays((height, width), pose_cpu)
                rgb = _call_scene_rgb(scene, origins.reshape(-1, 3).cuda(),
                                      directions.reshape(-1, 3).cuda(), score_threshold=0.003,
                                      view_matrix=_c2w_to_view_matrix(pose_cpu, device="cuda"),
                                      proj_matrix=projection)
                yield rgb.reshape(height, width, 3).clamp(0, 1).cpu().numpy()
    return generate(), width, height, {"clip": clip, "trajectory": "interp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    factory = gsplat_frames if cfg["backend"] == "gsplat" else mlp_frames
    frames, width, height, metadata = factory(cfg, args.workspace)
    count = write_video(frames, args.output, width, height, cfg["render"]["fps"],
                         cfg["render"]["save_frames"])
    write_json(args.output.with_suffix(".json"), {"frames": count, "width": width, "height": height,
                                                "fps": cfg["render"]["fps"], **metadata})


if __name__ == "__main__":
    main()
