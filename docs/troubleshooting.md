# Troubleshooting

| Symptom | First action |
| --- | --- |
| MLP native override rejects TITAN RTX | Use `configs/mlp-titan.json` or `mlp-pilot.json` with `kernel_backend: auto`. `native` requires SM80+; the same MLP representation uses scalar FP32 CUDA on TITAN. |
| BF16/WMMA compile failure on TITAN | Rerun `bash scripts/install_backend.sh mlp`, verify the compatibility patch, and run the smoke test. A TITAN build should contain no BF16/WMMA kernels. Do not force the native build on SM75. |
| MLP source has unrecognized edits or a partial patch | Preserve your edited checkout and reinstall from the pinned source. The installer accepts only pristine or exactly patched source and does not discard unknown edits. |
| `nvcc` missing / CUDA mismatch | Activate the right conda environment, `export CUDA_HOME="$CONDA_PREFIX"`, check `which nvcc`, `nvcc --version`, and `python -c 'import torch; print(torch.version.cuda)'`. Use the matching environment file. |
| Conda cannot solve the environment | Use Miniforge/conda-forge on Linux x86-64, try `conda env create --solver libmamba -f environment.yml`, and inspect package availability. Do not silently substitute CUDA 13 for the pinned 12.x extension build. |
| CPU SIFT is slow | Enable `sfm.use_gpu` only with a CUDA-capable COLMAP build; otherwise use fewer redundant frames for a pilot. Many distro/conda builds are CPU-only. |
| Several components / one video absent | Inspect `reports/reconstruction.json`, reduce global keyframe stride, try exhaustive matching for a small pilot, or capture a bridging clip. Do not quietly accept only one flight. |
| Many frames rejected | Inspect extraction counts. Lower `min_sharpness` / `dedup_mae`, improve shutter speed, or capture more textured oblique views. |
| Too many candidate frames/pairs | Use a pilot subset, lower sampling FPS while retaining overlap, increase global stride, or explicitly raise the guard after estimating cost. No input clip is silently truncated. |
| Images don't register despite overlap | Check fixed optics, blur, exposure differences, repeated textures, rolling shutter, and shared camera assumptions. Try per-video calibration + gsplat when justified. |
| CUDA OOM | Reduce max image size, cap_max, and redundant views; close GPU-heavy applications. A Gaussian count cap is not a byte-level memory limit. |
| MLP host RAM guard fails | Reduce image size and count. Upstream stacks all float RGBs; lazy frame decoding requires loader changes beyond this wrapper. |
| MLP tile capacity overflow | The current upstream checks capacity and errors rather than truncating. Increase points_per_leaf to reduce primitive count, reduce scene complexity, or investigate partitioning. Raising capacity can increase memory and is restricted by the upstream supported capacities. |
| Scene looks disconnected/ghosted | Verify camera geometry/registration before longer training. Dynamic foliage, water, changing illumination, and camera calibration errors are not fixed by more iterations. |
| Browser cannot open native MLP checkpoint | Use native offline RGB rendering; ordinary Gaussian viewers need Gaussian PLY. See `rendering.md`. |
| Ellipse/spiral fails or flies underground | Use `interp` on one clip first. Synthetic orbit paths can be unsuitable for planar or partial captures. |
| Viewer is waiting forever | That is expected: `view` keeps serving until Ctrl+C. `run` trains headlessly and exits after rendering. |
| Changed settings rejected by cache | Use a fresh workspace, or restore the exact original preprocessing/training settings. Render-only changes are supported in-place. |

## Interrupted training

Completed pipeline stages are cached by source/config signatures. CUDA training resume is a separate backend feature.

The pinned gsplat example saves model parameters for evaluation/preview; its `--ckpt` branch does not restore optimizer/densification state for training. Rerunning an interrupted pipeline restarts Gaussian training from SfM. If partial model outputs matter to you, move `work/site/model/gsplat/` to an archive before rerunning. Completed training is skipped.

Native MLP saves `train_state_latest.pt` with optimizer state every 500 steps and at completion. Advanced native resume, using the same dataset and settings:

```bash
# In drone-scene-mlp; copy the rest of the exact train command from logs/commands.jsonl:
python third_party/mlp/train.py \
  --data /absolute/path/to/work/mlp-site/scene \
  --output /absolute/path/to/work/mlp-site/model/mlp \
  --dataset scannet --factor 1 --iterations 30000 \
  --camera-mode colmap-native --feature-type none \
  --split-mode llff --split-test-offset 0 --llffhold 8 \
  --max-points-per-leaf 32 --tile-size 8 --tile-order tilecenter \
  --training-backend legacy --ray-cache-size 2 --workers 0 \
  --no-persistent-workers --no-tile-mosaic --seed 2025 \
  --resume /absolute/path/to/work/mlp-site/model/mlp/train_state_latest.pt
```

`scene_ckpt.pt` is an inference export; do not use it to resume optimization. This manual command does not mark the orchestration stage complete. After it succeeds, use the standalone renderer with a config snapshot instead of pretending the pipeline cached it:

```bash
python -m drone_scene.render_backend --config configs/mlp-ampere.json \
  --workspace /absolute/path/to/work/mlp-site \
  --output /absolute/path/to/work/mlp-site/manual-render/scene.mp4
```

Use the original config you actually trained with. Rebuilding/removing a backend checkout requires reinstalling it at the lockfile commit. Do not remove the extension cache while another process is compiling into it.
