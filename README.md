# Drone scene pipeline

**Turn multiple overlapping MP4 drone videos into one connected reconstruction, train native MLP-Splatting or Gaussian splats, and render an RGB flythrough.** Linux, one NVIDIA GPU, FFmpeg, COLMAP, and conda.

## Start here: your TITAN RTX

Your GPU is **Turing, compute capability 7.5, 24 GB**. The upstream MLP-Splatting README specifies **8.0+ for its supplied configurations**. This project deliberately supports its native MLP route on Ampere or newer; it does **not** claim an untested Turing port. `TORCH_CUDA_ARCH_LIST=7.5` cannot add missing GPU instructions.

For your current workstation, start with the included **gsplat 3D Gaussian route**. It uses a CUDA 12.6 / PyTorch 2.7.1 stack that targets Turing. For native MLPs, use the separate CUDA 12.8 / PyTorch 2.10 environment and an Ampere-or-newer GPU. These are separate representations and environments; the pipeline never silently changes your selected backend.

The `CUDA 13.4` line in `nvidia-smi` describes the driver's supported CUDA level. It does not say which `nvcc` toolkit is installed. Keep the recent driver; install the specified toolkit inside conda. There is no need to downgrade the system driver.

## 1. Install once

Extract this repository and open its root directory. Install [Miniforge](https://github.com/conda-forge/miniforge) if you do not have conda. The environment files target Linux x86-64.

```bash
conda env create -f environment.yml
conda activate drone-scene
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/install_backend.sh gsplat
```

The installer clones exact upstream commits into `third_party/`, installs PyTorch before building CUDA extensions, and checks the active GPU/toolkit. Keep the cloned repositories: the pipeline calls their pinned entrypoints. Compilation can take several minutes. Close GPU-heavy desktop applications before training; your screenshot already has about 1.7 GB allocated to the desktop/apps.

`environment.yml` and the backend pins specify a reproducible starting stack, not a solved package lock for every Linux distribution. The conda solve, CUDA build, and GPU execution must be verified on your workstation. See [validation status](docs/validation.md).

## 2. Run a small pilot end to end

Put **two or more overlapping clips of the same static place** in `/data/drone/pilot/`. About 1–2 minutes per clip is a useful first experiment. Shell paths with spaces work. The pipeline discovers `.mp4` and `.MP4` recursively.

```bash
drone-scene run \
  --config configs/pilot.json \
  --videos /data/drone/pilot \
  --workspace work/pilot
```

That one command samples and filters frames, matches across all clips, reconstructs and checks a connected model, undistorts images, trains for 2,000 iterations, exports Gaussian PLY, and renders an MP4. A pilot checks the capture and setup; 2,000 iterations is not a final-quality training budget.

Useful outputs:

| Output | Pilot path |
| --- | --- |
| Portable Gaussian scene | `work/pilot/model/gsplat/ply/point_cloud_1999.ply` |
| Native trained checkpoint | `work/pilot/model/gsplat/ckpts/ckpt_1999_rank0.pt` |
| Exact rendered MP4 path | `work/pilot/latest-render.json` |
| Registration coverage per input video | `work/pilot/reports/reconstruction.json` |
| Frame filtering and source mapping | `work/pilot/reports/extraction.json` |
| Ready-to-train common dataset | `work/pilot/scene/` |
| Commands and tool output | `work/pilot/logs/` |

Open the PLY in [SuperSplat](https://superspl.at/editor) for browser inspection/editing. This PLY contains Gaussian scale, rotation, opacity, and spherical harmonics; it is not a triangle mesh or an MLP checkpoint.

## 3. Run the larger scene

```bash
drone-scene run \
  --config configs/titan.json \
  --videos /data/drone/full-capture \
  --workspace work/site
```

Defaults: **1 FPS**, longest edge **1600 px**, at most **2,500 candidate frames**, temporal + global-keyframe matching, **30,000 training iterations**, SH degree 2, and a **750,000 Gaussian growth cap**. The final PLY is `work/site/model/gsplat/ply/point_cloud_29999.ply`.

The cap bounds MCMC growth; it is **not a VRAM guarantee**. The initializer also needs to fit, so the pipeline checks that the SfM point count does not already exceed the cap. High-resolution overlapping splats can still exhaust VRAM. Reduce image size first, then the cap/frame count. See [large scenes and alternatives](docs/large-scenes.md).

Pass specific files instead of a directory if convenient:

```bash
drone-scene run --config configs/titan.json --workspace work/site \
  --videos '/data/flight 1.mp4' '/data/flight 2.mp4' /data/flight3.MP4
```

Use a **new workspace** when changing sampling, SfM, resolution, or an already-completed backend's training settings. This prevents mixing cameras from old images with a new run. Keep workspaces outside the input video directory so generated flythroughs cannot become inputs.

## Pipeline and connection checks

1. FFmpeg samples every clip at a controlled rate and size. A conservative frame budget rejects oversized jobs **before** decoding rather than silently dropping a video.
2. A thumbnail Laplacian score removes very blurry frames. Mean thumbnail difference removes near duplicates within each clip. Scores are heuristics; review extraction counts and lower thresholds if useful distant views are rejected.
3. All retained frames enter **one COLMAP database**. The default hybrid matcher includes temporal neighbors and a global lattice of keyframes **across videos and within each flight**. A sequential-only matcher would often leave separate flights disconnected.
4. COLMAP estimates cameras and a sparse point cloud. The largest component must contain at least **85% of all retained frames**, and **60% from every clip**. Reports list every component and per-video coverage. Unregistered views are reported and omitted from training. For stricter inclusion, raise both thresholds toward 1.0.
5. Undistortion creates a calibrated pinhole dataset. Exported filenames are flattened uniquely and receive a global numeric suffix so both loaders use the same periodic holdout order. Image IDs, camera IDs, 2D observations, and 3D tracks stay consistent.
6. The chosen backend trains from that common dataset. The gsplat route exports a real Gaussian PLY. The native MLP route keeps native neural primitive checkpoints.
7. An offline renderer produces a clean RGB MP4. By default it interpolates along the most-registered single clip, avoiding artificial jumps between separate flights. Change `render.clip` using IDs from the reports to choose a flight. Slerp preserves valid camera rotations.

The reconstruction has an arbitrary coordinate frame and scale. MP4 video normally provides no usable per-frame EXIF/GPS for COLMAP. This pipeline does not extract DJI SRT telemetry, recover metric scale, georeference, or compensate rolling shutter. See [capture and calibration](docs/capture.md).

## Preview, inspect, and resume

```bash
# No CUDA training or backend installation required for this check:
drone-scene doctor --preprocessing-only

# Print a plan without running external tools or creating files:
drone-scene run --config configs/titan.json --videos /data/drone/full-capture \
  --workspace work/site --dry-run

# Stop after the validated training dataset:
drone-scene run --config configs/titan.json --videos /data/drone/full-capture \
  --workspace work/site --until prepare

# Same command later skips completed stages whose inputs/config still match:
drone-scene run --config configs/titan.json --videos /data/drone/full-capture \
  --workspace work/site

# Interactive gsplat CUDA viewer (Ctrl+C to exit):
drone-scene view --config configs/titan.json --workspace work/site

# Rerender; editing only render.* settings does not retrain:
drone-scene render --config configs/titan.json --workspace work/site
```

The viewer uses port 8080. On a remote workstation, forward it with `ssh -L 8080:localhost:8080 alex@alex-ws`, then open `http://localhost:8080`. The upstream viewer is a development server without authentication; keep its port on a trusted network. PLY viewing in SuperSplat does not require your workstation GPU server.

**Resume means completed-stage caching.** Extraction and preparation rebuild their own partial outputs after interruption. SfM can reuse its database and remap. The gsplat example's `--ckpt` is evaluation/viewing, **not an optimizer resume flag**; interrupted gsplat training restarts training when rerun. Native MLP additionally saves `train_state_latest.pt` with optimizer state; advanced resume commands are in [troubleshooting](docs/troubleshooting.md). Do not run two jobs in the same workspace.

Inputs are identified by absolute path, size, and modification time, not a full hash of every video. Moving/changing footage requires a new workspace. Dataset/model corruption is not repaired by cache stamps; retain originals and use new workspaces when in doubt.

## Native MLP-Splatting route

This is the actual [kxhit/mlp-splatting](https://github.com/kxhit/mlp-splatting) code, pinned in `upstreams.lock.json`, with RGB-only supervision and native COLMAP intrinsics. It does not require LSeg/SAM features. It is an experimental outdoor starting recipe; the supplied upstream scenes are Replica/ScanNet, and this project makes no city-scale quality or speed claim.

**Run this on an Ampere-or-newer GPU, not your TITAN RTX:**

```bash
conda env create -f environment.mlp.yml
conda activate drone-scene-mlp
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/install_backend.sh mlp

drone-scene run --config configs/mlp-ampere.json \
  --videos /data/drone/overlapping-clips --workspace work/mlp-site
```

The final native checkpoint is `work/mlp-site/model/mlp/scene_ckpt.pt`. The renderer loads that representation directly and makes an MP4 from interpolated poses. It does not rename/convert it to Gaussian PLY. All views must share one calibration; digital zoom, different lenses, or independently calibrated clips require the gsplat route or upstream loader changes.

The upstream MLP loader stacks **all float RGB images in host RAM**, then copies a split. The pipeline prints an estimate and rejects training if its conservative RAM budget exceeds available memory. With many views, reduce resolution/frame count or plan loader/partitioning work before attempting the entire capture. Increasing `points_per_leaf` reduces initial neural primitive count; it does not implement streaming or LOD.

For additional rendering options and their representation requirements, read [rendering](docs/rendering.md). For a better fit to very large sites, read [alternatives](docs/large-scenes.md).

## Publish your repository

The supplied Git repository is ready to publish. The connected GitHub tools used during preparation do not expose new-repository creation, so no remote has been claimed or created.

On a workstation with [GitHub CLI](https://cli.github.com):

```bash
gh auth login
bash scripts/publish.sh dolokov/drone-scene-pipeline
```

The script creates a **private** repository and pushes the source. You can pass another `OWNER/NAME` you control. Videos, extracted data, checkpoints, and CUDA builds are ignored. If that remote name already exists, choose another name; the script does not overwrite a repository.

## Development and provenance

```bash
python -m pip install -e . pytest==8.3.5 ruff==0.11.2 scipy==1.15.2
python -m pytest -q
ruff check drone_scene tests
```

CPU tests cover binary camera/observation preservation, component coverage, cross-flight matching, stale cache rejection, trajectory rotations, command construction, and real FFmpeg video extraction/encoding. GPU installation/training is a separate workstation validation step. The CI workflow runs CPU checks, not expensive reconstruction/training.

This repository's orchestration code is MIT licensed. Upstream repositories/dependencies retain their own licenses. Native MLP-Splatting is MIT; gsplat is Apache-2.0. Review third-party terms before redistribution. See [sources and pins](docs/sources.md).
