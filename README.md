# Drone scene pipeline

**Turn multiple overlapping MP4 drone videos into one connected reconstruction, train native MLP-Splatting or Gaussian splats, and render an RGB flythrough.** Linux, one NVIDIA GPU, FFmpeg, COLMAP, and conda.

## Start here: MLP-Splatting on your TITAN RTX

Your GPU is **Turing, compute capability 7.5, 24 GB**. This repository now patches the pinned upstream renderer so the **same MLP representation can train and render on Turing using its FP32 CUDA kernels**. Ampere-or-newer GPUs retain the full native build, including the optional WMMA/BF16 entrypoints. There is no automatic conversion to Gaussians.

| Actual GPU capability | `training.mlp.kernel_backend: auto` selects | Implementation |
| --- | --- | --- |
| 7.5, including TITAN RTX | `turing` | Upstream scalar FP32 forward/backward; WMMA and BF16 code excluded at compile time |
| 8.0 or higher | `native` | Full upstream CUDA build and its original optional WMMA entrypoints |
| Below 7.5 | Error | Outside this pipeline's supported GPU target |

The MLP environment stays **CUDA 12.8 / PyTorch 2.10** on both GPU families. The patch changes compilation/dispatch, not the scene equations, native checkpoint schema, or restricted checkpoint-loading rules. FP32 is the pipeline's training precision on both paths. The TITAN path does not call CSR/Triton kernels. Expect it to be slower than an optimized tensor-core workflow; no speed or maximum scene size is promised.

The `CUDA 13.4` line in `nvidia-smi` describes the driver's supported CUDA level. It does not say which `nvcc` toolkit is installed. Keep the recent driver; install the specified toolkit inside conda. There is no need to downgrade the system driver.

```bash
conda env create -f environment.mlp.yml
conda activate drone-scene-mlp
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/install_backend.sh mlp

# Compile and exercise the real MLP kernels without a video or COLMAP solve:
drone-scene smoke --config configs/mlp-pilot.json --workspace work/mlp-smoke

# First end-to-end test with a few overlapping clips:
drone-scene run --config configs/mlp-pilot.json \
  --videos /data/drone/pilot --workspace work/mlp-pilot

# Larger run after checking the pilot:
drone-scene run --config configs/mlp-titan.json \
  --videos /data/drone/full-capture --workspace work/mlp-site
```

The first use compiles CUDA code and can take several minutes. The 2,000-iteration pilot is a setup/capture check. The full TITAN recipe uses 768 px images, a 750-frame budget, 64 points per octree leaf, and 30,000 iterations as conservative starting settings. These are not memory/performance guarantees.

**Validation boundary:** CPU regression and preprocessor checks pass; this conversation has no NVIDIA device or toolkit, so the new SM75 binary has not been compiled or executed here. Run the smoke test before spending time on reconstruction. It validates finite RGB/gradients, loss improvement, the selected binary's entrypoints, and the actual upstream inference checkpoint round-trip. The report goes to `work/mlp-smoke/smoke.json`.

The final scene is `work/mlp-site/model/mlp/scene_ckpt.pt`. `execution.json` next to it records the selected CUDA build and patch pin. Render it with the same config using `drone-scene render`. See [the compatibility implementation](docs/mlp-turing.md) for patch verification, native overrides, and how to test an existing installation.

## Gaussian alternative: install once

Extract this repository and open its root directory. Install [Miniforge](https://github.com/conda-forge/miniforge) if you do not have conda. The environment files target Linux x86-64.

```bash
conda env create -f environment.yml
conda activate drone-scene
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/install_backend.sh gsplat
```

The installer clones exact upstream commits into `third_party/`, installs PyTorch before building CUDA extensions, and checks the active GPU/toolkit. Keep the cloned repositories: the pipeline calls their pinned entrypoints. Compilation can take several minutes. Close GPU-heavy desktop applications before training; your screenshot already has about 1.7 GB allocated to the desktop/apps.

`environment.yml` and the backend pins specify a reproducible starting stack, not a solved package lock for every Linux distribution. The conda solve, CUDA build, and GPU execution must be verified on your workstation. See [validation status](docs/validation.md).

## Gaussian alternative: run a small pilot end to end

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

## Gaussian alternative: run the larger scene

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

This is the actual [kxhit/mlp-splatting](https://github.com/kxhit/mlp-splatting) code, pinned in `upstreams.lock.json`, with a checked compatibility patch, RGB-only supervision, and native COLMAP intrinsics. It does not require LSeg/SAM features. It is an experimental outdoor starting recipe; the supplied upstream scenes are Replica/ScanNet, and this project makes no city-scale quality or speed claim.

For **automatic selection on either GPU family**, use `configs/mlp-pilot.json` or `configs/mlp-titan.json` as shown above. To **require the full native build on an 8.0+ card**, use `configs/mlp-ampere.json`:

```bash
conda env create -f environment.mlp.yml
conda activate drone-scene-mlp
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/install_backend.sh mlp

drone-scene run --config configs/mlp-ampere.json \
  --videos /data/drone/overlapping-clips --workspace work/mlp-site
```

The final native checkpoint is `work/mlp-site/model/mlp/scene_ckpt.pt`, regardless of which CUDA build was selected. The renderer loads that representation directly and makes an MP4 from interpolated poses. It does not rename/convert it to Gaussian PLY. All views must share one calibration; digital zoom, different lenses, or independently calibrated clips require the gsplat route or upstream loader changes. A `native` override on TITAN RTX fails explicitly; use `auto` or `turing`.

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

CPU tests cover binary camera/observation preservation, component coverage, cross-flight matching, stale cache rejection, trajectory rotations, command construction, real FFmpeg video extraction/encoding, exact patch application, and both CUDA build selections. C++ preprocessing verifies native kernel preservation and Turing's WMMA/BF16 exclusion. An opt-in test runs the real GPU smoke check; see [validation status](docs/validation.md). The standard CI workflow runs CPU checks.

This repository's orchestration code is MIT licensed. Upstream repositories/dependencies retain their own licenses. Native MLP-Splatting is MIT; gsplat is Apache-2.0. The compressed pinned-source regression fixture retains upstream attribution and notices in `tests/fixtures/`. Review third-party terms before redistribution. See [sources and pins](docs/sources.md).
