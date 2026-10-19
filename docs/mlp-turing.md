# TITAN RTX MLP compatibility

This is a source patch to the actual neural-primitive renderer. It does not substitute Gaussian splats or a different network. The same training and offline-render commands operate on both GPU families.

## Dispatch

`training.mlp.kernel_backend` accepts:

| Value | Behavior |
| --- | --- |
| `auto` (default) | SM75 → Turing FP32 build; SM80+ → full native build |
| `native` | Require SM80+; fail clearly on TITAN RTX |
| `turing` | Force the scalar-compatible build on SM75 or newer, useful for comparison/debugging |

Use `configs/mlp-titan.json` or `configs/mlp-pilot.json` for automatic selection. `configs/mlp-ampere.json` explicitly requires `native`. The Python preflight and patched upstream extension loader enforce matching rules. Driver CUDA version is not used to infer compute capability.

The existing Gaussian route (`configs/titan.json`) remains an explicit alternative. Changing GPU build never switches the selected scene representation.

## What the patch changes

The pinned upstream file unconditionally includes WMMA headers that instantiate BF16 kernels. Those paths require SM80+. Merely setting an architecture variable to `7.5` does not make those instantiations legal.

`patches/mlp-turing-v1.patch` modifies two files:

1. `scannet/cumodel.py`: select the compatibility build on SM75; reject unsupported requests; use a separate JIT module name and `-DMLPSP_TURING_COMPAT=1`; disallow CSR/Triton training in this build.
2. `scannet/cuda/np_tile_kernels.cu`: conditionally exclude BF16 headers/specializations/launches, WMMA implementation includes, and their Python bindings. An attempted BF16 scalar call fails explicitly. FP32 and ordinary FP16 scalar code is retained.

The pipeline trains with upstream's existing FP32 `render_tiles_forward_tilecenter` and `render_tiles_backward_tilecenter_full` kernels. Alpha compositing, positional encoding, neural weights, geometry gradients, tile-capacity checks, and checkpoint layouts are retained. The wrapper's RGB recipe already used the `legacy` FP32 backward; this patch makes that route compile independently of the SM80-only optional code.

With the compatibility macro disabled, C++ preprocessing produces the same native CUDA tokens as the unpatched source, and native JIT build options stay identical. The original optional WMMA/BF16 entrypoints remain available on SM80+. The RGB pipeline does not promise tensor-core acceleration merely because those optional entrypoints are present.

The Turing build has a distinct cache name (`*_turing_fp32_v1_*`). Training and rendering set `TORCH_CUDA_ARCH_LIST` from the actual selected GPU. Do not reuse an old extension binary by manually copying build caches between cards.

## Exact patch verification and reinstall

The installer fetches commit `426e0a24a02124bb07ce0f35e67bb241fba484d4` and applies the patch before installing the MLP package. The manifest records SHA-256 hashes of both pristine files, both patched files, and the patch itself. `upstreams.lock.json` pins the patch hash as part of training-cache provenance.

Installation is idempotent for an exactly patched checkout. Unknown source changes and partial application fail instead of being overwritten. Runtime source checks allow only those exact patch edits and reject other tracked changes. The native build uses the same checked patch with its Turing branches inactive.

```bash
conda activate mlp
export CUDA_HOME="$CONDA_PREFIX"
bash scripts/install_backend.sh mlp
python -m drone_scene.mlp_compat verify third_party/mlp
```

If you have already installed the old unpatched MLP checkout, rerun the installer. The older Gaussian-only `drone-scene` environment contains a different torch version; create/activate `environment.mlp.yml` for MLP on either GPU family. PyTorch 2.10 restricted checkpoint loading is retained. No downgrade to older checkpoint-loading behavior is used.

When invoking upstream tools directly, the patched loader defaults to `auto` too. To force a build for direct upstream commands, set `MLPSP_CUDA_BACKEND=auto|turing|native` before the first extension call. The wrapper's config takes precedence when using this project's commands.

## Verify on the workstation

```bash
drone-scene smoke --config configs/mlp-pilot.json --workspace work/mlp-smoke

# Equivalent automated GPU regression, after installing pytest:
DRONE_SCENE_RUN_CUDA_TESTS=1 python -m pytest -q tests/test_mlp_cuda.py
```

The smoke test compiles the selected real extension, renders an 8×8 two-primitive scene, checks finite gradients for geometry and all MLP layers, optimizes appearance, and requires the toy loss to improve. It exports with the actual upstream checkpoint helper, reloads through the actual upstream inference loader, and compares RGB. It also checks whether WMMA entrypoints are present for the selected build. This does not measure capture quality or certify a large scene.

This execution environment has no NVIDIA GPU, nvcc, or torch installation. CPU tests exercise real patch application and full-source C++ preprocessing, including token-equivalent native output. GPU compilation/runtime must be confirmed with the provided smoke test. We do not report an SM75 benchmark or claim that a GPU test ran here.

## Limits

The TITAN route is RGB-only and uses scalar FP32 computation. It can be slow and still has the upstream full-frame rasterizer, eager image loader, tile capacities, and shared-calibration restrictions. Start with `mlp-pilot.json`; change sampling/training settings in a new workspace when scaling up. Native inference exports may store weights in FP16 on disk; the upstream loader restores FP32 runtime parameters, so this does not require BF16 support.
