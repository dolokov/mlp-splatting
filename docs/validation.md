# Validation status

Prepared and checked on 2026-10-16. No drone videos were attached to this conversation, and this execution environment has no COLMAP executable, conda, or NVIDIA GPU. The repository is an implemented, CPU-tested pipeline; it is not a claim that your footage has been reconstructed or that either CUDA environment has been built here.

## Checks completed

- **33 CPU tests passed** (`pytest 8.3.5`). Tests include real multi-MP4 FFmpeg extraction, frame-budget rejection, H.264 encoding at requested FPS, and odd-dimension handling.
- COLMAP binary tests preserve image IDs, camera IDs, 2D observations, and copied 3D point payloads while flattening names. Tests check common calibration, per-video component coverage, temporal/global pairs, and matching holdout sort order.
- Cache tests reject changed inputs/settings and missing outputs. CLI dry-run creates no files and needs no training GPU.
- Trajectory tests check Slerp endpoint preservation and valid orthonormal rotation matrices.
- GPU preflight tests simulate Turing and verify that native MLP is rejected before compilation and that toolkit matching uses actual nvcc/PyTorch versions.
- **Ruff checks passed** for pipeline and test code. Python compilation and `bash -n` checks passed for the scripts.
- Training command contracts were checked against source retrieved at the exact pinned commits: the actual upstream MLP `build_parser()` accepted the generated argv; gsplat's actual `Config` and `MCMCStrategy` field definitions accepted the generated argv through pinned `tyro 0.9.19`. This check extracted parser/config definitions without importing or executing CUDA code. It verifies argument names/types, not GPU behavior.
- Installation source inspection caught gsplat's GLM submodule requirement. The installer initializes recursive submodules at the pinned commit, and runtime checks reject missing/different submodules.

CPU checks ran on Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0, and SciPy 1.17.0 from the supplied execution runtime. The workstation conda recipes intentionally use Python 3.10/3.11 and NumPy 1.26.4 to match their research backends. CI also targets Python 3.10. CI has been provided; no hosted GitHub Actions run is claimed.

## Checks to run on your workstation

1. Create the selected conda environment and run `scripts/install_backend.sh`. This validates the solve, wheel/toolkit pairing, compiler, dependencies, submodules, and hardware visibility.
2. Run `drone-scene doctor --config configs/titan.json` on the TITAN RTX.
3. Run `configs/pilot.json` on a few genuinely overlapping clips. Inspect the reconstruction coverage report and PLY before raising the scene size/training budget.
4. Verify native GPU training, exported checkpoint/PLY, offline RGB render, and interactive viewer on that pilot. Measure memory and timing locally.
5. Only then launch the full capture. For MLP, use the separate environment and supported GPU floor; verify its outdoor quality and host-RAM use independently.

No GPU-training, renderer performance, full COLMAP reconstruction, conda solve, semantic editing, city-scale streaming, or survey-accuracy result is implied by the CPU checks.
