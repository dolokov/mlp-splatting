# Large drone sites and other options

## What this implementation actually scales

All videos share one SfM solve and coordinate frame. Frame budgets and sparse candidate matching control preprocessing cost; gsplat uses packed rasterization and capped MCMC growth. This is a practical **single-GPU, single-model baseline**, not an out-of-core city reconstruction system. Its only required model outputs are a connected COLMAP dataset, one backend's scene, and renders.

A TITAN RTX can be useful for a building/site experiment, but 24 GB does not determine the maximum acreage. Resolution, altitude, ground detail, view overlap, tree motion, primitive count, and the number of projected splat intersections all matter. No run time, FPS, or largest scene size is promised without a pilot on your footage.

If memory is tight, try these changes in a fresh config/workspace:

1. Lower `extraction.max_size` from 1600 to 1000–1200. This reduces rasterization cost directly.
2. Reduce redundant frames while preserving overlapping views and bridging flights. Do not lower FPS until adjacent cameras lose shared landmarks.
3. Lower Gaussian `cap_max` and keep SH degree 2 or try 1. The initial SfM points still need to be below the cap. Fewer primitives trade detail for memory.
4. Close GPU-heavy apps. Keep enough host RAM and fast SSD space for COLMAP, image caches, and outputs.
5. For native MLP, its eager loader and full-frame renderer are independent bottlenecks. Increase `training.mlp.points_per_leaf` to reduce initial primitive count and reduce ray-cache sizes. This still needs an Ampere-or-newer GPU for this route.

## Choose the representation for the deliverable

| Approach | Good reason to choose it | Tradeoffs / work remaining |
| --- | --- | --- |
| gsplat 3D Gaussians (included) | Useful first result on this TITAN; broad PLY viewing/editing ecosystem | A single high-detail site can still exceed memory; no built-in scene streaming |
| Native MLP-Splatting (included) | Experiment with expressive localized neural primitives and the paper's representation | Supported GPU floor here is 8.0+; outdoor tuning needed; native runtime; eager host-image loading |
| Scaffold-GS | MLP-conditioned Gaussian generation; a nearby idea if you want neural prediction with Gaussian rendering | Distinct from native MLP-Splatting; custom training/viewer/export assumptions; not wired into this CLI |
| Octree-GS / Octree-AnyGS | Explicit LOD structure for large scenes and view-dependent level selection | Additional environment/integration; use their renderer; ordinary PLY export may lose LOD/neural behavior |
| Block-based training | Larger area by fitting overlapping local scene blocks and streaming selected blocks | Requires shared global poses, block selection, overlap treatment, normalization transforms, seam handling, and a streaming renderer |
| Nerfstudio Splatfacto | A maintained training/viewer ecosystem with more general data processing tooling | Heavier environment; inspect current hardware requirements; not installed by this project |
| NeRF / radiance-field variants | Research alternative for difficult view-dependent appearance | Training/rendering cost and export differ; not a guaranteed solution to missing views or moving vegetation |
| OpenDroneMap / WebODM | Textured mesh, orthophoto, DEM, GIS workflows | Photogrammetry objectives; still images + calibrated/GPS inputs usually more useful for mapping than compressed video |

My suggested order for this capture: **small gsplat pilot → full connected SfM → evaluate quality/memory → choose MLP research or LOD/block engineering only if its benefits matter.** These are practical recommendations, not benchmark results on your footage.

## If the whole site does not fit

Keep the **global SfM solution** as the common geometry reference. Select overlapping spatial blocks of points and cameras from it. Train each block while recording the exact transform between its normalized model frame and the global frame. A renderer must apply those transforms, choose blocks/LODs, cull, and handle boundaries. Do not independently solve each MP4 and concatenate checkpoints or PLY files: scale, rotation, calibration, and exposure can disagree, and splats can double up at seams.

This repository does not implement block training, learned block fusion, LOD, or streaming. Implementing only a grid splitter would not make that system end to end. For a city-scale deployment, evaluate an existing LOD/block system before extending a monolithic research renderer.

## Matching at larger scale

The hybrid pair set is approximately `N * temporal_window + (N / global_stride)^2 / 2`. It trades robust cross-flight coverage against matching cost. It deliberately includes cross-flight candidates; it cannot invent visual overlap. If clips remain disconnected, decrease `global_stride`, use `exhaustive` for a small pilot, or capture a bridging flight.

`sfm.max_pairs` prevents accidentally starting a huge quadratic matching job. For thousands of views beyond this baseline, use the optional `vocab-tree` matcher with an existing COLMAP vocabulary tree (`sfm.vocab_tree`), or investigate hierarchical/local-feature retrieval and modern SfM options such as GLOMAP. Those tools have their own calibration/verification requirements. CPU SIFT is the portable default; enable `sfm.use_gpu` only if the installed COLMAP build supports it.
