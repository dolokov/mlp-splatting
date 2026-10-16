# Primary sources and exact upstream versions

Inspected on 2026-10-16. Mutable documentation pages may describe newer versions; runtime entrypoints are pinned in `upstreams.lock.json`.

| Source | Used for |
| --- | --- |
| [MLP-Splatting repository at 426e0a2](https://github.com/kxhit/mlp-splatting/tree/426e0a24a02124bb07ce0f35e67bb241fba484d4) | Native train/evaluate flags, checkpoint format, COLMAP loader, camera restriction, source license, dependencies |
| [MLP-Splatting project](https://www.shinjeongkim.com/mlp-splatting/) and [paper](https://arxiv.org/abs/2606.03877) | Native neural primitive representation and research scope |
| [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus) | TITAN RTX compute capability 7.5 |
| [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) | Matching torch/torchvision/CUDA wheel commands |
| [gsplat v1.5.3](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae) | Trainer, MCMC cap, checkpoint fields, PLY exporter, camera normalization, Viser viewer |
| [gsplat COLMAP example documentation](https://docs.gsplat.studio/main/examples/colmap.html) | Example training workflow |
| [COLMAP 3.11.1 feature CLI source](https://github.com/colmap/colmap/blob/3.11.1/src/colmap/exe/feature.cc) | matches_importer `pairs` mode really performs descriptor matching and geometric verification |
| [COLMAP CLI](https://colmap.github.io/cli.html) and [binary format](https://colmap.github.io/format.html) | SfM, undistortion, camera/image/observation structures |
| [SuperSplat](https://github.com/playcanvas/supersplat) | Gaussian editor/viewer option |
| [Viser](https://github.com/viser-project/viser) | Browser UI with a Python rendering server |
| [Scaffold-GS](https://github.com/city-super/Scaffold-GS) | Neural-conditioned Gaussian alternative |
| [Octree-GS](https://github.com/city-super/Octree-GS) / [Octree-AnyGS](https://github.com/city-super/Octree-AnyGS) | LOD alternatives for larger scenes |
| [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) | Alternative integrated Gaussian workflow |
| [OpenDroneMap](https://github.com/OpenDroneMap/ODM) and [outputs](https://docs.opendronemap.org/outputs/) | Mesh/orthophoto/DEM alternative |

The native MLP route's Ampere floor is a deliberate supported configuration boundary for this wrapper, based on the upstream supplied recipe requirements. Some upstream scalar code paths may merit Turing-port investigation, but this repository does not certify one or modify kernels to simulate support. Gaussian/Turing compatibility is an intended software configuration pending the user's GPU build and smoke test, not a benchmark claim.

The wrapper does not vendor source from these projects. The installer fetches pinned code from its original upstream, whose license and third-party notices remain authoritative.
