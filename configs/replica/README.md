# Replica configurations

The four scene configurations share the same training settings:
seed 2025, 60,000 iterations, constant learning rates,
8×8 tile-center rendering, four-view mosaics, and 512D LSeg features.

| Configuration | Training views | Excluded training images |
|---|---:|---|
| `room0_lseg.json` | 70 | None |
| `room1_lseg.json` | 69 | `rgb_009.png` |
| `office3_lseg.json` | 70 | None |
| `office4_lseg.json` | 63 | `rgb_102.png`, `rgb_108.png` |

Exclusions apply after the Feature-3DGS split (`llffhold=8`, offset 2).
Every scene retains all 10 test views. No image files are deleted.

Training uses record-based CUDA forward/backward computation, FP16 MLP
operands, FP32 Adam state, and a tile candidate capacity of 1,536.
Capacity overflow raises an error. The RGB loss is 0.9 L1 + 0.1 DSSIM;
feature distillation uses L1 loss. Features use a shared base plus four
directional residual tensors (`feature_slots=5`).
Feature loss updates feature attributes; geometry and appearance remain
RGB-supervised.

Learning rates are `1e-2` for log-scale, `1e-4` for position, and
`1e-3` for the other parameters. Quaternion parameters are normalized
after each optimizer step and during the forward pass.

Train and evaluate with the same configuration. Command-line arguments
override its values. Paths to datasets, checkpoints, and outputs are supplied
on the command line.
