<div align="center">

# MLP Splatting

### Object-Centric Neural Fields

Compact neural primitives for rendering, scene decomposition, and feature-aware interaction.

[Paper](https://arxiv.org/abs/2606.03877) ·
[Project page](https://www.shinjeongkim.com/mlp-splatting/) ·
[Interactive demo](https://www.shinjeongkim.com/mlp-splatting/#visualization)

<img src="assets/teaser.svg" alt="MLP Splatting overview" width="100%">

</div>

MLP Splatting represents a scene with spatially localized light-field MLPs.
Each primitive predicts radiance and opacity within an anisotropic Gaussian
support. Optional feature distillation supports open-vocabulary selection,
segmentation, and editing.

This repository provides a tile-based CUDA implementation with training and
evaluation entrypoints. The supplied Replica configurations use **m=4**.

## Installation

Requirements: Linux, Python 3.10+, and a CUDA toolkit with `nvcc`.
The supplied Replica configurations require an NVIDIA GPU with compute
capability 8.0 or newer. Install a supported PyTorch release with matching
torchvision. RTX 50-series GPUs require CUDA 12.8 or newer.

```bash
git clone https://github.com/kxhit/mlp-splatting.git
cd mlp-splatting

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Choose a PyTorch wheel compatible with your driver:
# https://pytorch.org/get-started/locally/
python -m pip install "torch>=2.10.0" "torchvision>=0.25.0"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_install.py
```

The CUDA extension compiles on first use. Set `TORCH_EXTENSIONS_DIR` to
choose its build cache. Install optional evaluation dependencies for LPIPS
and text-based semantic evaluation:

```bash
python -m pip install -r requirements-optional.txt
```

See [SECURITY.md](SECURITY.md) for checkpoint loading guidance.

## Data

### Replica

Download the
[Feature-3DGS processed Replica archive](https://drive.google.com/file/d/1sC2ZJUBRHKeWXXVUj7rIBEM-xaibvGw7/view?usp=sharing).
Each scene should contain:

```text
room0/
├── images/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── rgb_feature_langseg/         # 512D LSeg features
```

The configurations use the Feature-3DGS split with `llffhold=8` and offset 2.
Training exclusions are applied after splitting: room1 excludes
`rgb_009.png`; office4 excludes `rgb_102.png` and `rgb_108.png`.
Room0 and office3 use all training views. Source images remain on disk, and
evaluation uses the complete test split.

Retain the terms and citation of the
[Replica dataset](https://github.com/facebookresearch/Replica-Dataset).

### ScanNet

Prepare each ScanNet scene in the processed COLMAP layout below. Raw `.sens`,
depth, and pose files are not read directly by the loader.

```text
sceneXXXX_XX/
├── images/                     # RGB images, e.g. 000000.jpg
├── sparse/0/
│   ├── cameras.bin             # camera calibration
│   ├── images.bin              # registered image names and poses
│   └── points3D.bin            # non-empty point cloud for initialization
└── rgb_feature_langseg/        # optional for RGB-only training
    └── 000000.pt               # 512D LSeg features for 000000.jpg
```

RGB filenames must match the names in `images.bin`. For LSeg supervision,
provide one feature file per RGB image with the same filename stem. Supported
formats are `.pt`, `.npy`, and `.npz`, containing a tensor/array of shape
`512 × H × W` or `H × W × 512`. Dictionary/archive payloads should use the key
`feat`. If both layouts are possible, name the file with an explicit layout
suffix, such as `000000_feature_chw.pt` or `000000_feature_hwc.pt`.

ScanNet uses the COLMAP loader (`--dataset mip360`). Supply these data options
to both `train.py` and `evaluate.py`:

```text
--dataset mip360 --data /path/to/ScanNet/sceneXXXX_XX --factor 1
--split-mode scannet_feature3dgs --llffhold 8 --split-test-offset 2
```

Views are sorted by numeric filename suffix; zero-based indices 2, 10, 18,
and so on form the test split, with the remaining views used for training.
Use `--feature-type lseg` for feature-supervised training, or
`--feature-type none` for RGB-only training. `--feature-dir` can override the
default feature folder. With `--factor 2`, `4`, or `8`, provide the corresponding
pre-resized `images_2/`, `images_4/`, or `images_8/` directory with unchanged
filenames; the loader does not resize RGB images automatically.

All views must share one calibration. The default camera mode requires
undistorted `SIMPLE_PINHOLE` or `PINHOLE` cameras with centered principal points
and approximately square pixels. For non-centered intrinsics, unequal focal
lengths, or an `OPENCV` camera model, select `--camera-mode colmap-native` when
training. RGB evaluation restores the camera mode from the checkpoint.

## Training

```bash
python train.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/Replica/room0 \
  --output outputs/room0
```

Configurations are provided for `room0`, `room1`, `office3`, and `office4`:

```bash
for scene in room0 room1 office3 office4; do
  python train.py \
    --config "configs/replica/${scene}_lseg.json" \
    --data "/path/to/Replica/${scene}" \
    --output "outputs/${scene}"
done
```

Each configuration uses m=4 octree initialization, 60,000 iterations,
constant learning rates, 8×8 tiles, four-view mosaics, and 512D LSeg feature
supervision. Model, optimizer, and resolved configuration files are saved in
the output directory. See [configuration details](configs/replica/README.md).

The renderer checks tile and per-pixel capacities before processing records.
The default tile capacity is 1,536; exceeding a capacity raises an error.

## RGB evaluation

```bash
python evaluate.py \
  --config configs/replica/room0_lseg.json \
  --data /path/to/Replica/room0 \
  --checkpoint outputs/room0/scene_epoch_60000.pt \
  --output results/room0.json
```

Evaluation reports per-image and mean PSNR, SSIM, LPIPS, and render timing.
Use `--disable-lpips` to omit LPIPS. Training exclusions do not remove test
views. Command-line values override configuration values; use
`python train.py --help` and `python evaluate.py --help` for available options.

## Semantic evaluation

Use label maps, a label list, and text embeddings from the same semantic
embedding model as the supervised features:

```bash
python mip360/eval_lseg_table2.py \
  --data_dir /path/to/Replica/room0 \
  --scene_ckpt outputs/room0/scene_epoch_60000.pt \
  --factor 1 \
  --gt_mode semantic_png \
  --gt_semantic_dir /path/to/labels \
  --labels_file /path/to/labels.txt \
  --text_feature_pt /path/to/text_features.pt \
  --tile_order tilecenter \
  --tile_size 8 \
  --score_threshold 5e-3 \
  --save_json results/room0_semantics.json
```

Label maps and model-specific text embeddings are not bundled. Scores against
teacher-generated label maps measure agreement with that teacher.

## Repository layout

```text
train.py             training entrypoint
evaluate.py          RGB evaluation entrypoint
configs/replica/     four m=4 scene configurations
mip360/              model, trainer, loaders, and evaluation
mip360/cuda/         CUDA renderer and training kernels
replica/             Replica loader and launcher
scripts/             installation and package checks
tools/               inference and checkpoint utilities
tests/               CPU and CUDA regression tests
```

## License and citation

Original code is available under the [MIT License](LICENSE). Third-party
components retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The code license does not relicense datasets or third-party model weights.

```bibtex
@inproceedings{kim2026mlpsplatting,
  title     = {MLP Splatting: Object-Centric Neural Fields},
  author    = {Kim, Shinjeong and Cheng, Yuzhou and Kong, Xin and Kelly, Paul H. J. and Davison, Andrew J.},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

See [CITATION.cff](CITATION.cff) for citation-manager metadata.
