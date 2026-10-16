# Third-party notices

`scannet/utils/read_write_model.py` is based on COLMAP's Python model I/O
utility. Its BSD 3-Clause copyright notice, redistribution terms, and warranty
disclaimer are retained in full at the top of that source file.

- Project: [COLMAP](https://github.com/colmap/colmap)
- Copyright: ETH Zurich and UNC Chapel Hill

The original MLP Splatting code is distributed under the root MIT `LICENSE`.
That license does not replace the terms of third-party components, datasets,
pretrained foundation models, or paper/project-page media.

## Structural similarity (SSIM)

`scannet/utils/loss_utils.py` implements the windowed SSIM formulation used by
[pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim), authored by Evan Su.
We retain its MIT attribution in `licenses/pytorch-ssim-MIT.txt`. GRAPHDECO's
Gaussian-Splatting license explicitly identifies its SSIM helper as based on
this permissively licensed work; this does not extend to the rest of 3DGS.

## Historical code is not part of this distribution

Earlier private development revisions contained GRAPHDECO/Inria helpers
(including `graphics_utils.py`) with non-commercial restrictions. Those
revisions are not included in this clean publication history. They have not
been relicensed under MIT. The private archival copy must retain their original
notices and the full Gaussian-Splatting license. Do not merge that history or
copy those files into this release without reviewing their terms.

For historical attribution, the complete license is also retained as
[`licenses/Gaussian-Splatting-historical-LICENSE.md`](licenses/Gaussian-Splatting-historical-LICENSE.md),
from `graphdeco-inria/gaussian-splatting` revision
`54c035f7834b564019656c3e3fcc3646292f727d`. This reference does not grant
commercial rights to historical code or apply its restrictions to unrelated
original code in this release.

## External dependencies and data

PyTorch, torchvision, NVIDIA CUDA/CUB, NumPy, Pillow, ImageIO, LPIPS and optional
CLIP/W&B components are installed separately and retain their upstream licenses.
Replica/ScanNet images, semantic labels and teacher features are not distributed
under the project's code license. Follow the dataset and model providers' terms.
Paper/project illustrations are scholarly media, not a grant of rights to any
depicted dataset content. The code license does not relicense those assets.
