#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
backend="${1:-gsplat}"
if [[ "$backend" != gsplat && "$backend" != mlp ]]; then
  echo 'Usage: bash scripts/install_backend.sh gsplat|mlp' >&2
  exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo 'Activate the conda environment before installing.' >&2
  exit 2
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
python -m pip install 'pip>=24' 'setuptools>=69' wheel ninja
if [[ "$backend" == gsplat ]]; then
  python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
else
  python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
fi
python -m pip install -e .
# Fail before compiling CUDA if the active GPU/toolkit is unsupported.
python - "$backend" <<'PY'
import sys
from drone_scene.backends import gpu_check
gpu_check(sys.argv[1])
PY
export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
mkdir -p third_party
python - "$backend" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

backend = sys.argv[1]
lock = json.loads(Path('upstreams.lock.json').read_text())[backend]
destination = Path('third_party') / backend
if not destination.exists():
    subprocess.run(['git', 'clone', lock['url'], str(destination)], check=True)
if subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain']).strip():
    raise SystemExit(f'{destination} contains local changes; preserve them before reinstalling.')
subprocess.run(['git', '-C', str(destination), 'fetch', 'origin', lock['commit']], check=True)
subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', lock['commit']], check=True)
subprocess.run(['git', '-C', str(destination), 'submodule', 'update', '--init', '--recursive'], check=True)
PY
if [[ "$backend" == gsplat ]]; then
  python -m pip install -r requirements-gsplat.txt
  python -m pip install --no-build-isolation \
    'git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5'
  python -m pip install --no-build-isolation -e third_party/gsplat
  python -m drone_scene doctor --config configs/titan.json
else
  python -m pip install -r third_party/mlp/requirements.txt
  python -m pip install -e third_party/mlp --no-deps
  python -m drone_scene doctor --config configs/mlp-ampere.json
fi
echo 'Backend ready. The first render/training call may compile additional CUDA kernels.'
