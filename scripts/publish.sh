#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
repository="${1:-dolokov/drone-scene-pipeline}"
if ! command -v gh >/dev/null; then
  echo 'Install GitHub CLI (https://cli.github.com), then run gh auth login.' >&2
  exit 2
fi
gh auth status
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi
git add -- README.md docs drone_scene scripts tests configs .github .gitignore \
  LICENSE pyproject.toml environment.yml environment.mlp.yml requirements-gsplat.txt upstreams.lock.json
if ! git diff --cached --quiet; then
  git commit -m 'Add multi-video COLMAP, MLP-Splatting and TITAN Gaussian pipeline'
fi
# This creates a PRIVATE remote. No footage or model outputs are tracked.
gh repo create "$repository" --private --source . --remote origin --push
