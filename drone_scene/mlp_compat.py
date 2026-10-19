"""Apply and verify the one audited patch to the pinned upstream MLP renderer."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .common import PipelineError, checked_output, read_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches" / "mlp-turing-v1.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    spec = read_json(MANIFEST)
    patch = MANIFEST.parent / spec["patch"]
    lock = read_json(ROOT / "upstreams.lock.json")["mlp"]
    if (spec["upstream_commit"] != lock["commit"] or sha256(patch) != spec["patch_sha256"]
            or lock["compatibility_patch"]["sha256"] != spec["patch_sha256"]):
        raise PipelineError("MLP compatibility patch does not match its pinned manifest/lockfile")
    return spec, patch


def verify(repository: Path, *, allow_pristine=False):
    """Accept exactly pristine or exactly patched sources, never arbitrary upstream edits."""
    spec, _ = manifest()
    states = []
    for name, hashes in spec["files"].items():
        path = repository / name
        if not path.is_file():
            raise PipelineError(f"Pinned MLP source missing: {name}. Reinstall the backend.")
        actual = sha256(path)
        if actual == hashes["patched_sha256"]:
            states.append("patched")
        elif actual == hashes["original_sha256"]:
            states.append("pristine")
        else:
            raise PipelineError(f"MLP file {name} has unrecognized edits; preserve them before reinstalling.")
    changed = set(checked_output(["git", "-C", str(repository), "diff", "--name-only", "HEAD"]).splitlines())
    unexpected = changed - set(spec["files"])
    if unexpected:
        raise PipelineError(f"MLP source has unrelated edits: {sorted(unexpected)}")
    if len(set(states)) != 1:
        raise PipelineError("MLP patch is only partially applied; preserve this checkout and reinstall cleanly.")
    state = states[0]
    if state == "pristine" and not allow_pristine:
        raise PipelineError("MLP compatibility patch is missing. Run bash scripts/install_backend.sh mlp.")
    return state


def apply(repository: Path):
    if verify(repository, allow_pristine=True) == "patched":
        return "already applied"
    _, patch = manifest()
    checked_output(["git", "-C", str(repository), "apply", "--check", str(patch)])
    checked_output(["git", "-C", str(repository), "apply", str(patch)])
    verify(repository)
    return "applied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "verify"))
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    result = apply(args.repository.resolve()) if args.action == "apply" else verify(args.repository.resolve())
    print(f"MLP compatibility patch: {result}")


if __name__ == "__main__":
    main()
