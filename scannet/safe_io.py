"""Restricted CPU loading for trusted, tensor/state-dict checkpoints.

This reduces pickle attack surface; it is not a sandbox for hostile files.
Never retry a failed restricted load using unrestricted pickle.
"""
from pathlib import Path
import pickle

from packaging.version import InvalidVersion, Version
import torch


MIN_TORCH_VERSION = Version("2.10.0")


def require_supported_torch() -> None:
    try:
        version = Version(str(torch.__version__).split("+")[0])
    except InvalidVersion as exc:
        raise RuntimeError("Cannot verify PyTorch version; install stable PyTorch >=2.10.0.") from exc
    if version < MIN_TORCH_VERSION:
        raise RuntimeError(
            "Checkpoint loading requires stable PyTorch >=2.10.0; older releases "
            "have known weights_only vulnerabilities. Upgrade PyTorch and torchvision."
        )


def load_checkpoint(path: str | Path):
    """Read on CPU using restricted loading; never extend PyTorch's allowlist.

    PyTorch itself registers some built-in globals. Do not clear or mutate
    that process-wide registry here. Applications must not add untrusted ones.
    """
    require_supported_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Restricted checkpoint loading failed. Only trusted tensor/state-dict files "
            "are supported. Do not retry with weights_only=False or allowlist unknown classes."
        ) from exc
