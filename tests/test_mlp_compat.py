import ast
import gzip
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from drone_scene import mlp_compat
from drone_scene.common import PipelineError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mlp_checkout(tmp_path):
    fixture = ROOT / "tests/fixtures/mlp-port-sources.json.gz"
    original = json.loads(gzip.decompress(fixture.read_bytes()))
    directory = tmp_path / "mlp"
    for name, content in original.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (directory / "unrelated.py").write_text("original = True\n")
    for args in (["init", "-b", "main"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@local"], ["add", "."],
                 ["commit", "-m", "Pinned source fixture"]):
        subprocess.run(["git", "-C", str(directory), *args], check=True, capture_output=True)
    return directory, original


def build_config(source, capability):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "_tile_ext_build_config")
    namespace = {"os": os, "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: capability))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned MLP build config>", "exec"), namespace)
    return namespace["_tile_ext_build_config"]()


def test_patch_applies_exactly_and_is_idempotent(mlp_checkout):
    repo, _ = mlp_checkout
    assert mlp_compat.verify(repo, allow_pristine=True) == "pristine"
    with pytest.raises(PipelineError, match="missing"):
        mlp_compat.verify(repo)
    assert mlp_compat.apply(repo) == "applied"
    assert mlp_compat.verify(repo) == "patched"
    assert mlp_compat.apply(repo) == "already applied"


def test_patch_does_not_accept_other_user_edits(mlp_checkout):
    repo, _ = mlp_checkout
    mlp_compat.apply(repo)
    path = repo / "scannet/cumodel.py"
    before = path.read_text()
    path.write_text(before + "\nuser_edit = True\n")
    with pytest.raises(PipelineError, match="unrecognized edits"):
        mlp_compat.apply(repo)
    assert path.read_text().endswith("user_edit = True\n")
    path.write_text(before)
    (repo / "unrelated.py").write_text("user_edit = True\n")
    with pytest.raises(PipelineError, match="unrelated edits"):
        mlp_compat.verify(repo)


def test_patch_rejects_partial_installation(mlp_checkout):
    repo, original = mlp_checkout
    mlp_compat.apply(repo)
    (repo / "scannet/cumodel.py").write_text(original["scannet/cumodel.py"])
    with pytest.raises(PipelineError, match="partially"):
        mlp_compat.apply(repo)


def test_turing_gets_its_own_jit_binary_and_disallows_csr(mlp_checkout, monkeypatch):
    repo, _ = mlp_checkout
    mlp_compat.apply(repo)
    source = (repo / "scannet/cumodel.py").read_text()
    monkeypatch.setenv("MLPSP_CUDA_BACKEND", "auto")
    monkeypatch.setenv("MLPSP_TRAINING_BACKEND", "legacy")
    _, name, flags = build_config(source, (7, 5))
    assert "turing_fp32_v1" in name and "-DMLPSP_TURING_COMPAT=1" in flags
    monkeypatch.setenv("MLPSP_TRAINING_BACKEND", "csr")
    with pytest.raises(RuntimeError, match="CSR/Triton"):
        build_config(source, (7, 5))
    monkeypatch.setenv("MLPSP_CUDA_BACKEND", "native")
    with pytest.raises(RuntimeError, match="8.0"):
        build_config(source, (7, 5))


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (9, 0), (12, 0)])
def test_native_build_options_are_preserved(mlp_checkout, monkeypatch, capability):
    repo, original = mlp_checkout
    mlp_compat.apply(repo)
    monkeypatch.setenv("MLPSP_CUDA_BACKEND", "auto")
    monkeypatch.setenv("MLPSP_TRAINING_BACKEND", "legacy")
    patched = (repo / "scannet/cumodel.py").read_text()
    assert build_config(patched, capability) == build_config(original["scannet/cumodel.py"], capability)


def preprocess(source, compatibility, arch):
    # Expand only preprocessor selection, replacing external headers with visible
    # markers. This inspects every kernel/body without needing CUDA headers/GPU.
    source = re.sub(r'^#include\s+[<"]([^>"\n]+)[>"]',
                    lambda m: "INCLUDED_" + re.sub(r"\W", "_", m[1]) + ";", source, flags=re.M)
    result = subprocess.run(["g++", "-E", "-P", "-x", "c++", "-",
                             f"-DMLPSP_TURING_COMPAT={compatibility}", f"-D__CUDA_ARCH__={arch}"],
                            input=source, capture_output=True, text=True, check=True)
    return re.sub(r"\s+", " ", result.stdout).strip()


@pytest.mark.skipif(not shutil.which("g++"), reason="g++ preprocessor not installed")
def test_turing_build_excludes_all_bf16_and_wmma_but_keeps_cuda_backward(mlp_checkout):
    repo, original = mlp_checkout
    mlp_compat.apply(repo)
    patched = (repo / "scannet/cuda/np_tile_kernels.cu").read_text()
    turing = preprocess(patched, compatibility=1, arch=750)
    for forbidden in ("__nv_bfloat", "INCLUDED_cuda_bf16_h", "INCLUDED_rgb_wmma_cuh",
                      'm.def("render_tiles_rgb_wmma"', 'm.def("render_tiles_training_records_wmma"'):
        assert forbidden not in turing
    assert "render_tiles_kernel_tilecenter_order<float>" in turing
    assert 'm.def("render_tiles_backward_tilecenter_full"' in turing
    assert 'm.def("render_tiles_forward_tilecenter"' in turing
    # On SM80+ the preprocessed CUDA source is token-for-token unchanged.
    assert preprocess(patched, 0, 800) == preprocess(original["scannet/cuda/np_tile_kernels.cu"], 0, 800)
