"""Restricted loading regression tests; never execute arbitrary payloads."""
import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mip360.safe_io import load_checkpoint, require_supported_torch
from tools.convert_legacy_checkpoint import _load


def test_tensor_and_optimizer_containers_roundtrip(tmp_path):
    path = tmp_path / 'state.pt'
    torch.save({'scene_state_dict': {'weight': torch.arange(4.)},
                'optimizer_state_dict': {'state': {0: {'step': torch.tensor(3.)}}},
                'metadata': {'epoch': 3, 'labels': ['a', 'b']}}, path)
    value = load_checkpoint(path)
    assert torch.equal(value['scene_state_dict']['weight'], torch.arange(4.))
    assert value['scene_state_dict']['weight'].device.type == 'cpu'
    assert value['metadata']['epoch'] == 3


class UnsupportedObject:
    pass


@pytest.mark.parametrize('loader', [load_checkpoint, _load])
def test_custom_object_rejected_without_fallback(tmp_path, loader):
    path = tmp_path / 'object.pt'
    torch.save(UnsupportedObject(), path)
    with patch('mip360.safe_io.torch.load', wraps=torch.load) as observed:
        with pytest.raises(ValueError, match='Restricted checkpoint'):
            loader(path)
        assert observed.call_count == 1
        assert observed.call_args.kwargs == {'map_location': 'cpu', 'weights_only': True}


@pytest.mark.parametrize('version', ['2.4.0', '2.5.1', '2.9.1+cu128', '2.10.0.dev20250901', 'invalid'])
def test_old_or_unverifiable_runtime_rejected_before_io(version):
    with patch.object(torch, '__version__', version), patch.object(torch, 'load') as reader:
        with pytest.raises(RuntimeError):
            load_checkpoint('never-read.pt')
        reader.assert_not_called()


@pytest.mark.parametrize('version', ['2.10.0', '2.10.0+cu128', '2.11.0'])
def test_patched_version_accepted(version):
    with patch.object(torch, '__version__', version):
        require_supported_torch()


def test_loader_does_not_mutate_allowlist(tmp_path):
    path = tmp_path / 'tensor.pt'
    torch.save(torch.ones(1), path)
    before = torch.serialization.get_safe_globals()
    load_checkpoint(path)
    assert torch.serialization.get_safe_globals() == before


def test_runtime_loads_cannot_bypass_wrapper():
    root = Path(__file__).resolve().parents[1]
    direct = []
    for folder in ('mip360', 'replica', 'tools', 'scripts'):
        for path in (root / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'
                    and node.func.attr == 'load'):
                    direct.append((path.relative_to(root).as_posix(), node))
    assert len(direct) == 1 and direct[0][0] == 'mip360/safe_io.py'
    assert any(k.arg == 'weights_only' and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in direct[0][1].keywords)
