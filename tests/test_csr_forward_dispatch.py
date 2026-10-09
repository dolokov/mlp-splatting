"""CPU-only dispatch tests; numeric record contracts have separate CUDA tests."""
from types import SimpleNamespace

import pytest

from mip360.csr_training import _record_forward


@pytest.mark.parametrize("value,precision", [
    (None, None), ("scalar", None), (" SCALAR ", None),
    ("wmma_fp16", "float16"), ("wmma_bf16", "bfloat16"),
])
def test_record_forward_dispatch(monkeypatch, value, precision):
    monkeypatch.delenv("MLPSP_CSR_FORWARD_BACKEND", raising=False)
    if value is not None:
        monkeypatch.setenv("MLPSP_CSR_FORWARD_BACKEND", value)
    # The backward precision flag must not silently select the actual forward.
    monkeypatch.setenv("MLPSP_CSR_MLP_PRECISION", "bfloat16")
    calls, records, base = [], object(), [object() for _ in range(17)]
    def scalar(*args):
        calls.append(("scalar", args))
        return records
    def wmma(*args):
        calls.append(("wmma", args))
        return records
    ext = SimpleNamespace(render_tiles_training_records=scalar,
                          render_tiles_training_records_wmma=wmma)
    assert _record_forward(ext, base, 638, 478, .001, 1.5, 2.) is records
    assert len(calls) == 1
    assert calls[0][0] == ("scalar" if precision is None else "wmma")
    expected = (base, 638, 478, 8, .001, 1.5, 2.)
    assert calls[0][1] == expected + (() if precision is None else (precision,))
    assert calls[0][1][0] is base


@pytest.mark.parametrize("value", ["", "fp16", "wmma", "float32", "triton", "invalid"])
def test_record_forward_rejects_unknown_backend(monkeypatch, value):
    monkeypatch.setenv("MLPSP_CSR_FORWARD_BACKEND", value)
    # An invalid selector must fail before any extension method is accessed.
    with pytest.raises(ValueError, match="MLPSP_CSR_FORWARD_BACKEND"):
        _record_forward(object(), [], 638, 478, .001, 1.5, 2.)


def test_record_forward_preserves_extension_failure(monkeypatch):
    monkeypatch.setenv("MLPSP_CSR_FORWARD_BACKEND", "wmma_fp16")
    failure = RuntimeError("hit capacity exceeded")
    def fail(*args):
        raise failure
    ext = SimpleNamespace(render_tiles_training_records_wmma=fail)
    with pytest.raises(RuntimeError) as caught:
        _record_forward(ext, [], 638, 478, .001, 1.5, 2.)
    assert caught.value is failure
