"""CPU safety-policy tests plus CUDA regressions for fused finite detection."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from mip360 import primitive_gemm as gemm


@pytest.fixture(autouse=True)
def isolated_safety_counters(monkeypatch):
    monkeypatch.setattr(gemm, "_SAFETY_STATS", {key: 0 for key in gemm._SAFETY_STATS})


def _csr(counts, device="cpu"):
    counts = torch.tensor(counts, device=device, dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    records = int(offsets[-1])
    return SimpleNamespace(offsets=offsets, x6=torch.zeros(records, 6, device=device),
                           grad4=torch.ones(records, 4, device=device), relu_masks=None)


def test_group_layout_carries_exact_segment_boundaries():
    counts = [0, 1, 511, 512, 513, 1024]
    csr = _csr(counts)
    groups = gemm.prepare_groups(csr, 512)
    expected_primitives, expected_starts = [], []
    start = 0
    for p, count in enumerate(counts):
        for local in range(0, count, 512):
            expected_primitives.append(p)
            expected_starts.append(start + local)
        start += count
    assert groups.segment_size == 512
    assert groups.primitives.tolist() == expected_primitives
    assert groups.starts.tolist() == expected_starts
    assert groups.offsets.tolist() == [0, 0, 1, 2, 3, 5, 7]
    assert gemm.resolve_groups(csr, 512, groups) is groups


@pytest.mark.parametrize("segment_size", [256, 1024])
def test_group_layout_rejects_segment_size_mismatch(segment_size):
    csr = _csr([513])
    groups = gemm.prepare_groups(csr, 512)
    with pytest.raises(ValueError, match="Prepared groups use segment_size=512"):
        gemm.resolve_groups(csr, segment_size, groups)


def test_group_layout_rejects_raw_tuple_and_other_csr():
    csr = _csr([1, 2])
    groups = gemm.prepare_groups(csr)
    with pytest.raises(TypeError, match="GroupLayout"):
        gemm.resolve_groups(csr, 512, tuple(groups))
    with pytest.raises(ValueError, match="different CSR"):
        gemm.resolve_groups(_csr([1, 2]), 512, groups)


@pytest.mark.parametrize("tensor", ["source_offsets", "primitives", "starts", "offsets"])
def test_group_layout_rejects_inplace_mutation(tensor):
    csr = _csr([1, 2])
    groups = gemm.prepare_groups(csr)
    getattr(groups, tensor)[0] += 1
    with pytest.raises(ValueError, match="modified in place"):
        gemm.resolve_groups(csr, 512, groups)


@pytest.mark.parametrize("segment_size", [0, -1, 1.5, True])
def test_prepare_groups_rejects_invalid_segment_size(segment_size):
    with pytest.raises(ValueError, match="positive integer"):
        gemm.prepare_groups(_csr([1]), segment_size)


def test_prepared_groups_reject_float_segment_size():
    csr = _csr([1])
    groups = gemm.prepare_groups(csr)
    with pytest.raises(ValueError, match="positive integer"):
        gemm.resolve_groups(csr, 512.0, groups)


def test_feature_consumer_rejects_mismatched_layout_before_launch():
    from mip360.record_feature_loss import feature_weight_gradients
    csr = _csr([1])
    groups = gemm.prepare_groups(csr, 512)
    records = (None, None, torch.zeros(1, dtype=torch.int32), csr.x6,
               torch.zeros(1, 8), None, torch.zeros(1, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="Prepared groups use segment_size=512"):
        feature_weight_gradients(records, csr, torch.zeros(1, 1, 1, dtype=torch.int8),
                                 torch.zeros(1, 1), 1, 1, groups=groups, segment_size=1024)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_finite_path_returns_original_result_unchanged(dtype):
    original = torch.tensor([1.25, -2.0, 0.0])
    calls = []

    def launch(operand_dtype, scale):
        calls.append((operand_dtype, scale))
        return original, torch.tensor(0, dtype=torch.int32)

    result = gemm._checked_grouped_vjp(launch, dtype, 4096.0)
    assert result is original
    assert calls == [(dtype, 4096.0)]
    assert gemm.get_grouped_vjp_safety_stats() == {
        "calls": 1, "mixed_calls": int(dtype != torch.float32),
        "fp32_fallbacks": 0, "fp32_nonfinite_failures": 0}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nonfinite_mixed_result_retries_unscaled_fp32_and_reports(dtype):
    expected = torch.tensor([3.0])
    calls = []

    def launch(operand_dtype, scale):
        calls.append((operand_dtype, scale))
        if operand_dtype == torch.float32:
            return expected, torch.tensor(0, dtype=torch.int32)
        return torch.tensor([float("inf")]), torch.tensor(1, dtype=torch.int32)

    with pytest.warns(RuntimeWarning, match="retrying with IEEE FP32 before Adam"):
        result = gemm._checked_grouped_vjp(launch, dtype, 4096.0)
    assert result is expected
    assert calls == [(dtype, 4096.0), (torch.float32, 1.0)]
    stats = gemm.get_grouped_vjp_safety_stats()
    assert stats["fp32_fallbacks"] == 1
    assert stats["fp32_nonfinite_failures"] == 0


def test_nonfinite_fp32_raises_without_returning_gradients():
    calls = []

    def launch(operand_dtype, scale):
        calls.append((operand_dtype, scale))
        return torch.tensor([float("nan")]), torch.tensor(1, dtype=torch.int32)

    with pytest.raises(FloatingPointError, match="Adam must not step"):
        gemm._checked_grouped_vjp(launch, torch.float32, 1.0)
    assert calls == [(torch.float32, 1.0)]
    assert gemm.get_grouped_vjp_safety_stats()["fp32_nonfinite_failures"] == 1


def test_nonfinite_fp32_retry_raises_and_counts_both_events():
    def launch(operand_dtype, scale):
        return torch.tensor([float("inf")]), torch.tensor(1, dtype=torch.int32)

    with pytest.warns(RuntimeWarning, match="FP32 fallbacks=1"):
        with pytest.raises(FloatingPointError, match="Adam must not step"):
            gemm._checked_grouped_vjp(launch, torch.float16, 4096.0)
    stats = gemm.get_grouped_vjp_safety_stats(reset=True)
    assert stats["fp32_fallbacks"] == stats["fp32_nonfinite_failures"] == 1
    assert all(value == 0 for value in gemm.get_grouped_vjp_safety_stats().values())


@pytest.fixture
def cuda_vjp_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for actual reduction/VJP regressions")
    csr = _csr([33, 0, 513], device="cuda")
    csr.relu_masks = torch.full((len(csr.x6), 2), -1, device="cuda", dtype=torch.int32)
    freq = 2. ** torch.arange(6, device="cuda", dtype=torch.float32)
    attenuation = torch.ones_like(freq)
    weights = [torch.full((3, *shape), .01, device="cuda") for shape in
               [(32, 72), (32,), (32, 32), (32,), (4, 32), (4,)]]
    return csr, freq, attenuation, weights


def test_cuda_finite_flag_does_not_change_finite_result(cuda_vjp_inputs):
    csr, freq, attenuation, weights = cuda_vjp_inputs
    groups = gemm.prepare_groups(csr)
    raw, flag = gemm._launch_grouped_vjp(csr, freq, attenuation, weights, groups,
                                        torch.float16, 32, 4096.0)
    actual = gemm.grouped_weight_gradients(csr, freq, attenuation, weights, groups=groups)
    assert int(flag) == 0
    expected = [v.reshape(w.shape).contiguous() for v, w in
                zip(raw.split([2304, 32, 1024, 32, 128, 4], dim=1), weights)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert gemm.get_grouped_vjp_safety_stats()["fp32_fallbacks"] == 0


def test_cuda_actual_fp16_overflow_falls_back_to_exact_fp32(cuda_vjp_inputs):
    csr, freq, attenuation, weights = cuda_vjp_inputs
    # Finite FP32 adjoints overflow only after the default FP16 scaling.
    csr.grad4.fill_(20.0)
    masters_before = [w.clone() for w in weights]
    expected = gemm.grouped_weight_gradients(csr, freq, attenuation, weights, dtype=torch.float32)
    with pytest.warns(RuntimeWarning, match="retrying with IEEE FP32 before Adam"):
        actual = gemm.grouped_weight_gradients(csr, freq, attenuation, weights, dtype=torch.float16)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(torch.equal(a, b) for a, b in zip(weights, masters_before))
    assert all(bool(torch.isfinite(t).all()) for t in actual)
    assert gemm.get_grouped_vjp_safety_stats()["fp32_fallbacks"] == 1


def test_cuda_nonfinite_fp32_vjp_fails_closed(cuda_vjp_inputs):
    csr, freq, attenuation, weights = cuda_vjp_inputs
    csr.grad4[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Adam must not step"):
        gemm.grouped_weight_gradients(csr, freq, attenuation, weights, dtype=torch.float32)


@pytest.mark.parametrize("invalid_scale", [0, -1, float("inf"), float("nan")])
def test_cuda_invalid_loss_scale_rejected(cuda_vjp_inputs, invalid_scale):
    csr, freq, attenuation, weights = cuda_vjp_inputs
    with pytest.raises(ValueError, match="finite and strictly positive"):
        gemm.grouped_weight_gradients(csr, freq, attenuation, weights, loss_scale=invalid_scale)
