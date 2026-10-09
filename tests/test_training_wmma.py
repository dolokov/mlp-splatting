"""CUDA training-record and mixed-precision numerical regression tests.

Integer record topology and FP32 geometry are exact contracts. Learned
mixed-precision outputs use explicit numerical tolerances, and backward uses
the masks produced by its own forward pass.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import re

import pytest
import torch
import torch.nn.functional as F

from test_cuda_renderer import tile_ext, _base_args, _renderer_inputs


PRECISIONS = ("float16", "bfloat16")
WEIGHTS = ("W1", "b1", "W2", "b2", "W3", "b3")
STATS = ("calls", "frame_fallbacks", "fp32_nonfinite_failures",
         "count_emit_failures", "capacity_failures")
# The last counter covers kernel hit-capacity failures. Host candidate/input
# validation rejects before calls and before all five kernel-related counters.


def _stats(ext):
    values = ext.training_wmma_safety_stats()
    assert len(values) == len(STATS)
    return dict(zip(STATS, map(int, values)))


def _delta(ext, before):
    after = _stats(ext)
    result = {key: after[key] - before[key] for key in STATS}
    assert all(value >= 0 for value in result.values())
    return result


def _case(n=13, width=9, height=7, *, constant=False, mosaic=False):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("training WMMA requires SM80+")
    torch.manual_seed(42157 + n + width + height)
    data = _renderer_inputs(n)
    yy, xx = torch.meshgrid(torch.linspace(-.45, .45, height, device="cuda"),
                            torch.linspace(-.55, .55, width, device="cuda"), indexing="ij")
    data["rays_o"] = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(width * height, device="cuda")), -1)
    data["rays_d"] = torch.tensor([.025, -.015, 1.], device="cuda").repeat(width * height, 1)
    if mosaic:
        # Different origins AND directions inside a tile: no common-camera
        # assumption, including a mosaic boundary not aligned to a tile edge.
        ix = torch.arange(width * height, device="cuda")
        quadrant = ((ix % width >= width // 2).long()
                    + 2 * (ix // width >= height // 2).long())
        cameras = torch.tensor([[0., 0., 0.], [.19, 0., -.12],
                                [0., -.17, .09], [-.14, .13, -.08]], device="cuda")
        directions = torch.tensor([[0., 0., 1.], [-.09, .04, 1.],
                                   [.07, .08, 1.], [.12, -.05, 1.]], device="cuda")
        data["rays_o"] += cameras[quadrant]
        data["rays_d"] = directions[quadrant]
    data["rays_d"] = F.normalize(data["rays_d"], dim=-1).contiguous()
    data["positions"][:, :2].uniform_(-.2, .2)
    data["positions"][:, 2] = torch.linspace(.75, 1.75, n, device="cuda")
    data["log_scales"].uniform_(-.45, .15)
    data["quats"] = F.normalize(torch.randn(n, 4, device="cuda"), dim=-1).contiguous()
    data["freq"] = 2. ** torch.arange(6, device="cuda", dtype=torch.float32)
    data["attenuation"] = torch.exp(-.5 * data["freq"].square() * .0001)
    shapes = ((32, 72), (32,), (32, 32), (32,), (4, 32), (4,))
    for name, shape in zip(WEIGHTS, shapes):
        data[name] = torch.randn((n, *shape), device="cuda") * .035
    data["b3"][:, 3] = -2.  # Far from termination; topology must match FP32.
    if constant:
        for name in WEIGHTS:
            data[name].zero_()
        data["b3"][:, 0] = torch.linspace(-1., 1., n, device="cuda")
        data["b3"][:, 1] = -.25
        data["b3"][:, 2] = .5
        data["b3"][:, 3] = -2.
    tiles = ((width + 7) // 8) * ((height + 7) // 8)
    data["candidates"] = torch.arange(n, device="cuda", dtype=torch.int32).repeat(tiles)
    data["offsets"] = torch.arange(tiles + 1, device="cuda", dtype=torch.int32) * n
    data["grad_rgb"] = torch.randn(width * height, 3, device="cuda") * .1
    data["grad_t"] = torch.randn(width * height, device="cuda") * .1
    return data


def _args(data):
    return list(_base_args(data)) + [data["grad_rgb"], data["grad_t"]]


def _render(ext, data, width, height, precision, *, threshold=.001):
    return ext.render_tiles_training_records_wmma(
        _args(data), width, height, 8, threshold, 1.5, 2., precision)


def _reference(ext, data, width, height, *, threshold=.001):
    return ext.render_tiles_training_records(
        _args(data), width, height, 8, threshold, 1.5, 2.)


def _schema(records, data, width, height):
    assert len(records) == 8
    rgb, alpha, ids, x6, raw, masks, gates, offsets = records
    count = len(ids)
    shapes = ((width * height, 3), (width * height, 1), (count,),
              (count, 6), (count, 8), (count, 2), (count, 4), (width * height + 1,))
    dtypes = (torch.float32, torch.float32, torch.int32, torch.float32,
              torch.float32, torch.int32, torch.float32, torch.int64)
    for value, shape, dtype in zip(records, shapes, dtypes):
        assert tuple(value.shape) == shape
        assert value.dtype == dtype
        assert value.device == data["positions"].device
        assert value.is_contiguous()
        assert bool(torch.isfinite(value).all())
    assert int(offsets[0]) == 0 and int(offsets[-1]) == count
    assert bool((offsets.diff() >= 0).all())
    assert bool(((ids >= 0) & (ids < len(data["positions"]))).all())
    if count:
        assert bool(((raw[:, 4:6] >= 0) & (raw[:, 4:6] <= 1)).all())
        torch.testing.assert_close(gates.sum(-1), torch.ones(count, device=ids.device), atol=2e-6, rtol=2e-6)
        pixels = torch.repeat_interleave(torch.arange(width * height, device=ids.device),
                                        offsets.diff(), output_size=count)
        torch.testing.assert_close(x6[:, :3], data["rays_d"][pixels], atol=0, rtol=0)


def _record_parity(actual, expected, *, precision, exact=False):
    # Geometry, order, and gate computation retain the FP32 expression path.
    for index in (2, 3, 6, 7):
        torch.testing.assert_close(actual[index], expected[index], atol=0, rtol=0)
    torch.testing.assert_close(actual[4][:, 6], expected[4][:, 6], atol=0, rtol=0)
    if exact:
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    else:
        atol = .003 if precision == "bfloat16" else .0005
        for index in (0, 1, 4):
            torch.testing.assert_close(actual[index], expected[index], atol=atol, rtol=atol)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("width,height,mosaic", [(1, 1, False), (8, 8, False),
                                                   (13, 11, False), (17, 9, True)])
@pytest.mark.parametrize("constant", [False, True])
def test_training_wmma_schema_geometry_and_full_records(tile_ext, precision, width, height, mosaic, constant):
    data = _case(width=width, height=height, constant=constant, mosaic=mosaic)
    originals = [(tensor.clone(), tensor._version) for tensor in _args(data)]
    expected = _reference(tile_ext, data, width, height)
    before = _stats(tile_ext)
    actual = _render(tile_ext, data, width, height, precision)
    _schema(actual, data, width, height)
    assert len(actual[2]) > 0
    _record_parity(actual, expected, precision=precision, exact=constant)
    assert _delta(tile_ext, before) == dict(zip(STATS, (1, 0, 0, 0, 0)))
    for tensor, (saved, version) in zip(_args(data), originals):
        torch.testing.assert_close(tensor, saved, atol=0, rtol=0)
        assert tensor._version == version


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("kind", ["empty_scene", "empty_lists", "no_hits"])
def test_training_wmma_empty_outputs_are_initialized(tile_ext, precision, kind):
    width, height = 13, 11
    data = _case(0 if kind == "empty_scene" else 5, width, height)
    if kind == "empty_lists":
        data["candidates"] = data["candidates"].new_empty(0)
        data["offsets"].zero_()
    elif kind == "no_hits":
        data["rays_o"][:, 0] = 100.
    actual = _render(tile_ext, data, width, height, precision)
    _schema(actual, data, width, height)
    assert len(actual[2]) == 0
    for index in (0, 1, 7):
        assert bool((actual[index] == 0).all())
    _record_parity(actual, _reference(tile_ext, data, width, height), precision=precision, exact=True)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_training_wmma_stable_depth_ties_and_divergent_early_stop(tile_ext, precision):
    data = _case(12, 8, 8, constant=True)
    data["positions"].zero_()
    data["positions"][:, 2] = 1.
    data["log_scales"].zero_()
    data["quats"].zero_()
    data["quats"][:, 0] = 1.
    data["rays_o"].zero_()
    data["rays_o"][:, 0] = 100.
    selected = [0, 15, 16, 31, 32, 47, 48, 63]
    data["rays_o"][selected] = 0.
    data["rays_d"].zero_()
    data["rays_d"][:, 2] = 1.
    data["candidates"] = data["candidates"].flip(0).contiguous()
    data["b3"][:, 3] = 8.  # Each active pixel stops after the first tied primitive.
    actual = _render(tile_ext, data, 8, 8, precision)
    _schema(actual, data, 8, 8)
    _record_parity(actual, _reference(tile_ext, data, 8, 8), precision=precision, exact=True)
    counts = torch.zeros(64, device="cuda", dtype=torch.int64)
    counts[selected] = 1
    torch.testing.assert_close(actual[7].diff(), counts, atol=0, rtol=0)
    assert actual[2].tolist() == [11] * len(selected)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_training_wmma_masks_follow_actual_mixed_forward(tile_ext, precision):
    from mip360.primitive_csr import group_records
    from mip360.primitive_gemm import grouped_weight_gradients

    data = _case(1, 1, 1, constant=True)
    epsilon = 2. ** (-12 if precision == "float16" else -9)
    data["b1"][0, 1] = 1. + epsilon
    data["W2"][0] = torch.eye(32, device="cuda")
    data["b2"][0, 1] = -(1. + epsilon / 2.)
    data["W3"][0, :, 1] = .125
    # h1 rounds to 1 in the selected operand dtype. Thus actual h2 is
    # negative, although the scalar FP32 h2 remains positive. No denormals.
    actual = _render(tile_ext, data, 1, 1, precision)
    scalar = _reference(tile_ext, data, 1, 1)
    assert len(actual[2]) == 1
    assert actual[5].tolist() == [[2, 0]]
    assert scalar[5].tolist() == [[2, 2]]
    # This near-zero hidden unit hardly changes the image; image-only parity
    # cannot establish the correct training derivative.
    torch.testing.assert_close(actual[0], scalar[0], atol=.0001, rtol=.0001)
    csr = group_records(actual[2], actual[3], torch.ones(1, 4, device="cuda"), 1, actual[5])
    gradients = grouped_weight_gradients(csr, data["freq"], data["attenuation"],
                                        [data[name] for name in WEIGHTS], dtype=torch.float32)
    # FP32 activation recomputation sees positive h2 here. Only using the
    # actual saved mask correctly prevents its adjoint reaching W1/W2/b1/b2.
    for gradient in gradients[:4]:
        assert bool((gradient == 0).all())


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("bias_offset", [-.001, 0., .001])
def test_training_wmma_count_emit_agree_at_transmittance_boundary(tile_ext, precision, bias_offset):
    data = _case(2, 1, 1, constant=True)
    data["positions"].zero_()
    data["positions"][:, 2] = torch.tensor([1., 2.], device="cuda")
    data["rays_o"].zero_()
    data["rays_d"].zero_()
    data["rays_d"][:, 2] = 1.
    data["b3"][0, 3] = math.log(999.) + bias_offset
    before = _stats(tile_ext)
    actual = _render(tile_ext, data, 1, 1, precision)
    _schema(actual, data, 1, 1)
    _record_parity(actual, _reference(tile_ext, data, 1, 1), precision=precision, exact=True)
    # Evaluate the actual FP32 update, not a double-precision estimate of the
    # boundary; exact equality to 1e-3 must remain an active ray.
    still_active = bool((1. - actual[4][0, 5]) >= .001)
    assert actual[7].tolist() == [0, 2 if still_active else 1]
    assert _delta(tile_ext, before) == dict(zip(STATS, (1, 0, 0, 0, 0)))


def _compiled_capacity(ext, name):
    build = Path(ext.__file__).with_name("build.ninja")
    if build.is_file():
        values = re.findall(r"-D\s*" + re.escape(name) + r"=(\d+)\b", build.read_text())
        if values:
            return int(values[-1])
    default = Path(__file__).parents[1] / "mip360" / "cuda" / "np_tile_kernels.cu"
    source = Path(os.environ.get("MLPSP_CUDA_TEST_SOURCE", default))
    match = re.search(r"^\s*#\s*define\s+" + re.escape(name) + r"\s+(\d+)\b",
                      source.read_text(), re.MULTILINE)
    assert match, f"Cannot determine compiled {name}"
    return int(match.group(1))


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("kind", ["candidate", "hit"])
def test_training_wmma_capacity_raises_not_fallback_or_truncation(tile_ext, precision, kind):
    name = "MLPSP_MAX_TILE_PRIMS" if kind == "candidate" else "MLPSP_TILECENTER_MAX_HITS"
    capacity = _compiled_capacity(tile_ext, name)
    data = _case(capacity + 1, 1, 1, constant=True)
    data["positions"].zero_()
    data["positions"][:, 2] = 1.
    data["log_scales"].zero_()
    data["quats"].zero_()
    data["quats"][:, 0] = 1.
    data["rays_o"].zero_()
    data["rays_d"].zero_()
    data["rays_d"][:, 2] = 1.
    data["b3"][:, 3] = -20. if kind == "hit" else 8.
    before = _stats(tile_ext)
    with pytest.raises(RuntimeError, match="capacity"):
        _render(tile_ext, data, 1, 1, precision)
    delta = _delta(tile_ext, before)
    expected = (0, 0, 0, 0, 0) if kind == "candidate" else (1, 0, 0, 0, 1)
    assert delta == dict(zip(STATS, expected))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_training_wmma_just_below_hit_capacity_keeps_every_record(tile_ext, precision):
    count = _compiled_capacity(tile_ext, "MLPSP_TILECENTER_MAX_HITS") - 1
    data = _case(count, 1, 1, constant=True)
    data["positions"].zero_()
    data["positions"][:, 2] = 1.
    data["rays_o"].zero_()
    data["rays_d"].zero_()
    data["rays_d"][:, 2] = 1.
    data["log_scales"].zero_()
    data["b3"][:, 3] = -20.
    actual = _render(tile_ext, data, 1, 1, precision)
    assert actual[7].tolist() == [0, count]
    _record_parity(actual, _reference(tile_ext, data, 1, 1), precision=precision, exact=True)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("kind", ["weight", "hidden"])
def test_training_wmma_numeric_failure_retries_entire_fp32_frame(tile_ext, precision, kind):
    data = _case(3, 9, 7, constant=True, mosaic=True)
    large = 70000. if precision == "float16" else 3.4e38
    if kind == "weight":
        # Zero ray-x phase: finite scalar product large*0; mixed weight cast
        # overflows. W2/W3 stay zero, so FP32 fallback records remain finite.
        data["rays_d"][:, 0] = 0.
        data["W1"][0, 0, 0] = large
    else:
        data["b1"][0, 0] = large
    expected = _reference(tile_ext, data, 9, 7)
    before = _stats(tile_ext)
    actual = _render(tile_ext, data, 9, 7, precision)
    _schema(actual, data, 9, 7)
    _record_parity(actual, expected, precision=precision, exact=True)
    assert _delta(tile_ext, before) == dict(zip(STATS, (1, 1, 0, 0, 0)))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_training_wmma_nonfinite_fp32_retry_raises_without_records(tile_ext, precision):
    data = _case(1, 1, 1, constant=True)
    data["b1"][0, 0] = 3.e38
    data["W2"][0, 0, 0] = 3.e38
    before = _stats(tile_ext)
    with pytest.raises(RuntimeError, match="[Nn]onfinite|finite"):
        _render(tile_ext, data, 1, 1, precision)
    delta = _delta(tile_ext, before)
    assert delta["frame_fallbacks"] == 1
    assert delta["fp32_nonfinite_failures"] == 1
    assert delta["count_emit_failures"] == delta["capacity_failures"] == 0


@pytest.mark.parametrize("kind", ["half_master", "half_geometry", "invalid_id", "invalid_offset",
                                 "tile16", "precision", "shape", "nonfinite_unused_geometry"])
def test_training_wmma_rejects_invalid_contract(tile_ext, kind):
    data = _case(3, 9, 7)
    tile, precision = 8, "float16"
    if kind == "half_master":
        data["W1"] = data["W1"].half()
    elif kind == "half_geometry":
        data["log_scales"] = data["log_scales"].half()
    elif kind == "invalid_id":
        data["candidates"][-1] = 3
    elif kind == "invalid_offset":
        data["offsets"][0] = 1
    elif kind == "tile16":
        tile = 16
    elif kind == "precision":
        precision = "float32"
    elif kind == "shape":
        data["W1"] = data["W1"][:, :, :71].contiguous()
    else:
        data["positions"][-1, 0] = 100.
        data["log_scales"][-1, 1] = float("nan")
    before = _stats(tile_ext)
    with pytest.raises(RuntimeError):
        tile_ext.render_tiles_training_records_wmma(_args(data), 9, 7, tile, .001, 1.5, 2., precision)
    assert _delta(tile_ext, before)["frame_fallbacks"] == 0


def test_training_wmma_per_call_stream_and_record_lifetime(tile_ext):
    data = _case(4, 9, 7)
    original = _render(tile_ext, data, 9, 7, "float16")
    saved = [value.clone() for value in original]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # In-place master mutation must affect this call, including geometry.
        data["b3"][:, 0].add_(.7)
        data["log_scales"].add_(.15)
        changed = _render(tile_ext, data, 9, 7, "float16")
        expected = _reference(tile_ext, data, 9, 7)
    torch.cuda.current_stream().wait_stream(stream)
    _record_parity(changed, expected, precision="float16")
    assert not torch.equal(changed[0], original[0])
    for value, snapshot in zip(original, saved):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


def _fp32_parameter_vjp_with_saved_masks(csr, data):
    """Independent per-primitive PyTorch oracle for the existing VJP contract.

    The input/geometry stop-gradient remains intentional. This oracle tests the
    FP32 backward option on actual WMMA records, including its actual masks;
    it does not assert that mixed forward has FP32 training trajectories.
    """
    from mip360.cumodel import _encode_x6
    weights = [data[name] for name in WEIGHTS]
    result = [torch.zeros_like(value) for value in weights]
    bits = torch.arange(32, device=csr.x6.device, dtype=torch.int64)
    for p in range(len(weights[0])):
        begin, end = map(int, csr.offsets[p:p + 2])
        if begin == end:
            continue
        x = _encode_x6(csr.x6[begin:end], data["freq"], data["attenuation"])
        w1, b1, w2, b2, w3, _ = [value[p] for value in weights]
        h1 = torch.relu(x @ w1.T + b1)
        h2 = torch.relu(h1 @ w2.T + b2)
        masks = csr.relu_masks[begin:end].long()
        m1 = ((masks[:, :1] >> bits) & 1).bool()
        m2 = ((masks[:, 1:] >> bits) & 1).bool()
        d3 = csr.grad4[begin:end]
        d2 = (d3 @ w3) * m2
        d1 = (d2 @ w2) * m1
        grads = (d1.T @ x, d1.sum(0), d2.T @ h1, d2.sum(0), d3.T @ h2, d3.sum(0))
        for output, grad in zip(result, grads):
            output[p] = grad
    return result


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("constant", [False, True])
def test_training_wmma_saved_records_feed_reverse_and_grouped_parameter_vjp(tile_ext, precision, constant):
    from mip360.primitive_csr import group_records
    from mip360.primitive_gemm import grouped_weight_gradients

    torch.backends.cuda.matmul.allow_tf32 = False
    data = _case(9, 9, 7, constant=constant, mosaic=True)
    # Constant MLP makes the complete compositing/geometry oracle exact while
    # nonzero hidden biases still exercise every parameter-gradient shape.
    data["b1"].fill_(.125)
    data["b2"].fill_(.25)
    if not constant:
        # Strictly positive hidden activations avoid conflating permitted
        # forward rounding/mask changes with an incorrect downstream VJP.
        for name in ("W1", "W2", "W3"):
            data[name].mul_(.1)
    records = _render(tile_ext, data, 9, 7, precision)
    expected = tile_ext.render_tiles_backward_tilecenter_full(*_args(data), 9, 7, 8, 8, .001)
    d4, gp, gs, gq = tile_ext.primitive_records_reverse(
        records[2], records[7], records[3], records[4], data["log_scales"], data["quats"],
        data["grad_rgb"], data["grad_t"])
    forward_tolerance = 5e-5 if constant else (.003 if precision == "bfloat16" else .0005)
    for actual, reference in zip((gp, gs, gq), expected[6:]):
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=forward_tolerance)
    csr = group_records(records[2], records[3], d4, len(data["positions"]), records[5])
    gradients = grouped_weight_gradients(csr, data["freq"], data["attenuation"],
                                        [data[name] for name in WEIGHTS], dtype=torch.float32)
    oracle = _fp32_parameter_vjp_with_saved_masks(csr, data)
    for actual, reference, scalar in zip(gradients, oracle, expected[:6]):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=5e-5)
        torch.testing.assert_close(actual, scalar, atol=3e-5, rtol=forward_tolerance)
    if not constant:
        assert all(bool((value != 0).any()) for value in gradients)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("branch", ["feature", "rgb", "joint"])
def test_training_wmma_512d_feature_and_input_stop_gradient(tile_ext, monkeypatch, precision, branch):
    from mip360 import cumodel
    from mip360.csr_training import CSRTrainingFunction
    from test_record_feature_512 import _record_framebuffer, _mixed_target, _resize

    class ForwardProxy:
        def __getattr__(self, name):
            return getattr(tile_ext, name)

        def render_tiles_training_records(self, *args):
            return tile_ext.render_tiles_training_records_wmma(*args, precision)

    monkeypatch.setattr(cumodel, "_tile_ext", ForwardProxy())
    monkeypatch.setattr(cumodel, "_tile_ext_key", cumodel._tile_ext_build_config(8)[0])
    monkeypatch.setenv("MLPSP_CSR_MLP_PRECISION", "float32")
    # Exercise the actual WMMA records independently of the public environment
    # selector, which has its own dispatch tests.
    monkeypatch.setenv("MLPSP_CSR_FORWARD_BACKEND", "scalar")
    data = _case(7, 9, 7, mosaic=True)
    base = list(_base_args(data))
    for tensor in base:
        if tensor.dtype == torch.float32:
            tensor.requires_grad_()
    attributes = (torch.randn(7, 5, 512, device="cuda") * .2).requires_grad_()
    with torch.no_grad():
        records = _render(tile_ext, data, 9, 7, precision)
        prediction = _resize(_record_framebuffer(records, attributes, 9, 7), 9, 7, 5, 3)
        target = _mixed_target(prediction, torch.float32)
    rgb, feature_loss = CSRTrainingFunction.apply(*base, attributes, target,
                                                 (9, 7, .001, 1.5, 2.), None)
    loss = feature_loss if branch == "feature" else rgb.square().mean()
    if branch == "joint":
        loss = loss + feature_loss
    loss.backward()
    # No MLP-input/ray/frequency/attenuation adjoint is part of the release.
    for index in (0, 1, 7, 8):
        assert base[index].grad is None
    if branch == "feature":
        for index in (2, 3, 4, 9, 10, 11, 12, 13, 14):
            assert base[index].grad is None
    else:
        for index in (2, 3, 4, 9, 10, 11, 12, 13, 14):
            assert base[index].grad is not None
            assert bool(torch.isfinite(base[index].grad).all())
    if branch == "rgb":
        assert attributes.grad is None
    else:
        oracle_attributes = attributes.detach().clone().requires_grad_()
        oracle_raw = records[4].detach().clone().requires_grad_()
        oracle_gates = records[6].detach().clone().requires_grad_()
        oracle_records = list(records)
        oracle_records[4], oracle_records[6] = oracle_raw, oracle_gates
        dense = _resize(_record_framebuffer(oracle_records, oracle_attributes, 9, 7), 9, 7, 5, 3)
        F.l1_loss(dense, target).backward()
        assert oracle_raw.grad is None and oracle_gates.grad is None
        for first in range(0, 512, 128):
            torch.testing.assert_close(attributes.grad[..., first:first + 128],
                                       oracle_attributes.grad[..., first:first + 128], atol=2e-8, rtol=8e-5)
