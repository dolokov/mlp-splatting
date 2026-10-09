"""512D streamed-feature numerical parity and stop-gradient regression tests.

These exercise all four 128-channel forward blocks and eight 64-channel
backward blocks independently.
"""

import pytest
import torch
import torch.nn.functional as F

from test_cuda_renderer import tile_ext, _base_args, _renderer_inputs
from scannet.primitive_csr import group_records


CHANNELS = 512


def _case(width, height, slots):
    """Real renderer records: anisotropic geometry and a nonconstant 72-32-32-4 MLP."""
    torch.manual_seed(29117 + slots)
    count = 17
    data = _renderer_inputs(count)
    yy, xx = torch.meshgrid(
        torch.linspace(-0.55, 0.55, height, device="cuda"),
        torch.linspace(-0.65, 0.65, width, device="cuda"), indexing="ij",
    )
    data["rays_o"] = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(width * height, device="cuda")), -1)
    data["rays_d"] = torch.tensor([0.04, -0.03, 1.0], device="cuda").repeat(width * height, 1)
    data["rays_d"] = F.normalize(data["rays_d"], dim=-1)
    # The upper-left pixel deliberately has no hits. One primitive never hits.
    data["rays_o"][0, 0] = 100.0
    data["positions"][:, :2].uniform_(-0.2, 0.2)
    data["positions"][:, 2] = torch.linspace(0.7, 1.8, count, device="cuda")
    data["positions"][-1, 0] = -100.0
    data["log_scales"].uniform_(-0.3, 0.3)
    data["quats"].normal_()
    data["quats"] = F.normalize(data["quats"], dim=-1)
    data["freq"] = torch.pow(2.0, torch.arange(6, device="cuda"))
    data["attenuation"] = torch.exp(-0.5 * data["freq"].square() * 0.01**2)
    for name, shape in {
        "W1": (count, 32, 72), "b1": (count, 32),
        "W2": (count, 32, 32), "b2": (count, 32),
        "W3": (count, 4, 32), "b3": (count, 4),
    }.items():
        data[name] = torch.randn(shape, device="cuda") * 0.04
    # Keep enough transmittance to visit many overlapping primitives.
    data["b3"][:, 3] = -2.0
    tiles = ((width + 7) // 8) * ((height + 7) // 8)
    data["candidates"] = torch.arange(count, device="cuda", dtype=torch.int32).repeat(tiles)
    data["offsets"] = torch.arange(tiles + 1, device="cuda", dtype=torch.int32) * count
    shape = (count, CHANNELS) if slots == 1 else (count, slots, CHANNELS)
    attributes = torch.randn(shape, device="cuda") * 0.2
    # Distinct scales/signs across all channel blocks prevent accidental broadcast
    # or a missing later block from hiding in a global-average check.
    channel = torch.arange(CHANNELS, device="cuda")
    scale = (1.0 + channel // 128) * torch.where(channel % 3 == 0, -1.0, 1.0)
    attributes = (attributes * scale).contiguous()
    return data, attributes


def _resize(features, width, height, target_width, target_height):
    return F.interpolate(
        features.view(height, width, CHANNELS).permute(2, 0, 1).unsqueeze(0),
        size=(target_height, target_width), mode="bilinear", align_corners=False,
    ).squeeze(0).permute(1, 2, 0)


def _mixed_target(prediction, dtype):
    """Stable mixed L1 signs, including checkerboard spatial cancellation."""
    height, width, channels = prediction.shape
    yy = torch.arange(height, device=prediction.device)[:, None, None]
    xx = torch.arange(width, device=prediction.device)[None, :, None]
    cc = torch.arange(channels, device=prediction.device)[None, None, :]
    sign = torch.where((xx + 3 * yy + cc + cc // 128) % 5 < 2, -1.0, 1.0)
    magnitude = 0.12 + (cc % 13) * 0.01
    # Keep residuals comfortably away from zero even after half target rounding.
    return (prediction.detach() + sign * magnitude).to(dtype).contiguous()


def _record_framebuffer(records, attributes, width, height):
    """Independent autograd oracle: only feature attributes receive gradients."""
    _, _, ids, _, raw, _, gates, pixel_offsets = records
    slots = 1 if attributes.ndim == 2 else attributes.shape[1]
    if slots == 1:
        mixed = attributes[ids.long()]
    else:
        slot_values = attributes[ids.long()]
        residual_slots = slot_values[:, 1:] if slots == 5 else slot_values
        mixed = (residual_slots * gates.detach().unsqueeze(-1)).sum(dim=1)
        if slots == 5:
            mixed = mixed + slot_values[:, 0]
    weights = (raw[:, 4] * raw[:, 5]).detach()
    pixels = torch.repeat_interleave(
        torch.arange(width * height, device=ids.device),
        pixel_offsets.diff(), output_size=len(ids),
    )
    framebuffer = attributes.new_zeros((width * height, CHANNELS))
    return framebuffer.index_add(0, pixels, mixed * weights[:, None])


def _assert_channelwise_vjp(actual, expected, *, mixed):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    # Inspect each forward channel block separately so an error confined to
    # channels 128:512 cannot be diluted by correct low channels.
    for first in range(0, CHANNELS, 128):
        a = actual[..., first:first + 128]
        e = expected[..., first:first + 128]
        relative = (a - e).norm() / e.norm().clamp_min(1e-30)
        assert float(relative) < (0.003 if mixed else 5e-5), (first, float(relative))
        if not mixed:
            torch.testing.assert_close(a, e, atol=2e-8, rtol=5e-5)


@pytest.mark.parametrize("slots", [1, 4, 5])
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "size",
    [
        (33, 19, 23, 11),  # >512 records/primitive; fractional bilinear footprint
        (9, 7, 9, 7),      # identity with clamped right/bottom bilinear neighbors
        (1, 13, 1, 7),     # single-column boundary, including empty pixel
        (15, 1, 8, 1),     # single-row boundary
        (9, 7, 1, 1),      # strong downsampling to one target pixel
    ],
)
def test_512d_stream_and_feature_vjp_match_dense_oracles(tile_ext, slots, target_dtype, size):
    from scannet.record_feature_loss import streaming_feature_l1, feature_weight_gradients

    torch.backends.cuda.matmul.allow_tf32 = False
    width, height, target_width, target_height = size
    data, attributes = _case(width, height, slots)
    base = list(_base_args(data))
    grad_rgb = torch.zeros(width * height, 3, device="cuda")
    grad_t = torch.zeros(width * height, device="cuda")
    records = tile_ext.render_tiles_training_records(
        base + [grad_rgb, grad_t], width, height, 8, 0.001, 1.5, 2.0,
    )
    ids, x6, raw, masks, gates, offsets = records[2:]
    hits = offsets.diff()
    assert int(hits[0]) == 0
    assert int(hits.max()) >= 3
    assert int(torch.unique(ids).numel()) >= 3
    assert not bool((ids == len(attributes) - 1).any())
    if width == 33:
        assert int(torch.bincount(ids.long(), minlength=len(attributes)).max()) > 512
    if slots != 1:
        assert float(gates.std()) > 0.01
        torch.testing.assert_close(gates.sum(-1), torch.ones(len(ids), device="cuda"), atol=2e-6, rtol=2e-6)

    _, _, dense = tile_ext.render_tiles_forward_tilecenter_with_feat(
        *base, attributes, 1.5, 2.0, width, height, 8, 8, 0.001,
    )
    dense = dense.detach().requires_grad_()
    expected_prediction = _resize(dense, width, height, target_width, target_height)
    target = _mixed_target(expected_prediction, target_dtype)
    expected_loss = F.l1_loss(expected_prediction, target)
    expected_loss.backward()
    legacy_vjp = tile_ext.render_tiles_backward_featattr_tilecenter(
        *base, attributes, dense.grad.contiguous(), 1.5, 2.0, width, height, 8, 8, 0.001,
    )

    oracle_attributes = attributes.detach().clone().requires_grad_()
    # Leaf gates/raw with grad enabled make the intended detach explicit in the
    # independent oracle. Under the release contract neither is differentiated.
    oracle_raw = raw.detach().clone().requires_grad_()
    oracle_gates = gates.detach().clone().requires_grad_()
    oracle_records = (*records[:4], oracle_raw, masks, oracle_gates, offsets)
    oracle_dense = _record_framebuffer(oracle_records, oracle_attributes, width, height)
    oracle_prediction = _resize(oracle_dense, width, height, target_width, target_height)
    oracle_loss = F.l1_loss(oracle_prediction, target)
    oracle_loss.backward()
    assert oracle_raw.grad is None
    assert oracle_gates.grad is None
    torch.testing.assert_close(oracle_prediction, expected_prediction, atol=4e-6, rtol=4e-6)
    _assert_channelwise_vjp(legacy_vjp, oracle_attributes.grad, mixed=False)

    loss, signs, prediction = streaming_feature_l1(records, attributes, target, width, height, debug=True)
    torch.testing.assert_close(prediction, expected_prediction, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(loss, expected_loss, atol=2e-7, rtol=2e-6)
    expected_signs = (expected_prediction - target).sign().to(torch.int8)
    torch.testing.assert_close(signs, expected_signs, atol=0, rtol=0)
    for first in range(0, CHANNELS, 128):
        assert torch.equal(torch.unique(signs[..., first:first + 128]), torch.tensor([-1, 1], device="cuda", dtype=torch.int8))

    csr = group_records(ids, x6, torch.zeros((len(ids), 4), device="cuda"), len(attributes), masks)
    # 32 forces many independently reduced partials; 512 is the training default.
    for segment in (32, 512):
        for mixed in (False, True):
            actual = feature_weight_gradients(
                records, csr, signs, attributes, width, height, mixed=mixed, segment_size=segment,
            )
            _assert_channelwise_vjp(actual, legacy_vjp, mixed=mixed)
            _assert_channelwise_vjp(actual, oracle_attributes.grad, mixed=mixed)
            assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.parametrize("slots", [4, 5])
@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_512d_scene_keeps_feature_stop_gradient_and_rgb_d4(tile_ext, monkeypatch, slots, precision):
    from scannet import cumodel, csr_training

    monkeypatch.setattr(cumodel, "_tile_ext", tile_ext)
    monkeypatch.setattr(cumodel, "_tile_ext_key", cumodel._tile_ext_build_config(8)[0])
    monkeypatch.setenv("MLPSP_CSR_MLP_PRECISION", precision)
    torch.backends.cuda.matmul.allow_tf32 = False
    width, height, target_width, target_height = 13, 9, 8, 6
    data, attributes = _case(width, height, slots)
    scene = cumodel.NeuralPrimitiveSceneTileCUDA(
        num_primitives=len(attributes), hidden_dim=32, pe_frequencies=6,
        feature_dim_low=CHANNELS, feature_slots=slots,
    ).cuda()
    scene.set_image_size(width, height)
    scene.tile_order_mode_override = "tilecenter"
    with torch.no_grad():
        for name in ("positions", "log_scales", "quats", "W1", "b1", "W2", "b2", "W3", "b3"):
            getattr(scene, name).copy_(data[name])
        scene.feat_attr.copy_(attributes)
    options = dict(
        return_feature=True, score_threshold=0.001, bg_color=0.3,
        culled_prim_indices=data["candidates"], tile_offsets=data["offsets"],
    )
    with torch.no_grad():
        _, feature = scene(data["rays_o"], data["rays_d"], **options)
        target = _mixed_target(_resize(feature, width, height, target_width, target_height), torch.float16)
    rgb_adjoint = torch.linspace(-0.5, 0.7, width * height * 3, device="cuda").reshape(-1, 3) / (width * height)

    observed_d4 = []
    original_group = csr_training.group_records

    def capture_group(ids, x6, d4, n, masks=None):
        observed_d4.append(d4.detach().clone())
        return original_group(ids, x6, d4, n, masks)

    monkeypatch.setattr(csr_training, "group_records", capture_group)
    results = {}
    for mode in ("feature_only", "rgb_only", "joint"):
        for backend in ("legacy", "csr"):
            scene.zero_grad(set_to_none=True)
            rgb, second = scene(
                data["rays_o"], data["rays_d"], **options,
                feature_target=target if backend == "csr" else None,
            )
            feature_loss = second if backend == "csr" else F.l1_loss(
                _resize(second, width, height, target_width, target_height), target,
            )
            rgb_loss = (rgb * rgb_adjoint).sum()
            loss = 0.7 * feature_loss if mode == "feature_only" else rgb_loss
            if mode == "joint":
                loss = loss + 0.7 * feature_loss
            loss.backward()
            results[mode, backend] = (
                loss.detach(),
                {name: None if value.grad is None else value.grad.detach().clone()
                 for name, value in scene.named_parameters()},
            )
        expected_loss, expected = results[mode, "legacy"]
        actual_loss, actual = results[mode, "csr"]
        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=2e-6)
        for name, grad in expected.items():
            if grad is None:
                assert actual[name] is None, (mode, name)
            elif name == "feat_attr":
                _assert_channelwise_vjp(actual[name], grad, mixed=precision != "float32")
            elif precision == "float32" or name in {"positions", "log_scales", "quats"}:
                torch.testing.assert_close(actual[name], grad, atol=5e-6, rtol=5e-5)
            else:
                relative = (actual[name] - grad).norm() / grad.norm().clamp_min(1e-30)
                assert float(relative) < 0.003, (mode, name, float(relative))

    assert len(observed_d4) == 3
    assert torch.count_nonzero(observed_d4[0]) == 0  # feature-only cannot affect MLP outputs
    assert torch.count_nonzero(observed_d4[1]) > 0  # non-vacuous RGB backward
    torch.testing.assert_close(observed_d4[1], observed_d4[2], atol=0, rtol=0)
    for backend in ("legacy", "csr"):
        feature_gradients = results["feature_only", backend][1]
        rgb_gradients = results["rgb_only", backend][1]
        joint_gradients = results["joint", backend][1]
        assert feature_gradients["feat_attr"] is not None
        assert torch.count_nonzero(feature_gradients["feat_attr"]) > 0
        assert rgb_gradients["feat_attr"] is None
        for name, grad in feature_gradients.items():
            if name != "feat_attr":
                assert grad is None, (backend, "feature loss changed detached geometry/gates/MLP", name)
                if rgb_gradients[name] is not None:
                    torch.testing.assert_close(joint_gradients[name], rgb_gradients[name], atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(joint_gradients["feat_attr"], feature_gradients["feat_attr"], atol=2e-8, rtol=5e-5)
