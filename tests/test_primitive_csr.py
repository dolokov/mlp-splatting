"""CUDA oracle checks for the new record pipeline, including real MLP shapes."""
import pytest
import torch
import os

from test_cuda_renderer import tile_ext, _renderer_inputs, _base_args
from scannet.primitive_csr import group_records, reference_weight_gradients


@pytest.mark.parametrize("hidden,frequencies", [(1, 1), (32, 6)])
@pytest.mark.parametrize("width,height", [(1, 1), (8, 8), (9, 7)])
def test_csr_matches_rotated_tilecenter_gradients(tile_ext, hidden, frequencies, width, height):
    torch.manual_seed(7123 + hidden + width)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 24
    data = _renderer_inputs(n)
    yy, xx = torch.meshgrid(torch.linspace(-0.35, 0.35, height, device="cuda"),
                            torch.linspace(-0.35, 0.35, width, device="cuda"), indexing="ij")
    data["rays_o"] = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(width*height, device="cuda")), -1)
    data["rays_d"] = torch.tensor([0., 0., 1.], device="cuda").repeat(width*height, 1)
    data["positions"][:, :2].uniform_(-0.3, 0.3)
    data["positions"][:, 2].uniform_(0.8, 1.2)
    data["log_scales"].uniform_(-2.0, -1.2)
    data["quats"].normal_()
    data["quats"] /= data["quats"].norm(dim=-1, keepdim=True)
    tiles = ((width+7)//8) * ((height+7)//8)
    data["candidates"] = torch.arange(n, device="cuda", dtype=torch.int32).repeat(tiles)
    data["offsets"] = torch.arange(tiles+1, device="cuda", dtype=torch.int32) * n
    data["freq"] = 2. ** torch.arange(frequencies, device="cuda", dtype=torch.float32)
    data["attenuation"] = torch.exp(-0.5 * data["freq"].square() * 0.0001)
    for name, shape in {"W1": (n,hidden,12*frequencies), "b1": (n,hidden),
                        "W2": (n,hidden,hidden), "b2": (n,hidden),
                        "W3": (n,4,hidden), "b3": (n,4)}.items():
        data[name] = torch.randn(shape, device="cuda") * 0.1
    data["grad_rgb"] = torch.randn(width*height, 3, device="cuda")
    data["grad_t"] = torch.randn(width*height, device="cuda")
    args = list(_base_args(data)) + [data["grad_rgb"], data["grad_t"]]
    expected = tile_ext.render_tiles_backward_tilecenter_full(*args, width,height,8,8,0.001)
    ids, x6, grad4, gp, gs, gq, counts, masks = tile_ext.render_tiles_mlp_records(args,width,height,8,0.001)
    assert int(counts.sum()) == ids.numel()
    assert ids.numel() > 0
    assert bool(((ids >= 0) & (ids < n)).all())
    csr = group_records(ids, x6, grad4, n, masks)
    assert int(csr.offsets[-1]) == ids.numel()
    weights = [data[k] for k in ("W1","b1","W2","b2","W3","b3")]
    actual = reference_weight_gradients(csr,data["freq"],data["attenuation"],weights) + [gp,gs,gq]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=5e-4,rtol=5e-4)
    saved = tile_ext.render_tiles_training_records(args,width,height,8,.001,1.5,2.)
    rgb,alpha,saved_ids,saved_x6,raw,saved_masks,gates,pixel_offsets = saved
    rgb_ref,alpha_ref = tile_ext.render_tiles_forward_tilecenter(*args[:15],width,height,8,8,.001)
    torch.testing.assert_close(rgb,rgb_ref,atol=2.e-6,rtol=2.e-6)
    torch.testing.assert_close(alpha,alpha_ref,atol=2.e-6,rtol=2.e-6)
    torch.testing.assert_close(saved_ids,ids,atol=0,rtol=0)
    torch.testing.assert_close(saved_x6,x6,atol=0,rtol=0)
    torch.testing.assert_close(saved_masks,masks,atol=0,rtol=0)
    d4,gp2,gs2,gq2 = tile_ext.primitive_records_reverse(saved_ids,pixel_offsets,saved_x6,raw,
        data["log_scales"],data["quats"],data["grad_rgb"],data["grad_t"])
    for a,b in zip([d4,gp2,gs2,gq2],[grad4,gp,gs,gq]):
        torch.testing.assert_close(a,b,atol=5.e-5,rtol=5.e-5)
    if hidden == 32:
        from scannet.primitive_gemm import grouped_weight_gradients
        for dtype in (torch.float32,torch.float16):
            grouped = grouped_weight_gradients(csr,data["freq"],data["attenuation"],weights,dtype=dtype)
            for a,b in zip(grouped,expected[:6]):
                if dtype==torch.float32:
                    torch.testing.assert_close(a,b,atol=5e-4,rtol=5e-4)
                else:
                    assert (a-b).norm()/b.norm().clamp_min(1.e-30) < .003


def test_csr_rejects_hit_overflow(tile_ext):
    if os.environ.get("MLPSP_TILECENTER_MAX_HITS", "512") != "128":
        pytest.skip("this adversarial fixture targets the public 128-hit tile8 specialization")
    data = _renderer_inputs(130)
    data["b3"][:, 3] = -20.
    args = list(_base_args(data)) + [data["grad_rgb"], data["grad_t"]]
    with pytest.raises(RuntimeError, match="hit capacity"):
        tile_ext.render_tiles_mlp_records(args,1,1,8,0.001)


def test_saved_records_reject_configured_capacity_overflow(tile_ext):
    capacity=int(os.environ.get("MLPSP_TILECENTER_MAX_HITS","512"))
    data=_renderer_inputs(capacity+1)
    data["b3"][:,3]=-20.
    args=list(_base_args(data))+[data["grad_rgb"],data["grad_t"]]
    with pytest.raises(RuntimeError,match="hit capacity"):
        tile_ext.render_tiles_training_records(args,1,1,8,.001,1.5,2.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1., 1.e-6])
def test_grouped_gemm_vjp_matches_fp32_oracle(dtype, scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from scannet.primitive_gemm import grouped_weight_gradients
    torch.manual_seed(92)
    torch.backends.cuda.matmul.allow_tf32 = False
    counts = torch.tensor([0,1,31,32,33,511,512,513,1700,0],device="cuda")
    ids = torch.repeat_interleave(torch.arange(10,device="cuda",dtype=torch.int32),counts)
    order = torch.randperm(ids.numel(),device="cuda")
    ids = ids[order]
    x6 = torch.randn((len(ids),6),device="cuda")*.2
    d3 = torch.randn((len(ids),4),device="cuda")*scale
    csr = group_records(ids,x6,d3,10)
    freq = 2.**torch.arange(6,device="cuda",dtype=torch.float32)
    att = torch.exp(-.5*freq.square()*1.e-4)
    weights = [torch.randn((10,*shape),device="cuda")*.1 for shape in [(32,72),(32,),(32,32),(32,),(4,32),(4,)]]
    # Emulate exact FP32 activation masks saved by the record emitter.
    from scannet.cumodel import _encode_x6
    masks = torch.empty((len(ids),2),device="cuda",dtype=torch.int32)
    bit = 1 << torch.arange(32,device="cuda",dtype=torch.int64)
    for p in range(10):
        begin,end = map(int,csr.offsets[p:p+2])
        if begin==end:
            continue
        enc = _encode_x6(csr.x6[begin:end],freq,att)
        h1 = torch.relu(enc @ weights[0][p].T + weights[1][p])
        h2 = torch.relu(h1 @ weights[2][p].T + weights[3][p])
        masks[begin:end,0] = ((h1>0)*bit).sum(1).to(torch.int32)
        masks[begin:end,1] = ((h2>0)*bit).sum(1).to(torch.int32)
    csr.relu_masks = masks
    expected = reference_weight_gradients(csr,freq,att,weights)
    actual = grouped_weight_gradients(csr,freq,att,weights,dtype=dtype)
    for a,b in zip(actual,expected):
        assert torch.isfinite(a).all()
        if dtype == torch.float32:
            torch.testing.assert_close(a,b,atol=1.e-4*scale,rtol=3.e-4)
        else:
            relative_l2 = (a-b).norm()/b.norm().clamp_min(1.e-30)
            assert float(relative_l2) < (.01 if dtype==torch.float16 else .04)
    for empty in [0,9]:
        for grad in actual:
            assert (grad[empty]==0).all()
