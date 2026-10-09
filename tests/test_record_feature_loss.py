import pytest
import torch
from test_cuda_renderer import tile_ext,_renderer_inputs,_base_args
from mip360.primitive_csr import group_records


@pytest.mark.parametrize("slots",[1,4,5])
@pytest.mark.parametrize("size",[(1,1,1,1),(9,7,7,5),(8,8,8,8)])
def test_streamed_l1_matches_framebuffer(tile_ext,slots,size):
    from mip360.record_feature_loss import streaming_feature_l1,feature_weight_gradients
    torch.manual_seed(482)
    torch.backends.cuda.matmul.allow_tf32=False
    w,h,wf,hf=size
    n,c=12,17
    data=_renderer_inputs(n)
    yy,xx=torch.meshgrid(torch.linspace(-.7,.7,h,device="cuda"),torch.linspace(-.7,.7,w,device="cuda"),indexing="ij")
    data["rays_o"]=torch.stack((xx.flatten(),yy.flatten(),torch.zeros(w*h,device="cuda")),-1)
    data["rays_d"]=torch.tensor([0.,0.,1.],device="cuda").repeat(w*h,1)
    data["quats"].normal_()
    data["quats"] /= data["quats"].norm(dim=-1,keepdim=True)
    data["log_scales"].uniform_(-.6,0)
    data["positions"][:,:2].uniform_(-.3,.3)
    tiles=((w+7)//8)*((h+7)//8)
    data["candidates"]=torch.arange(n,device="cuda",dtype=torch.int32).repeat(tiles)
    data["offsets"]=torch.arange(tiles+1,device="cuda",dtype=torch.int32)*n
    base=list(_base_args(data))
    grgb=torch.zeros(w*h,3,device="cuda")
    gt=torch.zeros(w*h,device="cuda")
    attr=torch.randn((n,c) if slots==1 else (n,slots,c),device="cuda")*.2
    target=torch.randn(hf,wf,c,device="cuda")*.2
    records=tile_ext.render_tiles_training_records(base+[grgb,gt],w,h,8,.001,1.5,2.)
    _,_,features=tile_ext.render_tiles_forward_tilecenter_with_feat(*base,attr,1.5,2.,w,h,8,8,.001)
    features=features.detach().requires_grad_()
    image=features.view(h,w,c).permute(2,0,1).unsqueeze(0)
    resized=torch.nn.functional.interpolate(image,size=(hf,wf),mode="bilinear",align_corners=False).squeeze(0).permute(1,2,0)
    expected_loss=torch.nn.functional.l1_loss(resized,target)
    expected_loss.backward()
    expected_grad=tile_ext.render_tiles_backward_featattr_tilecenter(*base,attr,features.grad.contiguous(),1.5,2.,w,h,8,8,.001)
    loss,signs,pred=streaming_feature_l1(records,attr,target,w,h,debug=True)
    torch.testing.assert_close(pred,resized,atol=3.e-6,rtol=3.e-6)
    torch.testing.assert_close(loss,expected_loss,atol=2.e-7,rtol=2.e-6)
    torch.testing.assert_close(signs,(resized-target).sign().to(torch.int8),atol=0,rtol=0)
    ids,x6=records[2:4]
    csr=group_records(ids,x6,torch.empty((len(ids),4),device="cuda"),n,records[5])
    for mixed in (False,True):
        actual=feature_weight_gradients(records,csr,signs,attr,w,h,mixed=mixed)
        if not mixed:
            torch.testing.assert_close(actual,expected_grad,atol=2.e-6,rtol=2.e-5)
        else:
            assert float((actual-expected_grad).norm()/expected_grad.norm())<.003


@pytest.mark.parametrize("precision",["float32","float16"])
@pytest.mark.parametrize("feature_only",[False,True])
def test_scene_csr_autograd_contract(tile_ext,monkeypatch,precision,feature_only):
    from mip360 import cumodel
    monkeypatch.setattr(cumodel,"_tile_ext",tile_ext)
    monkeypatch.setattr(cumodel,"_tile_ext_key",cumodel._tile_ext_build_config(8)[0])
    monkeypatch.setenv("MLPSP_CSR_MLP_PRECISION",precision)
    torch.manual_seed(165)
    scene=cumodel.NeuralPrimitiveSceneTileCUDA(num_primitives=4,feature_dim_low=17,feature_slots=5).cuda()
    scene.set_image_size(9,7)
    with torch.no_grad():
        scene.positions[:,:2].uniform_(-.3,.3)
        scene.positions[:,2]=1.
        scene.log_scales.uniform_(-.5,0)
        scene.feat_attr.normal_(0,.1)
    ro=torch.randn(63,3,device="cuda")*.1
    rd=torch.tensor([0.,0.,1.],device="cuda").repeat(63,1)
    cull=torch.arange(4,device="cuda",dtype=torch.int32).repeat(2)
    offs=torch.tensor([0,4,8],device="cuda",dtype=torch.int32)
    target=torch.randn(5,7,17,device="cuda")*.1
    rgb_target=torch.randn(63,3,device="cuda")*.1
    results=[]
    for csr in (False,True):
        scene.zero_grad(set_to_none=True)
        rgb,second=scene(ro,rd,return_feature=True,bg_color=.3,culled_prim_indices=cull,tile_offsets=offs,
                         feature_target=target if csr else None)
        if csr:
            feature_loss=second
        else:
            small=torch.nn.functional.interpolate(second.reshape(7,9,17).permute(2,0,1).unsqueeze(0),
                       size=(5,7),mode="bilinear",align_corners=False).squeeze(0).permute(1,2,0)
            feature_loss=torch.nn.functional.l1_loss(small,target)
        loss=.7*feature_loss
        if not feature_only:
            loss=loss+torch.nn.functional.l1_loss(rgb,rgb_target)
        loss.backward()
        results.append((loss.detach(),{name:None if param.grad is None else param.grad.clone() for name,param in scene.named_parameters()}))
    torch.testing.assert_close(results[0][0],results[1][0],atol=2.e-6,rtol=2.e-6)
    for name,expected in results[0][1].items():
        actual=results[1][1][name]
        if expected is None:
            assert actual is None,name
        elif precision=="float32" or name in ("positions","log_scales","quats"):
            torch.testing.assert_close(actual,expected,atol=5.e-6,rtol=5.e-5)
        else:
            assert float((actual-expected).norm()/expected.norm().clamp_min(1.e-30))<.003,name
