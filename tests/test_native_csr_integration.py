"""CUDA integration tests for distorted mosaic rays and streamed 512D loss."""
import pytest
import torch
import torch.nn.functional as F
from mip360.camera import camera_rays
from mip360.cumodel import NeuralPrimitiveSceneTileCUDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CSR integration requires a CUDA device')
@pytest.mark.parametrize('mosaic', [False, True])
@pytest.mark.parametrize('backend', ['scalar', 'wmma_fp16'])
def test_native_csr_joint(mosaic, backend, monkeypatch):
    monkeypatch.setenv('MLPSP_CSR_FORWARD_BACKEND', backend)
    monkeypatch.setenv('MLPSP_CSR_MLP_PRECISION', 'float32' if backend == 'scalar' else 'float16')
    torch.manual_seed(901)
    h,w,n=9,11,17
    scene=NeuralPrimitiveSceneTileCUDA(num_primitives=n,tile_size=8,feature_dim_low=512,feature_slots=5).cuda()
    scene.set_image_size(w,h);scene.camera_mode='colmap-native'
    scene.tile_order_mode_override='tilecenter'
    with torch.no_grad():
        scene.positions.copy_(torch.randn(n,3,device='cuda')*.3)
        scene.positions[:,2]+=1.2
        scene.log_scales.fill_(-1.4)
        scene.feat_attr.normal_(0,.1)
    rd=camera_rays(h,w,[8,11,5,4,.1,-.02,.005,-.003]).reshape(-1,3).cuda()
    ro=torch.zeros_like(rd)
    if mosaic:
        ro.reshape(h,w,3)[:,8:,0]=.2
        ro.reshape(h,w,3)[8:,:,1]=.1
    target=torch.randn(5,7,512,device='cuda')*.1
    rgbtarget=torch.rand(h*w,3,device='cuda')
    results=[]
    for csr in [False,True]:
        scene.zero_grad(set_to_none=True)
        rgb,feature=scene(ro,rd,return_feature=True,score_threshold=.005,feature_target=target if csr else None)
        if not csr:
            pred=F.interpolate(feature.view(h,w,512).permute(2,0,1)[None],size=(5,7),mode='bilinear',align_corners=False)[0].permute(1,2,0)
            feature=F.l1_loss(pred,target)
        loss=F.mse_loss(rgb,rgbtarget)+.7*feature
        loss.backward()
        results.append((rgb.detach(),loss.detach(),{k:p.grad.clone() for k,p in scene.named_parameters() if p.grad is not None}))
    ref,actual=results
    mixed=backend!='scalar'
    torch.testing.assert_close(actual[0],ref[0],atol=1e-4 if mixed else 2e-6,rtol=1e-3 if mixed else 2e-5)
    torch.testing.assert_close(actual[1],ref[1],atol=1e-5 if mixed else 2e-7,rtol=1e-3 if mixed else 2e-5)
    assert actual[2].keys()==ref[2].keys()
    assert torch.count_nonzero(ref[2]['feat_attr'])>0
    assert torch.count_nonzero(ref[2]['positions'])>0
    for name,grad in ref[2].items():
        torch.testing.assert_close(actual[2][name],grad,atol=2e-5 if mixed else 2e-6,rtol=.02 if mixed else 2e-4,msg=name)
