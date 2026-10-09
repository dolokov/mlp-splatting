"""Native ray-cone culling agrees with exhaustive CUDA rendering and gradients."""
import pytest
import torch
from mip360.camera import camera_rays
from mip360.cumodel import NeuralPrimitiveSceneTileCUDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('feature_dim', [0, 512])
@pytest.mark.parametrize('mosaic', [False, True])
def test_native_forward_backward_against_all_candidates(feature_dim, mosaic):
    torch.manual_seed(2025)
    device=torch.device('cuda');n=17;h=9;w=11;t=8
    scene=NeuralPrimitiveSceneTileCUDA(num_primitives=n,tile_size=t,feature_dim_low=feature_dim).to(device)
    scene.set_image_size(w,h);scene.tile_order_mode_override='tilecenter'
    with torch.no_grad():
        scene.positions.copy_(torch.randn(n,3,device=device)*.4)
        scene.positions[:,2]+=1.2;scene.positions[-1]=torch.tensor([10.,10.,1.],device=device)
        scene.log_scales.copy_(torch.randn(n,3,device=device)*.3-2)
        scene.quats.copy_(torch.nn.functional.normalize(torch.randn(n,4,device=device),dim=-1))
    rd=camera_rays(h,w,[8,11,5,4,.1,-.02,.005,-.003]).reshape(-1,3).to(device)
    ro=torch.zeros_like(rd)
    if mosaic:
        ro.reshape(h,w,3)[:,8:,0]=.2
        ro.reshape(h,w,3)[8:,:,1]=.1
    nt=((w+t-1)//t)*((h+t-1)//t)
    full=torch.arange(n,device=device,dtype=torch.int32).repeat(nt)
    offsets=torch.arange(nt+1,device=device,dtype=torch.int32)*n

    def render(**kwargs):
        output=scene(ro,rd,score_threshold=.005,return_feature=bool(feature_dim),**kwargs)
        return torch.cat(output,dim=-1) if feature_dim else output

    reference=render(culled_prim_indices=full,tile_offsets=offsets)
    target=torch.rand_like(reference)
    ((reference-target)**2).mean().backward()
    grads={k:p.grad.detach().clone() for k,p in scene.named_parameters() if p.grad is not None}
    scene.zero_grad(set_to_none=True)
    scene.camera_mode='colmap-native'
    actual=render()
    torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
    ((actual-target)**2).mean().backward()
    for k,p in scene.named_parameters():
        if k in grads:
            torch.testing.assert_close(p.grad,grads[k],atol=5e-6,rtol=2e-4,msg=k)
