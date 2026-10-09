import torch
import pytest
from scannet.camera import camera_rays,distort_xy,undistort_xy,ray_cone_candidates

@pytest.mark.parametrize('params',[
 [592.906,599.204,320,240,-.0136426,.0401939,.0009821,-.0007264],
 [1054.8045,917.8632,320,240,.7779662,-.5630132,.1066415,.02926385],
 [1023.1212,451.6940,320,240,-.1347437,.08745083,.01950579,.04406346],
])
def test_distortion_inverse_full_real_frame(params):
    rays=camera_rays(480,640,params).double()
    projected=distort_xy(rays[...,:2],params[4:])
    projected=projected*torch.tensor(params[:2])+torch.tensor(params[2:4])
    yy,xx=torch.meshgrid(torch.arange(480)+.5,torch.arange(640)+.5,indexing='ij')
    assert (projected-torch.stack((xx,yy),-1)).abs().max()<.0001

def test_fx_fy_principal_point():
    rays=camera_rays(3,4,[10,20,1.5,.5],model='PINHOLE')
    assert torch.equal(rays[0,1],torch.tensor([0.,0.,1.]))
    assert torch.allclose(rays[2,3],torch.tensor([.2,.1,1.]))

def test_zero_distortion_matches_pinhole():
    assert torch.equal(camera_rays(23,35,[40,50,17,11,0,0,0,0]),camera_rays(23,35,[40,50,17,11],model='PINHOLE'))

def test_cone_does_not_drop_any_threshold_sphere_hit():
    torch.manual_seed(41)
    h,w,t=19,27,8
    rd=camera_rays(h,w,[30,18,9,11,.2,-.04,.01,-.02]).reshape(-1,3)
    rd=torch.nn.functional.normalize(rd,dim=-1);ro=torch.zeros_like(rd)
    pos=torch.randn(95,3);pos[:,2]+=1.5
    scales=torch.rand(95,3)*.5+.01
    pos[0]=0;scales[0]=1
    ids,offset=ray_cone_candidates(ro,rd,pos,scales.log(),w,h,t,.005,tile_chunk=3)
    radius=scales.amax(1)*(-2*torch.log(torch.tensor(.005))).sqrt()
    along=rd@pos.T
    perp=(pos.square().sum(1)[None]-along.square()).clamp_min(0)
    hits=(along>0)&(perp<radius.square()[None])
    for y in range(h):
        for x in range(w):
            tile=(y//t)*((w+t-1)//t)+x//t
            present=set(ids[offset[tile]:offset[tile+1]].tolist())
            assert set(hits[y*w+x].nonzero().flatten().tolist())<=present

def test_reject_mixed_origins_within_tile():
    ro=torch.zeros(64,3);ro[1,0]=1
    with pytest.raises(ValueError,match='single camera'):
        ray_cone_candidates(ro,torch.ones_like(ro),torch.ones(1,3),torch.zeros(1,3),8,8,8,.005)


def test_native_camera_validation_is_opt_in():
    from types import SimpleNamespace
    from scannet.data_validation import parse_supported_camera
    cam = SimpleNamespace(model='OPENCV', width=640, height=480,
                          params=[1023.,451.,320.,240.,-.13,.08,.019,.044])
    with pytest.raises(ValueError, match='Unsupported COLMAP camera model'):
        parse_supported_camera(cam, image_name='test.png')
    intr = parse_supported_camera(cam, image_name='test.png', camera_mode='colmap-native')
    assert intr['fy'] == 451. and intr['distortion'] == cam.params[4:]


def test_metadata_defaults_do_not_reinterpret_legacy_checkpoints():
    from scannet.camera import checkpoint_camera_mode
    assert checkpoint_camera_mode({}) == 'legacy-centered'
    for metadata in [{'camera_mode': 'unknown'}, {'camera_mode': 'colmap-native'}]:
        with pytest.raises(ValueError):
            checkpoint_camera_mode({'metadata': metadata})


def test_calibration_mismatch_fails_closed():
    from types import SimpleNamespace
    from scannet.camera import validate_checkpoint_calibration
    intr = {'model': 'OPENCV', 'fx': 1023., 'fy': 451., 'distortion': [-.13,.08,.019,.044]}
    checkpoint = {'metadata': {'camera_mode': 'colmap-native', 'camera_pixel_center': .5,
                  'camera_culling': 'raycone-v1', 'camera_intrinsics': intr}}
    dataset = SimpleNamespace(camera_mode='colmap-native', camera_intrinsics=intr.copy())
    validate_checkpoint_calibration(checkpoint, dataset)
    dataset.camera_intrinsics['fy'] = 1023.
    with pytest.raises(ValueError, match='calibration differ at fy'):
        validate_checkpoint_calibration(checkpoint, dataset)


def test_native_loader_scales_both_focals_and_principal_point():
    from scannet.scannet_loader import SceneDataset
    ds = SceneDataset.__new__(SceneDataset)
    ds.camera_mode = 'colmap-native'
    ds._native_dirs = None
    ds.camera_intrinsics = {'width': 8, 'height': 6, 'fx': 20., 'fy': 40.,
                            'cx': 3., 'cy': 1., 'distortion': [0.,0.,0.,0.]}
    pose = torch.eye(4)[:3]
    ro, rd = ds._get_rays((3,4), pose)
    torch.testing.assert_close(rd, camera_rays(3,4,[10.,20.,1.5,.5],model='PINHOLE'))
    assert not ro.any()


def test_no_hit_and_camera_inside_sphere():
    rd = camera_rays(8,8,[10.,10.,4.,4.],model='PINHOLE').reshape(-1,3)
    ro = torch.zeros_like(rd)
    positions = torch.tensor([[0.,0.,-10.],[0.,0.,0.]])
    scales = torch.full((2,3), .01).log()
    ids, offsets = ray_cone_candidates(ro,rd,positions,scales,8,8,8,.005)
    assert ids.tolist() == [1] and offsets.tolist() == [0,1]


@pytest.mark.parametrize('method', ['render_rgb_alpha_no_grad', 'render_rgb_t_no_grad'])
def test_unsupported_native_analysis_helpers_fail_closed(method):
    from scannet.cumodel import NeuralPrimitiveSceneTileCUDA
    scene = NeuralPrimitiveSceneTileCUDA(num_primitives=1)
    scene.camera_mode = 'colmap-native'
    with pytest.raises(NotImplementedError, match='Native camera'):
        getattr(scene, method)(torch.zeros(1,3), torch.ones(1,3))
