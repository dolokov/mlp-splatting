"""Opt-in native COLMAP rays and conservative ray-cone tile culling.

The historical centered-pinhole default is unchanged. Native cameras require
consistent rays, culling, and checkpoint metadata throughout train/evaluation.
"""
import math
import torch
import torch.nn.functional as F


def checkpoint_camera_mode(checkpoint):
    metadata = checkpoint.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('Checkpoint metadata must be a dictionary')
    mode = metadata.get('camera_mode', 'legacy-centered')
    if mode not in {'legacy-centered', 'colmap-native'}:
        raise ValueError(f'Unsupported checkpoint camera_mode={mode!r}')
    if mode == 'colmap-native':
        if metadata.get('camera_pixel_center') != 0.5 or metadata.get('camera_culling') != 'raycone-v1':
            raise ValueError('Native camera checkpoint has incompatible pixel-center/culling metadata')
        if not isinstance(metadata.get('camera_intrinsics'), dict):
            raise ValueError('Native camera checkpoint must record its COLMAP calibration')
    return mode


def validate_checkpoint_calibration(checkpoint, dataset):
    mode = checkpoint_camera_mode(checkpoint)
    if mode != getattr(dataset, 'camera_mode', 'legacy-centered'):
        raise ValueError('Checkpoint and dataset camera modes differ; retraining is required')
    if mode == 'colmap-native':
        expected = checkpoint['metadata']['camera_intrinsics']
        actual = dataset.camera_intrinsics
        if expected.keys() != actual.keys():
            raise ValueError('Checkpoint and dataset calibration fields differ')
        for key in expected:
            if isinstance(expected[key], str):
                same = expected[key] == actual[key]
            else:
                same = torch.allclose(torch.as_tensor(expected[key], dtype=torch.float64),
                                      torch.as_tensor(actual[key], dtype=torch.float64), rtol=1e-6, atol=1e-6)
            if not same:
                raise ValueError(f'Checkpoint and dataset calibration differ at {key}')


def distort_xy(xy, distortion):
    k1,k2,p1,p2=distortion
    x,y=xy.unbind(-1)
    r2=x*x+y*y
    radial=1+k1*r2+k2*r2*r2
    return torch.stack((x*radial+2*p1*x*y+p2*(r2+2*x*x),
                        y*radial+p1*(r2+2*y*y)+2*p2*x*y),-1)


def undistort_xy(target, distortion, iterations=40, tolerance=1e-10):
    """Newton inversion with analytic Jacobian and explicit failure on bad rays."""
    if target.dtype!=torch.float64:
        raise TypeError('invert in float64 before casting final ray directions')
    k1,k2,p1,p2=distortion
    xy=target.clone()
    for _ in range(iterations):
        x,y=xy.unbind(-1);r2=x*x+y*y
        radial=1+k1*r2+k2*r2*r2
        radial_derivative=k1+2*k2*r2
        dx=2*x*radial_derivative;dy=2*y*radial_derivative
        a=radial+x*dx+2*p1*y+6*p2*x
        b=x*dy+2*p1*x+2*p2*y
        c=y*dx+2*p1*x+2*p2*y
        d=radial+y*dy+6*p1*y+2*p2*x
        det=a*d-b*c
        residual=distort_xy(xy,distortion)-target
        if float(residual.abs().max())<=tolerance:break
        if not torch.isfinite(det).all() or bool((det.abs()<1e-12).any()):
            raise ValueError('Distortion inverse has a singular Jacobian')
        rx,ry=residual.unbind(-1)
        step=torch.stack(((d*rx-b*ry)/det,(-c*rx+a*ry)/det),-1)
        # Damping limits divergence for unreasonable COLMAP calibrations.
        length=step.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        xy=xy-step*(0.5/length).clamp(max=1)
    error=(distort_xy(xy,distortion)-target).abs().max()
    if not torch.isfinite(xy).all() or float(error)>tolerance:
        raise ValueError(f'Distortion inverse failed: normalized residual {float(error):.3g}')
    return xy


def camera_rays(height,width,params,model='OPENCV',pixel_center=0.5):
    params=[float(x) for x in params]
    if model=='PINHOLE' and len(params)==4:
        fx,fy,cx,cy=params;distortion=(0.,0.,0.,0.)
    elif model=='SIMPLE_PINHOLE' and len(params)==3:
        fx,cx,cy=params;fy=fx;distortion=(0.,0.,0.,0.)
    elif model=='OPENCV' and len(params)==8:
        fx,fy,cx,cy,*distortion=params
    else:raise ValueError(f'Unsupported model/parameter count: {model}/{len(params)}')
    if not all(math.isfinite(v) for v in params) or min(fx,fy)<=0 or min(height,width)<=0:
        raise ValueError('Invalid camera calibration')
    y,x=torch.meshgrid(torch.arange(height,dtype=torch.float64)+pixel_center,torch.arange(width,dtype=torch.float64)+pixel_center,indexing='ij')
    target=torch.stack(((x-cx)/fx,(y-cy)/fy),-1)
    xy=undistort_xy(target,distortion) if any(distortion) else target
    return torch.cat((xy,torch.ones_like(xy[...,:1])),-1).float()


@torch.no_grad()
def ray_cone_candidates(ray_o,ray_d,positions,log_scales,width,height,tile_size,score_threshold,tile_chunk=128):
    """Conservative bound for the threshold ellipsoid using its enclosing sphere.

    Every ray in a tile is inside the computed cone. If any of those half-rays
    intersects a primitive's threshold ellipsoid, its enclosing sphere must
    intersect the cone. Near-plane crossing/large spheres are retained. False
    positives are acceptable; the existing CUDA kernel applies the exact test.
    """
    if not 0<score_threshold<1:raise ValueError('score_threshold must be in (0,1)')
    if ray_o.shape!=(height*width,3) or ray_d.shape!=ray_o.shape:raise ValueError('ray shape mismatch')
    device=positions.device
    if ray_o.device!=device or ray_d.device!=device:raise ValueError('device mismatch')
    ntx=(width+tile_size-1)//tile_size;nty=(height+tile_size-1)//tile_size
    tiles=torch.arange(ntx*nty,device=device)
    yy,xx=torch.meshgrid(torch.arange(tile_size,device=device),torch.arange(tile_size,device=device),indexing='ij')
    px=((tiles%ntx)[:,None]*tile_size+xx.reshape(1,-1)).clamp(max=width-1)
    py=((tiles//ntx)[:,None]*tile_size+yy.reshape(1,-1)).clamp(max=height-1)
    idx=py*width+px
    origins=ray_o[idx[:,0]]
    if bool(((ray_o[idx]-origins[:,None]).abs().amax()>1e-6)):
        raise ValueError('Each tile must use a single camera origin')
    directions=F.normalize(ray_d[idx],dim=-1)
    axes=F.normalize(directions.mean(1),dim=-1)
    cos_theta=(directions*axes[:,None]).sum(-1).amin(1).clamp(-1,1)
    if bool((cos_theta<=0).any()):raise ValueError('Tile ray cone spans >=90 degrees')
    sin_theta=(1-cos_theta.square()).clamp_min(0).sqrt()
    radius=log_scales.exp().amax(-1)*math.sqrt(-2*math.log(score_threshold))
    radius=radius*(1+1e-5)+1e-7
    pairs=[];counts=[]
    for begin in range(0,len(tiles),tile_chunk):
        end=min(begin+tile_chunk,len(tiles))
        delta=positions[None]-origins[begin:end,None]
        dist=delta.norm(dim=-1)
        ratio=(radius[None]/dist.clamp_min(1e-20)).clamp(0,1)
        cos_beta=(delta*axes[begin:end,None]).sum(-1)/dist.clamp_min(1e-20)
        bound=cos_theta[begin:end,None]*(1-ratio.square()).clamp_min(0).sqrt()-sin_theta[begin:end,None]*ratio
        keep=(dist<=radius[None])|(cos_beta>=bound-2e-6)
        pair=keep.nonzero(as_tuple=False)
        pairs.append(pair[:,1].to(torch.int32))
        counts.append(keep.sum(1,dtype=torch.int64))
    ids=torch.cat(pairs)
    offset=torch.cat((torch.zeros(1,device=device,dtype=torch.int64),torch.cat(counts).cumsum(0)))
    if int(offset[-1])>=2**31:raise OverflowError('Tile candidate count exceeds int32 capacity')
    return ids,offset.to(torch.int32)
