"""Streaming feature L1 from saved splat records, without a 512D FP32 image.

The exact high-resolution feature blend is sampled with the original bilinear
downsampling operator. Only int8 L1 signs and scalar loss partials are retained.
Geometry/alpha/MLP are intentionally detached from this loss, as in the release.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _blend_pixel(Ids, Offsets, Raw, Gates, Attributes, pixel, channels,
                 CHANNELS:tl.constexpr, SLOTS:tl.constexpr, BLOCK:tl.constexpr):
    acc=tl.full((BLOCK,),0,tl.float32)
    begin=tl.load(Offsets+pixel)
    end=tl.load(Offsets+pixel+1)
    for r in range(begin,end):
        p=tl.load(Ids+r)
        w=tl.load(Raw+r*8+4)*tl.load(Raw+r*8+5)
        if SLOTS==1 or SLOTS==5:
            mix=tl.load(Attributes+p*SLOTS*CHANNELS+channels,channels<CHANNELS,0)
        else:
            mix=tl.full((BLOCK,),0,tl.float32)
        if SLOTS!=1:
            for s in tl.static_range(4):
                g=tl.load(Gates+r*4+s)
                v=tl.load(Attributes+(p*SLOTS+s+(1 if SLOTS==5 else 0))*CHANNELS+channels,channels<CHANNELS,0)
                mix=tl.fma(g,v,mix)
        acc=tl.fma(w,mix,acc)
    return acc


@triton.jit
def _stream_l1(Ids,Offsets,Raw,Gates,Attributes,Target,Signs,Loss,Prediction,
               W:tl.constexpr,H:tl.constexpr,WF:tl.constexpr,HF:tl.constexpr,
               CHANNELS:tl.constexpr,SLOTS:tl.constexpr,BLOCK:tl.constexpr,DEBUG:tl.constexpr):
    pix=tl.program_id(0)
    chunk=tl.program_id(1)
    cc=chunk*BLOCK+tl.arange(0,BLOCK)
    xf=(pix%WF+.5)*(W/WF)-.5
    yf=(pix//WF+.5)*(H/HF)-.5
    xf=tl.maximum(xf,0)
    yf=tl.maximum(yf,0)
    x0=xf.to(tl.int32)
    y0=yf.to(tl.int32)
    x1=tl.minimum(x0+1,W-1)
    y1=tl.minimum(y0+1,H-1)
    wx=xf-x0
    wy=yf-y0
    a=_blend_pixel(Ids,Offsets,Raw,Gates,Attributes,y0*W+x0,cc,CHANNELS,SLOTS,BLOCK)
    b=_blend_pixel(Ids,Offsets,Raw,Gates,Attributes,y0*W+x1,cc,CHANNELS,SLOTS,BLOCK)
    c=_blend_pixel(Ids,Offsets,Raw,Gates,Attributes,y1*W+x0,cc,CHANNELS,SLOTS,BLOCK)
    d=_blend_pixel(Ids,Offsets,Raw,Gates,Attributes,y1*W+x1,cc,CHANNELS,SLOTS,BLOCK)
    value=(1-wy)*((1-wx)*a+wx*b)+wy*((1-wx)*c+wx*d)
    gt=tl.load(Target+pix*CHANNELS+cc,cc<CHANNELS,0).to(tl.float32)
    delta=value-gt
    signs=(delta>0).to(tl.int8)-(delta<0).to(tl.int8)
    tl.store(Signs+pix*CHANNELS+cc,signs,cc<CHANNELS)
    loss=tl.sum(tl.where(cc<CHANNELS,tl.abs(delta),0),0)/(HF*WF*CHANNELS)
    tl.store(Loss+pix*tl.cdiv(CHANNELS,BLOCK)+chunk,loss)
    if DEBUG:
        tl.store(Prediction+pix*CHANNELS+cc,value,cc<CHANNELS)


def streaming_feature_l1(records, attributes, target, width, height, *, debug=False):
    _,_,ids,x6,raw,masks,gates,offsets=records
    if target.ndim!=3 or min(target.shape)<=0:
        raise ValueError("feature target must have nonempty HWC dimensions")
    if attributes.dtype!=torch.float32 or not attributes.is_contiguous():
        raise ValueError("feature master attributes must be contiguous FP32")
    if target.dtype not in (torch.float16,torch.float32):
        raise ValueError("feature target must be FP16 or FP32")
    if any(not t.is_cuda or t.device!=attributes.device for t in (ids,offsets,raw,gates,attributes,target)):
        raise ValueError("feature tensors and records must share a CUDA device")
    hf,wf,channels=target.shape
    slots=1 if attributes.ndim==2 else attributes.shape[1]
    if slots not in (1,4,5) or attributes.shape[-1]!=channels:
        raise ValueError("feature shape does not match the 1/4/5-slot renderer")
    if hf>height or wf>width:
        raise ValueError("streamed feature loss supports downsampling only")
    target=target.contiguous()
    signs=torch.empty(target.shape,device=target.device,dtype=torch.int8)
    partial=torch.empty((hf*wf,triton.cdiv(channels,128)),device=target.device,dtype=torch.float32)
    pred=torch.empty(target.shape,device=target.device,dtype=torch.float32) if debug else None
    _stream_l1[(hf*wf,triton.cdiv(channels,128))](ids,offsets,raw,gates,attributes,target,signs,partial,pred,
        W=width,H=height,WF=wf,HF=hf,CHANNELS=channels,SLOTS=slots,BLOCK=128,DEBUG=debug,
        num_warps=4,enable_fp_fusion=False)
    return partial.sum(),signs,pred


@triton.jit
def _resample_adjoint(Signs, pix, cc, valid,
                      W:tl.constexpr,H:tl.constexpr,WF:tl.constexpr,HF:tl.constexpr,CHANNELS:tl.constexpr):
    xx=pix%W
    yy=pix//W
    tx=tl.floor((xx+.5)*(WF/W)-.5).to(tl.int32)
    ty=tl.floor((yy+.5)*(HF/H)-.5).to(tl.int32)
    grad=tl.full((pix.shape[0],cc.shape[0]),0,tl.float32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            ix=tx+dx
            iy=ty+dy
            srcx=tl.maximum((ix+.5)*(W/WF)-.5,0)
            srcy=tl.maximum((iy+.5)*(H/HF)-.5,0)
            weight=tl.maximum(0,1-tl.abs(srcx-xx))*tl.maximum(0,1-tl.abs(srcy-yy))
            mask=valid & (ix>=0)&(ix<WF)&(iy>=0)&(iy<HF)
            s=tl.load(Signs+(iy[:,None]*WF+ix[:,None])*CHANNELS+cc[None,:],
                      mask[:,None]&(cc[None,:]<CHANNELS),0).to(tl.float32)
            grad+=weight[:,None]*s
    return grad


@triton.jit
def _feature_vjp(Signs,Order,RecordPixels,Raw,Gates,Offsets,GroupPrim,GroupStart,Partial,
                  W:tl.constexpr,H:tl.constexpr,WF:tl.constexpr,HF:tl.constexpr,
                  CHANNELS:tl.constexpr,SLOTS:tl.constexpr,SEGMENT:tl.constexpr,
                  MIXED:tl.constexpr,BLOCK:tl.constexpr=32,BC:tl.constexpr=64):
    group=tl.program_id(0)
    cc=tl.program_id(1)*BC+tl.arange(0,BC)
    p=tl.load(GroupPrim+group)
    begin=tl.load(GroupStart+group)
    end=tl.minimum(begin+SEGMENT,tl.load(Offsets+p+1))
    rr=tl.arange(0,BLOCK)
    ss=tl.arange(0,16)
    accum=tl.full((16,BC),0,tl.float32)
    dtype:tl.constexpr=tl.float16 if MIXED else tl.float32
    for base in range(begin,end,BLOCK):
        row=base+rr
        r=tl.load(Order+row,row<end,0)
        pix=tl.load(RecordPixels+r,row<end,0)
        trans=tl.load(Raw+r*8+4,row<end,0)
        alpha=tl.load(Raw+r*8+5,row<end,0)
        if SLOTS==1:
            gates=tl.full((BLOCK,16),1,tl.float32)
        else:
            slot=ss-(1 if SLOTS==5 else 0)
            gates=tl.load(Gates+r[:,None]*4+slot[None,:],
                           (row[:,None]<end)&(slot[None,:]>=0)&(slot[None,:]<4),0)
            if SLOTS==5:
                gates=tl.where(ss[None,:]==0,1,gates)
        a=tl.where((row[:,None]<end)&(ss[None,:]<SLOTS),
                    (trans*alpha)[:,None]*gates,0).to(dtype)
        # Scale-free sign adjoints have magnitude <= 1; normalize in FP32
        # after GEMM rather than casting ~1e-8 full-frame mean gradients.
        b=_resample_adjoint(Signs,pix,cc,row<end,W,H,WF,HF,CHANNELS).to(dtype)
        accum+=tl.dot(tl.trans(a),b,input_precision="ieee")
    out=Partial+group*SLOTS*CHANNELS+ss[:,None]*CHANNELS+cc[None,:]
    tl.store(out,accum/(WF*HF*CHANNELS),(ss[:,None]<SLOTS)&(cc[None,:]<CHANNELS))


@triton.jit
def _feature_reduce(Partial,Offsets,Output,PARAMS:tl.constexpr,BLOCK:tl.constexpr):
    p=tl.program_id(0)
    kk=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    start=tl.load(Offsets+p)
    end=tl.load(Offsets+p+1)
    acc=tl.full((BLOCK,),0,tl.float32)
    for group in range(start,end):
        acc+=tl.load(Partial+group*PARAMS+kk,kk<PARAMS,0)
    tl.store(Output+p*PARAMS+kk,acc,kk<PARAMS)


def feature_weight_gradients(records, csr, signs, attributes, width, height, *,
                             mixed=True,segment_size=512,groups=None):
    from .primitive_gemm import resolve_groups
    _,_,ids,x6,raw,masks,gates,pixel_offsets=records
    hf,wf,channels=signs.shape
    if signs.dtype!=torch.int8 or not signs.is_contiguous():
        raise ValueError("feature loss signs must be contiguous int8")
    slots=1 if attributes.ndim==2 else attributes.shape[1]
    if hf>height or wf>width:
        raise ValueError("streamed feature loss supports downsampling only")
    primitives,starts,group_offsets=resolve_groups(csr,segment_size,groups)
    pixels=torch.repeat_interleave(torch.arange(width*height,device=ids.device,dtype=torch.int32),
                                   pixel_offsets.diff(),output_size=len(ids))
    partial=torch.empty((len(primitives),slots,channels),device=ids.device,dtype=torch.float32)
    result=torch.empty_like(attributes)
    if len(primitives):
        _feature_vjp[(len(primitives),triton.cdiv(channels,64))](signs,csr.record_indices,pixels,raw,gates,
            csr.offsets,primitives,starts,partial,W=width,H=height,WF=wf,HF=hf,CHANNELS=channels,
            SLOTS=slots,SEGMENT=segment_size,MIXED=mixed,num_warps=4,enable_fp_fusion=False)
    _feature_reduce[(len(attributes),triton.cdiv(slots*channels,256))](partial,group_offsets,result,
        PARAMS=slots*channels,BLOCK=256,num_warps=4)
    return result
