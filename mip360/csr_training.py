"""Record-based CUDA training with grouped gradients and streamed feature loss."""
import os
import torch

from .primitive_csr import group_records
from .primitive_gemm import grouped_weight_gradients,prepare_groups
from .record_feature_loss import streaming_feature_l1,feature_weight_gradients


def _record_forward(ext, base, width, height, threshold, beta, temp):
    """Select the actual record-producing forward, independently of VJP operands.

    WMMA batches rays sharing a tile/primitive; it is not a global primitive-major
    forward. The extension owns count/emit agreement and whole-frame FP32 retry.
    The released scalar path remains the default.
    """
    backend = os.environ.get("MLPSP_CSR_FORWARD_BACKEND", "scalar").strip().lower()
    if backend == "scalar":
        return ext.render_tiles_training_records(base, width, height, 8, threshold, beta, temp)
    precision = {"wmma_fp16": "float16", "wmma_bf16": "bfloat16"}.get(backend)
    if precision is None:
        raise ValueError("MLPSP_CSR_FORWARD_BACKEND must be scalar, wmma_fp16 or wmma_bf16")
    return ext.render_tiles_training_records_wmma(
        base, width, height, 8, threshold, beta, temp, precision)


class CSRTrainingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,*args):
        from .cumodel import NeuralPrimitiveSceneTileCUDA
        base=list(args[:15])
        attr,target,settings,bg=args[15:19]
        width,height,threshold,beta,temp=settings
        if len(base[7])!=6 or tuple(base[9].shape[1:])!=(32,72):
            raise ValueError("CSR training currently supports only the released 72→32→32→4 MLP")
        ext=NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static(8)
        dummy_rgb=torch.zeros_like(base[0])
        dummy_t=torch.zeros(len(base[0]),device=base[0].device,dtype=torch.float32)
        records=_record_forward(ext,base+[dummy_rgb,dummy_t],width,height,threshold,beta,temp)
        loss,signs,_=streaming_feature_l1(records,attr,target,width,height)
        rgb,alpha=records[:2]
        if bg is not None:
            rgb=rgb+(1-alpha).clamp(0,1)*bg.reshape(1,-1)
        ctx.save_for_backward(*base,attr,*records,signs)
        ctx.settings=settings
        ctx.bg=bg
        ctx.precision=os.environ.get("MLPSP_CSR_MLP_PRECISION","float16")
        if ctx.precision not in ("float32","float16","bfloat16"):
            raise ValueError("MLPSP_CSR_MLP_PRECISION must be float32, float16 or bfloat16")
        ctx.set_materialize_grads(False)
        return rgb,loss

    @staticmethod
    def backward(ctx,grad_rgb,grad_feature):
        from .cumodel import NeuralPrimitiveSceneTileCUDA
        tensors=ctx.saved_tensors
        base=tensors[:15]
        attr=tensors[15]
        records=tensors[16:24]
        signs=tensors[24]
        width,height,threshold,beta,temp=ctx.settings
        ext=NeuralPrimitiveSceneTileCUDA._ensure_tile_ext_static(8)
        gp=gs=gq=None
        weight_grads=[None]*6
        if grad_rgb is not None:
            grad_rgb=grad_rgb.contiguous()
            gt=torch.zeros(len(base[0]),device=grad_rgb.device,dtype=torch.float32) if ctx.bg is None else (
                grad_rgb*ctx.bg.reshape(1,-1)).sum(-1)
            d4,gp,gs,gq=ext.primitive_records_reverse(records[2],records[7],records[3],records[4],
                                                    base[3],base[4],grad_rgb,gt)
        else:
            d4=torch.zeros((len(records[2]),4),device=attr.device,dtype=torch.float32)
        csr=group_records(records[2],records[3],d4,len(attr),records[5])
        groups=prepare_groups(csr)
        if grad_rgb is not None:
            weight_grads=grouped_weight_gradients(csr,base[7],base[8],base[9:15],
                dtype=getattr(torch,ctx.precision),groups=groups)
        ga=None
        if grad_feature is not None:
            ga=feature_weight_gradients(records,csr,signs,attr,width,height,groups=groups,
                mixed=ctx.precision!="float32")*grad_feature
        return (None,None,gp,gs,gq,None,None,None,None,*weight_grads,ga,None,None,None)


def render_training_csr(base,attr,target,width,height,threshold,beta,temp,bg):
    return CSRTrainingFunction.apply(*base,attr,target,(width,height,threshold,beta,temp),bg)
