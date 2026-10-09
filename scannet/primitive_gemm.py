"""Grouped MLP backward computation with FP32 output-owned reductions.

One CUDA program owns a bounded segment of a primitive's CSR records. It
recomputes the two activations and evaluates all three weight gradients using
GEMM, rather than issuing one weight atomic per ray/hit. A second kernel reduces
segment partials into FP32 master-weight gradients. No change to compositing or
the established stop-gradient geometry contract is made here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@dataclass(frozen=True)
class GroupLayout:
    """Prepared segments bound to their CSR and segment length.

    Tensor identity/version checks are host-only: reusing a layout cannot
    silently reinterpret 512-row segments as 1024-row segments, or use offsets
    that have subsequently been modified in place.
    """

    primitives: torch.Tensor
    starts: torch.Tensor
    offsets: torch.Tensor
    segment_size: int
    source_offsets: torch.Tensor
    num_records: int
    tensor_versions: tuple[int, int, int, int]

    def __iter__(self):
        return iter((self.primitives, self.starts, self.offsets))

    def validate(self, csr, segment_size):
        if segment_size != self.segment_size:
            raise ValueError(
                f"Prepared groups use segment_size={self.segment_size}, "
                f"but the consumer requested {segment_size}"
            )
        if self.source_offsets is not csr.offsets or self.num_records != len(csr.x6):
            raise ValueError("Prepared groups belong to a different CSR layout")
        tensors = (self.source_offsets, self.primitives, self.starts, self.offsets)
        if tuple(t._version for t in tensors) != self.tensor_versions:
            raise ValueError("Prepared groups or their CSR offsets were modified in place")
        return self


_SAFETY_STATS = {"calls": 0, "mixed_calls": 0, "fp32_fallbacks": 0, "fp32_nonfinite_failures": 0}


def get_grouped_vjp_safety_stats(*, reset=False):
    """Return process-local counters for run reports; resetting is explicit."""
    snapshot = dict(_SAFETY_STATS)
    if reset:
        for key in _SAFETY_STATS:
            _SAFETY_STATS[key] = 0
    return snapshot


def _checked_grouped_vjp(launch, dtype, loss_scale):
    """Never deliver a nonfinite MLP gradient to the caller's Adam step.

    ``launch`` returns the unmodified result and its fused reduction flag. The
    successful finite path performs no rescaling or gradient replacement.
    """
    _SAFETY_STATS["calls"] += 1
    mixed = dtype != torch.float32
    _SAFETY_STATS["mixed_calls"] += int(mixed)
    result, nonfinite = launch(dtype, loss_scale)
    if not bool(nonfinite.item()):
        return result
    if mixed:
        _SAFETY_STATS["fp32_fallbacks"] += 1
        warnings.warn(
            f"[CSR safety] Nonfinite {dtype} MLP VJP; retrying with IEEE FP32 "
            f"before Adam (FP32 fallbacks={_SAFETY_STATS['fp32_fallbacks']}).",
            RuntimeWarning,
            stacklevel=2,
        )
        # The original FP32 master tensors and exact ReLU masks are reused.
        # Disable loss scaling for the retry; never skip an optimizer step or
        # zero a primitive to hide a mixed-precision overflow.
        result, nonfinite = launch(torch.float32, 1.0)
        if not bool(nonfinite.item()):
            return result
    _SAFETY_STATS["fp32_nonfinite_failures"] += 1
    raise FloatingPointError(
        "[CSR safety] IEEE FP32 MLP VJP is nonfinite; no gradients were returned "
        "and Adam must not step. Inspect the saved model, records and loss. "
        f"Safety counters: {get_grouped_vjp_safety_stats()}"
    )


@triton.jit
def _grouped_vjp(X, D3, Masks, Offsets, GroupPrim, GroupStart, Freq, Att,
                 W1, B1, W2, B2, W3, Partial,
                 PRECISION: tl.constexpr, SCALE: tl.constexpr,
                 SEGMENT: tl.constexpr, BLOCK: tl.constexpr, EXACT_MASKS: tl.constexpr):
    group = tl.program_id(0)
    p = tl.load(GroupPrim + group)
    begin = tl.load(GroupStart + group)
    end = tl.minimum(begin + SEGMENT, tl.load(Offsets + p + 1))
    ii = tl.arange(0, 128)
    hh = tl.arange(0, 32)
    oo = tl.arange(0, 16)
    rr = tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float32 if PRECISION == 0 else (tl.float16 if PRECISION == 1 else tl.bfloat16)
    w1 = tl.load(W1 + p*2304 + hh[None,:]*72 + ii[:,None], ii[:,None]<72, 0).to(dtype)
    w2 = tl.load(W2 + p*1024 + hh[None,:]*32 + hh[:,None]).to(dtype)
    w3 = tl.load(W3 + p*128 + oo[None,:]*32 + hh[:,None], oo[None,:]<4, 0).to(dtype)
    b1 = tl.load(B1 + p*32 + hh)
    b2 = tl.load(B2 + p*32 + hh)
    # FP32 accumulators across the entire segment, never half accumulators.
    gw1 = tl.full((32,128), 0, tl.float32)
    gw2 = tl.full((32,32), 0, tl.float32)
    gw3 = tl.full((16,32), 0, tl.float32)
    gb1 = tl.full((32,), 0, tl.float32)
    gb2 = tl.full((32,), 0, tl.float32)
    gb3 = tl.full((16,), 0, tl.float32)
    f = tl.load(Freq + (ii % 36) % 6)
    a = tl.load(Att + (ii % 36) % 6)
    for base in range(begin, end, BLOCK):
        rows = base + rr
        x = tl.load(X + rows[:,None]*6 + ((ii%36)//6)[None,:], rows[:,None]<end, 0)
        phase = x * f[None,:] * a[None,:]
        encoded = tl.where(ii[None,:]<36, libdevice.sin(phase), libdevice.cos(phase))
        encoded = tl.where((ii[None,:]<72) & (rows[:,None]<end), encoded, 0).to(dtype)
        h1f = tl.maximum(tl.dot(encoded, w1, input_precision="ieee") + b1[None,:], 0)
        h1 = h1f.to(dtype)
        h2f = tl.maximum(tl.dot(h1, w2, input_precision="ieee") + b2[None,:], 0)
        h2 = h2f.to(dtype)
        d3 = (tl.load(D3 + rows[:,None]*4 + oo[None,:],
                      (rows[:,None]<end) & (oo[None,:]<4), 0) * SCALE).to(dtype)
        if EXACT_MASKS:
            mask1 = tl.load(Masks+rows*2,rows<end,0).to(tl.uint32)
            mask2 = tl.load(Masks+rows*2+1,rows<end,0).to(tl.uint32)
            active1 = ((mask1[:,None] >> hh[None,:]) & 1) != 0
            active2 = ((mask2[:,None] >> hh[None,:]) & 1) != 0
        else:
            active1 = h1f>0
            active2 = h2f>0
        d2f = tl.dot(d3, tl.trans(w3), input_precision="ieee") * active2
        d2 = d2f.to(dtype)
        d1f = tl.dot(d2, tl.trans(w2), input_precision="ieee") * active1
        d1 = d1f.to(dtype)
        gw1 += tl.dot(tl.trans(d1), encoded, input_precision="ieee")
        gw2 += tl.dot(tl.trans(d2), h1, input_precision="ieee")
        gw3 += tl.dot(tl.trans(d3), h2, input_precision="ieee")
        # Use the same adjoints for biases and matrices (cast before reduction).
        gb1 += tl.sum(d1.to(tl.float32), axis=0)
        gb2 += tl.sum(d2.to(tl.float32), axis=0)
        gb3 += tl.sum(d3.to(tl.float32), axis=0)
    out = Partial + group*3524
    tl.store(out + hh[:,None]*72 + ii[None,:], gw1/SCALE, ii[None,:]<72)
    tl.store(out + 2304 + hh, gb1/SCALE)
    tl.store(out + 2336 + hh[:,None]*32 + hh[None,:], gw2/SCALE)
    tl.store(out + 3360 + hh, gb2/SCALE)
    tl.store(out + 3392 + oo[:,None]*32 + hh[None,:], gw3/SCALE, oo[:,None]<4)
    tl.store(out + 3520 + oo, gb3/SCALE, oo<4)


@triton.jit
def _reduce_partials(Partial, GroupOffsets, Output, Nonfinite, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    kk = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
    start = tl.load(GroupOffsets+p)
    end = tl.load(GroupOffsets+p+1)
    total = tl.full((BLOCK,), 0, tl.float32)
    for group in range(start, end):
        total += tl.load(Partial + group*3524 + kk, kk<3524, 0)
    tl.store(Output+p*3524+kk, total, kk<3524)
    # Preserve the FP32 arithmetic/store above. Only exceptional blocks write
    # the flag, avoiding a separate read of the complete gradient buffer.
    bad = (kk < 3524) & ((total != total) | (tl.abs(total) == float("inf")))
    if tl.sum(bad.to(tl.int32), axis=0) != 0:
        tl.atomic_or(Nonfinite, 1)


def prepare_groups(csr, segment_size=512):
    if not isinstance(segment_size, int) or isinstance(segment_size, bool) or segment_size <= 0:
        raise ValueError("segment_size must be a positive integer")
    if csr.offsets.ndim != 1 or csr.offsets.dtype != torch.int64 or not csr.offsets.is_contiguous():
        raise ValueError("CSR offsets must be contiguous int64")
    if csr.offsets.numel() < 1:
        raise ValueError("CSR offsets must include the initial zero")
    counts = triton.cdiv(csr.offsets.diff(), segment_size)
    group_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    total = int(group_offsets[-1])
    primitives = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts, output_size=total)
    local = torch.arange(total, device=counts.device) - group_offsets[primitives]
    starts = csr.offsets[primitives] + local*segment_size
    tensors = (csr.offsets, primitives, starts, group_offsets)
    return GroupLayout(primitives, starts, group_offsets, segment_size,
                       csr.offsets, len(csr.x6), tuple(t._version for t in tensors))


def resolve_groups(csr, segment_size=512, groups=None):
    """Require metadata when a caller shares groups between MLP/feature VJPs."""
    if not isinstance(segment_size, int) or isinstance(segment_size, bool) or segment_size <= 0:
        raise ValueError("segment_size must be a positive integer")
    if groups is None:
        return prepare_groups(csr, segment_size)
    if not isinstance(groups, GroupLayout):
        raise TypeError("groups must be a GroupLayout returned by prepare_groups")
    return groups.validate(csr, segment_size)


def _launch_grouped_vjp(csr, freq, attenuation, weights, groups, dtype, block_size, scale):
    """One unguarded launch, shared by the normal path and the FP32 retry."""
    w1,b1,w2,b2,w3,b3 = weights
    n = len(w1)
    primitives, starts, offsets = groups
    partial = torch.empty((len(primitives),3524),device=w1.device,dtype=torch.float32)
    result = torch.empty((n,3524),device=w1.device,dtype=torch.float32)
    nonfinite = torch.zeros((),device=w1.device,dtype=torch.int32)
    precision = [torch.float32,torch.float16,torch.bfloat16].index(dtype)
    if len(primitives):
        _grouped_vjp[(len(primitives),)](csr.x6,csr.grad4,csr.relu_masks,csr.offsets,primitives,starts,freq,attenuation,
            w1,b1,w2,b2,w3,partial,PRECISION=precision,SCALE=scale,
            SEGMENT=groups.segment_size,BLOCK=block_size,EXACT_MASKS=csr.relu_masks is not None,num_warps=4)
    if n:
        _reduce_partials[(n,triton.cdiv(3524,256))](partial,offsets,result,nonfinite,BLOCK=256,num_warps=4)
    return result, nonfinite


@torch.no_grad()
def grouped_weight_gradients(csr, freq, attenuation, weights, *, dtype=torch.float16,
                             segment_size=512, block_size=32, loss_scale=None, groups=None):
    """Ada-compatible Tensor Core path for the released 72→32→32→4 MLP.

    float32 selects an IEEE-FP32 oracle (no TF32). Half/BF16 selects true MMA
    operands and FP32 accumulation. FP16 adjoints are scaled before casting to
    avoid underflow on full-image mean losses, and unscaled in FP32 afterwards.
    A fused finite check preserves finite results unchanged and retries any
    nonfinite mixed result in unscaled IEEE FP32 before returning gradients.
    Master parameters are never mutated or cast in place. Runtime fallback
    warnings and get_grouped_vjp_safety_stats() make recovery visible in reports.
    """
    w1,b1,w2,b2,w3,b3 = weights
    n = len(w1)
    if [tuple(w.shape[1:]) for w in weights] != [(32,72),(32,),(32,32),(32,),(4,32),(4,)]:
        raise ValueError("grouped backward requires the 72→32→32→4 architecture")
    if any(len(w) != n for w in weights) or csr.offsets.numel() != n + 1:
        raise ValueError("CSR offsets and all master tensors must have the same primitive count")
    if csr.x6.ndim != 2 or csr.x6.shape[1] != 6 or csr.grad4.shape != (len(csr.x6), 4):
        raise ValueError("CSR inputs/adjoints must have shapes (records,6) and (records,4)")
    if freq.numel()!=6 or attenuation.numel()!=6:
        raise ValueError("grouped backward requires six positional frequencies")
    if dtype not in [torch.float32,torch.float16,torch.bfloat16]:
        raise ValueError("unsupported GEMM operand dtype")
    if block_size not in (32,64) or segment_size<=0 or segment_size % block_size:
        raise ValueError("segment size must be a positive multiple of the 32/64-row block")
    for tensor in [csr.offsets,csr.x6,csr.grad4,freq,attenuation,*weights]:
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.device!=w1.device:
            raise ValueError("all inputs must be contiguous tensors on one CUDA device")
    if any(t.dtype != torch.float32 for t in [csr.x6,csr.grad4,freq,attenuation,*weights]):
        raise ValueError("master weights, records and encoding parameters must be FP32")
    if csr.relu_masks is not None and (
        csr.relu_masks.dtype!=torch.int32 or not csr.relu_masks.is_contiguous()
        or csr.relu_masks.device!=w1.device or csr.relu_masks.shape!=(len(csr.x6),2)
    ):
        raise ValueError("saved ReLU masks must be contiguous CUDA int32 with shape (records,2)")
    groups = resolve_groups(csr,segment_size,groups)
    scale = float(4096 if dtype==torch.float16 else 1) if loss_scale is None else float(loss_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("loss_scale must be finite and strictly positive")
    result = _checked_grouped_vjp(
        lambda operand_dtype, operand_scale: _launch_grouped_vjp(
            csr,freq,attenuation,weights,groups,operand_dtype,block_size,operand_scale),
        dtype,scale)
    sizes = [2304,32,1024,32,128,4]
    return [v.reshape(w.shape).contiguous() for v,w in zip(result.split(sizes,dim=1),weights)]
