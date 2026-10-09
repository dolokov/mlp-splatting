"""Primitive-major training records and an independent FP32 gradient reference.

The reference uses one PyTorch GEMM per primitive for numerical validation.
Training uses the grouped CUDA implementation in primitive_gemm.
"""

from dataclasses import dataclass

import torch

from .cumodel import _encode_x6


@dataclass
class PrimitiveCSR:
    offsets: torch.Tensor
    record_indices: torch.Tensor
    x6: torch.Tensor
    grad4: torch.Tensor
    relu_masks: torch.Tensor | None = None


def group_records(primitive_ids, x6, grad4, num_primitives: int, relu_masks=None) -> PrimitiveCSR:
    """Group records by primitive, preserving pixel order inside each group."""
    order = torch.argsort(primitive_ids, stable=True)
    counts = torch.bincount(primitive_ids.to(torch.int64), minlength=num_primitives)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return PrimitiveCSR(offsets, order, x6[order], grad4[order],
                        None if relu_masks is None else relu_masks[order])


@torch.no_grad()
def reference_weight_gradients(csr: PrimitiveCSR, freq, attenuation, weights):
    """Return W1,b1,W2,b2,W3,b3 gradients with one owner per primitive.

    grad4 is the pre-activation output adjoint emitted by the original FP32
    compositing backward, so sigmoid/softplus must not be differentiated twice.
    Geometry retains the original renderer's gradient contract.
    """
    w1, b1, w2, b2, w3, b3 = weights
    grads = [torch.zeros_like(w) for w in weights]
    offsets = csr.offsets.cpu().tolist()
    for p, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if begin == end:
            continue
        encoded = _encode_x6(csr.x6[begin:end], freq, attenuation)
        h1 = torch.relu(encoded @ w1[p].T + b1[p])
        h2 = torch.relu(h1 @ w2[p].T + b2[p])
        d3 = csr.grad4[begin:end]
        d2 = (d3 @ w3[p]) * (h2 > 0)
        d1 = (d2 @ w2[p]) * (h1 > 0)
        grads[0][p] = d1.T @ encoded
        grads[1][p] = d1.sum(0)
        grads[2][p] = d2.T @ h1
        grads[3][p] = d2.sum(0)
        grads[4][p] = d3.T @ h2
        grads[5][p] = d3.sum(0)
    return grads
