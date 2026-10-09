"""Candidate-count expansion: no truncation, stable order and backward parity."""
import pytest
import torch
from test_cuda_renderer import tile_ext
from test_training_wmma import _case, _args, _render, _reference, _record_parity

@pytest.mark.parametrize('n', [768, 769, 1024, 1536])
def test_full_candidate_tail_and_gradients(tile_ext, n):
    data = _case(n, 1, 1, constant=True)
    # All candidates except the LAST are nonintersecting. Truncating the list
    # at the old capacity loses the only contributor for n > 768.
    data['positions'].zero_(); data['positions'][:, 0] = 100.
    data['positions'][-1] = torch.tensor([0., 0., 1.], device='cuda')
    data['log_scales'].zero_(); data['quats'].zero_(); data['quats'][:, 0] = 1.
    data['rays_o'].zero_(); data['rays_d'].zero_(); data['rays_d'][:, 2] = 1.
    data['b1'].fill_(.125); data['b2'].fill_(.25)
    records = _render(tile_ext, data, 1, 1, 'float16')
    assert records[2].tolist() == [n - 1]
    _record_parity(records, _reference(tile_ext, data, 1, 1), precision='float16', exact=True)
    reference = tile_ext.render_tiles_backward_tilecenter_full(*_args(data), 1, 1, 8, 8, .001)
    d4, gp, gs, gq = tile_ext.primitive_records_reverse(
        records[2], records[7], records[3], records[4], data['log_scales'], data['quats'],
        data['grad_rgb'], data['grad_t'])
    for actual, expected in zip((gp, gs, gq), reference[6:]):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
    from mip360.primitive_csr import group_records
    from mip360.primitive_gemm import grouped_weight_gradients
    from test_training_wmma import WEIGHTS
    csr = group_records(records[2], records[3], d4, n, records[5])
    grads = grouped_weight_gradients(csr, data['freq'], data['attenuation'],
                                   [data[k] for k in WEIGHTS], dtype=torch.float32)
    for actual, expected in zip(grads, reference[:6]):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)

def test_1537_still_rejected(tile_ext):
    with pytest.raises(RuntimeError, match='tile candidate capacity 1536 exceeded'):
        _render(tile_ext, _case(1537, 1, 1, constant=True), 1, 1, 'float16')
