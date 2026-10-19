import shutil
import subprocess

import numpy as np
import pytest

from drone_scene.common import PipelineError
from drone_scene.web_viewer import ASSETS, pack_scene


def state():
    return {'positions': np.array([[0, 0, 2]], dtype=np.float32),
            'log_scales': np.zeros((1, 3)), 'quats': np.array([[2, 0, 0, 0]]),
            'W1': np.zeros((1, 2, 12)), 'b1': np.zeros((1, 2)),
            'W2': np.zeros((1, 2, 2)), 'b2': np.zeros((1, 2)),
            'W3': np.zeros((1, 4, 2)), 'b3': np.array([[0, 0, 0, 1]])}


def test_pack_matches_shader_layout():
    packed, manifest = pack_scene(state(), [1], [.5])
    assert manifest['stride_floats'] == 54
    assert manifest['byte_length'] == 56 * 4
    np.testing.assert_array_equal(packed[:10], [0, 0, 2, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(packed[-6:], [0, 0, 0, 1, 1, .5])
    assert packed.dtype == np.dtype('<f4')


@pytest.mark.parametrize('key,value', [
    ('positions', np.array([[float('nan'), 0, 0]])),
    ('quats', np.zeros((1, 4))), ('W3', np.zeros((1, 3, 2))),
])
def test_invalid_export_fails(key, value):
    data = state()
    data[key] = value
    with pytest.raises(PipelineError):
        pack_scene(data, [1], [1])


def test_shader_template_substitution():
    import re

    shader = (ASSETS / 'render.wgsl').read_text()
    replacements = {'N': 1, 'H': 2, 'F': 1, 'INPUT': 12}
    shader = re.sub(r'\$(INPUT|N|H|F)\b', lambda m: str(replacements[m[1]]), shader)
    assert '$' not in shader
    assert 'array<f32, 12>' in shader
    assert 'textureStore' in shader


@pytest.mark.skipif(not shutil.which('node'), reason='Node not installed')
def test_browser_javascript_syntax():
    subprocess.run(['node', '--check', str(ASSETS / 'viewer.js')], check=True)
