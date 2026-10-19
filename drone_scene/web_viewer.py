"""Export native MLP weights and serve a browser-GPU reference viewer (no streaming)."""
from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import sys
from pathlib import Path

import numpy as np

from .backends import checkpoint, upstream
from .common import PipelineError, read_json, write_json
from .mlp_dataset import selected_image_names

ASSETS = Path(__file__).resolve().parent / 'web'


def pack_scene(state, frequencies, attenuation):
    """Pack little-endian FP32 primitive records shared by the WGSL reference renderer."""
    arrays = {key: np.asarray(state[key], dtype=np.float32) for key in
              ('positions', 'log_scales', 'quats', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3')}
    n, hidden, encoded = arrays['W1'].shape
    frequencies = np.asarray(frequencies, dtype=np.float32)
    attenuation = np.asarray(attenuation, dtype=np.float32)
    if not (n > 0 and 0 < hidden <= 64 and 0 < len(frequencies) <= 10
            and encoded == len(frequencies) * 12 and attenuation.shape == frequencies.shape):
        raise PipelineError('Unsupported WebGPU scene dimensions (H<=64, F<=10 required)')
    expected = {'positions': (n, 3), 'log_scales': (n, 3), 'quats': (n, 4),
                'W1': (n, hidden, encoded), 'b1': (n, hidden),
                'W2': (n, hidden, hidden), 'b2': (n, hidden),
                'W3': (n, 4, hidden), 'b3': (n, 4)}
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise PipelineError(f'Invalid WebGPU tensor {key}: expected finite {shape}')
    lengths = np.linalg.norm(arrays['quats'], axis=1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise PipelineError('Cannot export zero-length primitive quaternion')
    arrays['quats'] = arrays['quats'] / lengths
    if not np.isfinite(frequencies).all() or not np.isfinite(attenuation).all():
        raise PipelineError('Invalid encoding frequencies/attenuation')
    records = np.concatenate([arrays[k].reshape(n, -1) for k in expected], axis=1)
    packed = np.concatenate([records.ravel(), frequencies, attenuation]).astype('<f4')
    return packed, {'version': 1, 'count': n, 'hidden': hidden, 'frequencies': len(frequencies),
                    'stride_floats': records.shape[1], 'byte_length': packed.nbytes}


def export_scene(cfg, work, output):
    import torch

    if cfg['backend'] != 'mlp':
        raise PipelineError('The WebGPU prototype exports native MLP checkpoints only')
    repo = upstream('mlp')
    sys.path.insert(0, str(repo))
    from scannet.camera import validate_checkpoint_calibration
    from scannet.checkpoints import infer_scene_spec, normalize_scene_state_dict, split_checkpoint
    from scannet.scannet_loader import SceneDataset

    source = checkpoint(work, cfg)
    raw = torch.load(source, map_location='cpu', weights_only=True)
    state = normalize_scene_state_dict(split_checkpoint(raw).scene_state_dict)
    spec = infer_scene_spec(state)
    metadata = raw.get('metadata', {})
    kind = metadata.get('encoding_kind', 'attenuated_phase_v1')
    if kind not in {'attenuated_phase_v1', 'fourier_features_v1'}:
        raise PipelineError(f'Unsupported encoding: {kind}')
    dataset = SceneDataset(str(work / 'scene'), factor=1, llffhold=cfg['training']['test_every'],
                           split='train', feature_type='none', ray_cache_size=1,
                           split_mode='llff', split_test_offset=0, camera_mode='colmap-native')
    validate_checkpoint_calibration(raw, dataset)
    camera = dataset.camera_intrinsics
    if any(abs(float(v)) > 1e-8 for v in camera['distortion']):
        raise PipelineError('WebGPU prototype requires undistorted pinhole cameras')
    freq = state.get('encoder.freq_bands')
    if freq is None:
        freq = 2. ** torch.arange(spec.pe_frequencies) * (torch.pi if kind == 'attenuated_phase_v1' else 1)
    freq = freq.detach().float().cpu().numpy()
    att = (np.exp(-0.5 * freq ** 2 * float(metadata.get('encoding_sigma', .01)) ** 2)
           if kind == 'attenuated_phase_v1' else np.ones_like(freq))
    packed, manifest = pack_scene({k: v.detach().float().cpu().numpy() for k, v in state.items()
                                  if torch.is_tensor(v)}, freq, att)
    manifest.update({'camera': camera, 'threshold': .003, 'encoding_kind': kind,
                     'cameras': [{'name': Path(name).name, 'pose': pose.tolist()}
                                 for name, pose in zip(selected_image_names(dataset), dataset.poses)],
                     'source_checkpoint': source.name,
                     'limitations': ['Brute-force ray candidates', '128 hits/ray; magenta on overflow',
                                     'No CUDA/browser pixel parity certification', 'Black background']})
    if output.exists() and any(output.iterdir()):
        raise PipelineError(f'Export directory is not empty: {output}; choose a new --output')
    output.mkdir(parents=True, exist_ok=True)
    packed.tofile(output / 'scene.bin')
    write_json(output / 'scene.json', manifest)
    for name in ('index.html', 'viewer.js', 'render.wgsl'):
        shutil.copyfile(ASSETS / name, output / name)
    return manifest


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Serve only regular export assets, with correct shader/module content types."""
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.js': 'text/javascript', '.wgsl': 'text/plain', '.json': 'application/json'}

    def do_GET(self):
        from urllib.parse import unquote, urlsplit
        allowed = {'/', '/index.html', '/viewer.js', '/render.wgsl', '/scene.json', '/scene.bin'}
        if unquote(urlsplit(self.path).path) not in allowed:
            self.send_error(404)
            return
        super().do_GET()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--serve', action='store_true', help='Serve exported files on localhost')
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise PipelineError('port must be between 1 and 65535')
        cfg = read_json(args.config.resolve())
        manifest = export_scene(cfg, args.workspace.resolve(), args.output.resolve())
        print(f'Exported {manifest["count"]} primitives ({manifest["byte_length"] / 2**20:.1f} MiB) to {args.output.resolve()}')
        if args.serve:
            handler = functools.partial(ViewerHandler, directory=str(args.output.resolve()))
            with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
                print(f'Open http://localhost:{args.port}/ in a WebGPU-capable browser. Ctrl+C stops serving.', flush=True)
                server.serve_forever()
        return 0
    except (PipelineError, OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
