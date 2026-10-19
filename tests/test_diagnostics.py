import struct
from unittest.mock import Mock

import pytest
from PIL import Image

from drone_scene.common import PipelineError, read_json, write_json
from drone_scene.diagnostics import coverage, generate_report, open_report, point_statistics


def fixture_report(tmp_path, model_factory, points=120):
    names = ['a/00000001.jpg', 'b/00000001.jpg']
    model, images = model_factory('diagnostic-model', names)
    with (model / 'points3D.bin').open('wb') as handle:
        handle.write(struct.pack('<Q', points))
        for i in range(points):
            handle.write(struct.pack('<QdddBBBdQiiii', i, 0., 0., 1., 1, 2, 3, 0.5, 2,
                                     images[0].image_id, 0, images[1].image_id, 0))
    videos = []
    for clip in ('a', 'b'):
        target = tmp_path / 'raw' / clip / '00000001.jpg'
        target.parent.mkdir(parents=True)
        Image.new('RGB', (32, 24), 'red').save(target)
        videos.append({'clip': clip, 'source': '<script>alert(1)</script>', 'duration': 10.,
                       'selected': [{'name': target.name, 'timestamp_seconds': 2., 'sharpness': 12.}],
                       'dropped': {'blur': 1}, 'runners_up': []})
    write_json(tmp_path / 'reports/extraction.json', {'videos': videos,
               'retained': {'a': [names[0]], 'b': [names[1]]}})
    write_json(tmp_path / 'reports/reconstruction.json', {
        'components': [{'model': str(model), 'registered': 2, 'per_video': {'a': 1, 'b': 1}}],
        'failures': []})
    return {'extraction': {'fps': 1}, 'sfm': {}}, model, images


def test_geometry_and_self_contained_html(tmp_path, model_factory):
    cfg, _, _ = fixture_report(tmp_path, model_factory)
    path, data = generate_report(tmp_path, cfg)
    assert data['geometry']['point_count'] == 120
    assert data['geometry']['shared_landmarks'] == {'a / b': 120}
    assert data['geometry']['track_length']['median'] == 2
    text = path.read_text()
    assert '<svg' in text and 'data:image/jpeg;base64,' in text
    assert '<script>alert' not in text
    assert '&lt;script&gt;' in text
    assert data['clips'][0]['selection']['end_gap_seconds'] == 8
    assert data['status'] == 'WARN'
    assert read_json(tmp_path / 'reports/diagnostics.json')['stage'] == 'reconstruction'


def test_configured_gate_and_original_failure_are_visible(tmp_path, model_factory):
    cfg, _, _ = fixture_report(tmp_path, model_factory)
    cfg['sfm']['quality_gate'] = {'min_median_track_length': 3, 'max_registered_gap_seconds': 4}
    path, data = generate_report(tmp_path, cfg, error='Original reconstruction failure')
    assert data['status'] == 'FAIL'
    assert len(data['failures']) == 3
    assert 'Original reconstruction failure' in path.read_text()


def test_truncated_points_fail_clearly(tmp_path, model_factory):
    _, model, images = fixture_report(tmp_path, model_factory)
    (model / 'points3D.bin').write_bytes(struct.pack('<Q', 1))
    with pytest.raises(PipelineError, match='Truncated'):
        point_statistics(model / 'points3D.bin', images)


def test_extraction_only_and_legacy_timestamps(tmp_path):
    write_json(tmp_path / 'reports/extraction.json', {'videos': [
        {'clip': 'a', 'source': 'input.mp4', 'duration': 5, 'dropped': {}}],
        'retained': {'a': ['a/00000003.jpg']}})
    _, data = generate_report(tmp_path, {'extraction': {'fps': 2}, 'sfm': {}})
    assert data['stage'] == 'extraction'
    assert data['geometry'] is None
    assert data['clips'][0]['frames'][0]['timestamp_seconds'] == 1
    assert data['clips'][0]['approximate_legacy_timestamps']


def test_empty_and_boundary_coverage():
    assert coverage([], 10)['max_gap_seconds'] == 10
    assert coverage([1, 2, 8], 10) == {
        'start_gap_seconds': 1, 'end_gap_seconds': 2,
        'max_internal_gap_seconds': 6, 'max_gap_seconds': 6}


def test_firefox_is_detached_and_missing_browser_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr('drone_scene.diagnostics.shutil.which', lambda _: '/usr/bin/firefox')
    popen = Mock()
    monkeypatch.setattr('drone_scene.diagnostics.subprocess.Popen', popen)
    path = tmp_path / 'report space.html'
    open_report(path)
    assert popen.call_args.args[0] == ['/usr/bin/firefox', path.as_uri()]
    assert popen.call_args.kwargs['start_new_session'] is True
    popen.assert_called_once()
    monkeypatch.setattr('drone_scene.diagnostics.shutil.which', lambda _: None)
    open_report(path)
    popen.assert_called_once()


def test_zero_points_are_unavailable_and_fail(tmp_path, model_factory):
    cfg, _, _ = fixture_report(tmp_path, model_factory, points=0)
    _, data = generate_report(tmp_path, cfg)
    assert data['status'] == 'FAIL'
    assert data['geometry']['stored_point_error_pixels']['median'] is None


def test_pipeline_failure_writes_and_opens_report_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from drone_scene import cli
    from drone_scene.common import Runner

    source = tmp_path / 'input.mp4'
    source.touch()
    work = tmp_path / 'workspace'
    cfg = read_json(cli.backends.ROOT / 'configs/pilot.json')
    write_json(work / 'reports/extraction.json', {'videos': [
        {'clip': 'a', 'source': str(source), 'duration': 2, 'selected': [], 'dropped': {}}],
        'retained': {'a': []}})
    monkeypatch.setattr(cli.backends, 'doctor', lambda *a, **kw: None)
    def fake_stage(runner, name, signature, outputs, action):
        if name == 'sfm':
            raise PipelineError('No overlapping model')
    monkeypatch.setattr(cli, 'stage', fake_stage)
    opened = Mock()
    monkeypatch.setattr(cli, 'open_report', opened)
    args = SimpleNamespace(videos=[str(source)], until='sfm', no_open_report=False)
    with pytest.raises(PipelineError, match='No overlapping model'):
        cli.run_pipeline(args, cfg, Runner(work))
    opened.assert_called_once()
    data = read_json(work / 'reports/diagnostics.json')
    assert data['status'] == 'FAIL'
    assert 'No overlapping model' in data['failures']


@pytest.mark.parametrize('value', [0, -1, True, float('nan')])
def test_invalid_quality_thresholds(tmp_path, value):
    from drone_scene import cli
    from drone_scene.config import load_config

    cfg = read_json(cli.backends.ROOT / 'configs/pilot.json')
    cfg['sfm']['quality_gate'] = {'max_point_error_p95': value}
    path = tmp_path / 'config.json'
    write_json(path, cfg)
    with pytest.raises(PipelineError):
        load_config(path)
