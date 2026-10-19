"""Offline, dependency-free quality diagnostics and SVG/HTML reporting."""
from __future__ import annotations

import base64
import html
import io
import itertools
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .colmap_model import read_images, safe_name, unpack
from .common import PipelineError, read_json, write_json


def coverage(times, duration):
    times = sorted(set(times))
    boundaries = [0., *times, duration]
    return {"start_gap_seconds": times[0] if times else duration,
            "end_gap_seconds": duration - times[-1] if times else duration,
            "max_internal_gap_seconds": max((b - a for a, b in zip(times, times[1:])), default=0),
            "max_gap_seconds": max((b - a for a, b in zip(boundaries, boundaries[1:])), default=0)}


def point_statistics(path, images):
    clips = {im.image_id: str(safe_name(im.name).parts[0]) for im in images}
    errors, lengths, shared = [], [], {}
    with Path(path).open('rb') as handle:
        count, = unpack(handle, 'Q')
        if count > 100_000_000:
            raise PipelineError('Invalid point count')
        for _ in range(count):
            record = unpack(handle, 'QdddBBBd')
            error = record[-1]
            if not math.isfinite(error) or error < 0:
                raise PipelineError('Invalid stored point reprojection error')
            length, = unpack(handle, 'Q')
            if length > 10_000_000:
                raise PipelineError('Invalid point track length')
            observed = set()
            for _ in range(length):
                image_id, _ = unpack(handle, 'ii')
                if image_id not in clips:
                    raise PipelineError('Point track references missing image')
                observed.add(clips[image_id])
            for a, b in itertools.combinations(sorted(observed), 2):
                key = a + ' / ' + b
                shared[key] = shared.get(key, 0) + 1
            errors.append(error)
            lengths.append(length)
    def distribution(values):
        if not values:
            return {"count": 0, "median": None, "p95": None, "histogram": []}
        counts, edges = np.histogram(values, bins=min(20, max(1, len(values))))
        return {"count": len(values), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "histogram": [[float(edges[i]), float(edges[i + 1]), int(n)]
                              for i, n in enumerate(counts)]}
    return {"point_count": count, "stored_point_error_pixels": distribution(errors),
            "track_length": distribution(lengths), "shared_landmarks": shared}


def collect(work, cfg, error=None):
    extraction = read_json(work / 'reports/extraction.json')
    reconstruction_path = work / 'reports/reconstruction.json'
    reconstruction = read_json(reconstruction_path) if reconstruction_path.exists() else None
    registered, geometry = set(), None
    warnings, failures = [], list(reconstruction.get('failures', [])) if reconstruction else []
    if error:
        failures.append(str(error))
    components = reconstruction.get('components', []) if reconstruction else []
    if components:
        model = Path(max(components, key=lambda c: c['registered'])['model'])
        try:
            images = read_images(model / 'images.bin')
            registered = {im.name for im in images}
            geometry = point_statistics(model / 'points3D.bin', images)
        except (PipelineError, OSError, ValueError) as exc:
            failures.append(f'Geometry diagnostics unavailable: {exc}')
    clips = []
    for video in extraction['videos']:
        clip = video['clip']
        selected = video.get('selected')
        approximate = selected is None
        if approximate:
            selected = [{"name": Path(name).name,
                         "timestamp_seconds": (int(Path(name).stem) - 1) / cfg['extraction']['fps'],
                         "sharpness": None} for name in extraction['retained'][clip]]
        frames = [{**s, "registered": f"{clip}/{s['name']}" in registered} for s in selected]
        duration = video['duration']
        kept_times = [s['timestamp_seconds'] for s in frames]
        solved_times = [s['timestamp_seconds'] for s in frames if s['registered']]
        clips.append({"clip": clip, "source": video['source'], "duration": duration,
                      "approximate_legacy_timestamps": approximate, "frames": frames,
                      "selection": coverage(kept_times, duration),
                      "registration": coverage(solved_times, duration) if reconstruction else None,
                      "dropped": video['dropped'], "runners_up": video.get('runners_up', [])})
        threshold = cfg['extraction'].get('max_gap_seconds',
                                        max(2., 2 * cfg['extraction'].get('selection_window_seconds', 0),
                                            2 / cfg['extraction']['fps']))
        if clips[-1]['selection']['max_gap_seconds'] > threshold:
            warnings.append(f'{clip}: selection gap exceeds {threshold:g} seconds; inspect capture or shorten windows.')
        if reconstruction and clips[-1]['registration']['max_gap_seconds'] > threshold:
            warnings.append(f'{clip}: registration gap exceeds {threshold:g} seconds; inspect matching and coverage.')
    if reconstruction and not components:
        failures.append('No reconstruction component is available.')
    if geometry:
        if geometry['point_count'] < 100:
            failures.append('Fewer than 100 triangulated points.')
        if geometry['stored_point_error_pixels']['p95'] is None:
            warnings.append('Point error statistics unavailable: no reconstructed points.')
        elif geometry['stored_point_error_pixels']['p95'] > 4:
            warnings.append('Stored point error p95 exceeds 4 px (heuristic); inspect calibration and matches.')
        if geometry['track_length']['median'] is not None and geometry['track_length']['median'] < 3:
            warnings.append('Median track length is below 3 (heuristic); inspect landmark support.')
        if len(clips) > 1:
            connected = {clips[0]['clip']}
            edges = [key.split(' / ') for key in geometry['shared_landmarks']]
            for _ in clips:
                for a, b in edges:
                    if a in connected or b in connected:
                        connected.update((a, b))
            if len(connected) < len(clips):
                warnings.append('Shared-landmark graph does not connect every source clip; inspect bridging views.')
    policy = cfg['sfm'].get('quality_gate', {})
    checks = [("max_point_error_p95", geometry['stored_point_error_pixels']['p95'] if geometry else None, 'max'),
              ("min_median_track_length", geometry['track_length']['median'] if geometry else None, 'min'),
              ("max_registered_gap_seconds", max((c['registration']['max_gap_seconds'] for c in clips
                                                  if c['registration']), default=None), 'max')]
    for key, value, direction in checks:
        if key in policy and reconstruction:
            if value is None or (value > policy[key] if direction == 'max' else value < policy[key]):
                failures.append(f'Quality gate {key}: measured {value}, required {policy[key]}.')
    return {"status": 'FAIL' if failures else 'WARN' if warnings else 'PASS',
            "stage": 'reconstruction' if reconstruction else 'extraction',
            "warnings": warnings, "failures": failures, "clips": clips,
            "components": components, "geometry": geometry}


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def histogram(distribution, label):
    bins = distribution['histogram']
    if not bins:
        return '<p>Unavailable: no points.</p>'
    maximum = max(b[2] for b in bins) or 1
    bars = ''.join(f'<rect x="{i * 600 / len(bins):.2f}" y="{140 - n / maximum * 130:.2f}" '
                   f'width="{max(1, 600 / len(bins) - 2):.2f}" height="{n / maximum * 130:.2f}" fill="#5aa9e6">'
                   f'<title>{lo:.3g}–{hi:.3g}: {n}</title></rect>'
                   for i, (lo, hi, n) in enumerate(bins))
    return (f'<svg role="img" aria-label="{esc(label)}" viewBox="0 0 600 165">{bars}'
            f'<text x="0" y="160">{bins[0][0]:.3g}</text>'
            f'<text x="520" y="160">{bins[-1][1]:.3g}</text></svg>')


def render(work, data):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Pipeline diagnostics</title>',
             '<style>body{font:16px system-ui;margin:32px;background:#101827;color:#e5edf5}'
             'section{background:#1c293b;padding:20px;margin:20px 0;border-radius:12px}'
             'table{border-collapse:collapse;width:100%;margin:12px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #41516a}'
             'th{background:#263951}tbody tr:nth-child(even){background:#223146}'
             '.PASS{color:#60df9b}.WARN{color:#ffd166}.FAIL{color:#ff7b86}'
             'svg{width:100%;max-width:900px;background:#142035}svg text{fill:#e5edf5;font-size:12px}'
             '.gallery{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:180px}img{max-width:180px}'
             'figcaption{font-size:12px;overflow-wrap:anywhere}.heat{display:inline-block;padding:12px;margin:4px;background:#294b68}'
             '</style>', f'<h1>Pipeline diagnostics — <span class="{data["status"]}">{data["status"]}</span></h1>',
             f'<p>Stage: {esc(data["stage"])}. Timestamps use the sampling grid, not exact source PTS. '
             'Warnings are heuristic unless a hard quality gate is configured.</p>']
    for kind in ('failures', 'warnings'):
        parts.append(f'<section class="{"FAIL" if kind == "failures" else "WARN"}"><h2>{kind.title()}</h2><ul>' +
                     ''.join(f'<li>{esc(item)}</li>' for item in data[kind]) + '</ul>' +
                     ('None.' if not data[kind] else '') + '</section>')
    parts.append('<section><h2>Registration summary</h2><table><tr><th>Clip</th><th>Retained</th>'
                 '<th>Registered</th><th>Registered fraction</th></tr>')
    total_kept = total_registered = 0
    for clip in data['clips']:
        kept = len(clip['frames'])
        solved = sum(f['registered'] for f in clip['frames'])
        total_kept += kept
        total_registered += solved
        ratio = solved / kept if kept else 0
        color = 'PASS' if ratio >= .85 else 'WARN' if ratio >= .6 else 'FAIL'
        value = f'{ratio:.1%}' if data['stage'] == 'reconstruction' else 'Unavailable'
        parts.append(f'<tr><td>{esc(clip["clip"])}</td><td>{kept}</td><td>'
                     f'{solved if data["stage"] == "reconstruction" else "Unavailable"}</td>'
                     f'<td class="{color if data["stage"] == "reconstruction" else ""}">{value}</td></tr>')
    overall = f'{total_registered / total_kept:.1%}' if total_kept and data['stage'] == 'reconstruction' else 'Unavailable'
    parts.append(f'</table><p>Overall registration: {overall}. Table colors use display-only '
                 '85% / 60% bands; configured registration gates determine acceptance.</p></section>')
    for clip in data['clips']:
        parts.append(f'<section><h2>{esc(clip["clip"])}</h2><p>{esc(clip["source"])}</p>')
        if clip['approximate_legacy_timestamps']:
            parts.append('<p class="WARN">Legacy report: timestamps inferred approximately from filenames and configured FPS.</p>')
        parts.append(table(['Coverage', 'Start gap (s)', 'End gap (s)', 'Largest internal gap (s)', 'Largest gap (s)'],
                           [[label, *[f'{v:.2f}' for v in stats.values()]]
                            for label, stats in [('Selected', clip['selection']), ('Registered', clip['registration'])] if stats]))
        frames, duration = clip['frames'], max(clip['duration'], 0.001)
        parts.append('<h3>Temporal coverage</h3><p>Gray: no observation; yellow: retained; green: registered. Hover for timestamp.</p>')
        parts.append('<svg viewBox="0 0 900 50"><rect width="900" height="28" fill="#485568"/>')
        for frame in frames:
            color = '#60df9b' if frame['registered'] else '#ffd166'
            parts.append(f'<rect x="{frame["timestamp_seconds"] / duration * 895:.2f}" width="3" height="28" fill="{color}">'
                         f'<title>{esc(frame["name"])}: {frame["timestamp_seconds"]:.2f}s</title></rect>')
        parts.append(f'<text x="0" y="46">0 s</text><text x="820" y="46">{duration:.2f} s</text></svg>')
        # Fixed-size occupancy bins make long empty intervals visible without enormous plots.
        parts.append('<h3>Selected / registered frames per time bucket</h3>')
        edges = np.linspace(0, duration, 21)
        selected_counts = np.histogram([f['timestamp_seconds'] for f in frames], edges)[0]
        registered_counts = np.histogram([f['timestamp_seconds'] for f in frames if f['registered']], edges)[0]
        parts.append(table(['Interval (s)', 'Selected', 'Registered'],
                           [[f'{edges[i]:.1f}–{edges[i+1]:.1f}', int(a),
                             int(registered_counts[i]) if data['stage'] == 'reconstruction' else 'Unavailable']
                            for i, a in enumerate(selected_counts)]))
        scored = [f for f in frames if f['sharpness'] is not None]
        parts.append('<h3>Selected sharpness over time</h3>')
        if scored:
            maximum = max(f['sharpness'] for f in scored) or 1
            parts.append('<svg viewBox="0 0 900 150">' + ''.join(
                f'<circle cx="{f["timestamp_seconds"] / duration * 890 + 5:.2f}" '
                f'cy="{135 - f["sharpness"] / maximum * 125:.2f}" r="3" fill="#5aa9e6">'
                f'<title>{f["timestamp_seconds"]:.2f}s: {f["sharpness"]:.2f}</title></circle>' for f in scored) +
                f'<text x="4" y="148">0–{duration:.1f}s; sharpness 0–{maximum:.1f}</text></svg>')
        else:
            parts.append('<p>Unavailable.</p>')
        parts.append(table(['Rejection reason', 'Count'], clip['dropped'].items()))
        parts.append('<h3>Contact sheet (up to 24 evenly distributed selected views)</h3><div class="gallery">')
        for index in np.unique(np.linspace(0, max(0, len(frames) - 1), min(24, len(frames)), dtype=int)):
            frame = frames[index]
            try:
                name = safe_name(f'{clip["clip"]}/{frame["name"]}')
                with Image.open(work / 'raw' / str(name)) as image:
                    image.thumbnail((180, 110))
                    buffer = io.BytesIO()
                    image.convert('RGB').save(buffer, format='JPEG', quality=65)
                encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
                parts.append(f'<figure><img alt="{esc(frame["name"])}" src="data:image/jpeg;base64,{encoded}">'
                             f'<figcaption>{esc(frame["name"])} · {frame["timestamp_seconds"]:.2f}s · '
                             f'sharpness {esc(frame["sharpness"])}</figcaption></figure>')
            except (OSError, PipelineError):
                parts.append('<p>Thumbnail unavailable.</p>')
        parts.append('</div><details><summary>Runner-up metadata (bounded sample)</summary>' +
                     table(['Filename', 'Time (s)', 'Sharpness', 'Reason'],
                           [[r['name'], r['timestamp_seconds'], r['sharpness'], r['reason']]
                            for r in clip['runners_up']]) + '</details></section>')
    parts.append('<section><h2>Reconstruction components</h2>' + table(
        ['Model', 'Registered', 'Per-clip counts'],
        [[c['model'], c['registered'], c['per_video']] for c in data['components']]) + '</section>')
    geometry = data['geometry']
    if geometry:
        parts.append('<section><h2>Landmark geometry</h2><p>Stored COLMAP per-point errors, not independently recomputed observation residuals.</p>')
        for key, title in [('stored_point_error_pixels', 'Stored point error (pixels)'), ('track_length', 'Track length (observations)')]:
            stats = geometry[key]
            parts.append(f'<h3>{title}</h3>' + table(['Count', 'Median', '95th percentile'],
                         [[stats['count'], stats['median'], stats['p95']]]) + histogram(stats, title))
        parts.append('<h3>Cross-flight shared-landmark heatmap</h3>')
        names = [c['clip'] for c in data['clips']]
        peak = max(geometry['shared_landmarks'].values(), default=1)
        parts.append('<table><tr><th>Clip</th>' + ''.join(f'<th>{esc(n)}</th>' for n in names) + '</tr>')
        for a in names:
            parts.append(f'<tr><th>{esc(a)}</th>')
            for b in names:
                n = geometry['shared_landmarks'].get(' / '.join(sorted((a, b))), 0)
                parts.append(f'<td style="background:rgba(40,180,130,{0.1 + 0.8 * n / peak:.2f})">{n if a != b else "—"}</td>')
            parts.append('</tr>')
        parts.append('</table><p>Counts are points observed by both clips; indirect connections are valid. '
                     'No metric-scale or survey-accuracy claim is implied.</p></section>')
    else:
        parts.append('<section><h2>Landmark geometry</h2><p>Unavailable: reconstruction has not completed or could not be read.</p></section>')
    return ''.join(parts) + '</html>'


def generate_report(work, cfg, error=None):
    work = Path(work)
    data = collect(work, cfg, error)
    write_json(work / 'reports/diagnostics.json', data)
    path = work / 'reports/diagnostics.html'
    temporary = path.with_suffix('.html.tmp')
    temporary.write_text(render(work, data), encoding='utf-8')
    temporary.replace(path)
    return path, data


def open_report(path):
    firefox = shutil.which('firefox')
    if not firefox:
        print(f'Warning: Firefox not found. Open report: {path}')
        return
    try:
        subprocess.Popen([firefox, Path(path).resolve().as_uri()], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        print(f'Warning: could not launch Firefox ({exc}). Open report: {path}')
