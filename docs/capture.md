# Capture, calibration, and data quality

## Multiple videos need shared landmarks

Different clips can contribute to one scene only when they see enough of the same static 3D structure. Flying separate areas with no connecting overlap produces separate SfM components, regardless of the splatting method. Use a bridging flight or a co-registered mapping/control-point workflow to connect them.

For a recapture, practical starting targets are roughly 75–85% forward overlap and 65–80% side overlap, supplemented by oblique views of facades. These are capture-planning heuristics, not a guaranteed reconstruction threshold. Include loops and several altitudes when safe. Keep shutter speed fast enough to avoid motion blur. Lock focus, exposure, white balance, focal length, and image size where possible. A compressed moving video is usually a worse mapping source than sharp original still images.

Nadir-only views can reconstruct rooftops/ground while leaving facade holes. High-altitude imagery cannot recover close-up facade detail that was never observed. Trees, water, cars, people, changing shadows, reflections, haze, repeated roofs, and low-texture surfaces all complicate camera solving and appearance fitting.

## Calibration modes

`sfm.camera_mode: shared` assumes **the same camera, lens, focal length, crop, stabilization behavior, and image dimensions** for all clips. It uses a single OPENCV calibration in COLMAP; undistortion then provides pinhole images. This is appropriate for one fixed-optics drone capture if those assumptions hold.

Set `sfm.camera_mode: per-video` for multiple fixed lenses, different resolutions, or different fixed crops **and select gsplat**. Frames are grouped into separate source folders so COLMAP can calibrate each clip independently. The native MLP loader requires one shared calibration and will reject a dataset with differing cameras.

Continuous zoom or electronic stabilization that changes intrinsics within a clip violates both modes. Split stable segments, use original unstabilized recordings, or extend camera modeling first. Do not force all frames into a shared camera just to satisfy MLP's loader.

FFmpeg applies its normal display rotation metadata when sampling; the actual extracted dimensions are checked. Every clip is resized without upscaling. Resize calibration happens through COLMAP on the extracted frames and through its undistorter; arbitrary resizing after SfM would invalidate it.

## Geographic coordinates and scale

COLMAP's monocular reconstruction is defined only up to a similarity transform. Matching videos aligns them to one arbitrary frame; it does not establish meter scale or GPS coordinates. This pipeline does not read telemetry from MP4/SRT, automatically align drone GPS, or provide survey-grade output.

For a known distance or reliable GPS/control points, perform a deliberate model alignment/scale step and verify coordinate conventions before downstream use. Keep enough well-distributed control observations to test the result. For GIS/orthophoto/DEM deliverables, evaluate ODM/WebODM and purpose-captured calibrated stills.

## Filtering

## HTML quality diagnostics

Pipeline runs generate `reports/diagnostics.html` and `reports/diagnostics.json`
after extraction and reconstruction. HTML contains inline SVG plots, colored tables,
bounded embedded contact sheets, temporal occupancy, sharpness, component coverage,
stored COLMAP point-error and track-length histograms, and shared-landmark connectivity.
It works offline without a plotting package. Failed reconstruction quality checks
also generate a report when extraction metadata is available.

Firefox opens the final applicable report in a detached process, including on a
reconstruction failure. Pass `--no-open-report` for headless/automated runs. Missing
Firefox produces a warning, not a failed pipeline. Dry runs do not write or open reports.

Optional configuration fragments (merge into the existing sections):

```json
{
  "extraction": {"max_gap_seconds": 4.0},
  "sfm": {
    "quality_gate": {
      "max_point_error_p95": 4.0,
      "min_median_track_length": 3.0,
      "max_registered_gap_seconds": 6.0
    }
  }
}
```

These numbers are illustrative, not tuned guarantees. Selection gap limits generate
warnings. Explicit `quality_gate` thresholds fail before preparation/training; missing
required metrics also fail. Existing registration and minimum-point checks remain.
Without explicit geometry thresholds, error p95 above 4 pixels and median track
length below 3 are warnings only. Gaps include clip boundaries. Empty windows are
not automatically filled. Stored errors are COLMAP's per-point errors, not independently
recomputed per-observation residuals. Cross-flight heatmap counts points observed in
both clips; indirect connections are valid, and no minimum reliable bridge strength
is asserted. Older extraction reports infer timestamps from filenames and configured
FPS and are labeled approximate. Up to 200 rejected window/duplicate candidates per
clip are recorded as runner-up metadata; rejected JPEGs remain deleted. Contact
sheets include at most 24 evenly spaced retained views per clip.

### Optional sharpest-in-window selection

Existing configs retain fixed-FPS sampling. To select better nearby frames, add these
fields to the `extraction` object in your config:

```json
{
  "fps": 5.0,
  "selection_window_seconds": 2.0,
  "max_candidates": 5000,
  "max_frames": 500
}
```

This is a config fragment; keep the existing size and quality settings. It scores
5 candidates per second and keeps at most one usable frame per 2-second bucket,
including a partial final bucket. Start with 1–2 seconds rather than 10 seconds
to avoid sacrificing viewpoint overlap. These are starting heuristics, not guarantees.
Frames below `min_sharpness` are rejected first. Remaining candidates are ranked
by sharpness, with earliest timestamp breaking ties. If the winner duplicates the
previous retained frame, the next-sharpest candidate is tried. Selection never
crosses clip boundaries, and output remains chronological.

`selection_window_seconds: 0` (or omission) disables windowed selection.
`fps` always controls candidate sampling; `max_candidates` limits decoded candidates
across all clips and defaults to `max_frames`. `max_frames` limits retained frames.
Exceeding either budget fails instead of silently truncating footage. Candidate
JPEGs are temporarily written to disk, so dense sampling increases extraction cost.
Changing these settings requires a new workspace under the existing cache policy.

The extraction report includes each selected frame's filename, sharpness, approximate
sample timestamp, window-rejection counts, and the largest gap between selected
frames in each clip. Timestamps refer to the FFmpeg sampling grid, not exact source
presentation timestamps. Empty/blurry/duplicate windows can produce gaps; no poor
frame is inserted to hide them. Even adjacent populated windows can produce gaps
approaching twice the window size. Inspect the report and shorten windows if needed.
The existing minimum of eight usable frames per clip still applies.

Frame names preserve the sampled index and source clip ID. `reports/extraction.json` records every retained frame name and drop counts per reason. Sampling uses FFmpeg's CFR `fps` filter; timestamps can be inferred approximately from sampled indices/FPS, but are not a lossless record of original variable-frame-rate timestamps.

Laplacian variance is measured on 256×144 grayscale thumbnails. `min_sharpness=8` is a starting value; texture and compression change this score. Lower it toward zero to disable blur rejection. `dedup_mae=0.8` is mean absolute grayscale difference on the same thumbnails; zero disables duplicate rejection. A small score can describe either duplicate frames or a slowly moving, low-texture scene, so inspect retained counts.

Periodic holdout uses every eighth registered view by default. Adjacent video frames can be nearly identical, so this can overestimate novel-view performance. For rigorous evaluation, design a separate flight/trajectory holdout and adapt the backend split rather than reporting periodic-frame metrics as evidence of unseen-area generalization.
