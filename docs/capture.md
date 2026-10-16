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

Frame names preserve the sampled index and source clip ID. `reports/extraction.json` records every retained frame name and drop counts per reason. Sampling uses FFmpeg's CFR `fps` filter; timestamps can be inferred approximately from sampled indices/FPS, but are not a lossless record of original variable-frame-rate timestamps.

Laplacian variance is measured on 256×144 grayscale thumbnails. `min_sharpness=8` is a starting value; texture and compression change this score. Lower it toward zero to disable blur rejection. `dedup_mae=0.8` is mean absolute grayscale difference on the same thumbnails; zero disables duplicate rejection. A small score can describe either duplicate frames or a slowly moving, low-texture scene, so inspect retained counts.

Periodic holdout uses every eighth registered view by default. Adjacent video frames can be nearly identical, so this can overestimate novel-view performance. For rigorous evaluation, design a separate flight/trajectory holdout and adapt the backend split rather than reporting periodic-frame metrics as evidence of unseen-area generalization.
