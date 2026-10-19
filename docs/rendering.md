# Rendering choices

Your trained representation determines which renderer can consume it. MLP-Splatting and ordinary 3D Gaussians are not interchangeable file formats.

| Choice | Input | Execution | Status in this repo |
| --- | --- | --- | --- |
| RGB flythrough MP4; optional PNG frames | Native MLP checkpoint or gsplat checkpoint + dataset | NVIDIA CUDA server/workstation; FFmpeg encodes | Implemented by `drone-scene render` |
| Interactive free-fly browser UI backed by workstation | gsplat checkpoint + dataset | gsplat/Viser runs CUDA on your workstation; browser receives frames | Implemented by `drone-scene view` |
| Browser-local editing and sharing | Standard Gaussian PLY exported by gsplat | SuperSplat / PlayCanvas GPU renderer; deployment is separate | PLY export implemented; editor/hosting external |
| Other browser Gaussian viewers | Compatible standard Gaussian PLY or converted splat format | WebGL/WebGPU on the viewer's device | External; check SH/axis/scale conventions and scene size |
| Unity / Unreal / Blender | Gaussian PLY plus a compatible Gaussian rendering plugin | Application-specific renderer/shader | External; a PLY import alone does not create Gaussian rendering |
| Custom native MLP interactive viewer | Native MLP checkpoint and its calibrated ray generator | Python CUDA renderer, e.g. streamed through Viser | Additional engineering; not packaged here |
| Browser-native MLP rendering | Exported native geometry + MLP weights | Local WebGPU reference renderer | Experimental prototype; see limitations below |
| Metric textured mesh / orthophoto | Photogrammetry output, e.g. ODM | Blender, engine, GIS, standard mesh viewer | Separate reconstruction route; see `large-scenes.md` |

## Included offline rendering

```bash
drone-scene render --workspace work/site --config configs/titan.json
```

`latest-render.json` gives the exact MP4 location. Each distinct `render` configuration gets a separate directory. Set `render.save_frames` to `true` to also save exact RGB PNG frames. MP4 uses H.264, CRF 18, configured FPS, and pads odd dimensions by at most one pixel instead of changing projection calibration. PNGs retain original dimensions.

`interp` interpolates the capture path with linear camera centers and quaternion Slerp. It uses the most-registered single clip unless `render.clip` names an ID such as `v001_...` from `reports/extraction.json`. Camera paths outside observed coverage can expose holes, sky floaters, or artifacts: synthesized geometry is not guaranteed behind objects or under roofs.

The Gaussian renderer also accepts `ellipse` and `spiral` paths from the pinned gsplat examples. These are experimental paths fitted in the normalized training frame. Near-planar overhead flight paths can make ellipse/spiral estimates unstable or place cameras outside the trained volume; start with `interp`. Native MLP supports `interp` here.

The offline Gaussian renderer reconstructs the same `Parser(normalize=True)` frame used during training and uses the same intrinsics. Do not edit the scene's calibration after training. The native MLP renderer uses its training loader's normalization/ray generator and validates checkpoint calibration.

## Gaussian PLY

Open `model/gsplat/ply/point_cloud_<last-zero-based-step>.ply` in SuperSplat. The exported coordinates are the **normalized gsplat training frame**, not GPS or meter coordinates. The point cloud is an appearance representation, not a mesh, collision shape, survey surface, or relightable material model. SH degree 2 is chosen to reduce storage and compute; raise to 3 for an experiment if memory permits.

For very large scenes, a browser/mobile viewer usually needs pruning, compression, chunks, and LOD. A single enormous PLY is not automatically streamable. Exporting/chunking a large scene is separate from creating a connected reconstruction.

## Native MLP checkpoints

### Experimental browser-GPU reference viewer

An initial WebGPU prototype is included. It downloads exported geometry and neural
weights once, then renders locally on the browser GPU. The HTTP server only serves
static files; it does not render or stream frames. Export with your completed native
inference checkpoint, original scene dataset, and training config, from the MLP environment:

```bash
python -m drone_scene.web_viewer \
  --config configs/mlp-pilot.json \
  --workspace /absolute/path/to/your/completed-pilot \
  --output /absolute/path/to/a/new-webgpu-export \
  --serve --port 8088
```

Open `http://localhost:8088/` in a WebGPU-capable browser. The page reports missing
WebGPU/adapter support, shader compilation errors, and device loss. Do not assume
your installed Firefox/Linux configuration exposes WebGPU. No unsafe browser flags
or rendering fallback are enabled automatically. Stop the server with Ctrl+C.
For an existing export, serve its directory on localhost with a static HTTP server;
the exporter intentionally refuses to overwrite a nonempty output directory.

Select a captured camera, drag to look, and use W/S, A/D, Q/E to translate. Start
at 48–96 pixels wide and increase only after measuring render time. Save PNG captures
the current browser image. This is a brute-force correctness-oriented implementation,
**not a real-time performance claim**: every ray tests every primitive. It uses the
native closest-point features, positional encoding, MLP and depth-sorted opacity
compositing, with a black background and a decay threshold of 0.003. More than 128
hits on a ray produce magenta instead of silently truncating. Hidden widths above
64, more than 10 encoding frequencies, nonfinite weights, invalid quaternions, and
nonzero camera distortion are rejected. GPU buffer limits are checked in the browser.

**Validation limitation:** export packing and browser source syntax have automated
tests. Browser shader execution, CUDA pixel parity, browser compatibility and FPS
on your TITAN are not certified. Validate a captured view against CUDA before using
the viewer to judge reconstruction quality. Tile culling/batched neural evaluation
remain future optimization work. Exports contain the learned scene weights; treat
them as scene data when sharing or hosting.

The native primitive predicts view-dependent RGB and opacity using learned neural weights. Renaming the checkpoint or exporting primitive centers to `.ply` discards that learned function. No lossless standard Gaussian export is provided by this pipeline. A Gaussian approximation would require training/distillation against rendered MLP views and evaluating the resulting quality loss.

The authors' interactive project demo uses a separate runtime; the experimental viewer above is this repository's own implementation, not their web runtime. Ordinary splat viewers cannot open native checkpoints. Native CUDA RGB rendering is also included here. Semantic feature selection/editing would additionally require feature supervision and UI work; RGB-only training does not produce those features.

For a native interactive viewer, reuse `mlp_frames`' checkpoint and camera setup, replace its fixed path with a Viser camera callback, and render with `scene(...)`. Respect the native camera modes and tile-capacity checks. Keep view sizes small initially. This is a development direction, not a currently implemented command.
