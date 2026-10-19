from __future__ import annotations

import math

from .common import PipelineError, read_json


def load_config(path):
    cfg = read_json(path)
    required = {"backend", "extraction", "sfm", "training", "render"}
    if set(cfg) != required:
        raise PipelineError(f"Config requires exactly {sorted(required)}")
    if cfg["backend"] not in {"gsplat", "mlp"}:
        raise PipelineError("backend must be gsplat or mlp")
    e, s, t, r = (cfg[k] for k in ("extraction", "sfm", "training", "render"))
    window = e.get("selection_window_seconds", 0)
    if (isinstance(window, bool) or not isinstance(window, (int, float))
            or not math.isfinite(window) or window < 0):
        raise PipelineError("selection_window_seconds must be finite and nonnegative")
    if window and window * e["fps"] < 1:
        raise PipelineError("selection windows must contain at least one candidate interval")
    candidates = e.get("max_candidates", e["max_frames"])
    if type(candidates) is not int or candidates < 1:
        raise PipelineError("max_candidates must be a positive integer")
    policy = s.get("quality_gate", {})
    allowed = {"max_point_error_p95", "min_median_track_length", "max_registered_gap_seconds"}
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise PipelineError("Unknown reconstruction quality_gate settings")
    thresholds = list(policy.items())
    if "max_gap_seconds" in e:
        thresholds.append(("max_gap_seconds", e["max_gap_seconds"]))
    for key, value in thresholds:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise PipelineError(f"{key} must be finite and positive")
    for name, value in [("fps", e["fps"]), ("max_size", e["max_size"]),
                        ("max_frames", e["max_frames"]), ("temporal_window", s["temporal_window"]),
                        ("global_stride", s["global_stride"]), ("iterations", t["iterations"]),
                        ("fps", r["fps"])]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise PipelineError(f"{name} must be positive")
    integer_fields = [("max_size", e["max_size"]), ("max_frames", e["max_frames"]),
                      ("temporal_window", s["temporal_window"]), ("global_stride", s["global_stride"]),
                      ("max_pairs", s["max_pairs"]), ("iterations", t["iterations"]),
                      ("test_every", t["test_every"]), ("cap_max", t["gsplat"]["cap_max"]),
                      ("sh_degree", t["gsplat"]["sh_degree"]),
                      ("points_per_leaf", t["mlp"]["points_per_leaf"]),
                      ("interpolation_steps", r["interpolation_steps"]),
                      ("render.max_frames", r["max_frames"]), ("port", r["port"])]
    for name, value in integer_fields:
        if type(value) is not int or value < (0 if name == "sh_degree" else 1):
            raise PipelineError(f"{name} must be a positive integer (SH degree may be zero)")
    if type(s["use_gpu"]) is not bool or type(r["save_frames"]) is not bool:
        raise PipelineError("use_gpu and save_frames must be JSON booleans")
    if r["max_frames"] < 2 or r["port"] > 65535 or e["max_size"] < 32:
        raise PipelineError("render.max_frames>=2, port<=65535, and max_size>=32 are required")
    if s["camera_mode"] not in {"shared", "per-video"}:
        raise PipelineError("camera_mode must be shared or per-video")
    if s["matcher"] not in {"hybrid", "exhaustive", "vocab-tree"}:
        raise PipelineError("matcher must be hybrid, exhaustive, or vocab-tree")
    if not 0 < s["min_registered_fraction"] <= 1:
        raise PipelineError("min_registered_fraction must be in (0, 1]")
    if not 0 < s["min_per_video_fraction"] <= 1:
        raise PipelineError("min_per_video_fraction must be in (0, 1]")
    if e["min_sharpness"] < 0 or e["dedup_mae"] < 0:
        raise PipelineError("Quality thresholds cannot be negative")
    if t["test_every"] < 2 or t["gsplat"]["cap_max"] < 100:
        raise PipelineError("test_every must be >=2 and cap_max must be >=100")
    if t["gsplat"]["sh_degree"] not in range(4):
        raise PipelineError("sh_degree must be 0..3")
    if t["mlp"].get("kernel_backend", "auto") not in {"auto", "native", "turing"}:
        raise PipelineError("training.mlp.kernel_backend must be auto, native, or turing")
    if r["trajectory"] not in {"interp", "ellipse", "spiral"}:
        raise PipelineError("trajectory must be interp, ellipse, or spiral")
    return cfg
