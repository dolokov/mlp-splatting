"""On-device MLP compile/forward/backward/update/checkpoint smoke test; no footage needed."""
from __future__ import annotations

import os
import sys

from .backends import gpu_check, mlp_requested, upstream
from .common import PipelineError, write_json


def run(cfg, work):
    if cfg["backend"] != "mlp":
        raise PipelineError("smoke requires an MLP config, e.g. configs/mlp-pilot.json")
    import torch
    import torch.nn.functional as functional

    env = gpu_check("mlp", mlp_kernel_backend=mlp_requested(cfg))
    env.update({"MAX_JOBS": "2", "MLPSP_TILE_ORDER": "tilecenter",
                "MLPSP_TILE_CANDIDATE_CAPACITY": "1536",
                "TORCH_EXTENSIONS_DIR": str(work / "torch-extensions")})
    os.environ.update(env)
    repo = upstream("mlp")
    sys.path.insert(0, str(repo))
    from scannet.checkpoints import make_inference_checkpoint
    from scannet.cumodel import NeuralPrimitiveSceneTileCUDA, _tile_ext_build_config
    from scannet.eval_rgb_psnr import _load_scene_tile_cuda

    torch.manual_seed(17)
    scene = NeuralPrimitiveSceneTileCUDA(num_primitives=2).cuda()
    scene.set_image_size(8, 8)
    with torch.no_grad():
        scene.positions.copy_(torch.tensor([[-.15, 0., 2.], [.2, .1, 2.5]], device="cuda"))
        scene.log_scales.fill_(-.4)
        scene.b1.fill_(.02)
        scene.b2.fill_(.02)
        scene.b3[:, 3].fill_(.5)
    y, x = torch.meshgrid(torch.arange(8, device="cuda"), torch.arange(8, device="cuda"), indexing="ij")
    rays_d = functional.normalize(torch.stack(((x - 3.5) / 20, (y - 3.5) / 20,
                                               torch.ones_like(x)), -1).float().reshape(-1, 3), dim=-1)
    rays_o = torch.zeros_like(rays_d)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    def image(model):
        return model(rays_o, rays_d, score_threshold=.003,
                     culled_prim_indices=indices, tile_offsets=offsets)

    # Exercise the real custom backward for geometry and every MLP layer, then
    # optimize appearance in a controlled toy problem (not reconstruction quality).
    target = torch.tensor([.15, .7, .25], device="cuda").expand(64, 3)
    optimizer = torch.optim.Adam([scene.W3, scene.b3], lr=.05)
    losses = []
    for _ in range(8):
        scene.zero_grad(set_to_none=True)
        rgb = image(scene)
        if not bool(torch.isfinite(rgb).all()) or not bool((rgb.abs().sum() > 0)):
            raise PipelineError("MLP smoke: forward output is nonfinite or empty")
        loss = (rgb - target).square().mean()
        loss.backward()
        for name in ("positions", "log_scales", "quats", "W1", "b1", "W2", "b2", "W3", "b3"):
            grad = getattr(scene, name).grad
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise PipelineError(f"MLP smoke: invalid {name} gradient")
        if not bool(scene.b3.grad.abs().sum() > 0):
            raise PipelineError("MLP smoke: zero appearance gradient")
        losses.append(float(loss.detach()))
        optimizer.step()
    with torch.inference_mode():
        reference = image(scene)
        final_loss = float((reference - target).square().mean())
    if final_loss >= losses[0]:
        raise PipelineError("MLP smoke: optimizer did not improve the toy appearance loss")

    # Use the actual upstream inference export and loader, including FP16-on-disk
    # weights restored to FP32. No representation conversion is involved.
    work.mkdir(parents=True, exist_ok=True)
    path = work / "smoke-scene.pt"
    exported = make_inference_checkpoint({"scene_state_dict": scene.state_dict()})
    torch.save(exported, path)
    loaded = _load_scene_tile_cuda(str(path), torch.device("cuda:0"), min_points=64, tile_size=8)
    loaded.set_image_size(8, 8)
    with torch.inference_mode():
        restored = image(loaded)
    if loaded.W1.dtype != torch.float32 or not torch.allclose(restored, reference, atol=.003, rtol=.003):
        raise PipelineError("MLP smoke: native checkpoint round-trip changed the RGB output excessively")
    extension = scene._ensure_tile_ext()
    has_wmma = hasattr(extension, "render_tiles_rgb_wmma")
    if has_wmma != (env["MLPSP_CUDA_BACKEND"] == "native"):
        raise PipelineError("MLP smoke: compiled extension does not match selected GPU backend")
    torch.cuda.synchronize()
    report = {"passed": True, "kernel_backend": env["MLPSP_CUDA_BACKEND"],
              "cuda_arch": env["TORCH_CUDA_ARCH_LIST"], "extension": _tile_ext_build_config()[1],
              "initial_loss": losses[0], "final_loss": final_loss,
              "checkpoint_max_rgb_error": float((restored - reference).abs().max()),
              "wmma_entrypoints": has_wmma, "checkpoint": str(path)}
    write_json(work / "smoke.json", report)
    print(f"MLP GPU smoke passed ({report['kernel_backend']}): {work / 'smoke.json'}")
    return report
