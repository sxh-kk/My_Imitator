"""Environment smoke checks; synthetic inputs do not measure benchmark performance."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORKSPACE = Path(__file__).resolve().parents[1]
REPO = WORKSPACE / "The-Imitator-Game"
ARTIFACTS = WORKSPACE / "environment" / "checks"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
# ACT uses the PyTorch implementations of Transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def core():
    import torch
    import mani_skill
    import gymnasium as gym

    assert Path(mani_skill.__file__).resolve().is_relative_to(REPO)
    assert torch.cuda.is_available()
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = (x @ x.T).float().square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for module in (
        "examples.baselines.act.train_act_imitator",
        "examples.baselines.act.eval_act_imitator",
        "examples.baselines.lerobot_dataset.lerobot_paired_dataset",
    ):
        importlib.import_module(module)
    packages = ["torch", "torchvision", "torchcodec", "numpy", "sapien",
                "gymnasium", "lerobot", "transformers", "diffusers", "huggingface-hub"]
    return {
        "python": sys.version,
        "prefix": sys.prefix,
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "mani_skill_path": mani_skill.__file__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "benchmark_envs": sorted(k for k in gym.registry if any(
            name in k for name in ("PlaceMugRack", "PickRemoteControl", "PourKettle"))),
    }


def video():
    from torchcodec.decoders import VideoDecoder
    from examples.baselines.lerobot_dataset.video_utils import decode_video_frames
    target = ARTIFACTS / "synthetic-video.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=64x64:rate=10:duration=1", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(target),
    ], check=True)
    decoder = VideoDecoder(str(target), device="cpu")
    frames = decoder.get_frames_at(indices=[0, 3, 7]).data
    assert list(frames.shape) == [3, 3, 64, 64]
    result = decode_video_frames(str(target), [0.0, 0.3, 0.7], 0.05, "torchcodec")
    assert list(result.shape) == [3, 3, 64, 64]
    return {"frames": len(decoder), "decoded_shape": list(result.shape),
            "backend": "torchcodec (CPU decoding)", "video": str(target)}


def render(shader):
    import numpy as np
    import torch
    import gymnasium as gym
    import mani_skill
    from PIL import Image

    env = gym.make(
        "Empty-v1", robot_uids="panda", num_envs=1,
        obs_mode="rgb", render_mode="rgb_array", reward_mode="none",
        sim_backend="physx_cpu", render_backend="gpu",
        sensor_configs={"shader_pack": shader, "width": 128, "height": 128},
    )
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(3):
            obs, _, _, _, _ = env.step(torch.zeros(env.action_space.shape))
        rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        assert rgb.shape == (128, 128, 3)
        assert np.isfinite(rgb).all() and rgb.std() > 1
        target = ARTIFACTS / f"panda-{shader}.png"
        Image.fromarray(rgb.astype(np.uint8)).save(target)
        return {"env": "Empty-v1", "robot": "panda", "shader": shader,
                "physics": "physx_cpu", "renderer": "gpu", "steps": 3,
                "rgb_shape": list(rgb.shape), "rgb_std": float(rgb.std()),
                "image": str(target)}
    finally:
        env.close()


def act():
    import torch
    from examples.baselines.act.train_act_imitator import TrainingArgs
    from examples.baselines.act.act.detr_video.backbone import build_backbone
    from examples.baselines.act.act.detr_video.transformer import build_transformer
    from examples.baselines.act.act.detr_video.detr_vae import DETRVAE, build_encoder
    from examples.baselines.encoders.task_encoder.video_backbone import FrozenVideoBackbone

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    args = TrainingArgs()
    encoder = FrozenVideoBackbone(
        backbone_type="dinov2_vitl14", num_sampled_frames=2, latent_dim=256,
    ).cuda()
    # First call downloads official DINOv2 weights if absent (about 1.2 GB).
    human_video = torch.randint(0, 256, (1, 2, 224, 224, 3), dtype=torch.uint8)
    z = encoder.encode(human_video=human_video)["z"]
    assert list(z.shape) == [1, 256]
    model = DETRVAE(
        [build_backbone(args)], build_transformer(args), build_encoder(args),
        args.state_dim, args.action_dim, args.pred_horizon, args.task_latent_dim,
    ).cuda()
    model.train()
    obs = {"state": torch.randn(1, 18, device="cuda"),
           "rgb": torch.rand(1, 1, 3, 224, 224, device="cuda")}
    target = torch.randn(1, 24, 16, device="cuda")
    optimizer = torch.optim.AdamW(list(model.parameters()) + encoder.trainable_params(), lr=1e-4)
    before = model.action_head.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction, (mu, logvar) = model(obs, actions=target, video_feature=z)
        loss = (prediction.float() - target).abs().mean()
        loss = loss + 0.01 * (-0.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
    assert list(prediction.shape) == [1, 24, 16] and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in list(model.parameters()) + encoder.trainable_params() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert encoder.cls_adapter[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in encoder._backbone.parameters())
    optimizer.step()
    assert not torch.equal(before, model.action_head.weight)
    model.eval()
    with torch.no_grad():
        prediction, _ = model(obs, video_feature=z.detach())
    assert torch.isfinite(prediction).all()
    return {"inputs": "synthetic; no benchmark data", "video_frames": 2,
            "encoder": "pretrained facebook/dinov2-large, frozen",
            "encoder_revision": getattr(encoder._backbone.config, "_commit_hash", None),
            "backbone": "pretrained ResNet18", "action_shape": list(prediction.shape),
            "loss": float(loss.detach()), "optimizer_steps": 1,
            "adapter_gradient": True, "backbone_frozen": True,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["core", "video", "render", "render-rt", "act"])
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {"stage": args.stage}
    try:
        fn = {"core": core, "video": video, "render": lambda: render("default"),
              "render-rt": lambda: render("rt-fast"), "act": act}[args.stage]
        report.update(status="passed", result=fn())
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    report["seconds"] = round(time.time() - started, 3)
    output = ARTIFACTS / f"{args.stage}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
