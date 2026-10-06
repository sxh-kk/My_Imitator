"""Read-only ACT shape trace with a synthetic sample, not an IG-10K rollout.

Uses real HumanSimPairedDataset.__getitem__, collate, ACTAgent and pretrained
DINOv2/ResNet18. The underlying decoded sim/human samples and normalizer
statistics are fixtures. No optimizer step, source patch, or simulator step.
Run in the configured imitator environment. Model files must already be cached.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent / "The-Imitator-Game"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from torch.utils.data import DataLoader
from examples.baselines.act.train_act_imitator import ACTAgent, TrainingArgs
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset, InputMode, PairedDatasetConfig, TaskMapper, get_collate_fn,
)
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer


def describe(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {k: describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    return value


def main():
    torch.manual_seed(0)
    config = PairedDatasetConfig(include_depth=False, num_frames=10, horizon=24)
    sim_id, human_id = "L0_TwoRobotPlaceMugRack-v1", "human_H57"
    sim_sample = {
        "states": torch.zeros(1, 18), "actions": torch.zeros(24, 16),
        "view_1": torch.rand(1, 3, 224, 224), "dataset_idx": torch.tensor(0),
        "repo_id": sim_id,
    }
    human_video = torch.rand(10, 224, 224, 3)
    # Bypass disk-backed __init__; the paired __getitem__ and collate stay intact.
    paired = HumanSimPairedDataset.__new__(HumanSimPairedDataset)
    paired.config, paired.input_mode = config, InputMode.VIDEO_ONLY
    paired.skip_human_video = False
    paired.valid_indices = [0]
    paired.sim_dataset = [sim_sample]
    paired.task_mapper = TaskMapper(config.task_mapping_file)
    paired.task_to_human_indices = {human_id: {"repos": [{"repo_id": human_id}]}}
    paired.human_dataset = SimpleNamespace(_get_target_item=lambda _: {"video": human_video})
    sample = paired[0]
    batch = next(iter(DataLoader(paired, batch_size=1, collate_fn=get_collate_fn("video_only"))))
    report = {
        "scope": "synthetic sample at decoded-dataset boundary; no real IG-10K files or robot rollout",
        "configuration": {"batch": 1, "human_frames": 10, "encoder_frames": 10,
                          "obs_horizon": 1, "pred_horizon": 24, "hidden_dim": 256,
                          "decoder_layers": 4, "cache": False},
        "sim_sample": describe(sim_sample), "paired_sample": describe(sample),
        "collated_batch": describe(batch), "training": {}, "inference": {},
    }
    args = TrainingArgs(frozen_backbone_num_frames=10)
    agent = ACTAgent(args, torch.device("cuda"), dataloader=None)
    agent.task_encoder._load_backbone()
    report["dino_revision"] = agent.task_encoder._backbone.config._commit_hash
    phase = "training"
    handles, latest = [], {}

    def hook(name, output_converter=lambda v: v):
        def record(module, inputs, output):
            report[phase][name] = {"input": describe(inputs), "output": describe(output_converter(output))}
            if name == "act_transformer":
                latest["decoder_stack"] = output.detach()
            if name == "action_head":
                report[phase]["action_head_uses_first_decoder_layer"] = bool(
                    torch.equal(inputs[0], latest["decoder_stack"][0]))
                report[phase]["action_head_uses_last_decoder_layer"] = bool(
                    torch.equal(inputs[0], latest["decoder_stack"][-1]))
        return record

    modules = {
        "cls_adapter": agent.task_encoder.cls_adapter,
        "seq_adapter": agent.task_encoder.seq_adapter,
        "task_layernorm": agent.ftask_norm,
        "cvae_encoder": agent.model.encoder,
        "cvae_latent_proj": agent.model.latent_proj,
        "resnet_joiner": agent.model.backbones[0],
        "image_projection": agent.model.input_proj,
        "act_encoder": agent.model.transformer.encoder,
        "act_transformer": agent.model.transformer,
        "action_head": agent.model.action_head,
    }
    for name, module in modules.items():
        handles.append(module.register_forward_hook(hook(name)))
    handles.append(agent.task_encoder._backbone.register_forward_hook(
        hook("dino", lambda output: output.last_hidden_state)))

    # DINO receives a keyword argument, so record it separately.
    def dino_input(module, inputs, kwargs):
        report[phase]["dino_pixel_values"] = describe(kwargs["pixel_values"])
    handles.append(agent.task_encoder._backbone.register_forward_pre_hook(dino_input, with_kwargs=True))

    try:
        with torch.inference_mode():
            agent.train()
            report[phase]["model_observation"] = describe(agent._preprocess_obs_for_model(batch["robot_obs"]))
            losses = agent.compute_loss(batch)
            report[phase]["loss_shapes"] = describe(losses)
            phase = "inference"
            agent.prepare_for_eval(batch["human_video"], batch["robot_first_frame_obs"])
            report[phase]["cached_task_z"] = describe(agent._cached_task_z)
            obs = {k: v.cuda() for k, v in batch["robot_obs"].items()}
            actions = agent.get_action(obs)
            report[phase]["predicted_chunk"] = describe(actions)
            report[phase]["cvae_encoder_called"] = "cvae_encoder" in report[phase]
            # t=0: only one chunk exists, so default temporal aggregation selects its first action.
            raw_action = actions[:, 0]
            normalizer = ActionNormalizer()
            normalizer.add_dataset_stats(0, sim_id, {
                "action.qpos_gripper_actions": {"q01": [-1.] * 16, "q99": [1.] * 16},
            }, "observation.qpos_gripper_states", "action.qpos_gripper_actions")
            denorm = normalizer.denormalize_action(raw_action, 0).cpu().numpy()
            report[phase]["robot_action_dict_shapes"] = {
                "panda_wristcam-0": list(denorm[:, :8].shape),
                "panda_wristcam-1": list(denorm[:, 8:16].shape),
            }
            report[phase]["action_normalization_note"] = "synthetic q01/q99; shapes only, no real control values"
    finally:
        for handle in handles:
            handle.remove()
    target = DOCS / "shape_trace.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {target}")
    print("Decoder output:", report["training"]["act_transformer"]["output"]["shape"])
    print("First decoder layer selected:", report["training"]["action_head_uses_first_decoder_layer"])
    print("Actions:", report["inference"]["predicted_chunk"]["shape"])


if __name__ == "__main__":
    main()
