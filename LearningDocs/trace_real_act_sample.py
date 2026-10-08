"""用中文说明，跟踪一条真实 PlaceMugRack 样本如何进入 ACT。

读取已完成实验的 A50 数据、人类 episode 0 特征缓存和所选 checkpoint。
执行一次带真实动作标签的训练前向，再执行一次不带标签的推理前向。
不执行反向传播、参数更新、checkpoint 保存或仿真步进。
添加 --interactive 后，会在七个位置暂停，可以在 main() 中检查变量。
"""
import argparse
import json
import os
from pathlib import Path
import sys

DOCS = Path(__file__).resolve().parent
STUDY = DOCS.parent / "experiments/act_placemugrack_study"
sys.path.insert(0, str(STUDY))
from common import ROOT, TASK, build_args, build_dataset, seed_all, sha256

# 这条学习路径只使用 PyTorch；避免额外初始化 TensorFlow/Flax。
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from torch.utils.data import DataLoader
from examples.baselines.act.train_act_imitator import ACTAgent, get_collate_fn


# 保留源码中的英文变量名，同时在终端解释其含义，便于对照原仓库。
LABELS = {
    "sample": "配对后的单条样本：机器人观测、动作标签和同任务人类视频",
    "raw_state": "未归一化的当前状态：双臂各 7 个臂关节和 2 个手指关节",
    "raw_action": "未归一化的当前动作标签：双臂各 7 个臂命令和 1 个夹爪命令",
    "batch_before_cache": "启用特征缓存前的 batch：包含实际读取的人类视频",
    "batch": "缓存启用后的训练 batch：通过 human_repo_id 查询视频特征",
    "cached_raw": "冻结 DINO 的缓存特征：cls 是视频全局特征，seq 是选取的 patch 特征",
    "obs_for_model": "送入 ACT 的机器人观测：state 为关节状态，rgb 为机器人相机图像",
    "fixed_video": "本轮训练实际固定的人类视频帧：episode 0 的 10 帧",
    "raw_cls": "DINO 每帧 CLS token 沿时间平均后的 1024 维特征",
    "raw_seq": "DINO 全部帧的 patch tokens 中选出的最多 32 个特征",
    "task_z": "人类视频经过 Adapter 和 LayerNorm 后的 256 维任务条件",
    "pred_train": "训练分支预测的 24 步动作块，用于与 robot_actions 计算 L1",
    "mu": "CVAE 潜变量分布的均值，每条样本 32 维",
    "logvar": "CVAE 潜变量分布的对数方差，每条样本 32 维",
    "eval_task_z": "推理准备阶段缓存的任务条件，随后供 get_action 使用",
    "prediction": "不输入真实动作标签时，策略预测的 24 步动作块",
    "latent_inference": "推理时使用的 32 维零潜变量，尚未经过 latent_out_proj",
}


def describe(value):
    """只记录 shape、dtype、device，避免打印整幅图像或大量特征值。"""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    return value


def show(name, value):
    print(f"\n{name}（{LABELS.get(name, name)}）:\n"
          f"  {json.dumps(describe(value), ensure_ascii=False)}", flush=True)


def explain(message, commands, interactive):
    """在暂停前给出中文解释；交互模式额外展示可直接输入的检查命令。"""
    print(f"\n中文解释：{message}", flush=True)
    if interactive:
        print("接下来在 (Pdb) 中逐行检查：", flush=True)
        for command, meaning in commands:
            print(f"  {command}\n    → {meaning}", flush=True)
        print("检查完输入 c，继续到下一个暂停点；输入 q 可退出。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0, help="要跟踪的机器人样本索引，默认从第 0 帧开始")
    parser.add_argument("--batch-size", type=int, default=1, help="学习用 batch 大小，允许 1–8，默认 1")
    parser.add_argument("--interactive", action="store_true", help="在七个步骤进入 pdb，交互查看变量")
    parser.add_argument("--out", type=Path, default=DOCS / "real_sample_trace.json", help="跟踪结果 JSON 的保存路径")
    cli = parser.parse_args()
    cli.out = cli.out.resolve()
    seed_all(1)
    # 与本轮独立评估一致，关闭 TF32；训练前向仍使用 BF16。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache_dir = ROOT / "prepared/human_cache_ep0"
    cache_file = cache_dir / "backbone_dinov2_vitl14/raw_features.pt"
    checkpoint = ROOT / "runs/A50_s1/checkpoints/best_model.pt"
    cache_hash, checkpoint_hash = sha256(cache_file), sha256(checkpoint)
    # 复用实际实验的数据路径和模型配置；workers=0 使读取发生在当前进程。
    args = build_args("A50", 1, cli.batch_size, 0, cache_dir)
    print("\n维度说明：shape=张量形状，dtype=数值类型，device=所在设备；B=batch 大小。", flush=True)

    # 第 1 步：构建真正从磁盘读数据的配对 dataset，此时尚未跳过视频读取。
    paired = build_dataset(args)
    assert len(paired.sim_dataset.lerobot_dataset.datasets) == 1
    assert 0 <= cli.sample_index < len(paired)
    assert 1 <= cli.batch_size <= 8
    assert cli.sample_index + cli.batch_size <= len(paired)
    print(f"\nSTEP 1｜数据集与任务配对：{len(paired)} 个机器人时刻；{paired.paired_tasks}", flush=True)
    explain("这里按任务 ID 将机器人轨迹映射到人类示范，并不要求两段视频的帧号或 episode 号相同。",
            [("p paired.paired_tasks", "查看配对的人类任务和机器人任务"),
             ("p paired.task_mapper.get_human_task_from_sim(TASK)", "查看 PlaceMugRack 对应的人类任务")], cli.interactive)
    if cli.interactive:
        breakpoint()

    # 第 2 步：同时查看原始数值、未来动作索引和归一化后的配对样本。
    # hf_dataset 读取原始 parquet，尚未经过 sim 包装层的数值归一化。
    actual_sim_idx = paired.valid_indices[cli.sample_index]
    sim_ds = paired.sim_dataset.main_dataset
    row = sim_ds.hf_dataset[actual_sim_idx]
    raw_state = row[paired.sim_dataset.state_key]    # (18,)：当前关节状态。
    raw_action = row[paired.sim_dataset.action_key]  # (16,)：当前时刻的监督动作。
    query_indices, padding = sim_ds._get_query_indices(int(row["index"]), int(row["episode_index"]))
    # 从当前时刻开始读 24 步；超出本 episode 的索引截到最后一帧。
    future_indices = query_indices[paired.sim_dataset.action_key]
    mapping = sim_ds._absolute_to_relative_idx
    raw_actions = torch.stack([
        sim_ds.hf_dataset[j if mapping is None else mapping[j]][paired.sim_dataset.action_key]
        for j in future_indices
    ])
    sample = paired[cli.sample_index]  # 调用原版 HumanSimPairedDataset.__getitem__。
    normalizer = paired.sim_dataset.normalizer
    assert torch.equal(sample["robot_actions"], normalizer.normalize_action(raw_actions, 0))
    assert torch.equal(sample["robot_obs"]["states"], normalizer.normalize_state(raw_state.unsqueeze(0), 0))
    print(f"\nSTEP 2｜单条真实样本：{sample['sample_id']}；机器人 episode={int(row['episode_index'])}，frame={int(row['frame_index'])}", flush=True)
    show("sample", sample)
    show("raw_state", raw_state)
    show("raw_action", raw_action)
    print("24 步动作对应的原始行索引：", future_indices, flush=True)
    explain("state 每臂记录 7 个臂关节和 2 个手指，共 18 维；action 每臂只有 1 个夹爪命令，共 16 维。sample 内的数值已经归一化。",
            [("p raw_state[:7], raw_state[7:9]", "查看左臂关节和两个手指的原始状态"),
             ("p sample['robot_actions'].shape", "查看单条样本的 24×16 动作标签"),
             ("p padding['action.qpos_gripper_actions_is_pad'].tolist()", "查看哪些未来位置超出 episode 尾部")], cli.interactive)
    if cli.interactive:
        breakpoint()

    # 第 3 步：真正的 DataLoader 将若干配对样本堆成 batch。
    # 重读图像可能产生不同的随机增强；同一索引的 state/action 标签不变。
    loader = DataLoader(paired, batch_size=cli.batch_size,
                        sampler=list(range(cli.sample_index, cli.sample_index + cli.batch_size)),
                        num_workers=0, collate_fn=get_collate_fn(args.input_mode))
    batch_before_cache = next(iter(loader))
    print("\nSTEP 3｜collate 后的真实 batch：此时仍包含人类视频", flush=True)
    show("batch_before_cache", batch_before_cache)
    explain("collate 用 torch.stack 在最前面添加 B 维。例如 (24,16) 动作标签变为 (B,24,16)。机器人 view_1 的第二维是观测时间窗。",
            [("p batch_before_cache['robot_actions'].shape", "查看堆叠后的动作标签"),
             ("p batch_before_cache['human_video'].shape", "查看 B 条人类视频，每条 10 帧")], cli.interactive)
    if cli.interactive:
        breakpoint()

    # 第 4 步：原版 ACTAgent 构造函数启用冻结特征缓存和 skip_human_video。
    agent = ACTAgent(args, torch.device("cuda"), dataloader=loader)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["experiment"]["stats_sha256"] == sha256(Path(args.sim_root) / TASK / "meta/stats.json")
    agent.load_state_dict(payload["agent_state_dict"], strict=True)  # 使用已经训练的模型参数。
    checkpoint_step = int(payload["iteration"])
    del payload
    batch = next(iter(loader))
    assert paired.skip_human_video and "human_video" not in batch
    assert torch.equal(batch["robot_actions"], batch_before_cache["robot_actions"])
    # 用 human_repo_id 查询已有 raw features，训练 step 无需重新运行 DINO。
    cached_raw = agent.task_encoder._raw_cache.get_batch(batch["human_repo_id"], torch.device("cuda"))
    backbone = agent.task_encoder.backbone
    obs_for_model = agent._preprocess_obs_for_model({k: v.cuda() for k, v in batch["robot_obs"].items()})
    report = {
        "scope": "真实 A50 数据与所选 checkpoint；各执行一次训练和推理前向，不更新参数，不运行仿真",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step, "batch_size": cli.batch_size,
        "sample_index": cli.sample_index, "sample_id": sample["sample_id"],
        "robot_episode": int(row["episode_index"]), "robot_frame": int(row["frame_index"]),
        "future_indices": future_indices, "padding": describe(padding),
        "state_and_action_match_original_rows": True,
        "sample": describe(sample), "batch_before_cache": describe(batch_before_cache),
        "batch": describe(batch), "cached_raw": describe(cached_raw),
        "obs_for_model": describe(obs_for_model), "hooks": {},
    }
    print(f"\nSTEP 4｜正式训练的缓存路径：已加载 step {checkpoint_step} 的 checkpoint", flush=True)
    show("batch", batch)
    show("cached_raw", cached_raw)
    show("obs_for_model", obs_for_model)
    explain("batch 没有 human_video 是因为视频特征已经缓存。raw_cls 经过可训练 Adapter 变为任务条件；机器人图像则转为 CHW，送给另一个 ResNet18 编码器。",
            [("p 'human_video' in batch", "应为 False：数据读取跳过了视频解码"),
             ("p cached_raw['cls'].shape", "DINO 缓存全局特征为 (B,1024)"),
             ("p backbone.cls_adapter[0].weight.requires_grad", "应为 True：Adapter 仍可训练")], cli.interactive)
    if cli.interactive:
        breakpoint()

    # 给现有模块注册 hook，只观察输入输出，不替换任何网络计算。
    # report 记录形状；live 保留部分实际张量，便于在 pdb 中进一步检查。
    phase, handles, live = "dino_precompute", [], {}

    def record(name, converter=lambda value: value):
        """生成记录模块输入输出的回调函数，phase 区分训练与推理。"""
        def hook(module, inputs, kwargs, output):
            output = converter(output)
            report["hooks"].setdefault(phase, {}).setdefault(name, []).append({
                "input": describe(inputs), "kwargs": describe(kwargs), "output": describe(output)})
            if name in {"model", "task_norm", "latent_out", "action_head", "transformer"}:
                live.setdefault(phase, {})[name] = {"input": inputs, "kwargs": kwargs, "output": output}
        return hook

    modules = {
        "cls_adapter": backbone.cls_adapter, "seq_adapter": backbone.seq_adapter,
        "task_norm": agent.ftask_norm, "cvae_encoder": agent.model.encoder,
        "latent_proj": agent.model.latent_proj, "latent_out": agent.model.latent_out_proj,
        "robot_backbone": agent.model.backbones[0], "image_proj": agent.model.input_proj,
        "video_feature_proj": agent.model.video_feature_proj,
        "act_encoder": agent.model.transformer.encoder, "transformer": agent.model.transformer,
        "action_head": agent.model.action_head, "model": agent.model,
    }
    for name, module in modules.items():
        handles.append(module.register_forward_hook(record(name), with_kwargs=True))
    try:
        # 第 5 步：重算本轮实际固定的人类训练视频，用于理解缓存的来源。
        # 它与 batch_before_cache 重新采样/增强的视频可能不同。
        fixed_video = torch.load(ROOT / "prepared/human_ep0.pt", weights_only=True).cuda()
        backbone._load_backbone()
        backbone.to("cuda").eval()
        handles.append(backbone._backbone.register_forward_hook(
            record("dino", lambda output: output.last_hidden_state), with_kwargs=True))
        with torch.no_grad():  # DINO 主干冻结，重算原始特征不需要梯度。
            raw_cls, raw_seq = backbone._encode_dino(fixed_video)
        cls_error = float((raw_cls - cached_raw["cls"][:1]).abs().max())
        seq_error = float((raw_seq - cached_raw["seq"][:1]).abs().max())
        assert torch.allclose(raw_cls, cached_raw["cls"][:1], atol=1e-5, rtol=1e-5), cls_error
        assert torch.allclose(raw_seq, cached_raw["seq"][:1], atol=1e-5, rtol=1e-5), seq_error
        report["dino_cache_max_abs_error"] = {"cls": cls_error, "seq": seq_error}
        print("\nSTEP 5｜缓存来源：固定人类视频帧 → DINO → 原始特征", flush=True)
        show("fixed_video", fixed_video)
        show("raw_cls", raw_cls)
        show("raw_seq", raw_seq)
        print("重算 DINO 与已有缓存的最大绝对误差（cls、seq）：", cls_error, seq_error, flush=True)
        explain("10 帧先合并为图像 batch。每帧有 1 个 CLS 和 256 个 patch token；CLS 沿时间平均得到 raw_cls。这次重算仅用于学习检查，正式训练直接读缓存。",
                [("p report['hooks']['dino_precompute']['dino'][0]['output']['shape']", "查看逐帧 DINO 输出：[10,257,1024]"),
                 ("p raw_cls.shape, raw_seq.shape", "查看时间汇总和 patch 选取后的特征")], cli.interactive)
        if cli.interactive:
            breakpoint()

        # 第 6 步：调用真正的 compute_loss，提供真实动作标签，保留梯度图。
        # 使用正式训练的 BF16 精度；此脚本不执行 backward 或 optimizer.step。
        phase = "training"
        agent.train()
        backbone._backbone.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = agent.compute_loss(batch)
        pred_train, (mu, logvar) = live[phase]["model"]["output"]  # 动作预测与 CVAE 分布参数。
        task_z = live[phase]["task_norm"]["output"]  # 人类视频得到的任务条件，区别于 CVAE latent。
        assert tuple(pred_train.shape) == (cli.batch_size, 24, 16)
        assert tuple(mu.shape) == tuple(logvar.shape) == (cli.batch_size, 32)
        assert "dino" not in report["hooks"][phase]
        report["losses"] = {k: float(v.detach()) for k, v in losses.items()}
        report["training_loss_requires_grad"] = losses["loss"].requires_grad
        report["training_dino_calls"] = 0
        report["first_decoder_layer_selected"] = torch.equal(
            live[phase]["action_head"]["input"][0], live[phase]["transformer"]["output"][0])
        print("\nSTEP 6｜训练前向：缓存特征 → task_z；当前观测与动作标签 → CVAE → 动作块", flush=True)
        show("task_z", task_z)
        show("pred_train", pred_train)
        show("mu", mu)
        show("logvar", logvar)
        print("训练损失（loss=总损失，l1=动作重建误差，kl=潜变量分布约束）：", report["losses"], flush=True)
        print("总损失是否保留梯度图：", losses["loss"].requires_grad, flush=True)
        explain("task_z 是 256 维视频条件；mu/logvar 描述另一种 32 维 CVAE 潜变量。24 个 learned queries 各预测一个 16 维动作，所以输出 (B,24,16)。当前损失为 L1+10×KL。",
                [("p task_z.shape, mu.shape, logvar.shape", "对比视频任务条件与 CVAE 分布参数的维度"),
                 ("p agent.model.query_embed.weight.shape", "查看 24 个可训练动作查询向量"),
                 ("p pred_train.shape", "查看动作块预测，与 robot_actions 形状一致")], cli.interactive)
        if cli.interactive:
            breakpoint()

        # 第 7 步：使用同一个离线机器人观测调用推理接口，不启动仿真。
        phase = "inference_prepare"
        agent.prepare_for_eval(fixed_video, batch["robot_first_frame_obs"])  # 编码一次视频，缓存任务条件。
        eval_task_z = agent._cached_task_z
        phase = "inference"
        prediction = agent.get_action({k: v.cuda() for k, v in batch["robot_obs"].items()})  # 内部传 actions=None。
        latent_inference = live[phase]["latent_out"]["input"][0]
        assert tuple(prediction.shape) == (cli.batch_size, 24, 16)
        assert torch.count_nonzero(latent_inference) == 0
        assert "cvae_encoder" not in report["hooks"][phase]
        assert live[phase]["model"]["kwargs"]["actions"] is None
        report["predicted_chunk"] = describe(prediction)
        report["inference_cvae_calls"] = 0
        report["inference_latent_is_zero"] = True
        print("\nSTEP 7｜推理前向：不输入真实动作标签，跳过 CVAE encoder", flush=True)
        show("eval_task_z", eval_task_z)
        show("prediction", prediction)
        show("latent_inference", latent_inference)
        explain("DETRVAE 根据 actions 是否为 None 选择分支。推理没有真实动作标签，直接使用 32 维零 latent。零 latent 经过带 bias 的投影后，256 维 token 不必为零。",
                [("p live['inference']['model']['kwargs']['actions']", "应为 None：未来真实动作没有进入推理"),
                 ("p 'cvae_encoder' in report['hooks']['inference']", "应为 False：推理没有调用 CVAE encoder"),
                 ("p torch.count_nonzero(latent_inference).item()", "应为 0：投影前的 latent 全为零")], cli.interactive)
        if cli.interactive:
            breakpoint()
    finally:
        for handle in handles:
            handle.remove()
    # 回读文件 hash，确认这次学习跟踪没有改动训练缓存或 checkpoint。
    assert sha256(cache_file) == cache_hash
    assert sha256(checkpoint) == checkpoint_hash
    report["checkpoint_and_feature_cache_unchanged"] = True
    report["status"] = "passed"
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nTRACE PASSED｜跟踪通过；结果已保存到 {cli.out}", flush=True)


if __name__ == "__main__":
    main()
