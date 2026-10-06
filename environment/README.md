# Imitator 本机环境

创建日期：2026-10-06。对应研究方案：[`../IMITATOR_RESEARCH_PLAN.md`](../IMITATOR_RESEARCH_PLAN.md)。

## 使用

```bash
source /home/zxc/Imitator/scripts/activate_imitator.sh
# 等价于 conda activate imitator，然后进入 The-Imitator-Game 目录
python -m examples.baselines.act.train_act_imitator --help
python -m examples.baselines.act.eval_act_imitator --help
```

Python 环境：`/home/zxc/miniconda3/envs/imitator`，Python 3.12。

源码：`/home/zxc/Imitator/The-Imitator-Game`；上游 commit：
`d6d16ec511bc389e0a207692730c137bc022ef14`；本地分支：`setup/imitator-env`。
本地依赖修改与生成的 `uv.lock` 共同定义安装版本。ManiSkill 从此源码目录 editable 安装。

## 本次验证结果

以下检查均已通过，详细机器记录见 [`setup-report.json`](setup-report.json)：

- CUDA 可用，GPU 为 RTX 4090，PyTorch CUDA runtime 为 12.8；BF16 矩阵运算和反向传播正常。
- ManiSkill 导入路径指向此项目；ACT 训练、评估与配对数据模块导入正常，两个 `--help` 入口正常退出。
- TorchCodec 解码一秒合成视频正常，项目自带视频读取函数可取出指定时间戳的帧。
- `Empty-v1` 中 Panda 使用 CPU 物理和 GPU 渲染完成 reset 与 3 次 step；`default` 和 `rt-fast` 均产生有效 RGB 图像。
- 官方预训练 DINOv2-L＋ResNet18＋ACT 用合成输入完成前向、反向和一次 AdamW 更新；ACT 输出 `[1, 24, 16]`，Adapter 有梯度，DINOv2 主干保持冻结。
- `uv sync --locked --no-dev --check` 通过，当前环境与锁文件一致。

预训练 DINOv2-L 的缓存 revision：`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`。
ACT 检查只使用 batch 1、两帧视频；约 1.47 GiB 的 PyTorch 分配峰值不能用于估算正式 batch 的总显存。
基础渲染场景不含 IG-10K 专用资产，尚未运行 PlaceMugRack / PickRemoteControl 任务。

当前空间：`/home` 可用约 **98.92 GiB**，根分区约 **200.09 GiB**。
Conda 环境约 12.08 GiB，uv 缓存约 13.20 GiB，DINOv2 与 ResNet18 缓存合计约 1.17 GiB。
未下载 IG-10K 数据集和 benchmark 资产压缩包。

## 路径与空间

Conda 环境变量已通过 `conda env config vars set -n imitator` 保存，激活时生效、退出时还原：

- `UV_PROJECT_ENVIRONMENT=/home/zxc/miniconda3/envs/imitator`：`uv sync` 安装到指定 Conda 环境。
- `UV_CACHE_DIR=/var/tmp/imitator-game-zxc/uv-cache`。
- `UV_LINK_MODE=copy`：缓存与 Conda 环境位于不同分区，使用复制。
- `HF_HOME=/var/tmp/imitator-game-zxc/hf-cache`。
- `HF_LEROBOT_HOME=/var/tmp/imitator-game-zxc/lerobot`。
- `TORCH_HOME=/var/tmp/imitator-game-zxc/torch-cache`。
- `MS_ASSET_DIR=/var/tmp/imitator-game-zxc/maniskill`。
- `TMPDIR=/var/tmp/imitator-game-zxc/tmp`。

注意：此版本源码会在 `MS_ASSET_DIR` 后再拼接 `data`，所以任务资产应放到
`/var/tmp/imitator-game-zxc/maniskill/data`。不要照抄 README 中的 `~/.maniskill/data`。

`/var/tmp` 路径适合本次缓存与启动阶段。长期训练数据、checkpoint 和研究结果应使用明确的持久目录并备份，且不应依赖操作系统的临时文件保留策略。

## 本地依赖处理

保留官方完整依赖集合，使用上游已有的 `tool.uv.override-dependencies` 机制固定兼容版本：

- Torch 2.7.1+cu128、Torchvision 0.22.1+cu128，匹配 TorchCodec 0.5。
- NumPy 1.26.4、Gymnasium 0.29.1，保持仓库数值与仿真接口约束。
- Transformers 4.53.3、Diffusers 0.36.0、HF Hub 0.35.3、PEFT 0.17.1。
- 两个 OpenCV 分发包固定 4.11.0.86，避免新版本要求 NumPy 2。
- TensorFlow / TensorFlow-CPU 固定 2.19.1，Zarr 固定 2.18.7。
- SciPy 1.15.3、ml-dtypes 0.5.1、Numpydantic 1.6.11、ContourPy 1.3.2、Tifffile 2025.3.30 保持 NumPy 1.x 兼容；Gradio 6.3.0 保持 HF Hub 0.x 兼容。
- r3m 使用固定 git commit `b2334e726887fa0206962d7984c69c5fb09cceab`。
- 不再全局允许任意预发布版本；锁定目标为本机 Linux x86_64。

上游 LeRobot 0.5.0 的 PyPI 依赖声明与仓库自身约束不一致，例如 Gymnasium、NumPy、Draccus、Diffusers 和 HF Hub。上游原本就通过 override 绕过这些冲突；本次延续该机制，并分别验证本项目实际使用的接口。原始声明保存在 `lerobot-0.5.0-requirements.json`。最终静态依赖审计及实际测试结果以同目录报告为准，不把 override 等同于所有 LeRobot 功能都兼容。

**静态依赖检查尚非全绿**：[`pip-check.txt`](pip-check.txt) 保存了 7 条声明冲突：

- LeRobot 的 Diffusers、HF Hub、NumPy、Draccus、Gymnasium 要求与本仓库选择不同，共 5 条。
- Rerun 要求 NumPy >= 2，而本仓库使用 1.26.4。
- dlimp 要求 TensorFlow 2.15.0，而本仓库要求 >= 2.16.1，本机安装 2.19.1。

已排除额外的新版本 SciPy 等兼容问题，以上 7 条延续上游显式覆盖涉及的冲突。ACT 导入、视频与合成计算检查通过；完整真实数据 loader、Rerun 可视化和 dlimp / π 系列链路未验证。扩展这些功能时应单独核查，不能用当前结果声称所有 baseline 均可运行。

系统 `nvcc` 仍为 12.0；当前验证使用 Torch wheel 携带的 CUDA 12.8 runtime。若以后编译自定义 CUDA 扩展，应再对齐编译工具链，本次没有修改系统 CUDA。

仓库 README 的 LeRobot 补丁 URL 在配置时返回 HTTP 404，且目标路径写为 Python 3.11，与项目要求的 Python 3.12 不符。本次没有执行该覆盖命令，也没有修改已安装包的元数据。

## 重新同步依赖

```bash
source /home/zxc/Imitator/scripts/activate_imitator.sh
uv sync --locked --no-dev
```

若要在这台机器重建已删除的环境，可先执行
`conda env create -f /home/zxc/Imitator/environment/conda-imitator.yml`，再运行上面的同步命令。

保留 `uv.lock` 和本地 `pyproject.toml` 修改；直接还原上游依赖文件再同步会丢失本次兼容约束。锁文件也记录可选开发依赖，但此次安装未启用开发组。
[`upstream-compat.patch`](upstream-compat.patch) 保存相对固定上游 commit 的改动；[`installed-requirements.txt`](installed-requirements.txt) 保存安装快照。复建以源码、兼容补丁和 `uv.lock` 为准。

## 验证命令

```bash
source /home/zxc/Imitator/scripts/activate_imitator.sh
python ../scripts/check_environment.py core
python ../scripts/check_environment.py video
python ../scripts/check_environment.py render
python ../scripts/check_environment.py render-rt
python ../scripts/check_environment.py act
```

每项输出 JSON 至 `environment/checks/`。渲染测试保存 PNG；视频测试保存一秒合成 MP4。
ACT 检查首次会下载官方 DINOv2-L 和 ResNet18 预训练权重，以两帧合成视频运行一次参数更新。
该检查使用 ACT 默认 18 维状态、16 维动作、24 步预测；只验证计算链路，不验证数据配对、任务闭环或 benchmark 成功率。

真实任务训练还需要 IG-10K 数据子集、任务资产和训练/评估配置；若使用作者 checkpoint，还需配套归一化与模型参数。

## 依据

- [固定版本官方仓库](https://github.com/imitator-game/The-Imitator-Game/tree/d6d16ec511bc389e0a207692730c137bc022ef14)。
- [TorchCodec 与 Torch 版本对应关系](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)。
- [LeRobot 0.5.0 发布元数据](https://pypi.org/pypi/lerobot/0.5.0/json)。
