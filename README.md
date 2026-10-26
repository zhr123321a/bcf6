# BCF6 reproduction

**Block-Compressed Features (BCf / BCF6)** 的独立复现项目，依据 Ubisoft La Forge 的 *Real-Time Neural Materials using Block-Compressed Features*（Eurographics 2024）。目标是在 Linux 上完成训练、标准 BC6H 导出和保存重载评测，并接入现有 external-25 材质数据集。

**当前采用“论文明确配置 + 本仓库补充选择”，尚未确认与作者内部实现完全一致。** 特征尺寸、网络宽度、batch、阶段步数及前两阶段 LR/gamma 有原文依据；signed SF16、连续参数的具体表示、初始化编码器、Adam 细节及 scheduler 调用约定等由本仓库选择。建议先读[逐项设置审计](docs/PAPER_SETTINGS_AUDIT_ZH.md)，其中列出当前值、来源、影响和代码位置。

激活位置也属于补充解释：§5.1 只称 MLP 使用 ReLU，没有逐层指定。当前代码把 ReLU 放在隐藏层之后、输出层保持线性；这两项位置选择不能归为论文已经明确的逐层配置。

当前状态（2026-10-26，北京时间）：**训练器、Mode 10 编码、BC6H DDS 导出和独立重载已实现。Linux 的 6 项测试及真实数据短程训练通过。** 首个论文步数任务为 lubricant_spray / 1K / 8 输出 / 206,000 步，已交给持久调度器。完整训练状态与实现选择见 [复现记录 R01](docs/REPRODUCTION_R01_ZH.md)。硬件压缩纹理 shader 与 Windows 验收待完成。

原论文称方法为 BCf；后续 BCF1 文献使用 BCF6 区分 BC6 与 BC1 路线。本项目是独立实现，作者项目入口见 [Clément Weinreich 主页](https://clement-w.github.io/)。

## 阅读入口

- [逐项设置审计：论文设置与复现选择](docs/PAPER_SETTINGS_AUDIT_ZH.md)
- [训练与推理逻辑（中文）](docs/METHOD_ZH.md)
- [四档 bpp、存储组成与计算方法](docs/STORAGE_ZH.md)
- [复现记录与实现选择 R01](docs/REPRODUCTION_R01_ZH.md)
- [现有数据集适配与评测口径](docs/DATASET_PROTOCOL_ZH.md)
- [实现顺序、验收与待核实细节](docs/IMPLEMENTATION_PLAN.md)
- [论文四档特征规格](configs/paper_profiles.json)
- [训练阶段规格](configs/training_spec.json)
- [外部 25 场评测规格](configs/external25_protocol.json)

核心流程：每个材质单独优化四组神经特征和一个小 MLP；5,000 步自由特征预训练后，用 BC6 编码选择块分区并初始化端点/索引；固定分区再训练 200,000 步；量化并冻结特征，微调解码器 1,000 步。部署时保存四张带 mip 的 BC6H 特征纹理和 FP16 网络参数，硬件采样后执行一次小 MLP，输出 PBR 材质属性供常规着色使用。

首轮采用论文常用的 8 个输出（法线 XY）。评分同时报告这 8 个通道以及由 XY 重建法线 Z 后的 RGB9；`--outputs 9` 是另一个需要单独训练的适配配置。公开仓库收录源码、测试、说明和公开配置，数据与训练产物保留在运行目录。独立参考解码器的来源和许可见 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 建议的代码阅读顺序

| 文件 / 入口 | 阅读重点 |
|---|---|
| [train.py](bcf6/train.py) → `main` | 输入身份、三阶段切换、优化循环、保存后评价 |
| [train.py](bcf6/train.py) → `make_optimizer` | LR/gamma、Adam 细节和阶段重置 |
| [model.py](bcf6/model.py) → `BCF6.features` | 四组特征采样、LOD bias、拼接成 12 维再输入 MLP |
| [model.py](bcf6/model.py) → `FeatureMip.texture` | 端点 `[B,4,3]`、索引 `[B,16]`、分区 `[B]`，以及可微解码 |
| [model.py](bcf6/model.py) → `initialize_blocks` | 本仓库 PCA 初始化编码器如何选择固定分区 |
| [codec.py](bcf6/codec.py) | 标准整数解码、anchor、128-bit 打包和 DDS 布局 |
| [deployment.py](bcf6/deployment.py) | 实际文件重载、FP16 参数、质量与 CPU/GPU 重放 |
| [storage.py](bcf6/storage.py)、[tests/](tests/) | 载荷计数和验证样例 |

源码已展开紧凑语句，核心函数增加中文说明、张量形状和阶段注释。连续训练与最终标准整数解码分别实现，建议对照阅读 `FeatureMip.texture` 和 `codec.decode_parameters`。

## Linux 环境与基础检查

在已有 PyTorch CUDA 环境内运行；`requirements.txt` 中的版本下限不是已测试版本矩阵。本次验证环境为 PyTorch 2.7.1+cu118、NumPy 和 GCC 11.4。先安装与驱动相容的 PyTorch CUDA 包；以下命令从仓库根目录执行，不需要另设 `PYTHONPATH`。

```bash
git clone https://github.com/zhr123321a/bcf6.git
cd bcf6
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
python -m bcf6.train --help
python -m bcf6.storage --width 2048 --height 2048 --outputs 8
```

单元测试使用小型 CPU 样例，显式屏蔽 GPU。若 CUDA 检查为 False，先解决 PyTorch/驱动环境；若重载找不到 `build/libbcdec_ref.so`，先执行两条 CMake 命令。

## 准备数据

输入 `identity.json` 最少包含：

```json
{
  "scene": "material",
  "width": 2048,
  "height": 2048,
  "target_f32": "/absolute/path/to/target.f32",
  "target_sha256": "SHA256_OF_THE_TARGET_FILE"
}
```

目标是 little-endian float32、HWC、9 通道：Diffuse RGB、Normal RGB、AO、roughness、metalness。输入必须与已有评测口径一致；加载器不改变颜色空间或法线方向。

具体通道为 `0..2=Diffuse RGB`、`3..5=Normal XYZ 的 [0,1] 图像编码`、`6..8=AO/roughness/metalness`。文件无头，大小必须为 `W*H*9*4` 字节。已有冻结目标/身份文件应直接复用；移动目标位置后需更新 `target_f32` 路径，目标内容与 SHA256 不变。

若从相同尺寸、已按上述语义整理的 **8-bit RGB PNG** 开始，可使用下例。它只做 `/255` 保留原图编码，不做 sRGB 线性化、法线归一化、尺寸调整或缺失通道填充。

```bash
mkdir -p data/my_material
# 先将 diffuse.png、normal.png、orm.png 放入该目录。
python - <<'PY'
from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image

folder = Path("data/my_material").resolve()
arrays = []
for name in ("diffuse.png", "normal.png", "orm.png"):
    with Image.open(folder / name) as image:
        if image.mode != "RGB":
            raise ValueError(f"{name}: expected 8-bit RGB, got {image.mode}")
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
if len({a.shape for a in arrays}) != 1:
    raise ValueError("All three textures must have identical shapes")
target = folder / "target.f32"
np.concatenate(arrays, axis=-1).astype("<f4").tofile(target)
h, w, _ = arrays[0].shape
identity = dict(scene=folder.name, width=w, height=h,
                target_f32=str(target),
                target_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(folder / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
print(folder / "identity.json")
PY
```

HDR、16-bit、alpha、displacement 或其他通道/色彩约定需明确适配，不能直接套用此 PNG 示例。现有 external-25 包括 2048²、2048×1024、1024²，训练保持目标原尺寸。

## 先做短程训练

选一张允许使用的空闲物理 GPU；下例 `CUDA_VISIBLE_DEVICES=0` 按实际占用调整。一次只启动一个材质/档位；共享服务器上的多任务应使用已有资源调度器。

```bash
CUDA_VISIBLE_DEVICES=0 python -m bcf6.train \
  --identity data/my_material/identity.json \
  --out runs/my_material_1k_debug --profile 1K --outputs 8 --seed 1 \
  --warmup-steps 20 --block-steps 20 --finetune-steps 5 \
  --batch-side 512 --threads 2 --checkpoint-every 10 --log-every 10
```

这是 45 步流程检查，PSNR 不用于判断收敛。初始化、导出或 CPU 重放可能暂时没有逐步日志；全流程通过后才写 `complete.json`。核对 `status=complete`、`replay.continuation_allowed=true`；轻微数值 warning 允许继续，但保留 `strict_pass=false`。

## 使用论文已公开的步数与 batch

```bash
CUDA_VISIBLE_DEVICES=0 python -m bcf6.train \
  --identity data/my_material/identity.json \
  --out runs/my_material_1k_seed1 --profile 1K --outputs 8 --seed 1 \
  --warmup-steps 5000 --block-steps 200000 --finetune-steps 1000 \
  --batch-side 512 --threads 2 --checkpoint-every 1000 --log-every 100
```

总计 206,000 次更新：§6.1 的 205k 指前两阶段，额外 1k 来自 §6.2 的量化后微调。此命令匹配公开步数/batch，其余仍有本仓库选择；不会自动运行全 25 场。

| 参数 | 默认值 / 含义 |
|---|---|
| `--profile` | `1K`；另有 `0.5K`、`2K`、`2K++` |
| `--outputs` | `8`；`9` 是原生 RGB9 适配，需单独训练 |
| `--batch-side` | `512`；每步 `512²=262144` 个 UV，不是 batch=512 |
| `--warmup-steps / --block-steps / --finetune-steps` | `5000 / 200000 / 1000` |
| `--seed` | `1`；本仓库选择，不保证跨平台逐位相同 |
| `--device / --threads` | `cuda / 2`；CPU 可调试，不建议跑完整训练 |
| `--checkpoint-every / --log-every` | `1000 / 100`；阶段末也保存 |
| `--resume` | 同版本检查点路径 |

LR/gamma、Adam 和采样边界目前在源码中固定，没有对应 CLI。修改位置为 `train.make_optimizer`、`train.reference_mips`、`model.sample_image`；改动时同步协议记录并新建实验目录。**当前每次 update 后指数衰减**，主阶段 LR 为 `.01*.9999**t` / `.001*.9999**t`，最后 1k 步重置 MLP LR。衰减间隔和状态重置尚未由原文确认，详见[设置审计](docs/PAPER_SETTINGS_AUDIT_ZH.md)。

## 中断、恢复与版本

捕获 Ctrl+C/SIGTERM 后，训练在当前 update 结束时保存 `latest.pt`、写出 `stopped_checkpoint.json`，以返回码 **75** 退出。恢复需同一源码、目标和原始训练参数；不能只加载网络权重。

```bash
# 原命令参数保持一致，只添加 --resume；非默认参数也需原样传入。
CUDA_VISIBLE_DEVICES=0 python -m bcf6.train \
  --identity data/my_material/identity.json \
  --out runs/my_material_1k_seed1 --profile 1K --outputs 8 --seed 1 \
  --resume runs/my_material_1k_seed1/latest.pt
```

检查点含阶段/步数、模型、Adam、scheduler、Python/Torch/CUDA RNG 及数据/源码 SHA256。源码按原始字节校验，**只改注释也改变身份**。本注释版保留 R01 的数值运算，但旧 R01 检查点应使用原提交 `95617d6` 或实验冻结副本恢复；不要关闭身份检查硬接新版本。旧部署 DDS 的读取仅要求格式兼容，不依赖训练源码哈希。

已有 `protocol.json` 却没传 `--resume` 时拒绝覆盖；已有 `deployment/` 时导出也拒绝覆盖。当前恢复入口覆盖训练中断，没有实现导出/评分失败后的自动重试。已完成结果保留，调试重跑另设 `--out`。

## 运行目录与参考推理

```text
runs/my_material_1k_seed1/
  protocol.json             # 数据/源码身份和本次约定
  progress.json             # 最近记录的阶段、步数、loss、LR、时间
  loss.jsonl                # 随机 batch/LOD 损失，不是全图 PSNR 曲线
  latest.pt
  warmup_final.pt / block_final.pt / finetune_final.pt
  deployment/
    features_0.dds ... features_3.dds
    decoder_fp16.bin
    metadata.json           # 形状、参数布局、SHA256、实际字节
  replay.json
  complete.json             # 保存质量、载荷 bpp、耗时、验收状态
```

保存后的参考推理：

```python
import torch
from bcf6.deployment import load_model, prediction_rgb9

model, metadata = load_model("runs/my_material_1k_seed1/deployment", "cuda")
uv = torch.tensor([[0.5, 0.5]], device="cuda")  # [N,2]，归一化 UV
with torch.no_grad():
    material8 = model(uv, scale=0.0)          # Diffuse RGB、Normal XY、ORM
    material9 = prediction_rgb9(material8)   # 补充从 XY 重建的正 Normal Z
print(material9, metadata["payload_bpp"])
```

这个参考路径将 BC6H 独立解码为浮点纹理后执行 PyTorch 采样和 MLP，用于验证保存产物。保存网络为 FP16，参考运算为 FP32。`trained8_psnr` 对应监督通道，`rgb9_psnr` 还含 Z 重建误差；均为原尺寸 mip0、不 RGB8 舍入、不截断其他预测。论文主表跨 mip/材质汇总 MSE 后转 PSNR，口径不同。

论文的实时路径直接使用 GPU BC6H 硬件采样，本仓库尚未测量该帧时。`configs/` 是来源说明/设计规格，**当前 CLI 不解析这些 JSON**，修改其中数值不会改变训练。实际选择写入每次运行的 `protocol.json`。

## 来源

- Weinreich, De Oliveira, Houdard, Nader. [Real-Time Neural Materials using Block-Compressed Features](https://arxiv.org/html/2311.16121v2)，重点为 §3–§6.2、表 1。DOI: [10.1111/cgf.15013](https://doi.org/10.1111/cgf.15013)。
- Microsoft: [BC6H Format](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format)，用于标准位流、端点、索引及精确解码规则。
- [BCF1 论文](https://arxiv.org/html/2506.06040v1)，用于核实 BCF6/BCF1 的命名关系。

实现与验收以 Linux 为先。Windows 运行及跨平台保存产物验收待后续执行。

Windows CMake 命令为 `cmake -S . -B build -A x64`、`cmake --build build --config Release --parallel 2`，需 MSVC 及匹配的 PyTorch CUDA 环境，DLL 输出到 `build/bcdec_ref.dll`。上述 Bash 命令不直接等同于 PowerShell 语法；Windows 计算仍 pending。
