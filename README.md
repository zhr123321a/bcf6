# BCF6 reproduction

**Block-Compressed Features (BCf / BCF6)** 的独立复现项目，依据 Ubisoft La Forge 的 *Real-Time Neural Materials using Block-Compressed Features*（Eurographics 2024）。目标是在 Linux 上完成训练、标准 BC6H 导出和保存重载评测，并接入现有 external-25 材质数据集。

当前状态（2026-10-26，北京时间）：**训练器、Mode 10 编码、BC6H DDS 导出和独立重载已实现。Linux 的 6 项测试及真实数据短程训练通过。** 首个论文步数任务为 lubricant_spray / 1K / 8 输出 / 206,000 步，已交给持久调度器。完整训练状态与实现选择见 [复现记录 R01](docs/REPRODUCTION_R01_ZH.md)。硬件压缩纹理 shader 与 Windows 验收待完成。

原论文称方法为 BCf；后续 BCF1 文献使用 BCF6 区分 BC6 与 BC1 路线。本项目是独立实现，作者项目入口见 [Clément Weinreich 主页](https://clement-w.github.io/)。

## 阅读入口

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

## Linux 使用

在已有 PyTorch CUDA 环境内运行；`requirements.txt` 中的版本下限不是已测试版本矩阵。本次验证环境为 PyTorch 2.7.1+cu118、NumPy 和 GCC 11.4。

```bash
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest discover -s tests -v
python -m bcf6.storage --width 2048 --height 2048 --outputs 8
python -m bcf6.train --identity /path/to/material/identity.json \
  --out runs/material_1k_seed1 --profile 1K --outputs 8 --seed 1
```

训练默认使用 5,000 / 200,000 / 1,000 步和 512×512 个采样位置。运行目录保存阶段检查点、优化器和 RNG 状态、源码/目标 SHA256、损失与进度；完成后从实际部署文件独立重载、评分并做 CPU/GPU 重放，生成 `complete.json`。中断恢复使用相同配置并附加 `--resume runs/material_1k_seed1/latest.pt`，源码和数据身份必须一致。

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

保存后的参考推理：

```python
import torch
from bcf6.deployment import load_model, prediction_rgb9

model, metadata = load_model("runs/material_1k_seed1/deployment", "cuda")
uv = torch.tensor([[0.5, 0.5]], device="cuda")
with torch.no_grad():
    material8 = model(uv, scale=0.0)
    material9 = prediction_rgb9(material8)
```

这个参考路径将 BC6H 独立解码为浮点纹理后执行 PyTorch 采样和 MLP，用于验证保存产物。论文的实时路径直接使用 GPU BC6H 硬件采样；本仓库尚未测量该路径的帧时。`configs/` 保留论文规格及未公开的空值；当前可执行入口使用上述 CLI，实际选择写入每次运行的 `protocol.json`。

## 来源

- Weinreich, De Oliveira, Houdard, Nader. [Real-Time Neural Materials using Block-Compressed Features](https://arxiv.org/html/2311.16121v2)，重点为 §3–§6.2、表 1。DOI: [10.1111/cgf.15013](https://doi.org/10.1111/cgf.15013)。
- Microsoft: [BC6H Format](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format)，用于标准位流、端点、索引及精确解码规则。
- [BCF1 论文](https://arxiv.org/html/2506.06040v1)，用于核实 BCF6/BCF1 的命名关系。

实现与验收以 Linux 为先。Windows 运行及跨平台保存产物验收待后续执行。
