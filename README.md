# BCF6 reproduction

**Block-Compressed Features (BCf / BCF6)** 的独立复现项目，依据 Ubisoft La Forge 的 *Real-Time Neural Materials using Block-Compressed Features*（Eurographics 2024）。目标是在 Linux 上完成训练、标准 BC6H 导出和保存重载评测，并接入现有 external-25 材质数据集。

当前状态（2026-10-26，北京时间）：**仓库与复现规格已建立；训练器、位流导出器和运行时尚未实现，尚无本仓库训练结果。** `configs/` 是来源核实后的设计规格，不能直接作为可执行训练配置。

原论文称方法为 BCf；后续 BCF1 文献使用 BCF6 区分 BC6 与 BC1 路线。本项目是独立实现，作者项目入口见 [Clément Weinreich 主页](https://clement-w.github.io/)。

## 阅读入口

- [训练与推理逻辑（中文）](docs/METHOD_ZH.md)
- [现有数据集适配与评测口径](docs/DATASET_PROTOCOL_ZH.md)
- [实现顺序、验收与待核实细节](docs/IMPLEMENTATION_PLAN.md)
- [论文四档特征规格](configs/paper_profiles.json)
- [训练阶段规格](configs/training_spec.json)
- [外部 25 场评测规格](configs/external25_protocol.json)

核心流程：每个材质单独优化四组神经特征和一个小 MLP；5,000 步自由特征预训练后，用 BC6 编码选择块分区并初始化端点/索引；固定分区再训练 200,000 步；量化并冻结特征，微调解码器 1,000 步。部署时保存四张带 mip 的 BC6H 特征纹理和 FP16 网络参数，硬件采样后执行一次小 MLP，输出 PBR 材质属性供常规着色使用。

原论文设置与本数据集适配分别记录。论文常用 8 个输出（法线 XY），现有 RGB9 比较需要显式的 9 输出变体。公开仓库只收录原创源码、说明和不含本机路径的配置；材质原图、目标张量、权重、实验日志和机器凭据保留在各自数据/运行目录。

## 来源

- Weinreich, De Oliveira, Houdard, Nader. [Real-Time Neural Materials using Block-Compressed Features](https://arxiv.org/html/2311.16121v2)，重点为 §3–§6.2、表 1。DOI: [10.1111/cgf.15013](https://doi.org/10.1111/cgf.15013)。
- Microsoft: [BC6H Format](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format)，用于标准位流、端点、索引及精确解码规则。
- [BCF1 论文](https://arxiv.org/html/2506.06040v1)，用于核实 BCF6/BCF1 的命名关系。

实现与验收以 Linux 为先。Windows 运行及跨平台保存产物验收待后续执行。
