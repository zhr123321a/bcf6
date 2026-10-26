# external-25 数据集适配协议

本项目目标是使用现有 external-25 的相同材质输入，建立可与 DASTC、DBC 和 NTC 比较的 BCF6 结果。2026-10-26（北京时间）已绑定 lubricant_spray 的冻结输入，完成单场短程训练、实际保存与重放验证；完整步数任务进入持久队列，详见 [R01](REPRODUCTION_R01_ZH.md)。当前实现尚未报告 valid-channel mask 质量、逐 mip 分数或全 25 场结果，下文保留后续评测要求。

## 数据与输出

- 数据集范围为现有外部 25 场材质，包含 RTXNTC 示例、ambientCG、Poly Haven 和 lubricant_spray；与同样有 25 个目录的内部 Tencent 数据集区分。
- 复用已有输入 manifest 与原域 HWC RGB9 float32 target；导入时核实文件 SHA256、尺寸、通道布局和数据类型。路径通过用户本机配置提供，不写入公共规格。
- RGB9 顺序为 Diffuse RGB、Normal RGB、ORM（AO、roughness、metalness）。保持原始尺寸、现有色彩编码与法线约定，不擅自翻转法线、不额外施加 sRGB 线性化。
- 保留历史输入中已登记的合成常量通道和 valid-channel mask；附加报告有效通道分数。位移和 alpha 不属于当前 RGB9 比较范围。
- 按用户本轮“论文官方配置”的要求，首轮采用 normal XY、8 输出，标为 **BCF6-paper8，12→16→8**；同时报告 8 通道质量与 XY 重建正 Z 后的 RGB9 质量。后续可单独训练 **BCF6-RGB9，12→16→9** 适配，不能混用配置身份。法线 Z 重建误差计入 RGB9 分数。

## 分辨率与 mip

先实现论文表 1 的四组方形特征规格。目标材质按实际 W×H 输入，包括 2048×1024 等矩形材质；不为套用论文示例而把目标拉伸成方形。

若后续为等码率比较采用保持宽高比的特征配置，需要单独命名并记录实际形状。论文方形特征配置与这种适配不可共用同一配置身份。

保留 BCf 的独立 mip 参数及连续尺度训练。新增参考 mip 的生成滤波器、边界规则、坐标约定和 mip 尾端策略必须冻结。当前 DASTC 主表为 mip0 质量；BCF6 的完整部署载荷仍计入所有保存的 mip，同时另报逐 mip 与连续 UV/LOD 检查。论文按 mip/材质汇总 MSE 的分数与本项目逐场 dB 均值分别报告。

## 保存后质量与码率

主质量取真实 BC6H 文件和保存网络重新载入后的预测，使用原域浮点 RGB9、mip0、peak=1：

$$
\operatorname{MSE}=\frac{1}{9HW}\sum_{p,c}(\hat M_{p,c}-M_{p,c})^2,
\qquad \operatorname{PSNR}=-10\log_{10}(\operatorname{MSE}).
$$

不截断预测、不做 RGB8 舍入。零 MSE 保存为零，并显式表示无限 PSNR。多场主均值使用逐场 dB 等权平均，同时记录实际完成场数和缺失场景。

主码率为 `payload_bpp = 8 * payload_bytes / (W * H)`，byte scope 为 `payload_excluding_metadata_v1`。计入所有 BC6H 块（含各级 mip、模式/分区位和边缘补齐）、FP16 网络权重/偏置、实际数值 scale/bias 等。排除 DDS 头、架构描述、JSON 文本和校验清单；另存 `package_bytes` 与各文件 SHA256。网络若有对齐填充，记录逻辑参数字节与实际保存字节，主载荷按实际保存的数值缓冲计数。

原域 CPU/GPU 回放保留严格诊断：max absolute < 5e-5 且 RMS < 1e-6。轻微有限数值差异记录 warning 并允许继续；max absolute >= 0.05 或 RMS >= 0.01，以及非有限值、形状/格式/身份或损坏问题阻断对应产物。记录 `pixel_strict_pass`、`strict_pass`、`continuation_allowed`、阈值及 measurement policy 版本。BC6H **标准块解码本身**应按规范精确匹配参考解码器；材质网络回放容差不替代位流正确性验收。

## 执行与结果边界

Linux 上先完成单材质训练—导出—独立重载—评分，再扩展 25 场。多模型运行使用持久调度器，检查物理 GPU、显存与已有进程，共享资源上限 6 GPU / 32 CPU，成功缓存复用，失败隔离。跨场顺序使用终态依赖，只有真正需要产物的节点使用成功依赖。实际任务清单与预算在实现可运行后冻结。

Linux 质量通过后再测硬件压缩纹理常驻的 shader 推理。将 BC6H 整图先解码成浮点纹理的测试属于参考验证，不作为硬件 BC6H 推理性能结果。Windows 运行和跨平台保存重载另行记录，当前 pending。

本仓库不分发数据集、目标张量或训练模型。数据获取与再分发遵循各素材原始许可。
