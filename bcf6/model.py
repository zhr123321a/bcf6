"""BCF6 的表示、连续 BC6H 松弛和特征采样。

张量约定：纹理 [H,W,3]，UV [N,2]，网络输入 [N,12]。
单个 mip 的 B 个 4×4 块保存端点 [B,4,3]、连续索引 [B,16]、分区 [B]。
训练用连续松弛；codec.py 负责最终标准整数解码，两条路径分别验收。
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .codec import (
    PARTITIONS,
    WEIGHTS,
    unquantize,
    pack_blocks,
    decode_parameters,
    blocks_to_image,
)
from .storage import PROFILES, mip_shapes


def halfcode_to_float(code):
    """把有符号 half 位码连续延拓为浮点值；整数位置与有限 FP16 位型一致。

    低于 1024 的位码对应 subnormal，其余按 exponent/mantissa 分段。
    这是本次选择的可微延拓，不应与论文式 (9) 的字面表达混为一谈。"""
    a = code.abs().clamp(max=31743)
    exponent = torch.floor(a / 1024)
    normal = (
        torch.sign(code) * torch.pow(2.0, exponent - 15) * (1 + a / 1024 - exponent)
    )
    return torch.where(a < 1024, code * (2.0**-24), normal)


def float_to_halfcode(value):
    """初始化用的反映射：浮点潜特征 → 有符号 half-code 坐标。"""
    a = value.abs().clamp(min=2.0**-24, max=65504.0)
    exponent = torch.floor(torch.log2(a)).clamp(-14, 15)
    normal = (exponent + 15) * 1024 + (a / torch.pow(2.0, exponent) - 1) * 1024
    code = torch.where(a < 2.0**-14, a * (2.0**24), normal)
    return torch.where(value == 0, torch.zeros_like(code), torch.sign(value) * code)


def torch_integer_palette(q):
    """计算精确 signed Mode 10 palette，输出 [B,2,8,3]。

    轴依次是块、分区、3-bit 索引、RGB。整数移位/舍入用于初始化评分，
    这条 no-grad 路径不提供训练中的连续梯度。"""
    a = q.abs().to(torch.int32)
    u = torch.where(a == 0, 0, torch.where(a >= 31, 32767, ((a << 15) + 16384) >> 5))
    u = torch.where(q < 0, -u, u).reshape(-1, 2, 2, 3)
    w = torch.as_tensor(WEIGHTS, device=q.device).view(1, 1, 8, 1)
    v = (u[:, :, 0, None] * (64 - w) + u[:, :, 1, None] * w + 32) >> 6
    code = (v.abs() * 31) >> 5
    bits = code | torch.where(v < 0, 32768, 0)
    return bits.to(torch.int16).view(torch.float16).float()


@torch.no_grad()
def initialize_blocks(image, chunk_blocks=4096):
    """本仓库的初始化编码器；论文未指定作者使用的编码器。

    对每块遍历 32 个分区，在 half-code 域做两组 PCA 线拟合，再按
    精确 BC6H 浮点重建 MSE 选分区。返回连续初值 endpoints [B,4,3]、
    interpolation weights [B,16]、固定 partition IDs [B]。"""
    h, w, _ = image.shape
    x = image.reshape(h // 4, 4, w // 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(-1, 16, 3)
    table = torch.as_tensor(PARTITIONS & 1, device=x.device, dtype=torch.long)
    qlevels = torch.arange(32, device=x.device, dtype=torch.int32)
    ulut = torch.as_tensor(unquantize(np.arange(32, dtype=np.int32)), device=x.device)
    clut = ((ulut * 31) >> 5).float()
    resultq = []
    resulti = []
    resultp = []
    for start in range(0, len(x), chunk_blocks):
        values = x[start : start + chunk_blocks]
        code = float_to_halfcode(values)
        n = len(values)
        row = torch.arange(n, device=x.device)
        best = torch.full((n,), float("inf"), device=x.device)
        bq = torch.empty((n, 4, 3), device=x.device)
        bi = torch.empty((n, 16), device=x.device)
        bp = torch.empty(n, dtype=torch.long, device=x.device)
        for p in range(32):
            endpoints = []
            for subset in (0, 1):
                sel = code[:, table[p] == subset]
                mean = sel.mean(1)
                centered = sel - mean[:, None]
                cov = centered.transpose(1, 2) @ centered
                axis = cov.diagonal(dim1=1, dim2=2).argmax(-1)
                direction = F.one_hot(axis, 3).to(code.dtype)
                for _ in range(8):
                    v = (cov @ direction[..., None])[..., 0]
                    norm = v.norm(dim=-1, keepdim=True)
                    direction = torch.where(
                        norm > 1e-12, v / norm.clamp_min(1e-12), direction
                    )
                proj = (centered * direction[:, None]).sum(-1)
                endpoints += [
                    mean + proj.min(1).values[:, None] * direction,
                    mean + proj.max(1).values[:, None] * direction,
                ]
            raw = torch.stack(endpoints, 1)
            # Nearest actual endpoint code includes zero and the saturated edge.
            q = (raw.abs()[..., None] - clut).abs().argmin(-1).to(
                torch.int32
            ) * torch.sign(raw).to(torch.int32)
            palette = torch_integer_palette(q)
            candidates = palette[:, table[p]]
            error = ((candidates - values[:, :, None]) ** 2).sum(-1)
            ix = error.argmin(-1)
            score = error.gather(2, ix[..., None])[..., 0].sum(-1)
            improved = score < best
            best = torch.minimum(best, score)
            bq[improved] = q[improved].float()
            bi[improved] = (
                torch.as_tensor(WEIGHTS, device=x.device)[ix[improved]].float() / 64
            )
            bp[improved] = p
        resultq.append(bq)
        resulti.append(bi)
        resultp.append(bp)
    return torch.cat(resultq), torch.cat(resulti), torch.cat(resultp)


class FeatureMip(nn.Module):
    """一个 mip 的三种状态：free 像素参数、block 连续块参数、quantized 精确纹理。"""

    def __init__(self, width, height, mode="free"):
        """创建当前状态的参数；分区及查表数据是 buffer，不参与梯度更新。"""
        super().__init__()
        self.width = width
        self.height = height
        self.mode = mode
        n = (width // 4) * (height // 4)
        if mode == "free":
            self.pixels = nn.Parameter(torch.randn(height, width, 3) * 0.05)
        else:
            self.endpoints = nn.Parameter(torch.zeros(n, 4, 3))
            self.indices = nn.Parameter(torch.full((n, 16), 0.5))
            self.register_buffer("partitions", torch.zeros(n, dtype=torch.long))
            self.register_buffer(
                "subset", torch.zeros(n, 16, dtype=torch.long), persistent=False
            )
            self.register_buffer(
                "unquant_lut",
                torch.as_tensor(
                    unquantize(np.arange(-31, 32, dtype=np.int32)), dtype=torch.float32
                ),
                persistent=False,
            )
            if mode == "quantized":
                self.register_buffer("frozen_texture", torch.zeros(height, width, 3))

    def refresh_partition_table(self):
        """把每块固定分区展开为每 texel 的 subset 0/1，供解码和重载使用。"""
        if self.mode != "free":
            self.subset = torch.as_tensor(
                PARTITIONS & 1, device=self.partitions.device, dtype=torch.long
            )[self.partitions]

    def texture(self):
        """恢复 HWC 特征纹理；block 模式保留端点和索引梯度。

        端点位于连续 [-31,31] code 坐标，先对标准反量化 LUT 线性插值，
        再混合两端点并作连续 half 转换。这里省去标准整数路径的舍入；
        最终冻结时才使用 codec.decode_parameters 的精确结果。"""
        if self.mode == "free":
            return self.pixels
        if self.mode == "quantized":
            return self.frozen_texture
        e = self.endpoints.clamp(-31, 31) + 31
        lo = e.floor().long()
        hi = (lo + 1).clamp(max=62)
        f = e - lo
        u = self.unquant_lut[lo] * (1 - f) + self.unquant_lut[hi] * f
        row = torch.arange(len(e), device=e.device)[:, None]
        a, b = u[row, 2 * self.subset], u[row, 2 * self.subset + 1]
        t = self.indices.clamp(0, 1)[..., None]
        code = ((1 - t) * a + t * b) * (31.0 / 32.0)
        decoded = halfcode_to_float(code)
        return (
            decoded.reshape(self.height // 4, self.width // 4, 4, 4, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(self.height, self.width, 3)
        )

    @torch.no_grad()
    def initialize_from(self, image):
        """用预训练自由特征初始化四端点、连续索引和固定分区。"""
        e, i, p = initialize_blocks(image)
        self.endpoints.copy_(e)
        self.indices.copy_(i)
        self.partitions.copy_(p)
        self.refresh_partition_table()

    @torch.no_grad()
    def integer_parameters(self):
        """端点舍入至整数 code，连续索引映射到最近标准 3-bit 权重。

        权重是 [0,9,18,27,37,46,55,64]/64，并非均匀的 idx/7。"""
        q = self.endpoints.round().clamp(-31, 31).cpu().numpy().astype(np.int32)
        weights = torch.as_tensor(WEIGHTS, device=self.indices.device).float() / 64
        idx = (
            (self.indices.clamp(0, 1)[..., None] - weights)
            .abs()
            .argmin(-1)
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        p = self.partitions.cpu().numpy().astype(np.int32)
        return q, idx, p

    @torch.no_grad()
    def freeze_quantized(self):
        """量化并精确解码一次，冻结特征；此后仅 MLP 接受更新。"""
        q, idx, p = self.integer_parameters()
        data = blocks_to_image(decode_parameters(q, idx, p), self.width, self.height)
        self.mode = "quantized"
        self.register_buffer(
            "frozen_texture", torch.as_tensor(data, device=self.endpoints.device)
        )
        self.endpoints.requires_grad_(False)
        self.indices.requires_grad_(False)

    @torch.no_grad()
    def constrain(self):
        """限制端点/索引范围；当前每步调用所有 mip，未做活跃层优化。"""
        if self.mode == "block":
            self.endpoints.clamp_(-31, 31)
            self.indices.clamp_(0, 1)


def sample_image(image, uv, mode="bilinear"):
    """把 HWC 纹理和 [0,1] UV 转成 grid_sample 布局，返回 [N,C]。

    border 和 align_corners=False 是本次约定；(.5/W,.5/H) 对齐像素中心。"""
    pixels = image.permute(2, 0, 1)[None]
    grid = (uv * 2 - 1).reshape(1, 1, -1, 2)
    return F.grid_sample(
        pixels, grid, mode=mode, padding_mode="border", align_corners=False
    )[0, :, 0].T


class BCF6(nn.Module):
    """四个独立 mip 金字塔的特征拼接，加单隐藏层 12→16→c MLP。"""

    def __init__(
        self, profile="1K", outputs=8, target_size=(2048, 2048), mode="free", sides=None
    ):
        """profile 固定方形特征尺寸；target_size=(W,H) 用于 LOD bias 和 bpp。

        sides 仅供小型测试；CLI 的四档直接使用论文表 1。隐藏 ReLU 后
        使用线性输出层；自由特征 N(0,.05²) 与 Linear 默认初始化是补充选择。"""
        super().__init__()
        self.profile = profile
        self.outputs = outputs
        self.target_size = tuple(target_size)
        self.mode = mode
        self.sides = list(PROFILES[profile] if sides is None else sides)
        self.pyramids = nn.ModuleList(
            [
                nn.ModuleList([FeatureMip(w, h, mode) for w, h in mip_shapes(n, n)])
                for n in self.sides
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, outputs)
        )
        if mode == "quantized":
            for pyramid in self.pyramids:
                for mip in pyramid:
                    mip.endpoints.requires_grad_(False)
                    mip.indices.requires_grad_(False)

    def features(self, uv, scale):
        """每组选择两个相邻 mip 做 bilinear+LOD，拼成 [N,12]。

        采用尺寸比的 log2 bias，再 clamp 到本组有效 LOD。当前先完整
        解码选中 mip 再采样，是用于正确性验证的 PyTorch 实现。"""
        result = []
        for side, pyramid in zip(self.sides, self.pyramids):
            bias = math.log2(
                max(side / self.target_size[0], side / self.target_size[1])
            )
            lod = max(0.0, min(float(scale) + bias, len(pyramid) - 1))
            lo = int(lod)
            hi = min(lo + 1, len(pyramid) - 1)
            fraction = lod - lo
            value = sample_image(pyramid[lo].texture(), uv)
            if fraction > 0:
                value = (1 - fraction) * value + fraction * sample_image(
                    pyramid[hi].texture(), uv
                )
            result.append(value)
        return torch.cat(result, -1)

    def forward(self, uv, scale):
        """UV/scale 控制采样，不额外拼成网络输入；输出 [N,c] 材质通道。"""
        return self.decoder(self.features(uv, scale))

    @torch.no_grad()
    def to_blocks(self):
        """warmup 结束后转块参数；保留 MLP，逐 mip 压缩自由特征。"""
        model = BCF6(
            self.profile, self.outputs, self.target_size, "block", self.sides
        ).to(next(self.parameters()).device)
        model.decoder.load_state_dict(self.decoder.state_dict())
        for oldp, newp in zip(self.pyramids, model.pyramids):
            for old, new in zip(oldp, newp):
                new.initialize_from(old.texture())
        return model

    @torch.no_grad()
    def freeze_quantized(self):
        """所有特征转为最终精确离散值，进入 decoder-only 微调。"""
        for pyramid in self.pyramids:
            for mip in pyramid:
                mip.freeze_quantized()
        self.mode = "quantized"

    @torch.no_grad()
    def constrain(self):
        """对全部 mip 做约束；这也是当前尚可优化的每步额外开销。"""
        for pyramid in self.pyramids:
            for mip in pyramid:
                mip.constrain()

    def config(self):
        """返回架构/形状描述；训练状态与部署 metadata 均使用此配置。"""
        return dict(
            profile=self.profile,
            outputs=self.outputs,
            target_size=list(self.target_size),
            mode=self.mode,
            sides=self.sides,
        )

    @classmethod
    def from_state(cls, config, state, device="cpu"):
        """先重建正确阶段的参数形状，再载入状态并恢复派生 subset 表。"""
        model = cls(**config).to(device)
        model.load_state_dict(state)
        for pyramid in model.pyramids:
            for mip in pyramid:
                mip.refresh_partition_table()
        return model
