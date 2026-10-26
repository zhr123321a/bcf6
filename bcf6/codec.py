"""Mode 10 的标准 BC6H 位流、精确整数解码和 DDS 文件布局。

本模块不构造可微训练图。q 为整数端点 [B,4,3]，idx 为 [0,7] 索引
[B,16]，p 为 [0,31] 分区 [B]。一个块固定 16 字节、解出 16 个 RGB texel。
训练器选 signed SF16；编解码测试同时覆盖 unsigned。独立 bcdec 用于交叉验证。
"""

from pathlib import Path
import ctypes
import re
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "third_party/bcdec/bcdec.h"
WEIGHTS = np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.int32)


def partition_table():
    """读取 bcdec 的规范分区表；最低位是 subset，高位 128 标记 anchor。"""
    text = HEADER.read_text(encoding="utf-8-sig")
    start = text.index("partition_sets[32][4][4]")
    body = text[start : text.index("};", start)]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S).split("=", 1)[1]
    values = [int(x) for x in re.findall(r"\d+", body)]
    if len(values) != 32 * 16:
        raise ValueError("Unexpected standard partition table")
    return np.asarray(values, dtype=np.uint8).reshape(32, 16)


PARTITIONS = partition_table()
# (endpoint, channel, least significant bit, bit count), in physical order.
FIELDS = [
    (0, 0, 0, 6),
    (3, 1, 4, 1),
    (3, 2, 0, 1),
    (3, 2, 1, 1),
    (2, 2, 4, 1),
    (0, 1, 0, 6),
    (2, 1, 5, 1),
    (2, 2, 5, 1),
    (3, 2, 2, 1),
    (2, 1, 4, 1),
    (0, 2, 0, 6),
    (3, 1, 5, 1),
    (3, 2, 3, 1),
    (3, 2, 5, 1),
    (3, 2, 4, 1),
    (1, 0, 0, 6),
    (2, 1, 0, 4),
    (1, 1, 0, 6),
    (3, 1, 0, 4),
    (1, 2, 0, 6),
    (2, 2, 0, 4),
    (2, 0, 0, 6),
    (3, 0, 0, 6),
]


def validate_parameters(endpoints, indices, partitions, signed):
    """检查形状、整数 dtype 和位宽范围；拒绝静默裁剪错误输入。"""
    q = np.asarray(endpoints)
    idx = np.asarray(indices)
    p = np.asarray(partitions)
    n = len(p)
    if q.shape != (n, 4, 3) or idx.shape != (n, 16) or p.shape != (n,):
        raise ValueError("Invalid Mode 10 parameter shapes")
    for x in [q, idx, p]:
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("Integer parameters required")
    if np.any((p < 0) | (p > 31)) or np.any((idx < 0) | (idx > 7)):
        raise ValueError("Partition/index out of range")
    lo, hi = (-32, 31) if signed else (0, 63)
    if np.any((q < lo) | (q > hi)):
        raise ValueError("Endpoint out of range")
    return q.astype(np.int32), idx.astype(np.int32), p.astype(np.int32)


def unquantize(q, signed=True):
    """标准整数端点反量化，包含零值和饱和端点的特殊分支。"""
    q = np.asarray(q, dtype=np.int32)
    if signed:
        a = np.abs(q)
        u = np.where(a == 0, 0, np.where(a >= 31, 32767, ((a << 15) + 16384) >> 5))
        return np.where(q < 0, -u, u)
    return np.where(q == 0, 0, np.where(q == 63, 65535, ((q << 16) + 32768) >> 6))


def finish_unquantize(v, signed=True):
    """标准缩放后重解释 FP16 位型，返回 float32 特征值。"""
    if signed:
        mag = (np.abs(v) * 31) >> 5
        bits = mag | np.where(v < 0, 32768, 0)
    else:
        bits = (v * 31) >> 6
    return np.asarray(bits, dtype="<u2").view("<f2").astype(np.float32)


def decode_parameters(endpoints, indices, partitions, signed=True):
    """精确整数解码到 [B,16,3]；包含标准权重表和右移舍入。"""
    q, idx, p = validate_parameters(endpoints, indices, partitions, signed)
    subset = (PARTITIONS[p] & 1).astype(np.int32)
    u = unquantize(q, signed)
    row = np.arange(len(p))[:, None]
    a, b = u[row, 2 * subset], u[row, 2 * subset + 1]
    w = WEIGHTS[idx][..., None]
    return finish_unquantize((a * (64 - w) + b * w + 32) >> 6, signed)


def canonicalize(endpoints, indices, partitions, signed=True):
    """必要时交换分区两端点并反转索引，使两个 anchor 的最高位为 0。

    anchor 少存一位；这次变换保持块的解码结果不变。"""
    q, idx, p = validate_parameters(endpoints, indices, partitions, signed)
    table = PARTITIONS[p]
    for subset in (0, 1):
        anchors = np.argmax(table == (128 + subset), axis=1)
        flip = idx[np.arange(len(p)), anchors] >= 4
        q[flip, 2 * subset : 2 * subset + 2] = q[flip, 2 * subset : 2 * subset + 2][
            :, ::-1
        ]
        m = flip[:, None] & ((table & 1) == subset)
        idx = np.where(m, 7 - idx, idx)
    return q, idx, p


def pack_blocks(endpoints, indices, partitions, signed=True):
    """把 q/idx/p 打包为连续 BC6H Mode 10 块。

    5 mode + 72 endpoint + 5 partition + 46 index = 128 bits。
    位序按规范排列，不能直接拼接四个 RGB 端点数组。"""
    q, idx, p = canonicalize(endpoints, indices, partitions, signed)
    q &= 63
    buf = np.zeros((len(p), 16), dtype=np.uint8)
    pos = 0

    def write(v, count):
        nonlocal pos
        for bit in range(count):
            buf[:, pos // 8] |= (((np.asarray(v) >> bit) & 1) << (pos % 8)).astype(
                np.uint8
            )
            pos += 1

    write(np.full(len(p), 30), 5)
    for e, c, s, b in FIELDS:
        write(q[:, e, c] >> s, b)
    write(p, 5)
    assert pos == 82
    for part in range(32):
        rows = np.flatnonzero(p == part)
        if not len(rows):
            continue
        at = 82
        for pixel in range(16):
            count = 2 if PARTITIONS[part, pixel] & 128 else 3
            for bit in range(count):
                buf[rows, at // 8] |= (
                    ((idx[rows, pixel] >> bit) & 1) << (at % 8)
                ).astype(np.uint8)
                at += 1
        assert at == 128
    return buf.tobytes()


def unpack_blocks(data, signed=True):
    """仅接受 Mode 10，按物理字段恢复 q/idx/p 以校验及再次导出。"""
    if len(data) % 16:
        raise ValueError("Partial BC6H block")
    buf = np.frombuffer(data, dtype=np.uint8).reshape(-1, 16)
    pos = 0

    def read(count):
        nonlocal pos
        out = np.zeros(len(buf), dtype=np.int32)
        for bit in range(count):
            out |= ((buf[:, pos // 8].astype(np.int32) >> (pos % 8)) & 1) << bit
            pos += 1
        return out

    if np.any(read(5) != 30):
        raise ValueError("Expected BC6H Mode 10")
    q = np.zeros((len(buf), 4, 3), dtype=np.int32)
    for e, c, s, b in FIELDS:
        q[:, e, c] |= read(b) << s
    p = read(5)
    if signed:
        q = np.where(q >= 32, q - 64, q)
    idx = np.zeros((len(buf), 16), dtype=np.int32)
    for part in range(32):
        rows = np.flatnonzero(p == part)
        at = 82
        for pixel in range(16):
            count = 2 if PARTITIONS[part, pixel] & 128 else 3
            for bit in range(count):
                idx[rows, pixel] |= (
                    (buf[rows, at // 8].astype(np.int32) >> (at % 8)) & 1
                ) << bit
                at += 1
    return q, idx, p


def blocks_to_image(blocks, width, height):
    """按块行优先和块内 texel 行优先顺序还原 HWC 图像，去除边缘补齐。"""
    bw, bh = (width + 3) // 4, (height + 3) // 4
    return (
        blocks.reshape(bh, bw, 4, 4, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(bh * 4, bw * 4, 3)[:height, :width]
        .copy()
    )


def image_to_blocks(image):
    """将 HWC 图像按 4×4 分块；不足整块的边缘使用 edge padding。"""
    h, w, c = image.shape
    if c != 3:
        raise ValueError("Expected three feature channels")
    a = np.pad(image, ((0, (-h) % 4), (0, (-w) % 4), (0, 0)), mode="edge")
    return (
        a.reshape(a.shape[0] // 4, 4, a.shape[1] // 4, 4, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 16, 3)
    )


def save_dds(path, width, height, mip_payloads, signed=True):
    """保存 148 字节 DDS+DX10 头及所有 mip；signed 对应 DXGI_FORMAT 96。"""
    if not mip_payloads:
        raise ValueError("Empty mip chain")
    for i, data in enumerate(mip_payloads):
        w, h = max(1, width >> i), max(1, height >> i)
        if len(data) != ((w + 3) // 4) * ((h + 3) // 4) * 16:
            raise ValueError("Mip payload size mismatch")
    h = [124, 0xA1007, height, width, len(mip_payloads[0]), 0, len(mip_payloads)] + [
        0
    ] * 11
    h += [32, 4, int.from_bytes(b"DX10", "little"), 0, 0, 0, 0, 0]
    h += [0x401008, 0, 0, 0, 0]
    assert len(h) == 31
    raw = (
        b"DDS "
        + struct.pack("<31I", *h)
        + struct.pack("<5I", 96 if signed else 95, 3, 0, 1, 0)
        + b"".join(mip_payloads)
    )
    Path(path).write_bytes(raw)
    return {
        "package_bytes": len(raw),
        "payload_bytes": len(raw) - 148,
        "header_bytes": 148,
    }


def load_dds(path):
    """核对 DDS 格式和完整 mip 字节，拒绝截断及多余尾部。"""
    raw = Path(path).read_bytes()
    if len(raw) < 148 or raw[:4] != b"DDS ":
        raise ValueError("Missing DDS header")
    h = struct.unpack("<31I", raw[4:128])
    dx = struct.unpack("<5I", raw[128:148])
    if (
        h[0] != 124
        or h[18] != 32
        or h[20] != int.from_bytes(b"DX10", "little")
        or dx[0] not in (95, 96)
        or dx[1:4] != (3, 0, 1)
    ):
        raise ValueError("Unsupported BC6H DDS format")
    width, height, count = h[3], h[2], h[6]
    if min(width, height, count) < 1 or count > max(width, height).bit_length():
        raise ValueError("Invalid DDS dimensions/mip count")
    offset = 148
    mips = []
    for i in range(count):
        w, hh = max(1, width >> i), max(1, height >> i)
        n = ((w + 3) // 4) * ((hh + 3) // 4) * 16
        mips.append((w, hh, raw[offset : offset + n]))
        offset += n
    if offset != len(raw):
        raise ValueError("DDS truncated or has trailing bytes")
    return dx[0] == 96, mips


def independent_decode(data, width, height, signed=True, library=None):
    """通过 C ABI 调用 vendored bcdec，交叉验证本仓库的打包/解码。

    build/libbcdec_ref.so（Linux）或 build/bcdec_ref.dll（Windows）
    必须先用 CMake 编译。该结果不依赖可微训练公式。"""
    import os

    libpath = (
        Path(library)
        if library
        else ROOT
        / "build"
        / ("bcdec_ref.dll" if os.name == "nt" else "libbcdec_ref.so")
    )
    lib = ctypes.CDLL(str(libpath))
    fn = lib.bcf6_decode_image
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    expected = ((width + 3) // 4) * ((height + 3) // 4) * 16
    if len(data) != expected:
        raise ValueError("Invalid image block count")
    src = np.frombuffer(data, dtype=np.uint8).copy()
    out = np.empty((height, width, 3), dtype=np.float32)
    if fn(src.ctypes.data, width, height, int(signed), out.ctypes.data) != 0:
        raise RuntimeError("Independent BC6H decoding failed")
    return out
