"""单材质训练入口：读取原域 RGB9 → 三阶段优化 → 真正保存 → 独立重载评分。

阅读顺序：main 的阶段循环 → make_optimizer → model.FeatureMip.texture。
论文依据及本次补充约定见 docs/PAPER_SETTINGS_AUDIT_ZH.md。只有步数和
batch 匹配时 paper_step_and_batch_settings 才为真；该字段不证明全部设置等同作者实现。
"""

from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import BCF6, sample_image
from .deployment import (
    export_model,
    load_model,
    score_model,
    replay_check,
    save_json,
    sha,
)
from .storage import PROFILES


def now():
    """返回北京时间，用于用户可读的训练和检查点时间。"""
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))
    ).isoformat()


def load_reference(identity_path, device):
    """校验目标 SHA256/尺寸/有限值，返回 HWC float32 RGB9 与输入身份。

    通道顺序为 Diffuse RGB、Normal RGB、AO/roughness/metalness。
    加载器不改变色彩编码、法线方向或归一化范围。"""
    identity = json.loads(Path(identity_path).read_text())
    path = Path(identity["target_f32"])
    if sha(path) != identity["target_sha256"]:
        raise ValueError("Target SHA256 mismatch")
    w, h = int(identity["width"]), int(identity["height"])
    if path.stat().st_size != w * h * 9 * 4:
        raise ValueError("Target size mismatch")
    a = np.fromfile(path, dtype="<f4").reshape(h, w, 9)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite target")
    return torch.from_numpy(a).to(device), identity


def reference_mips(target, outputs):
    """选出监督通道，并用 2×2 均值逐级构造独立参考 mip。

    8 输出时跳过原始 Normal Z（通道 5）。均值滤波和最短边截止 4
    是本次实现选择；不能从特征表的 mip 数推定作者参考图的尾端策略。"""
    sel = [0, 1, 2, 3, 4, 6, 7, 8] if outputs == 8 else list(range(9))
    current = target[:, :, sel]
    levels = [current]
    while min(current.shape[:2]) > 4:
        current = (
            F.avg_pool2d(current.permute(2, 0, 1)[None], 2)[0]
            .permute(1, 2, 0)
            .contiguous()
        )
        levels.append(current)
    return levels


def reference_sample(levels, uv, scale):
    """参考侧采用相邻 mip 的 bicubic 采样，再按小数 LOD 线性混合。"""
    lod = max(0.0, min(scale, len(levels) - 1))
    lo = int(lod)
    hi = min(lo + 1, len(levels) - 1)
    f = lod - lo
    a = sample_image(levels[lo], uv, "bicubic")
    return (1 - f) * a + f * sample_image(levels[hi], uv, "bicubic") if f > 0 else a


def source_identity():
    """对源码原始字节做身份校验；修改注释也会使旧检查点的恢复校验失败。"""
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "bcf6").glob("*.py")) + [
        root / "native/bcdec_ref.cpp",
        root / "third_party/bcdec/bcdec.h",
    ]
    files = {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
    return {
        "files": files,
        "combined_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def make_optimizer(model, stage):
    """构造当前阶段的 Adam 和指数衰减器。

    前两阶段 LR/gamma 取论文数值；beta、epsilon、每次更新衰减、阶段
    重建 Adam，以及 finetune 的 .001/.9999 是复现选择。LR 的坐标单位
    由连续端点参数化决定，数值相同不等于作者内部优化轨迹相同。"""
    groups = [{"params": list(model.decoder.parameters()), "lr": 0.001}]
    if stage != "finetune":
        groups.insert(
            0,
            {
                "params": list(model.pyramids.parameters()),
                "lr": 0.05 if stage == "warmup" else 0.01,
            },
        )
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
    gamma = {"warmup": 0.9995, "block": 0.9999, "finetune": 0.9999}[stage]
    return optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


def checkpoint(path, model, optimizer, scheduler, stage, done, total, args, identity):
    """原子保存阶段、模型、Adam/scheduler 和随机状态，支持同版本恢复。"""
    obj = {
        "model_config": model.config(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stage": stage,
        "stage_updates": done,
        "total_updates": total,
        "args": vars(args),
        "target_sha256": identity["target_sha256"],
        "source": source_identity(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_rng": random.getstate(),
        "saved_at_bjt": now(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def main():
    """执行 warmup → block 初始化/优化 → 量化冻结 → MLP 微调 → 保存评测。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", choices=PROFILES, default="1K")
    p.add_argument("--outputs", type=int, choices=[8, 9], default=8)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--block-steps", type=int, default=200000)
    p.add_argument("--finetune-steps", type=int, default=1000)
    p.add_argument("--batch-side", type=int, default=512)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if (
        min(
            args.warmup_steps,
            args.block_steps,
            args.finetune_steps,
            args.batch_side,
            args.checkpoint_every,
            args.log_every,
        )
        < 1
    ):
        p.error("Positive steps and batch required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "protocol.json").exists() and args.resume is None:
        raise ValueError("Existing run; use a new output or explicit resume")
    target, identity = load_reference(args.identity, device)
    h, w, _ = target.shape
    levels = reference_mips(target, args.outputs)
    # 仅比较三阶段步数与 batch；official 不表示“全部官方设置”。
    # 初始化、signedness、衰减调用等补充选择见 protocol 和设置审计。
    official = (
        args.warmup_steps == 5000
        and args.block_steps == 200000
        and args.finetune_steps == 1000
        and args.batch_side == 512
    )
    protocol = dict(
        start_bjt=now(),
        source=source_identity(),
        target_sha256=identity["target_sha256"],
        scene=identity["scene"],
        width=w,
        height=h,
        profile=args.profile,
        outputs=args.outputs,
        seed=args.seed,
        phase_steps={
            "warmup": args.warmup_steps,
            "block": args.block_steps,
            "finetune": args.finetune_steps,
        },
        batch_side=args.batch_side,
        paper_step_and_batch_settings=official,
        loss="mean over batch and output channels",
        signedness="SF16; reproduction choice",
        initialization="all 32 Mode10 partitions; PCA in half-code domain; exact palette MSE selection",
        endpoint_parameters="continuous signed 6-bit code units, piecewise linear unquantize LUT",
        index_parameters="continuous normalized BC6 interpolation weight; nearest standard 3-bit weight at export",
        uv_sampling="uniform independent samples arranged as a batch-side squared batch",
        reference_mip_generation="2x2 average, down to 4 texels on shorter side",
        sampler="border padding; align_corners=False; per-feature LOD size bias",
        optimizer="Adam beta .9/.999 epsilon 1e-8; new optimizer at each phase",
        scheduler="gamma .9995 warmup/.9999 block, called after every update",
        finetune_learning_rate="reset decoder LR .001, gamma .9999; disclosed choice not specified by paper",
        decoder="12-16-c, hidden ReLU, linear output; FP32 training; FP16 saved weights",
        full_author_implementation_equivalence="unverified; no official trainer available",
        platform="Linux" if os.name == "posix" else "Windows",
        torch_version=torch.__version__,
        device=str(device),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        windows_validation="pending",
    )
    if args.resume is None:
        save_json(args.out / "protocol.json", protocol)
    stages = ["warmup", "block", "finetune"]
    counts = [args.warmup_steps, args.block_steps, args.finetune_steps]
    model = BCF6(args.profile, args.outputs, (w, h)).to(device)
    stage_idx = 0
    done = 0
    total = 0
    optimizer, scheduler = make_optimizer(model, "warmup")
    if args.resume:
        # 连注释修改也会改变源码身份；旧检查点应使用旧版源码恢复。
        # 模型、Adam、scheduler 和随机状态必须一起还原。
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            saved["target_sha256"] != identity["target_sha256"]
            or saved["source"]["combined_sha256"]
            != source_identity()["combined_sha256"]
        ):
            raise ValueError("Resume target or source identity mismatch")
        for name in [
            "profile",
            "outputs",
            "seed",
            "batch_side",
            "warmup_steps",
            "block_steps",
            "finetune_steps",
        ]:
            if saved["args"][name] != getattr(args, name):
                raise ValueError("Resume protocol mismatch: " + name)
        model = BCF6.from_state(saved["model_config"], saved["model"], device)
        stage_idx = stages.index(saved["stage"])
        done = saved["stage_updates"]
        total = saved["total_updates"]
        optimizer, scheduler = make_optimizer(model, saved["stage"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        random.setstate(saved["python_rng"])
    # 信号只置标志；update 完成后保存 latest.pt，以 75 退出。
    stop = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    history = args.out / "loss.jsonl"
    started = time.monotonic()
    last = time.monotonic()
    for k in range(stage_idx, 3):
        stage = stages[k]
        if k != stage_idx:
            done = 0
            if stage == "block":
                # 保留 MLP；每级自由特征经编码器选择固定分区和块初值。
                print(
                    json.dumps({"event": "block_initialization_started", "bjt": now()}),
                    flush=True,
                )
                t = time.monotonic()
                model = model.to_blocks()
                print(
                    json.dumps(
                        {
                            "event": "block_initialization_complete",
                            "seconds": time.monotonic() - t,
                            "bjt": now(),
                        }
                    ),
                    flush=True,
                )
            else:
                # 先离散量化并精确解码，再做 decoder-only 微调。
                model.freeze_quantized()
            optimizer, scheduler = make_optimizer(model, stage)
        while done < counts[k]:
            # N=batch_side² 个 IID UV，共享一个连续尺度。
            # MSE 对样本和通道同时求均值，不是全部 mip 的逐像素损失。
            uv = torch.rand(args.batch_side**2, 2, device=device)
            scale = random.uniform(0, len(levels) - 1)
            optimizer.zero_grad(set_to_none=True)
            ref = reference_sample(levels, uv, scale)
            pred = model(uv, scale)
            loss = (pred - ref).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            model.constrain()
            # 每次 update 后衰减是本次选择；原文未披露调用间隔。
            scheduler.step()
            done += 1
            total += 1
            if done == 1 or done % args.log_every == 0 or done == counts[k]:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                row = dict(
                    bjt=now(),
                    stage=stage,
                    stage_step=done,
                    total_step=total,
                    loss=float(loss.detach()),
                    scale=scale,
                    lr=[g["lr"] for g in optimizer.param_groups],
                    elapsed_seconds=time.monotonic() - started,
                    interval_seconds=time.monotonic() - last,
                )
                with history.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                save_json(args.out / "progress.json", row)
                print(json.dumps(row), flush=True)
                last = time.monotonic()
            if stop[0] or done % args.checkpoint_every == 0 or done == counts[k]:
                checkpoint(
                    args.out / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    stage,
                    done,
                    total,
                    args,
                    identity,
                )
                if stop[0]:
                    save_json(
                        args.out / "stopped_checkpoint.json",
                        dict(bjt=now(), stage=stage, total_step=total),
                    )
                    return 75
        # Keep phase boundaries; don't overwrite prior phase checkpoints.
        checkpoint(
            args.out / (stage + "_final.pt"),
            model,
            optimizer,
            scheduler,
            stage,
            done,
            total,
            args,
            identity,
        )
    # 最终质量从实际 DDS 和 FP16 MLP 独立重载后测量。
    # 既有 deployment 目录会拒绝覆盖；这里没有导出失败的自动恢复协议。
    exportdir = args.out / "deployment"
    meta = export_model(
        model,
        exportdir,
        {
            "scene": identity["scene"],
            "target_sha256": identity["target_sha256"],
            "source": source_identity(),
        },
    )
    reloaded, _ = load_model(exportdir, device)
    score = score_model(reloaded, target)
    cpu_model, _ = load_model(exportdir, "cpu")
    replay = replay_check(cpu_model, reloaded)
    save_json(args.out / "replay.json", replay)
    if not replay["continuation_allowed"]:
        raise ValueError("Saved material replay gross mismatch")
    package = sum(f.stat().st_size for f in exportdir.iterdir() if f.is_file())
    result = dict(
        status="complete",
        completed_bjt=now(),
        scene=identity["scene"],
        profile=args.profile,
        steps=total,
        paper_step_and_batch_settings=official,
        quality=score,
        replay=replay,
        payload_bytes=meta["payload_bytes"],
        payload_bpp=meta["payload_bpp"],
        package_bytes=package,
        feature_bytes_by_layer=meta["feature_bytes_by_layer"],
        decoder_bytes=meta["decoder_bytes"],
        elapsed_seconds=time.monotonic() - started,
        source=source_identity(),
        target_sha256=identity["target_sha256"],
        hardware_shader_validation="pending",
        windows_validation="pending",
    )
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    save_json(args.out / "complete.json", result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
