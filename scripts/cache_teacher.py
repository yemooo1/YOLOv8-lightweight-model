# -*- coding: utf-8 -*-
"""
阶段② · 教师输出离线缓存生成（方案 4.3.2）
==========================================

在有大显存/快 GPU 的机器（L40S/L20，或 4060 fp16）上，按 epoch 预跑冻结教师，
把「增强后的图像 + 教师 logits + 教师颈部特征」逐 step 落盘，供 1080 Ti 等弱机器
以 cache 模式训练学生（scripts/train_distill.py --teacher-mode cache）。

示例：
    python scripts/cache_teacher.py \
        --teacher yolov8l.pt --data data/coco_sub100.yaml \
        --epochs 100 --batch 16 --imgsz 640 --amp \
        --cache-dir runs/distill_cache/coco_sub100

只补生成第 e 轮（例如上次中断）：
    python scripts/cache_teacher.py ... --start-epoch 12 --end-epoch 13

注意：
    * 增强参数（mosaic/mixup/close_mosaic/imgsz/batch）必须与学生训练一致，
      meta.json 会记录这些值，学生侧启动时做校验；
    * cache 模式学生会关闭 multi_scale，本脚本固定 imgsz；
    * 磁盘估算（fp16，640x640，yolov8-l 教师）：每张图约
      0.85MB 图像 + 2.4MB logits + 1.7MB 特征 ≈ 5MB；sub10 单 epoch 约 57GB。
      --no-feats / --no-boxes 可进一步压缩（相应蒸馏项在训练时自动需要关闭，见提示）。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from copy import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.data.build import seed_worker  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr  # noqa: E402

from src.distill.cache import (  # noqa: E402
    cache_epoch_dir,
    save_cache_meta,
    batch_to_cpu_uint8,
)
from src.distill.model import _pred_dict  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="预计算 YOLOv8-L 教师输出并落盘")
    p.add_argument("--teacher", default="yolov8l.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--start-epoch", type=int, default=0, help="起始 epoch（含），用于断点补生成")
    p.add_argument("--end-epoch", type=int, default=None, help="结束 epoch（不含），默认=epochs")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-images", default=False, help="数据集图像缓存 False/ram/disk")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--teacher-batch", type=int, default=8, help="教师单次前向微批次")
    p.add_argument("--keep", type=int, default=0, help="只保留最近 N 个 epoch（0=全部保留）")
    # 与训练一致的增强配方（默认值必须与 train_distill.py 对齐）
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--fraction", type=float, default=1.0)
    # 体积控制
    p.add_argument("--no-feats", action="store_true", help="不存颈部特征（学生必须 --kd-feat-gain 0）")
    p.add_argument("--no-boxes", action="store_true", help="不存框分布（学生必须 --kd-box-gain 0）")
    return p


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_cfg(args):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.task = "detect"
    cfg.imgsz = args.imgsz
    cfg.batch = args.batch
    cfg.cache = args.cache_images
    cfg.fraction = args.fraction
    cfg.rect = False
    cfg.single_cls = False
    cfg.classes = None
    # 增强
    cfg.mosaic = args.mosaic
    cfg.mixup = args.mixup
    cfg.close_mosaic = args.close_mosaic
    cfg.copy_paste = 0.0
    return cfg


def run_teacher_chunked(teacher, img: torch.Tensor, micro: int, device_type: str, amp: bool) -> dict:
    scores, boxes, feats = [], [], []
    for x in torch.split(img, micro, dim=0):
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=amp and device_type == "cuda"
        ):
            pred = _pred_dict(teacher(x))
        scores.append(pred["scores"].detach())
        boxes.append(pred["boxes"].detach())
        feats.append(pred["feats"])
    out = {
        "scores": torch.cat(scores, 0),
        "boxes": torch.cat(boxes, 0),
        "feats": [torch.cat([b[i] for b in feats], 0) for i in range(len(feats[0]))],
    }
    return out


def filter_teacher_outputs(pred: dict, no_feats: bool, no_boxes: bool) -> dict:
    pred = {
        "scores": pred["scores"],
        "boxes": pred["boxes"] if not no_boxes else None,
        "feats": pred["feats"] if not no_feats else [],
    }
    return pred


def main() -> None:
    args = build_parser().parse_args()
    end_epoch = args.end_epoch if args.end_epoch is not None else args.epochs
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # ---- 数据 ----
    data = check_det_dataset(args.data)
    gs = 32
    base_cfg = build_data_cfg(args)

    # ---- 教师 ----
    LOGGER.info(f"{colorstr('教师模型')}: {args.teacher} → {device} (amp={args.amp})")
    teacher = load_checkpoint(args.teacher)[0].to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    nc = int(teacher.model[-1].nc)
    reg_max = int(teacher.model[-1].reg_max)

    # ---- 探针：记录通道/anchor 数等元信息 ----
    with torch.no_grad():
        dummy = torch.zeros(2, 3, args.imgsz, args.imgsz, device=device)
        probe = _pred_dict(teacher(dummy))
    feat_channels = [int(f.shape[1]) for f in probe["feats"]]
    n_anchors = int(probe["scores"].shape[-1])

    meta = {
        "teacher": str(args.teacher),
        "nc": nc,
        "reg_max": reg_max,
        "imgsz": args.imgsz,
        "n_anchors": n_anchors,
        "feat_channels": feat_channels,
        "batch": args.batch,
        "epochs": args.epochs,
        "base_seed": args.seed,
        "aug": {
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "close_mosaic": args.close_mosaic,
            "no_feats": bool(args.no_feats),
            "no_boxes": bool(args.no_boxes),
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    save_cache_meta(cache_dir, meta)
    LOGGER.info(f"meta.json 已写入 {cache_dir / 'meta.json'}：{json.dumps(meta, ensure_ascii=False)}")
    if args.no_feats or args.no_boxes:
        LOGGER.warning(
            f"体积裁剪：no_feats={args.no_feats}, no_boxes={args.no_boxes}；"
            f"学生训练时必须对应设置 --kd-feat-gain 0 / --kd-box-gain 0。"
        )

    # ---- 逐 epoch 生成 ----
    for epoch in range(args.start_epoch, end_epoch):
        seed_all(args.seed + epoch)
        cfg = copy(base_cfg)
        dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", rect=False, stride=gs)
        if args.close_mosaic > 0 and epoch >= args.epochs - args.close_mosaic:
            dataset.close_mosaic(hyp=copy(cfg))  # 与 trainer 的最后 N 轮关 Mosaic 行为一致
            LOGGER.info(f"epoch {epoch}: close_mosaic 生效（mosaic/mixup/copy_paste=0）")

        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            pin_memory=device_type == "cuda",
        )

        epoch_dir = cache_epoch_dir(cache_dir, epoch)
        if epoch_dir.exists():
            shutil.rmtree(epoch_dir)  # 重跑该 epoch 时先清旧分片，避免混入残留
        epoch_dir.mkdir(parents=True, exist_ok=True)

        n = len(loader)
        for step, batch in enumerate(loader):
            img = batch["img"].to(device, non_blocking=True).float() / 255.0
            pred = run_teacher_chunked(teacher, img, args.teacher_batch, device_type, args.amp)
            pred = filter_teacher_outputs(pred, args.no_feats, args.no_boxes)
            pred_cpu = {
                "scores": pred["scores"].detach().to("cpu", torch.float16),
                "boxes": None if args.no_boxes else pred["boxes"].detach().to("cpu", torch.float16),
                "feats": [] if args.no_feats else [f.detach().to("cpu", torch.float16) for f in pred["feats"]],
            }
            shard = {"batch": batch_to_cpu_uint8(batch), "__teacher_cache__": pred_cpu}
            torch.save(shard, epoch_dir / f"shard_{step:06d}.pt")
            if step % 20 == 0 or step == n - 1:
                LOGGER.info(f"epoch {epoch} [{step + 1}/{n}] → {epoch_dir.name}/shard_{step:06d}.pt")

        LOGGER.info(f"epoch {epoch} 缓存完成：{n} 个分片 @ {epoch_dir}")

        # ---- 磁盘清理：只保留最近 keep 个 epoch ----
        if args.keep > 0:
            old = cache_epoch_dir(cache_dir, epoch - args.keep)
            if old.is_dir():
                shutil.rmtree(old)
                LOGGER.info(f"已清理旧缓存 {old}（keep={args.keep}）")

    LOGGER.info(f"全部缓存任务完成：{cache_dir}")


if __name__ == "__main__":
    main()
