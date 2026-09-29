# -*- coding: utf-8 -*-
"""
阶段② · 蒸馏训练入口
====================

示例（集群 L40S/L20，在线教师，与阶段①相同的 SGD+cosine 配方）：
    python scripts/train_distill.py \
        --model src/distill/yolov8n_student.yaml \
        --pretrained runs/baseline/baseline_n_sub100_ep100/weights/best.pt \
        --teacher yolov8l.pt --teacher-mode online \
        --data data/coco_sub100.yaml --epochs 100 --batch 32 --amp \
        --project runs/distill --name distill_n_sub100_ep100

1080 Ti（amp=False + 离线缓存，先在集群上跑 cache_teacher.py 生成缓存）：
    python scripts/train_distill.py \
        --model src/distill/yolov8n_student.yaml --pretrained yolov8n.pt \
        --teacher-mode cache --cache-dir runs/distill_cache/coco_sub100 \
        --data data/coco_sub100.yaml --epochs 100 --batch 16 --no-amp \
        --project runs/distill --name distill_n_sub100_cache

权重初始化与续训（显式区分）：
    * --pretrained xxx.pt  仅做学生初始化（start_epoch=0，推荐：阶段① best.pt 或 yolov8n.pt）
    * 不提供 --resume；蒸馏 best/last 只保存学生，需要重来时再次用 --pretrained 启动即可
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接从项目根目录 `python scripts/train_distill.py` 运行
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distill.trainer import DISTILL_DEFAULTS, DistillDetectionTrainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLOv8-L → 轻量学生 知识蒸馏训练")

    # ---- 模型 / 数据 ----
    p.add_argument("--model", default="src/distill/yolov8n_student.yaml", help="学生结构 yaml")
    p.add_argument("--pretrained", default="yolov8n.pt", help="学生初始化权重（阶段① best.pt 或 yolov8n.pt）")
    p.add_argument("--data", required=True, help="数据 yaml，如 data/coco_sub100.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32, help="学生训练 batch（基线 yolov8n=32）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cuda 设备号，如 0；留空自动选择")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=False, help="数据集缓存：False / ram / disk")
    p.add_argument("--project", default="runs/distill")
    p.add_argument("--name", default="distill_run")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--plots", action="store_true", help="是否输出训练曲线/混淆矩阵等图")

    # ---- AMP（1080Ti 必须 --no-amp；4060/L40S 必须 --amp）----
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")

    # ---- 训练配方（默认与阶段①基线完全一致）----
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--multi-scale", type=float, default=0.0)

    # ---- 蒸馏教师 ----
    p.add_argument("--teacher", default="yolov8l.pt", help="online 模式教师权重")
    p.add_argument("--teacher-mode", choices=["online", "cache"], default="online")
    p.add_argument("--teacher-batch", type=int, default=4, help="教师微批次（4060 8GB 建议 2~4）")
    p.add_argument("--cache-dir", default="", help="cache 模式的教师缓存目录")

    # ---- 蒸馏损失超参 ----
    p.add_argument("--kd-tau", type=float, default=DISTILL_DEFAULTS["kd_tau"], help="KL 温度")
    p.add_argument("--kd-cls-gain", type=float, default=DISTILL_DEFAULTS["kd_cls_gain"])
    p.add_argument("--kd-box-gain", type=float, default=DISTILL_DEFAULTS["kd_box_gain"], help="0=关闭框分布蒸馏")
    p.add_argument("--kd-feat-gain", type=float, default=DISTILL_DEFAULTS["kd_feat_gain"], help="0=关闭 CWD")
    p.add_argument("--cwd-tau", type=float, default=DISTILL_DEFAULTS["cwd_tau"])
    p.add_argument("--kd-adaptive", dest="kd_adaptive", action="store_true", default=True, help="置信度自适应权重")
    p.add_argument("--no-kd-adaptive", dest="kd_adaptive", action="store_false")
    p.add_argument("--kd-adaptive-gamma", type=float, default=DISTILL_DEFAULTS["kd_adaptive_gamma"])
    p.add_argument("--kd-warmup", type=float, default=DISTILL_DEFAULTS["kd_warmup_epochs"], help="蒸馏损失爬升轮数")

    # ---- 阶段③ 结构重参数化 ----
    p.add_argument("--rep", dest="use_reparam", action="store_true", default=False,
                   help="学生 C2f→RepC2f（训练多分支，部署前 fuse 为单路；用 scripts/reparam_deploy.py 融合）")
    p.add_argument("--no-rep", dest="use_reparam", action="store_false")

    # ---- 固定输入尺寸下的 cuDNN 加速（约 5~10%；放弃严格确定性，结果有微小随机差异）----
    p.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", default=False,
                   help="torch.backends.cudnn.benchmark=True（multi_scale=0、固定 640 时安全）")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ultralytics 官方识别的参数走 overrides
    overrides = dict(
        model=str(Path(args.model)),
        pretrained=str(Path(args.pretrained)),
        data=str(Path(args.data)),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        cache=args.cache,
        project=str(Path(args.project)),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        plots=args.plots,
        amp=args.amp,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        multi_scale=args.multi_scale,
        compile=False,
        verbose=True,
    )

    if args.cudnn_benchmark:
        # 必须在 trainer 构造前生效：BaseTrainer.__init__ 会用 deterministic 参数
        # 调 init_seeds()，开启 torch.use_deterministic_algorithms(True) +
        # cudnn.deterministic=True，之后再改全局状态也无法让 benchmark 选算法
        overrides["deterministic"] = False

    trainer = DistillDetectionTrainer(overrides=overrides)

    # 蒸馏专属参数（官方 cfg 不认识，构造后注入）
    trainer.args.teacher_mode = args.teacher_mode
    trainer.args.teacher_weights = args.teacher
    trainer.args.teacher_batch = args.teacher_batch
    trainer.args.cache_dir = args.cache_dir
    trainer.args.kd_tau = args.kd_tau
    trainer.args.kd_cls_gain = args.kd_cls_gain
    trainer.args.kd_box_gain = args.kd_box_gain
    trainer.args.kd_feat_gain = args.kd_feat_gain
    trainer.args.cwd_tau = args.cwd_tau
    trainer.args.kd_adaptive = args.kd_adaptive
    trainer.args.kd_adaptive_gamma = args.kd_adaptive_gamma
    trainer.args.kd_warmup_epochs = args.kd_warmup
    trainer.args.use_reparam = args.use_reparam

    if args.teacher_mode == "cache":
        if not args.cache_dir:
            raise SystemExit("teacher-mode=cache 时必须用 --cache-dir 指定缓存目录")
        # 缓存分辨率已固定，多尺度训练会造成师生特征空间尺寸不一致
        trainer.args.multi_scale = 0.0

    if args.cudnn_benchmark:
        import torch
        # deterministic=True 会强制 cudnn.deterministic 算法，使 benchmark 失效
        trainer.args.deterministic = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # init_seed(deterministic=False) 不会重置此项

    trainer.train()


if __name__ == "__main__":
    main()
