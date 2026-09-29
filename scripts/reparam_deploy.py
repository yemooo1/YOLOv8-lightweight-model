# -*- coding: utf-8 -*-
"""
阶段③ · 重参数化模型部署转换
============================

把 ``--rep`` 训练得到的 RepC2f 检查点（训练态多分支）：

  1. 数值校验：融合前/后在真实图片上逐框对比（NMS 后 box/类别/置信度应一致）；
  2. fuse 成纯单路 3x3 卷积（RepConv 三分支 → 一个 Conv2d + bias）；
  3. 保存 *_deploy.pt（纯 DetectionModel，与普通学生同构，可直接 val/导出）；
  4. 导出 ONNX（opset12 / imgsz640）；
  5. benchmark：训练态 vs 融合态参数量，以及与普通学生 ONNX 的 CPU 延迟对比。

用法：
    python scripts/reparam_deploy.py \
        --weights runs/watermelon/distill_rep_s2n_ep100/weights/best.pt \
        --plain-onnx runs/watermelon/distill_s2n_yolov8n_ep100/weights/best.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 必须先 import，pickle 才能解析 RepC2f/RepBottleneck 类
import src.reparam  # noqa: F401,E402


def benchmark_onnx(path: Path, img, runs: int = 50, warmup: int = 10) -> float:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = img.astype("float32")
    for _ in range(warmup):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / runs * 1000.0  # ms/张


def main():
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="RepC2f 训练态检查点（--rep 产出的 best/last.pt）")
    ap.add_argument("--plain-onnx", default="", help="普通学生 ONNX，用于延迟对比（可选）")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    w = Path(args.weights)
    assert w.exists(), w
    out_dir = w.parent
    stem = w.stem  # best / last

    yolo = YOLO(str(w))
    model = yolo.model
    model.eval()
    n_train = sum(p.numel() for p in model.parameters())

    # ---- 1. 融合前推理（取一张真实图，NMS 后结果）----
    test_imgs = sorted((ROOT / "datasets/watermelon/images/test").glob("*"))
    if test_imgs:
        probe = str(test_imgs[0])
        pil = Image.open(probe).convert("RGB").resize((args.imgsz, args.imgsz))
    else:
        probe = None
        pil = Image.fromarray(np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype="uint8"))
    pre = yolo.predict(pil, conf=0.25, imgsz=args.imgsz, device="cpu", verbose=False)[0]
    pre_boxes = np.round(pre.boxes.xyxy.cpu().numpy(), 3)
    pre_cls = pre.boxes.cls.cpu().numpy()
    pre_conf = np.round(pre.boxes.conf.cpu().numpy(), 4)
    print(f"[pre-fuse ] {len(pre_boxes)} boxes", flush=True)

    # ---- 2. fuse（RepConv 三分支 → 单 3x3；Conv+BN 折叠）----
    model.fuse(verbose=False)
    n_fused = sum(p.numel() for p in model.parameters())
    n_repconv = sum(1 for m in model.modules()
                    if type(m).__name__ == "RepConv" and hasattr(m, "conv1"))
    assert n_repconv == 0, f"仍存在未融合 RepConv: {n_repconv}"

    # ---- 3. 融合后推理 + 逐框数值一致性 ----
    post = yolo.predict(pil, conf=0.25, imgsz=args.imgsz, device="cpu", verbose=False)[0]
    post_boxes = np.round(post.boxes.xyxy.cpu().numpy(), 3)
    post_cls = post.boxes.cls.cpu().numpy()
    post_conf = np.round(post.boxes.conf.cpu().numpy(), 4)
    n_diff = len(pre_boxes) != len(post_boxes)
    cls_match = (not n_diff) and np.array_equal(pre_cls, post_cls)
    box_err = float(np.abs(pre_boxes - post_boxes).max()) if not n_diff else 9.9
    conf_err = float(np.abs(pre_conf - post_conf).max()) if not n_diff else 9.9
    print(f"[post-fuse] {len(post_boxes)} boxes | cls_match={cls_match} "
          f"max_box_err={box_err:.2e} max_conf_err={conf_err:.2e}", flush=True)
    if n_diff or not cls_match or box_err > 1e-2 or conf_err > 1e-3:
        print("WARNING: 融合前后结果超出容差，请检查！")

    # ---- 4. 保存融合态 pt + 导出 ONNX ----
    deploy_pt = out_dir / f"{stem}_deploy.pt"
    ckpt = torch.load(w, map_location="cpu", weights_only=False)
    torch.save(
        {"model": model, "train_args": ckpt.get("train_args", {}), "names": model.names},
        deploy_pt,
    )
    del ckpt
    print(f"saved {deploy_pt}", flush=True)

    deploy_yolo = YOLO(str(deploy_pt))
    deploy_onnx = Path(
        deploy_yolo.export(format="onnx", opset=args.opset, imgsz=args.imgsz,
                          simplify=True, dynamic=False, half=False)
    )
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        n_layers, n_p, n_grad, n_g = deploy_yolo.model.info(verbose=True, imgsz=args.imgsz)
    print(f"saved {deploy_onnx}", flush=True)

    # ---- 5. benchmark ----
    print("\n===== summary =====")
    print(f"训练态（多分支）参数 : {n_train/1e6:.3f}M")
    print(f"部署态（融合后）参数 : {n_fused/1e6:.3f}M")
    print(f"融合态 GFLOPs@{args.imgsz} : {n_g:.1f}")
    print(f"融合一致性: box_maxerr={box_err:.2e} conf_maxerr={conf_err:.2e} cls_match={cls_match}")

    x = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    t_rep = benchmark_onnx(deploy_onnx, x)
    print(f"CPU 延迟: RepC2f 融合 ONNX = {t_rep:.2f} ms/张")
    if args.plain_onnx and Path(args.plain_onnx).exists():
        t_plain = benchmark_onnx(Path(args.plain_onnx), x)
        print(f"CPU 延迟: 普通学生  ONNX = {t_plain:.2f} ms/张")
        print(f"延迟差异: {(t_rep-t_plain)/t_plain*100:+.1f}%（理论应≈0：部署图同构）")
    print("DEPLOY_DONE")


if __name__ == "__main__":
    main()
