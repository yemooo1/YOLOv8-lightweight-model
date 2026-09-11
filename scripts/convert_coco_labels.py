# -*- coding: utf-8 -*-
"""
S1 · COCO json 标注 → YOLO txt 格式
====================================

Ultralytics 会在图片路径里把 `images` 段替换成 `labels` 来找标签文件，
所以目录必须成对：

    datasets/coco/
    ├── images/train2017/xxx.jpg       ← 解压得到
    ├── images/val2017/xxx.jpg
    ├── labels/train2017/xxx.txt       ← 本脚本生成
    ├── labels/val2017/xxx.txt
    └── annotations/instances_*.json

类别映射用 COCO 的 category_id 排序后的稠密索引（0–79），与 Ultralytics
官方 coco.yaml 的 names 完全一致。注意 COCO 原始 id 是 1–90 有跳号，
直接拿 id 当类别号会错位。

用法:
    E:/AIC/.venv/Scripts/python.exe scripts/convert_coco_labels.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
COCO_ROOT = DEFAULT_COCO_ROOT
ANN_DIR = COCO_ROOT / "annotations"
LABEL_DIR = COCO_ROOT / "labels"


def convert(split: str, skip_crowd: bool = True) -> dict:
    ann_file = ANN_DIR / f"instances_{split}.json"
    if not ann_file.exists():
        print(f"  ⚠ 找不到 {ann_file}，跳过 {split}")
        return {}

    print(f"\n[{split}]")
    print(f"  读取 {ann_file.name} …", flush=True)
    t0 = time.time()
    with open(ann_file, "r", encoding="utf-8") as f:
        ann = json.load(f)

    # category_id -> 稠密索引（与 Ultralytics coco.yaml 一致）
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    print(f"  类别数: {len(cats)}（id {cats[0]['id']}..{cats[-1]['id']} → 索引 0..{len(cats)-1}）")

    # 图片信息
    images = {im["id"]: im for im in ann["images"]}
    print(f"  图片数: {len(images):,}")

    # 按图片分组标注
    per_img = defaultdict(list)
    for a in ann["annotations"]:
        if skip_crowd and a.get("iscrowd", 0):
            continue
        per_img[a["image_id"]].append(a)

    out_dir = LABEL_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_boxes = 0
    n_skipped = 0
    n_empty = 0

    for img_id, im in images.items():
        w_img, h_img = im["width"], im["height"]
        stem = Path(im["file_name"]).stem
        lines = []

        for a in per_img.get(img_id, []):
            x, y, bw, bh = a["bbox"]
            cls = id2idx.get(a["category_id"])
            if cls is None:
                n_skipped += 1
                continue
            if bw <= 0 or bh <= 0:
                n_skipped += 1
                continue

            # 裁剪到图像范围内（COCO 存在越界框）
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w_img), x + bw), min(float(h_img), y + bh)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                n_skipped += 1
                continue

            # YOLO: 归一化的 中心x 中心y 宽 高
            cx = (x1 + bw / 2) / w_img
            cy = (y1 + bh / 2) / h_img
            nw = bw / w_img
            nh = bh / h_img
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 无目标的图片也要生成空 txt（YOLO 视为背景图）
        if not lines:
            n_empty += 1
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                             encoding="utf-8")
        n_written += 1
        n_boxes += len(lines)

        if n_written % 20000 == 0:
            print(f"    已写 {n_written:,}/{len(images):,} …", flush=True)

    el = time.time() - t0
    print(f"  完成: {n_written:,} 个 txt, {n_boxes:,} 个框, 用时 {el:.0f}s")
    if n_empty:
        print(f"  其中 {n_empty:,} 张为无目标背景图（空 txt）")
    if n_skipped:
        print(f"  跳过 {n_skipped:,} 个异常框（类别未知 / 宽高非正 / 越界）")

    return {"images": n_written, "boxes": n_boxes, "classes": len(cats)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="COCO 标注转 YOLO txt")
    ap.add_argument("--coco-root", type=str, default=None,
                    help="COCO 数据目录（默认 项目/datasets/coco；集群上指向 jhaidata）")
    args = ap.parse_args()

    global COCO_ROOT, ANN_DIR, LABEL_DIR
    if args.coco_root:
        COCO_ROOT = Path(args.coco_root).resolve()
        ANN_DIR = COCO_ROOT / "annotations"
        LABEL_DIR = COCO_ROOT / "labels"

    print("=" * 68)
    print("S1 · COCO 标注转 YOLO 格式")
    print("=" * 68)

    if not ANN_DIR.exists():
        print(f"✗ 找不到 {ANN_DIR}，请先解压 annotations_trainval2017.zip")
        sys.exit(1)

    results = {}
    for split in ("val2017", "train2017"):     # 先转小的，便于快速验证
        results[split] = convert(split)

    # 汇总
    print()
    print("=" * 68)
    print("汇总")
    print("=" * 68)
    total_boxes = 0
    for split, r in results.items():
        if r:
            print(f"  {split:12s} {r['images']:>7,} txt   {r['boxes']:>9,} 框")
            total_boxes += r["boxes"]
    print(f"  {'合计':12s} {'':>7}       {total_boxes:>9,} 框")
    print()
    print("COCO2017 官方参考值: train2017 约 860k 框 / val2017 约 36k 框（不含 crowd）")
    print("若你的数字明显偏离，说明标注文件或映射有误。")


if __name__ == "__main__":
    main()
