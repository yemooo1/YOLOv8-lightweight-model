# -*- coding: utf-8 -*-
"""
S1 · COCO2017 按类别分层抽样，生成子集配置
==========================================

读 datasets/coco/annotations/instances_train2017.json，按「主类别」分层抽样
10% / 25% / 50%（seed=0），保证各类别在子集中的占比与全集一致。

输出
----
  data/coco_sub10.yaml / coco_sub25.yaml / coco_sub50.yaml  数据集配置
  data/sub{10,25,50}_train.txt                              子集图片路径列表
  data/val2017.txt                                          验证集列表（完整）
  data/manifest.csv                                         各子集汇总（图数/实例数/覆盖率）
  data/manifest_per_class.csv                               逐类别分布对比

用法
----
    E:/AIC/.venv/Scripts/python.exe scripts/make_subset.py
    E:/AIC/.venv/Scripts/python.exe scripts/make_subset.py --ratios 0.10 0.25 0.50 --seed 0
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
COCO_ROOT = DEFAULT_COCO_ROOT
TRAIN_ANN = COCO_ROOT / "annotations" / "instances_train2017.json"
VAL_ANN = COCO_ROOT / "annotations" / "instances_val2017.json"
DATA_DIR = PROJECT_ROOT / "data"          # yaml 与 manifest 的输出目录
LIST_DIR = COCO_ROOT                      # 图片列表 txt 必须放在 path 下：
                                          # yaml 里的相对路径是相对 `path` 解析的
                                          # （官方 coco.yaml 也把 train2017.txt 放在这）


# --------------------------------------------------------------------------
# 读取与统计
# --------------------------------------------------------------------------
def load_annotations(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到标注文件: {path}\n"
            f"  请先运行 scripts/download_coco.py 并解压 annotations_trainval2017.zip"
        )
    print(f"  读取 {path.name} …", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_annotations(ann: dict):
    """建立索引：图片信息、每图的实例类别、每类的图片与实例统计。"""
    # 注意：images 列表用 "id"，而 annotations 里用 "image_id" 指向它
    images = {im["id"]: im["file_name"] for im in ann["images"]}
    cat_names = {c["id"]: c["name"] for c in ann["categories"]}

    # 每张图的实例列表，以及每张图的主类别（实例数最多的类）
    per_image_cats = defaultdict(list)          # image_id -> [cat_id, ...]
    inst_per_cat = Counter()                    # cat_id -> 实例数
    for a in ann["annotations"]:
        if a.get("iscrowd", 0):
            # 保留 crowd 标注用于统计，但不参与主类别判定
            inst_per_cat[a["category_id"]] += 1
            continue
        per_image_cats[a["image_id"]].append(a["category_id"])
        inst_per_cat[a["category_id"]] += 1

    # 主类别 = 该图实例数最多的类别；并列时取 cat_id 较小者，保证确定性
    primary = {}
    for img_id, cats in per_image_cats.items():
        primary[img_id] = min(Counter(cats).items(),
                              key=lambda kv: (-kv[1], kv[0]))[0]

    # 没有任何非 crowd 标注的图片：归到一个哨兵层，抽样时按比例带上
    all_ids = set(images.keys())
    no_ann = sorted(all_ids - set(primary.keys()))

    return images, cat_names, primary, inst_per_cat, no_ann


def class_stats(image_ids, per_image_cats, cat_ids):
    """给定图片集合，统计各类别的图片数与实例数。"""
    img_cnt = Counter()
    inst_cnt = Counter()
    for img_id in image_ids:
        cats = per_image_cats.get(img_id, [])
        img_cnt.update(set(cats))
        inst_cnt.update(cats)
    return img_cnt, inst_cnt


# --------------------------------------------------------------------------
# 分层抽样
# --------------------------------------------------------------------------
def stratified_sample(primary: dict, no_ann: list, ratio: float, seed: int) -> list:
    """
    按主类别分层抽样。

    每个类别层内按比例抽取，且至少保留 1 张 —— 否则 10% 子集下
    「吹风机」这类长尾类别会整类消失，使消融结论失去意义。
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for img_id, cat in primary.items():
        by_class[cat].append(img_id)

    picked = []
    for cat in sorted(by_class):
        ids = sorted(by_class[cat])            # 排序保证不同机器结果一致
        n = max(1, round(len(ids) * ratio))
        picked.extend(rng.sample(ids, min(n, len(ids))))

    # 无标注图片按同样比例带上
    if no_ann:
        n = max(0, round(len(no_ann) * ratio))
        picked.extend(rng.sample(no_ann, min(n, len(no_ann))))

    return sorted(picked)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def write_txt(path: Path, image_ids, images, split: str):
    """写图片路径列表（Ultralytics 的 train/val 支持 txt 列表）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for img_id in image_ids:
            f.write(f"./images/{split}/{images[img_id]}\n")


def write_yaml(path: Path, train_txt: str, val_txt: str, cat_names: dict, note: str):
    """
    cat_names 是 {原始 category_id: 名称}（COCO 的 id 是 1..90 有跳号）。
    YAML 里必须写成 0..79 的稠密索引，否则 Ultralytics 会报
    "80-class dataset requires class indices 0-79"。
    索引顺序 = 按 category_id 排序，与转换脚本和官方 coco.yaml 一致。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# COCO2017 {note}",
        f"# 由 scripts/make_subset.py 生成，勿手工编辑",
        f"path: {COCO_ROOT.as_posix()}",
        f"train: {train_txt}",
        f"val: {val_txt}",
        "names:",
    ]
    for idx, cid in enumerate(sorted(cat_names)):
        lines.append(f"  {idx}: {cat_names[cid]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="COCO2017 分层子集生成")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.25, 0.50],
                    help="抽样比例，1.0 表示全集（tag=sub100）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coco-root", type=str, default=None,
                    help="COCO 数据目录（默认 项目/datasets/coco；集群上指向 jhaidata）")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="yaml/CSV 输出目录（默认 项目/data）")
    args = ap.parse_args()

    global COCO_ROOT, TRAIN_ANN, VAL_ANN, DATA_DIR, LIST_DIR
    if args.coco_root:
        COCO_ROOT = Path(args.coco_root).resolve()
        TRAIN_ANN = COCO_ROOT / "annotations" / "instances_train2017.json"
        VAL_ANN = COCO_ROOT / "annotations" / "instances_val2017.json"
        LIST_DIR = COCO_ROOT
    if args.out_dir:
        DATA_DIR = Path(args.out_dir).resolve()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("S1 · COCO2017 分层子集生成")
    print("=" * 68)

    train_ann = load_annotations(TRAIN_ANN)
    val_ann = load_annotations(VAL_ANN)

    images, cat_names, primary, inst_per_cat, no_ann = index_annotations(train_ann)
    val_images = {im["id"]: im["file_name"] for im in val_ann["images"]}
    val_per_image = defaultdict(list)
    for a in val_ann["annotations"]:
        val_per_image[a["image_id"]].append(a["category_id"])

    print(f"\n  全集: {len(images):,} 张训练图, {len(val_images):,} 张验证图, "
          f"{len(cat_names)} 个类别")
    print(f"        无标注图片 {len(no_ann)} 张")

    # 验证集列表（固定为完整 val2017，所有子集共用）
    val_txt = LIST_DIR / "val2017.txt"
    write_txt(val_txt, sorted(val_images), val_images, "val2017")
    print(f"  已写 {val_txt.name}: {len(val_images):,} 张")

    # 全集逐类别分布（后续所有子集都跟它对比）
    per_image_all = _per_image_all(train_ann)
    full_img_cnt, full_inst_cnt = class_stats(sorted(images), per_image_all, cat_names)

    per_class_rows = []
    summary_rows = []

    for ratio in args.ratios:
        tag = f"sub{int(round(ratio * 100))}"
        subset_ids = stratified_sample(primary, no_ann, ratio, args.seed)

        # 抽样后的真实分布
        img_cnt, inst_cnt = class_stats(subset_ids, per_image_all, cat_names)

        subset_images = {i: images[i] for i in subset_ids}
        train_txt = LIST_DIR / f"{tag}_train.txt"
        write_txt(train_txt, subset_ids, subset_images, "train2017")

        yaml_path = DATA_DIR / f"coco_{tag}.yaml"
        write_yaml(
            yaml_path,
            train_txt=train_txt.name,      # 相对 `path`（= datasets/coco）解析
            val_txt=val_txt.name,
            cat_names=cat_names,
            note=f"分层子集 {int(ratio*100)}% (seed={args.seed}, {len(subset_ids):,} 张训练图)",
        )

        n_inst = sum(inst_cnt.values())
        covered = sum(1 for c in cat_names if img_cnt.get(c, 0) > 0)
        missing = [cat_names[c] for c in cat_names if img_cnt.get(c, 0) == 0]

        summary_rows.append({
            "subset": tag,
            "yaml": yaml_path.name,
            "ratio": f"{ratio:.2f}",
            "seed": args.seed,
            "train_images": len(subset_ids),
            "train_instances": n_inst,
            "val_images": len(val_images),
            "categories_total": len(cat_names),
            "categories_covered": covered,
            "categories_missing": len(missing),
        })

        for cid in sorted(cat_names):
            per_class_rows.append({
                "subset": tag,
                "category_id": cid,
                "category": cat_names[cid],
                "full_images": full_img_cnt.get(cid, 0),
                "subset_images": img_cnt.get(cid, 0),
                "full_instances": full_inst_cnt.get(cid, 0),
                "subset_instances": inst_cnt.get(cid, 0),
                "image_ratio": (f"{img_cnt.get(cid,0)/full_img_cnt[cid]:.4f}"
                                if full_img_cnt.get(cid, 0) else ""),
            })

        print(f"\n  [{tag}] {len(subset_ids):,} 张训练图 / {n_inst:,} 实例 "
              f"| 类别覆盖 {covered}/{len(cat_names)}"
              + (f" | 缺失: {', '.join(missing[:5])}" if missing else ""))
        print(f"         -> {yaml_path.name}")

    # 写入 CSV
    manifest = DATA_DIR / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    per_class_csv = DATA_DIR / "manifest_per_class.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader()
        w.writerows(per_class_rows)

    print(f"\n  已写 {manifest}")
    print(f"  已写 {per_class_csv}")
    print("\nS1 子集生成完成。")


def _per_image_all(ann: dict) -> dict:
    """每张图的全部标注类别（含 crowd），用于分布统计。"""
    d = defaultdict(list)
    for a in ann["annotations"]:
        d[a["image_id"]].append(a["category_id"])
    return d


if __name__ == "__main__":
    main()
