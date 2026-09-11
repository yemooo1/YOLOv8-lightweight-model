# -*- coding: utf-8 -*-
"""
S1 · 解压 COCO2017 数据集
=========================

把 datasets/coco/ 下的三个 zip 解开。zip 内部已带顶层目录名
（train2017/、val2017/、annotations/），所以统一解到 datasets/coco/ 即可。

用法:
    E:/AIC/.venv/Scripts/python.exe scripts/extract_coco.py
    E:/AIC/.venv/Scripts/python.exe scripts/extract_coco.py --keep-zip   # 保留 zip（默认保留）
    E:/AIC/.venv/Scripts/python.exe scripts/extract_coco.py --delete-zip # 解压后删除 zip 省空间
"""

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COCO_ROOT = PROJECT_ROOT / "datasets" / "coco"
COCO_ROOT = DEFAULT_COCO_ROOT

# (zip 文件名, 解压目标子目录, 解压后应存在的目录, 预期条目数)
#
# 图片必须落在 images/ 下：Ultralytics 找标签文件的方式是把图片路径里的
# `images` 段替换成 `labels`（见 ultralytics/data/utils.py: img2label_paths）。
# 若图片直接放在 datasets/coco/train2017/，替换无从发生，训练会报找不到标签。
ASSETS = [
    ("annotations_trainval2017.zip", "", "annotations", 6),
    ("val2017.zip", "images", "images/val2017", 5001),
    ("train2017.zip", "images", "images/train2017", 118288),
]


def extract_one(zip_name: str, sub_dir: str, check_dir: str, expected: int) -> bool:
    zp = COCO_ROOT / zip_name
    dest = COCO_ROOT / sub_dir if sub_dir else COCO_ROOT
    target = COCO_ROOT / check_dir

    if not zp.exists():
        print(f"  ⚠ 找不到 {zip_name}，跳过")
        return False

    if target.exists() and any(target.iterdir()):
        n = sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"  {check_dir}/ 已存在（{n:,} 个文件），跳过")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  解压 {zip_name} → {sub_dir or '.'}/ …", flush=True)
    t0 = time.time()

    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        if len(names) != expected:
            print(f"    ⚠ 条目数 {len(names)} 与预期 {expected} 不符，仍继续解压")
        z.extractall(dest)

    el = time.time() - t0
    n = sum(1 for _ in target.rglob("*") if _.is_file())
    print(f"    完成: {n:,} 个文件, 用时 {el:.0f}s ({n/el:.0f} 文件/秒)")
    return True


def main():
    ap = argparse.ArgumentParser(description="解压 COCO2017")
    ap.add_argument("--delete-zip", action="store_true", help="解压后删除 zip 文件")
    ap.add_argument("--coco-root", type=str, default=None,
                    help="COCO 数据目录（默认 项目/datasets/coco；集群上指向 jhaidata）")
    args = ap.parse_args()

    global COCO_ROOT
    if args.coco_root:
        COCO_ROOT = Path(args.coco_root).resolve()

    print("=" * 68)
    print("S1 · 解压 COCO2017")
    print("=" * 68)
    free = shutil.disk_usage(COCO_ROOT).free / 1024**3 if COCO_ROOT.exists() else 0
    print(f"目标目录: {COCO_ROOT}")
    print(f"可用空间: {free:.1f} GB\n")

    for zip_name, sub_dir, check_dir, expected in ASSETS:
        extract_one(zip_name, sub_dir, check_dir, expected)
        print()

    # 结果核对
    print("=" * 68)
    print("解压结果核对")
    print("=" * 68)
    checks = [
        ("images/train2017", COCO_ROOT / "images" / "train2017"),
        ("images/val2017", COCO_ROOT / "images" / "val2017"),
        ("annotations/instances_train2017.json",
         COCO_ROOT / "annotations" / "instances_train2017.json"),
        ("annotations/instances_val2017.json",
         COCO_ROOT / "annotations" / "instances_val2017.json"),
    ]
    ok = True
    for label, p in checks:
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {label:42s} {n:>7,} 个文件 ✓")
        elif p.is_file():
            print(f"  {label:42s} {p.stat().st_size/1024**2:>7.1f} MB ✓")
        else:
            print(f"  {label:42s} 缺失 ✗")
            ok = False

    if args.delete_zip:
        print()
        for zip_name, *_ in ASSETS:
            zp = COCO_ROOT / zip_name
            if zp.exists():
                zp.unlink()
                print(f"  已删除 {zip_name}")

    print()
    print("解压完成。" if ok else "解压存在缺失，请检查上面的 ✗ 项。")


if __name__ == "__main__":
    main()
