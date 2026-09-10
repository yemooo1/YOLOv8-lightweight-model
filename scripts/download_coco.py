# -*- coding: utf-8 -*-
"""
S1 前置 · COCO2017 数据集下载器
================================

官方源单线程实测约 200 KB/s（20 GB 需约 27 小时），本脚本用多线程分片
把速度提到 7 MB/s 左右（约 1 小时）。支持断点续传：每个分片单独落盘，
中断后重跑只补缺失的分片。

用法:
    E:/AIC/.venv/Scripts/python.exe scripts/download_coco.py
    E:/AIC/.venv/Scripts/python.exe scripts/download_coco.py --only val2017.zip
"""

import argparse
import concurrent.futures
import shutil
import sys
import time
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "coco"
PARTS_DIR = DATA_DIR / ".parts"

# 官方源，与 Ultralytics coco.yaml 的拉取来源一致
BASE = "http://images.cocodataset.org"
ASSETS = [
    (f"{BASE}/zips/train2017.zip", "train2017.zip"),
    (f"{BASE}/zips/val2017.zip", "val2017.zip"),
    (f"{BASE}/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
]

CHUNK = 16 * 1024 * 1024  # 每个分片 16 MB


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_part(url: str, part_path: Path, start: int, end: int, retries: int = 3) -> int:
    """下载一个分片；若已完整则跳过（断点续传的关键）。"""
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return expected

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r, open(part_path, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            if part_path.stat().st_size == expected:
                return expected
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


def download(url: str, name: str, threads: int) -> Path:
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {name} 已存在（{dest.stat().st_size/1024**3:.2f} GB），跳过")
        return dest

    total = head_size(url)
    n_parts = (total + CHUNK - 1) // CHUNK
    print(f"  {name}: {total/1024**3:.2f} GB，{n_parts} 个分片，{threads} 线程")

    pdir = PARTS_DIR / name
    pdir.mkdir(parents=True, exist_ok=True)

    ranges = []
    for i in range(n_parts):
        s = i * CHUNK
        ranges.append((i, s, min(s + CHUNK - 1, total - 1)))

    t0 = time.time()
    done = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(threads) as ex:
        futures = {
            ex.submit(fetch_part, url, pdir / f"part{i:05d}", s, e): i
            for i, s, e in ranges
        }
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as exc:
                failed += 1
                print(f"    分片失败: {exc}", flush=True)
            if done % 20 == 0 or done == n_parts:
                el = time.time() - t0
                got = sum(p.stat().st_size for p in pdir.glob("part*"))
                sp = got / el / 1024 / 1024 if el > 0 else 0
                eta = (total - got) / (got / el) if got else 0
                print(f"    {got/total*100:5.1f}%  {done}/{n_parts} 分片  "
                      f"{sp:5.1f} MB/s  剩余约 {eta/60:.0f} 分钟", flush=True)

    if failed:
        print(f"  ⚠ {failed} 个分片失败，请重跑本脚本补齐")

    # 合并分片
    print(f"  合并分片 -> {name} …", flush=True)
    with open(dest, "wb") as out:
        for i in range(n_parts):
            with open(pdir / f"part{i:05d}", "rb") as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
    shutil.rmtree(pdir)
    print(f"  完成: {dest}  ({dest.stat().st_size/1024**3:.2f} GB)")
    return dest


def main():
    ap = argparse.ArgumentParser(description="COCO2017 多线程下载器")
    ap.add_argument("--threads", type=int, default=32, help="并发线程数（默认 32）")
    ap.add_argument("--only", type=str, default=None, help="只下载指定文件，如 val2017.zip")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据目录: {DATA_DIR}")
    print(f"可用空间: {shutil.disk_usage(DATA_DIR).free/1024**3:.1f} GB\n")

    for url, name in ASSETS:
        if args.only and args.only != name:
            continue
        print(f"[{name}]")
        download(url, name, args.threads)
        print()

    print("下载阶段结束。")


if __name__ == "__main__":
    main()
