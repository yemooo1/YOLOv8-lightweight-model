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
# 默认数据目录在项目内；可用 --data-dir 覆盖（学校集群上指向 jhaidata 数据区）
DATA_DIR = PROJECT_ROOT / "datasets" / "coco"
PARTS_DIR = DATA_DIR / ".parts"

# 两个源，文件内容与大小完全一致（已核对 Content-Length）：
#   official —— images.cocodataset.org，国内单线程约 200 KB/s，且传输中途频繁断连
#   hf       —— HuggingFace 国内镜像，实测单线程 7 MB/s，稳定
OFFICIAL = "http://images.cocodataset.org"
HF_MIRROR = "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/main"


def build_assets(source: str):
    if source == "hf":
        return [
            (f"{HF_MIRROR}/train2017.zip", "train2017.zip"),
            (f"{HF_MIRROR}/val2017.zip", "val2017.zip"),
            (f"{HF_MIRROR}/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
        ]
    return [
        (f"{OFFICIAL}/zips/train2017.zip", "train2017.zip"),
        (f"{OFFICIAL}/zips/val2017.zip", "val2017.zip"),
        (f"{OFFICIAL}/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
    ]

CHUNK = 16 * 1024 * 1024  # 每个分片 16 MB


def acquire_lock(lock_path: Path):
    """
    单实例锁（Windows / Linux 双平台）。

    必须的：两个下载进程同时往同一个分片文件追加写（open 'ab'）会交错
    写入，产生「大小正确但 CRC 错误」的损坏文件 —— 这个坑已经踩过一次，
    当时坏了 train2017 和 val2017 两个文件。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except ImportError:
        pass
    try:
        import msvcrt
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("✗ 已有另一个 download_coco.py 在运行。")
        print("  同时运行多个下载进程会写坏分片文件，请先等待它结束，")
        print("  或用以下命令清理残留进程：")
        print("  powershell -Command \"Get-CimInstance Win32_Process -Filter \\\"Name='python.exe'\\\""
              " | Where-Object { $_.CommandLine -like '*download_coco*' }"
              " | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }\"")
        sys.exit(1)
    return fh


def verify_zip(path: Path) -> str | None:
    """CRC 校验整包，返回第一个损坏的成员名；完好则返回 None。"""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip()
    except Exception as exc:
        return f"<无法读取: {exc}>"


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_part(url: str, part_path: Path, start: int, end: int, retries: int = 6) -> int:
    """
    下载一个分片，支持两级断点续传：

      1) 分片已完整        → 直接跳过
      2) 分片只下了一半    → 用 Range 从断点「续写」，而不是从头重下

    第 2 点是必需的。官方源会在传输途中掐断连接；若每次重试都从头开始，
    大分片就会陷入「下到几 MB 就断 → 重新从头下 → 又断」的死循环，
    永远到不了 100%。实测有 5 个分片卡在这个循环里。
    """
    expected = end - start + 1

    for attempt in range(retries):
        have = part_path.stat().st_size if part_path.exists() else 0
        if have == expected:
            return expected
        if have > expected:          # 异常：文件比预期大，丢弃重下
            part_path.unlink()
            have = 0

        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start + have}-{end}"})
            # 超时设短：连接卡死时快速失败进入下一次重试，
            # 否则进程会一直等一个永不返回的连接
            with urllib.request.urlopen(req, timeout=45) as r, open(part_path, "ab") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))

    return part_path.stat().st_size if part_path.exists() else 0


def download(url: str, name: str, threads: int,
             merge_only: bool = False, force: bool = False,
             verify: bool = True) -> Path:
    dest = DATA_DIR / name

    if dest.exists() and dest.stat().st_size > 0 and not force:
        if verify:
            print(f"  {name} 已存在，正在做 CRC 校验 …", flush=True)
            bad = verify_zip(dest)
            if bad is None:
                print(f"  {name} ✓ CRC 校验通过（{dest.stat().st_size/1024**3:.2f} GB），跳过")
                return dest
            print(f"  ✗ {name} CRC 校验失败于 {bad}")
            print(f"    删除损坏文件并重新下载（这类损坏源于多进程并发写同一分片）")
            dest.unlink()
            shutil.rmtree(PARTS_DIR / name, ignore_errors=True)
        else:
            print(f"  {name} 已存在（{dest.stat().st_size/1024**3:.2f} GB），跳过校验")
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

    failed = 0
    if merge_only:
        bad = [i for i, s, e in ranges
               if not (pdir / f"part{i:05d}").exists()
               or (pdir / f"part{i:05d}").stat().st_size != (e - s + 1)]
        if bad:
            raise RuntimeError(
                f"--merge-only 模式下仍有 {len(bad)} 个分片不完整：{bad[:10]}…\n"
                f"  请去掉 --merge-only 正常下载以补齐。"
            )
        print(f"  全部 {n_parts} 个分片完整，跳过下载直接合并")
    else:
        t0 = time.time()
        done = 0
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
                    # 注意：done 计的是「本次提交的任务数」，包含一开始就
                    # 判定已完整而立即返回的分片，所以这里的速率会虚高
                    got = sum(p.stat().st_size for p in pdir.glob("part*"))
                    print(f"    {done}/{n_parts} 任务完成  "
                          f"磁盘已有 {got/total*100:5.1f}%", flush=True)

        if failed:
            print(f"  ⚠ {failed} 个分片失败，请重跑本脚本补齐（会自动跳过已完成的分片）")

    # 合并分片
    print(f"  合并分片 -> {name} …", flush=True)
    with open(dest, "wb") as out:
        for i in range(n_parts):
            with open(pdir / f"part{i:05d}", "rb") as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
    shutil.rmtree(pdir)
    print(f"  完成: {dest}  ({dest.stat().st_size/1024**3:.2f} GB)")

    if verify:
        print(f"  正在做 CRC 校验（整包逐成员）…", flush=True)
        bad = verify_zip(dest)
        if bad is None:
            print(f"  {name} ✓ CRC 校验通过")
        else:
            print(f"  ✗ {name} CRC 校验失败于 {bad}")
            print(f"    文件已保留供排查；重下请加 --force")
            raise RuntimeError(f"{name} CRC 校验失败")

    return dest


def main():
    ap = argparse.ArgumentParser(description="COCO2017 多线程下载器")
    ap.add_argument("--threads", type=int, default=32, help="并发线程数（默认 32）")
    ap.add_argument("--only", type=str, default=None, help="只下载指定文件，如 val2017.zip")
    ap.add_argument("--merge-only", action="store_true",
                    help="跳过分片下载，直接用已有分片合并（分片已齐时用）")
    ap.add_argument("--source", choices=["official", "hf"], default="official",
                    help="下载源：official=COCO 官网（慢/易断），hf=HuggingFace 国内镜像（快）")
    ap.add_argument("--force", action="store_true",
                    help="即使文件已存在也重新下载（用于修复 CRC 损坏）")
    ap.add_argument("--no-verify", action="store_true", help="跳过 CRC 校验（快，但不安全）")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="数据存放目录（默认 项目/datasets/coco；集群上指向 jhaidata）")
    args = ap.parse_args()

    global DATA_DIR, PARTS_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
        PARTS_DIR = DATA_DIR / ".parts"

    lock = acquire_lock(PARTS_DIR / ".download.lock")   # noqa: F841 持有到进程结束
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据目录: {DATA_DIR}")
    print(f"可用空间: {shutil.disk_usage(DATA_DIR).free/1024**3:.1f} GB\n")

    for url, name in build_assets(args.source):
        if args.only and args.only != name:
            continue
        print(f"[{name}]")
        download(url, name, args.threads, merge_only=args.merge_only,
                 force=args.force, verify=not args.no_verify)
        print()

    print("下载阶段结束。")


if __name__ == "__main__":
    main()
