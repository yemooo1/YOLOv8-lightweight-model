# -*- coding: utf-8 -*-
"""
阶段② · 教师输出离线缓存（方案 4.3.2）
======================================

为什么需要
----------
1080 Ti（Pascal, sm_61, amp=False）上在线跑 YOLOv8-L 教师约有 7.3x 墙钟开销；
因此在 L40S/L20（或 4060 fp16）上用 ``scripts/cache_teacher.py`` 按 epoch 预计算
教师输出（含增强后的图像本身，保证师生严格对齐），本地训练时直接读缓存。

缓存布局
--------
``<cache_dir>/meta.json``             全局元信息（通道、reg_max、增强签名……）
``<cache_dir>/epoch000/shard_000000.pt``   每个 optimizer step 一个分片
``<cache_dir>/epoch000/shard_000001.pt``
...

每个分片（torch.save 的 dict）：
    {
      "batch":   {"img": uint8 (B,3,H,W), "cls", "bboxes", "batch_idx", "im_file"},
      TEACHER_KEY: {"scores": fp16 (B,nc,A), "boxes": fp16 (B,4*reg_max,A),
                    "feats": [fp16 (B,Ct_i,H_i,W_i) x3]}
    }

一致性保证（关键）
------------------
* 图像以「增强后」的形态写入分片，师生看到的是同一张图，不依赖随机种子回放；
* 缓存模式下必须关闭 multi_scale（trainer 启动时强制），所有分片分辨率一致；
* 每个 epoch 的增强在缓存生成时已烘焙（含 close_mosaic 最后 10 轮关 Mosaic）；
* 学生侧只是一个按顺序读取分片的普通 Dataset，DataLoader 的 shuffle/增强全部失效，
  数据顺序即缓存生成时的顺序。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# batch dict 中携带教师输出的键名（下划线前缀避免与 ultralytics 的 batch 键冲突）
TEACHER_KEY = "__teacher_cache__"

CACHE_DTYPE = torch.float16  # 教师输出落盘精度（fp16 体积减半，蒸馏数值精度足够）


def cache_meta_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "meta.json"


def cache_epoch_dir(cache_dir: str | Path, epoch: int) -> Path:
    return Path(cache_dir) / f"epoch{epoch:03d}"


def save_cache_meta(cache_dir: str | Path, meta: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_meta_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_cache_meta(cache_dir: str | Path) -> dict:
    with open(cache_meta_path(cache_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def teacher_outputs_to_cpu(teacher_preds: dict, dtype: torch.dtype = CACHE_DTYPE) -> dict:
    """把教师一个 batch 的输出搬到 CPU + fp16，供落盘。"""
    return {
        "scores": teacher_preds["scores"].detach().to("cpu", dtype),
        "boxes": teacher_preds["boxes"].detach().to("cpu", dtype),
        "feats": [f.detach().to("cpu", dtype) for f in teacher_preds["feats"]],
    }


def batch_to_cpu_uint8(batch: dict) -> dict:
    """保留训练损失需要的最小 batch 字段，图像转 uint8（数据集侧本来就是 uint8）。"""
    img = batch["img"]
    if img.dtype != torch.uint8:
        # 防御：万一上游已归一化为 float（[0,1]），还原回 uint8
        img = (img * 255.0).round().clamp_(0, 255).to(torch.uint8)
    out = {
        "img": img.detach().to("cpu"),
        "cls": batch["cls"].detach().to("cpu"),
        "bboxes": batch["bboxes"].detach().to("cpu"),
        "batch_idx": batch["batch_idx"].detach().to("cpu"),
        "im_file": list(batch.get("im_file", [])),
    }
    return out


def write_cache_shard(cache_dir: str | Path, epoch: int, step: int, batch: dict, teacher_preds: dict) -> Path:
    epoch_dir = cache_epoch_dir(cache_dir, epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    path = epoch_dir / f"shard_{step:06d}.pt"
    torch.save({"batch": batch_to_cpu_uint8(batch), TEACHER_KEY: teacher_outputs_to_cpu(teacher_preds)}, path)
    return path


class DistillCacheDataset(Dataset):
    """按 epoch 目录顺序读取教师缓存分片。__getitem__ 直接返回「一个完整 batch」。"""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.meta = load_cache_meta(self.cache_dir)
        self.epoch = 0
        self._files: list[Path] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        epoch_dir = cache_epoch_dir(self.cache_dir, epoch)
        if not epoch_dir.is_dir():
            raise FileNotFoundError(
                f"找不到蒸馏缓存目录 {epoch_dir}。\n"
                f"请先在 GPU 机器上运行 scripts/cache_teacher.py 生成第 {epoch} 轮缓存，"
                f"或改用 --teacher-mode online。"
            )
        files = sorted(epoch_dir.glob("shard_*.pt"))
        if not files:
            raise FileNotFoundError(f"{epoch_dir} 下没有 shard_*.pt 分片文件。")
        if self._files and len(files) != len(self._files):
            raise RuntimeError(
                f"缓存分片数量在 epoch 间不一致：{len(self._files)} vs {len(files)}（{epoch_dir}），"
                f"说明不同 epoch 用了不同 batch 配置，请重新生成缓存。"
            )
        self.epoch = epoch
        self._files = files

    def close_mosaic(self, hyp=None) -> None:  # trainer 在最后若干轮会回调，缓存模式无需处理
        return

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> dict:
        # DataLoader(batch_size=None) 不做自动 batching/collate，直接返回本分片。
        # 把教师输出挂到 batch 上，展平成 trainer 期望的单 dict 结构。
        shard = torch.load(self._files[index], map_location="cpu", weights_only=False)
        batch = shard["batch"]
        batch[TEACHER_KEY] = shard[TEACHER_KEY]
        return batch


class _CacheSampler:
    """仅提供 __len__ 的占位 sampler，供 trainer 计算 final_batch_size。"""

    def __init__(self, num_samples: int):
        self._n = num_samples

    def __len__(self) -> int:
        return self._n


class DistillCacheLoader:
    """鸭子类型对齐 ultralytics.InfiniteDataLoader 的最小接口。

    trainer 每个 epoch 会重新 enumerate(loader)，并调用 loader.reset()、读取
    loader.num_workers / loader.dataset / loader.sampler / loader.batch_size /
    loader.drop_last，这里逐一提供。
    """

    def __init__(self, dataset: DistillCacheDataset, num_workers: int = 2, batch_size: int | None = None):
        self.dataset = dataset
        self.num_workers = num_workers
        # 分片在生成缓存时已固定 batch 大小，训练侧必须与其一致
        self.batch_size = int(batch_size or dataset.meta.get("batch") or 1)
        self.drop_last = False
        # sampler 只用于 trainer 的 final_batch_size 检查：len(sampler) % batch_size
        self.sampler = _CacheSampler(len(dataset) * self.batch_size)
        # batch_size=None → DataLoader 不做二次 batching，逐条吐出已组好 batch 的分片
        self._loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=False,
        )

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        yield from self._loader

    def reset(self) -> None:
        """兼容 InfiniteDataLoader.reset()（每个 epoch 重新 enumerate 时自动重建迭代器）。"""
        return
