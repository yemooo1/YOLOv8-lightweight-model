# -*- coding: utf-8 -*-
"""西瓜数据集蒸馏前置：训练 yolov8s 教师（nc=2），配方与西瓜 n 基线完全一致。
存在 last.pt 时自动断点续训。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from ultralytics import YOLO

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

    last = ROOT / "runs" / "watermelon" / "watermelon_yolov8s_ep100" / "weights" / "last.pt"
    if last.exists():
        print(f"RESUME from {last}", flush=True)
        model = YOLO(str(last))
        model.train(resume=True)
        print("TEACHER_DONE", flush=True)
        return

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(ROOT / "watermelon_local.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=15,
        workers=2,  # Windows 页面文件有限：worker 过多会 WinError 1455
        seed=0,
        project=str(ROOT / "runs" / "watermelon"),
        name="watermelon_yolov8s_ep100",
        exist_ok=True,
        plots=True,
    )
    print("TEACHER_DONE", flush=True)


if __name__ == "__main__":
    main()
