# -*- coding: utf-8 -*-
"""西瓜推理 demo：在 test 集挑含 unripe 的图，用蒸馏学生 best/last 两权重推理并拼图对比。"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_DIR = ROOT / "datasets/watermelon/images/test"
LBL_DIR = ROOT / "datasets/watermelon/labels/test"
OUT_DIR = ROOT / "runs/watermelon/demo"

BEST = ROOT / "runs/watermelon/distill_s2n_yolov8n_ep100/weights/best.pt"
LAST = ROOT / "runs/watermelon/distill_s2n_yolov8n_ep100/weights/last.pt"

N_UNRIPE = 8   # 含 unripe 目标的图
N_RIPE = 4     # 仅 ripe 的图
SEED = 0


def pick_images():
    unripe_imgs, ripe_imgs = [], []
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
    for lbl in LBL_DIR.glob("*.txt"):
        classes = {line.split()[0] for line in lbl.read_text().splitlines() if line.strip()}
        img = next((p for ext in exts for p in [IMG_DIR / f"{lbl.stem}{ext}"] if p.exists()), None)
        if img is None:
            continue
        if "1" in classes:
            unripe_imgs.append(img)
        elif classes == {"0"}:
            ripe_imgs.append(img)
    rng = random.Random(SEED)
    rng.shuffle(unripe_imgs)
    rng.shuffle(ripe_imgs)
    picked = unripe_imgs[:N_UNRIPE] + ripe_imgs[:N_RIPE]
    print(f"candidate unripe={len(unripe_imgs)}, ripe-only={len(ripe_imgs)}, picked={len(picked)}", flush=True)
    return picked


def main():
    from PIL import Image
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = pick_images()
    models = [("best", "best(high-recall)", YOLO(str(BEST))),
              ("last", "last(high-precision)", YOLO(str(LAST)))]

    rows = []
    for i, img in enumerate(imgs):
        panels = []
        for key, _tag, m in models:
            r = m.predict(source=str(img), conf=0.25, imgsz=640, device=0, verbose=False)[0]
            plot = r.plot(line_width=2, font_size=12)  # BGR ndarray
            pil = Image.fromarray(plot[:, :, ::-1])
            panels.append(pil)
            sub = OUT_DIR / key
            sub.mkdir(exist_ok=True)
            pil.save(sub / f"{i:02d}_{img.stem[:20]}.jpg", quality=88)
        rows.append(panels)

    # 统一行高，每行横向拼 best | last
    row_h = min(min(p.height for p in row) for row in rows)
    laid = []
    for row in rows:
        rs = [p.resize((int(p.width * row_h / p.height), row_h)) for p in row]
        c = Image.new("RGB", (rs[0].width + rs[1].width + 10, row_h), (255, 255, 255))
        c.paste(rs[0], (0, 0))
        c.paste(rs[1], (rs[0].width + 10, 0))
        laid.append(c)

    W = max(c.width for c in laid)
    H = sum(c.height for c in laid) + 10 * (len(laid) - 1)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for c in laid:
        sheet.paste(c, (0, y))
        y += c.height + 10
    out = OUT_DIR / "comparison_best_vs_last.jpg"
    sheet.save(out, quality=85)
    print(f"DEMO_DONE -> {out}  ({len(imgs)} pairs, left=best/high-recall right=last/high-precision)", flush=True)


if __name__ == "__main__":
    main()
