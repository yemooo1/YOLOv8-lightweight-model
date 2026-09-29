# -*- coding: utf-8 -*-
"""生成「同图检测效果对比」图：教师 / 基线 / 蒸馏 / RepC2f 部署权重。

测试图从 ``cluster_upload/watermelon.zip`` 内 ``images/test/`` 抽取（优先选含
unripe 类别的样本），临时落到系统临时目录，不写入仓库。

用法：``python scripts/plot_detections.py``  → 输出 ``figures/detect_compare.png``
"""

import pathlib
import sys
import tempfile
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # RepC2f 权重反序列化需要 src.reparam

from ultralytics import YOLO  # noqa: E402

ZIP = ROOT / "cluster_upload" / "watermelon.zip"
N_IMG = 3

MODELS = [
    ("教师 yolov8s", "runs/watermelon/watermelon_yolov8s_ep100/weights/best.pt"),
    ("基线 yolov8n", "runs/watermelon/watermelon_yolov8n_ep100/weights/best.pt"),
    ("蒸馏学生", "runs/watermelon/distill_s2n_yolov8n_ep100/weights/best.pt"),
    ("RepC2f 部署", "runs/detect/runs/watermelon/distill_rep_s2n_yolov8n_ep100/weights/best_deploy.pt"),
]


def pick_test_images():
    """从 zip 里挑若干测试图：优先含 unripe（class 1）标签的样本。"""
    with zipfile.ZipFile(ZIP) as z:
        names = [n.replace("\\", "/") for n in z.namelist()]
        imgs = sorted(n for n in names if n.startswith("images/test/") and n.lower().endswith(".jpg"))
        picked = []
        for img in imgs:
            stem = pathlib.PurePosixPath(img).stem
            lbl = f"labels/test/{stem}.txt"
            if lbl not in names:
                continue
            content = z.read(lbl).decode("utf-8", "ignore")
            classes = {ln.split()[0] for ln in content.strip().splitlines() if ln.strip()}
            if "1" in classes:  # unripe 少数类
                picked.append(img)
            if len(picked) >= N_IMG:
                break
        picked = picked or imgs[:N_IMG]

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="wm_test_"))
        paths = []
        for name in picked:
            out = tmp / pathlib.PurePosixPath(name).name
            out.write_bytes(z.read(name))
            paths.append(out)
    return paths


def main():
    images = pick_test_images()
    print("测试图：", [p.name for p in images])

    fig, axes = plt.subplots(len(images), len(MODELS), figsize=(4.0 * len(MODELS), 3.2 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)

    for col, (title, rel) in enumerate(MODELS):
        weight = ROOT / rel
        print(f"加载 {title}: {rel}")
        model = YOLO(str(weight))
        for row, img in enumerate(images):
            res = model.predict(str(img), imgsz=640, conf=0.25, verbose=False)[0]
            axes[row][col].imshow(res.plot()[:, :, ::-1])
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(title, fontsize=13)

    fig.suptitle("同图检测效果对比（西瓜 test 集样本，conf=0.25）", fontsize=15)
    fig.tight_layout()
    out = ROOT / "figures" / "detect_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已输出 {out}")


if __name__ == "__main__":
    main()