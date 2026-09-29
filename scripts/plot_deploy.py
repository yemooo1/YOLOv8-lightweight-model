# -*- coding: utf-8 -*-
"""生成参赛材料用图：西瓜集 test 指标对比 + 部署前后（PyTorch / TensorRT）对比。

数据来源
--------
* 教师 / 基线 / 蒸馏：``runs/watermelon/eval_test/summary.json``（独立 test 集 1041 张）
* RepC2f 部署权重：融合后 ``best_deploy.pt`` 在 test 集上的实测
* 延迟 / 体积：集群 L40S 单卡实测（imgsz=640，不含前后处理）

用法：``python scripts/plot_deploy.py``  → 输出 ``figures/deploy_summary.png``
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUM = json.loads((ROOT / "runs/watermelon/eval_test/summary.json").read_text(encoding="utf-8"))

# RepC2f 融合部署权重实测（PyTorch FP32 / TensorRT FP16，同一 test 集）
PT = {"mAP50": 0.633, "mAP50_95": 0.430, "latency": 0.69, "size": 11.7}
TRT = {"mAP50": 0.631, "mAP50_95": 0.421, "latency": 0.45, "size": 8.3}

models = [
    ("教师\nyolov8s\n11.14M", SUM["teacher_yolov8s"]["mAP50"], SUM["teacher_yolov8s"]["mAP50-95"]),
    ("基线\nyolov8n\n3.01M", SUM["baseline_yolov8n"]["mAP50"], SUM["baseline_yolov8n"]["mAP50-95"]),
    ("蒸馏学生\n3.01M", SUM["student_distill_best"]["mAP50"], SUM["student_distill_best"]["mAP50-95"]),
    ("RepC2f\n部署权重 3.006M", PT["mAP50"], PT["mAP50_95"]),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- 左图：四组模型精度对比（独立 test 集） ----
labels = [m[0] for m in models]
map50 = [m[1] for m in models]
map5095 = [m[2] for m in models]
x = range(len(models))
w = 0.36
b1 = ax1.bar([i - w / 2 for i in x], map50, w, label="mAP@0.5", color="#2E6FB7")
b2 = ax1.bar([i + w / 2 for i in x], map5095, w, label="mAP@0.5:0.95", color="#8FB8DE")
ax1.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
ax1.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, 0.82)
ax1.set_ylabel("精度（test split，1041 张）")
ax1.set_title("四组模型精度对比（西瓜成熟度，2 类）", fontsize=12)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(axis="y", alpha=0.25)

# ---- 右图：部署前后延迟对比 ----
backends = ["PyTorch\nFP32", "TensorRT\nFP16"]
lat = [PT["latency"], TRT["latency"]]
bars = ax2.bar(backends, lat, 0.45, color=["#B7A98F", "#2E6FB7"])
ax2.bar_label(bars, fmt="%.2f ms", fontsize=10, padding=2)
ax2.set_ylim(0, max(lat) * 1.95)
ax2.set_ylabel("推理耗时（ms/图，640×640）")
ax2.set_title("部署前后推理延迟（L40S，不含前后处理）", fontsize=12)
ax2.grid(axis="y", alpha=0.25)
ax2.annotate(
    f"加速比 {PT['latency'] / TRT['latency']:.2f}×\n"
    f"mAP@0.5:0.95  {PT['mAP50_95']:.3f} → {TRT['mAP50_95']:.3f}（−{PT['mAP50_95'] - TRT['mAP50_95']:.2f} 点）\n"
    f"体积  {PT['size']} MB → {TRT['size']} MB（−29%）",
    xy=(0.5, 0.99), xycoords="axes fraction", ha="center", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F2F6FB", edgecolor="#2E6FB7", alpha=0.9),
)

fig.tight_layout()
out = ROOT / "figures" / "deploy_summary.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"已输出 {out}")