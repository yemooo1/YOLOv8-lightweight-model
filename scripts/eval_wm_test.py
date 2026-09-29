# -*- coding: utf-8 -*-
"""西瓜：在独立 test 集上同条件评估 教师/蒸馏学生best/蒸馏学生last/n基线。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = str(ROOT / "watermelon_local.yaml")
WEIGHTS = {
    "teacher_yolov8s":      ROOT / "runs/watermelon/watermelon_yolov8s_ep100/weights/best.pt",
    "student_distill_best": ROOT / "runs/watermelon/distill_s2n_yolov8n_ep100/weights/best.pt",
    "student_distill_last": ROOT / "runs/watermelon/distill_s2n_yolov8n_ep100/weights/last.pt",
    "baseline_yolov8n":     ROOT / "runs/watermelon/watermelon_yolov8n_ep100/weights/best.pt",
}


def main():
    from ultralytics import YOLO

    results = {}
    for tag, w in WEIGHTS.items():
        if not w.exists():
            print(f"SKIP {tag}: {w} missing")
            continue
        print(f"\n===== {tag} =====", flush=True)
        m = YOLO(str(w))
        r = m.val(data=DATA, split="test", imgsz=640, batch=16, device=0,
                  workers=0, plots=False, verbose=False, exist_ok=True,
                  project=str(ROOT / "runs/watermelon/eval_test"), name=tag)
        per = {}
        for i, c in r.names.items():
            per[c] = {
                "P": round(float(r.box.p[i]), 4),
                "R": round(float(r.box.r[i]), 4),
                "mAP50": round(float(r.box.ap50[i]), 4),
                "mAP50-95": round(float(r.box.ap[i]), 4),
            }
        results[tag] = {
            "P": round(float(r.box.mp), 4),
            "R": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
            "per_class": per,
            "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 3),
        }

    out = ROOT / "runs/watermelon/eval_test/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== TEST-SET SUMMARY =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
