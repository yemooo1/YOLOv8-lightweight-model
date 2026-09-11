#!/bin/bash
# 统一评测作业：在完整 val2017 上复测权重，输出 mAP@0.5 / mAP@0.5:0.95
# 提交方式：bash val_model.sh <run目录名>
#   例：bash val_model.sh baseline/n_sub10_ep100
source "$(dirname "$0")/common.sh"

RUN_SUBDIR="${1:?用法: bash val_model.sh <baseline/xxx 运行目录名>}"
WEIGHTS="$RUNS_DIR/$RUN_SUBDIR/weights/best.pt"
TAG="${2:-sub100}"   # 默认在全集 val 上评（val 始终是完整 val2017，与训练子集无关）

if [ ! -f "$WEIGHTS" ]; then
    echo "✗ 找不到权重: $WEIGHTS"
    echo "  现有运行目录："
    find "$RUNS_DIR" -maxdepth 3 -name best.pt 2>/dev/null
    exit 1
fi

install_python_deps
cd "$PROJECT_DIR"
echo "========== 评测 $WEIGHTS =========="
yolo detect val \
    model="$WEIGHTS" \
    data="$PROJECT_DIR/data/coco_${TAG}.yaml" \
    imgsz=640 \
    batch=32 \
    device=0 \
    amp=True \
    workers=8 \
    project="$RUNS_DIR/eval" \
    name="$(basename "$RUN_SUBDIR")_val" \
    exist_ok=True
echo "========== 评测完成 =========="
