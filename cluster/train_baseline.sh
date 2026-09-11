#!/bin/bash
# 基线训练作业（成员A 的 8 组"数据量-mAP"实验）
# ------------------------------------------------------------
# 提交方式：工程目录选 jhupload，启动命令：
#     bash train_baseline.sh n  sub10          # yolov8n + 10% 子集
#     bash train_baseline.sh n  sub25
#     bash train_baseline.sh n  sub50
#     bash train_baseline.sh n  sub100         # 全量 COCO
#     bash train_baseline.sh s  sub10          # yolov8s + 10% 子集
#     bash train_baseline.sh s  sub25
#     bash train_baseline.sh s  sub50
#     bash train_baseline.sh s  sub100
# 可选第 3/4 个参数覆盖轮数与 batch：
#     bash train_baseline.sh n sub10 3 16      # 冒烟：3 轮
# 资源：L40S/L20 单卡独占；Ada 架构必须 amp=True（与 1080Ti 相反）
source "$(dirname "$0")/common.sh"

SIZE="${1:?用法: bash train_baseline.sh <n|s> <sub10|sub25|sub50|sub100> [epochs] [batch]}"
TAG="${2:?缺少子集标签}"
EPOCHS="${3:-100}"

# 默认 batch：L20/L40S 48G 显存实测余量充足
if [ "$SIZE" = "n" ]; then
    BATCH="${4:-32}"
elif [ "$SIZE" = "s" ]; then
    BATCH="${4:-16}"
else
    echo "✗ 模型规格只支持 n 或 s，收到: $SIZE"
    exit 1
fi

case "$TAG" in
    sub10|sub25|sub50|sub100) ;;
    *) echo "✗ 子集标签只支持 sub10/sub25/sub50/sub100，收到: $TAG"; exit 1 ;;
esac

DATA_YAML="$PROJECT_DIR/data/coco_${TAG}.yaml"
if [ ! -f "$DATA_YAML" ]; then
    echo "✗ 找不到 $DATA_YAML，请先完成 job_02_prepare.sh"
    exit 1
fi

install_python_deps

# 训练配方（参赛方案 2.3）：SGD + cosine，lr0=0.01，Mosaic/MixUp，最后10轮关 Mosaic
NAME="baseline_${SIZE}_${TAG}_ep${EPOCHS}"
cd "$PROJECT_DIR"
echo "========== 开始训练: $NAME (batch=$BATCH) =========="
yolo detect train \
    model="yolov8${SIZE}.pt" \
    data="$DATA_YAML" \
    epochs="$EPOCHS" \
    imgsz=640 \
    batch="$BATCH" \
    device=0 \
    amp=True \
    optimizer=SGD \
    lr0=0.01 \
    lrf=0.01 \
    cos_lr=True \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    mosaic=1.0 \
    mixup=0.15 \
    close_mosaic=10 \
    workers=8 \
    project="$RUNS_DIR/baseline" \
    name="$NAME" \
    exist_ok=True

echo "========== 训练完成: $NAME =========="
echo "权重与曲线: $RUNS_DIR/baseline/$NAME/"
