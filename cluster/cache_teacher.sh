#!/bin/bash
# 教师输出离线缓存作业（供 1080Ti 等弱卡离线蒸馏使用）
# ------------------------------------------------------------
# 提交方式：工程目录选 jhupload，启动命令：
#     bash cache_teacher.sh sub100                 # 全 100 epoch 缓存
#     bash cache_teacher.sh sub100 100 16          # 自定义轮数 / batch
# 断点补生成：
#     bash cache_teacher.sh sub100 100 16 30 59    # 只补 epoch 30~58（--start-epoch 30，不含 end）
# 资源：建议在 L40S/L20/4060 等快卡上跑（amp=True），生成的缓存目录拷到弱卡机器即可
# 注意：缓存与后续训练的 batch / imgsz / epochs / mosaic 配置必须一致，训练启动时会强校验
source "$(dirname "$0")/common.sh"

TAG="${1:?用法: bash cache_teacher.sh <sub10|sub25|sub50|sub100> [epochs] [batch] [start_epoch] [end_epoch]}"
EPOCHS="${2:-100}"
BATCH="${3:-16}"
START_EPOCH="${4:-0}"
END_EPOCH="${5:-$EPOCHS}"

case "$TAG" in
    sub10|sub25|sub50|sub100) ;;
    *) echo "✗ 子集标签只支持 sub10/sub25/sub50/sub100，收到: $TAG"; exit 1 ;;
esac

DATA_YAML="$PROJECT_DIR/data/coco_${TAG}.yaml"
if [ ! -f "$DATA_YAML" ]; then
    echo "✗ 找不到 $DATA_YAML，请先完成 job_02_prepare.sh"
    exit 1
fi

CACHE_DIR="$RUNS_DIR/distill_cache/coco_${TAG}"

install_python_deps

cd "$PROJECT_DIR"
echo "========== 开始生成教师缓存: coco_${TAG} epoch[$START_EPOCH,$END_EPOCH) -> $CACHE_DIR =========="
python scripts/cache_teacher.py \
    --teacher yolov8l.pt \
    --data "$DATA_YAML" \
    --cache-dir "$CACHE_DIR" \
    --epochs "$EPOCHS" \
    --start-epoch "$START_EPOCH" \
    --end-epoch "$END_EPOCH" \
    --batch "$BATCH" \
    --imgsz 640 \
    --device 0 \
    --amp \
    --mosaic 1.0 \
    --mixup 0.15 \
    --close-mosaic 10 \
    --workers 8

echo "========== 缓存生成完成: $CACHE_DIR =========="
echo "拷贝到弱卡机器后用 train_distill.sh 的 cache 模式启动学生训练"
