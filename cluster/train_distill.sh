#!/bin/bash
# 阶段② 蒸馏训练作业（yolov8l 教师 -> <5M 轻量学生，三损失 + 自适应权重）
# ------------------------------------------------------------
# 提交方式：工程目录选 jhupload，启动命令：
#     bash train_distill.sh sub100                    # 在线教师（L40S/L20 推荐）
#     bash train_distill.sh sub100 100 32             # 自定义轮数 / batch
#     bash train_distill.sh sub100 100 16 online ""   # 显式 online
# 1080Ti（amp=False + 离线缓存，先在快卡跑 cache_teacher.sh）：
#     bash train_distill.sh sub100 100 16 cache "$RUNS_DIR/distill_cache/coco_sub100"
# 可选第 6 个参数开启阶段③ RepC2f 重参数化：
#     bash train_distill.sh sub100 100 32 online "" 1
# 学生默认用阶段① yolov8n 基线 best.pt 暖启；教师在线模式默认 yolov8l.pt
# 资源：在线模式师生同卡，建议 >=16G 显存；8G 卡（4060/1080Ti）用 cache 模式或降 batch
source "$(dirname "$0")/common.sh"

TAG="${1:?用法: bash train_distill.sh <sub10|sub25|sub50|sub100> [epochs] [batch] [online|cache] [cache_dir] [use_rep]}"
EPOCHS="${2:-100}"
BATCH="${3:-32}"
MODE="${4:-online}"
CACHE_DIR="${5:-}"
USE_REP="${6:-0}"

case "$TAG" in
    sub10|sub25|sub50|sub100) ;;
    *) echo "✗ 子集标签只支持 sub10/sub25/sub50/sub100，收到: $TAG"; exit 1 ;;
esac

DATA_YAML="$PROJECT_DIR/data/coco_${TAG}.yaml"
if [ ! -f "$DATA_YAML" ]; then
    echo "✗ 找不到 $DATA_YAML，请先完成 job_02_prepare.sh"
    exit 1
fi

PRETRAINED="$RUNS_DIR/baseline/baseline_n_${TAG}_ep${EPOCHS}/weights/best.pt"
if [ ! -f "$PRETRAINED" ]; then
    echo "! 未找到阶段① 基线权重 $PRETRAINED"
    echo "  回退使用官方 yolov8n.pt 初始化（建议先跑完 train_baseline.sh n $TAG）"
    PRETRAINED="yolov8n.pt"
fi

MODE_ARGS=(--teacher-mode "$MODE")
if [ "$MODE" = "cache" ]; then
    [ -z "$CACHE_DIR" ] && CACHE_DIR="$RUNS_DIR/distill_cache/coco_${TAG}"
    if [ ! -d "$CACHE_DIR" ]; then
        echo "✗ cache 模式但缓存目录不存在: $CACHE_DIR"
        echo "  请先在快卡上执行: bash cache_teacher.sh $TAG $EPOCHS <batch>"
        exit 1
    fi
    MODE_ARGS+=(--cache-dir "$CACHE_DIR" --no-amp)          # 1080Ti 等 Pascal 卡必须关 AMP
else
    MODE_ARGS+=(--teacher yolov8l.pt --amp)                 # Ada/安培卡必须开 AMP
fi

REP_ARGS=()
[ "$USE_REP" = "1" ] && REP_ARGS=(--rep)

install_python_deps

NAME="distill_n_${TAG}_ep${EPOCHS}"
[ "$USE_REP" = "1" ] && NAME="distill_rep_n_${TAG}_ep${EPOCHS}"
cd "$PROJECT_DIR"
echo "========== 开始蒸馏: $NAME (mode=$MODE batch=$BATCH rep=$USE_REP) =========="
python scripts/train_distill.py \
    --model src/distill/yolov8n_student.yaml \
    --pretrained "$PRETRAINED" \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --optimizer SGD \
    --lr0 0.01 \
    --lrf 0.01 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --close-mosaic 10 \
    --seed 0 \
    --project "$RUNS_DIR/distill" \
    --name "$NAME" \
    "${MODE_ARGS[@]}" \
    "${REP_ARGS[@]}"

echo "========== 蒸馏完成: $NAME =========="
echo "权重与曲线: $RUNS_DIR/distill/$NAME/ （best.pt 为纯学生，可直接 val / 导出）"
