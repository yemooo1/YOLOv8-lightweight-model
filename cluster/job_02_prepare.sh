#!/bin/bash
# 作业 02：标注转 YOLO 格式 + 分层抽样子集（10%/25%/50%/100%）
# 提交方式：工程目录选 jhupload，启动命令  bash job_02_prepare.sh
# 前置：job_01_download_coco.sh 完成（annotations/ 与 images/ 齐全）
# 资源建议：CPU 规格即可（8核64G 优先，转换标注是 CPU/IO 密集）
source "$(dirname "$0")/common.sh"

echo "========== [1/2] COCO json 标注 -> YOLO txt（约 12.3 万张） =========="
python "$PROJECT_DIR/scripts/convert_coco_labels.py" --coco-root "$COCO_DIR"

echo "========== [2/2] 分层抽样：sub10 / sub25 / sub50 / sub100（全集） =========="
python "$PROJECT_DIR/scripts/make_subset.py" \
    --ratios 0.10 0.25 0.50 1.0 \
    --seed 0 \
    --coco-root "$COCO_DIR" \
    --out-dir "$PROJECT_DIR/data"

echo "---------- 生成的数据集配置 ----------"
ls -la "$PROJECT_DIR/data"
echo "---------- COCO 根目录下列表文件 ----------"
ls -la "$COCO_DIR"/*.txt

echo "========== 数据准备完成（可以提交 train_baseline.sh 训练作业） =========="
