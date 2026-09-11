#!/bin/bash
# 作业 01：下载并解压 COCO2017（约 20GB，支持分片断点续传，中断后重跑自动补齐）
# 提交方式：工程目录选 jhupload，启动命令  bash job_01_download_coco.sh
# 资源建议：GPU 规格均可（本作业不用 GPU；也可选 CPU 8核64G，省出显卡）
source "$(dirname "$0")/common.sh"

echo "========== [0/3] 存储空间与目录 =========="
df -h "$DATA_HOME" || true
mkdir -p "$COCO_DIR"

echo "========== [1/3] 下载 COCO2017 三个包（hf-mirror 国内源） =========="
# 纯标准库脚本，无需安装额外依赖
# 若 hf 源不通，把 --source hf 改成 --source official
python "$PROJECT_DIR/scripts/download_coco.py" \
    --data-dir "$COCO_DIR" \
    --source hf \
    --threads 32

echo "========== [2/3] 解压（images/ 与 annotations/ 布局） =========="
python "$PROJECT_DIR/scripts/extract_coco.py" --coco-root "$COCO_DIR"

echo "========== [3/3] 落盘核对 =========="
du -sh "$COCO_DIR"/* 2>/dev/null || true
df -h "$DATA_HOME" || true
echo "========== COCO 下载解压完成（下一步：job_02_prepare.sh） =========="
