#!/bin/bash
# 学校集群环境探针作业（L40S/L20 + jhinno/pytorch:2.1.0）
# 逐项验证：显卡 -> Python/torch/CUDA -> GPU 实算 -> 装 ultralytics -> 跑通 1 轮训练
set -e

# pip 用户级安装的 yolo 命令在 ~/.local/bin，需加入 PATH
export PATH="$HOME/.local/bin:$PATH"

echo "========== [1/5] nvidia-smi =========="
nvidia-smi

echo "========== [2/5] Python / torch 版本 =========="
python -V
python env_check.py

echo "========== [3/5] 安装 ultralytics（清华源，numpy 锁 1.26.4 兼容 torch2.1） =========="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy==1.26.4" "ultralytics==8.4.145"

echo "========== [4/5] ultralytics 版本确认 =========="
yolo checks

echo "========== [5/5] coco8 跑通 1 轮真实 GPU 训练 =========="
yolo detect train model=yolov8n.pt data=coco8.yaml epochs=1 imgsz=320 device=0 amp=True name=env_probe

echo "========== 探针全部通过 =========="
