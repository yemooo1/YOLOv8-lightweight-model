#!/bin/bash
# 作业 00：项目就位 —— 解压代码 zip + 预下载预训练权重
# 提交方式：工程目录选 jhupload，启动命令  bash job_00_setup.sh
# 前置：先把 YOLOv8-lightweight-model.zip 上传到 jhupload 根目录
source "$(dirname "$0")/common.sh"

echo "========== [1/3] jhupload 当前内容 =========="
ls -lah "$UPLOAD_DIR"

echo "========== [2/3] 解压项目代码 =========="
ZIP="$UPLOAD_DIR/YOLOv8-lightweight-model.zip"
if [ ! -f "$ZIP" ]; then
    echo "✗ 找不到 $ZIP"
    echo "  请先在「我的数据」把 YOLOv8-lightweight-model.zip 上传到 jhupload 根目录"
    exit 1
fi
python - "$ZIP" "$UPLOAD_DIR" <<'PY'
import sys, zipfile, pathlib
zip_path, dest = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    z.extractall(dest)
print(f"解压完成：{len(names)} 个文件 -> {dest}")
PY

echo "---------- 项目目录结构 ----------"
ls -la "$PROJECT_DIR"
find "$PROJECT_DIR" -maxdepth 2 -type f | sort | head -40

echo "========== [3/3] 预下载预训练权重（yolov8n / yolov8s） =========="
# 探针已证明容器能访问 github（yolov8n.pt 自动下载成功）
# 权重直接放项目根目录：训练作业 cd 到项目目录后 model=yolov8n.pt 即可命中
cd "$PROJECT_DIR"
python - <<'PY'
from pathlib import Path
import urllib.request

targets = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
}
for rel, url in targets.items():
    p = Path(rel)
    if p.exists() and p.stat().st_size > 1_000_000:
        print(f"{rel} 已存在（{p.stat().st_size/1e6:.1f} MB），跳过")
        continue
    print(f"下载 {url} -> {rel}")
    urllib.request.urlretrieve(url, p)
    print(f"  完成 {p.stat().st_size/1e6:.1f} MB")
PY

echo "========== setup 完成：项目与权重已就位 =========="
