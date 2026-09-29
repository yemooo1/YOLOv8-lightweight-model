#!/bin/bash
# 西瓜成熟度检测训练脚本（单作业自包含：数据集缺失时自动从 jhupload/watermelon.zip 解压）
# ------------------------------------------------------------
# 提交方式：工程目录选 jhupload，启动命令：
#     bash train_watermelon.sh n            # yolov8n（推荐）
#     bash train_watermelon.sh s            # yolov8s
#     bash train_watermelon.sh m            # yolov8m
# 可选第 2/3 个参数覆盖轮数与 batch：
#     bash train_watermelon.sh n 100 16
# 资源：L40S/L20 单卡独占
source "$(dirname "$0")/common.sh"

SIZE="${1:?用法: bash train_watermelon.sh <n|s|m|l|x> [epochs] [batch]}"
EPOCHS="${2:-100}"

case "$SIZE" in
    n) BATCH="${3:-32}" ;;
    s) BATCH="${3:-16}" ;;
    m) BATCH="${3:-8}"  ;;
    l) BATCH="${3:-4}"  ;;
    x) BATCH="${3:-2}"  ;;
    *) echo "✗ 模型规格只支持 n/s/m/l/x，收到: $SIZE"; exit 1 ;;
esac

echo "========== 环境检查 =========="
echo "UPLOAD_DIR     = $UPLOAD_DIR"
echo "DATA_HOME      = $DATA_HOME"
echo "WATERMELON_DIR = $WATERMELON_DIR"
echo "---------- UPLOAD_DIR 内容 ----------"
ls -lh "$UPLOAD_DIR" 2>&1 | head -30
echo "---------- DATA_HOME 内容 ----------"
ls -lh "$DATA_HOME" 2>&1 | head -30

# ---- 数据集自动就位：缺失时直接从 watermelon.zip 解压 ----
if [ ! -d "$WATERMELON_DIR/images/train" ]; then
    echo "========== 数据集不在 $WATERMELON_DIR，尝试自动解压 =========="
    ZIP="$UPLOAD_DIR/watermelon.zip"
    if [ ! -f "$ZIP" ]; then
        echo "✗ 找不到 $ZIP"
        echo "  请确认 jhupload 根目录下有 watermelon.zip"
        exit 1
    fi
    ls -lh "$ZIP"
    python - "$ZIP" "$WATERMELON_DIR" <<'PY'
import sys, zipfile, pathlib
zip_path, dest = sys.argv[1], pathlib.Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    # PowerShell Compress-Archive 会用反斜杠作路径分隔符，Linux 上必须归一化为 /
    norm = [(i, i.filename.replace("\\", "/")) for i in z.infolist()]
    files = [(i, n) for i, n in norm if not n.endswith("/")]
    print(f"压缩包内 {len(norm)} 个条目，示例：")
    for _, n in files[:5]:
        print("   ", n)

    # 探测是否需要剥离根目录前缀（如 西瓜成熟度检测数据集/images/...）
    prefix = ""
    if files:
        parts = pathlib.PurePosixPath(files[0][1]).parts
        if parts and parts[0] not in ("images", "labels", "data.yaml"):
            prefix = parts[0] + "/"
            print(f"检测到根目录前缀: {prefix}")

    count = 0
    for info, name in files:
        rel = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if not rel:
            continue
        if not any(k in rel for k in ("images/", "labels/", "data.yaml")):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(out, "wb") as dst:
            dst.write(src.read())
        count += 1
    print(f"解压 {count} 个文件 -> {dest}")
PY
fi

# ---- 再次校验 ----
echo "========== 数据集校验 =========="
ls -l "$WATERMELON_DIR" 2>&1
if [ ! -d "$WATERMELON_DIR/images/train" ]; then
    echo "✗ 仍然找不到 $WATERMELON_DIR/images/train"
    echo "  上面已列出 $WATERMELON_DIR 的实际内容，请把这段日志发出来"
    exit 1
fi
for split in train valid test; do
    n_img=$(ls "$WATERMELON_DIR/images/$split" 2>/dev/null | wc -l)
    n_lbl=$(ls "$WATERMELON_DIR/labels/$split" 2>/dev/null | wc -l)
    echo "  $split : images=$n_img  labels=$n_lbl"
done

# ---- 运行时生成 data.yaml ----
DATA_YAML="$UPLOAD_DIR/watermelon_runtime.yaml"
cat > "$DATA_YAML" <<EOF
path: $WATERMELON_DIR
train: images/train
val: images/valid
test: images/test
names:
  0: ripe
  1: unripe
nc: 2
EOF
echo "---------- 生成的 data.yaml ----------"
cat "$DATA_YAML"

install_python_deps

# ---- 训练 ----
cd "$UPLOAD_DIR"
NAME="watermelon_yolov8${SIZE}_ep${EPOCHS}"
echo "========== 开始训练: $NAME (batch=$BATCH) =========="

# 用环境变量传参给 python，避免 yolo 命令的 PATH 问题与嵌套引号问题
export WM_DATA_YAML="$DATA_YAML"
export WM_MODEL="yolov8${SIZE}.pt"
export WM_EPOCHS="$EPOCHS"
export WM_BATCH="$BATCH"
export WM_NAME="$NAME"
export WM_PROJECT="$RUNS_DIR/watermelon"

python - <<'PY'
import os
import torch
import ultralytics
from ultralytics import YOLO

print(f"ultralytics {ultralytics.__version__}")
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu = {torch.cuda.get_device_name(0)}")

model = YOLO(os.environ["WM_MODEL"])
model.train(
    data=os.environ["WM_DATA_YAML"],
    epochs=int(os.environ["WM_EPOCHS"]),
    imgsz=640,
    batch=int(os.environ["WM_BATCH"]),
    device=0,
    amp=True,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=15,
    workers=8,
    project=os.environ["WM_PROJECT"],
    name=os.environ["WM_NAME"],
    exist_ok=True,
)
PY

echo "========== 训练完成: $NAME =========="
echo "权重与曲线: $RUNS_DIR/watermelon/$NAME/"
