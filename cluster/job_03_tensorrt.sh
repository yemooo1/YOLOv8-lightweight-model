#!/bin/bash
# 作业 03：TensorRT 部署验证（探测 → 安装 → 构建 FP16 引擎 → 精度/延迟对比）
# ------------------------------------------------------------
# 提交方式：工程目录选 jhupload，启动命令：
#     bash job_03_tensorrt.sh                                  # 用默认权重路径
#     bash job_03_tensorrt.sh <best.pt 路径>                    # 指定权重
# 资源：L40S/L20 单卡独占（TensorRT 构建必须真卡，不能用 CPU）
# 说明：pip 装的 tensorrt 不含 trtexec，这里全部走 Python API（ultralytics 内部实现）
source "$(dirname "$0")/common.sh"

# 自定义层 src.reparam 放在 jhupload 根目录；反序列化含 RepC2f 的权重时必须能 import 到
export PYTHONPATH="$UPLOAD_DIR${PYTHONPATH:+:$PYTHONPATH}"

WEIGHTS="${1:-$RUNS_DIR/watermelon/watermelon_yolov8n_ep100/weights/best.pt}"
TRT_DIR="$RUNS_DIR/tensorrt"
mkdir -p "$TRT_DIR"

echo "========== [1/5] 探测：容器里是否已有 TensorRT =========="
python - <<'PY'
try:
    import tensorrt as trt
    print("✓ 已预装 tensorrt:", trt.__version__)
except Exception as e:
    print("✗ 未预装 tensorrt:", type(e).__name__, str(e)[:120])
PY
echo "--- trtexec ---"
which trtexec || echo "✗ 无 trtexec（pip 包本来就不含，属正常）"
echo "--- libnvinfer ---"
ls /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -i nvinfer | head -5 || echo "✗ 无 libnvinfer"

echo "========== [2/5] 安装依赖（ultralytics + tensorrt-cu12） =========="
install_python_deps

if python -c "import tensorrt" 2>/dev/null; then
    echo "✓ tensorrt 已可用，跳过安装"
else
    echo "尝试从清华源安装 tensorrt-cu12 ..."
    if python -m pip install -i "$PIP_MIRROR" --no-cache-dir tensorrt-cu12 2>&1 | tail -15; then
        python -c "import tensorrt as trt; print('✓ 安装成功, 版本:', trt.__version__)" \
            || { echo "✗ 装了但导入失败，看上面的报错"; exit 1; }
    else
        echo "✗ tensorrt-cu12 安装失败"
        echo "  常见原因：容器 CUDA 版本与 wheel 不匹配（可试 tensorrt-cu12==10.7.0 等更低版本）"
        echo "  兜底方案：放弃 TensorRT，用已有 ONNX + CPU 延迟数据做 PPT 部署章节"
        exit 1
    fi
fi

echo "========== [3/5] 准备权重与数据配置 =========="
if [ ! -f "$WEIGHTS" ]; then
    echo "✗ 找不到权重：$WEIGHTS"
    echo "  本地训好的权重要先上传到 jhupload，再指定路径，例如："
    echo "    bash job_03_tensorrt.sh $UPLOAD_DIR/best_deploy.pt"
    exit 1
fi
ls -lh "$WEIGHTS"

# 重参数化权重（wm_rep_deploy.pt）的 pickle 引用了 src.reparam.rep_c2f，必须能导入
if [ ! -f "$UPLOAD_DIR/src/reparam/rep_c2f.py" ] && [ -f "$UPLOAD_DIR/src.zip" ]; then
    echo "从 src.zip 解压自定义模块 -> $UPLOAD_DIR/src"
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$UPLOAD_DIR/src.zip" "$UPLOAD_DIR"
fi
if [ -f "$UPLOAD_DIR/src/reparam/rep_c2f.py" ]; then
    python -c "import src.reparam; print('✓ src.reparam 可导入，含 RepC2f 的权重可正常加载')" \
        || echo "⚠ src.reparam 导入失败，含 RepC2f 的权重要报 ModuleNotFoundError"
else
    echo "⚠ 缺 $UPLOAD_DIR/src/reparam/rep_c2f.py：只能加载不含自定义模块的权重"
fi

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
echo "data.yaml -> $DATA_YAML"

echo "========== [4/5] 构建 TensorRT FP16 引擎 =========="
export TRT_WEIGHTS="$WEIGHTS"
export TRT_OUT="$TRT_DIR"
export TRT_DATA="$DATA_YAML"

python - <<'PY'
import os, shutil, time
import torch
from ultralytics import YOLO

w = os.environ["TRT_WEIGHTS"]
print(f"torch {torch.__version__}  gpu = {torch.cuda.get_device_name(0)}")

# 1) PyTorch 基准（延迟 + test 精度）
print("\n---------- [A] PyTorch 基准 ----------")
pt_model = YOLO(w)
t0 = time.time()
pt_metrics = pt_model.val(data=os.environ["TRT_DATA"], split="test",
                          imgsz=640, device=0, verbose=False)
pt_time = time.time() - t0
pt_speed = pt_metrics.speed  # {'preprocess','inference','loss','postprocess'}
print(f"PyTorch  mAP50={pt_metrics.box.map50:.4f}  mAP50-95={pt_metrics.box.map:.4f}")
print(f"PyTorch  inference={pt_speed['inference']:.2f} ms/img  (val 总耗时 {pt_time:.1f}s)")

# 2) 构建 TensorRT FP16 引擎（Python API，无需 trtexec）
print("\n---------- [B] 导出 TensorRT FP16 ----------")
engine_path = pt_model.export(format="engine", half=True, imgsz=640, device=0, verbose=False)
print(f"引擎文件: {engine_path}")

# 3) TensorRT 精度 + 延迟
print("\n---------- [C] TensorRT 对比 ----------")
trt_model = YOLO(str(engine_path))
trt_metrics = trt_model.val(data=os.environ["TRT_DATA"], split="test",
                            imgsz=640, device=0, verbose=False)
trt_speed = trt_metrics.speed
print(f"TensorRT mAP50={trt_metrics.box.map50:.4f}  mAP50-95={trt_metrics.box.map:.4f}")
print(f"TensorRT inference={trt_speed['inference']:.2f} ms/img")

# 4) 汇总（PPT 直接用这张表）
print("\n========== 部署对比汇总（test 集） ==========")
print(f"{'后端':<12}{'精度':<10}{'mAP50':<10}{'mAP50-95':<12}{'推理 ms/图':<12}{'加速比'}")
print("-" * 66)
print(f"{'PyTorch':<12}{'FP32':<10}{pt_metrics.box.map50:<10.4f}{pt_metrics.box.map:<12.4f}"
      f"{pt_speed['inference']:<12.2f}{'1.00x'}")
accel = pt_speed["inference"] / trt_speed["inference"] if trt_speed["inference"] else 0
print(f"{'TensorRT':<12}{'FP16':<10}{trt_metrics.box.map50:<10.4f}{trt_metrics.box.map:<12.4f}"
      f"{trt_speed['inference']:<12.2f}{accel:.2f}x")

# 5) 引擎落盘到持久目录
dst = os.path.join(os.environ["TRT_OUT"], os.path.basename(str(engine_path)))
if not os.path.exists(dst):
    shutil.copy(str(engine_path), dst)
print(f"\n引擎已保存: {dst}")
print(f"⚠ 引擎与 GPU 架构绑定，此引擎仅能在 Ada 卡（L40S/L20）上运行")
PY

echo "========== [5/5] 完成 =========="
ls -lh "$TRT_DIR"
echo "上面的「部署对比汇总」表可直接用于 PPT 部署章节"