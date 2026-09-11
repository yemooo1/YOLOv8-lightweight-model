# 集群作业公共环境（被 job_*.sh / train_baseline.sh 以 source 方式引用）
# 探针作业（446/448/450/451）实测结论：
#   - yolo 命令装在 ~/.local/bin，不在默认 PATH
#   - 容器内 ~/.config 不可写，ultralytics 退化到 /tmp（每次作业清空）
#   - jhupload / jhaidata 都是持久目录链接，产物必须落在这两个区
set -e

# ---- 门户家目录与持久目录（用户名 chenyungui，不同机器提交时只需改这里） ----
PORTAL_HOME="${PORTAL_HOME:-/public/home/chenyungui}"
UPLOAD_DIR="$(readlink -f "$PORTAL_HOME/jhupload")"      # 代码/脚本/训练产物
DATA_HOME="$(readlink -f "$PORTAL_HOME/jhaidata")"       # 大数据集（20GB COCO）

# ---- 项目布局 ----
PROJECT_DIR="$UPLOAD_DIR/YOLOv8-lightweight-model"       # zip 解压后的项目目录
COCO_DIR="$DATA_HOME/coco"                               # COCO 数据根目录
RUNS_DIR="$UPLOAD_DIR/runs"                              # 所有训练输出

# ---- 运行时环境 ----
export PATH="$HOME/.local/bin:$PATH"
export YOLO_CONFIG_DIR="$UPLOAD_DIR/.ultralytics"        # ultralytics 配置持久化
export YOLO_AUTOINSTALL=False                            # 不自动装包/弹窗

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

install_python_deps() {
    echo "========== 安装 Python 依赖（集群清单） =========="
    python -m pip install -i "$PIP_MIRROR" \
        -r "$PROJECT_DIR/cluster/requirements_cluster.txt"
}

echo "UPLOAD_DIR = $UPLOAD_DIR"
echo "DATA_HOME  = $DATA_HOME"
echo "PROJECT_DIR= $PROJECT_DIR"
echo "COCO_DIR   = $COCO_DIR"
echo "RUNS_DIR   = $RUNS_DIR"
