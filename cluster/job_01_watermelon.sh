#!/bin/bash
# 作业 01：西瓜数据集就位（从 jhupload 解压到 jhaidata）
# 前置：本地把 西瓜成熟度检测数据集.zip 上传到 jhupload 根目录
# 资源：CPU 规格即可（不需要 GPU）
source "$(dirname "$0")/common.sh"

ZIP="$UPLOAD_DIR/watermelon.zip"

echo "========== [1/3] 检查上传的数据集包 =========="
if [ ! -f "$ZIP" ]; then
    echo "✗ 找不到 $ZIP"
    echo "  请先在「我的数据」上传 西瓜成熟度检测数据集.zip 到 jhupload 根目录"
    exit 1
fi
ls -lh "$ZIP"

echo "========== [2/3] 解压到 $WATERMELON_DIR =========="
mkdir -p "$WATERMELON_DIR"
python - "$ZIP" "$WATERMELON_DIR" <<'PY'
import sys, zipfile, pathlib
zip_path, dest = sys.argv[1], pathlib.Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

# zip 可能是 Windows 打的包，条目名用反斜杠（images\test\xxx.jpg），先统一成正斜杠
norm = lambda n: n.replace("\\", "/")

with zipfile.ZipFile(zip_path) as z:
    names = [norm(n) for n in z.namelist()]
    # 探测压缩包结构：第一个有效文件的路径前缀
    # 可能是扁平的 images/...，也可能是 西瓜成熟度检测数据集/images/...
    first_valid = next((n for n in names if not n.endswith("/")), "")
    prefix = ""
    # 如果包含根目录名（非 images/labels/data.yaml 开头），去掉它
    parts = pathlib.PurePosixPath(first_valid).parts
    if parts and parts[0] not in ("images", "labels", "data.yaml"):
        prefix = parts[0] + "/"
        print(f"检测到根目录前缀: {prefix}")

    count = 0
    for info in z.infolist():
        name = norm(info.filename)
        rel = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if not rel or info.is_dir():
            continue
        # 只保留数据相关路径
        if not any(k in rel for k in ("images/", "labels/", "data.yaml")):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(out, "wb") as dst:
            dst.write(src.read())
        count += 1
    print(f"解压 {count} 个文件 -> {dest}")
PY

echo "========== [3/3] 落盘核对 =========="
du -sh "$WATERMELON_DIR" 2>/dev/null || true
total_img=0
for split in train valid test; do
    n_img=$(ls "$WATERMELON_DIR/images/$split" 2>/dev/null | wc -l)
    n_lbl=$(ls "$WATERMELON_DIR/labels/$split" 2>/dev/null | wc -l)
    echo "  $split : images=$n_img  labels=$n_lbl"
    total_img=$((total_img + n_img))
done
if [ "$total_img" -eq 0 ]; then
    echo "✗ 图片数为 0，解压没生效（多半是压缩包内部路径分隔符问题）"
    exit 1
fi
cat "$WATERMELON_DIR/data.yaml" 2>/dev/null || echo "⚠ 原始 data.yaml 缺失"

echo "========== 数据集就位完成（下一步：train_watermelon.sh） =========="
