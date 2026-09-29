# 蒸馏训练使用文档（YOLOv8-L 教师 → <5M 轻量学生）

> 适用代码：`src/distill/`、`scripts/train_distill.py`、`scripts/cache_teacher.py`
> 框架版本：**ultralytics==8.4.145（必须锁版本）**、torch≥2.0
> 教师：YOLOv8-L（冻结）；学生：3.16M 参数（`src/distill/yolov8n_student.yaml`）
> 损失：原生检测损失 + Logit 蒸馏（sigmoid 二元对称 KL）+ 框分布蒸馏（DFL bin KL）+ 特征蒸馏（CWD），带置信度自适应权重

---

## 1. 环境配置

### 1.1 安装依赖

```bash
pip install ultralytics==8.4.145
# 或使用项目清单
pip install -r requirements.txt
# 集群环境
pip install -r cluster/requirements_cluster.txt
```

> ⚠️ 必须锁定 8.4.145：蒸馏代码依赖该版本 Detect 头训练态输出 dict 结构
> （`{"boxes", "scores", "feats"}`）与 `DistillationModel` 的冻结/EMA 行为，跨版本不保证可用。

### 1.2 按显卡选择 AMP

| GPU | 架构 | AMP | 教师模式 |
|---|---|---|---|
| L40S / L20 / RTX 4060 | Ada/Ampere | **`--amp` 必须开** | online 优先 |
| GTX 1080 Ti | Pascal | **`--no-amp` 必须关**（fp16 内核缺失会报错） | cache 推荐 |
| CPU 冒烟 | — | `--no-amp` | online（coco8 即可） |

### 1.3 Windows 本地额外设置（PowerShell）

```powershell
# ultralytics 默认写 %AppData%，沙箱/无权限环境需重定向
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
# 同时装了多个 OpenMP 运行库时防止重复加载中断
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
# DataLoader workers 在 Windows 上建议 0~2（过高会报页面文件不足 WinError 1455）
```

### 1.4 训练前三样前置条件

1. **数据 yaml**：集群 `data/coco_sub{10,25,50,100}.yaml`；本地西瓜集 `watermelon_local.yaml`
2. **教师权重**：online 模式准备 `yolov8l.pt`（或自训教师 best.pt，如西瓜实验的 yolov8s best）
3. **学生初始化权重**（推荐阶段① 基线 best.pt；也可用官方 `yolov8n.pt`，355/355 张量全量迁移）

## 2. 一分钟冒烟（coco8 + CPU）

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -B scripts/train_distill.py `
    --model src/distill/yolov8n_student.yaml --pretrained yolov8n.pt `
    --teacher yolov8n.pt --data coco8.yaml `
    --epochs 2 --batch 4 --imgsz 320 --device cpu --workers 0 --no-amp `
    --project runs/distill --name smoke_online
```

预期：训练完成、验证 mAP 正常打印、`weights/best.pt` 落盘且为纯学生（无教师/对齐层键）。

## 3. 教师输出离线缓存（弱卡/1080Ti 场景）

在线模式下教师前向约占总耗时的一部分（1080Ti 上更明显）。可用快卡先把每个 epoch 的
教师输出（分类 logits、框分布、颈部特征）离线落盘，弱卡训练时直接读缓存。

### 3.1 生成缓存

集群：

```bash
bash cluster/cache_teacher.sh sub100                 # 100 epoch 全量
bash cluster/cache_teacher.sh sub100 100 16 30 59    # 断点补生成 epoch 30~58
```

手动命令：

```bash
python scripts/cache_teacher.py \
    --teacher yolov8l.pt --data data/coco_sub100.yaml \
    --cache-dir runs/distill_cache/coco_sub100 \
    --epochs 100 --batch 16 --imgsz 640 --amp \
    --mosaic 1.0 --mixup 0.15 --close-mosaic 10
```

输出结构：`<cache-dir>/epoch{NNN}/shard_{NNNNNN}.pt`，fp16 存储（图像 uint8）。

### 3.2 缓存与训练必须一致的参数

| 参数 | 原因 |
|---|---|
| `--batch` | 分片按 batch 切分，不一致**启动直接报错** |
| `--imgsz` / `--multi-scale 0` | 分辨率不同会导致师生特征图尺寸不匹配；cache 模式强制 multi_scale=0 |
| `--epochs` / `--close-mosaic` / `--mosaic` / `--mixup` | 第 N epoch 的增强状态必须与缓存对齐，缺 epoch 会报错 |

### 3.3 磁盘与裁剪

- 占用经验值：约 **5MB/图/epoch**（含三层 fp16 特征）；11k 图 × 100 epoch ≈ 5TB，大盘再用
- `--no-feats`：不存颈部特征（训练必须 `--kd-feat-gain 0`）
- `--no-boxes`：不存框分布（训练必须 `--kd-box-gain 0`）
- `--keep N`：只保留最近 N 个 epoch（滚动清理）
- `--start-epoch/--end-epoch`：断点补生成，不重跑已有分片
- 生成后整个缓存目录拷到弱卡机器即可，无需教师权重

## 4. 蒸馏训练

### 4.1 online / cache 模式选择

| 场景 | 模式 | 理由 |
|---|---|---|
| L40S/L20（48G）、RTX 4060（8G，batch 16~32） | **online** | 一条命令，无需 5TB 缓存盘 |
| GTX 1080 Ti / 显存 <8G | **cache** | 避免师生同卡在线前向开销与显存压力 |

### 4.2 集群提交

```bash
# 在线（推荐）
bash cluster/train_distill.sh sub100
bash cluster/train_distill.sh sub100 100 32                 # 自定义轮数/batch
# 离线缓存（1080Ti）
bash cluster/train_distill.sh sub100 100 16 cache "$RUNS_DIR/distill_cache/coco_sub100"
# 阶段③ RepC2f 重参数化（第 6 个参数传 1）
bash cluster/train_distill.sh sub100 100 32 online "" 1
```

脚本默认：阶段① `baseline_n_<tag>_ep<epochs>/best.pt` 暖启（找不到则回退 yolov8n.pt）、
配方与基线一致（SGD+cosine、mosaic 1.0、mixup 0.15、末 10 轮关 mosaic）。

### 4.3 本地命令（PowerShell，西瓜集示例）

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
# 在线
python -B -u scripts/train_distill.py `
    --model src/distill/yolov8n_student.yaml `
    --pretrained runs/watermelon/watermelon_yolov8n_ep100/weights/best.pt `
    --teacher runs/watermelon/watermelon_yolov8s_ep100/weights/best.pt `
    --teacher-mode online --teacher-batch 4 `
    --data watermelon_local.yaml `
    --epochs 100 --batch 32 --imgsz 640 --device 0 --workers 2 --amp `
    --mosaic 1.0 --mixup 0.1 --close-mosaic 15 --seed 0 `
    --project runs/watermelon --name distill_s2n_yolov8n_ep100

# 重参数化版：加 --rep（部署前需用 scripts/reparam_deploy.py 融合）
# 缓存版：--teacher-mode cache --cache-dir <目录> --no-amp
```

### 4.4 蒸馏超参表（命令行可覆盖）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--kd-tau` | 4.0 | KL 温度（logit 与 CWD 共用各自温度项） |
| `--kd-cls-gain` | 4.0 | 分类 logit KL 权重 |
| `--kd-box-gain` | 1.0 | DFL 框分布 KL 权重（**0 关闭**；框精度不足时可调大） |
| `--kd-feat-gain` | 8.0 | CWD 特征蒸馏权重（**0 关闭**） |
| `--cwd-tau` | 4.0 | CWD 通道 softmax 温度 |
| `--kd-adaptive / --no-kd-adaptive` | 开 | 自适应权重 `w = mean(1-conf_student)^γ`（学生越不确定越听教师） |
| `--kd-adaptive-gamma` | 1.0 | 自适应指数 γ |
| `--kd-warmup` | 1.0 | 蒸馏损失线性爬升轮数 |
| `--teacher-batch` | 4 | 在线教师微批次大小（8G 卡建议 2~4） |
| `--rep / --no-rep` | 关 | 学生 C2f→RepC2f 多分支训练 |
| `--cudnn-benchmark` | 关 | 固定输入下 cuDNN 自动选核（放弃严格确定性；小模型实测无收益） |

训练日志每行会打印 `kd_cls_loss / kd_box_loss / kd_feat_loss / kd_weight`，
验证阶段蒸馏项记 0（仅原生检测指标）。

### 4.5 初始化与续训说明

- `--pretrained xxx.pt` 只做学生初始化（start_epoch=0）；官方/自训权重均可
- **不提供 `--resume`**：蒸馏 best/last 只保存纯学生；需要重来时再次 `--pretrained` 启动
- 首 epoch warmup ramp=0 时蒸馏项无梯度属设计行为，ramp>0 后正常

## 5. 产物、验证与导出

- 输出目录：`<project>/<name>/weights/best.pt | last.pt`
- **保存的是纯 `DetectionModel` 学生**：教师权重、1×1 对齐层在保存时剥离，
  可直接被官方 `val` / ONNX / TensorRT 工具链使用
- 学生规模：**3,006,038 参数（3.01M）/ 8.1 GFLOPs@640**，满足 <5M 要求

```bash
# 验证（兼容 cluster/val_model.sh）
yolo detect val model=runs/distill/<name>/weights/best.pt data=<data.yaml> imgsz=640 split=test

# ONNX 导出
yolo export model=<best.pt> format=onnx opset=12 imgsz=640

# RepC2f 训练权重：先融合成单路（内置零误差校验 + ONNX 导出 + CPU 延迟基准）
python scripts/reparam_deploy.py --weights <rep_best.pt> --plain-onnx <普通学生.onnx>
# 产物 *._deploy.pt / *_deploy.onnx 才是部署权重（3.006M，与普通学生部署图同构）

# TensorRT
trtexec --onnx=best_deploy.onnx --saveEngine=best_deploy_fp16.engine --fp16
```

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| cache 模式启动报 batch/epoch 不一致 | 缓存与训练参数必须与 §3.2 表一致，重新生成缺失分片 |
| 1080Ti 训练 fp16 内核报错 | 必须 `--no-amp`，并优先用 cache 模式 |
| 4060/8G 在线 OOM | `--batch 16 --teacher-batch 2`；或改 cache 模式；关掉占显存的其他程序 |
| Windows `WinError 1455 页面文件太小` | `--workers 0` 或 2；增大系统虚拟内存 |
| Windows spawn 递归/freeze_support 报错 | 自写入口脚本须加 `if __name__ == "__main__":` 保护 |
| `OMP: Error #15` | 设 `KMP_DUPLICATE_LIB_OK=TRUE` |
| 首批 kd_* 损失为 0、无梯度 | warmup 设计行为，ramp>0 后正常 |
| 非 n 教师报 `.view()` 张量连续错误 | 已修复（losses.py 使用 reshape）；确认使用当前版本代码 |
| best.pt 出现在很早 epoch（valid 尖峰） | 小验证集 fitness 波动，务必以独立 test 复测结果选型，不要只看 best |

## 7. 代码位置速查

| 模块 | 内容 |
|---|---|
| `src/distill/yolov8n_student.yaml` | 学生结构（depth/width 显式展开） |
| `src/distill/losses.py` | binary logit KL / DFL KL / CWD |
| `src/distill/model.py` | `DistillDetModel`：双模式、对齐层、自适应权重、warmup |
| `src/distill/cache.py` | 教师缓存 Dataset/DataLoader（鸭子类型复用官方主循环） |
| `src/distill/trainer.py` | `DistillDetectionTrainer`（最小侵入，保存纯学生） |
| `src/reparam/rep_c2f.py` | RepConv/RepC2f（fuse 等价融合） |
| `scripts/reparam_deploy.py` | 部署融合 + ONNX + 零误差校验 + 延迟基准 |
