# YOLOv8 轻量化目标检测 · 参赛方案

> **参赛主题**：面向边缘部署的轻量化目标检测算法
> —— 基于知识蒸馏与结构重参数化的高效推理方案
>
> **竞赛名称**：边缘AI模型轻量化
> **团队成员**：林子煌、曾子键、王濛
> **时间线**：2026-09 ~ 2026-10.1

---

## 📋 四阶段工作流程

| 阶段 | 内容 | 负责 | 状态 | 产出 |
|:---:|------|:---:|:---:|------|
| **① 基线训练** | YOLOv8-n / YOLOv8-s × COCO 子集 (10%/25%/50%/100%) 共 8 组实验 | A | ✅ 已完成 | 8 个 best.pt + mAP 曲线 |
| **② 知识蒸馏** | YOLOv8-s（教师）→ YOLOv8-n 学生（西瓜 2 类主线） | B | ✅ 已完成 | 蒸馏后学生权重 |
| **③ 结构重参数化** | 训练时多分支 → 推理时单分支融合（RepC2f） | B | ✅ 已完成 | 融合部署权重 best_deploy.pt |
| **④ 部署优化** | ONNX → TensorRT FP16 实测（L40S）；INT8 / 边缘板未做 | C | ✅ 已完成 | FP16 引擎 + 精度/延迟对比数据 |

### 阶段间依赖链

```
① 基线训练 ──确立数据量-mAP关系──▶ ② 知识蒸馏 ──获得轻量权重──▶ ③ 重参数化 ──简化计算图──▶ ④ 部署
                                        ▲
                                        │ 教师权重
                                   （COCO预训练YOLOv8-L）
```

---

## 一、环境配置

### 1.1 本地开发（三人统一）

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

**硬性约束**：
| 项目 | 要求 | 原因 |
|------|------|------|
| PyTorch | `2.7.1+cu126` （不可 ≥2.8） | 成员A 的 GTX 1080 Ti 是 Pascal (sm_61)，PyTorch 2.8+ 的 cu128/cu129 已移除该内核 |
| ultralytics | `8.4.145`（三人版本完全一致） | 保证权重互载与 mAP 可复现 |
| 1080 Ti 训练 | 必须 `amp=False` | Pascal 架构 FP16 吞吐仅 FP32 的 1/64，AMP 无加速反而更慢 |
| RTX 4060 8GB | 必须 `amp=True`，避免训练 YOLOv8-L | 显存约束 |

自检命令：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1.2 学校 HPC 集群

| 项目 | 值 |
|------|----|
| 容器镜像 | `jhinno/pytorch:2.1.0` |
| GPU | L40S / L20（48GB 显存） |
| 共享内存 | `shm-size=8192MB`（避免 DataLoader 多进程报错） |
| 持久化目录 | `jhupload/`（代码+产物）、`jhaidata/`（数据集） |
| 容器内路径 | 会随作业销毁，产物必须落在持久化目录 |
| 训练 AMP | `amp=True`（Ada 架构有 FP16 加速） |
| numpy 版本 | 必须显式 `pip install numpy==1.26.4`（兼容 PyTorch 2.1.0） |

集群 Python 依赖清单见 [cluster/requirements_cluster.txt](cluster/requirements_cluster.txt)。

---

## 二、数据流水线

### 2.1 COCO2017 数据集

| 项目 | 值 |
|------|----|
| 训练集 | train2017：118,287 张 |
| 验证集 | val2017：5,000 张 |
| 类别数 | 80 类 |
| 总大小 | ~20 GB |

### 2.2 集群作业依赖链（严格串行）

```
job_00_setup.sh            # 项目解压 + 预下载 yolov8n/s.pt 权重
    │
    ▼
job_01_download_coco.sh     # HF 镜像多线程下载 COCO + 解压（支持断点续传）
    │
    ▼
job_02_prepare.sh           # COCO json → YOLO txt + 分层抽样子集
    │
    ▼
train_baseline.sh           # 基线训练（8 组，见 §三）
    │
    ▼
val_model.sh                # 在完整 val2017 上统一评测
```

每个作业必须**看到上一步日志的成功结尾**再提交下一个，禁止并行提交。

提交方式：
- 工程目录选 `jhupload`
- 启动命令示例：`bash job_00_setup.sh`

### 2.3 分层子集

由 [scripts/make_subset.py](scripts/make_subset.py) 按主类别分层抽样，seed=0，验证集固定为完整 val2017。

| 子集标签 | 比例 | 训练图数（约） | 生成的配置文件 |
|:---:|:---:|:---:|---|
| sub10 | 10% | ~11,800 | `data/coco_sub10.yaml` |
| sub25 | 25% | ~29,500 | `data/coco_sub25.yaml` |
| sub50 | 50% | ~59,100 | `data/coco_sub50.yaml` |
| sub100 | 100% | 118,287 | `data/coco_sub100.yaml` |

> ⚠️ `data/` 和 `datasets/` 目录**不随仓库提交**，由 `job_02_prepare.sh` 在集群上生成。

---

## 三、阶段① · 基线训练 ✅

### 3.1 实验矩阵（2 × 4 = 8 组）

训练命令格式：
```bash
bash train_baseline.sh <n|s> <sub10|sub25|sub50|sub100>
```

| 模型 \ 子集 | sub10 | sub25 | sub50 | sub100 |
|------------:|:-----:|:-----:|:-----:|:------:|
| **yolov8n** (3.2M) | ✅ | ✅ | ✅ | ✅ |
| **yolov8s** (11.2M) | ✅ | ✅ | ✅ | ✅ |

### 3.2 训练配方（所有组统一）

| 参数 | 值 |
|------|----|
| epochs | 100 |
| imgsz | 640 |
| batch | n: 32 / s: 16 |
| optimizer | SGD |
| lr0 | 0.01 |
| lrf | 0.01 |
| lr schedule | Cosine annealing |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup | 3 epochs |
| mosaic | 1.0 |
| mixup | 0.15 |
| close_mosaic | 最后 10 epochs |

### 3.3 基线实验结果

> 💡 训练脚本最后一轮**自动在完整 val2017 上做 COCO eval**，以下 mAP 直接取训练日志中的 `Average Precision` 行即可。如需复测可提交 `val_model.sh`。

| 实验 | 参数量 (M) | FLOPs (G) | mAP@0.5 | mAP@0.5:0.95 | 推理速度 (ms) |
|------|:----------:|:---------:|:-------:|:------------:|:-------------:|
| yolov8n + sub10 | 3.2 | 8.7 | **0.434** | **0.294** | 0.5 |
| yolov8n + sub25 | 3.2 | 8.7 | **0.479** | **0.331** | 0.5 |
| yolov8n + sub50 | 3.2 | 8.7 | **0.499** | **0.350** | 0.5 |
| yolov8n + sub100 | 3.2 | 8.7 | **0.516** | **0.364** | 0.5 |
| yolov8s + sub10 | 11.2 | 28.6 | **0.553** | **0.389** | 0.8 |
| yolov8s + sub25 | 11.2 | 28.6 | **0.561** | **0.399** | 0.9 |
| yolov8s + sub50 | 11.2 | 28.6 | **0.589** | **0.424** | 0.9 |
| yolov8s + sub100 | 11.2 | 28.6 | **0.611** | **0.444** | 0.8 |

> 📌 训练产物位置：`jhupload/runs/baseline/<name>/weights/best.pt`

### 3.4 阶段① 结论与后续决策

- [ ] 确定哪个子集规模足够支撑后续知识蒸馏
- [ ] 确定学生网络初始架构（以 yolov8n 为起点或自定义更轻量结构）
- [ ] 准备教师权重（COCO 预训练 YOLOv8-L）

---

## 四、阶段② · 知识蒸馏 🔄

### 4.1 核心思路

```
教师网络 YOLOv8-L (43.7M, 165.2 GFLOPs)
    │ 冻结参数，仅推理
    │ 提供两种监督信号：
    │   1. Logit-level (KL divergence)  —— 输出分布拟合
    │   2. Feature-level (CWD)         —— 中间特征匹配
    ▼
学生网络 <5M 参数  ——  训练目标：既拟合真实标签，又拟合教师信号
```

### 4.2 蒸馏损失设计

| 损失组件 | 公式 | 权重 |
|----------|------|:----:|
| 检测损失（原生） | L_cls + L_box + L_dfl | 1.0 |
| Logit 蒸馏损失 | KL(σ(z_T / τ) ‖ σ(z_S / τ)) × τ² | 自适应 |
| Feature 蒸馏损失 | CWD（通道对齐的 Wasserstein 距离） | 自适应 |

### 4.3 关键技术点（方案创新）

1. **自适应蒸馏权重机制**：根据学生输出置信度动态调整蒸馏损失权重——置信度低时多听教师，置信度高时多信标签
2. **离线蒸馏**：预缓存教师输出到磁盘，避免 1080 Ti 上在线蒸馏的 7.3x 墙钟成本
3. **特征对齐层**：在学生颈部添加 1×1 卷积，将学生特征通道数映射到教师对应层，才能计算 CWD

### 4.4 进度追踪

- [ ] 学生网络结构设计（目标 <5M 参数）
- [ ] 蒸馏损失实现（CWD + KL + 自适应权重）
- [ ] 教师输出离线缓存脚本
- [ ] 蒸馏训练脚本（支持本地 amp=False / 集群 amp=True）
- [ ] 蒸馏训练实验（用阶段①确定的最优子集）

---

## 五、阶段③ · 结构重参数化 ⏳

### 5.1 原理

训练时使用多分支结构获取更高精度，推理时将分支融合为单一卷积，在**零精度损失**下加速推理：

```
训练时:           推理时:
    ┌ Conv ─┐
    ├ Conv ─┼─→ fuse ──→ Single Conv
    └ BN ───┘
```

### 5.2 应用位置

| 位置 | 训练时多分支 | 推理时融合 |
|------|-------------|-----------|
| 学生网络 Backbone 的 3×3 卷积 | 3×3 Conv + 1×1 Conv + BN | 单个 3×3 Conv |
| 学生网络 Neck 的 C2f 模块 | 多分支 Bottleneck | 简化计算图 |

### 5.3 进度追踪

- [ ] 定义 reparameterization 模块
- [ ] 训练路径（带多分支）vs 推理路径（融合后）的代码分离
- [ ] 验证融合前后权重输出一致性（数值误差 < 1e-5）
- [ ] 导出 reparameterized.pt

---

## 六、阶段④ · 部署优化 ✅

### 6.1 导出路径

```
PyTorch (.pt) → ONNX (opset 12) → TensorRT FP16 → （INT8 / 边缘板：未执行）
```

| 步骤 | 方式 | 产出 | 状态 |
|------|------|------|:---:|
| PyTorch → ONNX | `yolo export format=onnx opset=12` | `best_deploy.onnx` | ✅ |
| ONNX → TensorRT FP16 | `yolo export format=engine half=True`（ultralytics 内部走 TensorRT Python API） | `wm_rep_deploy.engine` 8.3 MB | ✅ |
| INT8 量化 | 需约 500 张校准图 + 重新构建引擎 | — | ⏸ 时间不足，跳过 |

> 注：pip 安装的 TensorRT wheel **不含 `trtexec`** 命令行工具，故改用 ultralytics 的 `format=engine` 导出（内部等价于 ONNX → engine 两步）。

### 6.2 实测结果（西瓜集 test split，1041 张）

| 指标 | PyTorch FP32 | TensorRT FP16 | 变化 |
|------|:------------:|:-------------:|:----:|
| mAP@0.5 | 0.6328 | 0.6308 | −0.20 点 |
| mAP@0.5:0.95 | 0.4303 | 0.4207 | −0.96 点 |
| 推理耗时 (640×640, 不含前后处理) | 0.69 ms/图 | **0.45 ms/图** | **1.53× 加速** |
| 部署体积 | 11.7 MB (.pt) | **8.3 MB (.engine)** | −29% |

**测试条件**：NVIDIA L40S（Ada, sm_89）、imgsz=640、西瓜 test split 全量 1041 张。引擎与 GPU 架构绑定，仅能在 Ada 卡（L40S / L20）上运行。

### 6.3 进度追踪

- [x] ONNX 导出脚本 + 验证
- [x] TensorRT FP16 引擎构建 + 验证（精度回退 < 1 点，加速 1.53×）
- [ ] ~~TensorRT INT8 量化~~（时间不足；FP16 数据已足够支撑部署章节）
- [ ] ~~边缘设备实测~~（改用 L40S 推算，展示时如实标注测试硬件）
- [ ] 精度-速度帕累托曲线出图（与 yolov8n/s 基线对比）

---

## 七、项目结构

```
YOLOv8-lightweight-model/
├── README.md                     ← 本文件
├── requirements.txt              ← 本地开发依赖（三人统一）
├── 参赛方案.docx                 ← 原始方案文档（参考）
│
├── src/                          ← 源码
│   └── YOLOv8.py                 ← 教师模型加载 + 环境自检 + 蒸馏占位
│
├── scripts/                      ← 数据准备脚本
│   ├── download_coco.py          ← COCO 多线程下载（支持断点续传）
│   ├── extract_coco.py           ← ZIP 解压
│   ├── convert_coco_labels.py    ← COCO json → YOLO txt
│   ├── make_subset.py            ← 分层抽样子集
│   └── find_client.py            ← （空文件 · 预留）
│
├── cluster/                      ← 学校 HPC 集群作业脚本
│   ├── common.sh                 ← 公共环境（被所有 job 脚本 source）
│   ├── job_00_setup.sh           ← 项目就位 + 预下载权重
│   ├── job_01_download_coco.sh   ← 下载 COCO2017
│   ├── job_02_prepare.sh         ← 标注转换 + 子集生成
│   ├── train_baseline.sh         ← 基线训练（8 组实验）
│   ├── val_model.sh              ← 统一评测
│   ├── env_probe.sh              ← 环境探针（调试用）
│   └── requirements_cluster.txt  ← 集群 Python 依赖（与本地不同）
│
├── cluster_upload/               ← 上传到集群的打包副本（含 zip，不含 env_probe.sh / requirements_cluster.txt）
│
├── data/                         ← 生成产物：yaml + manifest（job_02 产出）
│   ├── coco_sub10.yaml
│   ├── coco_sub25.yaml
│   ├── coco_sub50.yaml
│   ├── coco_sub100.yaml
│   ├── manifest.csv
│   └── manifest_per_class.csv
│
├── datasets/                     ← 生成产物：COCO2017 数据（~20GB）
│   ├── images/{train2017,val2017}/
│   ├── labels/{train2017,val2017}/
│   ├── annotations/
│   └── *.txt                     ← 子集图片列表
│
└── runs/                         ← 训练产物（集群持久化目录）
    ├── baseline/                 ← 基线训练输出
    └── eval/                     ← 评测输出
```

> ⚠️ `data/`、`datasets/`、`runs/` 均为**运行时生成**，不随仓库提交。

---

## 八、协作分工

| 成员 | 阶段① 基线 | 阶段② 蒸馏 | 阶段③ 重参数化 | 阶段④ 部署 | GPU 环境 |
|:---:|:----------:|:----------:|:--------------:|:----------:|---------|
| **A** | 主责        | 辅助验证    | —              | —          | GTX 1080 Ti (sm_61)，**amp=False** |
| **B** | —          | 主责        | 主责            | —          | RTX 4060 8GB，**amp=True**，避免训练 YOLOv8-L |
| **C** | —          | —           | —              | 主责        | （待确认）TensorRT/ONNX 环境 |

三人共同：
- 统一使用 `ultralytics==8.4.145`
- 统一评测管线（`val_model.sh` 输出为最终基准）
- 集群作业**严格串行**，上一步日志成功结尾再提下一步

---

## 九、常见问题

### Q1: 集群上 `yolo` 命令找不到？
`yolo` 装在 `~/.local/bin`，不在默认 PATH。解决方法见 `cluster/common.sh`，已自动 `export PATH="$HOME/.local/bin:$PATH"`。

### Q2: 训练 DataLoader 卡死或报 shared memory 错误？
集群作业必须设置 `shm-size=8192MB`，否则多进程 DataLoader 会失败。

### Q3: 1080 Ti 训练报 "no kernel image is available for execution on the device"？
说明 PyTorch ≥ 2.8，已移除 sm_61 内核。降级到 `torch==2.7.1+cu126`：
```bash
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### Q4: 集群容器里的产物为什么看不到了？
容器内路径（如 `/tmp`、工作目录）在作业销毁后会清空。所有产物必须落在 `jhupload/` 或 `jhaidata/`。

### Q5: COCO 下载中断了怎么办？
直接重跑 `bash job_01_download_coco.sh`，下载脚本支持分片断点续传，会自动补齐缺失部分。

---

*最后更新：2026-09-14*
