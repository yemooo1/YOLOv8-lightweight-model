# -*- coding: utf-8 -*-
"""
阶段一 · 知识蒸馏：教师模型加载与训练环境准备
================================================

参赛方案：面向边缘部署的轻量化目标检测算法
         —— 基于知识蒸馏与结构重参数化的高效推理方案

本文件的职责
------------
1. 训练环境自检：虚拟环境、CUDA、GPU 架构、AMP 状态
2. 加载教师模型 YOLOv8-L（Ultralytics 官方 COCO 预训练权重）
3. 冻结教师参数（推理模式，不参与梯度更新）
4. 输出教师规格报告（参数量 / FLOPs）
5. 教师前向冒烟测试，确认教师可用于蒸馏

硬件约束（GTX 1080 Ti / Pascal sm_61 / 11 GB）
---------------------------------------------
* FP16 吞吐仅为 FP32 的 1/64，AMP 无加速反而更慢；且 Ultralytics 的
  AMP 失败 GPU 黑名单（1660/1650/1630/T400-T2000/K40m）不含 1080 Ti，
  默认 amp=True 会静默生效 —— 必须在所有训练入口显式关闭。
* 教师前向 165.2 GFLOPs vs 学生约 26 GFLOPs，倍率约 6.3x。在线蒸馏会把
  一次 100 epoch 训练从 ~24 h 拉长到 ~176 h，因此正式训练必须走
  「离线蒸馏」（预缓存教师输出）或云端 GPU。

用法
----
    E:/AIC/.venv/Scripts/python.exe src/YOLOv8.py
"""

import sys
from pathlib import Path

# Windows 控制台默认使用 GBK 编码，直接 print 中文或 ✓ 这类符号会乱码，
# 甚至抛 UnicodeEncodeError 让脚本中断。这里强制标准输出走 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent              # E:\AIC
TEACHER_WEIGHTS = PROJECT_ROOT / "yolov8l.pt"
SMOKE_IMAGE = PROJECT_ROOT / "bus.jpg"

DEVICE = 0                                  # GTX 1080 Ti
IMGSZ = 640
# 1080 Ti 上是 Pascal，AMP 必须关闭；此处集中定义，供后续训练脚本共用
USE_AMP = False


# --------------------------------------------------------------------------
# 1. 环境自检
# --------------------------------------------------------------------------
def check_environment() -> dict:
    """检查训练环境是否满足方案要求，返回关键信息字典。"""
    import torch

    print("=" * 68)
    print("1. 训练环境自检")
    print("=" * 68)

    info = {}

    # 虚拟环境
    in_venv = sys.prefix != sys.base_prefix
    info["venv"] = sys.prefix
    print(f"  Python 解释器 : {sys.executable}")
    print(f"  虚拟环境      : {'是' if in_venv else '否（建议使用 E:/AIC/.venv）'}")
    if not in_venv:
        print("    ⚠ 未在虚拟环境中运行，依赖版本可能不受控")

    # PyTorch / CUDA
    info["torch"] = torch.__version__
    info["cuda_built"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA (编译)   : {torch.version.cuda}")
    print(f"  CUDA (可用)   : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。本方案需要 GPU 训练，请检查驱动与 PyTorch 安装。"
        )

    # GPU 架构
    props = torch.cuda.get_device_properties(DEVICE)
    info["gpu"] = props.name
    info["vram_gb"] = props.total_memory / 1024**3
    info["capability"] = torch.cuda.get_device_capability(DEVICE)
    info["arch_list"] = torch.cuda.get_arch_list()
    print(f"  GPU           : {props.name}")
    print(f"  显存          : {info['vram_gb']:.1f} GB")
    print(f"  计算能力      : sm_{info['capability'][0]}{info['capability'][1]}")

    # 关键校验：PyTorch 是否真的带 sm_61 内核
    #（PyTorch >= 2.8 的 cu128/cu129 构建已移除 Pascal，会报
    #  "no kernel image is available for execution on the device"）
    sm_tag = f"sm_{info['capability'][0]}{info['capability'][1]}"
    has_kernel = any(sm_tag in a for a in info["arch_list"])
    info["has_sm_kernel"] = has_kernel
    print(f"  内核架构列表  : {info['arch_list']}")
    if not has_kernel:
        raise RuntimeError(
            f"当前 PyTorch 不包含 {sm_tag} 内核，无法在这张卡上运行。\n"
            f"  修复：pip install torch==2.7.1 torchvision==0.22.1 "
            f"--index-url https://download.pytorch.org/whl/cu126"
        )
    print(f"  {sm_tag} 内核    : 存在 ✓")

    # AMP 状态
    print(f"  AMP 开关      : {'开启' if USE_AMP else '关闭'}")
    if not USE_AMP:
        print("    ✓ 1080 Ti 为 Pascal 架构，FP16 吞吐仅为 FP32 的 1/64，")
        print("      AMP 无加速效果。所有训练入口必须显式传 amp=False。")

    return info


# --------------------------------------------------------------------------
# 2. 加载教师模型
# --------------------------------------------------------------------------
def load_teacher():
    """加载 YOLOv8-L 教师模型并冻结。"""
    from ultralytics import YOLO

    print()
    print("=" * 68)
    print("2. 加载教师模型 YOLOv8-L")
    print("=" * 68)

    if not TEACHER_WEIGHTS.exists():
        print(f"  本地未找到 {TEACHER_WEIGHTS.name}，将从 Ultralytics 官方仓库下载…")

    print(f"  权重路径 : {TEACHER_WEIGHTS}")
    teacher = YOLO(str(TEACHER_WEIGHTS))

    # 移到 GPU 并设为推理模式
    # 教师只提供监督信号，永远是 eval 模式：
    #   - 不更新参数（requires_grad=False）
    #   - BN 使用训练时累积的 running 统计量，不随学生训练漂移
    model = teacher.model
    model.to(f"cuda:{DEVICE}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("  已冻结教师：eval 模式 + requires_grad=False")
    print(f"  教师任务 : {teacher.task}")

    return teacher


# --------------------------------------------------------------------------
# 3. 规格报告
# --------------------------------------------------------------------------
def report_teacher(teacher) -> dict:
    """输出教师模型的参数量与计算量，作为蒸馏配置的依据。"""
    model = teacher.model

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print("=" * 68)
    print("3. 教师规格报告")
    print("=" * 68)
    print(f"  总参数量   : {total/1e6:.1f} M")
    print(f"  可训练参数 : {trainable/1e6:.1f} M  (应为 0)")

    if trainable != 0:
        raise RuntimeError(
            f"教师仍有 {trainable} 个可训练参数，蒸馏前必须全部冻结。"
        )

    # FLOPs 需要一次前向才能统计
    model.info(imgsz=IMGSZ, verbose=False)

    return {"params": total}


# --------------------------------------------------------------------------
# 4. 冒烟测试
# --------------------------------------------------------------------------
def smoke_test(teacher) -> None:
    """用一张真实图片跑一次教师前向，确认教师可用于蒸馏。"""
    import torch

    print()
    print("=" * 68)
    print("4. 教师前向冒烟测试")
    print("=" * 68)

    if not SMOKE_IMAGE.exists():
        print(f"  ⚠ 找不到测试图 {SMOKE_IMAGE}，跳过冒烟测试")
        return

    # 直接对底层 nn.Module 做一次前向，模拟蒸馏时的调用方式
    from ultralytics.data.augment import LetterBox
    import cv2
    import numpy as np

    img = cv2.imread(str(SMOKE_IMAGE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    letterbox = LetterBox((IMGSZ, IMGSZ), auto=False)
    img = letterbox(image=img)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(f"cuda:{DEVICE}")

    with torch.inference_mode():
        torch.cuda.synchronize()
        out = teacher.model(x)
        torch.cuda.synchronize()

    # YOLOv8 检测头在训练模式下返回 (decoded, raw) 的列表
    print(f"  输入张量     : {tuple(x.shape)}")
    print(f"  输出类型     : {type(out).__name__}")
    if isinstance(out, (list, tuple)):
        print(f"  输出元素数   : {len(out)}")
        for i, o in enumerate(out):
            if torch.is_tensor(o):
                print(f"    [{i}] tensor {tuple(o.shape)}  dtype={o.dtype}")
            elif isinstance(o, (list, tuple)):
                # 各层原始特征列表 —— Feature-level 蒸馏（CWD）的特征源
                print(f"    [{i}] {type(o).__name__}（{len(o)} 项）:")
                for j, t in enumerate(o):
                    if torch.is_tensor(t):
                        print(f"         ({j}) {tuple(t.shape)}")
            elif isinstance(o, dict):
                print(f"    [{i}] dict（{len(o)} 项）:")
                for k, v in o.items():
                    if torch.is_tensor(v):
                        print(f"         {k}: tensor {tuple(v.shape)}")
                    elif isinstance(v, (list, tuple)):
                        print(f"         {k}: {type(v).__name__}（{len(v)} 项）")
                        for j, t in enumerate(v):
                            if torch.is_tensor(t):
                                print(f"              ({j}) {tuple(t.shape)}")
                    else:
                        print(f"         {k}: {type(v).__name__}")
            else:
                print(f"    [{i}] {type(o).__name__}")
    elif torch.is_tensor(out):
        print(f"  输出形状     : {tuple(out.shape)}")

    print(f"  显存占用     : {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print("  ✓ 教师前向正常，可用于蒸馏")


# --------------------------------------------------------------------------
# 5. 蒸馏训练入口（待实现）
# --------------------------------------------------------------------------
def prepare_distillation(teacher):
    """
    蒸馏训练入口占位。

    下一步需要补齐的三件事（阶段一剩余工作）：
      1) 学生网络：改进的轻量网络，目标 < 5M 参数（方案 2.1 阶段一）
      2) 蒸馏损失：Feature-level(CWD) + Logit-level(KL) + 关系蒸馏
      3) 自适应蒸馏权重机制（方案创新点 1）

    注意：本地 1080 Ti 上必须走离线蒸馏（预缓存教师输出到磁盘），
          在线蒸馏的墙钟成本约 7.3x，100 epoch 需 ~176 h，不可行。
    """
    print()
    print("=" * 68)
    print("5. 蒸馏训练准备")
    print("=" * 68)
    print("  教师已就绪，以下为阶段一待实现项：")
    print("    [ ] 学生网络结构（目标 < 5M 参数）")
    print("    [ ] 蒸馏损失：CWD + KL + 关系蒸馏")
    print("    [ ] 自适应蒸馏权重机制")
    print("    [ ] 教师输出离线缓存（本地训练的前提）")
    print()
    print("  数据集：MS COCO 2017（118K 训练 / 5K 验证）")
    print("  训练配方：SGD + cosine annealing，初始 lr=0.01，")
    print("            Mosaic / MixUp / 随机翻转缩放")


# --------------------------------------------------------------------------
def main():
    check_environment()
    teacher = load_teacher()
    report_teacher(teacher)
    smoke_test(teacher)
    prepare_distillation(teacher)

    print()
    print("=" * 68)
    print("阶段一环境准备完成。")
    print("=" * 68)


if __name__ == "__main__":
    main()
