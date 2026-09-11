# -*- coding: utf-8 -*-
"""环境探针：验证 torch 能识别并实际调用 GPU（由 env_probe.sh 调用）"""
import torch

print("torch 版本      :", torch.__version__)
print("CUDA 编译版本   :", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("GPU 数量        :", torch.cuda.device_count())
print("GPU 型号        :", torch.cuda.get_device_name(0))

# 做一次真实矩阵乘法，确认不只是"能看到卡"而是真的能算
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum().item()
print("GPU 矩阵乘法通过, 校验值:", round(y, 2))
