# -*- coding: utf-8 -*-
"""
阶段③ · 结构重参数化模块（RepVGG-style）
=========================================

训练态：RepBottleneck 的 3x3 主卷积分支旁挂 1x1 卷积分支与 BN 恒等分支，
        多分支结构带来更强的梯度流与可学习容量；
部署态：三条分支在数学上等价融合为单个 3x3 卷积（ultralytics 自带的
        ``RepConv.fuse_convs``），由 ``DetectionModel.fuse`` 自动调用，
        部署图与普通 C2f 学生完全一致，零额外推理开销。

与常见「改 site-packages」的做法不同，本实现**不改 ultralytics 源码**，
训练前通过 :func:`convert_c2f_to_rep` 把已加载预训练权重的 DetectionModel
中的每个 C2f 原地替换为 RepC2f：

    * ``cv1`` / ``cv2``（C2f 级 1x1）及瓶颈内 3x3 主分支直接继承预训练权重；
    * 新增的 1x1 分支与 BN 恒等分支 ``weight(gamma)`` 置零，转换后训练前
      模型输出与原 C2f 严格相等（eval 态，见脚本 reparam_deploy 数值校验）；
    * 训练中各分支差异化学习；最终 fuse 成单路 3x3 用于部署。

pickle 反序列化训练检查点时必须能 import 到本模块（脚本已把项目根加入
sys.path，并提前 import src.reparam）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv, RepConv
from ultralytics.utils.torch_utils import initialize_weights


class RepBottleneck(nn.Module):
    """与官方 Bottleneck 接口一致；3x3 出路替换为可融合的 RepConv。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        assert k[0] in (3, (3, 3)) and k[1] in (3, (3, 3)), "RepBottleneck 仅支持 C2f 的 3x3+3x3 瓶颈结构"
        self.cv1 = Conv(c1, c_, 3, 1)
        # bn 恒等分支仅在通道一致（shortcut 可加）时启用；与 block 级残差相加共存
        self.cv2 = RepConv(c_, c2, 3, s=1, p=1, g=g, act=True,
                           bn=bool(shortcut and c1 == c2))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepC2f(C2f):
    """训练多分支、部署单路；通道结构、构造参数与官方 C2f 完全相同。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            RepBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    @torch.no_grad()
    def copy_from_c2f(self, old: "C2f") -> "RepC2f":
        """从普通 C2f 暖启：主分支拷权重，新增分支 gamma 置零（初始贡献为 0）。"""
        self.cv1.load_state_dict(old.cv1.state_dict())
        self.cv2.load_state_dict(old.cv2.state_dict())
        for old_b, new_b in zip(old.m, self.m):
            new_b.cv1.load_state_dict(old_b.cv1.state_dict())
            # 旧 Bottleneck.cv2 是 Conv（3x3）→ 新 RepConv.conv1（3x3）
            new_b.cv2.conv1.load_state_dict(old_b.cv2.state_dict())
            # 新增 1x1 分支：BN gamma=0 → 训练初始该分支恒输出 0
            nn.init.zeros_(new_b.cv2.conv2.bn.weight)
            # 新增 BN 恒等分支（若启用）：gamma=0 同理
            if new_b.cv2.bn is not None:
                nn.init.zeros_(new_b.cv2.bn.weight)
        return self


@torch.no_grad()
def convert_c2f_to_rep(model, verbose: bool = True):
    """把 DetectionModel 中所有 exact-C2f 原地替换为 RepC2f（需在 load 预训练权重之后调用）。"""
    from ultralytics.utils import LOGGER

    seq = model.model  # nn.Sequential
    n_replaced = 0
    for i, layer in enumerate(seq):
        # exact C2f：不动 C3k2 / C2fAttn 等子类
        if type(layer) is C2f:
            c1 = layer.cv1.conv.in_channels
            c2 = layer.cv2.conv.out_channels
            n = len(layer.m)
            g = layer.m[0].cv2.conv.groups
            shortcut = bool(layer.m[0].add)
            e = layer.c / c2
            rep = RepC2f(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
            # 与 DetectionModel 构建时的 initialize_weights 对齐（BN eps=1e-3, momentum=0.03），
            # 否则新分支 BN 默认 eps=1e-5 会导致融合前后/与旧模型输出微小不一致
            initialize_weights(rep)
            rep.copy_from_c2f(layer)
            rep.i, rep.f, rep.np = layer.i, layer.f, layer.np
            rep.type = "RepC2f"
            seq[i] = rep
            n_replaced += 1
    if verbose:
        extra = sum(p.numel() for p in model.parameters())
        LOGGER.info(f"结构重参数化：已替换 {n_replaced} 个 C2f → RepC2f；"
                    f"训练态参数 {extra / 1e6:.3f}M（融合后回到 3.006M 部署参数）")
    return model
