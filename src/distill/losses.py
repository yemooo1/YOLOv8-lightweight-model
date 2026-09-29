# -*- coding: utf-8 -*-
"""
阶段② · 蒸馏损失组件
====================

1. ``binary_logit_kd``  —— 分类 Logit 蒸馏：温度 sigmoid 下的对称二元 KL（对应方案 KL 项）
2. ``box_dist_kd``     —— DFL 框分布蒸馏：reg_max 个 bin 上的 KL
3. ``CWDLoss``         —— Feature 蒸馏：通道对齐 Wasserstein 距离
                          (Shu et al., ICCV 2021, "Channel-wise Knowledge Distillation")

约定（重要）：
    * 每个多尺度组件内部先归约为「标量」，再跨尺度聚合，避免不同分辨率张量直接相加；
    * 所有 KL 项乘 tau^2（Hinton 温度补偿，保证梯度尺度不随温度萎缩）；
    * 教师张量一律 detach（教师冻结，不回传梯度）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_pair(student: torch.Tensor, teacher: torch.Tensor):
    """对齐 dtype / shape，教师 detach。"""
    teacher = teacher.detach().to(dtype=student.dtype, device=student.device)
    return student, teacher


def binary_logit_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float) -> torch.Tensor:
    """分类 logit 的对称二元 KL 蒸馏。

    Args:
        student_logits: (B, A, nc) 学生分类 logits（sigmoid 前）。
        teacher_logits: (B, A, nc) 教师分类 logits。
        tau: 蒸馏温度，越大教师软标签越平滑。

    Returns:
        标量 loss（已乘 tau^2，逐元素 mean 归一）。
    """
    s, t = _as_pair(student_logits, teacher_logits)
    # log-sigmoid 形式，避免 fp16 下 sigmoid→log 的数值不稳定
    log_p_s = F.logsigmoid(s / tau)
    log_1m_p_s = F.logsigmoid(-s / tau)
    with torch.no_grad():
        p_t = torch.sigmoid(t / tau)
        log_p_t = F.logsigmoid(t / tau)
        log_1m_p_t = F.logsigmoid(-t / tau)
    # KL(p_t || p_s)，逐元素后对 batch×anchor×class 取均值
    kl = p_t * (log_p_t - log_p_s) + (1.0 - p_t) * (log_1m_p_t - log_1m_p_s)
    return kl.mean() * (tau ** 2)


def box_dist_kd(student_boxes: torch.Tensor, teacher_boxes: torch.Tensor, reg_max: int, tau: float) -> torch.Tensor:
    """DFL 框分布蒸馏：对 4 条边各自 reg_max 个 bin 的分布做 KL(p_t || p_s)。

    Args:
        student_boxes: (B, A, 4*reg_max) 学生框分布 logits。
        teacher_boxes: (B, A, 4*reg_max) 教师框分布 logits。
        reg_max: DFL 积分区间数（YOLOv8 = 16）。
        tau: 分布温度。

    Returns:
        标量 loss（已乘 tau^2）。
    """
    s, t = _as_pair(student_boxes, teacher_boxes)
    s = s.reshape(*s.shape[:-1], 4, reg_max)
    t = t.reshape(*t.shape[:-1], 4, reg_max)
    log_p_s = F.log_softmax(s / tau, dim=-1)
    with torch.no_grad():
        log_p_t = F.log_softmax(t / tau, dim=-1)
    # KL(p_t || p_s) = sum p_t * (log p_t - log p_s)，在 bin 维求和后整体取均值
    kl = torch.clip(F.kl_div(log_p_s, log_p_t, reduction="none", log_target=True).sum(-1), min=0)
    return kl.mean() * (tau ** 2)


class CWDLoss(torch.nn.Module):
    """通道级 Wasserstein 距离（特征蒸馏）。

    对每个通道在 H×W 空间上的响应做 softmax 归一，得到该通道的「空间注意力分布」，
    再最小化师生该分布之间的对称 KL。前景/背景无需标注，逐像素稠密对齐。
    """

    def __init__(self, tau: float = 4.0):
        super().__init__()
        self.tau = tau

    def forward(self, student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor]) -> torch.Tensor:
        """多尺度 CWD，每个尺度先归约为标量再平均。

        Args:
            student_feats: 已通过 1x1 对齐层的学生特征列表 [(B, Ct, H_i, W_i), ...]。
            teacher_feats: 教师颈部特征列表，空间分辨率与学生一一对应。
        """
        assert len(student_feats) == len(teacher_feats)
        scale_losses = []
        for s, t in zip(student_feats, teacher_feats):
            s, t = _as_pair(s, t)
            n, c, h, w = s.shape
            # reshape 而非 view：Detect 头返回的颈部特征/对齐层输出可能非连续
            s = s.reshape(n * c, h * w)
            t = t.reshape(n * c, h * w)
            log_p_s = F.log_softmax(s / self.tau, dim=1)  # (N*C, HW)
            with torch.no_grad():
                log_p_t = F.log_softmax(t / self.tau, dim=1)
            # 对称 KL：KL(t||s) + KL(s||t)，空间维求和、通道维均值
            kl_ts = torch.clip(F.kl_div(log_p_s, log_p_t, reduction="none", log_target=True).sum(1), min=0)
            kl_st = torch.clip(F.kl_div(log_p_t, log_p_s, reduction="none", log_target=True).sum(1), min=0)
            scale_losses.append(0.5 * (kl_ts + kl_st).mean() * (self.tau ** 2))
        return torch.stack(scale_losses).mean()
