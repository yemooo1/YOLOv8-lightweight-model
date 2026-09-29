# -*- coding: utf-8 -*-
"""
阶段② · 蒸馏包装模型
====================

把「冻结教师 + 可训练学生 + 1x1 特征对齐层」组装成一个符合 ultralytics 训练协议的
nn.Module：

    trainer 调用 self.model(batch_dict) → DistillDetModel.loss()
        ├── 学生原生检测损失（v8DetectionLoss：box / cls / dfl）
        ├── 分类 Logit 蒸馏（对称二元 KL，温度 tau）
        ├── DFL 框分布蒸馏（KL，可选）
        └── 特征蒸馏 CWD（学生经 1x1 对齐层后与教师颈部特征比对）

    trainer 验证/推理调用 self.model(tensor) → 直接走学生 predict

教师有两种来源：
    * online：每个 step 在线推理教师（默认，L40S/L20 推荐），支持微批次分块以省显存；
    * cache ：从 scripts/cache_teacher.py 预生成的分片读取（1080 Ti 推荐）。

设计说明：不使用 forward hook——ultralytics 8.4.145 的 Detect 头训练态输出 dict 自带
"feats"（三个颈部输入特征），eval 态输出 (decoded, dict)，直接取用即可。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ultralytics.nn.distill_model import DistillationModel as _BaseDistillModel
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import copy_attr

from .cache import TEACHER_KEY, load_cache_meta
from .losses import CWDLoss, binary_logit_kd, box_dist_kd


def _pred_dict(out) -> dict:
    """Detect 头 eval 态返回 (y, preds_dict)，train 态直接返回 preds_dict。"""
    if isinstance(out, tuple):
        out = out[1]
    return out


class DistillDetModel(_BaseDistillModel):
    """教师-学生蒸馏包装模型（检测任务）。

    继承官方 ``DistillationModel`` 仅用于通过 trainer 的 ``isinstance`` 检查
    （框架据此把 ``teacher_model.`` 加入冻结名单、保存时剥离 EMA 中的教师副本）；
    __init__/loss/forward 等均为自有实现（无 hook、CWD+KL+自适应权重）。
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_weights: str | Path | None = None,
        mode: str = "online",
        cache_dir: str | Path | None = None,
        teacher_batch: int = 4,
    ):
        # 显式跳过官方 __init__（它会注册 hook、构建 MLP projector）
        nn.Module.__init__(self)
        assert mode in {"online", "cache"}, f"teacher_mode 只支持 online/cache，收到 {mode}"
        self.mode = mode
        self.teacher_batch = max(int(teacher_batch), 1)

        # ---- 学生 ----
        self.student_model = student_model
        device = next(student_model.parameters()).device
        imgsz = int(student_model.args.imgsz)
        ch_in = int(student_model.yaml.get("channels", 3))
        student_reg_max = int(student_model.model[-1].reg_max)
        student_nc = int(student_model.model[-1].nc)

        # ---- 教师 ----
        self.teacher_model: nn.Module | None = None
        teacher_feat_channels: list[int] | None = None
        if mode == "online":
            assert teacher_weights, "online 模式必须提供 teacher_weights（如 yolov8l.pt）"
            teacher_model = load_checkpoint(str(teacher_weights))[0]
            teacher_model = teacher_model.to(device)
            self.teacher_model = teacher_model
            self._freeze_teacher()
        else:
            assert cache_dir, "cache 模式必须提供 cache_dir"
            meta = load_cache_meta(cache_dir)
            self._check_meta(meta, imgsz, student_nc, student_reg_max)
            teacher_feat_channels = meta["feat_channels"]

        # ---- 探针前向，拿到师生颈部特征通道 ----
        student_model.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, ch_in, imgsz, imgsz, device=device)
            s_feats = _pred_dict(student_model(dummy))["feats"]
            student_channels = [int(f.shape[1]) for f in s_feats]
            if mode == "online":
                t_feats = _pred_dict(self.teacher_model(dummy))["feats"]
                teacher_feat_channels = [int(f.shape[1]) for f in t_feats]
        student_model.train()
        assert teacher_feat_channels is not None

        # ---- 1x1 特征对齐层（方案 4.3.3）：学生通道 → 教师通道 ----
        self.projector = nn.ModuleList(
            [nn.Conv2d(cs, ct, kernel_size=1, stride=1, padding=0, bias=True)
             for cs, ct in zip(student_channels, teacher_feat_channels)]
        ).to(device)

        # ---- 损失算子 ----
        self.cwd = CWDLoss(tau=float(getattr(student_model.args, "cwd_tau", 4.0))).to(device)

        # ---- 训练进度（自适应权重 + warmup 用）----
        self._epoch = 0
        self._epochs = 1
        self._nb = 1
        self._step = 0

        # 复制学生的 stride/names/nc/yaml/model 等属性，使本包装器对外"就是"学生
        # （trainer / validator 会访问 .stride / .model[-1] / .names 等）
        copy_attr(self, student_model)

    # ------------------------------------------------------------------ #
    # 基础工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_meta(meta: dict, imgsz: int, nc: int, reg_max: int) -> None:
        for k, got, expect in [
            ("imgsz", meta.get("imgsz"), imgsz),
            ("nc", meta.get("nc"), nc),
            ("reg_max", meta.get("reg_max"), reg_max),
        ]:
            if got is not None and int(got) != int(expect):
                raise ValueError(f"教师缓存 meta 的 {k}={got} 与学生 {expect} 不一致，请用相同配置重新生成缓存。")

    def _freeze_teacher(self) -> None:
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()  # 任何时候教师都保持 eval/frozen
        return self

    # 父类的 pickle 钩子引用了我们没有的 hook 字典，这里覆盖为默认行为
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_progress(self, epoch: int, epochs: int, nb: int) -> None:
        """每个 epoch 开始由 trainer 回调注入训练进度。"""
        self._epoch = int(epoch)
        self._epochs = max(int(epochs), 1)
        self._nb = max(int(nb), 1)
        self._step = 0

    def _args(self, key, default):
        return getattr(self.student_model.args, key, default)

    # ------------------------------------------------------------------ #
    # 前向协议
    # ------------------------------------------------------------------ #
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):  # 训练 / 训练中验证
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)  # 纯推理走学生

    def fuse(self, verbose: bool = True, imgsz: int | list = 640):
        """导出/推理时丢弃教师与对齐层，只返回融合后的学生。"""
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    def set_head_attr(self, **kwargs):
        self.student_model.set_head_attr(**kwargs)

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value):
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    # ------------------------------------------------------------------ #
    # 教师推理
    # ------------------------------------------------------------------ #
    def _run_teacher_online(self, img: torch.Tensor) -> dict:
        """在线教师：微批次分块推理，拼接全部输出（4060 8GB 也能跑）。"""
        self._freeze_teacher()
        scores, boxes, feats = [], [], []
        for x in torch.split(img, self.teacher_batch, dim=0):
            with torch.no_grad():
                pred = _pred_dict(self.teacher_model(x))
            scores.append(pred["scores"])
            boxes.append(pred["boxes"])
            feats.append(pred["feats"])
        return {
            "scores": torch.cat(scores, dim=0),
            "boxes": torch.cat(boxes, dim=0),
            "feats": [torch.cat([b[i] for b in feats], dim=0) for i in range(len(feats[0]))],
        }

    @staticmethod
    def _prep_cached(teacher_cache: dict, device: torch.device, dtype: torch.dtype) -> dict:
        # 缓存固定 fp16：amp=False 时学生是 fp32、amp=True 时学生处于 autocast fp16，
        # 统一转成与学生 logits 相同的 dtype，避免 KL 损失里出现 dtype 不一致
        return {
            "scores": teacher_cache["scores"].to(device=device, dtype=dtype, non_blocking=True),
            # 缓存生成时可用 --no-boxes/--no-feats 裁剪体积，此处允许为 None / 空
            "boxes": (
                teacher_cache["boxes"].to(device=device, dtype=dtype, non_blocking=True)
                if teacher_cache.get("boxes") is not None
                else None
            ),
            "feats": [
                f.to(device=device, dtype=dtype, non_blocking=True) for f in teacher_cache.get("feats", [])
            ],
        }

    # ------------------------------------------------------------------ #
    # 损失
    # ------------------------------------------------------------------ #
    def loss(self, batch: dict, preds=None):
        device = batch["img"].device
        bs = int(batch["img"].shape[0])

        if not self.training:
            # 训练中的验证只算检测损失，蒸馏项记 0（与 ultralytics 官方 DistillationModel 行为一致）。
            # validator 已先做过推理（eval 态返回 (y, preds_dict)），直接复用避免重复前向。
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            # 原生 loss_items 的值是 0 维标量，这里必须同为 0 维，否则 validator 累加时报形状错误
            z = torch.zeros((), device=device)
            loss_items["kd_cls_loss"] = z
            loss_items["kd_box_loss"] = z
            loss_items["kd_feat_loss"] = z
            loss_items["kd_weight"] = z
            return regular_loss, loss_items

        zero = torch.zeros(1, device=device)

        # 学生前向（train 态 Detect 直接返回 dict）
        s_preds = self.student_model(batch["img"])
        regular_loss, loss_items = self.student_model.loss(batch, s_preds)

        # ---- 教师输出 ----
        teacher_cache = batch.pop(TEACHER_KEY, None)
        with torch.no_grad():
            if teacher_cache is not None:
                t_preds = self._prep_cached(teacher_cache, device, s_preds["scores"].dtype)
            elif self.mode == "online":
                t_preds = self._run_teacher_online(batch["img"])
            else:
                raise RuntimeError("cache 模式但 batch 中没有教师缓存，请检查 DistillCacheLoader 是否生效。")

        # ---- 形状自检（第一批）----
        self._assert_compatible(s_preds, t_preds)

        # ---- Logit 对齐到 (B, A, nc) / (B, A, 4*reg_max) ----
        s_scores = s_preds["scores"].permute(0, 2, 1).contiguous()
        t_scores = t_preds["scores"].permute(0, 2, 1).contiguous()
        s_boxes = s_preds["boxes"].permute(0, 2, 1).contiguous()

        reg_max = int(self.student_model.model[-1].reg_max)
        tau = float(self._args("kd_tau", 4.0))
        g_cls = float(self._args("kd_cls_gain", 4.0))
        g_box = float(self._args("kd_box_gain", 1.0))
        g_feat = float(self._args("kd_feat_gain", 8.0))

        l_cls = binary_logit_kd(s_scores, t_scores, tau=tau)
        if g_box > 0 and t_preds["boxes"] is not None:
            t_boxes = t_preds["boxes"].permute(0, 2, 1).contiguous()
            l_box = box_dist_kd(s_boxes, t_boxes, reg_max=reg_max, tau=tau)
        else:
            l_box = zero

        # ---- 特征对齐后做 CWD（裁剪缓存或权重为 0 时跳过）----
        if g_feat > 0 and len(t_preds["feats"]) == len(self.projector):
            s_feats_aligned = [proj(f) for proj, f in zip(self.projector, s_preds["feats"])]
            l_feat = self.cwd(s_feats_aligned, t_preds["feats"])
        else:
            l_feat = zero

        # ---- 自适应蒸馏权重（方案 4.3.1）----
        # 学生整体置信度低 → 多听教师；置信度高 → 多信真值标签
        if bool(self._args("kd_adaptive", True)):
            conf = s_scores.detach().sigmoid().amax(dim=-1).mean().clamp(0.0, 1.0)
            gamma = float(self._args("kd_adaptive_gamma", 1.0))
            w_adapt = (1.0 - conf).pow(gamma)
        else:
            w_adapt = torch.ones((), device=device)

        # ---- 蒸馏 warmup 线性爬升（前 warmup_epochs 轮 0→1，稳定初期检测训练）----
        warmup = float(self._args("kd_warmup_epochs", 1.0))
        if warmup > 0:
            ramp = min(1.0, (self._epoch + self._step / self._nb) / warmup)
        else:
            ramp = 1.0
        self._step += 1

        distill = (g_cls * l_cls + g_box * l_box + g_feat * l_feat) * (w_adapt * ramp)
        distill_scaled = (distill * bs).reshape(1)  # 与检测损失一样乘 batch size，保持梯度尺度一致

        loss_items["kd_cls_loss"] = (g_cls * l_cls).detach()
        loss_items["kd_box_loss"] = (g_box * l_box).detach()
        loss_items["kd_feat_loss"] = (g_feat * l_feat).detach()
        loss_items["kd_weight"] = (w_adapt * ramp).detach()

        return torch.cat([regular_loss, distill_scaled]), loss_items

    def _assert_compatible(self, s_preds: dict, t_preds: dict) -> None:
        if getattr(self, "_checked", False):
            return
        if s_preds["scores"].shape != t_preds["scores"].shape:
            raise RuntimeError(
                f"师生分类 logits 形状不一致：student {tuple(s_preds['scores'].shape)} "
                f"vs teacher {tuple(t_preds['scores'].shape)}。请确认同 nc / imgsz / reg_max，"
                f"且 cache 模式已关闭 multi_scale。"
            )
        if t_preds["boxes"] is not None and s_preds["boxes"].shape != t_preds["boxes"].shape:
            raise RuntimeError(
                f"师生框分布 logits 形状不一致：student {tuple(s_preds['boxes'].shape)} "
                f"vs teacher {tuple(t_preds['boxes'].shape)}。"
            )
        if len(s_preds["feats"]) != len(t_preds["feats"]):
            raise RuntimeError("师生特征层数不一致。")
        for i, (fs, ft) in enumerate(zip(s_preds["feats"], t_preds["feats"])):
            if fs.shape[-2:] != ft.shape[-2:]:
                raise RuntimeError(
                    f"第 {i} 层特征空间分辨率不一致：student {tuple(fs.shape)} vs teacher {tuple(ft.shape)}，"
                    f"cache 模式请确认 imgsz 与生成缓存时相同且关闭了 multi_scale。"
                )
        self._checked = True
