# -*- coding: utf-8 -*-
"""
阶段② · 蒸馏训练器
==================

对 ultralytics DetectionTrainer 的最小侵入扩展（只重写 5 个钩子方法）：

    get_model              构建「学生 + 教师 + 对齐层」DistillDetModel
    set_model_attributes   数据集 nc/names/args 同时写到学生与包装器
    get_dataloader         cache 模式下训练集改读教师缓存分片
    save_model             best/last 只保存学生（纯 DetectionModel 检查点，
                           可直接被 val_model.sh / ONNX / TensorRT 使用）
    plot_training_labels   cache 模式无原始 labels，跳过

其余训练循环（SGD/cosine、warmup、AMP、mosaic、EMA、checkpoint 选择）完全复用
阶段① 的官方实现，保证 8 组基线与蒸馏实验的训练配方一致、可复现。
"""

from __future__ import annotations

from copy import deepcopy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .cache import DistillCacheDataset, DistillCacheLoader
from .model import DistillDetModel
from ..reparam.rep_c2f import convert_c2f_to_rep

# 蒸馏专属超参默认值（也可在命令行覆盖）
DISTILL_DEFAULTS = dict(
    teacher_weights="",       # online 模式的教师权重路径，如 yolov8l.pt
    teacher_mode="online",    # online | cache
    teacher_batch=4,          # 教师微批次（4060 8GB 建议 2~4）
    cache_dir="",             # cache 模式分片目录
    kd_tau=4.0,               # KL 蒸馏温度
    kd_cls_gain=4.0,          # 分类 logit KL 权重
    kd_box_gain=1.0,          # DFL 框分布 KL 权重（0 关闭）
    kd_feat_gain=8.0,         # CWD 特征蒸馏权重（0 关闭）
    cwd_tau=4.0,              # CWD softmax 温度
    kd_adaptive=True,         # 是否启用置信度自适应蒸馏权重
    kd_adaptive_gamma=1.0,    # 自适应指数：w=(1-conf)^gamma
    kd_warmup_epochs=1.0,     # 蒸馏损失前 N 轮线性爬升
    use_reparam=False,        # 阶段③：C2f→RepC2f 重参数化（训练多分支，部署 fuse 单路）
)


class DistillDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=..., overrides=None, _callbacks=None):
        if cfg is ...:
            from ultralytics.utils import DEFAULT_CFG
            cfg = DEFAULT_CFG
        super().__init__(cfg, overrides, _callbacks)
        # 蒸馏专属参数挂到 args（框架 cfg 不认识这些键，不能走 overrides）
        for k, v in DISTILL_DEFAULTS.items():
            setattr(self.args, k, v)
        if self.args.teacher_mode == "cache":
            if not self.args.cache_dir:
                raise ValueError("teacher_mode=cache 时必须指定 cache_dir")
            if float(self.args.multi_scale) > 0:
                # 缓存分辨率在生成时已固定，多尺度训练会导致师生特征空间尺寸不一致
                self.args.multi_scale = 0.0
        if self.args.compile:
            raise ValueError("蒸馏训练与 torch.compile 不兼容（冻结教师需要 find_unused_parameters）。")
        self.add_callback("on_train_epoch_start", self._on_distill_epoch_start)

    # -------------------------------------------------------------- #
    # 模型构建
    # -------------------------------------------------------------- #
    def get_model(self, cfg=None, weights=None, verbose=True):
        student = DetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        student = self.set_model_names_for_load(student)
        if weights:
            student.load(weights)  # 兼容 yolov8n.pt / 阶段① best.pt / 蒸馏产出的学生 best.pt
        if getattr(self.args, "use_reparam", False):
            # 必须在 load 预训练权重之后：先按 C2f 名字对齐迁移，再原地换 RepC2f（新分支零初始化）
            convert_c2f_to_rep(student, verbose=verbose and RANK == -1)
        student.args = self.args  # 包装器构造时要读 imgsz 与蒸馏超参
        model = DistillDetModel(
            student_model=student,
            teacher_weights=self.args.teacher_weights or None,
            mode=self.args.teacher_mode,
            cache_dir=self.args.cache_dir or None,
            teacher_batch=self.args.teacher_batch,
        )
        n_student = sum(p.numel() for p in student.parameters())
        n_proj = sum(p.numel() for p in model.projector.parameters())
        n_teacher = (
            sum(p.numel() for p in model.teacher_model.parameters()) if model.teacher_model is not None else 0
        )
        self._log_param_budget(n_student, n_proj, n_teacher)
        return model

    @staticmethod
    def _log_param_budget(n_student: int, n_proj: int, n_teacher: int) -> None:
        from ultralytics.utils import LOGGER
        LOGGER.info(
            f"蒸馏模型参数：学生 {n_student / 1e6:.2f}M + 1x1对齐层 {n_proj / 1e6:.2f}M "
            f"= {(n_student + n_proj) / 1e6:.2f}M（目标 <5M）"
            + (f"；教师 {n_teacher / 1e6:.2f}M（冻结，不计入部署体积）" if n_teacher else "；教师走离线缓存")
        )

    def set_model_attributes(self):
        wrapper = unwrap_model(self.model)
        student = wrapper.student_model
        student.nc = self.data["nc"]
        student.names = self.data["names"]
        student.args = self.args
        # copy_attr 在包装器构造时复制过同名属性，这里同步刷新
        wrapper.nc = student.nc
        wrapper.names = student.names
        wrapper.args = self.args

    # -------------------------------------------------------------- #
    # cache 模式的数据通道
    # -------------------------------------------------------------- #
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "train" and self.args.teacher_mode == "cache":
            dataset = DistillCacheDataset(self.args.cache_dir)
            cache_bs = int(dataset.meta.get("batch") or 0)
            if cache_bs and int(batch_size) != cache_bs:
                raise ValueError(
                    f"训练 batch_size={batch_size} 与教师缓存 meta 中的 batch={cache_bs} 不一致："
                    f"缓存分片是按固定 batch 落盘的，请用 --batch {cache_bs}，或重新生成缓存。"
                )
            workers = min(int(self.args.workers), 4)
            return DistillCacheLoader(dataset, num_workers=workers, batch_size=cache_bs or batch_size)
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def plot_training_labels(self):
        if self.args.teacher_mode == "cache":
            return  # 缓存分片不含完整训练 labels
        super().plot_training_labels()

    def get_validator(self):
        # DetectionValidator 会把 args 整体当 overrides 重新做 get_cfg 合法性校验，
        # 蒸馏专属键官方 cfg 不认识 → 构造 validator 前临时摘除，构造后恢复
        custom = {k: getattr(self.args, k) for k in DISTILL_DEFAULTS if hasattr(self.args, k)}
        for k in custom:
            delattr(self.args, k)
        try:
            validator = super().get_validator()
        finally:
            for k, v in custom.items():
                setattr(self.args, k, v)
        return validator

    # -------------------------------------------------------------- #
    # epoch 回调：切换缓存 epoch、注入蒸馏进度
    # -------------------------------------------------------------- #
    def _on_distill_epoch_start(self, trainer=None):
        wrapper = unwrap_model(self.model)
        if self.args.teacher_mode == "cache":
            self.train_loader.dataset.set_epoch(self.epoch)
        nb = len(self.train_loader)
        wrapper.set_progress(epoch=self.epoch, epochs=self.epochs, nb=nb)

    # -------------------------------------------------------------- #
    # 检查点：只落学生权重
    # -------------------------------------------------------------- #
    def save_model(self):
        ema_wrapper = self.ema.ema
        if hasattr(ema_wrapper, "student_model"):
            # 序列化为纯 DetectionModel：best.pt 可直接 val/export，不携带教师与对齐层。
            # 对齐层只服务训练；学生颈部自身结构未变，推理完全不依赖它。
            self.ema.ema = ema_wrapper.student_model
            try:
                return super().save_model()
            finally:
                self.ema.ema = ema_wrapper
        return super().save_model()
