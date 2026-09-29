# -*- coding: utf-8 -*-
"""阶段② 知识蒸馏模块包。"""

from .model import DistillDetModel
from .trainer import DISTILL_DEFAULTS, DistillDetectionTrainer

__all__ = ["DistillDetModel", "DistillDetectionTrainer", "DISTILL_DEFAULTS"]
