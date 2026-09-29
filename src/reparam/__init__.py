# -*- coding: utf-8 -*-
"""阶段③ 结构重参数化：训练多分支（RepVGG-style）、部署单路融合。"""

from .rep_c2f import RepBottleneck, RepC2f, convert_c2f_to_rep

__all__ = ["RepBottleneck", "RepC2f", "convert_c2f_to_rep"]
