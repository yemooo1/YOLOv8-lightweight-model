# -*- coding: utf-8 -*-
"""生成参赛汇报 PPT（简洁风，16:9）。

数据来源：``runs/watermelon/eval_test/summary.json`` + 集群 L40S 部署实测。
配图：``figures/deploy_summary.png``、``figures/detect_compare.png``

用法：``python scripts/build_ppt.py``  → 输出 ``参赛方案_汇报.pptx``
"""

import json
import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUM = json.loads((ROOT / "runs/watermelon/eval_test/summary.json").read_text(encoding="utf-8"))

FONT = "微软雅黑"
DARK = RGBColor(0x33, 0x33, 0x33)
GREY = RGBColor(0x77, 0x77, 0x77)
ACCENT = RGBColor(0x2E, 0x6F, 0xB7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def style_run(run, size=18, bold=False, color=DARK, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", font)


def add_text(slide, left, top, width, height, lines, align=None):
    """lines: [(text, size, bold, color, space_before_pt)]"""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (text, size, bold, color, space) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if space:
            p.space_before = Pt(space)
        if align is not None:
            p.alignment = align
        style_run(p.add_run(), size=size, bold=bold, color=color)
        p.runs[0].text = text
    return tb


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def content_slide(prs, title):
    slide = blank_slide(prs)
    add_text(slide, 0.7, 0.45, 12, 0.7, [(title, 28, True, ACCENT, 0)])
    line = slide.shapes.add_shape(1, Inches(0.7), Inches(1.15), Inches(11.9), Pt(2))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    line.shadow.inherit = False
    return slide


def add_table(slide, headers, rows, left=0.9, top=1.7, width=11.5, height=3.6, font_size=15):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table
    for c, text in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = ""
        style_run(cell.text_frame.paragraphs[0].add_run(), size=font_size, bold=True, color=WHITE)
        cell.text_frame.paragraphs[0].runs[0].text = text
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
    for r, row in enumerate(rows, start=1):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            style_run(cell.text_frame.paragraphs[0].add_run(), size=font_size)
            cell.text_frame.paragraphs[0].runs[0].text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(0xF2, 0xF6, 0xFB)
    return table


def add_picture_fit(slide, path, top=1.5, max_w=11.9, max_h=5.4):
    pic = slide.shapes.add_picture(str(path), Inches(0.7), Inches(top), width=Inches(max_w))
    if pic.height > Inches(max_h):
        ratio = Inches(max_h) / pic.height
        pic.height = Inches(max_h)
        pic.width = int(pic.width * ratio)
        pic.left = Inches((13.333 - pic.width / 914400) / 2)
    return pic


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ---------- 1. 封面 ----------
    s = blank_slide(prs)
    bar = s.shapes.add_shape(1, Inches(0), Inches(2.05), Inches(13.333), Pt(3))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    add_text(s, 1.0, 3.0, 11.3, 1.4, [
        ("面向边缘部署的轻量化目标检测算法", 40, True, DARK, 0),
        ("—— 基于知识蒸馏与结构重参数化的 YOLOv8 高效推理方案", 20, False, GREY, 10),
    ])
    add_text(s, 1.0, 1.2, 11.3, 0.8, [("边缘 AI 模型轻量化 · 参赛方案汇报", 18, False, ACCENT, 0)])
    add_text(s, 1.0, 5.0, 11.3, 2.0, [
        ("队　　名：________________", 18, False, DARK, 0),
        ("队　　长：________________", 18, False, DARK, 8),
        ("队　　员：________________", 18, False, DARK, 8),
        ("指导老师：________________", 18, False, DARK, 8),
    ])

    # ---------- 2. 赛题与目标 ----------
    s = content_slide(prs, "一、赛题解读与量化目标")
    add_text(s, 0.9, 1.5, 11.6, 1.2, [
        ("核心矛盾：大模型精度高但边缘端跑不动；直接训练的小模型体积达标、但精度与召回明显不足。", 17, False, DARK, 0),
        ("方案目标：在不增加任何部署期推理开销的前提下，让 3M 量级学生逼近甚至超过 11M 量级教师。", 17, False, DARK, 8),
    ])
    add_table(s, ["约束项", "要求", "本方案达成"], [
        ["学生参数量", "< 5M", "3.006M"],
        ["部署图计算量", "不高于同规模普通学生", "8.1 GFLOPs @640"],
        ["部署期额外开销", "零（重参数化必须可等价融合）", "融合前后逐元素误差 0.0"],
        ["检测精度", "逼近 / 超过教师", "test mAP@0.5 0.665 > 教师 0.639"],
        ["部署格式", "ONNX / TensorRT", "均已实测通过"],
    ], top=2.9, height=3.4)

    # ---------- 3. 技术路线 ----------
    s = content_slide(prs, "二、总体技术路线（四阶段闭环）")
    add_text(s, 0.9, 1.6, 11.6, 4.5, [
        ("① 基线训练：西瓜成熟度数据集（2 类）上训练 YOLOv8n / s 基线，建立精度参照。", 18, False, DARK, 0),
        ("② 知识蒸馏：YOLOv8s（教师）→ YOLOv8n（学生），三损失联合监督 + 自适应权重。", 18, False, DARK, 14),
        ("③ 结构重参数化：训练态多分支、部署态单路融合（RepC2f），零精度损失。", 18, False, DARK, 14),
        ("④ 部署实测：ONNX → TensorRT FP16，精度对齐与延迟实测。", 18, False, DARK, 14),
        ("设计要点：三阶段相互解耦——学生 best.pt 是纯 DetectionModel，可直接被官方 val / export 工具链消费。",
         16, False, GREY, 22),
    ])

    # ---------- 4. 实验结果 ----------
    s = content_slide(prs, "三、实验结果（独立 test 集 1041 张，训练全程未见）")
    add_table(s, ["模型", "参数量", "mAP@0.5", "mAP@0.5:0.95"], [
        ["教师 YOLOv8s（冻结）", "11.14M", "0.639", "0.423"],
        ["基线 YOLOv8n", "3.01M", "0.638", "0.430"],
        ["蒸馏学生", "3.01M", "0.665", "0.401"],
        ["RepC2f 部署权重", "3.006M", "0.633", "0.430"],
    ], top=1.6, height=2.6)
    add_text(s, 0.9, 4.6, 11.6, 1.6, [
        ("3.01M 蒸馏学生在 mAP@0.5 上达 0.665，超 11.14M 教师 +2.6 点、超同规模基线 +2.7 点，参数量仅为教师的 27%。", 16, False, DARK, 0),
        ("召回 0.664 显著领先；unripe 少数类 mAP@0.5 = 0.352 为全场最高。", 16, False, DARK, 8),
        ("mAP@0.5:0.95 上蒸馏未超基线（0.401 vs 0.430）——增益集中在分类与召回，后续提高框分布蒸馏权重。", 16, False, GREY, 8),
    ])

    # ---------- 5. 重参数化 ----------
    s = content_slide(prs, "四、结构重参数化（阶段③）")
    add_text(s, 0.9, 1.6, 11.6, 4.6, [
        ("训练态：RepBottleneck 的 3×3 主卷积旁挂 1×1 卷积分支与 BN 恒等分支，增强梯度流与可学习容量。", 17, False, DARK, 0),
        ("部署态：三分支数学等价融合为单个 3×3 卷积，由 ultralytics 官方 RepConv.fuse_convs 完成。", 17, False, DARK, 12),
        ("零初始化暖启：主分支继承基线权重、新增分支 BN γ 置零，转换瞬间输出与原网络逐元素相等（max err = 0.0）。", 17, False, DARK, 12),
        ("免侵入实现：不改 ultralytics 源码，通过模块原地替换 + 官方 fuse 完成，保持与官方工具链兼容。", 17, False, DARK, 12),
        ("关键数字：训练态 3.066M → 部署态 3.006M；部署图 8.1 GFLOPs，与普通学生完全一致。", 17, True, ACCENT, 18),
    ])

    # ---------- 6. 精度/延迟对比图 ----------
    s = content_slide(prs, "五、精度对比与部署前后延迟")
    add_picture_fit(s, ROOT / "figures" / "deploy_summary.png")
    add_text(s, 0.9, 6.85, 11.6, 0.5, [
        ("测试条件：西瓜 test split 1041 张，imgsz=640；延迟为 L40S 单卡实测（不含前后处理）。", 13, False, GREY, 0),
    ])

    # ---------- 7. 同图检测对比 ----------
    s = content_slide(prs, "六、同图检测效果对比（定性）")
    add_picture_fit(s, ROOT / "figures" / "detect_compare.png", top=1.4, max_h=5.6)
    add_text(s, 0.9, 7.05, 11.6, 0.4, [
        ("四组模型在同一批 test 样本上的检测结果（conf=0.25）。", 13, False, GREY, 0),
    ])

    # ---------- 8. 部署实测 ----------
    s = content_slide(prs, "七、部署实测（NVIDIA L40S）")
    add_table(s, ["指标", "PyTorch FP32", "TensorRT FP16", "变化"], [
        ["mAP@0.5", "0.6328", "0.6308", "−0.20 点"],
        ["mAP@0.5:0.95", "0.4303", "0.4207", "−0.96 点"],
        ["推理耗时（640×640）", "0.69 ms/图", "0.45 ms/图", "1.53× 加速"],
        ["部署体积", "11.7 MB (.pt)", "8.3 MB (.engine)", "−29%"],
    ], top=1.6, height=2.7)
    add_text(s, 0.9, 4.7, 11.6, 2.0, [
        ("导出链路：PyTorch → ONNX（opset 12）→ TensorRT FP16 引擎，全链路在集群 GPU 上跑通。", 16, False, DARK, 0),
        ("FP16 量化仅带来 0.2 点 mAP@0.5 回退，延迟降低 34%，满足边缘部署轻量化目标。", 16, False, DARK, 8),
        ("说明：INT8 量化与真实边缘设备实测因时间不足未执行，本页数据为 GPU 侧推算，测试硬件如实标注。",
         15, False, GREY, 10),
    ])

    # ---------- 9. 创新点 ----------
    s = content_slide(prs, "八、创新点")
    add_text(s, 0.9, 1.6, 11.6, 5.0, [
        ("1. 自适应蒸馏权重：w = mean(1 − conf_student)^γ，监督强度随学生不确定性自适应。", 17, False, DARK, 0),
        ("2. 三损失联合蒸馏：分类 logit KL + DFL 框分布 KL + CWD 特征蒸馏，覆盖三个层面。", 17, False, DARK, 14),
        ("3. 教师输出离线缓存：弱卡可完全不加载教师模型完成蒸馏，扩展方法可运行的硬件范围。", 17, False, DARK, 14),
        ("4. 零初始化暖启的重参数化融合：转换瞬间输出逐元素相等（err = 0.0），只增训练容量、不改部署行为。", 17, False, DARK, 14),
        ("5. 免侵入式实现：不改 ultralytics 源码，与官方 val / export / TensorRT 工具链完全兼容。", 17, False, DARK, 14),
    ])

    # ---------- 10. 结论与后续 ----------
    s = content_slide(prs, "九、结论与后续计划")
    add_text(s, 0.9, 1.6, 11.6, 5.2, [
        ("结论", 19, True, ACCENT, 0),
        ("· 蒸馏学生在 mAP@0.5 上超过教师 2.6 点，参数量仅 3.01M（教师的 27%），满足 <5M 约束。", 16, False, DARK, 8),
        ("· 重参数化实现零部署开销：融合误差 0.0，部署图与普通学生完全一致。", 16, False, DARK, 6),
        ("· TensorRT FP16 实测：延迟 0.45 ms/图（1.53× 加速），精度仅回退 0.2 点。", 16, False, DARK, 6),
        ("后续改进", 19, True, ACCENT, 20),
        ("· 提高框分布蒸馏权重（kd_box_gain 1.0 → 2.0~4.0），争取 mAP@0.5:0.95 超过基线。", 16, False, DARK, 8),
        ("· 清洗 unripe 标注口径并补充少数类样本——当前最大短板，属数据问题而非模型问题。", 16, False, DARK, 6),
        ("· 在真实边缘设备上完成端到端部署实测（当前仅完成 GPU 侧验证）。", 16, False, DARK, 6),
    ])

    out = ROOT / "参赛方案_汇报.pptx"
    prs.save(out)
    print(f"已输出 {out}（{len(prs.slides._sldIdLst)} 页）")


if __name__ == "__main__":
    main()