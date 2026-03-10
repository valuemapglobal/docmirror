"""
VLM 输出后处理 + Ensemble 融合 (VLM Postprocess + Ensemble)
============================================================

解析 VLM (Qwen2.5-VL) 的 Markdown 输出，并与 Pipeline 提取结果融合。

核心逻辑:
    1. parse_vlm_markdown:  将 VLM Markdown 解析为 Block 列表
    2. ensemble_results:    VLM + Pipeline 结果择优融合
    3. vlm_markdown_to_benchmark_md: VLM Markdown → OmniDocBench 评测格式

融合策略 (Hybrid):
    - 数字 PDF: Pipeline 文本 (100% 准确) + VLM 表格/公式
    - 扫描件:   VLM 优先 (Pipeline OCR 质量低)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VLM Markdown 清理
# ═══════════════════════════════════════════════════════════════════════════════

def clean_vlm_markdown(md: str) -> str:
    """清理 VLM 输出的 Markdown，去除常见瑕疵。

    VLM 常见问题:
      - 输出 ```markdown ... ``` 包裹
      - 多余的解释文字
      - 思考过程 <think>...</think>
      - 空行过多
    """
    if not md:
        return ""

    # 去除 <think>...</think> 块 (Qwen3.5 的思考过程)
    md = re.sub(r"<think>.*?</think>", "", md, flags=re.DOTALL)

    # 去除 ```markdown ... ``` 包裹
    md = re.sub(r"^```(?:markdown|md|html)?\s*\n", "", md)
    md = re.sub(r"\n```\s*$", "", md)

    # 去除开头的解释性文字 (如 "Here is the converted markdown:")
    lines = md.split("\n")
    while lines and _is_meta_line(lines[0]):
        lines.pop(0)

    # 压缩连续空行 (3+ → 2)
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md.strip()


def _is_meta_line(line: str) -> bool:
    """判断是否为 VLM 添加的元信息行。"""
    line = line.strip().lower()
    if not line:
        return False
    meta_patterns = [
        "here is", "here's", "below is", "the converted",
        "i've converted", "i have converted",
        "markdown output", "markdown format",
    ]
    return any(p in line for p in meta_patterns)


# ═══════════════════════════════════════════════════════════════════════════════
# VLM Markdown → OmniDocBench 评测格式
# ═══════════════════════════════════════════════════════════════════════════════

def vlm_markdown_to_benchmark_md(vlm_md: str) -> str:
    """将 VLM Markdown 直接转换为 OmniDocBench 评测格式。

    OmniDocBench 评测脚本期望:
      - 文本: 纯 Markdown 段落
      - 表格: HTML <table> 格式 (用于 TEDS 评分)
      - 公式: $...$ 或 $$...$$ LaTeX (用于 CDM 评分)
      - 标题: # ## ### (用于结构评分)

    VLM 的输出通常已经符合这个格式, 只需清理。
    """
    md = clean_vlm_markdown(vlm_md)

    # 确保 Markdown 表格转为 HTML (某些 VLM 可能输出 Markdown 表格)
    md = _convert_md_tables_to_html(md)

    return md


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown 表格 → HTML 表格转换
# ═══════════════════════════════════════════════════════════════════════════════

# Markdown 表格分隔行: | --- | --- | 或 |:---:|:---|---:|
_MD_TABLE_SEP_RE = re.compile(r"^\|[\s:]*-+[\s:]*(\|[\s:]*-+[\s:]*)*\|?\s*$")


def _convert_md_tables_to_html(md: str) -> str:
    """将 Markdown 表格转为 HTML <table>。

    VLM 可能输出 Markdown 格式的表格而非 HTML，需要转换以获得 TEDS 评分。
    """
    lines = md.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        # 检测 Markdown 表格开始
        if i + 1 < len(lines) and _MD_TABLE_SEP_RE.match(lines[i + 1].strip()):
            # 收集整个表格
            table_lines = [lines[i]]  # 表头行
            i += 2  # 跳过分隔行
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            # 转换
            html = _md_table_lines_to_html(table_lines)
            result_lines.append(html)
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines)


def _md_table_lines_to_html(table_lines: List[str]) -> str:
    """将 Markdown 表格行列表转为 HTML。"""
    if not table_lines:
        return ""

    def parse_row(line: str) -> List[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [cell.strip() for cell in line.split("|")]

    html_parts = ["<table>"]

    # 表头
    header_cells = parse_row(table_lines[0])
    html_parts.append("<thead><tr>")
    for cell in header_cells:
        html_parts.append(f"<th>{cell}</th>")
    html_parts.append("</tr></thead>")

    # 数据行
    if len(table_lines) > 1:
        html_parts.append("<tbody>")
        for line in table_lines[1:]:
            cells = parse_row(line)
            html_parts.append("<tr>")
            for cell in cells:
                html_parts.append(f"<td>{cell}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody>")

    html_parts.append("</table>")
    return "".join(html_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble 融合
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_page_markdown(
    vlm_md: Optional[str],
    pipeline_md: Optional[str],
    has_text_layer: bool = True,
) -> str:
    """融合 VLM 和 Pipeline 的 Markdown 输出。

    融合策略:
      1. VLM 可用 + 扫描件 → VLM 优先 (Pipeline OCR 质量低)
      2. VLM 可用 + 数字 PDF → VLM 优先但用 Pipeline 验证
      3. VLM 不可用 → Pipeline 兜底

    Args:
        vlm_md: VLM 输出的 Markdown (可能为 None)
        pipeline_md: Pipeline 输出的 Markdown (可能为 None)
        has_text_layer: PDF 是否有文字层

    Returns:
        最终 Markdown 字符串
    """
    vlm_clean = clean_vlm_markdown(vlm_md) if vlm_md else ""
    pipeline_clean = pipeline_md.strip() if pipeline_md else ""

    # 情况 1: VLM 无输出 → Pipeline 兜底
    if not vlm_clean:
        logger.debug("[VLM-Ensemble] VLM 无输出, 使用 Pipeline")
        return pipeline_clean

    # 情况 2: Pipeline 无输出 → VLM
    if not pipeline_clean:
        logger.debug("[VLM-Ensemble] Pipeline 无输出, 使用 VLM")
        return vlm_markdown_to_benchmark_md(vlm_clean)

    # 情况 3: 两者都有 → 择优融合
    vlm_score = _score_markdown_quality(vlm_clean)
    pipeline_score = _score_markdown_quality(pipeline_clean)

    logger.info(
        f"[VLM-Ensemble] VLM={vlm_score:.1f}, Pipeline={pipeline_score:.1f}, "
        f"text_layer={has_text_layer}"
    )

    # 扫描件: VLM 几乎总是更好
    if not has_text_layer:
        return vlm_markdown_to_benchmark_md(vlm_clean)

    # 数字 PDF: VLM 通常表格/公式更好
    # 如果 VLM 得分显著高于 Pipeline, 用 VLM
    if vlm_score > pipeline_score * 0.9:
        return vlm_markdown_to_benchmark_md(vlm_clean)

    # 否则用 Pipeline (数字 PDF 文本层更可靠)
    return pipeline_clean


def _score_markdown_quality(md: str) -> float:
    """简单打分: 评估 Markdown 内容的质量。

    评分维度:
      - 长度 (越长越可能包含更多信息)
      - 结构标记 (表格 HTML, LaTeX 公式, 标题)
      - 无效内容比例 (乱码, 重复)
    """
    if not md:
        return 0.0

    score = 0.0

    # 长度分 (上限 100)
    score += min(len(md) / 50.0, 100.0)

    # 结构分
    if "<table" in md.lower():
        score += 30  # 有 HTML 表格
    if re.search(r"\|.*\|.*\|", md):
        score += 15  # 有 Markdown 表格
    if re.search(r"\$.*?\$", md):
        score += 20  # 有 LaTeX 公式
    if re.search(r"^#{1,3}\s", md, re.MULTILINE):
        score += 10  # 有标题

    # 惩罚: 连续重复行
    lines = md.split("\n")
    unique_lines = set(l.strip() for l in lines if l.strip())
    if len(lines) > 5:
        uniqueness = len(unique_lines) / len(lines)
        if uniqueness < 0.5:
            score *= 0.5  # 大量重复, 减半

    return score
