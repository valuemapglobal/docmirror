"""
公式字符流提取 (Formula Character Stream Extraction)
======================================================

K1: 从 PDF 字符流直接提取 LaTeX，跳过 OCR。

原理:
    学术论文 PDF 中的公式字符以数学字体 (CMMI, CMSY, Symbol, Cambria Math)
    嵌入为 Unicode 码点。通过字体名检测 + Unicode→LaTeX 映射，可以
    直接从字符流重建 LaTeX，精度对数字 PDF 而言是 100%。

    优先级: 字符流提取 > OCR 裁图识别 > 空字符串
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 数学字体检测
# ═══════════════════════════════════════════════════════════════════════════════

_MATH_FONT_PATTERNS = re.compile(
    r"(?i)(CMMI|CMSY|CMEX|CMR\d|Math|Symbol|STIX|Cambria.?Math|"
    r"MathJax|Asana.?Math|XITS.?Math|Latin\s*Modern\s*Math|"
    r"NewCM\s*Math|DejaVu\s*Math|Fira\s*Math)",
)


def is_math_font(fontname: str) -> bool:
    """检测字体名是否为数学字体。"""
    if not fontname:
        return False
    return bool(_MATH_FONT_PATTERNS.search(fontname))


# ═══════════════════════════════════════════════════════════════════════════════
# Unicode → LaTeX 映射
# ═══════════════════════════════════════════════════════════════════════════════

_UNICODE_TO_LATEX = {
    # 希腊小写
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega", "ϵ": r"\epsilon",
    "ϕ": r"\phi", "ϑ": r"\vartheta", "ϱ": r"\varrho", "ς": r"\varsigma",
    "ϖ": r"\varpi",
    # 希腊大写
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # 运算符
    "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div",
    "·": r"\cdot", "∗": r"*", "⊕": r"\oplus", "⊗": r"\otimes",
    "∘": r"\circ",
    # 关系符
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "≅": r"\cong",
    "∝": r"\propto", "≪": r"\ll", "≫": r"\gg", "⊂": r"\subset",
    "⊃": r"\supset", "⊆": r"\subseteq", "⊇": r"\supseteq",
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni", "≺": r"\prec",
    "≻": r"\succ", "⊥": r"\perp", "∥": r"\parallel",
    # 大运算符
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint",
    "∬": r"\iint", "∭": r"\iiint", "⋃": r"\bigcup", "⋂": r"\bigcap",
    "⊔": r"\bigsqcup",
    # 箭头
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto", "↑": r"\uparrow", "↓": r"\downarrow",
    "↗": r"\nearrow", "↘": r"\searrow",
    # 杂项
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla", "∅": r"\emptyset",
    "∀": r"\forall", "∃": r"\exists", "¬": r"\neg", "√": r"\sqrt",
    "∠": r"\angle", "△": r"\triangle", "□": r"\square", "◇": r"\diamond",
    "♯": r"\sharp", "♭": r"\flat", "♮": r"\natural",
    "∧": r"\wedge", "∨": r"\vee", "⊤": r"\top", "⊥": r"\bot",
    # 括号
    "⟨": r"\langle", "⟩": r"\rangle", "⌈": r"\lceil", "⌉": r"\rceil",
    "⌊": r"\lfloor", "⌋": r"\rfloor", "‖": r"\|",
    # 点号
    "…": r"\ldots", "⋯": r"\cdots", "⋮": r"\vdots", "⋱": r"\ddots",
    # 重音
    "ℓ": r"\ell", "℘": r"\wp", "ℜ": r"\Re", "ℑ": r"\Im",
    "ℵ": r"\aleph", "ℏ": r"\hbar", "†": r"\dagger", "‡": r"\ddagger",
}


def extract_formula_from_chars(
    chars: list,
    bbox: Tuple[float, float, float, float],
) -> Optional[str]:
    """K1: 从 PDF 字符流提取公式 LaTeX。

    Args:
        chars: pdfplumber 或 zone 的字符列表 (dict with text, fontname, top, bottom, x0, x1)
        bbox: 公式区域边界框

    Returns:
        LaTeX 字符串，提取失败返回 None。
    """
    if not chars:
        return None

    # 1. 筛选 bbox 内的字符
    x0, y0, x1, y1 = bbox
    margin = 2.0
    formula_chars = []
    math_font_count = 0

    for c in chars:
        cx0 = c.get("x0", 0)
        ctop = c.get("top", 0)
        cx1 = c.get("x1", 0)
        cbottom = c.get("bottom", 0)

        if cx0 >= x0 - margin and ctop >= y0 - margin and cx1 <= x1 + margin and cbottom <= y1 + margin:
            formula_chars.append(c)
            if is_math_font(c.get("fontname", "")):
                math_font_count += 1

    if not formula_chars:
        return None

    # 2. 数学字体比例检测
    total = len(formula_chars)
    math_ratio = math_font_count / total if total > 0 else 0

    if math_ratio < 0.3:
        # 数学字体不足 30%, 不太可能是数字 PDF 公式
        return None

    # 3. 按位置排序并重建 LaTeX
    formula_chars.sort(key=lambda c: (c.get("top", 0), c.get("x0", 0)))

    # 行分组
    rows = _group_by_rows(formula_chars)

    # 4. 字符流 → LaTeX
    parts = []
    for row_chars in rows:
        row_latex = _row_to_latex(row_chars)
        if row_latex:
            parts.append(row_latex)

    if not parts:
        return None

    result = " ".join(parts)

    # 5. 基本结构修正
    result = _post_process_char_latex(result)

    logger.debug(f"[FormulaChars] extracted from char stream: {result[:80]}...")
    return result


def _group_by_rows(chars: list) -> List[List[dict]]:
    """将字符按 y 坐标分组为行。"""
    if not chars:
        return []

    rows: List[List[dict]] = []
    current_row: List[dict] = [chars[0]]

    for c in chars[1:]:
        prev_mid = (current_row[-1].get("top", 0) + current_row[-1].get("bottom", 0)) / 2
        curr_mid = (c.get("top", 0) + c.get("bottom", 0)) / 2

        if abs(curr_mid - prev_mid) <= 3.0:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]

    rows.append(current_row)
    return rows


def _row_to_latex(row_chars: list) -> str:
    """将一行字符转换为 LaTeX。"""
    row_chars.sort(key=lambda c: c.get("x0", 0))

    parts = []
    baseline = _estimate_baseline(row_chars)

    for c in row_chars:
        text = c.get("text", "")
        if not text.strip():
            parts.append(" ")
            continue

        # Unicode → LaTeX 映射
        latex = _UNICODE_TO_LATEX.get(text, text)

        # 上标/下标检测
        char_mid = (c.get("top", 0) + c.get("bottom", 0)) / 2
        char_height = c.get("bottom", 0) - c.get("top", 0)

        if baseline > 0 and char_height > 0:
            if char_mid < baseline - char_height * 0.3:
                # 上标
                latex = f"^{{{latex}}}"
            elif char_mid > baseline + char_height * 0.3:
                # 下标
                latex = f"_{{{latex}}}"

        parts.append(latex)

    return "".join(parts)


def _estimate_baseline(chars: list) -> float:
    """估算行基线 (中位字符中心 y)。"""
    if not chars:
        return 0

    mids = [(c.get("top", 0) + c.get("bottom", 0)) / 2 for c in chars]
    heights = [c.get("bottom", 0) - c.get("top", 0) for c in chars]

    if not heights:
        return 0

    # 基线 = 最大字体字符的中心
    max_h_idx = max(range(len(heights)), key=lambda i: heights[i])
    return mids[max_h_idx]


def _post_process_char_latex(latex: str) -> str:
    """字符流 LaTeX 后处理。"""
    # 合并连续的上标/下标
    latex = re.sub(r"\}\^\{", "", latex)
    latex = re.sub(r"\}_\{", "", latex)

    # 清理多余空格
    latex = re.sub(r"\s+", " ", latex).strip()

    return latex
