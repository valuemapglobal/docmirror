"""
表格结构修复引擎 (Table Structure Fix Engine)
=============================================

通用、泛化的表格结构后处理修复模块。
在 OCR/提取之后、最终输出之前执行，修复常见的表格结构缺陷。

4 个独立修复函数 + 1 个统一入口:
    1. merge_split_rows     — 合并被拆分的多行记录
    2. clean_cell_text      — 清理单元格内多余空格/换行
    3. split_concat_cells   — 拆分粘连单元格 (如 余额+账号)
    4. align_row_columns    — 对齐行列数到表头

设计原则:
    - 纯函数, 无状态, 无副作用
    - 每次修复都做安全检查, 不确定时不修改
    - 对空表格/单行表格直接返回
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1: 合并被拆分的多行记录
# ═══════════════════════════════════════════════════════════════════════════════

# 纯时间模式 (HH:MM:SS 或 HH:MM)
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
# 日期模式 (YYYY-MM-DD 或 YYYY.MM.DD 或 YYYY/MM/DD)
_DATE_RE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$")


def merge_split_rows(table: List[List[str]]) -> List[List[str]]:
    """合并被拆分的多行记录。

    规则 (按优先级):
      R1: 行首为纯时间 (HH:MM:SS) 且上一行首列为日期 → 合并时间到日期
      R2: 行大部分列为空 (>60%) 且非汇总行 → 合并非空列到上一行

    泛化: 不依赖特定列名或银行格式。
    """
    if not table or len(table) < 3:
        return table

    header = table[0]
    col_count = len(header)
    result = [header]
    i = 1

    while i < len(table):
        row = table[i]

        # 确保列数一致 (防御)
        if len(row) < col_count:
            row = row + [""] * (col_count - len(row))

        # ── R1: 纯时间行 → 合并到上一行日期 ──
        first_cell = row[0].strip() if row else ""
        if (
            result  # 有上一行
            and len(result) > 1  # 不是表头
            and _TIME_ONLY_RE.match(first_cell)
        ):
            prev_row = list(result[-1])
            prev_first = prev_row[0].strip()

            if _DATE_RE.match(prev_first):
                # 合并: "2025-12-24" + "01:21:34" → "2025-12-24 01:21:34"
                prev_row[0] = f"{prev_first} {first_cell}"
                # 合并其他非空列
                for j in range(1, min(len(row), len(prev_row))):
                    if row[j].strip() and not prev_row[j].strip():
                        prev_row[j] = row[j]
                    elif row[j].strip() and prev_row[j].strip():
                        prev_row[j] = prev_row[j] + " " + row[j]
                result[-1] = prev_row
                i += 1
                continue

        # ── R2: 大部分列为空 → 合并到上一行 ──
        non_empty = sum(1 for c in row if c.strip())
        empty_ratio = 1 - (non_empty / col_count) if col_count > 0 else 0

        if (
            empty_ratio > 0.6
            and len(result) > 1  # 不是表头
            and non_empty > 0  # 不是全空行
            and not _is_summary_row(row)  # 不是汇总行
        ):
            prev_row = list(result[-1])
            for j in range(min(len(row), len(prev_row))):
                if row[j].strip() and not prev_row[j].strip():
                    prev_row[j] = row[j]
                elif row[j].strip() and prev_row[j].strip():
                    # 追加 (对方户名等多行文本)
                    prev_row[j] = prev_row[j] + row[j]
            result[-1] = prev_row
            i += 1
            continue

        result.append(row)
        i += 1

    return result


def _is_summary_row(row: List[str]) -> bool:
    """检测汇总行 (不应被合并)。"""
    text = "".join(str(c) for c in row)
    summary_keywords = ["总收入", "总支出", "合计", "总计", "小计", "本页", "累计"]
    return any(kw in text for kw in summary_keywords)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: 清理单元格内文本
# ═══════════════════════════════════════════════════════════════════════════════

# 中文字符之间的空格 (应移除)
_CJK_SPACE_RE = re.compile(
    r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])"
)


def clean_cell_text(text: str) -> str:
    """清理单元格内多余空格/换行。

    规则:
      - 中文字符之间的空格 → 移除 (PDF 多行文本重组产物)
      - 保留: 英文之间、数字之间、中英之间的空格
      - 首尾空白去除
    """
    if not text or not text.strip():
        return text.strip()

    # 替换换行为空格
    text = text.replace("\n", " ").replace("\r", "")

    # 多次迭代移除中文间空格 (处理连续 "A B C" → "ABC")
    prev = ""
    while prev != text:
        prev = text
        text = _CJK_SPACE_RE.sub(r"\1\2", text)

    return text.strip()


def clean_table_cells(table: List[List[str]]) -> List[List[str]]:
    """对表格所有单元格执行文本清理。"""
    return [
        [clean_cell_text(cell) if isinstance(cell, str) else cell for cell in row]
        for row in table
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3: 拆分粘连单元格
# ═══════════════════════════════════════════════════════════════════════════════

# 数字→字母边界 (如 "110.9731080243CNYFC")
_NUM_ALPHA_BOUNDARY_RE = re.compile(
    r"(\d{1,3}\.\d{2})"           # 金额部分 (如 110.97)
    r"(\d{5,}[A-Z]*\d*)"         # 账号部分 (如 31080243CNYFC0445)
)


def split_concatenated_cells(
    table: List[List[str]],
) -> List[List[str]]:
    """拆分粘连单元格 — 当行列数少于表头时尝试拆分。

    规则:
      - 只在行列数 < 表头列数时触发
      - 检测 数字.数字+数字字母 的粘连模式
      - 拆分后列数应等于表头列数
    """
    if not table or len(table) < 2:
        return table

    header = table[0]
    header_col_count = len(header)
    result = [header]

    for row in table[1:]:
        if len(row) >= header_col_count:
            result.append(row)
            continue

        # 尝试拆分粘连单元格
        deficit = header_col_count - len(row)
        if deficit <= 0:
            result.append(row)
            continue

        new_row = []
        splits_done = 0
        for cell in row:
            if splits_done >= deficit and len(new_row) + (len(row) - len(new_row)) <= header_col_count:
                new_row.append(cell)
                continue

            # 检测金额+账号粘连
            m = _NUM_ALPHA_BOUNDARY_RE.match(str(cell))
            if m and splits_done < deficit:
                new_row.append(m.group(1))  # 金额
                new_row.append(m.group(2))  # 账号
                splits_done += 1
            else:
                new_row.append(cell)

        # 如果拆分后列数匹配, 使用新行
        if len(new_row) == header_col_count:
            result.append(new_row)
        else:
            result.append(row)  # 拆分失败, 保留原行

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4: 对齐行列数
# ═══════════════════════════════════════════════════════════════════════════════

def align_row_columns(table: List[List[str]]) -> List[List[str]]:
    """对齐所有行的列数到表头列数。

    规则:
      - 列数少于表头 → 末尾补空字符串
      - 列数多于表头 → 尾部多余列合并到最后一列
    """
    if not table:
        return table

    header = table[0]
    target = len(header)
    result = [header]

    for row in table[1:]:
        if len(row) == target:
            result.append(row)
        elif len(row) < target:
            result.append(row + [""] * (target - len(row)))
        else:
            # 多余列合并到最后一列
            merged = row[:target - 1] + [" ".join(str(c) for c in row[target - 1:] if c)]
            result.append(merged)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5: 下划线页脚清理
# ═══════════════════════════════════════════════════════════════════════════════

# 匹配 ≥3 个连续下划线 (页脚分隔线)
_UNDERLINE_RE = re.compile(r"_{3,}")


def strip_underline_footer(table: List[List[str]]) -> List[List[str]]:
    """清理单元格中下划线拼接的页脚统计信息。

    模式: "4085.26___支出交易总额:...___收入交易总额:...___合计笔数:..."
    规则: 截断到第一个 ___（保留前面的实际数据值）。

    泛化: 不依赖特定列名, 任何包含 ___ 的单元格都处理。
    """
    for row in table:
        for ci in range(len(row)):
            cell = row[ci] or ""
            m = _UNDERLINE_RE.search(cell)
            if m:
                row[ci] = cell[: m.start()].rstrip()
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6: 裁剪尾部空列
# ═══════════════════════════════════════════════════════════════════════════════


def trim_trailing_empty_columns(table: List[List[str]]) -> List[List[str]]:
    """裁剪全为空的尾部列。

    泛化: 只裁尾部, 不影响中间的空列。
    """
    if not table or not table[0]:
        return table

    col_count = max(len(row) for row in table)
    trim_to = col_count
    for ci in range(col_count - 1, -1, -1):
        all_empty = all(
            not (row[ci] if ci < len(row) else "").strip()
            for row in table
        )
        if all_empty:
            trim_to = ci
        else:
            break

    if trim_to < col_count:
        table = [row[:trim_to] for row in table]
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 7: 纯数字空格合并
# ═══════════════════════════════════════════════════════════════════════════════

# 匹配 "数字 数字" 模式 (中间只有空格, 无字母/汉字)
_DIGIT_SPACE_RE = re.compile(r"^[\d\s]+$")


def merge_digit_spaces(table: List[List[str]]) -> List[List[str]]:
    """合并纯数字单元格中间的空格。

    模式: "6216911304 963684" → "6216911304963684"
    规则: 只对纯数字+空格的 cell 生效, 不影响含字母/汉字的 cell。

    泛化: 自动检测, 不依赖列名, 适用一切银行流水。
    """
    for row in table[1:]:  # 跳过表头
        for ci in range(len(row)):
            cell = (row[ci] or "").strip()
            if cell and _DIGIT_SPACE_RE.match(cell) and " " in cell:
                row[ci] = cell.replace(" ", "")
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 8: 清理粘连在数据值后面的双语列标题
# ═══════════════════════════════════════════════════════════════════════════════

# 常见的双语列标题尾缀 (英文部分) — 按长度降序匹配
_BILINGUAL_SUFFIXES = [
    "Counterparty Institution", "Counterparty Name",
    "Transaction Amount", "Transaction Date",
    "Account Balance", "Abstract Code",
    "Serial Number", "Description",
    "Debit", "Credit",
]
# 构建正则: 匹配 "中文...英文尾缀" 模式
_BILINGUAL_SUFFIX_RE = re.compile(
    r"([\u4e00-\u9fff][\u4e00-\u9fff\s]*)\s*("
    + "|".join(re.escape(s) for s in _BILINGUAL_SUFFIXES)
    + r")\s*$"
)


def strip_header_labels_from_cells(table: List[List[str]]) -> List[List[str]]:
    """清理数据单元格中粘连的双语列标题后缀。

    模式: "0.90借方Debit" → "0.90"
          "浦发银行重庆分行营业部对手机构 Counterparty Institution" → "浦发银行重庆分行营业部"

    规则: 检测 cell 末尾的 "中文+英文" 列标题组合, 截断到中文部分之前。
    泛化: 不依赖特定列, 基于通用双语列标题关键词。
    """
    for row in table[1:]:  # 跳过表头
        for ci in range(len(row)):
            cell = (row[ci] or "").strip()
            if not cell or len(cell) < 5:
                continue
            m = _BILINGUAL_SUFFIX_RE.search(cell)
            if m:
                # 截断到中文列标题开始之前
                row[ci] = cell[: m.start()].rstrip()
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 9: 移除全空表格
# ═══════════════════════════════════════════════════════════════════════════════


def remove_empty_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """移除所有 cell 均为空的表格。

    泛化: 只删全空表格, 不影响有任何数据的表格。
    """
    result = []
    for table in tables:
        has_content = any(
            (cell or "").strip()
            for row in table
            for cell in row
        )
        if has_content:
            result.append(table)
        else:
            logger.debug("[DocMirror] removed empty table")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 11: 拆分粘连在户名开头的账号数字
# ═══════════════════════════════════════════════════════════════════════════════

# ≥10位连续数字 + 中文 (账号粘连户名)
_ACCT_PREFIX_RE = re.compile(r"^(\d{10,})([\u4e00-\u9fff].*)$")


def split_account_from_name(table: List[List[str]]) -> List[List[str]]:
    """拆分粘连在户名列开头的长数字账号。

    模式: "7065018800015镇江一生一世好游戏有限公司" →
          对方账户="7065018800015"  对方户名="镇江一生一世好游戏有限公司"

    规则:
      - 如果 cell 以 ≥10 位连续数字开头, 后接中文 → 拆分
      - 数字部分合并到前一列 (如果前一列表头含 "账户"/"账号")
      - 泛化: 不依赖列名 hard-coding, 基于表头内容匹配
    """
    if not table or len(table) < 2 or len(table[0]) < 2:
        return table

    header = table[0]

    # 找 "对方账户"/"对方账号" 列和其右邻列
    acct_col = None
    for ci, h in enumerate(header):
        h_text = (h or "").strip()
        if ("账户" in h_text or "账号" in h_text) and "对方" in h_text:
            if ci + 1 < len(header):
                acct_col = ci
                break

    if acct_col is None:
        return table

    name_col = acct_col + 1

    for row in table[1:]:
        if name_col >= len(row):
            continue
        cell = (row[name_col] or "").strip()
        m = _ACCT_PREFIX_RE.match(cell)
        if m:
            digits, name = m.group(1), m.group(2)
            # 合并数字到账户列 (prepend, 用空格分隔已有值)
            existing = (row[acct_col] or "").strip()
            row[acct_col] = (digits + " " + existing).strip() if existing else digits
            row[name_col] = name.strip()

    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 12: 剥离货币前缀
# ═══════════════════════════════════════════════════════════════════════════════

# 匹配 "RMB 352.10" 或 "CNY352.10" 或 "USD 1,000.00"
_CURRENCY_PREFIX_RE = re.compile(
    r"^(RMB|CNY|USD|EUR|JPY|HKD|GBP)\s*"
    r"([\-\d,]+\.?\d*)\s*$"
)


def strip_currency_prefix(table: List[List[str]]) -> List[List[str]]:
    """剥离单元格中的货币代码前缀。

    模式: "RMB 352.10" → "352.10", "RMB7.77" → "7.77"
    规则: 只处理 "货币代码+数字" 的纯金额 cell, 不影响含文字的 cell。
    泛化: 支持 RMB/CNY/USD/EUR/JPY/HKD/GBP。
    """
    for row in table[1:]:  # 跳过表头
        for ci in range(len(row)):
            cell = (row[ci] or "").strip()
            if not cell:
                continue
            m = _CURRENCY_PREFIX_RE.match(cell)
            if m:
                row[ci] = m.group(2)
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════════════════

def fix_table_structure(table: List[List[str]]) -> List[List[str]]:
    """表格结构修复统一入口。

    按顺序执行:
      1. 行合并 (日期+时间, 多行单元格)
      2. 粘连单元格拆分 (余额+账号)
      3. 列数对齐
      4. 单元格文本清理
      5. 下划线页脚清理
      6. 尾部空列裁剪
      7. 纯数字空格合并
      8. 双语列标题清理
      9. 账号户名拆分
      10. 货币前缀剥离

    Args:
        table: 原始表格 (二维字符串列表, 第 0 行为表头)。

    Returns:
        修复后的表格。
    """
    if not table or len(table) < 2:
        return table

    original_rows = len(table)

    table = merge_split_rows(table)              # Fix 1
    table = split_concatenated_cells(table)       # Fix 3
    table = align_row_columns(table)              # Fix 4
    table = clean_table_cells(table)              # Fix 2
    table = strip_underline_footer(table)         # Fix 5
    table = trim_trailing_empty_columns(table)    # Fix 6
    table = merge_digit_spaces(table)             # Fix 7
    table = strip_header_labels_from_cells(table) # Fix 8
    table = split_account_from_name(table)        # Fix 11
    table = strip_currency_prefix(table)          # Fix 12
    table = remove_empty_interior_columns(table)  # Fix 13

    fixed_rows = len(table)
    if fixed_rows != original_rows:
        logger.info(
            f"[DocMirror] table_structure_fix: "
            f"{original_rows} → {fixed_rows} rows "
            f"(merged {original_rows - fixed_rows})"
        )

    return table


def remove_empty_interior_columns(table: List[List[str]]) -> List[List[str]]:
    """删除全空的内部列 (含表头也为空或为相邻列重复)。

    交通银行等双行表头场景: 借方发生额/贷方发生额 被 post_process 合并后,
    产生空列 + 重复列名。本函数只删除 **所有数据行都为空** 的列。

    Args:
        table: 修复后的表格, 第 0 行为表头。

    Returns:
        删除空列后的表格。
    """
    if not table or len(table) < 2:
        return table

    n_cols = len(table[0])
    if n_cols <= 1:
        return table

    # 找出所有数据行全为空的列
    empty_cols: set = set()
    for ci in range(n_cols):
        if all(
            not (row[ci] if ci < len(row) else "").strip()
            for row in table[1:]  # skip header
        ):
            empty_cols.add(ci)

    if not empty_cols:
        return table

    # 构建表头出现次数, 用于检测重复
    header_vals = [(table[0][ci] if ci < len(table[0]) else "").strip() for ci in range(n_cols)]
    header_counts: dict = {}
    for h in header_vals:
        header_counts[h] = header_counts.get(h, 0) + 1

    # 删除条件: 数据全空 AND (表头为空 OR 表头是重复值)
    cols_to_remove: set = set()
    for ci in empty_cols:
        hv = header_vals[ci]
        if not hv or header_counts.get(hv, 0) > 1:
            cols_to_remove.add(ci)

    if not cols_to_remove:
        return table

    keep = [ci for ci in range(n_cols) if ci not in cols_to_remove]
    result = []
    for row in table:
        result.append([row[ci] if ci < len(row) else "" for ci in keep])

    logger.debug(
        f"[fix] removed {len(cols_to_remove)} empty interior columns: "
        f"{sorted(cols_to_remove)}"
    )
    return result




