"""
表格后处理 (Table Post-processing)
====================================

从 layout_analysis.py 拆分的表格后处理系统。
包含 post_process_table、_strip_preamble、_fix_header_by_vocabulary、
_clean_cell、_merge_split_rows、_extract_summary_entities 等。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..utils.text_utils import (
    _is_cjk_char, _smart_join, normalize_text, normalize_table, parse_amount,
    _RE_DATE_COMPACT, _RE_DATE_HYPHEN, _RE_TIME, _RE_ONLY_CJK,
)
from ..utils.vocabulary import (
    KNOWN_HEADER_WORDS,
    VOCAB_BY_CATEGORY,
    _is_data_row,
    _is_header_row,
    _is_junk_row,
    _normalize_for_vocab,
    _score_header_by_vocabulary,
    _RE_IS_AMOUNT,
    _RE_IS_DATE,
    _RE_VALID_DATE,
)

logger = logging.getLogger(__name__)

def _extract_preamble_kv(rows: List[List[str]]) -> Dict[str, str]:
    """从 pre-header 行中提取 KV 元数据对。

    规则: 相邻非空单元格满足 (中文标签, 数值/日期) 模式时提取为 KV 对。
    示例行: ['汇出总金额（借方）', None, '3,507,280.66', None, '汇入总金额（贷方）', ...]
            → None 被跳过后 → ['汇出总金额（借方）', '3,507,280.66', '汇入总金额（贷方）', ...]
    """
    kv: Dict[str, str] = {}
    for row in rows:
        # 先过滤掉 None/空格, 得到紧凑的非空 cell 列表
        cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
        i = 0
        while i < len(cells) - 1:
            key = cells[i]
            val = cells[i + 1]
            # key: 非空, 包含汉字, 不像纯数值 / 日期
            # val: 非空, 是金额/日期/纯数字
            if (
                key and val
                and re.search(r"[\u4e00-\u9fff]", key)
                and not _RE_IS_DATE.match(key)
                and not _RE_IS_AMOUNT.match(key.replace(",", ""))
            ):
                clean_val = val.replace(",", "").replace("¥", "").replace(" ", "")
                is_num_or_date = bool(
                    _RE_IS_DATE.search(val) or
                    (_RE_IS_AMOUNT.match(clean_val) if clean_val else False)
                )
                if is_num_or_date:
                    kv[key] = val
                    i += 2  # 跳过 value
                    continue
            i += 1
    return kv


def _strip_preamble(
    rows: List[List[str]],
    confirmed_header: List[str],
    categories: Optional[List[str]] = None,
) -> List[List[str]]:
    """去除续表页开头的重复汇总行和重复表头行。

    Args:
        rows: 待过滤的行列表
        confirmed_header: 已确认的表头行
        categories: vocab 匹配所用的文档类别; 默认为 ["BANK_STATEMENT"]
    """
    if not confirmed_header or not rows:
        return rows

    # 确认表头非空 cell 的集合
    header_cells = {
        _normalize_for_vocab(c).strip()
        for c in confirmed_header
        if c and c.strip()
    }

    if not categories:
        categories = ["BANK_STATEMENT"]

    max_scan = min(10, len(rows))

    # 两阶段扫描:
    # 阶段1: 扫描前 max_scan 行, 找到最后一个 vocab_score >= 3 的行 (重复表头行)
    last_header_idx = -1
    for i in range(max_scan):
        vs = _score_header_by_vocabulary(rows[i], categories=categories)
        if vs >= 3:
            last_header_idx = i

    if last_header_idx >= 0:
        # F-7: 剥离保护 — 最多剥离 5 行
        if last_header_idx > 5:
            logger.warning(
                f"[v2] strip_preamble: vocab header at row {last_header_idx} "
                f"(> 5 rows) — capping to avoid data loss"
            )
            last_header_idx = 5
        logger.debug(
            f"[v2] strip_preamble: skip rows 0-{last_header_idx} "
            f"(vocab repeated header at row {last_header_idx})"
        )
        return rows[last_header_idx + 1:]

    # 阶段2: 无重复表头, 尝试 header-similarity 匹配
    for i in range(max_scan):
        row = rows[i]
        norm_cells = {
            _normalize_for_vocab(c).strip()
            for c in row if c and c.strip()
        }
        if header_cells and norm_cells:
            overlap = len(norm_cells & header_cells) / len(header_cells)
            if overlap >= 0.5:
                logger.debug(
                    f"[v2] strip_preamble: skip rows 0-{i} "
                    f"(header overlap={overlap:.2f})"
                )
                return rows[i + 1:]
        # 一旦遇到真实数据行, 停止相似度检测
        if _is_data_row(row):
            break

    return rows


def post_process_table(
    table_data: List[List[str]],
    confirmed_header: Optional[List[str]] = None,
) -> Tuple[Optional[List[List[str]]], Dict[str, str]]:
    """通用表格后处理 — 无关键词依赖。

    Args:
        table_data: 原始二维表格
        confirmed_header: 已确认的表头 (用于续表 preamble 过滤)

    Returns:
        Tuple of (processed_table, preamble_kv):
            processed_table: 处理后的表格, 或 None
            preamble_kv: 从表头前汇总行提取的 KV 对 (可能为空 dict)
    """
    if not table_data or len(table_data) < 2:
        return table_data, {}

    table_data = normalize_table(table_data)

    # ── 如有 confirmed_header, 先剥离续表页前置汇总行 ──
    if confirmed_header:
        table_data = _strip_preamble(table_data, confirmed_header)
        if not table_data:
            return None, {}

    # ── 词表匹配优先 (BANK_STATEMENT 范围): 在前 10 行中找匹配已知列名最多的行 ──
    _CATEGORIES = ["BANK_STATEMENT"]
    header_row_idx = -1
    best_vocab_score = 0
    for i, row in enumerate(table_data[:10]):
        vs = _score_header_by_vocabulary(row, categories=_CATEGORIES)
        if vs > best_vocab_score:
            best_vocab_score = vs
            header_row_idx = i

    # ── Fallback: 结构启发式 ──
    if best_vocab_score < 3:
        header_row_idx = -1
        for i, row in enumerate(table_data[:5]):
            if _is_header_row(row):
                header_row_idx = i
                break
        if header_row_idx == -1:
            for i, row in enumerate(table_data[1:6], 1):
                if _is_data_row(row):
                    header_row_idx = 0
                    break
            if header_row_idx == -1:
                return table_data, {}

    # ── pre-header 行提取为 KV 元数据 (通过返回值传出, 无全局状态) ──
    preamble_kv: Dict[str, str] = {}
    if header_row_idx > 0:
        preamble_rows = table_data[:header_row_idx]
        preamble_kv = _extract_preamble_kv(preamble_rows)
        if preamble_kv:
            logger.debug(f"[v2] preamble KV extracted: {preamble_kv}")

    header = table_data[header_row_idx]
    data_rows = list(table_data[header_row_idx + 1:])
    # 剥离 header 之后紧跟着的 preamble 行 (汇总行/重复表头), 不论 header 在第几行
    data_rows = _strip_preamble(data_rows, header)

    # ── Fix 2: 先修复粘连表头, 确保后续 _clean_cell 使用正确的列名 ──
    try:
        preliminary = [header] + data_rows
        preliminary = _fix_header_by_vocabulary(preliminary)
        header = preliminary[0]
        data_rows = preliminary[1:]
    except Exception as e:
        logger.debug(f"[v2] header fix rollback: {e}")

    # ── 预过滤: 移除 junk 行和短行, 提取表尾汇总 KV ──
    try:
        clean_rows = []
        tail_junk_rows = []
        for r in data_rows:
            if len(r) < 2:
                continue
            if _is_junk_row(r):
                tail_junk_rows.append(r)
                continue
            clean_rows.append(r)

        # 优化 A: 从表尾 junk 行 (合计/总计) 提取汇总 KV
        if tail_junk_rows:
            tail_kv = _extract_preamble_kv(tail_junk_rows)
            if tail_kv:
                preamble_kv.update(tail_kv)
                logger.debug(f"[v2] tail summary KV: {tail_kv}")

        data_rows = clean_rows
    except Exception as e:
        logger.debug(f"[v2] junk filter rollback: {e}")

    # ── Fix 3: 统一由 _merge_split_rows 处理所有 fragment 合并 ──
    try:
        merged = _merge_split_rows([header] + data_rows)
        header = merged[0]
        data_rows = merged[1:]
    except Exception as e:
        logger.debug(f"[v2] merge_split rollback: {e}")

    # ── 数据行清洗: 列对齐 + 单元格清洗 ──
    result: List[List[str]] = [header]

    for row in data_rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]

        try:
            row = [_clean_cell(cell, col_name) for cell, col_name in zip(row, header)]
        except Exception as e:
            logger.debug(f"[v2] clean_cell rollback: {e}")
        result.append(row)

    return result, preamble_kv


def _find_vocab_words_in_string(
    s: str,
    categories: Optional[List[str]] = None,
) -> List[Tuple[str, int, int]]:
    """贪心最长匹配: 在字符串中找出所有已知表头词及其位置 (NFKC + 繁简归一化)。

    Args:
        s: 待匹配字符串
        categories: 限制匹配的 category 列表; 为 None 时使用全量词表
    """
    s = _normalize_for_vocab(s)
    vocab = (
        frozenset().union(*(VOCAB_BY_CATEGORY.get(c, frozenset()) for c in categories))
        if categories else KNOWN_HEADER_WORDS
    )
    sorted_vocab = sorted(vocab, key=len, reverse=True)

    found: List[Tuple[str, int, int]] = []
    used: set = set()

    for word in sorted_vocab:
        start = 0
        while True:
            idx = s.find(word, start)
            if idx == -1:
                break
            end = idx + len(word)
            if not any(i in used for i in range(idx, end)):
                found.append((word, idx, end))
                used.update(range(idx, end))
            start = idx + 1

    return sorted(found, key=lambda x: x[1])


def _fix_header_by_vocabulary(
    table: List[List[str]],
) -> List[List[str]]:
    """词表驱动的表头修正: 只修复表头列名, 不改变列数和数据行。

    策略: 将表头拼接后用词表匹配找出更多列名,
    然后将匹配到的列名按位置顺序填回原有列。
    """
    if not table or len(table) < 2:
        return table

    header = table[0]
    n_cols = len(header)
    old_score = _score_header_by_vocabulary(header)

    concat = "".join((c or "").strip() for c in header)
    if not concat:
        return table

    found = _find_vocab_words_in_string(concat)

    # Guard 1: 匹配到的词数必须显著多于已有匹配 (标志粘连)
    min_improvement = max(3, old_score + 3) if old_score >= 3 else old_score * 2 + 1
    if len(found) < min_improvement:
        return table
    # Guard 2: 至少 3 个词匹配
    if len(found) < 3:
        return table
    # Guard 3: 词表词必须覆盖拼接串主体 (≥50%)
    # 注: 用去空格后的长度计算, 因为 PDF 中表头列名间常有大量空格
    concat_nospace = concat.replace(" ", "").replace("\u3000", "")
    covered = sum(end - start for _, start, end in found)
    if covered / max(len(concat_nospace), 1) < 0.5:
        return table

    # 只替换表头行, 不动数据行
    new_header = [w for w, _, _ in found]
    if len(new_header) > n_cols:
        new_header = new_header[:n_cols]
    elif len(new_header) < n_cols:
        new_header += header[len(new_header):]

    logger.info(
        f"[v2] vocab header fix: score {old_score}→{len(found)}, "
        f"header {header[:3]}→{new_header[:3]}"
    )

    result = [new_header] + table[1:]
    return result


def _clean_cell(cell: str, col_name: str) -> str:
    """通用单元格清洗 (按列名特征自适应)。"""
    cell = (cell or "").strip()
    if not cell:
        return cell

    col_lower = col_name.lower()

    # ── F-5: 账号/ID 类列保护 — 原样返回，不做格式化 ──
    _ID_KEYWORDS = ["账号", "卡号", "序号", "编号", "凭证", "流水号",
                    "日志号", "account", "储种", "地区"]
    if any(kw in col_lower for kw in _ID_KEYWORDS):
        return cell

    # ── F-4: 日期时间完整保留 ──
    if any(kw in col_lower for kw in ["日期", "时间", "date"]):
        # 先从原始 cell (含空格) 提取时间
        time_match = _RE_TIME.search(cell)

        compact = cell.replace(" ", "")
        date_match = _RE_DATE_HYPHEN.search(compact)
        if not date_match:
            raw_match = _RE_DATE_COMPACT.search(compact)
            if raw_match:
                d = raw_match.group(1)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                date_match = _RE_DATE_HYPHEN.search(date_str)
                # 尝试从紧凑日期后面提取 HHMMSS (如 20250921162345)
                if not time_match:
                    after_date = compact[raw_match.end():]
                    hhmmss = re.match(r"(\d{2})(\d{2})(\d{2})", after_date)
                    if hhmmss:
                        h, m, s = int(hhmmss.group(1)), int(hhmmss.group(2)), int(hhmmss.group(3))
                        if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                            time_match = type('M', (), {'group': lambda self: f"{h:02d}:{m:02d}:{s:02d}"})()

        if date_match:
            # 也尝试从 compact 中找标准时间格式 (HH:MM:SS)
            if not time_match:
                time_match = _RE_TIME.search(compact)
            return f"{date_match.group()} {time_match.group()}" if time_match else date_match.group()

    if any(kw in col_lower for kw in ["金额", "余额", "发生", "amount", "balance"]):
        return parse_amount(cell)

    if any(kw in col_lower for kw in ["币", "currency"]):
        cleaned = _RE_ONLY_CJK.sub("", cell)
        return cleaned if cleaned else cell

    return cell


def _merge_split_rows(table: List[List[str]]) -> List[List[str]]:
    """合并被拆分的行 (F-2 增强版)。"""
    if len(table) < 2:
        return table

    # F-2: 页分隔符/注释行正则
    _RE_ANNOTATION = re.compile(
        r"^[-=─—━]{3,}$|接下页|续[上下]?页|第\d+页.*共|page\s*\d+|^[-=]{5,}",
        re.IGNORECASE,
    )
    _RE_SUMMARY = re.compile(r"合计|共计|总计|小计|期末余额|期初余额")

    def _row_type(row):
        """判断行类型: 'data', 'fragment', 'junk', 'summary'。"""
        row_text = "".join(str(c or "") for c in row).strip()
        if not row_text:
            return "junk"

        # 注释/分隔行
        if _RE_ANNOTATION.search(row_text):
            return "junk"

        # 合计/汇总行
        if _RE_SUMMARY.search(row_text):
            if re.search(r"打印时间|打印日期|操作员", row_text):
                return "junk"
            return "summary"

        first = (row[0] if row else "").strip()
        has_content = any((c or "").strip() for c in row[1:]) if len(row) > 1 else False

        # 空首列 + 有内容 → fragment
        if not first and has_content:
            return "fragment"

        # 日期锚定: 银行流水中每笔交易首行必有日期, 无日期 = 续行
        has_date = any(_RE_VALID_DATE.search(c or "") for c in row)
        if has_date:
            return "data"

        # Fix 3: 金额检测 — 若行含金额且第一列非空, 视为独立数据行
        has_amount = any(
            _RE_IS_AMOUNT.match((c or "").strip().replace(",", "").replace("¥", ""))
            for c in row if (c or "").strip()
        )
        if has_amount and first:
            return "data"

        # F-2: 低填充率行 (<50% 列有值) 且无日期 → fragment
        filled = sum(1 for c in row if (c or "").strip())
        if filled > 0 and filled < len(row) * 0.5:
            return "fragment"

        has_any = any((c or "").strip() for c in row)
        if has_any:
            return "fragment"

        return "data"

    def _is_header(row):
        return not any(
            any(ch.isdigit() for ch in cell)
            for cell in row if cell.strip()
        )

    def _merge_into(target, source):
        for i, cell in enumerate(source):
            val = (cell or "").strip()
            if val and i < len(target):
                existing = (target[i] or "").strip()
                if existing:
                    target[i] = _smart_join(existing, val)
                else:
                    target[i] = val

    # Pass 1: 过滤 junk 行 (注释/分隔符)
    filtered = []
    for row in table:
        rt = _row_type(row)
        if rt != "junk":
            filtered.append(row)

    if len(filtered) < 2:
        return filtered if filtered else table

    # Pass 2: 反向扫描 — header 之下的 fragment 合并到下一个数据行
    result = list(filtered)
    i = len(result) - 1
    while i >= 1:
        if _row_type(result[i]) == "fragment":
            j = i - 1
            while j >= 0 and _row_type(result[j]) == "fragment":
                j -= 1
            if j >= 0 and _is_header(result[j]):
                k = i + 1
                while k < len(result) and _row_type(result[k]) == "fragment":
                    k += 1
                if k < len(result) and _row_type(result[k]) == "data":
                    _merge_into(result[k], result[i])
                    result.pop(i)
        i -= 1

    # Pass 3: 正向扫描 — fragment 合并到前一个数据行
    merged = [result[0]]
    seen_data = False
    for row in result[1:]:
        rt = _row_type(row)
        if rt == "fragment":
            if seen_data and merged and not _is_header(merged[-1]):
                _merge_into(merged[-1], row)
            # else: 首条数据行之前的 fragment → 丢弃 (跨页残留)
        else:
            if rt in ("data", "summary"):
                seen_data = True
            merged.append(row)

    return merged


def _extract_summary_entities(chars: list, out: dict):
    """从 summary zone 的 chars 提取 key-value 对。

    增强: 支持同行多 KV 粘连检测 (如 "户名:XX币种:YY")。
    """
    if not chars:
        return

    row_map = defaultdict(list)
    for c in chars:
        y_key = round(c["top"] / 3) * 3
        row_map[y_key].append(c)

    lines = []
    for y_key in sorted(row_map.keys()):
        row_chars = sorted(row_map[y_key], key=lambda c: c["x0"])
        parts = []
        for i, c in enumerate(row_chars):
            if i > 0 and c["x0"] - row_chars[i - 1]["x1"] > 10:
                parts.append("  ")
            parts.append(c["text"])
        lines.append("".join(parts))

    full = "\n".join(lines)
    for segment in re.split(r'\s{2,}|\n', full):
        segment = segment.strip()
        if not segment:
            continue
        _parse_kv_segment(segment, out)


# 常见 KV key 的模式 (中文短词 + 冒号)
_KV_EMBEDDED_RE = re.compile(
    r"([\u4e00-\u9fff]{2,6})"  # 2~6 个中文字符 (key)
    r"[：:]"                    # 冒号分隔符
)


def _parse_kv_segment(segment: str, out: dict):
    """解析单个 segment 为 KV 对, 支持同行粘连检测。

    例如: "户名:重庆中链农科技有限公司币种:人民币"
    → 户名=重庆中链农科技有限公司, 币种=人民币
    """
    # 尝试多种分隔符: 全角冒号、半角冒号、等号、Tab
    for delim in ["：", ":", "=", "\t"]:
        if delim not in segment:
            continue

        k, v = segment.split(delim, 1)
        k, v = k.strip(), v.strip()
        if not k or not v or len(k) >= 20:
            break

        # ── 检查 v 中是否嵌入了另一个 KV 对 ──
        # Pass 1: 用已知 KV 关键词精确匹配 (高精度)
        split_pos = _find_embedded_kv_by_keywords(v)
        if split_pos is None:
            # Pass 2: 扫描所有 "冒号" 位置, 取冒号前最短的 CJK 词 (泛化)
            split_pos = _find_embedded_kv_by_colon_scan(v)

        if split_pos is not None and split_pos > 0:
            first_value = v[:split_pos].strip()
            rest = v[split_pos:].strip()
            if first_value:
                out[k] = first_value
            if rest:
                _parse_kv_segment(rest, out)
            return

        # 无嵌套, 直接记录
        out[k] = v
        break


# 常见的 KV 关键词 (用于精确匹配嵌入的 key)
_COMMON_KV_KEYWORDS = [
    "币种", "户名", "账号", "卡号", "账户", "类型", "日期",
    "姓名", "编号", "状态", "备注", "摘要", "金额", "余额",
    "开户行", "起止日期", "起始日期", "截止日期", "打印日期",
    "总笔数", "总金额", "页码", "机构",
]


def _find_embedded_kv_by_keywords(v: str) -> "int | None":
    """用已知关键词在 value 中查找嵌入的 key:value 对。"""
    best_pos = None
    for kw in _COMMON_KV_KEYWORDS:
        for delim in ["：", ":"]:
            pattern = kw + delim
            idx = v.find(pattern)
            if idx > 0:  # 必须有前面的 value 部分
                if best_pos is None or idx > best_pos:
                    best_pos = idx  # 取最靠后的匹配
    return best_pos


def _find_embedded_kv_by_colon_scan(v: str) -> "int | None":
    """扫描冒号位置, 检查冒号前是否有 2~4 个中文字符 (疑似 key)。"""
    best_pos = None
    for delim in ["：", ":"]:
        pos = 0
        while True:
            idx = v.find(delim, pos)
            if idx <= 0:
                break
            # 检查冒号前的中文字符数
            cjk_before = 0
            scan = idx - 1
            while scan >= 0 and '\u4e00' <= v[scan] <= '\u9fff':
                cjk_before += 1
                scan -= 1
            # 2~4 个中文字 + 前面有非中文内容 → 可能是嵌入的 key
            if 2 <= cjk_before <= 4 and scan >= 0:
                key_start = idx - cjk_before
                if best_pos is None or key_start > best_pos:
                    best_pos = key_start
            pos = idx + 1
    return best_pos
