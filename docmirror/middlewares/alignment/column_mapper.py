"""
列映射中间件 (Column Mapper)
==============================

三层递进式列映射:
    Tier 1 (精确匹配): hints.yaml 中的标准名 + 别名
    Tier 2 (模糊匹配): 编辑距离 + 子串包含 + 同义词
    Tier 3 (LLM N选M): 仅对未匹配列请求 LLM

从 v1 的 processor.py 移植核心映射逻辑和 TARGET_COLUMNS 定义。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 标准化目标列 — 从 YAML 加载 (fallback 到硬编码)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_column_config() -> dict:
    """从 column_aliases.yaml 加载列映射配置。"""
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent / "configs" / "column_aliases.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"[ColumnMapper] Failed to load column_aliases.yaml: {e}, using defaults")
        return {}

_COL_CONFIG = _load_column_config()

TARGET_COLUMNS = _COL_CONFIG.get("target_columns", [
    "序号", "交易时间", "对方账号与户名", "摘要", "用途", "备注",
    "交易金额", "账户余额", "钞汇", "币别",
])

SKIP_COLUMNS = set(_COL_CONFIG.get("skip_columns", [
    "企业流水号", "凭证种类", "凭证号", "交易介质编号",
    "账户明细编号-交易流水号", "凭证类型凭证号码", "凭证类型 凭证号码",
    "凭证类型", "凭证号码", "交易渠道", "交易机构", "对方行名",
    "借贷状态", "支/收", "借贷标志", "收支", "柜员流水号", "摘要代码",
]))

COLUMN_ALIASES: Dict[str, List[str]] = _COL_CONFIG.get("column_aliases", {
    "序号": ["流水号", "序", "编号", "Seq", "No", "SeqNo", "序号"],
    "交易时间": ["交易日期", "日期", "交易日", "记账日期", "入账日期", "Date", "Transaction Date", "记帐日期", "交易时间"],
    "对方账号与户名": ["对方户名", "对方名称", "收/付款人", "对方账户", "Counterparty", "对手方", "对方信息", "交易对手信息", "交易对手", "对方户名/账号", "对方账户/对方银行", "交易机构对方户名/账号"],
    "摘要": ["交易摘要", "摘要", "说明", "用途/摘要", "备注/摘要", "Description", "Summary", "交易类型", "业务摘要", "交易方式", "序号摘要", "序号 摘要"],
    "用途": ["用途", "附言", "Purpose", "Remark", "交易地点/附言", "交易地点"],
    "备注": ["备注", "Memo", "Note", "附注"],
    "交易金额": ["金额", "交易金额", "发生额", "Amount", "收入金额", "支出金额", "人民币", "交易额", "收入", "支出"],
    "账户余额": ["余额", "账户余额", "Balance", "结存", "可用余额", "期末余额", "本次余额", "账户余额现转标志交易渠道", "账户余额 现转标志 交易渠道"],
    "钞汇": ["钞汇标志", "钞汇", "Cash/Transfer", "币别钞汇", "币别 钞汇", "现转标志", "现转"],
    "币别": ["币种", "币别", "Currency", "CCY"],
})

# 借贷分列关键字 (不参与普通列映射，由 split amount 逻辑处理)
INCOME_KEYWORDS = set(_COL_CONFIG.get("income_keywords", ["收入", "贷方", "存入", "收入金额", "贷方发生额", "Credit"]))
EXPENSE_KEYWORDS = set(_COL_CONFIG.get("expense_keywords", ["支出", "借方", "支取", "支出金额", "借方发生额", "Debit"]))
# 发生额类关键字 (用于检测空表头邻接列分列模式)
AMOUNT_LIKE_KEYWORDS = set(_COL_CONFIG.get("amount_like_keywords", ["发生额", "金额", "交易金额", "交易额", "Amount"]))


# ═══════════════════════════════════════════════════════════════════════════════
# Header-Data Alignment & Amount Split (已提取为独立模块)
# ═══════════════════════════════════════════════════════════════════════════════
from .header_alignment import infer_column_type, verify_header_data_alignment
from .amount_splitter import detect_split_amount as _detect_split_amount_fn

# 表头名 → 期望的数据列类型 (供 header_alignment 使用)
_HEADER_TYPE_EXPECTATIONS: Dict[str, str] = {
    "交易时间": "date", "交易日期": "date", "日期": "date",
    "记账日期": "date", "入账日期": "date", "交易日": "date", "Date": "date",
    "交易金额": "amount", "金额": "amount", "发生额": "amount", "Amount": "amount",
    "账户余额": "amount", "余额": "amount", "结存": "amount", "Balance": "amount",
    "序号": "seq", "流水号": "seq",
}
for _std_name, _aliases in COLUMN_ALIASES.items():
    if _std_name in _HEADER_TYPE_EXPECTATIONS:
        _expected_type = _HEADER_TYPE_EXPECTATIONS[_std_name]
        for _alias in _aliases:
            if _alias not in _HEADER_TYPE_EXPECTATIONS:
                _HEADER_TYPE_EXPECTATIONS[_alias] = _expected_type


class ColumnMapper(BaseMiddleware):
    """
    列映射中间件。

    将从 BaseResult 提取的原始表头映射到标准列名。
    输出 ``EnhancedResult.enhanced_data["standardized_tables"]`` — 多表结构。
    """

    def process(self, result: EnhancedResult) -> EnhancedResult:
        """执行列映射并生成标准化表格 (支持多表)。"""
        if result.base_result is None:
            return result

        # 仅对 bank_statement 场景执行完整映射
        if result.scene not in ("bank_statement", "unknown"):
            logger.info(f"[ColumnMapper] scene={result.scene}, skipping bank_statement mapping")
            return result

        # 按机构合并 hints 中的列别名（含 scene column_map）
        effective_aliases = self._effective_column_aliases(result)

        # 获取所有表格块
        table_blocks = result.base_result.table_blocks
        if not table_blocks:
            logger.info("[ColumnMapper] No table blocks found")
            return result

        # ── 多页表格合并: 按顺序分组 ──
        groups = self._merge_table_blocks(table_blocks)
        if not groups:
            return result

        # ── 为每个表组生成标准化结果 ──
        standardized_tables = []

        for idx, group in enumerate(groups):
            raw_headers = group["header"]
            data_rows = group["rows"]

            if not data_rows:
                continue

            # ── 表头-数据列对齐校验 ──
            raw_headers = self._verify_header_data_alignment(
                raw_headers, data_rows, result,
            )
            group["header"] = raw_headers  # 回写，供后续使用

            # 执行三层映射（传入按机构合并后的别名）
            mapping, unmapped = self._map_columns(raw_headers, effective_aliases)

            # 检测收入/支出分列 (F-6: 传入数据行用于验证)
            has_split_amount, split_income_idx, split_expense_idx = (
                self._detect_split_amount(raw_headers, mapping, data_rows)
            )

            # 生成标准化表格
            block_id = group["block"].block_id if group["block"] else f"table_{idx}"
            std_table = self._standardize(
                raw_headers, data_rows, mapping,
                has_split_amount=has_split_amount,
                split_income_idx=split_income_idx,
                split_expense_idx=split_expense_idx,
                block_id=block_id,
                result=result,
            )

            mapped_count = sum(1 for v in mapping.values() if v is not None)

            table_entry = {
                "table_id": f"table_{idx}",
                "headers": std_table[0] if std_table else [],
                "rows": std_table[1:] if std_table else [],
                "row_count": len(std_table) - 1 if std_table else 0,
                "column_mapping": mapping,
                "unmapped_columns": unmapped,
                "has_split_amount": has_split_amount,
                "source_block_id": block_id,
            }
            standardized_tables.append(table_entry)

            logger.info(
                f"[ColumnMapper] table_{idx}: mapped {mapped_count}/{len(raw_headers)} "
                f"columns | rows={table_entry['row_count']} | unmapped={unmapped}"
            )

        result.enhanced_data["standardized_tables"] = standardized_tables

        # 向后兼容: standardized_table = 最大表的完整二维数组
        if standardized_tables:
            main = max(standardized_tables, key=lambda t: t["row_count"])
            result.enhanced_data["standardized_table"] = (
                [main["headers"]] + main["rows"]
            )
            result.enhanced_data["standardized_headers"] = main["headers"]
            result.enhanced_data["column_mapping"] = main["column_mapping"]
            result.enhanced_data["unmapped_columns"] = main["unmapped_columns"]
            result.enhanced_data["raw_headers"] = main["headers"]
            result.enhanced_data["has_split_amount"] = main["has_split_amount"]

        return result

    # 表头质量检测正则 (含日期或金额 → 可能是数据行误做表头)
    _RE_HEADER_IS_DATA = re.compile(
        r'\d{4}[/-]\d{2}[/-]\d{2}'   # 日期
        r'|^\d[\d,]*\.\d{2}$'        # 金额
        r'|^\d{10,}$'                # 长数字串 (账号)
        r'|[：:].{3,}'               # KV 格式 (如 "账号:621460...")
        r'|银行.{0,4}(交易|流水|明细|对账)'  # 标题文字
        r'|\d{6}\*{2,}'              # 掩码账号 (如 621460****)
    )

    def _effective_column_aliases(self, result: EnhancedResult) -> Dict[str, List[str]]:
        """
        合并通用 column_aliases 与机构维度的 scene column_map。
        当 result.enhanced_data["institution"] 存在时，从 hints.scenes 中取
        {institution}_bank_statement 的 column_map，将 raw_header -> standard_name 转为别名列表。
        """
        aliases = dict(self.config.get("column_aliases", {}))
        # 深拷贝一层，避免修改 config
        for k, v in list(aliases.items()):
            if isinstance(v, list):
                aliases[k] = list(v)
            else:
                aliases[k] = [v] if isinstance(v, str) else []

        inst = result.enhanced_data.get("institution")
        hints = self.config.get("hints") or {}
        scenes = hints.get("scenes") or []
        if inst and scenes:
            scene_name = f"{inst}_bank_statement"
            for s in scenes:
                if s.get("name") == scene_name:
                    column_map = s.get("column_map") or {}
                    for raw_name, spec in column_map.items():
                        if isinstance(spec, dict):
                            std = spec.get("standard_name")
                        else:
                            std = None
                        if std and raw_name:
                            aliases.setdefault(std, []).append(raw_name)
                    logger.debug(f"[ColumnMapper] merged aliases from scene={scene_name}")
                    break
        return aliases

    def _merge_table_blocks(self, table_blocks):
        """
        按顺序合并多个 table block (多页续表) — **中间件级分组**。

        与 ``core.table_merger.merge_cross_page_tables`` (Step 4) 互补:
          - **table_merger** (Step 4): 提取阶段合并跨页 Block 的 raw_content
          - **本方法** (Step 7): 列映射阶段做逻辑分组 (header + data rows),
            支持弹性合并、多表分离

        算法 (Sequential Grouping):
            1. 按块顺序遍历
            2. 判断每个块的首行是 表头 还是 数据行 (首列含日期 → 数据行)
            3. 首行是表头 → 检查是否与当前组表头匹配:
               - 匹配 → 跳过重复表头, 数据行加入当前组
               - 不匹配 → 开启新组
            4. 首行是数据行 (续页) → 全部行加入当前组
            5. 返回最大的组
        """
        import re
        _date_re = re.compile(r'^\d{4}[-/]?\d{2}[-/]?\d{2}')

        # 过滤有效表格
        valid = [
            b for b in table_blocks
            if isinstance(b.raw_content, list) and len(b.raw_content) >= 2
        ]
        if not valid:
            valid = [
                b for b in table_blocks
                if isinstance(b.raw_content, list) and len(b.raw_content) >= 1
            ]
            if not valid:
                return None, None
            return valid[0].raw_content, valid[0]

        def _is_data_row(row):
            """判断行是否是数据行 (首列含日期)。"""
            return row and _date_re.match(str(row[0]).strip())

        def _headers_match(h1, h2):
            """
            判断两个表头是否属于同一张表。

            算法: 拼接字符串比较 — 消除列边界差异 (粘连不可拆)。
            例: ['序号摘要', '币别钞汇'] vs ['序号', '摘要', '币别', '钞汇']
              → "序号摘要币别钞汇" == "序号摘要币别钞汇" → 同一张表
            """
            if not h1 or not h2:
                return False
            # 去空格拼接: 消除列边界差异
            s1 = "".join(str(c).strip() for c in h1 if str(c).strip())
            s2 = "".join(str(c).strip() for c in h2 if str(c).strip())
            if not s1 or not s2:
                return False
            # 精确匹配 (覆盖所有粘连场景)
            if s1 == s2:
                return True
            # 容忍微小差异 (OCR 偶尔丢字/多字)
            if len(s1) > 5 and len(s2) > 5:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, s1, s2).ratio()
                return ratio >= 0.85
            return False

        # ── 顺序分组 ──
        # 每个 group = { "header": [...], "rows": [...], "block": first_block }
        groups = []
        current_group = None

        for block in valid:
            first_row = block.raw_content[0]

            if _is_data_row(first_row):
                # ── 续页: 首行是数据, 加入当前组 ──
                if current_group is None:
                    # 没有当前组 → 无法确定表头, 跳过
                    logger.debug("[ColumnMapper] skip orphan continuation block (no header group)")
                    continue

                col_diff = abs(len(first_row) - len(current_group["header"]))
                if col_diff <= 3: # Changed from 2 to 3
                    for row in block.raw_content:
                        padded = self._pad_row(row, len(current_group["header"]))
                        current_group["rows"].append(padded)
                    logger.info(
                        f"[ColumnMapper] continuation: +{len(block.raw_content)}r "
                        f"into group (header='{current_group['header'][0]}')"
                    )
                else:
                    logger.debug(
                        f"[ColumnMapper] skip cont block: col diff {col_diff} > 2"
                    )
            else:
                # ── 首行是表头 ──
                if current_group is not None and _headers_match(
                    current_group["header"], first_row
                ):
                    # 重复表头 → 跳过表头行, 数据行加入当前组
                    for row in block.raw_content[1:]:
                        padded = self._pad_row(row, len(current_group["header"]))
                        current_group["rows"].append(padded)
                    logger.info(
                        f"[ColumnMapper] repeat header merge: +{len(block.raw_content)-1}r"
                    )
                elif current_group is not None and len(block.raw_content) > 2:
                    # 表头不匹配但列数兼容 → 弹性合并 (检查数据行含日期)
                    col_diff = abs(len(first_row) - len(current_group["header"]))
                    if col_diff <= 2:
                        sample_rows = block.raw_content[1:4]
                        date_hits = sum(1 for r in sample_rows if _is_data_row(r))
                        if date_hits >= 2:
                            for row in block.raw_content[1:]:
                                padded = self._pad_row(row, len(current_group["header"]))
                                current_group["rows"].append(padded)
                            logger.info(
                                f"[ColumnMapper] elastic merge: +{len(block.raw_content)-1}r "
                                f"(col_diff={col_diff}, date_hits={date_hits})"
                            )
                            continue

                    # 真正不同的表 → 开启新组
                    groups.append(current_group)
                    current_group = {
                        "header": list(first_row),
                        "rows": list(block.raw_content[1:]),
                        "block": block,
                    }
                    logger.info(
                        f"[ColumnMapper] new table group: header='{first_row[:3]}' "
                        f"({len(block.raw_content)-1}r)"
                    )
                else:
                    # 没有当前组 或 块太小不值得弹性合并 → 开启新组
                    if current_group is not None:
                        groups.append(current_group)
                    current_group = {
                        "header": list(first_row),
                        "rows": list(block.raw_content[1:]),
                        "block": block,
                    }

        # 最后一组
        if current_group is not None:
            groups.append(current_group)

        if not groups:
            return []

        if len(groups) > 1:
            logger.info(
                f"[ColumnMapper] {len(groups)} table groups found: "
                + ", ".join(
                    f"group{i}({len(g['rows'])}r)" for i, g in enumerate(groups)
                )
            )
        else:
            logger.info(
                f"[ColumnMapper] 1 table group: {len(groups[0]['rows'])} data rows"
            )

        return groups

    @staticmethod
    def _pad_row(row, target_len):
        """补齐或截断行到目标长度。"""
        row = list(row)
        if len(row) < target_len:
            row += [""] * (target_len - len(row))
        elif len(row) > target_len:
            row = row[:target_len]
        return row

    def _map_columns(
        self,
        raw_headers: List[str],
        extra_aliases: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Dict[str, Optional[str]], List[str]]:
        """
        三层递进式列映射。

        Args:
            raw_headers: 原始表头列表
            extra_aliases: 标准名 -> 别名列表（含 hints + 机构 scene column_map）；None 时用 config。

        Returns:
            (mapping, unmapped)
            mapping: {raw_header: standard_name or None}
            unmapped: 未映射的原始列名列表
        """
        mapping: Dict[str, Optional[str]] = {}
        used_targets: Set[str] = set()
        unmapped: List[str] = []

        if extra_aliases is None:
            extra_aliases = self.config.get("column_aliases", {})
        # 统一为 standard_name -> list of aliases
        normalized: Dict[str, List[str]] = {}
        for k, v in (extra_aliases or {}).items():
            normalized[k] = list(v) if isinstance(v, list) else ([v] if isinstance(v, str) else [])
        extra_aliases = normalized

        for raw_h in raw_headers:
            raw_clean = raw_h.strip()
            if not raw_clean:
                mapping[raw_h] = None
                continue

            # 跳过已知的非标准列
            if raw_clean in SKIP_COLUMNS:
                mapping[raw_h] = None
                unmapped.append(raw_clean)
                continue

            # 跳过借贷分列 (由 split amount 逻辑单独处理)
            if raw_clean in INCOME_KEYWORDS or raw_clean in EXPENSE_KEYWORDS:
                mapping[raw_h] = None
                continue

            # ── Tier 1: 精确匹配 ──
            target = self._tier1_exact(raw_clean, used_targets, extra_aliases)
            if target:
                mapping[raw_h] = target
                used_targets.add(target)
                continue

            # ── Tier 2: 模糊匹配 ──
            target = self._tier2_fuzzy(raw_clean, used_targets)
            if target:
                mapping[raw_h] = target
                used_targets.add(target)
                continue

            # ── Tier 2.5: 粘连列名子串匹配 ──
            # 处理 char-level 提取导致的列名粘连, 如 "序号交易日期" 包含 "交易日期"
            target = self._tier25_merged(raw_clean, used_targets)
            if target:
                mapping[raw_h] = target
                used_targets.add(target)
                continue

            # ── Tier 3: LLM (预留) ──
            mapping[raw_h] = None
            unmapped.append(raw_clean)

        return mapping, unmapped

    def _tier1_exact(
        self,
        raw: str,
        used: Set[str],
        extra_aliases: Dict[str, List[str]],
    ) -> Optional[str]:
        """精确匹配: 标准名 + 别名。"""
        # 直接匹配 TARGET_COLUMNS
        for target in TARGET_COLUMNS:
            if target not in used and raw == target:
                return target

        # 别名匹配
        all_aliases = dict(COLUMN_ALIASES)
        for t, aliases in extra_aliases.items():
            if t in all_aliases:
                all_aliases[t] = list(set(all_aliases[t] + aliases))
            else:
                all_aliases[t] = aliases

        for target, aliases in all_aliases.items():
            if target in used:
                continue
            for alias in aliases:
                if raw == alias:
                    return target

        return None

    def _tier2_fuzzy(self, raw: str, used: Set[str]) -> Optional[str]:
        """
        模糊匹配: 子串包含 + 编辑距离。

        优化: 优先子串包含 (高精度)，再用编辑距离兜底。
        """
        best_target = None
        best_score = 0.0

        for target, aliases in COLUMN_ALIASES.items():
            if target in used:
                continue

            # 子串包含
            all_candidates = [target] + aliases
            for cand in all_candidates:
                # 短词护栏: ≤3字符的候选词要求高匹配率, 避免误匹配
                if len(cand) <= 3 or len(raw) <= 3:
                    if cand == raw:
                        score = 1.0
                    elif cand in raw and len(cand) >= len(raw) * 0.5:
                        score = len(cand) / len(raw)
                    elif raw in cand and len(raw) >= len(cand) * 0.5:
                        score = len(raw) / len(cand)
                    else:
                        continue
                elif cand in raw or raw in cand:
                    score = len(min(cand, raw, key=len)) / len(max(cand, raw, key=len))
                else:
                    continue
                if score > best_score:
                    best_score = score
                    best_target = target

            # 编辑距离 (仅对短串)
            if len(raw) <= 10:
                for cand in all_candidates:
                    if len(cand) <= 10:
                        dist = self._edit_distance(raw, cand)
                        max_len = max(len(raw), len(cand))
                        score = 1.0 - dist / max_len if max_len > 0 else 0.0
                        if score > best_score:
                            best_score = score
                            best_target = target

        return best_target if best_score >= 0.6 else None

    def _tier25_merged(self, raw: str, used: Set[str]) -> Optional[str]:
        """
        粘连列名子串匹配。

        处理 char-level 提取中多个列名粘连为一个字符串的场景。
        例如: "序号交易日期" 包含 "交易日期", "凭证种类借方发生额" 包含 "借方发生额"

        策略: 优先匹配最长的候选 (避免短词误命中)
        """
        if len(raw) <= 4:
            # 太短不做粘连拆分
            return None

        best_target = None
        best_len = 0

        # 检查标准列名及其别名是否为 raw 的子串
        for target, aliases in COLUMN_ALIASES.items():
            if target in used:
                continue
            all_candidates = [target] + aliases
            for cand in all_candidates:
                if len(cand) >= 2 and cand in raw and len(cand) > best_len:
                    best_len = len(cand)
                    best_target = target

        # 也检查 INCOME/EXPENSE 关键字
        if best_target is None:
            for kw in INCOME_KEYWORDS | EXPENSE_KEYWORDS:
                if kw in raw and len(kw) > best_len:
                    # 不映射到标准列, 交给 split amount 处理
                    return None

        if best_target and best_len >= 2:
            logger.debug(
                f"[ColumnMapper] tier2.5 merged match: "
                f"'{raw}' → '{best_target}' (substr len={best_len})"
            )

        return best_target

    # ═══════════════════════════════════════════════════════════════════════════
    # Header-Data Alignment — 委托给 header_alignment 模块
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _infer_column_type(
        data_rows: List[List[str]], col_idx: int, sample_size: int = 30,
    ) -> Dict[str, float]:
        """推断单列数据类型分布 — 委托给 header_alignment 模块。"""
        return infer_column_type(data_rows, col_idx, sample_size)

    def _verify_header_data_alignment(
        self,
        headers: List[str],
        data_rows: List[List[str]],
        result: "EnhancedResult",
    ) -> List[str]:
        """验证表头与数据列对齐 — 委托给 header_alignment 模块。"""
        return verify_header_data_alignment(
            headers=headers,
            data_rows=data_rows,
            header_type_expectations=_HEADER_TYPE_EXPECTATIONS,
            mutation_recorder=result,
            middleware_name=self.name,
        )

    def _detect_split_amount(
        self, headers: List[str], mapping: Dict[str, Optional[str]],
        data_rows: Optional[List[List[str]]] = None,
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """检测收入/支出分列 — 委托给 amount_splitter 模块。"""
        return _detect_split_amount_fn(
            headers=headers,
            mapping=mapping,
            income_keywords=INCOME_KEYWORDS,
            expense_keywords=EXPENSE_KEYWORDS,
            amount_like_keywords=AMOUNT_LIKE_KEYWORDS,
            data_rows=data_rows,
        )

    def _standardize(
        self,
        raw_headers: List[str],
        data_rows: List[List[str]],
        mapping: Dict[str, Optional[str]],
        has_split_amount: bool,
        split_income_idx: Optional[int],
        split_expense_idx: Optional[int],
        block_id: str,
        result: EnhancedResult,
    ) -> List[List[str]]:
        """
        生成标准化表格 — 保留原始列名和全部数据。

        仅做两项标准化:
            1. 收入/支出分列合并为单一带符号金额列
            2. 日期/金额单元格清洗
        """
        # ── 建立表头 ──
        out_headers = list(raw_headers)

        # 记录列映射 mutation (仅记录, 不用于重建表结构)
        for rh, target in mapping.items():
            if target and rh != target:
                result.record_mutation(
                    middleware_name=self.name,
                    target_block_id=block_id,
                    field_changed="column_name",
                    old_value=rh,
                    new_value=target,
                    confidence=0.9,
                    reason="column_mapping",
                )

        # ── 收入/支出分列定位 ──
        income_idx = split_income_idx
        expense_idx = split_expense_idx

        if has_split_amount and income_idx is None:
            for i, h in enumerate(raw_headers):
                h_clean = h.strip()
                if h_clean in INCOME_KEYWORDS:
                    income_idx = i
                elif h_clean in EXPENSE_KEYWORDS:
                    expense_idx = i

        # ── 动态查找日期/金额/余额列索引 (用于单元格清洗) ──
        date_col_idx = self._find_col_idx(raw_headers, mapping, "交易时间")
        amount_col_idx = self._find_col_idx(raw_headers, mapping, "交易金额")
        balance_col_idx = self._find_col_idx(raw_headers, mapping, "账户余额")

        # ── 构建数据行 ──
        std_rows = [out_headers]

        for row in data_rows:
            out_row = [str(cell).strip() if cell else "" for cell in row]
            # 补齐或截断到表头长度
            if len(out_row) < len(out_headers):
                out_row += [""] * (len(out_headers) - len(out_row))
            elif len(out_row) > len(out_headers):
                out_row = out_row[:len(out_headers)]

            # 合并收入/支出到金额列
            if has_split_amount and income_idx is not None and expense_idx is not None:
                if amount_col_idx is not None:
                    target_idx = amount_col_idx
                else:
                    # 没有现成金额列 → 用收入列位置
                    target_idx = income_idx

                income_val = row[income_idx].strip() if income_idx < len(row) else ""
                expense_val = row[expense_idx].strip() if expense_idx < len(row) else ""

                income_num = self._parse_amount(income_val)
                expense_num = self._parse_amount(expense_val)

                if expense_num and abs(expense_num) > 0.001:
                    out_row[target_idx] = f"-{abs(expense_num):.2f}"
                elif income_num and abs(income_num) > 0.001:
                    out_row[target_idx] = f"{income_num:.2f}"
                else:
                    out_row[target_idx] = "0.00"

                # Fix B: 清空原始借贷列 (非 target 列), 防止同时出现收入+支出
                if target_idx != income_idx and income_idx < len(out_row):
                    out_row[income_idx] = ""
                if target_idx != expense_idx and expense_idx < len(out_row):
                    out_row[expense_idx] = ""

            # 单元格清洗: 日期/金额列
            out_row = self._clean_std_row(out_row, date_col_idx, amount_col_idx, balance_col_idx)

            std_rows.append(out_row)

        return std_rows

    # ── 日期/金额正则 (编译一次复用) ──
    _RE_DATE = re.compile(r'(\d{4}[-/.]?\d{2}[-/.]?\d{2})')
    _RE_AMOUNT = re.compile(r'^([+-]?\d[\d,]*\.?\d*)')

    @staticmethod
    def _find_col_idx(
        headers: List[str],
        mapping: Dict[str, Optional[str]],
        target_name: str,
    ) -> Optional[int]:
        """通过 column_mapping 反查原始列索引。"""
        for i, h in enumerate(headers):
            if mapping.get(h) == target_name:
                return i
            if h.strip() == target_name:
                return i
        return None

    def _clean_std_row(
        self,
        row: List[str],
        date_idx: Optional[int],
        amount_idx: Optional[int],
        balance_idx: Optional[int],
    ) -> List[str]:
        """
        按列类型清洗行 (动态索引)。

        - 日期列: 只保留日期部分 (YYYY-MM-DD)
        - 金额/余额列: 只保留数值部分
        """
        # Fix A: 日期列 — 提取 YYYY-MM-DD 并保留时间部分
        if date_idx is not None and date_idx < len(row) and row[date_idx]:
            m = self._RE_DATE.search(row[date_idx])
            if m:
                d = m.group(1).replace("/", "-").replace(".", "-")
                if len(d) == 8 and "-" not in d:
                    d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                # 保留日期后的时间 (HH:MM 或 HH:MM:SS)
                after = row[date_idx][m.end():]
                time_m = re.search(r'\d{2}:\d{2}(?::\d{2})?', after)
                if time_m:
                    d = f"{d} {time_m.group()}"
                row[date_idx] = d

        # 金额列 & 余额列: 只保留数值
        for idx in (amount_idx, balance_idx):
            if idx is not None and idx < len(row):
                val = row[idx].strip()
                if val:
                    cleaned = val.replace(",", "").replace("，", "").replace("¥", "")
                    m = self._RE_AMOUNT.match(cleaned)
                    if m:
                        try:
                            row[idx] = f"{float(m.group(1)):.2f}"
                        except ValueError:
                            row[idx] = ""
                    else:
                        row[idx] = ""

        return row

    @staticmethod
    def _is_number(s: str) -> bool:
        """检查字符串是否为数字。"""
        try:
            float(s.replace(",", "").replace("，", ""))
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _parse_amount(s: str) -> Optional[float]:
        """解析金额字符串为 float，失败返回 None。"""
        if not s or not s.strip():
            return None
        try:
            return float(s.strip().replace(",", "").replace("，", ""))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein 编辑距离。"""
        if len(s1) < len(s2):
            return ColumnMapper._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[-1]
