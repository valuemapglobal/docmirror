"""
编排层 (Orchestrator)
======================

系统的"大脑" — 负责全流程编排:
    1. 调用 CoreExtractor 生成 BaseResult
    2. 根据 enhance_mode 动态构建中间件管线
    3. 执行管线，收集结果
    4. 桥接输出为 v1 兼容的 ParserOutput

三种增强模式:
    - ``raw``:      仅提取，不增强
    - ``standard``: SceneDetector + EntityExtractor + InstitutionDetector + ColumnMapper + Validator
    - ``full``:     Standard + Repairer

异常降级策略:
    - 中间件失败时默认 skip 继续执行
    - 保证始终返回有效结果 (即使 status="partial")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from ..core.extraction.extractor import CoreExtractor
from ..middlewares.base import BaseMiddleware, MiddlewarePipeline
from ..middlewares.scene_detector import SceneDetector
from ..middlewares.institution_detector import InstitutionDetector
from ..middlewares.entity_extractor import EntityExtractor
from ..middlewares.column_mapper import ColumnMapper
from ..middlewares.validator import Validator
from ..middlewares.repairer import Repairer
from ..middlewares.language_detector import LanguageDetector
from ..middlewares.generic_entity_extractor import GenericEntityExtractor
from ..models.enhanced import EnhancedResult
from ..models.domain import BaseResult
from ..configs.settings import DocMirrorSettings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 中间件注册表 — Open/Closed Principle
# ═══════════════════════════════════════════════════════════════════════════════

MIDDLEWARE_REGISTRY: Dict[str, Type[BaseMiddleware]] = {
    "SceneDetector": SceneDetector,
    "EntityExtractor": EntityExtractor,
    "InstitutionDetector": InstitutionDetector,
    "ColumnMapper": ColumnMapper,
    "Validator": Validator,
    "Repairer": Repairer,
    # ── 跨格式通用中间件 ──
    "LanguageDetector": LanguageDetector,
    "GenericEntityExtractor": GenericEntityExtractor,
}

# 管线配置: enhance_mode → 中间件列表
PIPELINE_CONFIGS: Dict[str, List[str]] = {
    "raw": [],
    "standard": ["SceneDetector", "EntityExtractor", "InstitutionDetector", "ColumnMapper", "Validator"],
    "full": ["SceneDetector", "EntityExtractor", "InstitutionDetector", "ColumnMapper", "Validator", "Repairer"],
}

# 需要 hints 注入的中间件名称
_HINTS_CONSUMERS = {"ColumnMapper"}


# ═══════════════════════════════════════════════════════════════════════════════
# hints.yaml 缓存 (mtime-based)
# ═══════════════════════════════════════════════════════════════════════════════

_hints_cache: Optional[Dict[str, Any]] = None
_hints_mtime: float = 0.0


def _load_hints_cached() -> Dict[str, Any]:
    """加载 hints.yaml 配置 (mtime 缓存)。"""
    global _hints_cache, _hints_mtime
    try:
        import yaml
        hints_path = Path(__file__).resolve().parent.parent / "configs" / "hints.yaml"
        if hints_path.exists():
            mtime = hints_path.stat().st_mtime
            if mtime != _hints_mtime or _hints_cache is None:
                with open(hints_path, "r", encoding="utf-8") as f:
                    _hints_cache = yaml.safe_load(f) or {}
                _hints_mtime = mtime
                logger.debug("[DocMirror] hints.yaml reloaded (mtime changed)")
    except Exception as e:
        logger.debug(f"[DocMirror] Failed to load hints.yaml: {e}")
    return _hints_cache or {}


class Orchestrator:
    """
    MultiModal 编排器 — 全流程管理。

    使用方式::

        orchestrator = Orchestrator()
        result = await orchestrator.run_pipeline(
            file_path=Path("bank_statement.pdf"),
            enhance_mode="full",
        )

        # 获取 v1 兼容输出
        parser_output = result.to_parser_output()
    """

    def __init__(
        self,
        settings: Optional[DocMirrorSettings] = None,
        config: Optional[Dict[str, Any]] = None,
        fail_strategy: Optional[str] = None,
        seal_detector_fn: Optional[Callable] = None,
    ):
        self.settings = settings or DocMirrorSettings.from_env()
        self.config = config or self.settings.to_dict()
        self.extractor = CoreExtractor(seal_detector_fn=seal_detector_fn)
        self.pipeline = MiddlewarePipeline(
            fail_strategy=fail_strategy or self.settings.fail_strategy
        )

    async def run_pipeline(
        self,
        file_path: Path,
        enhance_mode: Literal["raw", "standard", "full"] = "standard",
        file_type: str = "pdf",
        **kwargs,
    ) -> EnhancedResult:
        """
        执行完整解析管线。

        Args:
            file_path:    PDF 文件路径。
            enhance_mode: 增强模式 (raw/standard/full)。

        Returns:
            EnhancedResult: 包含 BaseResult + 增强数据 + Mutations。
        """
        t0 = time.time()

        logger.info(
            f"[DocMirror] Orchestrator ▶ "
            f"file={Path(file_path).name} | mode={enhance_mode}"
        )

        # ═══ Step 1: 核心提取 → BaseResult ═══
        base_result = await self.extractor.extract(file_path)

        # 检查提取结果有效性
        if not base_result.pages and not base_result.full_text:
            error_msg = base_result.metadata.get("error", "Empty extraction result")
            logger.warning(f"[DocMirror] Extraction failed: {error_msg}")
            result = EnhancedResult.from_base_result(base_result)
            result.status = "failed"
            result.add_error(error_msg)
            result.enhanced_data["enhance_mode"] = enhance_mode
            return result

        # ═══ Step 2: 初始化 EnhancedResult ═══
        result = EnhancedResult.from_base_result(base_result)
        result.enhanced_data["enhance_mode"] = enhance_mode

        # ═══ Step 2.5: 策略自适应 (基于 PreAnalyzer) ═══
        pre_analysis = base_result.metadata.get("pre_analysis", {})
        recommended = pre_analysis.get("recommended_strategy", "standard")
        strategy_params = pre_analysis.get("strategy_params", {})
        result.enhanced_data["pre_analysis"] = pre_analysis

        # fast 策略: 降级增强模式
        effective_mode = enhance_mode
        if recommended == "fast" and enhance_mode == "full":
            effective_mode = "standard"
            logger.info("[DocMirror] PreAnalyzer: fast strategy → downgrade full→standard")
        # LLM 启用: 由 strategy_params 驱动
        if strategy_params.get("enable_llm", False):
            self.config.setdefault("SceneDetector", {})["enable_llm"] = True
            self.config.setdefault("Repairer", {})["enable_llm"] = True
            logger.info("[DocMirror] PreAnalyzer: deep strategy → enable LLM middlewares")

        # ═══ Step 3: 构建中间件管线 ═══
        if effective_mode == "raw":
            logger.info("[DocMirror] Raw mode — skipping middleware pipeline")
        else:
            middlewares = self._build_middlewares(effective_mode, file_type)
            result = self.pipeline.execute(middlewares, result)

        # ═══ Step 4: 设置最终状态 ═══
        elapsed = (time.time() - t0) * 1000
        result.processing_time = elapsed

        # ═══ Step 4.5: Mutation 分析 (认知反馈闭环) ═══
        if result.mutations:
            try:
                from .middlewares.mutation_analyzer import MutationAnalyzer
                analyzer = MutationAnalyzer()
                analysis = analyzer.analyze(result.mutations)
                result.enhanced_data["mutation_analysis"] = analysis.to_dict()
            except Exception as e:
                logger.debug(f"[DocMirror] MutationAnalyzer error: {e}")

        # 确保 table block 有内容
        if not base_result.table_blocks:
            if result.status == "success":
                result.status = "partial"
                result.add_error("No tables found in document")

        logger.info(
            f"[DocMirror] Orchestrator ◀ status={result.status} | "
            f"scene={result.scene} | "
            f"mutations={result.mutation_count} | "
            f"elapsed={elapsed:.0f}ms"
        )

        return result

    def _build_middlewares(
        self, enhance_mode: str, file_type: str = "pdf",
    ) -> List[BaseMiddleware]:
        """
        根据格式 + 增强模式构建中间件列表。

        基于注册表模式，新增中间件只需:
          1. 在 MIDDLEWARE_REGISTRY 中注册
          2. 在 configs/pipeline_registry.py 中添加到对应格式+模式列表
        """
        from ..configs.pipeline_registry import get_pipeline_config
        middleware_names = get_pipeline_config(file_type, enhance_mode)
        middlewares = []

        hints = None  # 惰性加载

        for name in middleware_names:
            cls = MIDDLEWARE_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"[DocMirror] Unknown middleware: {name}")
                continue

            mw_config = self.config.get(name, {})

            # 特殊处理: 需要 hints 注入的中间件
            if name in _HINTS_CONSUMERS:
                if hints is None:
                    hints = _load_hints_cached()
                if hints:
                    mw_config["column_aliases"] = hints.get("column_aliases", {})
                    mw_config["hints"] = hints

            middlewares.append(cls(config=mw_config))

        return middlewares
