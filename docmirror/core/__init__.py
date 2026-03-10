"""Core extraction layer for MultiModal."""

from .extraction.extractor import CoreExtractor
from .extraction.foundation import FitzEngine, PDFPlumberEngine, OCREngine
from .extraction.pre_analyzer import PreAnalyzer, PreAnalysisResult

__all__ = ["CoreExtractor", "FitzEngine", "PDFPlumberEngine", "OCREngine",
           "PreAnalyzer", "PreAnalysisResult"]

# ── Backward-compatible shims ──────────────────────────────────────────
# Many callers still use flat paths like ``core.extractor``, ``core.text_utils``,
# etc.  Register the canonical subdirectory modules under the old names so
# that  ``from docmirror.core.extractor import ...``  keeps
# working without touching every call-site.
import importlib as _il, sys as _sys

_SHIM_MAP = {
    # extraction/
    f"{__name__}.extractor":       f"{__name__}.extraction.extractor",
    f"{__name__}.foundation":      f"{__name__}.extraction.foundation",
    f"{__name__}.pre_analyzer":    f"{__name__}.extraction.pre_analyzer",
    # layout/
    f"{__name__}.layout_analysis": f"{__name__}.layout.layout_analysis",
    f"{__name__}.layout_model":    f"{__name__}.layout.layout_model",
    f"{__name__}.graph_router":    f"{__name__}.layout.graph_router",
    # table/
    f"{__name__}.table_merger":    f"{__name__}.table.merger",
    f"{__name__}.table_postprocess": f"{__name__}.table.postprocess",
    f"{__name__}.table_extraction": f"{__name__}.table.extraction",
    # ocr/
    f"{__name__}.ocr_fallback":    f"{__name__}.ocr.fallback",
    # utils/
    f"{__name__}.text_utils":      f"{__name__}.utils.text_utils",
    f"{__name__}.vocabulary":      f"{__name__}.utils.vocabulary",
    f"{__name__}.watermark":       f"{__name__}.utils.watermark",
    # output/
    f"{__name__}.markdown_exporter": f"{__name__}.output.markdown_exporter",
    f"{__name__}.visualizer":      f"{__name__}.output.visualizer",
}
for _old, _new in _SHIM_MAP.items():
    if _old not in _sys.modules:
        try:
            _sys.modules[_old] = _il.import_module(_new)
        except ImportError:
            pass  # optional dependencies (e.g. layout_model) may not be available
