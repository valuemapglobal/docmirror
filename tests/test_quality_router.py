"""
Unit tests for the Adaptive Quality Router and Benchmark Framework.
"""

import pytest
import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# AdaptiveQualityRouter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class FakeZone:
    """Minimal zone stub for testing."""
    def __init__(self, zone_type="text", confidence=0.9, bbox=(0, 0, 100, 100)):
        self.type = zone_type
        self.confidence = confidence
        self.bbox = bbox


class TestAdaptiveQualityRouter:
    def _make_router(self, params=None):
        from docmirror.core.extraction.quality_router import AdaptiveQualityRouter
        return AdaptiveQualityRouter(params)

    def test_default_returns_rule(self):
        """Empty strategy_params → current behavior (rule-based)."""
        router = self._make_router()
        zone = FakeZone("text", 0.9)
        strategy = router.recommend(zone, page_has_text=True)
        assert strategy.extract_method == "rule"
        assert strategy.skip_extraction is False

    def test_high_confidence_text_uses_rule(self):
        """High-confidence text zone on digital page → rule-based."""
        router = self._make_router({"ocr_dpi": [200]})
        zone = FakeZone("text", 0.95)
        strategy = router.recommend(zone, page_has_text=True)
        assert strategy.extract_method == "rule"

    def test_low_confidence_table_uses_enhanced(self):
        """Low-confidence table zone → ocr_enhanced."""
        router = self._make_router({"ocr_dpi": [200, 300]})
        zone = FakeZone("data_table", 0.5)
        strategy = router.recommend(zone, page_has_text=True)
        assert strategy.extract_method == "ocr_enhanced"
        assert strategy.ocr_dpi == 300

    def test_formula_low_quality_uses_enhanced(self):
        """Formula zone with low page quality → ocr_enhanced."""
        router = self._make_router({"ocr_dpi": [200, 300]})
        zone = FakeZone("formula", 0.6)
        strategy = router.recommend(zone, page_has_text=True, page_quality=50)
        assert strategy.extract_method == "ocr_enhanced"

    def test_scanned_page_routing(self):
        """Scanned page → OCR with appropriate DPI."""
        router = self._make_router({"ocr_dpi": [150, 200, 300]})
        zone = FakeZone("text", 0.5)
        strategy = router.recommend(
            zone, page_has_text=False, is_scanned_page=True, page_quality=45
        )
        assert strategy.extract_method == "ocr_enhanced"
        assert strategy.ocr_dpi == 300

    def test_scanned_watermark_mid_quality(self):
        """Scanned page mid-quality → watermark separation enabled."""
        router = self._make_router({"ocr_dpi": [200], "skip_watermark_filter": False})
        zone = FakeZone("text", 0.7)
        strategy = router.recommend(
            zone, page_has_text=False, is_scanned_page=True, page_quality=70
        )
        assert strategy.enable_watermark_separation is True

    def test_should_enhance_table_low_confidence(self):
        """Low extraction confidence + high empty ratio → re-extract."""
        router = self._make_router()
        table = [["a", "b", "c"], ["", "", ""], ["", "", ""], ["x", "", ""]]
        assert router.should_enhance_table(table, 0.4) is True

    def test_should_enhance_table_high_confidence(self):
        """High extraction confidence → no re-extract."""
        router = self._make_router()
        table = [["a", "b"], ["1", "2"]]
        assert router.should_enhance_table(table, 0.9) is False

    def test_none_strategy_params_safe(self):
        """None strategy_params should not crash."""
        router = self._make_router(None)
        zone = FakeZone("data_table", 0.8)
        strategy = router.recommend(zone, page_has_text=True)
        assert strategy.extract_method in ("rule", "ocr_standard", "ocr_enhanced")


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkMetrics:
    def test_cer_identical(self):
        from tests.benchmark.metrics import compute_cer
        assert compute_cer("hello world", "hello world") == 0.0

    def test_cer_empty_gt(self):
        from tests.benchmark.metrics import compute_cer
        assert compute_cer("", "") == 0.0
        assert compute_cer("abc", "") == 1.0

    def test_cer_partial(self):
        from tests.benchmark.metrics import compute_cer
        cer = compute_cer("helo world", "hello world")
        assert 0.0 < cer < 0.2  # one char difference

    def test_teds_identical(self):
        from tests.benchmark.metrics import compute_teds
        table = [["a", "b"], ["1", "2"]]
        assert compute_teds(table, table) == 1.0

    def test_teds_different(self):
        from tests.benchmark.metrics import compute_teds
        pred = [["a", "b"], ["1", "2"]]
        gt = [["x", "y", "z"], ["3", "4", "5"]]
        score = compute_teds(pred, gt)
        assert 0.0 < score < 1.0

    def test_reading_order_identical(self):
        from tests.benchmark.metrics import compute_reading_order_accuracy
        assert compute_reading_order_accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_reading_order_reversed(self):
        from tests.benchmark.metrics import compute_reading_order_accuracy
        score = compute_reading_order_accuracy([2, 1, 0], [0, 1, 2])
        assert score == 0.0  # fully discordant


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Matcher Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridMatcher:
    def test_exact_match(self):
        from tests.benchmark.hybrid_matcher import hybrid_match
        assert hybrid_match("hello", "hello") is True

    def test_unicode_normalization(self):
        from tests.benchmark.hybrid_matcher import normalize_unicode
        assert normalize_unicode("α") == "a"
        assert normalize_unicode("∫") == "\\int"

    def test_latex_equivalence(self):
        from tests.benchmark.hybrid_matcher import is_latex_equivalent
        assert is_latex_equivalent("\\dfrac{1}{2}", "\\frac{1}{2}") is True
        assert is_latex_equivalent("\\left(x\\right)", "(x)") is True

    def test_fuzzy_match(self):
        from tests.benchmark.hybrid_matcher import fuzzy_segment_match
        # Same text with minor difference
        assert fuzzy_segment_match(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
        ) is True

    def test_hybrid_end_to_end(self):
        from tests.benchmark.hybrid_matcher import hybrid_match
        # Unicode variant should match
        assert hybrid_match("α + β = γ", "a + b = g") is True
        # Completely different → no match
        assert hybrid_match("hello world", "goodbye moon") is False
