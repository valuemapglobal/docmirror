"""
VLM 推理引擎 (VLM Inference Engine)
=====================================

基于 Ollama HTTP API 的 VLM 推理引擎。
支持 Qwen2.5-VL 系列模型，用于文档页面理解。

核心 API:
    - page_to_markdown:    页面图片 → 完整 Markdown (文本 + 表格 + 公式)
    - recognize_table:     表格区域图片 → HTML <table>
    - recognize_formula:   公式区域图片 → LaTeX
    - is_available:        检查 Ollama 服务是否可用

设计原则:
    - 异步 HTTP (httpx), 不阻塞事件循环
    - 超时重试: 默认 120s 超时, 失败自动回退到 Pipeline
    - 零侵入: VLM 不可用时管线自动降级, 不影响现有功能
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── 默认配置 ──
_DEFAULT_MODEL = "qwen2.5vl:3b"
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 600  # 秒 (CPU 推理 VLM 较慢, 需要充足时间)
_DEFAULT_TEMPERATURE = 0.1  # 低温度 = 高确定性


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 模板 (针对 OmniDocBench 评测维度优化)
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_PAGE_TO_MARKDOWN = """Convert this document page to Markdown format. Rules:
1. Plain text paragraphs as-is, preserve paragraph structure
2. Headings: use # ## ### for title hierarchy
3. Tables: output as HTML <table><thead><tbody><tr><th><td> with colspan/rowspan if merged cells exist
4. Math formulas: inline $...$ and display $$...$$, output LaTeX
5. Maintain original reading order (left-to-right, top-to-bottom, column by column)
6. Do NOT add any explanation, commentary, or description
7. Output ONLY the converted Markdown content"""

PROMPT_TABLE_TO_HTML = """Convert this table image to HTML format.
Rules:
1. Use <table><thead><tbody><tr><th><td> structure
2. Detect and include colspan/rowspan for merged cells
3. Preserve all cell text exactly as shown
4. Output ONLY the HTML table, no other text"""

PROMPT_FORMULA_TO_LATEX = """Convert this mathematical formula to LaTeX.
Rules:
1. Output ONLY the LaTeX code, no $$ delimiters
2. Use standard LaTeX commands
3. Be precise with subscripts, superscripts, fractions, etc.
4. Output ONLY the LaTeX, no explanation"""


class VLMEngine:
    """Qwen2.5-VL 推理引擎 (via Ollama HTTP API)。

    异步接口, 每次调用发送一张图片 + prompt, 返回模型输出。
    当 Ollama 服务不可用时, 所有方法优雅降级 (返回 None)。

    Usage::

        engine = VLMEngine()
        if engine.is_available():
            md = await engine.page_to_markdown(image_bytes)
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = _DEFAULT_TEMPERATURE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self._available: Optional[bool] = None  # 缓存可用性

        # 确保 localhost 绕过系统代理 (macOS 上 httpx 会拦截)
        import os
        no_proxy = os.environ.get("NO_PROXY", "")
        if "localhost" not in no_proxy:
            os.environ["NO_PROXY"] = f"{no_proxy},localhost,127.0.0.1" if no_proxy else "localhost,127.0.0.1"
            os.environ["no_proxy"] = os.environ["NO_PROXY"]

    # ───────────────────────────────────────────────────────────────────────
    # 可用性检查
    # ───────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """检查 Ollama 服务是否可用 (同步, 带缓存)。"""
        if self._available is not None:
            return self._available

        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5.0,
                proxy=None,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                self._available = any(self.model in m for m in models)
                if self._available:
                    logger.info(f"[VLM] Ollama 就绪: model={self.model}")
                else:
                    logger.warning(
                        f"[VLM] Ollama 运行中但模型 {self.model} 未安装. "
                        f"可用模型: {models}"
                    )
            else:
                self._available = False
        except Exception as e:
            logger.debug(f"[VLM] Ollama 不可用: {e}")
            self._available = False

        return self._available

    # ───────────────────────────────────────────────────────────────────────
    # 核心推理方法
    # ───────────────────────────────────────────────────────────────────────

    async def _call_ollama(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        max_tokens: int = 8192,
    ) -> Optional[str]:
        """底层 Ollama API 调用。

        Args:
            prompt: 文本提示
            image_bytes: 图片二进制数据 (PNG/JPEG)
            max_tokens: 最大输出 tokens

        Returns:
            模型输出文本, 失败返回 None
        """
        try:
            import httpx
        except ImportError:
            logger.error("[VLM] httpx 未安装, 请运行: pip install httpx")
            return None

        # 图片编码为 base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout, proxy=None) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )

            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                logger.warning(
                    f"[VLM] Ollama 返回 {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                return None

            data = resp.json()
            response_text = data.get("response", "")

            # 统计信息
            eval_count = data.get("eval_count", 0)
            speed = eval_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"[VLM] 推理完成: {eval_count} tokens, "
                f"{elapsed:.1f}s, {speed:.1f} tok/s"
            )

            return response_text

        except httpx.TimeoutException:
            elapsed = time.monotonic() - t0
            logger.warning(f"[VLM] 推理超时 ({elapsed:.1f}s > {self.timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"[VLM] 推理异常: {e}")
            return None

    # ───────────────────────────────────────────────────────────────────────
    # 高级 API
    # ───────────────────────────────────────────────────────────────────────

    async def page_to_markdown(self, image_bytes: bytes) -> Optional[str]:
        """页面图片 → 完整 Markdown (文本 + 表格 HTML + LaTeX 公式)。

        这是跑分的核心方法: 一次调用完成文本/表格/公式全部识别。

        Args:
            image_bytes: 页面渲染图片 (PNG/JPEG), 建议 200-300 DPI

        Returns:
            Markdown 字符串, 失败返回 None
        """
        return await self._call_ollama(
            PROMPT_PAGE_TO_MARKDOWN, image_bytes, max_tokens=8192
        )

    async def recognize_table(self, image_bytes: bytes) -> Optional[str]:
        """表格区域图片 → HTML <table> 结构。

        用于 Pipeline 提取的表格质量不佳时, 用 VLM 做二次识别。

        Returns:
            HTML 表格字符串, 失败返回 None
        """
        return await self._call_ollama(
            PROMPT_TABLE_TO_HTML, image_bytes, max_tokens=4096
        )

    async def recognize_formula(self, image_bytes: bytes) -> Optional[str]:
        """公式区域图片 → LaTeX 字符串。

        Returns:
            LaTeX 字符串 (不含 $$ 包裹), 失败返回 None
        """
        return await self._call_ollama(
            PROMPT_FORMULA_TO_LATEX, image_bytes, max_tokens=1024
        )

    # ───────────────────────────────────────────────────────────────────────
    # 工具方法
    # ───────────────────────────────────────────────────────────────────────

    def render_page_to_image(self, pdf_path, page_idx: int = 0, dpi: int = 216) -> Optional[bytes]:
        """将 PDF 页面渲染为 PNG 图片。

        Args:
            pdf_path: PDF 文件路径
            page_idx: 页码 (0-based)
            dpi: 渲染 DPI (216 = 3x 标准 72dpi, 平衡质量和速度)

        Returns:
            PNG bytes, 失败返回 None
        """
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            if page_idx >= len(doc):
                return None
            page = doc[page_idx]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        except Exception as e:
            logger.warning(f"[VLM] 页面渲染失败: {e}")
            return None
