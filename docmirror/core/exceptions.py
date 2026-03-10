"""
MultiModal 异常体系 (Exception Hierarchy)
======================================

统一的类型化异常层级，替代裸 Exception。

层级结构::

    MultiModalError (base)
    ├── ExtractionError      — CoreExtractor / 物理提取失败
    ├── LayoutAnalysisError   — 版面分析 / Zone 分区失败
    ├── MiddlewareError       — 中间件处理失败 (携带 middleware_name)
    └── ValidationError       — 数据校验不通过

使用指南:
    - 可恢复错误: 在 try/except 中捕获并 add_error(), 不终止管线
    - 不可恢复错误: 抛出，由上层 Pipeline 的 fail_strategy 决定处理方式
"""

from __future__ import annotations


class MultiModalError(Exception):
    """MultiModal 异常基类。"""

    def __init__(self, message: str = "", *, detail: str = ""):
        self.detail = detail
        super().__init__(message)


class ExtractionError(MultiModalError):
    """CoreExtractor 物理提取过程中的错误。

    示例: PDF 打开失败, pdfplumber 解析失败, 页面超上限等。
    """
    pass


class LayoutAnalysisError(MultiModalError):
    """版面分析 / Zone 分区 / 表格提取层的错误。"""
    pass


class MiddlewareError(MultiModalError):
    """中间件处理过程中的错误。

    Attributes:
        middleware_name: 出错的中间件名称。
    """

    def __init__(self, message: str = "", *, middleware_name: str = "", detail: str = ""):
        self.middleware_name = middleware_name
        super().__init__(message, detail=detail)

    def __str__(self):
        prefix = f"[{self.middleware_name}] " if self.middleware_name else ""
        return f"{prefix}{super().__str__()}"


class ValidationError(MultiModalError):
    """数据校验不通过。

    示例: 表格列数不一致, 日期覆盖率过低, 置信度低于阈值等。
    """
    pass
