"""
Pipeline 注册表 — 按格式注册中间件组合
==========================================

扩展方式: 在 FORMAT_PIPELINES 中添加新格式即可。
"""

from typing import Dict, List


# 格式 → { 增强模式 → 中间件列表 }
FORMAT_PIPELINES: Dict[str, Dict[str, List[str]]] = {
    "pdf": {
        "raw": [],
        "standard": [
            "SceneDetector",
            "EntityExtractor",
            "InstitutionDetector",
            "ColumnMapper",
            "Validator",
        ],
        "full": [
            "SceneDetector",
            "EntityExtractor",
            "InstitutionDetector",
            "ColumnMapper",
            "Validator",
            "Repairer",
        ],
    },
    "image": {
        "raw": [],
        "standard": ["LanguageDetector", "GenericEntityExtractor"],
    },
    "excel": {
        "raw": [],
        "standard": ["GenericEntityExtractor"],
    },
    "word": {
        "raw": [],
        "standard": ["LanguageDetector", "GenericEntityExtractor"],
    },
    # 通配 fallback: 未注册格式使用
    "*": {
        "raw": [],
        "standard": ["LanguageDetector"],
    },
}


def get_pipeline_config(file_type: str, enhance_mode: str = "standard") -> List[str]:
    """
    获取指定格式 + 增强模式的中间件列表。

    Args:
        file_type:    文件格式 (pdf, image, excel, word, ...)
        enhance_mode: 增强模式 (raw, standard, full)

    Returns:
        中间件名称列表 (按执行顺序)
    """
    fmt_config = FORMAT_PIPELINES.get(file_type, FORMAT_PIPELINES.get("*", {}))
    return fmt_config.get(enhance_mode, fmt_config.get("standard", []))
