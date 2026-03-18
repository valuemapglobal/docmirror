# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
MultiModal Data Model Layer.

Core Trinity:
    - domain.py:   BaseResult (frozen) \u2014 Immutable Extraction Result
    - mutation.py: Mutation            \u2014 Data Lineage Tracker
    - enhanced.py: EnhancedResult      \u2014 Final Enhanced Output
"""

from .entities.domain import BaseResult, Block, PageLayout, Style, TextSpan
from .entities.enhanced import EnhancedResult
from .tracking.mutation import Mutation

__all__ = [
    "Style",
    "TextSpan",
    "Block",
    "PageLayout",
    "BaseResult",
    "Mutation",
    "EnhancedResult",
]
