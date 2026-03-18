# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
DocMirror Configuration Package
================================

Provides centralized configuration for the DocMirror parsing engine.
Exports:
    - DocMirrorSettings: Global configuration dataclass with env var overrides.
    - default_settings: Pre-initialized singleton loaded from current environment.
"""

from .settings import DocMirrorSettings, default_settings

__all__ = ["DocMirrorSettings", "default_settings"]
