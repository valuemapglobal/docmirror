# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Detection middlewares \u2013 scene, language, institution."""

from .institution_detector import InstitutionDetector
from .language_detector import LanguageDetector
from .scene_detector import SceneDetector

__all__ = ["SceneDetector", "LanguageDetector", "InstitutionDetector"]
