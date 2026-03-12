"""
Backward Compatibility — Shared utilities for import-path shims.
================================================================

Multiple packages (core, models, middlewares) use the same pattern of
registering old import paths to point at new subpackage locations via
``sys.modules`` injection.  This module centralises that logic.

Usage::

    from docmirror._compat import register_shims

    register_shims({
        "docmirror.core.extractor": "docmirror.core.extraction.extractor",
        ...
    })
"""
from __future__ import annotations


import importlib
import sys
from typing import Dict


def register_shims(shim_map: Dict[str, str], *, silent: bool = True) -> None:
    """Register backward-compatible module aliases in ``sys.modules``.

    For each *(old_path, new_path)* pair, if *old_path* is not yet
    loaded, the *new_path* module is imported and placed at *old_path*
    so that ``import old_path`` keeps working transparently.

    Args:
        shim_map: Mapping of ``{old_module_path: new_module_path}``.
        silent:   If True (default), silently ignore ImportErrors
                  from optional dependencies.
    """
    for old_path, new_path in shim_map.items():
        if old_path not in sys.modules:
            try:
                sys.modules[old_path] = importlib.import_module(new_path)
            except ImportError:
                if not silent:
                    raise
