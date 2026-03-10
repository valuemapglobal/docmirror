"""
Domain Plugin Interface
=======================

Extensible plugin system for domain-specific document processing.
Each domain (bank_statement, invoice, contract, etc.) registers as a plugin
that provides:

1. Scene matching rules (keywords, patterns)
2. Entity extraction logic (key fields to extract)
3. Domain data construction (structured output model)
4. Column mapping hints (standard column names)

Built-in plugins are auto-discovered from ``docmirror.plugins.*``.
Third-party plugins can register via the ``docmirror.plugins`` entry point group.

Usage::

    from docmirror.plugins import registry

    # Get all registered plugins
    registry.list_plugins()

    # Get plugin for a specific domain
    plugin = registry.get("bank_statement")
    domain_data = plugin.build_domain_data(metadata, entities)
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnHint:
    """Standard column definition for tabular data."""
    standard_name: str
    aliases: Sequence[str] = ()
    description: str = ""
    required: bool = False


class DomainPlugin(ABC):
    """
    Abstract base class for domain plugins.

    Each plugin handles one document domain (e.g., bank_statement, invoice).
    Subclass this and register via the plugin registry to add new domains.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Unique domain identifier (e.g., 'bank_statement', 'invoice')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Bank Statement')."""
        ...

    @property
    def scene_keywords(self) -> Sequence[str]:
        """
        Keywords that indicate this domain when found in document text.
        Used by SceneDetector for automatic classification.
        Returns empty sequence if this plugin does not participate in scene detection.
        """
        return ()

    @property
    def identity_fields(self) -> Sequence[Tuple[str, Sequence[str]]]:
        """
        Identity field definitions: (display_name, candidate_keys...).
        Used by domain_registry for entity extraction.
        Returns empty sequence if not applicable.
        """
        return ()

    @property
    def standard_columns(self) -> Sequence[ColumnHint]:
        """
        Standard column definitions for tabular data in this domain.
        Used by ColumnMapper for column name standardization.
        Returns empty sequence if not applicable.
        """
        return ()

    def build_domain_data(
        self,
        metadata: Dict[str, Any],
        entities: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Build domain-specific data model from extracted metadata and entities.

        Returns a domain data object (e.g., BankStatementData) or None if
        insufficient data is available.

        Default implementation returns None (no domain-specific data).
        """
        return None

    def get_middleware_config(self) -> Dict[str, Any]:
        """
        Return plugin-specific middleware configuration overrides.

        Default implementation returns empty dict (no overrides).
        """
        return {}


class PluginRegistry:
    """
    Central registry for domain plugins.

    Plugins are registered in order of priority. The registry supports:
    - Built-in plugins (auto-discovered from docmirror.plugins.*)
    - Manual registration via register()
    - Entry point discovery (future: via importlib.metadata)
    """

    def __init__(self):
        self._plugins: Dict[str, DomainPlugin] = {}
        self._discovered = False

    def register(self, plugin: DomainPlugin, *, override: bool = False) -> None:
        """Register a domain plugin."""
        name = plugin.domain_name
        if name in self._plugins and not override:
            logger.warning(
                f"Plugin '{name}' already registered; use override=True to replace"
            )
            return
        self._plugins[name] = plugin
        logger.debug(f"Registered domain plugin: {name} ({plugin.display_name})")

    def get(self, domain_name: str) -> Optional[DomainPlugin]:
        """Get a registered plugin by domain name."""
        self._ensure_discovered()
        return self._plugins.get(domain_name)

    def list_plugins(self) -> Dict[str, str]:
        """Return {domain_name: display_name} for all registered plugins."""
        self._ensure_discovered()
        return {name: p.display_name for name, p in self._plugins.items()}

    def get_all_scene_keywords(self) -> Dict[str, Sequence[str]]:
        """Return {domain_name: keywords} for all plugins with scene keywords."""
        self._ensure_discovered()
        return {
            name: p.scene_keywords
            for name, p in self._plugins.items()
            if p.scene_keywords
        }

    def build_domain_data(
        self,
        domain_name: str,
        metadata: Dict[str, Any],
        entities: Dict[str, Any],
    ) -> Optional[Any]:
        """Build domain data using the appropriate plugin."""
        plugin = self.get(domain_name)
        if plugin is None:
            return None
        return plugin.build_domain_data(metadata, entities)

    def _ensure_discovered(self) -> None:
        """Auto-discover built-in plugins on first access."""
        if self._discovered:
            return
        self._discovered = True
        self._discover_builtin_plugins()

    def _discover_builtin_plugins(self) -> None:
        """Discover and load plugins from docmirror.plugins subpackage."""
        try:
            import docmirror.plugins as plugins_pkg
            for importer, modname, ispkg in pkgutil.iter_modules(plugins_pkg.__path__):
                if modname.startswith("_"):
                    continue
                try:
                    mod = importlib.import_module(f"docmirror.plugins.{modname}")
                    # Convention: each plugin module has a `plugin` attribute
                    if hasattr(mod, "plugin"):
                        self.register(mod.plugin)
                    elif hasattr(mod, "Plugin"):
                        self.register(mod.Plugin())
                except Exception as e:
                    logger.warning(f"Failed to load plugin docmirror.plugins.{modname}: {e}")
        except ImportError:
            logger.debug("No docmirror.plugins package found")


# Global singleton registry
registry = PluginRegistry()
