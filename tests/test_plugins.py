"""
Plugin system tests — verify plugin interface, registry, and built-in plugins.
"""

import pytest
from docmirror.plugins import DomainPlugin, PluginRegistry, registry
from docmirror.plugins.bank_statement import BankStatementPlugin


class TestPluginInterface:
    """Test the DomainPlugin interface and PluginRegistry."""

    def test_bank_statement_plugin_instantiable(self):
        """BankStatementPlugin should be instantiable."""
        plugin = BankStatementPlugin()
        assert plugin.domain_name == "bank_statement"
        assert plugin.display_name == "Bank Statement"

    def test_bank_statement_scene_keywords(self):
        """Plugin should provide scene keywords."""
        plugin = BankStatementPlugin()
        assert len(plugin.scene_keywords) > 0
        assert "bank statement" in plugin.scene_keywords

    def test_bank_statement_identity_fields(self):
        """Plugin should define identity fields."""
        plugin = BankStatementPlugin()
        fields = plugin.identity_fields
        assert len(fields) > 0
        names = [f[0] for f in fields]
        assert "account_holder" in names
        assert "account_number" in names


    def test_bank_statement_build_domain_data(self):
        """Plugin should build domain data from metadata and entities."""
        plugin = BankStatementPlugin()
        result = plugin.build_domain_data(
            metadata={"Account holder": "John Doe", "Account number": "1234"},
            entities={"bank_name": "HSBC"},
        )
        assert result is not None
        assert result.document_type == "bank_statement"
        assert result.bank_statement.account_holder == "John Doe"
        assert result.bank_statement.bank_name == "HSBC"


class TestPluginRegistry:
    """Test the PluginRegistry."""

    def test_register_and_get(self):
        """Should register and retrieve a plugin."""
        reg = PluginRegistry()
        plugin = BankStatementPlugin()
        reg.register(plugin)
        assert reg.get("bank_statement") is plugin

    def test_get_nonexistent_returns_none(self):
        """Should return None for unregistered domain."""
        reg = PluginRegistry()
        assert reg.get("nonexistent") is None

    def test_list_plugins(self):
        """Should list registered plugins."""
        reg = PluginRegistry()
        reg.register(BankStatementPlugin())
        plugins = reg.list_plugins()
        assert "bank_statement" in plugins
        assert plugins["bank_statement"] == "Bank Statement"

    def test_auto_discovery(self):
        """Global registry should auto-discover bank_statement plugin."""
        # Force re-discovery
        fresh_reg = PluginRegistry()
        fresh_reg._ensure_discovered()
        assert "bank_statement" in fresh_reg.list_plugins()

    def test_build_domain_data_via_registry(self):
        """Registry should delegate domain data building to plugin."""
        reg = PluginRegistry()
        reg.register(BankStatementPlugin())
        result = reg.build_domain_data(
            "bank_statement",
            metadata={"Account holder": "Alice"},
            entities={"bank_name": "BoC"},
        )
        assert result is not None
        assert result.bank_statement.account_holder == "Alice"

    def test_build_domain_data_unknown_returns_none(self):
        """Registry should return None for unknown domain."""
        reg = PluginRegistry()
        result = reg.build_domain_data("unknown", {}, {})
        assert result is None

    def test_duplicate_register_warns(self):
        """Duplicate registration without override should be ignored."""
        reg = PluginRegistry()
        p1 = BankStatementPlugin()
        p2 = BankStatementPlugin()
        reg.register(p1)
        reg.register(p2)  # Should warn, not replace
        assert reg.get("bank_statement") is p1

    def test_duplicate_register_with_override(self):
        """Duplicate registration with override should replace."""
        reg = PluginRegistry()
        p1 = BankStatementPlugin()
        p2 = BankStatementPlugin()
        reg.register(p1)
        reg.register(p2, override=True)
        assert reg.get("bank_statement") is p2

    def test_get_all_scene_keywords(self):
        """Should aggregate scene keywords from all plugins."""
        reg = PluginRegistry()
        reg.register(BankStatementPlugin())
        kw = reg.get_all_scene_keywords()
        assert "bank_statement" in kw
        assert len(kw["bank_statement"]) > 0
