# Creating Plugins

## Step 1: Define Your Plugin

Create a new file in `docmirror/plugins/` (for built-in) or your own package:

```python
from docmirror.plugins import DomainPlugin, ColumnHint

class InvoicePlugin(DomainPlugin):
    @property
    def domain_name(self) -> str:
        return "invoice"

    @property
    def display_name(self) -> str:
        return "Invoice"

    @property
    def scene_keywords(self):
        return ("invoice", "bill", "tax invoice", "receipt")

    @property
    def identity_fields(self):
        return (
            ("invoice_number", ("Invoice No", "Bill Number")),
            ("seller", ("Seller", "Vendor", "Supplier")),
            ("buyer", ("Buyer", "Customer")),
            ("total_amount", ("Total", "Amount Due")),
            ("invoice_date", ("Date", "Invoice Date")),
        )

    @property
    def standard_columns(self):
        return (
            ColumnHint("item", ("Item", "Description"), required=True),
            ColumnHint("quantity", ("Qty", "Quantity")),
            ColumnHint("unit_price", ("Unit Price", "Price")),
            ColumnHint("amount", ("Amount", "Total"), required=True),
        )

    def build_domain_data(self, metadata, entities):
        # Return your domain-specific data model
        return {
            "document_type": "invoice",
            "invoice_number": entities.get("invoice_number", ""),
            "total_amount": entities.get("total_amount", ""),
        }

# Required: module-level instance for auto-discovery
plugin = InvoicePlugin()
```

## Step 2: Register (Auto-Discovery)

Place your file in `docmirror/plugins/`. The registry automatically discovers
modules with a `plugin` attribute.

For third-party packages, use entry points in `pyproject.toml`:

```toml
[project.entry-points."docmirror.plugins"]
invoice = "my_package.plugins:InvoicePlugin"
```

## Step 3: Verify

```python
from docmirror.plugins import registry

print(registry.list_plugins())
# {'bank_statement': 'Bank Statement', 'invoice': 'Invoice'}

plugin = registry.get("invoice")
print(plugin.scene_keywords)
```

## Plugin API Reference

::: docmirror.plugins.DomainPlugin
    options:
      show_source: false
