# Plugin System

DocMirror uses a plugin architecture to separate generic document parsing from domain-specific business logic.

## Built-in Plugins

| Domain | Plugin | Description |
|--------|--------|-------------|
| `bank_statement` | `BankStatementPlugin` | Bank statement processing with institution detection, column mapping, and transaction extraction |

## How Plugins Work

```mermaid
graph LR
    A[Document] --> B[Core Parsing]
    B --> C{Scene Detection}
    C -->|bank_statement| D[BankStatementPlugin]
    C -->|invoice| E[InvoicePlugin]
    C -->|unknown| F[Generic Processing]
    D --> G[PerceptionResult]
    E --> G
    F --> G
```

Each plugin provides:

1. **Scene keywords** — trigger automatic domain classification
2. **Identity fields** — domain-specific entity definitions
3. **Standard columns** — column name standardization rules
4. **Domain data builder** — structured output model construction
