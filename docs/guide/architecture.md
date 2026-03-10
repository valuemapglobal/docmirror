# Architecture

## System Overview

```mermaid
graph TD
    A[Input Document] --> B[Dispatcher]
    B --> C[Format Adapter]
    C --> D[Core Extraction]
    D --> E[BaseResult]
    E --> F[Middleware Pipeline]
    F --> G[PerceptionResult]

    subgraph Adapters
        C1[PDF] -.-> C
        C2[Image] -.-> C
        C3[Word] -.-> C
        C4[Excel] -.-> C
        C5[PPT] -.-> C
        C6[Email] -.-> C
    end

    subgraph Core
        D1[Layout Analysis] -.-> D
        D2[OCR Engine] -.-> D
        D3[Table Extraction] -.-> D
        D4[Formula Recognition] -.-> D
    end

    subgraph Middlewares
        F1[Scene Detection] -.-> F
        F2[Entity Extraction] -.-> F
        F3[Column Mapping] -.-> F
        F4[Validation] -.-> F
    end
```

## Layer Architecture

| Layer | Module | Responsibility |
|-------|--------|---------------|
| **Dispatch** | `framework.dispatcher` | Route files to appropriate adapter |
| **Adapt** | `adapters.*` | Convert format → BaseResult |
| **Extract** | `core.extraction` | Low-level parsing (text, tables, layout) |
| **Enhance** | `middlewares.*` | Business logic pipeline |
| **Output** | `models.*` | Structured PerceptionResult |

## Data Flow

1. **Dispatcher** detects file type and selects adapter
2. **Adapter** converts raw document → immutable `BaseResult`
3. **Orchestrator** runs middleware pipeline on the result
4. **Middlewares** enhance: detect scene → extract entities → map columns → validate
5. **Builder** assembles final `PerceptionResult`

## Plugin System

Domain plugins extend DocMirror with business-specific logic:

```python
from docmirror.plugins import DomainPlugin

class InvoicePlugin(DomainPlugin):
    domain_name = "invoice"
    display_name = "Invoice"
    scene_keywords = ("invoice", "bill", "receipt")
    # ... implement build_domain_data()
```

See [Creating Plugins](../plugins/creating-plugins.md) for details.
