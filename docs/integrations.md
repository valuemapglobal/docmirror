# AI Integrations (RAG)

> **Status: Planned for v0.3**
>
> In-tree LangChain and LlamaIndex loaders have been removed in v0.2.
> We plan to submit official integration PRs upstream to `langchain-community`
> and `llama-index` once the DocMirror API stabilizes.

## Current Approach

You can integrate DocMirror with any RAG pipeline today using the core API:

```python
from docmirror import perceive_document

async def load_for_rag(file_path: str):
    result = await perceive_document(file_path)

    # Create chunks from content blocks
    chunks = []
    for block in result.content.blocks:
        if block.type == "table" and block.table:
            # Tables as Markdown for better LLM comprehension
            headers = block.table.headers or []
            rows = block.table.rows or []
            md = "| " + " | ".join(headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                md += "| " + " | ".join(str(c) for c in row) + " |\n"
            chunks.append({
                "text": md,
                "metadata": {"page": block.page, "type": "table"},
            })
        elif block.text and block.text.content:
            chunks.append({
                "text": block.text.content,
                "metadata": {"page": block.page, "type": block.type.value},
            })

    return chunks
```

## Integration Advantages

1. **MultiModal Fallback**: If a PDF is a scanned image, it automatically falls back to ONNX OCR.
2. **Tabular Context**: RAG LLMs receive perfectly structured Markdown tables instead of garbled text.
3. **No External APIs**: All processing runs locally with maximum data privacy.
4. **Trust Scoring**: Each parse includes a `validation_score` so you can filter low-quality results.
