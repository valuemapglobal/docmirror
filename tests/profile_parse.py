"""
Performance profiling script for DocMirror parsing pipeline.

Usage:
    python3 tests/profile_parse.py tests/fixtures/3.pdf

Outputs per-stage timing breakdown and memory usage.
"""
from __future__ import annotations

import asyncio
import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from pathlib import Path


async def profile_parse(file_path: Path) -> None:
    """Profile a single document parse with timing and memory tracking."""
    from docmirror.core.factory import perceive_document
    from docmirror.models.entities.document_types import DocumentType

    print(f"\n{'='*60}")
    print(f"  DocMirror Performance Profile")
    print(f"  File: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
    print(f"{'='*60}\n")

    # ── Memory tracking ──
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()

    # ── CPU profiling ──
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()

    result = await perceive_document(file_path, DocumentType.OTHER)

    elapsed = time.perf_counter() - t0
    profiler.disable()

    mem_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Results summary ──
    print(f"📊 Parse Result:")
    print(f"   Status:     {result.status}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Pages:      {result.content.page_count}")
    print(f"   Text:       {len(result.content.text)} chars")
    print(f"   Tables:     {len(result.tables)}")
    print(f"   Blocks:     {len(result.content.blocks)}")

    print(f"\n⏱  Timing:")
    print(f"   Total:      {elapsed * 1000:.0f} ms")
    if result.timing:
        print(f"   Reported:   {result.timing.elapsed_ms:.0f} ms")

    print(f"\n💾 Memory:")
    print(f"   Current:    {mem_after[0] / 1024 / 1024:.1f} MB")
    print(f"   Peak:       {mem_after[1] / 1024 / 1024:.1f} MB")
    print(f"   Delta:      {(mem_after[0] - mem_before[0]) / 1024 / 1024:.1f} MB")

    # ── Top 20 CPU hotspots ──
    print(f"\n🔥 Top 20 CPU Hotspots:")
    print(f"{'─'*80}")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    # Parse and format output
    lines = stream.getvalue().strip().splitlines()
    for line in lines[5:]:  # Skip header
        print(f"   {line}")

    # ── Per-module breakdown ──
    print(f"\n📦 Per-Module Time (cumulative, top 15):")
    print(f"{'─'*80}")
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats('cumulative')

    # Aggregate by module
    module_times: dict[str, float] = {}
    for (filename, lineno, name), (cc, nc, tt, ct, callers) in stats2.stats.items():
        # Extract module from filename
        if 'docmirror' in filename:
            parts = filename.split('docmirror/')
            if len(parts) > 1:
                module = 'docmirror/' + parts[-1].rsplit('.', 1)[0]
            else:
                module = filename
        elif 'site-packages' in filename:
            parts = filename.split('site-packages/')
            module = parts[-1].split('/')[0] if len(parts) > 1 else filename
        else:
            continue

        module_times[module] = module_times.get(module, 0.0) + ct

    sorted_modules = sorted(module_times.items(), key=lambda x: x[1], reverse=True)[:15]
    for module, ct in sorted_modules:
        pct = (ct / elapsed) * 100
        bar = '█' * int(pct / 2) + '░' * max(0, 50 - int(pct / 2))
        print(f"   {module:50s} {ct*1000:7.0f} ms ({pct:5.1f}%) {bar}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 tests/profile_parse.py <document_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    asyncio.run(profile_parse(file_path))


if __name__ == "__main__":
    main()
