# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for DocMirror document parsing engine.

Provides single-file and batch-directory parsing with rich progress
display, multiple output formats, and result persistence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Default output directory (relative to cwd)
DEFAULT_OUTPUT_DIR = Path("output")


def _safe_str(s: str) -> str:
    """Encode/decode to replace surrogates so console.print() never raises UnicodeEncodeError."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", errors="replace").decode("utf-8")


# Skip these when discovering files in a directory
SKIP_NAMES = {".DS_Store", ".gitkeep", "Thumbs.db"}


def discover_files(root: Path) -> list[Path]:
    """Recursively collect all files under *root* (excludes SKIP_NAMES)."""
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in SKIP_NAMES:
            files.append(p)
    return files


BANNER = r"""[cyan]
 ____             __  __ _
|  _ \  ___   ___|  \/  (_)_ __ _ __ ___  _ __
| | | |/ _ \ / __| |\/| | | '__| '__/ _ \| '__|
| |_| | (_) | (__| |  | | | |  | | | (_) | |
|____/ \___/ \___|_|  |_|_|_|  |_|  \___/|_|
[/cyan]
[bold white]Universal Document Parsing Engine[/bold white]
[yellow]Support us with a ⭐ on GitHub: https://github.com/valuemapglobal/docmirror[/yellow]
"""


def print_banner():
    console.print(Panel(BANNER, border_style="cyan", padding=(1, 2)))


def show_authors():
    console.print(
        Panel(
            "[bold cyan]Made with \u2764\ufe0f by[/bold cyan]\n[white]Adam Lin[/white]",
            title="Authors",
            border_style="cyan",
        )
    )
    console.print(
        "\n[yellow]Want your name here? Contribute to DocMirror at: https://github.com/valuemapglobal/docmirror[/yellow]\n"
    )


def save_result(result_dict: dict, source_path: Path, output_dir: Path) -> Path:
    """Save parse result as JSON to the output directory. Returns the saved file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{source_path.stem}.json"

    # Avoid overwriting: append a numeric suffix if the file already exists
    counter = 1
    while output_file.exists():
        output_file = output_dir / f"{source_path.stem}_{counter}.json"
        counter += 1

    output_file.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


async def parse_document(
    file_path: str,
    format_out: str,
    output_dir: Path,
    no_save: bool,
    skip_cache: bool = False,
    include_text: bool = False,
) -> None:
    from docmirror.core.factory import perceive_document
    from docmirror.models.entities.document_types import DocumentType

    path = Path(file_path).resolve()
    if not path.exists():
        console.print(f"[bold red]Error[/bold red]: File not found: {file_path}")
        return
    if path.is_dir():
        console.print(
            f"[bold red]Error[/bold red]: Path is a directory (use it as the batch root to parse all files inside): {path}"
        )
        return

    # ── Pipeline stage definitions for progress display ──
    STAGES = [
        (5, "[cyan]Loading document...[/cyan]"),
        (15, "[cyan]Extracting pages...[/cyan]"),
        (35, "[cyan]Detecting layout & tables...[/cyan]"),
        (55, "[cyan]Running OCR & text extraction...[/cyan]"),
        (70, "[cyan]Analyzing entities & structure...[/cyan]"),
        (85, "[cyan]Mapping columns & validating...[/cyan]"),
        (95, "[cyan]Building result...[/cyan]"),
    ]

    from rich.progress import BarColumn, TaskProgressColumn, TimeElapsedColumn

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    async def _animate_progress(progress, task_id):
        """Simulate stage-based progress while parsing runs."""
        import time

        start = time.monotonic()
        stage_idx = 0
        while not progress.tasks[task_id].finished:
            elapsed = time.monotonic() - start
            # Advance through stages based on elapsed time
            # Rough heuristic: ~2s per stage for a typical document
            target_stage = min(int(elapsed / 2.0), len(STAGES) - 1)
            while stage_idx <= target_stage and stage_idx < len(STAGES):
                pct, desc = STAGES[stage_idx]
                progress.update(task_id, completed=pct, description=desc)
                stage_idx += 1
            await asyncio.sleep(0.15)

    with progress:
        task_id = progress.add_task(
            STAGES[0][1],
            total=100,
        )
        # Start progress animation concurrently with parsing
        import time as _time

        _wall_start = _time.monotonic()
        anim_task = asyncio.create_task(_animate_progress(progress, task_id))
        try:
            result = await perceive_document(path, DocumentType.OTHER, skip_cache=skip_cache)
            progress.update(task_id, completed=100, description="[bold green]✅ Done![/bold green]")
            anim_task.cancel()
        except Exception as e:
            progress.update(task_id, completed=100, description="[bold red]❌ Failed[/bold red]")
            anim_task.cancel()
            console.print(f"[bold red]Critical Error:[/bold red] {_safe_str(str(e))}")
            return

    wall_elapsed_ms = (_time.monotonic() - _wall_start) * 1000

    # ── Display results (outside spinner) ──
    try:
        api_dict = result.to_api_dict(include_text=include_text)

        if result.success:
            console.print("\n[bold green]\u2705 Parsing Complete![/bold green]")

            table = Table(show_header=False, border_style="green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Status", str(result.status))
            table.add_row("Confidence", f"{result.confidence:.2%}")
            table.add_row("Pages", str(result.page_count))
            table.add_row("Tables Found", str(result.total_tables))
            table.add_row("Extracted Text", f"{len(result.full_text)} chars")
            table.add_row("Time Elapsed", f"{wall_elapsed_ms:.0f} ms")

            # Detect cached results: internal timing >> wall time
            is_cached = (
                result.parser_info
                and result.parser_info.elapsed_ms > 0
                and wall_elapsed_ms < result.parser_info.elapsed_ms * 0.5
                and wall_elapsed_ms < 2000
            )
            if is_cached:
                table.add_row("", "[dim italic]⚡ cached result[/dim italic]")

            console.print(table)

            effective_ms = max(wall_elapsed_ms, 1)
            speed = len(result.full_text) / (effective_ms / 1000)
            console.print(f"\n[bold magenta]\u26a1 BLAZING FAST:[/bold magenta] Processed at {speed:.0f} chars/sec!")
            console.print(
                "[dim]Copy this benchmark and share it on Twitter / V2EX to show off your speed! \u26a1[/dim]"
            )
        else:
            console.print("\n[bold red]\u274c Parsing Failed[/bold red]")
            if result.error:
                console.print(f"[red]{_safe_str(result.error.message)}[/red]")

            console.print("\n[bold yellow]Open Source Power[/bold yellow]")
            console.print("[white]Encountered an unsupported exotic format? This is how we improve![/white]")
            console.print("[white]Please attach the logs and a sample document by opening an issue at:[/white]")
            console.print("[cyan]https://github.com/valuemapglobal/docmirror/issues[/cyan]")

        # Save result to disk (both success and failure, for diagnostics)
        if not no_save:
            saved_path = save_result(api_dict, path, output_dir)
            console.print(f"\n[bold blue]\U0001f4be Result saved to:[/bold blue] [white]{saved_path}[/white]")

    except Exception as e:
        console.print(f"[bold red]Critical Error:[/bold red] {_safe_str(str(e))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocMirror - Universal Document Parsing Engine")
    parser.add_argument(
        "file", nargs="?", help="Path to a document or a directory (recursively parse all files under it)"
    )
    parser.add_argument("--format", default="markdown", choices=["markdown", "json", "text"], help="Output format")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save parse results (default: ./output)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save result to disk")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache and force a full re-parse")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="Skip files whose path contains SUBSTR (e.g. --exclude 工商银行); can be repeated",
    )
    parser.add_argument("--authors", action="store_true", help="Show contributors and authors")
    parser.add_argument("--include-text", action="store_true", help="Include full markdown text in output")

    args = parser.parse_args()

    if args.authors:
        print_banner()
        show_authors()
        return

    if not args.file:
        print_banner()
        parser.print_help()
        return

    print_banner()
    path = Path(args.file).resolve()
    if not path.exists():
        console.print(f"[bold red]Error[/bold red]: Path not found: {path}")
        return

    if path.is_dir():
        files = discover_files(path)
        if args.exclude:
            excluded = [f for f in files if any(pat in str(f) for pat in args.exclude)]
            files = [f for f in files if not any(pat in str(f) for pat in args.exclude)]
            if excluded:
                console.print(f"[dim]Excluding {len(excluded)} file(s) matching: {', '.join(args.exclude)}[/dim]")
        if not files:
            console.print(f"[bold yellow]No files found under[/bold yellow] {path}")
            return
        console.print(f"[bold cyan]Batch mode:[/bold cyan] {len(files)} file(s) under [white]{path}[/white]\n")

        async def _batch_parse():
            for i, fp in enumerate(files, 1):
                console.print(f"\n[bold blue][{i}/{len(files)}][/bold blue] {fp.name}")
                await parse_document(
                    str(fp), args.format, args.output_dir, args.no_save, args.skip_cache, args.include_text
                )

        asyncio.run(_batch_parse())
    else:
        asyncio.run(
            parse_document(args.file, args.format, args.output_dir, args.no_save, args.skip_cache, args.include_text)
        )


if __name__ == "__main__":
    main()
