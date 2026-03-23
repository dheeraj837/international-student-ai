#!/usr/bin/env python3
"""
Standalone ingestion script.
Run this once (or with --force) to populate the Qdrant vector DB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --force
"""

import sys
import argparse
import logging
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import track

from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.sources import USCIS_SOURCES

console = Console()
logging.basicConfig(level=logging.WARNING)  # suppress verbose logs in CLI


def main():
    parser = argparse.ArgumentParser(description="Ingest USCIS documents into Qdrant.")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    args = parser.parse_args()

    console.rule("[bold blue]International Student AI – Data Ingestion")
    console.print(f"\n[cyan]Sources to process:[/cyan] {len(USCIS_SOURCES)} USCIS pages")

    # Show source table
    table = Table(title="USCIS Source URLs", show_lines=True)
    table.add_column("Category", style="green", width=18)
    table.add_column("Title", style="white", width=45)
    table.add_column("URL", style="blue", width=55)

    for s in USCIS_SOURCES:
        table.add_row(s["category"], s["title"], s["url"])

    console.print(table)
    console.print()

    with console.status("[bold green]Running ingestion pipeline...", spinner="dots"):
        result = run_ingestion_pipeline(force=args.force)

    # Result summary
    status_color = "green" if result["status"] in ("success", "skipped") else "red"
    console.print(f"\n[bold {status_color}]Status:[/bold {status_color}] {result['status'].upper()}")
    console.print(f"  Sources processed : {result['sources_processed']}")
    console.print(f"  Chunks indexed    : {result['chunks_indexed']}")
    console.print(f"  Message           : {result['message']}")

    console.print("\n[bold green]✓ Done![/bold green] You can now start the API with:")
    console.print("  [yellow]docker compose up[/yellow]  or  [yellow]uvicorn app.main:app --reload[/yellow]\n")


if __name__ == "__main__":
    main()
