#!/usr/bin/env python3
"""
scripts/index_knowledge_base.py
────────────────────────────────
CLI script to ingest and index the operational knowledge base.
Run this before starting the API or running evaluations.

Usage:
  python scripts/index_knowledge_base.py --data-dir data/samples
  python scripts/index_knowledge_base.py --source servicenow
  python scripts/index_knowledge_base.py --rebuild   # full re-index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ticket_ingester import LocalTicketIngester
from src.ingestion.runbook_ingester import LocalRunbookIngester
from src.indexing.chunker import DocumentChunker
from src.indexing.embedder import Embedder
from src.indexing.vector_store import FAISSVectorStore

console = Console()


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_indexing(args):
    console.rule("[bold blue]RAG-AIOps Knowledge Base Indexer")

    config = load_config(args.config)

    # ── Ingestion ────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1: Ingesting documents...[/bold cyan]")
    documents = []

    data_dir = args.data_dir or "data/samples"

    # Tickets
    ticket_ingester = LocalTicketIngester({"path": f"{data_dir}/incidents"})
    if ticket_ingester.validate_connection():
        tickets = ticket_ingester.run()
        documents.extend(tickets)
        console.print(f"  ✓ Incident tickets: [green]{len(tickets)}[/green]")
    else:
        console.print(f"  ⚠ No tickets found at {data_dir}/incidents")

    # Runbooks & Post-mortems
    runbook_ingester = LocalRunbookIngester({
        "path": data_dir,
        "recursive": True,
        "patterns": ["*.md", "*.txt"],
    })
    if runbook_ingester.validate_connection():
        runbooks = runbook_ingester.run()
        documents.extend(runbooks)
        console.print(f"  ✓ Runbooks/docs/post-mortems: [green]{len(runbooks)}[/green]")

    if not documents:
        console.print("[red]No documents found. Check --data-dir path.[/red]")
        sys.exit(1)

    console.print(f"\n  Total documents ingested: [bold green]{len(documents)}[/bold green]")

    # ── Chunking ─────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2: Chunking documents...[/bold cyan]")
    chunker = DocumentChunker(config.get("chunking", {}))
    chunks = chunker.chunk_documents(documents)
    console.print(f"  ✓ Total chunks: [bold green]{len(chunks)}[/bold green]")

    # ── Embedding ─────────────────────────────────────────────
    console.print("\n[bold cyan]Step 3: Generating embeddings...[/bold cyan]")

    # For PoC without API key, use local embeddings
    emb_config = config["embedding"].copy()
    if args.local_embeddings:
        emb_config["provider"] = "local"
        console.print("  Using local sentence-transformers (no API key needed)")

    embedder = Embedder({**emb_config, **config["llm"], "cache": config.get("cache", {})})

    t0 = time.time()
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(chunks))
        # Process in batches
        batch_size = emb_config.get("batch_size", 100)
        chunk_embedding_pairs = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            pairs = embedder.embed_chunks(batch)
            chunk_embedding_pairs.extend(pairs)
            progress.update(task, advance=len(batch))

    elapsed = time.time() - t0
    console.print(f"  ✓ Embedded {len(chunk_embedding_pairs)} chunks in {elapsed:.1f}s")

    # ── Vector Indexing ───────────────────────────────────────
    console.print("\n[bold cyan]Step 4: Building FAISS HNSW index...[/bold cyan]")
    vs_config = {
        **config["vector_store"],
        "dimensions": embedder.dimensions,
    }
    vector_store = FAISSVectorStore(vs_config)

    if args.rebuild:
        console.print("  Rebuilding index from scratch (blue-green swap)...")
        vector_store.rebuild_index(chunk_embedding_pairs)
    else:
        console.print("  Incrementally adding to existing index...")
        vector_store.add_chunks(chunk_embedding_pairs)

    # ── Summary ───────────────────────────────────────────────
    console.rule("[bold green]Indexing Complete")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Documents ingested", str(len(documents)))
    table.add_row("Chunks created", str(len(chunks)))
    table.add_row("Vectors indexed", str(vector_store.total_vectors))
    table.add_row("Index type", "FAISS HNSW")
    table.add_row("Embedding dimensions", str(embedder.dimensions))
    table.add_row("Total time", f"{time.time() - t0:.1f}s")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Index operational knowledge base for RAG-AIOps"
    )
    parser.add_argument("--data-dir", default="data/samples", help="Path to sample data")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--rebuild", action="store_true", help="Full re-index (blue-green swap)")
    parser.add_argument("--local-embeddings", action="store_true",
                        help="Use local sentence-transformers (no OpenAI key needed)")
    args = parser.parse_args()
    run_indexing(args)


if __name__ == "__main__":
    main()
