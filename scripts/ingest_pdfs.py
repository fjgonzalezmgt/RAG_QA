"""Command-line entry point for PDF ingestion.

The script mirrors the Streamlit ingestion action for batch or scheduled use.
It reads configuration from `.env`, allows domain and PDF directory overrides,
and prints a concise ingestion summary.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_books.config import get_settings
from rag_books.ingest import PDFIngestor
from rag_books.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description="Ingest root-level PDFs into PostgreSQL/pgvector.")
    parser.add_argument("--pdf-dir", type=Path, help="Directory containing PDFs in its root.")
    parser.add_argument("--domain", help="Logical domain, for example literatura or sistemas_gestion.")
    parser.add_argument("--force", action="store_true", help="Replace indexed chunks for each source path.")
    return parser.parse_args()


def main() -> int:
    """Run PDF ingestion from the command line.

    Returns
    -------
    int
        Process exit code. Returns 0 when ingestion completes without errors.
    """

    setup_logging(ROOT)
    args = parse_args()
    settings = get_settings()

    rag_settings = settings.rag
    if args.pdf_dir:
        rag_settings = replace(rag_settings, pdf_dir=args.pdf_dir.resolve())
    if args.domain:
        rag_settings = replace(rag_settings, domain=args.domain)
    settings = replace(settings, rag=rag_settings)

    ingestor = PDFIngestor(settings=settings)
    result = ingestor.ingest_directory(
        pdf_dir=rag_settings.pdf_dir,
        domain=rag_settings.domain,
        force=args.force,
        progress=print,
    )

    print("")
    print(f"PDF directory: {result.pdf_dir}")
    print(f"Domain: {result.domain}")
    print(f"Files found: {result.files_found}")
    print(f"Documents ingested: {result.documents_ingested}")
    print(f"Documents skipped: {result.documents_skipped}")
    print(f"Chunks created: {result.chunks_created}")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"- {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
