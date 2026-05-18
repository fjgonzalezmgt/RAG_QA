"""PDF discovery and text extraction.

PDFs are assumed to live directly in the configured input folder. This module
extracts text page by page and captures lightweight metadata used during
ingestion and citation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page.

    Attributes
    ----------
    page_number
        One-based page number.
    text
        Extracted text content.
    """

    page_number: int
    text: str


@dataclass(frozen=True)
class PdfDocument:
    """Loaded PDF document.

    Attributes
    ----------
    path
        Source PDF path.
    title
        Optional title metadata.
    author
        Optional author metadata.
    content_hash
        SHA-256 hash of the file contents.
    pages
        Extracted page texts.
    """

    path: Path
    title: str | None
    author: str | None
    content_hash: str
    pages: list[PageText]


def file_sha256(path: Path) -> str:
    """Compute a SHA-256 hash for a file.

    Parameters
    ----------
    path
        File path.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pdf(path: Path) -> PdfDocument:
    """Load a PDF and extract text page by page.

    Parameters
    ----------
    path
        PDF file path.

    Returns
    -------
    PdfDocument
        Extracted document metadata and page text.
    """

    logger.info("Loading PDF '{}'.", path)
    reader = PdfReader(str(path))
    metadata = reader.metadata or {}
    pages: list[PageText] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page_number=index, text=text))

    title = _clean_metadata_value(metadata.get("/Title")) if metadata else None
    author = _clean_metadata_value(metadata.get("/Author")) if metadata else None

    document = PdfDocument(
        path=path,
        title=title,
        author=author,
        content_hash=file_sha256(path),
        pages=pages,
    )
    logger.info(
        "Loaded PDF '{}'. pages={}, title='{}'.",
        path.name,
        len(document.pages),
        document.title or "",
    )
    return document


def list_pdfs(pdf_dir: Path) -> list[Path]:
    """List root-level PDF files in a directory.

    Parameters
    ----------
    pdf_dir
        Folder to scan.

    Returns
    -------
    list[pathlib.Path]
        Sorted PDF paths.
    """

    logger.info("Listing root-level PDFs in '{}'.", pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdf_dir}")
    pdfs = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    logger.info("PDF listing finished. count={}.", len(pdfs))
    return pdfs


def _clean_metadata_value(value: object) -> str | None:
    """Normalize optional PDF metadata values."""

    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
