"""PDF discovery and text extraction.

PDFs are assumed to live directly in the configured input folder. This module
extracts text page by page and captures lightweight metadata used during
ingestion and citation.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import subprocess

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


def load_pdf(path: Path, ocr_fallback: bool = False) -> PdfDocument:
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

    if ocr_fallback and not any(page.text.strip() for page in pages):
        ocr_text = extract_text_with_pdftotext(path)
        if not ocr_text:
            ocr_text = extract_text_with_ocrmypdf(path)
        if ocr_text:
            pages = [PageText(page_number=index, text=text) for index, text in enumerate(split_pdftotext_pages(ocr_text), start=1)]

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


def list_pdfs(pdf_dir: Path, recursive: bool = True) -> list[Path]:
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
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(path for path in pdf_dir.glob(pattern) if path.is_file())
    logger.info("PDF listing finished. count={}.", len(pdfs))
    return pdfs


def extract_text_with_pdftotext(path: Path) -> str:
    """Try Poppler's pdftotext as a pragmatic fallback for difficult PDFs.

    Parameters
    ----------
    path
        PDF path to extract.

    Returns
    -------
    str
        Extracted text, or an empty string when extraction fails.
    """

    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pdftotext fallback unavailable for '{}': {}", path.name, exc)
        return ""
    if completed.returncode != 0:
        logger.warning("pdftotext fallback failed for '{}': {}", path.name, completed.stderr.strip())
        return ""
    return completed.stdout or ""


def extract_text_with_ocrmypdf(path: Path) -> str:
    """Run optional OCR through ocrmypdf when the command is installed.

    Parameters
    ----------
    path
        PDF path to OCR.

    Returns
    -------
    str
        OCR-extracted text, or an empty string when OCR is unavailable.
    """

    with tempfile.TemporaryDirectory(prefix="quality_ocr_") as temp_dir:
        output_path = Path(temp_dir) / "ocr.pdf"
        try:
            ocr_completed = subprocess.run(
                ["ocrmypdf", "--skip-text", "--quiet", str(path), str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("ocrmypdf fallback unavailable for '{}': {}", path.name, exc)
            return ""
        if ocr_completed.returncode != 0 or not output_path.exists():
            logger.warning("ocrmypdf fallback failed for '{}': {}", path.name, ocr_completed.stderr.strip())
            return ""
        return extract_text_with_pdftotext(output_path)


def split_pdftotext_pages(text: str) -> list[str]:
    """Split pdftotext output into page-sized strings when form feeds exist.

    Parameters
    ----------
    text
        Raw text returned by pdftotext.

    Returns
    -------
    list[str]
        Non-empty page texts.
    """

    pages = [page.strip() for page in text.split("\f")]
    return [page for page in pages if page]


def _clean_metadata_value(value: object) -> str | None:
    """Normalize optional PDF metadata values.

    Parameters
    ----------
    value
        Raw PDF metadata value.

    Returns
    -------
    str or None
        Cleaned metadata value.
    """

    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
