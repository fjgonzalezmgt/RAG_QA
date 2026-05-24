"""Text normalization and chunking utilities.

This module converts extracted PDF pages into overlapping chunks that are
suitable for embeddings. Chunks preserve page spans so final answers can cite
where retrieved text came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_loader import PageText


@dataclass(frozen=True)
class TextChunk:
    """Chunk of text ready for embedding.

    Attributes
    ----------
    index
        Zero-based chunk index within a document.
    content
        Chunk text.
    page_start, page_end
        Inclusive source page span.
    token_count
        Approximate token count.
    metadata
        Additional chunk metadata.
    """

    index: int
    content: str
    page_start: int
    page_end: int
    token_count: int
    metadata: dict[str, object] = field(default_factory=dict)


def split_pages(
    pages: list[PageText],
    chunk_size: int = 1800,
    chunk_overlap: int = 220,
) -> list[TextChunk]:
    """Split page text into overlapping chunks.

    Parameters
    ----------
    pages
        Page texts extracted from a PDF.
    chunk_size
        Approximate character length per chunk.
    chunk_overlap
        Number of trailing characters reused in the next chunk.

    Returns
    -------
    list[TextChunk]
        Ordered chunks with page spans.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    full_text, spans = _build_full_text(pages)
    if not full_text.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    text_len = len(full_text)

    while start < text_len:
        raw_end = min(start + chunk_size, text_len)
        end = _snap_to_boundary(full_text, start, raw_end)
        content = full_text[start:end].strip()

        if content:
            page_start, page_end = _pages_for_range(spans, start, end)
            chunks.append(
                TextChunk(
                    index=index,
                    content=content,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=estimate_tokens(content),
                    metadata=chunk_metadata(content, page_start, page_end),
                )
            )
            index += 1

        if end >= text_len:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""

    return max(1, len(text) // 4)


def chunk_metadata(content: str, page_start: int, page_end: int) -> dict[str, object]:
    """Infer lightweight operational metadata from chunk text."""

    metadata: dict[str, object] = {"page_start": page_start, "page_end": page_end}
    section = detect_section(content)
    if section:
        metadata.update(section)
    requirement_type = detect_requirement_type(content)
    if requirement_type:
        metadata["requirement_type"] = requirement_type
    key_terms = detect_key_terms(content)
    if key_terms:
        metadata["key_terms"] = key_terms
    entities = detect_entities(content)
    if entities:
        metadata["detected_entities"] = entities
    return metadata


def detect_section(content: str) -> dict[str, str]:
    """Detect a probable section heading or clause reference."""

    for line in content.splitlines()[:8]:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("[Page "):
            continue
        match = re.match(r"^(\d+(?:\.\d+)*)(?:\.|\s+|-|:)\s*(.{3,120})$", cleaned)
        if match:
            return {"section_number": match.group(1), "section_title": match.group(2).strip()}
        if re.match(r"^(section|seccion|clause|clausula)\b", cleaned, flags=re.IGNORECASE):
            return {"section_title": cleaned[:140]}
    clause = re.search(r"\b(?:clause|clausula|req(?:uirement)?\.?)\s*([A-Za-z0-9.-]+)\b", content, flags=re.IGNORECASE)
    if clause:
        return {"clause_ref": clause.group(1)}
    return {}


def detect_requirement_type(content: str) -> str | None:
    """Classify common QMS requirement language."""

    lowered = content.lower()
    if any(term in lowered for term in ("shall", "must", "debe ", "debera", "obligatorio", "required")):
        return "requirement"
    if any(term in lowered for term in ("record", "registro", "evidence", "evidencia")):
        return "evidence"
    if any(term in lowered for term in ("responsible", "responsable", "owner")):
        return "responsibility"
    if any(term in lowered for term in ("corrective action", "accion correctiva", "acción correctiva", "capa")):
        return "action"
    return None


def detect_key_terms(content: str) -> list[str]:
    """Extract a small controlled vocabulary useful for filtering and review."""

    terms = []
    vocabulary = [
        "CAPA",
        "SOP",
        "QMS",
        "auditoria",
        "audit",
        "reclamo",
        "complaint",
        "especificacion",
        "specification",
        "riesgo",
        "risk",
        "causa raiz",
        "root cause",
    ]
    lowered = content.lower()
    for term in vocabulary:
        if term.lower() in lowered:
            terms.append(term)
    return terms[:12]


def detect_entities(content: str) -> dict[str, list[str]]:
    """Detect common quality-system identifiers."""

    patterns = {
        "capa": r"\bCAPA[-_ ]?\d{4}[-_ ]?\d+\b",
        "audit": r"\bAUD[-_ ]?\d{4}[-_ ]?\d+\b",
        "sku": r"\bSKU[-_ ]?[A-Za-z0-9.-]+\b",
        "lot": r"\b(?:LOT|LOTE|BATCH)[-_ ]?[A-Za-z0-9.-]+\b",
    }
    entities: dict[str, list[str]] = {}
    for key, pattern in patterns.items():
        matches = sorted({match.group(0) for match in re.finditer(pattern, content, flags=re.IGNORECASE)})
        if matches:
            entities[key] = matches[:10]
    return entities


def _build_full_text(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages and track character spans by page."""

    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        cleaned = _normalize_text(page.text)
        if not cleaned:
            continue
        prefix = f"\n\n[Page {page.page_number}]\n"
        piece = prefix + cleaned
        start = cursor
        end = cursor + len(piece)
        parts.append(piece)
        spans.append((start, end, page.page_number))
        cursor = end

    return "".join(parts).strip(), spans


def _normalize_text(text: str) -> str:
    """Normalize line whitespace while preserving paragraph breaks."""

    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    compact: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                compact.append("")
            blank_seen = True
            continue
        compact.append(" ".join(line.split()))
        blank_seen = False
    return "\n".join(compact).strip()


def _snap_to_boundary(text: str, start: int, raw_end: int) -> int:
    """Move a chunk boundary to a nearby sentence or whitespace boundary."""

    if raw_end >= len(text):
        return len(text)
    window = text[start:raw_end]
    for boundary in ("\n\n", ". ", "? ", "! ", "; ", "\n", " "):
        position = window.rfind(boundary)
        if position >= max(80, len(window) // 2):
            return start + position + len(boundary)
    return raw_end


def _pages_for_range(spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    """Return page span overlapping a character range."""

    pages = [page for span_start, span_end, page in spans if span_start < end and span_end > start]
    if not pages:
        return 1, 1
    return min(pages), max(pages)
