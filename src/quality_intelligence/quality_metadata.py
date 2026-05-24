"""Quality/QMS metadata helpers.

The production path should load metadata from QMS, ERP, MES, LIMS, complaint,
CAPA, or audit systems. For a portfolio/demo repository, these helpers infer a
useful first layer of metadata from controlled file names.
"""

from __future__ import annotations

import re
from pathlib import Path


DOCUMENT_TYPE_ALIASES = {
    "SOP": "SOP",
    "PROCEDURE": "PROCEDURE",
    "PROCEDIMIENTO": "PROCEDURE",
    "CAPA": "CAPA",
    "AUDIT": "AUDIT",
    "AUDITORIA": "AUDIT",
    "COMPLAINT": "COMPLAINT",
    "RECLAMO": "COMPLAINT",
    "SPEC": "SPECIFICATION",
    "SPECIFICATION": "SPECIFICATION",
    "ESPECIFICACION": "SPECIFICATION",
    "QUALITY_REPORT": "QUALITY_REPORT",
    "REPORTE_CALIDAD": "QUALITY_REPORT",
    "LESSON_LEARNED": "LESSON_LEARNED",
    "LECCION_APRENDIDA": "LESSON_LEARNED",
    "DMAIC": "DMAIC",
    "KPI": "KPI",
    "INDICADOR": "KPI",
    "QMS": "QMS",
}


def infer_quality_metadata(pdf_path: Path) -> dict[str, object]:
    """Infer QMS metadata from a PDF path and controlled file name.

    Recommended pattern:
    ``<document_type>__<plant>__<process>__<product-or-customer>__<code>__rev-<revision>.pdf``

    Parameters
    ----------
    pdf_path
        Source PDF path.

    Returns
    -------
    dict[str, object]
        Metadata inferred from the file path.
    """

    metadata: dict[str, object] = {}
    stem = pdf_path.stem
    parts = [clean_metadata_value(part) for part in stem.split("__") if clean_metadata_value(part)]

    if parts:
        document_type = normalize_document_type(parts[0])
        if document_type:
            metadata["document_type"] = document_type
            metadata["document_type_raw"] = parts[0]

    if len(parts) > 1:
        metadata["plant"] = parts[1]
    if len(parts) > 2:
        metadata["process"] = parts[2]
    if len(parts) > 3:
        metadata.update(classify_product_or_customer(parts[3]))
    if len(parts) > 4:
        metadata["document_code"] = parts[4]

    revision = find_revision(parts) or find_revision([stem])
    if revision:
        metadata["revision"] = revision

    date_value = find_iso_date(stem)
    if date_value:
        metadata["document_date"] = date_value

    folder_parts = [clean_metadata_value(part.name) for part in pdf_path.parents[:3]]
    if folder_parts:
        metadata["source_folder"] = " / ".join(reversed([part for part in folder_parts if part]))

    return metadata


def normalize_document_type(value: str) -> str | None:
    """Normalize a document type token.

    Parameters
    ----------
    value
        Raw document type token.

    Returns
    -------
    str or None
        Normalized document type code.
    """

    key = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).upper().strip("_")
    return DOCUMENT_TYPE_ALIASES.get(key)


def classify_product_or_customer(value: str) -> dict[str, str]:
    """Classify the fourth file-name token as product or customer.

    Parameters
    ----------
    value
        Token from the controlled file name.

    Returns
    -------
    dict[str, str]
        Either product or customer metadata.
    """

    lowered = value.lower()
    if lowered.startswith(("cliente-", "customer-", "cust-")):
        return {"customer": value}
    return {"product": value}


def find_revision(values: list[str]) -> str | None:
    """Find a revision marker such as rev-03 or revision_2.

    Parameters
    ----------
    values
        Candidate strings to inspect.

    Returns
    -------
    str or None
        Revision identifier when found.
    """

    for value in values:
        match = re.search(r"\b(?:rev|revision|ver|version)[-_ ]?([A-Za-z0-9.]+)\b", value, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def find_iso_date(value: str) -> str | None:
    """Find an ISO date in a string.

    Parameters
    ----------
    value
        String to inspect.

    Returns
    -------
    str or None
        ISO date when found.
    """

    match = re.search(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", value)
    return match.group(1) if match else None


def clean_metadata_value(value: str) -> str:
    """Normalize file-name metadata tokens for display and filtering.

    Parameters
    ----------
    value
        Raw token value.

    Returns
    -------
    str
        Cleaned metadata token.
    """

    text = value.strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
