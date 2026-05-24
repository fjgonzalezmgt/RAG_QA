"""LLM-assisted metadata extraction for ingested quality documents."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from .config import DEFAULT_OPENAI_BASE_URL, OpenAISettings
from .embeddings import validate_base_url
from .text_splitter import TextChunk


DOCUMENT_METADATA_FIELDS = {
    "document_type",
    "document_code",
    "revision",
    "lifecycle_status",
    "document_date",
    "effective_date",
    "review_due_date",
    "plant",
    "process",
    "product",
    "customer",
    "owner_area",
    "approval_status",
    "approved_by",
    "approved_at",
    "qms_process",
    "source_system",
    "source_record_id",
    "confidentiality_level",
    "risk_level",
}

STRONG_DETERMINISTIC_FIELDS = {
    "document_type",
    "document_code",
    "revision",
    "plant",
    "process",
    "product",
    "customer",
    "document_date",
}

DOCUMENT_TYPE_ALIASES = {
    "PROCEDIMIENTO": "PROCEDURE",
    "PROCEDURE": "PROCEDURE",
    "SOP": "SOP",
    "AUDITORIA": "AUDIT",
    "AUDIT": "AUDIT",
    "RECLAMO": "COMPLAINT",
    "COMPLAINT": "COMPLAINT",
    "ESPECIFICACION": "SPECIFICATION",
    "SPECIFICATION": "SPECIFICATION",
    "CAPA": "CAPA",
    "DMAIC": "DMAIC",
    "INDICADOR": "KPI",
    "KPI": "KPI",
    "QMS": "QMS",
}


class MetadataEnrichmentClient:
    """Extract document-level metadata with an LLM and conservative validation."""

    def __init__(self, settings: OpenAISettings):
        self.settings = settings
        base_url = settings.base_url or DEFAULT_OPENAI_BASE_URL
        validate_base_url(base_url)
        self.client = OpenAI(api_key=settings.api_key, base_url=base_url)

    def enrich_document(
        self,
        pdf_path: Path,
        chunks: Sequence[TextChunk],
        existing_metadata: Mapping[str, object],
        max_chars: int,
    ) -> dict[str, object]:
        """Return metadata merged with LLM suggestions."""

        if not self.settings.has_real_api_key:
            logger.warning("LLM metadata enrichment skipped: OPENAI_API_KEY is missing or placeholder.")
            return dict(existing_metadata)

        sample = build_document_sample(chunks, max_chars=max_chars)
        if not sample:
            logger.warning("LLM metadata enrichment skipped for '{}': empty text sample.", pdf_path.name)
            return dict(existing_metadata)

        prompt = build_metadata_prompt(pdf_path.name, sample)
        try:
            raw_text = self._call_model(prompt)
            suggested = parse_metadata_json(raw_text)
        except Exception as exc:
            logger.warning("LLM metadata enrichment failed for '{}': {}", pdf_path.name, exc)
            return dict(existing_metadata)

        merged = merge_metadata(existing_metadata, suggested)
        logger.info(
            "LLM metadata enrichment finished for '{}'. suggested_fields={}, accepted_fields={}.",
            pdf_path.name,
            sorted(suggested.keys()),
            sorted(set(merged) - set(existing_metadata)),
        )
        return merged

    def _call_model(self, prompt: str) -> str:
        """Call the configured model and return response text."""

        if self.settings.chat_model.startswith(("gpt-5", "o")) and hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.settings.chat_model,
                input=[
                    {
                        "role": "developer",
                        "content": (
                            "Extract QMS document metadata. Return only one compact JSON object. "
                            "Use null for unknown values and do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                reasoning={"effort": self.settings.reasoning_effort},
                text={"verbosity": "low"},
            )
            return extract_response_text(response)

        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract QMS document metadata. Return only one compact JSON object. "
                        "Use null for unknown values and do not invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()


def build_document_sample(chunks: Sequence[TextChunk], max_chars: int) -> str:
    """Build a representative text sample from the first chunks."""

    parts: list[str] = []
    remaining = max(0, max_chars)
    for chunk in chunks:
        if remaining <= 0:
            break
        block = f"[pages {chunk.page_start}-{chunk.page_end}]\n{chunk.content.strip()}"
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if block:
            parts.append(block)
            remaining -= len(block)
    return "\n\n---\n\n".join(parts)


def build_metadata_prompt(file_name: str, sample: str) -> str:
    """Build the extraction prompt."""

    fields = ", ".join(sorted(DOCUMENT_METADATA_FIELDS))
    return (
        f"Source file name for traceability only: {file_name}\n\n"
        f"Extract these fields when directly supported by the text: {fields}.\n"
        "Also include confidence from 0.0 to 1.0 and evidence_notes as a short object mapping fields to evidence.\n"
        "Dates must be YYYY-MM-DD. document_type must use codes such as SOP, PROCEDURE, CAPA, AUDIT, "
        "COMPLAINT, SPECIFICATION, QUALITY_REPORT, LESSON_LEARNED, DMAIC, KPI or QMS.\n"
        "Do not infer metadata from the file name. Use only the document text sample as evidence.\n"
        "Return JSON only.\n\n"
        f"Document text sample:\n{sample}"
    )


def parse_metadata_json(text: str) -> dict[str, object]:
    """Parse and sanitize LLM JSON metadata."""

    raw = extract_json_object(text)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Metadata response must be a JSON object")

    clean: dict[str, object] = {}
    for key, value in parsed.items():
        if key in DOCUMENT_METADATA_FIELDS:
            normalized = normalize_metadata_value(key, value)
            if normalized not in (None, "", [], {}):
                clean[key] = normalized
        elif key in {"confidence", "evidence_notes"}:
            clean[key] = value
    return clean


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from a model response."""

    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in metadata response")
    return stripped[start : end + 1]


def normalize_metadata_value(key: str, value: Any) -> object:
    """Normalize one extracted metadata value."""

    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "n/a", "na", "none", "null"}:
        return None
    if key.endswith("_date") or key in {"document_date", "effective_date", "review_due_date"}:
        return text if re.match(r"^\d{4}-\d{2}-\d{2}$", text) else None
    if key == "document_type":
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", text).upper().strip("_")
        return DOCUMENT_TYPE_ALIASES.get(normalized, normalized)
    return text


def merge_metadata(existing: Mapping[str, object], suggested: Mapping[str, object]) -> dict[str, object]:
    """Merge LLM metadata with file-content suggestions as the preferred source."""

    merged = dict(existing)
    accepted: dict[str, object] = {}
    overridden: dict[str, dict[str, object]] = {}
    for key, value in suggested.items():
        if key in {"confidence", "evidence_notes"}:
            continue
        if key not in DOCUMENT_METADATA_FIELDS or value in (None, "", [], {}):
            continue
        current = merged.get(key)
        if current and current != value:
            overridden[key] = {"previous": current, "llm": value}
            merged[key] = value
            accepted[key] = value
        elif not current:
            merged[key] = value
            accepted[key] = value

    merged["llm_metadata"] = {
        "enabled": True,
        "confidence": suggested.get("confidence"),
        "accepted_fields": accepted,
        "overridden_fields": overridden,
        "evidence_notes": suggested.get("evidence_notes", {}),
    }
    return merged


def extract_response_text(response: object) -> str:
    """Extract text from an OpenAI Responses API object."""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                fragments.append(str(text))
    return "\n".join(fragments).strip()
