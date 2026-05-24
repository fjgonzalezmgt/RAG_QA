from quality_intelligence.metadata_enrichment import (
    build_document_sample,
    build_metadata_prompt,
    merge_metadata,
    parse_metadata_json,
)
from quality_intelligence.text_splitter import TextChunk


def test_build_metadata_prompt_treats_filename_as_traceability_only():
    prompt = build_metadata_prompt(
        "SOP__FakePlant__FakeProcess__rev-99.pdf",
        "Document title: Customer Complaint Report\nEffective date: 2026-05-24",
    )

    assert "traceability only" in prompt
    assert "Do not infer metadata from the file name" in prompt
    assert "Customer Complaint Report" in prompt


def test_parse_metadata_json_normalizes_and_filters_fields():
    text = """
    ```json
    {
      "document_type": "auditoria",
      "effective_date": "2026-05-24",
      "review_due_date": "24/05/2027",
      "plant": "Planta Norte",
      "unsupported": "ignore me",
      "confidence": 0.82,
      "evidence_notes": {"effective_date": "header"}
    }
    ```
    """

    metadata = parse_metadata_json(text)

    assert metadata["document_type"] == "AUDIT"
    assert metadata["effective_date"] == "2026-05-24"
    assert "review_due_date" not in metadata
    assert metadata["plant"] == "Planta Norte"
    assert "unsupported" not in metadata
    assert metadata["confidence"] == 0.82


def test_merge_metadata_prefers_llm_content_suggestions_over_filename_fallback():
    existing = {"document_type": "SOP", "plant": "FakePlant", "pages": 4}
    suggested = {
        "document_type": "COMPLAINT",
        "plant": "Planta Norte",
        "effective_date": "2026-05-24",
        "confidence": 0.9,
        "evidence_notes": {"document_type": "title says complaint"},
    }

    merged = merge_metadata(existing, suggested)

    assert merged["document_type"] == "COMPLAINT"
    assert merged["plant"] == "Planta Norte"
    assert merged["effective_date"] == "2026-05-24"
    assert merged["pages"] == 4
    assert merged["llm_metadata"]["overridden_fields"]["document_type"]["previous"] == "SOP"
    assert merged["llm_metadata"]["accepted_fields"]["effective_date"] == "2026-05-24"


def test_build_document_sample_uses_chunk_content():
    chunks = [
        TextChunk(
            index=0,
            content="First page content with approval status.",
            page_start=1,
            page_end=1,
            token_count=8,
        ),
        TextChunk(
            index=1,
            content="Second page content.",
            page_start=2,
            page_end=2,
            token_count=4,
        ),
    ]

    sample = build_document_sample(chunks, max_chars=80)

    assert "[pages 1-1]" in sample
    assert "First page content" in sample
    assert len(sample) <= 90
