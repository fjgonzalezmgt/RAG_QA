from quality_intelligence.pdf_loader import PageText
from quality_intelligence.text_splitter import split_pages


def test_split_pages_preserves_page_span_and_detects_metadata():
    pages = [
        PageText(
            page_number=1,
            text=(
                "1. Scope\n"
                "The process shall keep CAPA-2026-008 records as evidence for SKU-100."
            ),
        )
    ]

    chunks = split_pages(pages, chunk_size=120, chunk_overlap=20)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_start == 1
    assert chunk.page_end == 1
    assert chunk.metadata["section_number"] == "1"
    assert chunk.metadata["section_title"] == "Scope"
    assert chunk.metadata["requirement_type"] == "requirement"
    assert "CAPA" in chunk.metadata["key_terms"]
    assert chunk.metadata["detected_entities"]["capa"] == ["CAPA-2026-008"]
    assert chunk.metadata["detected_entities"]["sku"] == ["SKU-100"]


def test_split_pages_rejects_invalid_overlap():
    pages = [PageText(page_number=1, text="Some content")]

    try:
        split_pages(pages, chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
