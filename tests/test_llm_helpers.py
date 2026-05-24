from quality_intelligence.db import SearchResult
from quality_intelligence.llm import (
    build_context_block,
    extract_response_text,
    format_metadata,
    prefers_responses_api,
)
from quality_intelligence.retriever import RetrievedContext


def test_build_context_block_includes_operational_metadata():
    result = SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        file_name="SOP.pdf",
        title=None,
        chunk_index=0,
        page_start=2,
        page_end=3,
        content="The procedure shall keep records.",
        score=0.91,
        metadata={
            "document_type": "SOP",
            "revision": "03",
            "approval_status": "approved",
            "is_current": True,
            "plant": "PlantaNorte",
        },
    )

    block = build_context_block([RetrievedContext(source_id="S1", result=result)], max_chars=1000)

    assert "[S1] SOP.pdf, pp. 2-3" in block
    assert "document_type=SOP" in block
    assert "revision=03" in block
    assert "approval_status=approved" in block
    assert "is_current=True" in block
    assert "The procedure shall keep records." in block


def test_format_metadata_omits_empty_values():
    text = format_metadata({"document_code": "SOP-QA-014", "revision": "", "plant": None})

    assert text == "document_code=SOP-QA-014"


def test_prefers_responses_api_for_reasoning_models():
    assert prefers_responses_api("gpt-5.2")
    assert prefers_responses_api("o3")
    assert not prefers_responses_api("gpt-4.1")


def test_extract_response_text_supports_output_text_attribute():
    class Response:
        output_text = " answer "

    assert extract_response_text(Response()) == "answer"
