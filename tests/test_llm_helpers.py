from quality_intelligence.config import OpenAISettings, PROVIDER_LM_STUDIO, PROVIDER_OPENAI
from quality_intelligence.db import SearchResult
from quality_intelligence.llm import (
    build_context_block,
    context_char_limits,
    extract_response_text,
    format_metadata,
    history_char_limit,
    is_context_length_error,
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
    assert prefers_responses_api("gpt-5.6-luna")
    assert prefers_responses_api("o3")
    assert not prefers_responses_api("gpt-4.1")


def test_extract_response_text_supports_output_text_attribute():
    class Response:
        output_text = " answer "

    assert extract_response_text(Response()) == "answer"


def test_context_char_limits_cap_local_provider():
    settings = _settings(PROVIDER_LM_STUDIO)

    assert context_char_limits(settings, 50000) == [18000, 12000, 8000, 4000]
    assert history_char_limit(settings, 1) == 1200
    assert history_char_limit(settings, 2) == 500


def test_context_char_limits_keep_openai_requested_budget():
    settings = _settings(PROVIDER_OPENAI)

    assert context_char_limits(settings, 50000) == [50000, 24000, 12000, 8000, 4000]
    assert history_char_limit(settings, 1) == 4000


def test_context_length_error_detection_matches_lm_studio_message():
    error = RuntimeError("n_keep: 10103>= n_ctx: 8192. Try a larger context length.")

    assert is_context_length_error(error)


def _settings(provider: str) -> OpenAISettings:
    return OpenAISettings(
        provider=provider,
        api_key="test",
        base_url=None,
        chat_model="gpt-4.1",
        embedding_model="text-embedding-3-large",
        embedding_dim=768,
        embedding_batch_size=1,
        embedding_max_batch_chars=1000,
        temperature=0.2,
        reasoning_effort="medium",
        verbosity="high",
    )
