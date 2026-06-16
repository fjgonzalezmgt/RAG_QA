from dataclasses import replace

from quality_intelligence.config import (
    DEFAULT_LM_STUDIO_API_KEY,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_LM_STUDIO_EMBEDDING_DIM,
    PROVIDER_LM_STUDIO,
    get_settings,
    local_embedding_model_defaults,
    provider_default_settings,
)
from quality_intelligence.embeddings import format_embedding_input, supports_dimensions, validate_embedding_dimensions


def test_lm_studio_provider_defaults_do_not_require_real_api_key():
    settings = provider_default_settings(get_settings().openai, PROVIDER_LM_STUDIO)
    settings = replace(settings, api_key="")

    assert settings.is_local_provider
    assert settings.has_real_api_key
    assert settings.effective_api_key == DEFAULT_LM_STUDIO_API_KEY
    assert settings.effective_base_url == DEFAULT_LM_STUDIO_BASE_URL
    assert settings.embedding_dim == DEFAULT_LM_STUDIO_EMBEDDING_DIM


def test_local_embedding_model_omits_dimensions_parameter():
    assert not supports_dimensions("text-embedding-nomic-embed-text-v2-moe")
    assert not supports_dimensions("sfr-embedding-mistral")
    assert supports_dimensions("text-embedding-3-large")


def test_unknown_local_embedding_defaults_stay_within_pgvector_limit():
    defaults = local_embedding_model_defaults("sfr-embedding-mistral")

    assert defaults["embedding_dim"] == DEFAULT_LM_STUDIO_EMBEDDING_DIM
    assert defaults["document_prefix"] == ""
    assert defaults["query_prefix"] == ""


def test_embedding_dimension_validation_rejects_mismatched_vectors():
    try:
        validate_embedding_dimensions([[0.1, 0.2, 0.3]], expected_dim=2)
    except ValueError as exc:
        assert "dimension 3" in str(exc)
        assert "configured pgvector dimension is 2" in str(exc)
    else:
        raise AssertionError("Expected mismatched embedding dimension to raise ValueError")


def test_format_embedding_input_applies_prefix_once():
    assert format_embedding_input("  hello\nworld ", "search_document: ") == "search_document: hello world"
    assert format_embedding_input("hello world", "search_document:") == "search_document: hello world"
    assert (
        format_embedding_input("search_document: hello world", "search_document: ")
        == "search_document: hello world"
    )
