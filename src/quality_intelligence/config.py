"""Application configuration loaded from environment variables.

This module centralizes all runtime settings for the RAG application:
database access, OpenAI-compatible model provider settings, and
retrieval/chunking behavior. Settings are read from the project `.env` file
and exposed as frozen dataclasses so the rest of the code can pass
configuration explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_OPENAI = "openai"
PROVIDER_LM_STUDIO = "lm_studio"
SUPPORTED_AI_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_LM_STUDIO}
MAX_EMBEDDING_DIM = 2000
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_LM_STUDIO_CHAT_MODEL = "nvidia/nemotron-3-nano-4b"
DEFAULT_LM_STUDIO_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v2-moe"
DEFAULT_LM_STUDIO_EMBEDDING_DIM = 768
DEFAULT_LM_STUDIO_DOCUMENT_PREFIX = "search_document: "
DEFAULT_LM_STUDIO_QUERY_PREFIX = "search_query: "


def load_env() -> None:
    """Load local environment settings from the project `.env` file."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value returned when the variable is empty or missing.

    Returns
    -------
    int
        Parsed integer value.
    """

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    """Read a floating-point environment variable.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value returned when the variable is empty or missing.

    Returns
    -------
    float
        Parsed floating-point value.
    """

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value returned when the variable is empty or missing.

    Returns
    -------
    bool
        Parsed boolean value.
    """

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "si"}


def _optional_url_env(name: str) -> str | None:
    """Read and normalize an optional URL environment variable.

    Parameters
    ----------
    name
        Environment variable name.

    Returns
    -------
    str or None
        Normalized URL, or None when the variable is empty.

    Raises
    ------
    ValueError
        If the value cannot be interpreted as an HTTP(S) URL.
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    value = raw_value.strip().strip('"').strip("'")
    if not value:
        return None

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value.rstrip("/")

    if not parsed.scheme and parsed.netloc == "" and not value.startswith("/"):
        normalized = f"https://{value}".rstrip("/")
        logger.warning("{} did not include a protocol; normalized to '{}'.", name, normalized)
        return normalized

    raise ValueError(
        f"{name} must be empty or a full URL starting with http:// or https://. "
        f"Current value is invalid: {value!r}"
    )


def normalize_ai_provider(value: str | None) -> str:
    """Normalize an AI provider identifier.

    Parameters
    ----------
    value
        Raw provider text from the environment or UI.

    Returns
    -------
    str
        Canonical provider key.

    Raises
    ------
    ValueError
        If the provider is not supported.
    """

    raw = (value or PROVIDER_OPENAI).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "open_ai": PROVIDER_OPENAI,
        "openai": PROVIDER_OPENAI,
        "lmstudio": PROVIDER_LM_STUDIO,
        "lm_studio": PROVIDER_LM_STUDIO,
        "local": PROVIDER_LM_STUDIO,
        "local_llm": PROVIDER_LM_STUDIO,
        "local_models": PROVIDER_LM_STUDIO,
    }
    provider = aliases.get(normalized, normalized)
    if provider not in SUPPORTED_AI_PROVIDERS:
        raise ValueError(
            f"AI_PROVIDER must be one of {sorted(SUPPORTED_AI_PROVIDERS)}. "
            f"Current value is invalid: {value!r}"
        )
    return provider


def normalize_embedding_model_id(model: str) -> str:
    """Normalize an embedding model identifier for local default lookup."""

    return (model or "").strip().lower()


def local_embedding_model_defaults(model: str) -> dict[str, object]:
    """Return local embedding defaults for known LM Studio models.

    Parameters
    ----------
    model
        Embedding model identifier used by LM Studio.

    Returns
    -------
    dict[str, object]
        Defaults with ``embedding_dim``, ``document_prefix``, and
        ``query_prefix``.
    """

    normalized = normalize_embedding_model_id(model)
    if normalized in {
        "text-embedding-nomic-embed-text-v2-moe",
        "nomic-ai/nomic-embed-text-v2-moe",
        "nomic-embed-text-v2-moe",
    }:
        return {
            "embedding_dim": DEFAULT_LM_STUDIO_EMBEDDING_DIM,
            "document_prefix": DEFAULT_LM_STUDIO_DOCUMENT_PREFIX,
            "query_prefix": DEFAULT_LM_STUDIO_QUERY_PREFIX,
        }
    return {
        "embedding_dim": DEFAULT_LM_STUDIO_EMBEDDING_DIM,
        "document_prefix": "",
        "query_prefix": "",
    }


@dataclass(frozen=True)
class DatabaseSettings:
    """Database connection settings.

    Attributes
    ----------
    host
        PostgreSQL server hostname.
    port
        PostgreSQL server port.
    name
        Database name.
    user
        Database user.
    password
        Database password.
    sslmode
        PostgreSQL SSL mode.
    extensions_schema
        Schema where shared PostgreSQL extensions such as pgvector live.
    """

    host: str
    port: int
    name: str
    user: str
    password: str
    sslmode: str = "prefer"
    extensions_schema: str = "extensions"


@dataclass(frozen=True)
class OpenAISettings:
    """OpenAI-compatible API and model settings.

    Attributes
    ----------
    provider
        Model provider key: ``openai`` or ``lm_studio``.
    api_key
        API key. LM Studio accepts a placeholder such as ``lm-studio``.
    base_url
        Optional OpenAI-compatible API base URL.
    chat_model
        Model used to generate final RAG answers.
    embedding_model
        Model used to embed chunks and queries.
    embedding_dim
        Embedding vector dimension stored in pgvector.
    embedding_batch_size
        Maximum number of texts sent in one embeddings API request.
    embedding_max_batch_chars
        Maximum approximate character budget for each embeddings batch.
    temperature
        Sampling temperature for models that support it.
    reasoning_effort
        Reasoning effort for compatible reasoning models.
    verbosity
        Output verbosity for compatible GPT-5 family models.
    embedding_document_prefix
        Prefix applied before document/chunk text when embedding.
    embedding_query_prefix
        Prefix applied before retrieval questions when embedding.
    """

    provider: str
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_max_batch_chars: int
    temperature: float
    reasoning_effort: str
    verbosity: str
    embedding_document_prefix: str = ""
    embedding_query_prefix: str = ""

    @property
    def normalized_provider(self) -> str:
        """Return the canonical provider key."""

        return normalize_ai_provider(self.provider)

    @property
    def is_local_provider(self) -> bool:
        """Return whether the provider is LM Studio/local."""

        return self.normalized_provider == PROVIDER_LM_STUDIO

    @property
    def provider_label(self) -> str:
        """Return a user-facing provider label."""

        if self.is_local_provider:
            return "LM Studio local"
        return "OpenAI"

    @property
    def effective_base_url(self) -> str:
        """Return the base URL used by the OpenAI SDK client."""

        if self.base_url:
            return self.base_url
        if self.is_local_provider:
            return DEFAULT_LM_STUDIO_BASE_URL
        return DEFAULT_OPENAI_BASE_URL

    @property
    def effective_api_key(self) -> str:
        """Return the API key passed to the OpenAI SDK client."""

        if self.is_local_provider and not (self.api_key or "").strip():
            return DEFAULT_LM_STUDIO_API_KEY
        return self.api_key

    @property
    def has_real_api_key(self) -> bool:
        """Return whether the configured API key looks usable.

        Returns
        -------
        bool
            True when the key is non-empty and does not look like a placeholder.
        """

        if self.is_local_provider:
            return True
        value = (self.api_key or "").strip()
        if not value:
            return False
        lowered = value.lower()
        placeholders = ("mock", "replace", "placeholder", "your_", "your-", "sk-xxx")
        return not any(marker in lowered for marker in placeholders)

    @property
    def uses_openai_responses_api(self) -> bool:
        """Return whether to use OpenAI Responses API reasoning options."""

        model = self.chat_model
        return self.normalized_provider == PROVIDER_OPENAI and (model.startswith("gpt-5") or model.startswith("o"))


@dataclass(frozen=True)
class RagSettings:
    """Retrieval, ingestion, and chunking settings.

    Attributes
    ----------
    domain
        Logical domain and PostgreSQL schema name.
    pdf_dir
        Root folder containing PDF files to ingest.
    chunk_size
        Approximate character length of each text chunk.
    chunk_overlap
        Character overlap between neighboring chunks.
    top_k
        Number of final chunks sent to the LLM.
    candidate_k
        Number of initial vector matches considered before diversification.
    max_chunks_per_document
        Maximum number of selected chunks per source document.
    max_context_chars
        Maximum context characters sent to the LLM.
    """

    domain: str
    pdf_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidate_k: int
    max_chunks_per_document: int
    max_context_chars: int
    recursive_pdf_scan: bool
    pdf_text_fallback: bool
    llm_metadata_enrichment: bool
    llm_metadata_max_chars: int


@dataclass(frozen=True)
class Settings:
    """Complete application settings tree."""

    db: DatabaseSettings
    openai: OpenAISettings
    rag: RagSettings


def provider_default_settings(settings: OpenAISettings, provider: str) -> OpenAISettings:
    """Return settings seeded with defaults for a provider switch.

    Parameters
    ----------
    settings
        Existing settings used for shared tunables such as batch sizes.
    provider
        Target provider key.

    Returns
    -------
    OpenAISettings
        Settings with provider-specific endpoint, model, key, and prefixes.
    """

    normalized = normalize_ai_provider(provider)
    if normalized == PROVIDER_LM_STUDIO:
        embedding_model = os.getenv("LM_STUDIO_EMBEDDING_MODEL", DEFAULT_LM_STUDIO_EMBEDDING_MODEL)
        embedding_defaults = local_embedding_model_defaults(embedding_model)
        return replace(
            settings,
            provider=PROVIDER_LM_STUDIO,
            api_key=os.getenv("LM_STUDIO_API_KEY", DEFAULT_LM_STUDIO_API_KEY),
            base_url=_optional_url_env("LM_STUDIO_BASE_URL") or DEFAULT_LM_STUDIO_BASE_URL,
            chat_model=os.getenv("LM_STUDIO_CHAT_MODEL", DEFAULT_LM_STUDIO_CHAT_MODEL),
            embedding_model=embedding_model,
            embedding_dim=_int_env("LM_STUDIO_EMBEDDING_DIM", int(embedding_defaults["embedding_dim"])),
            embedding_document_prefix=os.getenv(
                "LM_STUDIO_EMBEDDING_DOCUMENT_PREFIX",
                str(embedding_defaults["document_prefix"]),
            ),
            embedding_query_prefix=os.getenv(
                "LM_STUDIO_EMBEDDING_QUERY_PREFIX",
                str(embedding_defaults["query_prefix"]),
            ),
        )

    return replace(
        settings,
        provider=PROVIDER_OPENAI,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=_optional_url_env("OPENAI_BASE_URL"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        embedding_dim=_int_env("OPENAI_EMBEDDING_DIM", 2000),
        embedding_document_prefix=os.getenv("OPENAI_EMBEDDING_DOCUMENT_PREFIX", ""),
        embedding_query_prefix=os.getenv("OPENAI_EMBEDDING_QUERY_PREFIX", ""),
    )


def get_settings() -> Settings:
    """Build application settings from environment variables.

    Returns
    -------
    Settings
        Parsed database, model-provider, and RAG settings.
    """

    load_env()
    provider = normalize_ai_provider(os.getenv("AI_PROVIDER") or os.getenv("MODEL_PROVIDER") or PROVIDER_OPENAI)
    if provider == PROVIDER_LM_STUDIO:
        base_url = _optional_url_env("LM_STUDIO_BASE_URL") or DEFAULT_LM_STUDIO_BASE_URL
        api_key = os.getenv("LM_STUDIO_API_KEY", DEFAULT_LM_STUDIO_API_KEY)
        chat_model = os.getenv("LM_STUDIO_CHAT_MODEL", DEFAULT_LM_STUDIO_CHAT_MODEL)
        embedding_model = os.getenv("LM_STUDIO_EMBEDDING_MODEL", DEFAULT_LM_STUDIO_EMBEDDING_MODEL)
        embedding_defaults = local_embedding_model_defaults(embedding_model)
        embedding_dim = _int_env("LM_STUDIO_EMBEDDING_DIM", int(embedding_defaults["embedding_dim"]))
        embedding_document_prefix = os.getenv(
            "LM_STUDIO_EMBEDDING_DOCUMENT_PREFIX",
            str(embedding_defaults["document_prefix"]),
        )
        embedding_query_prefix = os.getenv(
            "LM_STUDIO_EMBEDDING_QUERY_PREFIX",
            str(embedding_defaults["query_prefix"]),
        )
    else:
        base_url = _optional_url_env("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY", "")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_dim = _int_env("OPENAI_EMBEDDING_DIM", 2000)
        embedding_document_prefix = os.getenv("OPENAI_EMBEDDING_DOCUMENT_PREFIX", "")
        embedding_query_prefix = os.getenv("OPENAI_EMBEDDING_QUERY_PREFIX", "")

    return Settings(
        db=DatabaseSettings(
            host=os.getenv("DB_HOST", "localhost"),
            port=_int_env("DB_PORT", 5432),
            name=os.getenv("DB_NAME", "RAG_DB"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            sslmode=os.getenv("DB_SSLMODE", "prefer"),
            extensions_schema=os.getenv("DB_EXTENSIONS_SCHEMA", "extensions"),
        ),
        openai=OpenAISettings(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embedding_batch_size=_int_env("OPENAI_EMBEDDING_BATCH_SIZE", 64),
            embedding_max_batch_chars=_int_env("OPENAI_EMBEDDING_MAX_BATCH_CHARS", 240000),
            temperature=_float_env("OPENAI_TEMPERATURE", 0.2),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            verbosity=os.getenv("OPENAI_VERBOSITY", "high"),
            embedding_document_prefix=embedding_document_prefix,
            embedding_query_prefix=embedding_query_prefix,
        ),
        rag=RagSettings(
            domain=os.getenv("RAG_DOMAIN", "quality_intelligence"),
            pdf_dir=(PROJECT_ROOT / os.getenv("RAG_PDF_DIR", "./quality_knowledge_base")).resolve(),
            chunk_size=_int_env("RAG_CHUNK_SIZE", 1800),
            chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", 220),
            top_k=_int_env("RAG_TOP_K", 18),
            candidate_k=_int_env("RAG_CANDIDATE_K", 80),
            max_chunks_per_document=_int_env("RAG_MAX_CHUNKS_PER_DOCUMENT", 2),
            max_context_chars=_int_env("RAG_MAX_CONTEXT_CHARS", 50000),
            recursive_pdf_scan=_bool_env("RAG_RECURSIVE_PDF_SCAN", True),
            pdf_text_fallback=_bool_env("RAG_PDF_TEXT_FALLBACK", True),
            llm_metadata_enrichment=_bool_env("RAG_LLM_METADATA_ENRICHMENT", False),
            llm_metadata_max_chars=_int_env("RAG_LLM_METADATA_MAX_CHARS", 12000),
        ),
    )
