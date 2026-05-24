"""Application configuration loaded from environment variables.

This module centralizes all runtime settings for the RAG application:
database access, OpenAI model settings, and retrieval/chunking behavior.
Settings are read from the project `.env` file and exposed as frozen
dataclasses so the rest of the code can pass configuration explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


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
    """Read a boolean environment variable."""

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
    """OpenAI API and model settings.

    Attributes
    ----------
    api_key
        OpenAI API key.
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
    """

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

    @property
    def has_real_api_key(self) -> bool:
        """Return whether the configured API key looks usable.

        Returns
        -------
        bool
            True when the key is non-empty and does not look like a placeholder.
        """

        value = (self.api_key or "").strip()
        if not value:
            return False
        lowered = value.lower()
        placeholders = ("mock", "replace", "placeholder", "your_", "your-", "sk-xxx")
        return not any(marker in lowered for marker in placeholders)


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


def get_settings() -> Settings:
    """Build application settings from environment variables.

    Returns
    -------
    Settings
        Parsed database, OpenAI, and RAG settings.
    """

    load_env()
    base_url = _optional_url_env("OPENAI_BASE_URL")
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
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            embedding_dim=_int_env("OPENAI_EMBEDDING_DIM", 2000),
            embedding_batch_size=_int_env("OPENAI_EMBEDDING_BATCH_SIZE", 64),
            embedding_max_batch_chars=_int_env("OPENAI_EMBEDDING_MAX_BATCH_CHARS", 240000),
            temperature=_float_env("OPENAI_TEMPERATURE", 0.2),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            verbosity=os.getenv("OPENAI_VERBOSITY", "high"),
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
