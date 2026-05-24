"""OpenAI embeddings client and batching helpers.

The ingestion pipeline embeds document chunks in batches. Query-time retrieval
embeds a single user question. This module keeps batching, input normalization,
dimension handling, and base URL validation in one place.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from loguru import logger
from openai import OpenAI

from .config import DEFAULT_OPENAI_BASE_URL, OpenAISettings


class EmbeddingClient:
    """Client wrapper for OpenAI embeddings.

    Parameters
    ----------
    settings
        OpenAI settings containing API key, model, dimensions, and batching
        limits.
    """

    def __init__(self, settings: OpenAISettings):
        """Initialize the embeddings client.

        Parameters
        ----------
        settings
            OpenAI settings.
        """

        self.settings = settings
        self.client = _build_client(settings)

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed multiple texts using the OpenAI embeddings API.

        Parameters
        ----------
        texts
            Text strings to embed.
        batch_size
            Optional batch-size override. Defaults to configuration.

        Returns
        -------
        list[list[float]]
            Embedding vectors in the same order as ``texts``.
        """

        self._require_api_key()
        clean_texts = [normalize_embedding_text(text) for text in texts]
        effective_batch_size = batch_size or self.settings.embedding_batch_size
        logger.info(
            "Creating embeddings. texts={}, batch_size={}, max_batch_chars={}, model='{}', dimensions={}.",
            len(clean_texts),
            effective_batch_size,
            self.settings.embedding_max_batch_chars,
            self.settings.embedding_model,
            self.settings.embedding_dim,
        )
        embeddings: list[list[float]] = []

        for batch_index, batch in enumerate(
            _batched_by_size(
                clean_texts,
                max_items=effective_batch_size,
                max_chars=self.settings.embedding_max_batch_chars,
            ),
            start=1,
        ):
            batch_chars = sum(len(item) for item in batch)
            logger.debug("Embedding batch {}. items={}, chars={}.", batch_index, len(batch), batch_chars)
            response = self.client.embeddings.create(**self._embedding_request(batch))
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in ordered)

        logger.success("Embeddings created. vectors={}.", len(embeddings))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single retrieval query.

        Parameters
        ----------
        text
            User question or search query.

        Returns
        -------
        list[float]
            Query embedding vector.
        """

        logger.info("Embedding query. chars={}.", len(text))
        return self.embed_texts([text], batch_size=1)[0]

    def _embedding_request(self, batch: list[str]) -> dict[str, object]:
        """Build an embeddings API request payload.

        Parameters
        ----------
        batch
            Normalized text batch.

        Returns
        -------
        dict[str, object]
            Request payload for the embeddings API.
        """

        request: dict[str, object] = {
            "model": self.settings.embedding_model,
            "input": batch,
        }
        if supports_dimensions(self.settings.embedding_model):
            request["dimensions"] = self.settings.embedding_dim
        return request

    def _require_api_key(self) -> None:
        """Raise when no usable API key is configured."""

        if not self.settings.has_real_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing or still has a placeholder value.")


def _build_client(settings: OpenAISettings) -> OpenAI:
    """Create an OpenAI SDK client with an explicit base URL.

    Parameters
    ----------
    settings
        OpenAI settings.

    Returns
    -------
    OpenAI
        Configured OpenAI SDK client.
    """

    base_url = settings.base_url or DEFAULT_OPENAI_BASE_URL
    validate_base_url(base_url)
    kwargs: dict[str, str] = {"api_key": settings.api_key, "base_url": base_url}
    return OpenAI(**kwargs)


def normalize_embedding_text(text: str) -> str:
    """Normalize whitespace before embedding text.

    Parameters
    ----------
    text
        Raw text to normalize.

    Returns
    -------
    str
        Text with compact whitespace.
    """

    return " ".join((text or "").split())


def supports_dimensions(model: str) -> bool:
    """Return whether an embedding model supports custom dimensions.

    Parameters
    ----------
    model
        Embedding model name.

    Returns
    -------
    bool
        True when the model supports the ``dimensions`` parameter.
    """

    return model.startswith("text-embedding-3")


def validate_base_url(base_url: str) -> None:
    """Validate that a base URL is absolute and HTTP(S).

    Parameters
    ----------
    base_url
        OpenAI-compatible API base URL.

    Raises
    ------
    ValueError
        If the URL is not absolute HTTP(S).
    """

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "OPENAI_BASE_URL must start with http:// or https://. "
            f"Current value is invalid: {base_url!r}"
        )


def _batched_by_size(items: list[str], max_items: int, max_chars: int) -> Iterable[list[str]]:
    """Yield batches constrained by item count and character budget.

    Parameters
    ----------
    items
        Texts to batch.
    max_items
        Maximum number of texts per batch.
    max_chars
        Maximum approximate characters per batch.

    Yields
    ------
    list[str]
        Next batch of texts.
    """

    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    batch: list[str] = []
    batch_chars = 0

    for item in items:
        item_chars = len(item)
        would_exceed_items = len(batch) >= max_items
        would_exceed_chars = batch and batch_chars + item_chars > max_chars

        if would_exceed_items or would_exceed_chars:
            yield batch
            batch = []
            batch_chars = 0

        batch.append(item)
        batch_chars += item_chars

    if batch:
        yield batch
