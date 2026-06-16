"""OpenAI-compatible API connectivity checks used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from openai import OpenAI

from .config import OpenAISettings
from .embeddings import supports_dimensions, validate_base_url, validate_embedding_dimensions


@dataclass(frozen=True)
class OpenAIHealth:
    """Result of a model provider health check.

    Attributes
    ----------
    ok
        Whether the check succeeded.
    message
        User-facing status message.
    """

    ok: bool
    message: str


def check_model_provider_connection(settings: OpenAISettings) -> OpenAIHealth:
    """Verify that the configured provider, chat model, and embeddings work.

    Parameters
    ----------
    settings
        OpenAI settings.

    Returns
    -------
    OpenAIHealth
        Connectivity result and user-facing message.
    """

    if not settings.has_real_api_key:
        logger.warning("{} health check skipped: API key missing or placeholder.", settings.provider_label)
        return OpenAIHealth(ok=False, message=f"{settings.provider_label}: API key pendiente o placeholder.")

    base_url = settings.effective_base_url
    validate_base_url(base_url)
    kwargs: dict[str, str] = {"api_key": settings.effective_api_key, "base_url": base_url}

    try:
        logger.info(
            "Checking {} connectivity with chat model '{}' and embedding model '{}' using base_url '{}'.",
            settings.provider_label,
            settings.chat_model,
            settings.embedding_model,
            base_url,
        )
        client = OpenAI(**kwargs)
        if settings.uses_openai_responses_api and hasattr(client, "responses"):
            client.responses.create(
                model=settings.chat_model,
                input="Responde solamente: ok",
                reasoning={"effort": settings.reasoning_effort},
                text={"verbosity": "low"},
            )
        else:
            client.chat.completions.create(
                model=settings.chat_model,
                messages=[{"role": "user", "content": "Responde solamente: ok"}],
                max_tokens=5,
            )

        embedding_request: dict[str, object] = {
            "model": settings.embedding_model,
            "input": ["ok"],
        }
        if supports_dimensions(settings.embedding_model):
            embedding_request["dimensions"] = settings.embedding_dim
        embedding_response = client.embeddings.create(**embedding_request)
        embeddings = [item.embedding for item in sorted(embedding_response.data, key=lambda item: item.index)]
        validate_embedding_dimensions(embeddings, settings.embedding_dim)

        logger.success("{} connectivity OK.", settings.provider_label)
        return OpenAIHealth(
            ok=True,
            message=(
                f"Conectado a {settings.provider_label}. "
                f"Chat: {settings.chat_model}. Embeddings: {settings.embedding_model}."
            ),
        )
    except Exception as exc:
        logger.exception("{} connectivity check failed.", settings.provider_label)
        return OpenAIHealth(ok=False, message=f"No se pudo conectar a {settings.provider_label}: {exc}")


def check_openai_connection(settings: OpenAISettings) -> OpenAIHealth:
    """Backward-compatible alias for existing imports."""

    return check_model_provider_connection(settings)
