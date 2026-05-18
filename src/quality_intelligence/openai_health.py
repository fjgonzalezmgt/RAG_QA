"""OpenAI API connectivity checks used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from openai import OpenAI

from .config import DEFAULT_OPENAI_BASE_URL, OpenAISettings
from .embeddings import validate_base_url


@dataclass(frozen=True)
class OpenAIHealth:
    """Result of an OpenAI API health check.

    Attributes
    ----------
    ok
        Whether the check succeeded.
    message
        User-facing status message.
    """

    ok: bool
    message: str


def check_openai_connection(settings: OpenAISettings) -> OpenAIHealth:
    """Verify that the configured OpenAI API key and model are usable.

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
        logger.warning("OpenAI health check skipped: API key missing or placeholder.")
        return OpenAIHealth(ok=False, message="OPENAI_API_KEY pendiente o placeholder.")

    base_url = settings.base_url or DEFAULT_OPENAI_BASE_URL
    validate_base_url(base_url)
    kwargs: dict[str, str] = {"api_key": settings.api_key, "base_url": base_url}

    try:
        logger.info("Checking OpenAI API connectivity with model '{}' using base_url '{}'.", settings.chat_model, base_url)
        client = OpenAI(**kwargs)
        client.models.retrieve(settings.chat_model)
        logger.success("OpenAI API connectivity OK.")
        return OpenAIHealth(ok=True, message=f"Conectado a OpenAI. Modelo disponible: {settings.chat_model}.")
    except Exception as exc:
        logger.exception("OpenAI API connectivity check failed.")
        return OpenAIHealth(ok=False, message=f"No se pudo conectar a OpenAI: {exc}")
