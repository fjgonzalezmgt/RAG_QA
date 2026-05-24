"""LLM answer generation for retrieved context.

This module builds the final prompt from retrieved chunks and recent chat
history, then calls OpenAI Chat Completions. The chat history is used only for
conversation continuity; retrieved chunks remain the evidence source.
"""

from __future__ import annotations

from openai import OpenAI
from loguru import logger

from .config import DEFAULT_OPENAI_BASE_URL, OpenAISettings
from .domain_profiles import DomainProfile
from .embeddings import validate_base_url
from .retriever import RetrievedContext


class LLMClient:
    """OpenAI chat client for final RAG answers.

    Parameters
    ----------
    settings
        OpenAI settings controlling model, reasoning effort, and verbosity.
    """

    def __init__(self, settings: OpenAISettings):
        """Initialize the LLM client.

        Parameters
        ----------
        settings
            OpenAI settings.
        """

        self.settings = settings
        self.client = _build_client(settings)

    def answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        profile: DomainProfile,
        max_context_chars: int,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate an answer from a question and retrieved contexts.

        Parameters
        ----------
        question
            User question.
        contexts
            Retrieved context chunks.
        profile
            Domain prompt profile.
        max_context_chars
            Maximum context characters to include in the prompt.
        chat_history
            Optional recent chat messages for conversational continuity.

        Returns
        -------
        str
            Model-generated answer.
        """

        self._require_api_key()
        logger.info(
            "Generating LLM answer. model='{}', contexts={}, history_messages={}.",
            self.settings.chat_model,
            len(contexts),
            len(chat_history or []),
        )
        context_text = build_context_block(contexts, max_context_chars)
        if not context_text:
            logger.warning("LLM answer skipped because retrieved context is empty.")
            return "No encontre contexto relevante en la base vectorial para responder."

        history_text = build_history_block(chat_history or [])
        user_prompt = (
            "Pregunta:\n"
            f"{question.strip()}\n\n"
            "Historial reciente de la conversacion:\n"
            f"{history_text}\n\n"
            "Contexto recuperado:\n"
            f"{context_text}\n\n"
            "Instrucciones: responde con base en el contexto. Si el contexto no "
            "alcanza, dilo explicitamente. Usa el historial solo para entender "
            "referencias conversacionales, no como evidencia documental. Incluye "
            "citas [S1], [S2] donde aplique."
        )

        if prefers_responses_api(self.settings.chat_model):
            return self._answer_with_responses(profile.system_prompt, user_prompt)

        request = {
            "model": self.settings.chat_model,
            "messages": [
                {"role": instruction_role(self.settings.chat_model), "content": profile.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if supports_temperature(self.settings.chat_model):
            request["temperature"] = self.settings.temperature
        else:
            logger.info(
                "Omitting temperature for model '{}' because it only supports the default value.",
                self.settings.chat_model,
            )
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        answer = (content or "").strip()
        logger.success("LLM answer generated. chars={}.", len(answer))
        return answer

    def _answer_with_responses(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer through the Responses API for GPT-5 models."""

        request = {
            "model": self.settings.chat_model,
            "input": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": self.settings.reasoning_effort},
            "text": {"verbosity": self.settings.verbosity},
        }
        logger.info(
            "Calling Responses API with reasoning.effort='{}' and text.verbosity='{}'.",
            self.settings.reasoning_effort,
            self.settings.verbosity,
        )
        try:
            response = self.client.responses.create(**request)
        except AttributeError:
            logger.warning("OpenAI SDK does not expose Responses API; falling back to Chat Completions.")
            return self._answer_with_chat_fallback(system_prompt, user_prompt)
        answer = extract_response_text(response)
        logger.success("LLM answer generated through Responses API. chars={}.", len(answer))
        return answer

    def _answer_with_chat_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback for older SDK versions."""

        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {"role": instruction_role(self.settings.chat_model), "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _require_api_key(self) -> None:
        """Raise when no usable OpenAI API key is configured."""

        if not self.settings.has_real_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing or still has a placeholder value.")


def build_context_block(contexts: list[RetrievedContext], max_chars: int) -> str:
    """Build a cited context block for the LLM prompt.

    Parameters
    ----------
    contexts
        Retrieved chunks with citation ids.
    max_chars
        Maximum characters to include.

    Returns
    -------
    str
        Prompt-ready context block.
    """

    blocks: list[str] = []
    remaining = max_chars

    for item in contexts:
        result = item.result
        metadata_text = format_metadata(result.metadata)
        header = f"{item.citation} | score={result.score:.3f}"
        if metadata_text:
            header = f"{header} | {metadata_text}"
        block = f"{header}\n{result.content.strip()}"
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if block:
            blocks.append(block)
            remaining -= len(block)
        if remaining <= 0:
            break

    return "\n\n---\n\n".join(blocks)


def build_history_block(chat_history: list[dict[str, str]], max_turns: int = 6, max_chars: int = 4000) -> str:
    """Build a compact chat-history block.

    Parameters
    ----------
    chat_history
        Streamlit-style message dictionaries with ``role`` and ``content``.
    max_turns
        Maximum number of recent user/assistant turns.
    max_chars
        Maximum characters to include.

    Returns
    -------
    str
        Prompt-ready history block.
    """

    if not chat_history:
        return "Sin historial previo."

    recent = chat_history[-max_turns * 2 :]
    lines: list[str] = []
    remaining = max_chars

    for message in recent:
        role = "Usuario" if message.get("role") == "user" else "Asistente"
        content = (message.get("content") or "").strip()
        if not content:
            continue
        line = f"{role}: {content}"
        if len(line) > remaining:
            line = line[:remaining].rstrip()
        lines.append(line)
        remaining -= len(line)
        if remaining <= 0:
            break

    return "\n".join(lines) if lines else "Sin historial previo."


def supports_temperature(model: str) -> bool:
    """Return whether a model should receive ``temperature``."""

    return not model.startswith("gpt-5")


def prefers_responses_api(model: str) -> bool:
    """Return whether the model family should use Responses API semantics."""

    return model.startswith("gpt-5") or model.startswith("o")


def format_metadata(metadata: dict[str, object]) -> str:
    """Format operational metadata for prompt context headers."""

    parts: list[str] = []
    for key in (
        "document_code",
        "document_type",
        "revision",
        "lifecycle_status",
        "effective_date",
        "approval_status",
        "is_current",
        "plant",
        "process",
        "product",
        "customer",
        "risk_level",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def extract_response_text(response: object) -> str:
    """Extract text from an OpenAI Responses API object."""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                fragments.append(str(text))
    return "\n".join(fragments).strip()


def instruction_role(model: str) -> str:
    """Return the instruction role expected by a model family."""

    if model.startswith("gpt-5") or model.startswith("o"):
        return "developer"
    return "system"


def _build_client(settings: OpenAISettings) -> OpenAI:
    """Create an OpenAI SDK client with validated base URL."""

    base_url = settings.base_url or DEFAULT_OPENAI_BASE_URL
    validate_base_url(base_url)
    kwargs: dict[str, str] = {"api_key": settings.api_key, "base_url": base_url}
    return OpenAI(**kwargs)
