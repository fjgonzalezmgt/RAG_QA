"""Domain-specific prompts and labels for the RAG assistant.

Profiles let the same ingestion and retrieval stack serve different document
families. For example, literature uses interpretive guidance while management
system documents prioritize requirements, evidence, and responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    """Prompt profile for one document domain.

    Attributes
    ----------
    key
        Stable domain key and default schema name.
    label
        Human-readable label for Streamlit.
    system_prompt
        Developer/system instruction used by the LLM.
    """

    key: str
    label: str
    system_prompt: str


PROFILES: dict[str, DomainProfile] = {
    "quality_intelligence": DomainProfile(
        key="quality_intelligence",
        label="Quality Intelligence",
        system_prompt=(
            "Eres un Quality Intelligence Assistant para calidad, manufactura, "
            "supply chain, Lean Six Sigma y sistemas de gestion. Responde en "
            "espanol con criterio operativo y basado en evidencia. Usa solo el "
            "contexto recuperado como fuente documental; si falta evidencia, "
            "dilo claramente y separa hechos, inferencias y recomendaciones. "
            "Prioriza requisitos aplicables, desviaciones, riesgos, causa raiz, "
            "acciones, responsables, fechas, trazabilidad, impacto en proceso, "
            "producto, cliente y auditoria. Cuando sea util, estructura la "
            "respuesta en: resumen ejecutivo, evidencia, interpretacion operativa, "
            "riesgo, decision recomendada y brechas de informacion. Cita siempre "
            "las fuentes como [S1], [S2] y no inventes datos fuera del contexto."
        ),
    ),
}


def get_profile(key: str, custom_prompt: str | None = None) -> DomainProfile:
    """Return a domain profile, optionally overriding the prompt.

    Parameters
    ----------
    key
        Domain key such as ``quality_intelligence``.
    custom_prompt
        Optional prompt supplied from the UI.

    Returns
    -------
    DomainProfile
        Matching built-in profile, custom profile, or generic fallback.
    """

    if custom_prompt:
        return DomainProfile(key=key, label=key.replace("_", " ").title(), system_prompt=custom_prompt)
    return PROFILES.get(
        key,
        DomainProfile(
            key=key,
            label=key.replace("_", " ").title(),
            system_prompt=(
                "Eres un asistente RAG especializado. Responde en espanol, usa "
                "solo el contexto recuperado y cita fuentes como [S1], [S2]."
            ),
        ),
    )
