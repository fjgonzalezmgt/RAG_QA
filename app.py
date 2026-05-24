"""Streamlit interface for the Quality Intelligence Assistant.

The UI exposes configuration controls, PDF ingestion actions, indexed-document
inspection, OpenAI connectivity checks, and a chat interface with short-term
conversation memory.
"""

from __future__ import annotations

import sys
from datetime import date
from dataclasses import replace
from pathlib import Path

import streamlit as st
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quality_intelligence.config import get_settings
from quality_intelligence.db import VectorStore, validate_identifier
from quality_intelligence.domain_profiles import PROFILES, get_profile
from quality_intelligence.embeddings import EmbeddingClient
from quality_intelligence.ingest import PDFIngestor
from quality_intelligence.llm import LLMClient
from quality_intelligence.logging import setup_logging
from quality_intelligence.openai_health import check_openai_connection
from quality_intelligence.retriever import RAGRetriever


QUERY_MODES: dict[str, dict[str, str]] = {
    "Procedimiento": {
        "document_type": "SOP",
        "prompt": "Enfoca la respuesta en pasos aplicables, responsables, registros requeridos y controles.",
    },
    "CAPA / causa raiz": {
        "document_type": "CAPA",
        "prompt": "Enfoca la respuesta en problema, contencion, causa raiz, acciones, eficacia y recurrencia.",
    },
    "Auditoria": {
        "document_type": "Auditoria",
        "prompt": "Enfoca la respuesta en criterio, evidencia disponible, hallazgos, brechas y preparacion.",
    },
    "Reclamo cliente": {
        "document_type": "Reclamo",
        "prompt": "Enfoca la respuesta en cliente, producto, especificacion, historial, impacto y respuesta tecnica.",
    },
    "Especificacion": {
        "document_type": "Especificacion",
        "prompt": "Enfoca la respuesta en limites, tolerancias, criterios de aceptacion y conflictos documentales.",
    },
    "Indicadores / KPI": {
        "document_type": "Indicador",
        "prompt": "Enfoca la respuesta en tendencia, meta, variacion, eventos relacionados y acciones.",
    },
}

QUICK_QUESTIONS = {
    "Resumir documentos": "Describe los documentos del sistema, agrupados por tipo documental y contexto operativo.",
    "Brechas documentales": "Identifica brechas documentales relevantes: revision, vigencia, aprobacion, metadata y evidencia faltante.",
    "Preparar auditoria": "Prepara un briefing de auditoria con evidencia disponible, documentos clave, riesgos y brechas.",
    "Comparar versiones": "Compara versiones o revisiones disponibles y senala posibles documentos obsoletos o conflictos.",
    "CAPA similares": "Busca CAPA, reclamos o hallazgos similares y resume causas raiz, acciones y evidencia de efectividad.",
}

DOCUMENT_TYPE_FILTER_VALUES = {
    "Procedimiento": "PROCEDURE",
    "Auditoria": "AUDIT",
    "Reclamo": "COMPLAINT",
    "Especificacion": "SPECIFICATION",
    "Reporte de calidad": "QUALITY_REPORT",
    "Leccion aprendida": "LESSON_LEARNED",
    "Indicador": "KPI",
}


def _active_filters(raw_filters: dict[str, object]) -> dict[str, str]:
    """Return only populated operational filters."""

    active: dict[str, str] = {}
    for key, value in raw_filters.items():
        text = str(value or "").strip()
        if text:
            active[key] = text
    return active


def _date_filter_value(value: date | None) -> str:
    """Return ISO date text for optional Streamlit date inputs."""

    return value.isoformat() if value else ""


def _document_type_filter_value(value: str) -> str:
    """Normalize display labels to metadata document type codes."""

    return DOCUMENT_TYPE_FILTER_VALUES.get(value, value)


def _source_caption(item) -> str:
    """Build a compact evidence caption for the UI."""

    metadata = item.result.metadata or {}
    parts = [f"score={item.result.score:.3f}", item.result.file_name]
    for key in ("document_type", "revision", "effective_date", "approval_status", "is_current", "plant", "process", "product", "customer", "audit"):
        value = metadata.get(key) or metadata.get(f"{key}_code")
        if value:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def _quality_score(contexts) -> tuple[str, int, list[str]]:
    """Estimate document-confidence signal from retrieved evidence metadata."""

    if not contexts:
        return "Baja", 0, ["No se recupero evidencia documental."]

    issues: list[str] = []
    total = len(contexts)
    current = 0
    approved = 0
    with_revision = 0
    with_date = 0

    for item in contexts:
        metadata = item.result.metadata or {}
        if str(metadata.get("is_current", "true")).lower() in {"true", "1", "yes"}:
            current += 1
        else:
            issues.append(f"{item.citation}: documento no vigente.")
        approval = str(metadata.get("approval_status", "")).lower()
        if approval in {"approved", "aprobado", "vigente"}:
            approved += 1
        elif not approval:
            issues.append(f"{item.citation}: falta estado de aprobacion.")
        if metadata.get("revision"):
            with_revision += 1
        else:
            issues.append(f"{item.citation}: falta revision.")
        if metadata.get("effective_date") or metadata.get("document_date"):
            with_date += 1
        else:
            issues.append(f"{item.citation}: falta fecha documental/efectiva.")

    points = 0
    points += 25 if total >= 3 else 10
    points += round(25 * current / total)
    points += round(20 * approved / total)
    points += round(15 * with_revision / total)
    points += round(15 * with_date / total)
    label = "Alta" if points >= 80 else "Media" if points >= 55 else "Baja"
    return label, points, issues[:6]


def _render_confidence(label: str, points: int, issues: list[str]) -> None:
    """Render document confidence signal."""

    if label == "Alta":
        st.success(f"Confiabilidad documental: {label} ({points}/100)")
    elif label == "Media":
        st.warning(f"Confiabilidad documental: {label} ({points}/100)")
    else:
        st.error(f"Confiabilidad documental: {label} ({points}/100)")
    if issues:
        with st.expander("Brechas detectadas en la evidencia"):
            for issue in issues:
                st.write(f"- {issue}")


def _render_source_card(item) -> None:
    """Render one retrieved source with operational metadata."""

    metadata = item.result.metadata or {}
    st.markdown(f"**{item.citation}**")
    cols = st.columns(6)
    cols[0].metric("Score", f"{item.result.score:.3f}", help="Similitud semantica entre la pregunta y este fragmento.")
    cols[1].metric("Revision", str(metadata.get("revision") or "N/D"), help="Revision documental detectada o cargada en metadata.")
    cols[2].metric("Vigente", str(metadata.get("is_current", "N/D")), help="Indica si el documento es la version actual indexada.")
    cols[3].metric("Aprobacion", str(metadata.get("approval_status") or "N/D"), help="Estado de aprobacion documental si esta disponible.")
    cols[4].metric("Tipo", str(metadata.get("document_type") or metadata.get("document_type_code") or "N/D"), help="Clasificacion documental usada para filtrar y responder.")
    page_text = f"{item.result.page_start}-{item.result.page_end}" if item.result.page_start != item.result.page_end else str(item.result.page_start or "N/D")
    cols[5].metric("Paginas", page_text, help="Rango de paginas cubierto por el fragmento citado.")
    context_bits = []
    for key in ("plant", "process", "product", "customer", "risk_level"):
        value = metadata.get(key) or metadata.get(f"{key}_code")
        if value:
            context_bits.append(f"{key}={value}")
    if context_bits:
        st.caption(" | ".join(context_bits))
    st.write(item.result.content[:1200])


def _briefing_markdown(question: str, answer: str, contexts, label: str, points: int, issues: list[str]) -> str:
    """Build an exportable Markdown briefing."""

    lines = [
        "# Quality Intelligence Briefing",
        "",
        f"**Pregunta:** {question}",
        f"**Confiabilidad documental:** {label} ({points}/100)",
        "",
        "## Respuesta",
        answer,
        "",
        "## Brechas",
    ]
    lines.extend(f"- {issue}" for issue in issues) if issues else lines.append("- No se detectaron brechas principales en metadata recuperada.")
    lines.extend(["", "## Fuentes"])
    for item in contexts:
        lines.append(f"- **{item.citation}** score={item.result.score:.3f} | {_source_caption(item)}")
    return "\n".join(lines)


setup_logging(ROOT)
logger.info("Starting Streamlit app.")
st.set_page_config(page_title="Quality Intelligence Assistant", page_icon=":clipboard:", layout="wide")
st.title("Quality Intelligence Assistant")
st.caption("Inteligencia documental para calidad, operaciones, Lean Six Sigma y QMS basada en evidencia.")

base_settings = get_settings()
logger.info(
    "Settings loaded: db_host='{}', db_name='{}', domain='{}', pdf_dir='{}', chat_model='{}', embedding_model='{}'.",
    base_settings.db.host,
    base_settings.db.name,
    base_settings.rag.domain,
    base_settings.rag.pdf_dir,
    base_settings.openai.chat_model,
    base_settings.openai.embedding_model,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_by_turn" not in st.session_state:
    st.session_state.sources_by_turn = {}
if "openai_health" not in st.session_state:
    st.session_state.openai_health = check_openai_connection(base_settings.openai)

with st.sidebar:
    st.header("Configuracion del asistente")
    pdf_dir = st.text_input(
        "Repositorio documental",
        value=str(base_settings.rag.pdf_dir),
        help="Carpeta donde se buscan PDFs para ingesta.",
    )

    profile_keys = list(PROFILES.keys()) + ["custom"]
    default_index = (
        profile_keys.index(base_settings.rag.domain)
        if base_settings.rag.domain in profile_keys
        else profile_keys.index("custom")
    )
    profile_key = st.selectbox(
        "Dominio",
        profile_keys,
        index=default_index,
        format_func=lambda key: PROFILES[key].label if key in PROFILES else "Personalizado",
        help="Perfil de dominio usado para prompt, esquema y comportamiento RAG.",
    )

    if profile_key == "custom":
        domain = st.text_input(
            "Clave del dominio",
            value=base_settings.rag.domain,
            help="Tambien se usa como esquema PostgreSQL; usa solo letras, numeros y guion bajo.",
        )
        custom_prompt = st.text_area(
            "Prompt del dominio",
            value=(
                "Eres un asistente RAG especializado. Responde en espanol, "
                "usa solo el contexto recuperado y cita fuentes como [S1], [S2]."
            ),
            height=140,
        )
    else:
        domain = profile_key
        custom_prompt = None

    top_k = st.slider(
        "Evidencias recuperadas",
        min_value=1,
        max_value=50,
        value=base_settings.rag.top_k,
        help="Cantidad final de chunks enviados al modelo como evidencia.",
    )
    candidate_k = st.slider(
        "Candidatos para diversificar",
        min_value=top_k,
        max_value=200,
        value=max(base_settings.rag.candidate_k, int(top_k)),
        help="Pool inicial de resultados antes de limitar chunks repetidos por documento.",
    )
    max_chunks_per_document = st.slider(
        "Max chunks por documento",
        min_value=1,
        max_value=10,
        value=base_settings.rag.max_chunks_per_document,
        help="Evita que un solo documento domine toda la evidencia.",
    )
    chunk_size = st.number_input(
        "Tamano chunk",
        min_value=500,
        max_value=6000,
        value=base_settings.rag.chunk_size,
        step=100,
        help="Longitud aproximada de cada fragmento indexado.",
    )
    chunk_overlap = st.number_input(
        "Solape chunk",
        min_value=0,
        max_value=int(chunk_size) - 1,
        value=min(base_settings.rag.chunk_overlap, int(chunk_size) - 1),
        step=25,
        help="Texto reutilizado entre chunks consecutivos para no perder contexto.",
    )
    recursive_pdf_scan = st.toggle(
        "Incluir subcarpetas PDF",
        value=base_settings.rag.recursive_pdf_scan,
        help="Busca PDFs dentro de subcarpetas del repositorio documental.",
    )
    pdf_text_fallback = st.toggle(
        "Fallback de texto PDF",
        value=base_settings.rag.pdf_text_fallback,
        help="Intenta pdftotext/OCR opcional cuando pypdf no extrae contenido.",
    )
    llm_metadata_enrichment = st.toggle(
        "Extraer metadata con LLM",
        value=base_settings.rag.llm_metadata_enrichment,
        help="Lee el contenido del PDF durante ingesta y propone metadata documental en JSON.",
    )
    llm_metadata_max_chars = st.number_input(
        "Texto para metadata LLM",
        min_value=2000,
        max_value=50000,
        value=base_settings.rag.llm_metadata_max_chars,
        step=1000,
        help="Caracteres maximos del documento enviados al LLM para extraer metadata.",
        disabled=not llm_metadata_enrichment,
    )

    st.divider()
    st.header("Filtros operativos")
    query_mode = st.selectbox(
        "Modo de consulta",
        list(QUERY_MODES.keys()),
        index=0,
        help="Ajusta filtros sugeridos y enfoque de la respuesta.",
    )
    mode_document_type = QUERY_MODES[query_mode]["document_type"]
    plant_filter = st.text_input("Planta / sitio", placeholder="Ej. Planta Norte", help="Limita evidencia a una planta o sitio.")
    process_filter = st.text_input("Proceso / area", placeholder="Ej. Empaque", help="Limita evidencia por proceso, area o value stream.")
    product_filter = st.text_input("Producto / SKU", placeholder="Ej. QA-100", help="Filtra por producto, material o SKU.")
    customer_filter = st.text_input("Cliente", placeholder="Ej. ACME", help="Filtra por cliente o cuenta.")
    document_type_options = [
        "",
        "SOP",
        "Procedimiento",
        "CAPA",
        "Auditoria",
        "Reclamo",
        "Especificacion",
        "Reporte de calidad",
        "Leccion aprendida",
        "DMAIC",
        "Indicador",
    ]
    document_type_filter = st.selectbox(
        "Tipo documental",
        document_type_options,
        index=document_type_options.index(mode_document_type) if mode_document_type in document_type_options else 0,
        format_func=lambda value: "Todos" if not value else value,
        help="Tipo documental objetivo; el modo de consulta sugiere un valor inicial.",
    )
    audit_filter = st.text_input("Auditoria / hallazgo", placeholder="Ej. AUD-2026-014", help="Codigo de auditoria, hallazgo o referencia relacionada.")
    date_from_filter = st.date_input("Fecha desde", value=None, format="YYYY-MM-DD", help="Fecha minima documental o efectiva.")
    date_to_filter = st.date_input("Fecha hasta", value=None, format="YYYY-MM-DD", help="Fecha maxima documental o efectiva.")
    include_obsolete = st.toggle(
        "Incluir documentos obsoletos",
        value=False,
        help="Activalo solo para comparar versiones o investigar historial.",
    )

    health = st.session_state.openai_health
    if health.ok:
        st.success(health.message)
    else:
        st.warning(health.message)

    if st.button("Probar OpenAI API", width="stretch", help="Ejecuta una llamada corta para validar clave, endpoint y modelo."):
        st.session_state.openai_health = check_openai_connection(base_settings.openai)
        st.rerun()

    if st.button("Limpiar conversacion", width="stretch", help="Borra el historial visible de esta sesion Streamlit."):
        logger.info("Clearing chat conversation.")
        st.session_state.messages = []
        st.session_state.sources_by_turn = {}
        st.rerun()

rag_settings = replace(
    base_settings.rag,
    pdf_dir=Path(pdf_dir).expanduser().resolve(),
    domain=domain,
    top_k=int(top_k),
    candidate_k=int(candidate_k),
    max_chunks_per_document=int(max_chunks_per_document),
    chunk_size=int(chunk_size),
    chunk_overlap=int(chunk_overlap),
    recursive_pdf_scan=bool(recursive_pdf_scan),
    pdf_text_fallback=bool(pdf_text_fallback),
    llm_metadata_enrichment=bool(llm_metadata_enrichment),
    llm_metadata_max_chars=int(llm_metadata_max_chars),
)
settings = replace(base_settings, rag=rag_settings)
quality_filters = _active_filters(
    {
        "plant": plant_filter,
        "process": process_filter,
        "product": product_filter,
        "customer": customer_filter,
        "document_type": _document_type_filter_value(document_type_filter),
        "audit": audit_filter,
        "date_from": _date_filter_value(date_from_filter),
        "date_to": _date_filter_value(date_to_filter),
        "include_obsolete": "true" if include_obsolete else "",
    }
)

try:
    validate_identifier(settings.rag.domain)
except ValueError:
    logger.error("Invalid domain/schema identifier: '{}'.", settings.rag.domain)
    st.error(
        "El dominio tambien se usa como esquema PostgreSQL. Usa un identificador SQL valido, "
        "por ejemplo `quality_intelligence`."
    )
    st.stop()

store = VectorStore(settings.db, settings.rag.domain, settings.openai.embedding_dim)
embedding_client = EmbeddingClient(settings.openai)
retriever = RAGRetriever(store, embedding_client)
llm_client = LLMClient(settings.openai)
profile = get_profile(settings.rag.domain, custom_prompt=custom_prompt)
logger.info("Runtime components initialized for schema/domain '{}'.", settings.rag.domain)

left, right = st.columns([0.36, 0.64], gap="large")

with left:
    st.subheader("Base de conocimiento")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Dominio", settings.rag.domain, help="Esquema/dominio activo usado para la base vectorial.")
    metric_cols[1].metric("Top evidencias", settings.rag.top_k, help="Cantidad maxima de evidencias que se enviaran al modelo.")
    metric_cols[2].metric("Filtros", len(quality_filters), help="Numero de filtros operativos activos.")

    if st.button("Inicializar BD", width="stretch", help="Crea o actualiza tablas, columnas e indices necesarios."):
        try:
            logger.info("Initializing database schema for domain '{}'.", settings.rag.domain)
            store.ensure_schema()
            st.success("Esquema y tablas listos.")
        except Exception as exc:
            logger.exception("Database initialization failed.")
            st.error(f"No se pudo inicializar la BD: {exc}")

    if st.button("Ingerir PDFs", width="stretch", help="Lee PDFs, genera chunks, embeddings y actualiza el indice vectorial."):
        log_box = st.empty()
        try:
            logger.info("Starting PDF ingestion from '{}' for domain '{}'.", settings.rag.pdf_dir, settings.rag.domain)
            ingestor = PDFIngestor(settings=settings, store=store, embeddings=embedding_client)
            result = ingestor.ingest_directory(
                pdf_dir=settings.rag.pdf_dir,
                domain=settings.rag.domain,
                progress=lambda message: log_box.info(message),
            )
            st.success(
                f"Ingesta completa: {result.documents_ingested} documentos, "
                f"{result.chunks_created} chunks, {result.documents_skipped} omitidos."
            )
            if result.errors:
                st.warning("\n".join(result.errors))
        except Exception as exc:
            logger.exception("PDF ingestion failed.")
            st.error(f"No se pudo ejecutar la ingesta: {exc}")

    st.divider()
    doc_tab, gap_tab, history_tab = st.tabs(["Documentos", "Brechas", "Historial"])
    with doc_tab:
        try:
            logger.debug("Listing indexed documents for domain '{}'.", settings.rag.domain)
            documents = store.list_documents(domain=settings.rag.domain)
            st.dataframe(
                documents,
                width="stretch",
                hide_index=True,
                column_config={
                    "source_path": None,
                    "id": None,
                    "is_current": st.column_config.CheckboxColumn("Vigente"),
                    "chunks": st.column_config.NumberColumn("Chunks"),
                },
            )
        except Exception as exc:
            logger.warning("Document listing failed: {}", exc)
            st.info(f"Aun no hay indice disponible o la BD no responde: {exc}")
    with gap_tab:
        try:
            gaps = store.list_document_gaps(domain=settings.rag.domain)
            if gaps:
                st.dataframe(gaps, width="stretch", hide_index=True)
            else:
                st.success("No hay brechas documentales detectadas en los documentos indexados.")
        except Exception as exc:
            logger.warning("Document gap listing failed: {}", exc)
            st.info(f"No se pudieron leer brechas documentales: {exc}")
    with history_tab:
        try:
            sessions = store.list_recent_sessions(limit=20)
            if sessions:
                st.dataframe(sessions, width="stretch", hide_index=True)
            else:
                st.info("Aun no hay sesiones trazables guardadas.")
        except Exception as exc:
            logger.warning("Session listing failed: {}", exc)
            st.info(f"No se pudo leer el historial trazable: {exc}")

with right:
    title_col, clear_col = st.columns([0.75, 0.25])
    with title_col:
        st.subheader("Consulta operacional")
    with clear_col:
        if st.button("Limpiar chat", width="stretch", help="Limpia solo el panel de conversacion."):
            logger.info("Clearing chat conversation from chat panel.")
            st.session_state.messages = []
            st.session_state.sources_by_turn = {}
            st.rerun()

    quick_cols = st.columns(len(QUICK_QUESTIONS))
    for index, (label, prompt_text) in enumerate(QUICK_QUESTIONS.items()):
        if quick_cols[index].button(label, width="stretch", help=f"Pregunta rapida: {prompt_text}"):
            st.session_state.pending_question = prompt_text
            st.rerun()

    for turn_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and turn_index in st.session_state.sources_by_turn:
                with st.expander("Fuentes"):
                    for source in st.session_state.sources_by_turn[turn_index]:
                        st.markdown(f"**{source['citation']}**")
                        st.caption(source["caption"])
                        st.write(source["content"])

    if quality_filters:
        st.caption("Filtros activos: " + ", ".join(f"{key}={value}" for key, value in quality_filters.items()))

    typed_question = st.chat_input("Pregunta sobre procedimientos, CAPA, auditorias, reclamos, indicadores o QMS")
    question = typed_question or st.session_state.pop("pending_question", None)

    if question:
        mode_instruction = QUERY_MODES[query_mode]["prompt"]
        retrieval_question = f"{question}\n\nModo de consulta: {query_mode}. {mode_instruction}"
        logger.info("Received chat question. chars={}, domain='{}'.", len(question), settings.rag.domain)
        history_before_question = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                logger.info(
                    "Retrieving contexts. top_k={}, candidate_k={}, max_chunks_per_document={}.",
                    settings.rag.top_k,
                    settings.rag.candidate_k,
                    settings.rag.max_chunks_per_document,
                )
                contexts = retriever.retrieve(
                    retrieval_question,
                    domain=settings.rag.domain,
                    top_k=settings.rag.top_k,
                    candidate_k=settings.rag.candidate_k,
                    max_chunks_per_document=settings.rag.max_chunks_per_document,
                    filters=quality_filters,
                )
                logger.info("Retrieved {} contexts. Calling LLM.", len(contexts))
                confidence_label, confidence_points, confidence_issues = _quality_score(contexts)
                _render_confidence(confidence_label, confidence_points, confidence_issues)
                answer = llm_client.answer(
                    retrieval_question,
                    contexts=contexts,
                    profile=profile,
                    max_context_chars=settings.rag.max_context_chars,
                    chat_history=history_before_question,
                )
                st.write(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                assistant_turn_index = len(st.session_state.messages) - 1
                try:
                    session_id = store.save_retrieval_session(
                        question=question,
                        filters={**quality_filters, "query_mode": query_mode},
                        prompt_profile=profile.key,
                        top_k=settings.rag.top_k,
                        answer=answer,
                        contexts=contexts,
                    )
                    st.caption(f"Sesion trazable #{session_id}")
                except Exception as audit_exc:
                    logger.warning("Could not save retrieval audit session: {}", audit_exc)
                    st.caption("Respuesta generada; no se pudo guardar la trazabilidad en BD.")
                st.session_state.sources_by_turn[assistant_turn_index] = [
                    {
                        "citation": item.citation,
                        "caption": _source_caption(item),
                        "content": item.result.content[:1200],
                        "metadata": item.result.metadata or {},
                        "score": item.result.score,
                    }
                    for item in contexts
                ]
                logger.success("Assistant answer completed. chars={}.", len(answer))

                briefing = _briefing_markdown(question, answer, contexts, confidence_label, confidence_points, confidence_issues)
                st.download_button(
                    "Exportar briefing",
                    data=briefing,
                    file_name="quality_intelligence_briefing.md",
                    mime="text/markdown",
                    width="stretch",
                    help="Descarga respuesta, confiabilidad, brechas y fuentes en Markdown.",
                )

                with st.expander("Fuentes y evidencia", expanded=True):
                    for item in contexts:
                        _render_source_card(item)
                        st.divider()
            except Exception as exc:
                logger.exception("Chat answer failed.")
                error_message = f"No se pudo responder: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    elif not st.session_state.messages:
        st.write("Ingiere documentos tecnicos y escribe una pregunta para consultar evidencia operacional.")
