"""Streamlit interface for the RAG Books application.

The UI exposes configuration controls, PDF ingestion actions, indexed-document
inspection, OpenAI connectivity checks, and a chat interface with short-term
conversation memory.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_books.config import get_settings
from rag_books.db import VectorStore, validate_identifier
from rag_books.domain_profiles import PROFILES, get_profile
from rag_books.embeddings import EmbeddingClient
from rag_books.ingest import PDFIngestor
from rag_books.llm import LLMClient
from rag_books.logging import setup_logging
from rag_books.openai_health import check_openai_connection
from rag_books.retriever import RAGRetriever


def _active_filters(raw_filters: dict[str, object]) -> dict[str, str]:
    """Return only populated operational filters."""

    active: dict[str, str] = {}
    for key, value in raw_filters.items():
        text = str(value or "").strip()
        if text:
            active[key] = text
    return active


def _source_caption(item) -> str:
    """Build a compact evidence caption for the UI."""

    metadata = item.result.metadata or {}
    parts = [f"score={item.result.score:.3f}", item.result.file_name]
    for key in ("document_type", "plant", "process", "product", "customer", "audit"):
        value = metadata.get(key) or metadata.get(f"{key}_code")
        if value:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


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
    pdf_dir = st.text_input("Repositorio documental", value=str(base_settings.rag.pdf_dir))

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
    )

    if profile_key == "custom":
        domain = st.text_input("Clave del dominio", value=base_settings.rag.domain)
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

    top_k = st.slider("Evidencias recuperadas", min_value=1, max_value=50, value=base_settings.rag.top_k)
    candidate_k = st.slider(
        "Candidatos para diversificar",
        min_value=top_k,
        max_value=200,
        value=max(base_settings.rag.candidate_k, int(top_k)),
    )
    max_chunks_per_document = st.slider(
        "Max chunks por documento",
        min_value=1,
        max_value=10,
        value=base_settings.rag.max_chunks_per_document,
    )
    chunk_size = st.number_input("Tamano chunk", min_value=500, max_value=6000, value=base_settings.rag.chunk_size, step=100)
    chunk_overlap = st.number_input(
        "Solape chunk",
        min_value=0,
        max_value=int(chunk_size) - 1,
        value=min(base_settings.rag.chunk_overlap, int(chunk_size) - 1),
        step=25,
    )

    st.divider()
    st.header("Filtros operativos")
    plant_filter = st.text_input("Planta / sitio", placeholder="Ej. Planta Norte")
    process_filter = st.text_input("Proceso / area", placeholder="Ej. Empaque")
    product_filter = st.text_input("Producto / SKU", placeholder="Ej. QA-100")
    customer_filter = st.text_input("Cliente", placeholder="Ej. ACME")
    document_type_filter = st.selectbox(
        "Tipo documental",
        [
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
        ],
        format_func=lambda value: "Todos" if not value else value,
    )
    audit_filter = st.text_input("Auditoria / hallazgo", placeholder="Ej. AUD-2026-014")
    date_from_filter = st.text_input("Fecha desde", placeholder="YYYY-MM-DD")
    date_to_filter = st.text_input("Fecha hasta", placeholder="YYYY-MM-DD")

    health = st.session_state.openai_health
    if health.ok:
        st.success(health.message)
    else:
        st.warning(health.message)

    if st.button("Probar OpenAI API", width="stretch"):
        st.session_state.openai_health = check_openai_connection(base_settings.openai)
        st.rerun()

    if st.button("Limpiar conversacion", width="stretch"):
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
)
settings = replace(base_settings, rag=rag_settings)
quality_filters = _active_filters(
    {
        "plant": plant_filter,
        "process": process_filter,
        "product": product_filter,
        "customer": customer_filter,
        "document_type": document_type_filter,
        "audit": audit_filter,
        "date_from": date_from_filter,
        "date_to": date_to_filter,
    }
)

try:
    validate_identifier(settings.rag.domain)
except ValueError:
    logger.error("Invalid domain/schema identifier: '{}'.", settings.rag.domain)
    st.error(
        "El dominio tambien se usa como esquema PostgreSQL. Usa un identificador SQL valido, "
        "por ejemplo `literatura` o `sistemas_gestion`."
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
    metric_cols[0].metric("Dominio", settings.rag.domain)
    metric_cols[1].metric("Top evidencias", settings.rag.top_k)
    metric_cols[2].metric("Filtros", len(quality_filters))

    if st.button("Inicializar BD", width="stretch"):
        try:
            logger.info("Initializing database schema for domain '{}'.", settings.rag.domain)
            store.ensure_schema()
            st.success("Esquema y tablas listos.")
        except Exception as exc:
            logger.exception("Database initialization failed.")
            st.error(f"No se pudo inicializar la BD: {exc}")

    if st.button("Ingerir PDFs", width="stretch"):
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
    try:
        logger.debug("Listing indexed documents for domain '{}'.", settings.rag.domain)
        documents = store.list_documents(domain=settings.rag.domain)
        st.dataframe(documents, width="stretch", hide_index=True)
    except Exception as exc:
        logger.warning("Document listing failed: {}", exc)
        st.info(f"Aun no hay indice disponible o la BD no responde: {exc}")

with right:
    title_col, clear_col = st.columns([0.75, 0.25])
    with title_col:
        st.subheader("Consulta operacional")
    with clear_col:
        if st.button("Limpiar chat", width="stretch"):
            logger.info("Clearing chat conversation from chat panel.")
            st.session_state.messages = []
            st.session_state.sources_by_turn = {}
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

    question = st.chat_input("Pregunta sobre procedimientos, CAPA, auditorias, reclamos, indicadores o QMS")

    if question:
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
                    question,
                    domain=settings.rag.domain,
                    top_k=settings.rag.top_k,
                    candidate_k=settings.rag.candidate_k,
                    max_chunks_per_document=settings.rag.max_chunks_per_document,
                    filters=quality_filters,
                )
                logger.info("Retrieved {} contexts. Calling LLM.", len(contexts))
                answer = llm_client.answer(
                    question,
                    contexts=contexts,
                    profile=profile,
                    max_context_chars=settings.rag.max_context_chars,
                    chat_history=history_before_question,
                )
                st.write(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                assistant_turn_index = len(st.session_state.messages) - 1
                st.session_state.sources_by_turn[assistant_turn_index] = [
                    {
                        "citation": item.citation,
                        "caption": _source_caption(item),
                        "content": item.result.content[:1200],
                    }
                    for item in contexts
                ]
                logger.success("Assistant answer completed. chars={}.", len(answer))

                with st.expander("Fuentes"):
                    for item in contexts:
                        st.markdown(f"**{item.citation}**")
                        st.caption(_source_caption(item))
                        st.write(item.result.content[:1200])
            except Exception as exc:
                logger.exception("Chat answer failed.")
                error_message = f"No se pudo responder: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    elif not st.session_state.messages:
        st.write("Ingiere documentos tecnicos y escribe una pregunta para consultar evidencia operacional.")
