"""PDF ingestion pipeline.

The ingestor reads root-level PDFs from a configured folder, extracts text,
splits it into chunks, embeds those chunks through OpenAI, and persists document
metadata plus vectors in PostgreSQL/pgvector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loguru import logger

from .config import Settings, get_settings
from .db import VectorStore, validate_identifier
from .embeddings import EmbeddingClient
from .metadata_enrichment import MetadataEnrichmentClient
from .pdf_loader import list_pdfs, load_pdf
from .quality_metadata import infer_quality_metadata
from .text_splitter import split_pages


ProgressCallback = Callable[[str], None]


@dataclass
class IngestResult:
    """Summary of a PDF ingestion run.

    Attributes
    ----------
    pdf_dir
        Folder scanned for PDFs.
    domain
        Logical domain/schema used for ingestion.
    files_found
        Number of root-level PDF files discovered.
    documents_ingested
        Number of PDFs inserted or updated.
    documents_skipped
        Number of unchanged PDFs skipped.
    chunks_created
        Number of chunks persisted.
    errors
        Per-file errors captured during ingestion.
    """

    pdf_dir: Path
    domain: str
    files_found: int = 0
    documents_ingested: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)


class PDFIngestor:
    """Coordinate PDF loading, chunking, embedding, and persistence.

    Parameters
    ----------
    settings
        Optional application settings. Defaults to environment-loaded settings.
    store
        Optional vector store dependency.
    embeddings
        Optional embedding client dependency.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        embeddings: EmbeddingClient | None = None,
    ):
        """Initialize ingestion dependencies.

        Parameters
        ----------
        settings
            Optional application settings.
        store
            Optional vector store.
        embeddings
            Optional embedding client.
        """

        self.settings = settings or get_settings()
        self.store = store or VectorStore(
            self.settings.db,
            self.settings.rag.domain,
            self.settings.openai.embedding_dim,
        )
        self.embeddings = embeddings or EmbeddingClient(self.settings.openai)

    def ingest_directory(
        self,
        pdf_dir: Path | None = None,
        domain: str | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Ingest every root-level PDF in a directory.

        Parameters
        ----------
        pdf_dir
            Directory containing PDFs. Defaults to configured RAG folder.
        domain
            Domain/schema to ingest into. Defaults to configured domain.
        force
            When True, delete existing rows for each source path before ingest.
        progress
            Optional callback for user-facing progress messages.

        Returns
        -------
        IngestResult
            Summary of the ingestion run.
        """

        target_dir = (pdf_dir or self.settings.rag.pdf_dir).resolve()
        target_domain = domain or self.settings.rag.domain
        target_schema = validate_identifier(target_domain)
        logger.info(
            "Ingestion requested. pdf_dir='{}', domain='{}', force={}.",
            target_dir,
            target_domain,
            force,
        )
        store = self.store
        if store.schema != target_schema:
            logger.info("Switching vector store schema from '{}' to '{}'.", store.schema, target_schema)
            store = VectorStore(self.settings.db, target_schema, self.settings.openai.embedding_dim)
        result = IngestResult(pdf_dir=target_dir, domain=target_domain)

        store.ensure_schema()
        pdfs = list_pdfs(target_dir, recursive=self.settings.rag.recursive_pdf_scan)
        result.files_found = len(pdfs)
        logger.info("Found {} root-level PDF files in '{}'.", len(pdfs), target_dir)

        for pdf_path in pdfs:
            superseded_id: str | None = None
            try:
                logger.info("Processing PDF '{}'.", pdf_path.name)
                _notify(progress, f"Reading {pdf_path.name}")
                document = load_pdf(pdf_path, ocr_fallback=self.settings.rag.pdf_text_fallback)
                source_path = str(pdf_path.resolve())

                if force:
                    logger.info("Force mode active; deleting old rows for '{}'.", pdf_path.name)
                    store.delete_documents_by_source(target_domain, source_path)
                else:
                    existing_id = store.find_document_id(
                        target_domain,
                        source_path,
                        document.content_hash,
                    )
                    if existing_id:
                        result.documents_skipped += 1
                        logger.info("Skipping already indexed PDF '{}'. document_id='{}'.", pdf_path.name, existing_id)
                        _notify(progress, f"Skipping {pdf_path.name}; already indexed")
                        continue
                    superseded_id = store.mark_source_superseded(target_domain, source_path)

                chunks = split_pages(
                    document.pages,
                    chunk_size=self.settings.rag.chunk_size,
                    chunk_overlap=self.settings.rag.chunk_overlap,
                )
                logger.info("Split PDF '{}' into {} chunks.", pdf_path.name, len(chunks))
                if not chunks:
                    logger.warning("PDF '{}' has no extractable text.", pdf_path.name)
                    result.errors.append(f"{pdf_path.name}: no extractable text found")
                    continue

                document_metadata = {"pages": len(document.pages)}
                if target_domain == "quality_intelligence":
                    document_metadata.update(infer_quality_metadata(pdf_path))
                if superseded_id:
                    document_metadata["supersedes_document_id"] = superseded_id
                if self.settings.rag.llm_metadata_enrichment:
                    _notify(progress, f"Extracting metadata from {pdf_path.name}")
                    enricher = MetadataEnrichmentClient(self.settings.openai)
                    document_metadata = enricher.enrich_document(
                        pdf_path=pdf_path,
                        chunks=chunks,
                        existing_metadata=document_metadata,
                        max_chars=self.settings.rag.llm_metadata_max_chars,
                    )

                _notify(progress, f"Embedding {pdf_path.name} ({len(chunks)} chunks)")
                vectors = self.embeddings.embed_texts(
                    [chunk.content for chunk in chunks],
                    batch_size=self.settings.openai.embedding_batch_size,
                )

                doc_id = store.insert_document(
                    domain=target_domain,
                    source_path=source_path,
                    file_name=pdf_path.name,
                    title=document.title,
                    author=document.author,
                    content_hash=document.content_hash,
                    metadata=document_metadata,
                )
                inserted = store.insert_chunks(doc_id, target_domain, chunks, vectors)
                result.documents_ingested += 1
                result.chunks_created += inserted
                logger.success("Indexed PDF '{}'. chunks={}.", pdf_path.name, inserted)
                _notify(progress, f"Indexed {pdf_path.name}")
            except Exception as exc:
                logger.exception("Failed to ingest PDF '{}'.", pdf_path.name)
                result.errors.append(f"{pdf_path.name}: {exc}")

        logger.info(
            "Ingestion finished. found={}, ingested={}, skipped={}, chunks={}, errors={}.",
            result.files_found,
            result.documents_ingested,
            result.documents_skipped,
            result.chunks_created,
            len(result.errors),
        )
        return result


def _notify(progress: ProgressCallback | None, message: str) -> None:
    """Send a progress message when a callback is available."""

    if progress:
        progress(message)
