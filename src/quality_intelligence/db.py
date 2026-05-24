"""PostgreSQL/pgvector persistence and vector search.

This module owns all database interaction for the RAG system. Each domain is
stored in its own PostgreSQL schema, while pgvector can live in a shared
extension schema such as ``extensions``. The public API is intentionally small:
initialize schema objects, insert documents/chunks, list indexed documents, and
run similarity search.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import uuid4

import psycopg
from loguru import logger
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import DatabaseSettings
from .text_splitter import TextChunk


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

QUALITY_FILTER_ALIASES: dict[str, tuple[str, ...]] = {
    "plant": ("plant", "plant_code", "site", "site_code"),
    "process": ("process", "process_code", "area", "qms_process"),
    "product": ("product", "product_code", "sku", "material"),
    "customer": ("customer", "customer_code", "account"),
    "document_type": ("document_type", "document_type_code", "doc_type", "type"),
    "audit": ("audit", "audit_id", "audit_code"),
}

DOCUMENT_TYPE_LABELS: dict[str, str] = {
    "SOP": "SOP",
    "PROCEDURE": "Procedure",
    "CAPA": "CAPA",
    "AUDIT": "Audit",
    "COMPLAINT": "Complaint",
    "SPECIFICATION": "Specification",
    "QUALITY_REPORT": "Quality report",
    "LESSON_LEARNED": "Lesson learned",
    "DMAIC": "DMAIC project",
    "KPI": "Operational indicator",
    "QMS": "QMS document",
}


@dataclass(frozen=True)
class SearchResult:
    """One retrieved text chunk and its source metadata.

    Attributes
    ----------
    chunk_id
        Unique chunk identifier.
    document_id
        Source document identifier.
    file_name
        Source file name.
    title
        Optional PDF title metadata.
    chunk_index
        Chunk order within the document.
    page_start, page_end
        Inclusive page span covered by the chunk.
    content
        Chunk text.
    score
        Cosine similarity score expressed as ``1 - distance``.
    metadata
        Additional JSON metadata.
    """

    chunk_id: str
    document_id: str
    file_name: str
    title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    content: str
    score: float
    metadata: dict[str, object]


class VectorStore:
    """PostgreSQL-backed vector store for one RAG domain schema.

    Parameters
    ----------
    db
        Database connection settings.
    schema
        PostgreSQL schema used for this domain's ``documents`` and ``chunks``.
    embedding_dim
        Expected vector dimension for stored embeddings.
    """

    def __init__(self, db: DatabaseSettings, schema: str, embedding_dim: int):
        """Initialize a vector store for one domain schema.

        Parameters
        ----------
        db
            Database settings.
        schema
            Domain schema name.
        embedding_dim
            Expected vector dimension.
        """

        self.db = db
        self.schema = validate_identifier(schema)
        self.extensions_schema = validate_identifier(db.extensions_schema)
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

    def connect(self):
        """Open a PostgreSQL connection.

        Returns
        -------
        psycopg.Connection
            Connection configured with dictionary rows.
        """

        return psycopg.connect(
            host=self.db.host,
            port=self.db.port,
            dbname=self.db.name,
            user=self.db.user,
            password=self.db.password,
            sslmode=self.db.sslmode,
            row_factory=dict_row,
        )

    def ensure_schema(self) -> None:
        """Create required extension, schemas, tables, and vector indexes.

        Notes
        -----
        The pgvector extension is installed in ``db.extensions_schema``. Domain
        tables are created in ``self.schema``. If the embedding dimension
        changed and tables are empty, they are recreated automatically.
        """

        logger.info("Ensuring pgvector extension, schema '{}', and RAG tables.", self.schema)
        self._create_schema_tables()
        self._ensure_embedding_dimension()
        self._try_create_vector_index()
        logger.success("Database schema '{}' is ready.", self.schema)

    def _create_schema_tables(self) -> None:
        """Create extension schema, domain schema, and base RAG tables."""

        schema = qident(self.schema)
        document_types = qname(self.schema, "document_types")
        plants = qname(self.schema, "plants")
        processes = qname(self.schema, "processes")
        products = qname(self.schema, "products")
        customers = qname(self.schema, "customers")
        documents = qname(self.schema, "documents")
        chunks = qname(self.schema, "chunks")
        vector_type = qname(self.extensions_schema, "vector")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {qident(self.extensions_schema)}")
                cur.execute(f"CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA {qident(self.extensions_schema)}")
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {document_types} (
                        code TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        description TEXT,
                        retention_class TEXT,
                        default_risk_level TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.executemany(
                    f"""
                    INSERT INTO {document_types} (code, label)
                    VALUES (%s, %s)
                    ON CONFLICT (code) DO UPDATE SET label = EXCLUDED.label
                    """,
                    sorted(DOCUMENT_TYPE_LABELS.items()),
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {plants} (
                        plant_code TEXT PRIMARY KEY,
                        plant_name TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {processes} (
                        process_code TEXT PRIMARY KEY,
                        process_name TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {products} (
                        product_code TEXT PRIMARY KEY,
                        product_name TEXT,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {customers} (
                        customer_code TEXT PRIMARY KEY,
                        customer_name TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {documents} (
                        id UUID PRIMARY KEY,
                        domain TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        title TEXT,
                        author TEXT,
                        content_hash TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        is_current BOOLEAN NOT NULL DEFAULT TRUE,
                        supersedes_document_id UUID,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        document_code TEXT,
                        document_type_code TEXT,
                        revision TEXT,
                        lifecycle_status TEXT,
                        document_date DATE,
                        effective_date DATE,
                        review_due_date DATE,
                        owner_area TEXT,
                        plant_code TEXT,
                        process_code TEXT,
                        product_code TEXT,
                        customer_code TEXT,
                        qms_process TEXT,
                        source_system TEXT,
                        source_record_id TEXT,
                        confidentiality_level TEXT,
                        risk_level TEXT,
                        approval_status TEXT,
                        approved_by TEXT,
                        approved_at TIMESTAMPTZ,
                        UNIQUE (domain, source_path, content_hash)
                    )
                    """
                )
                for column_sql in document_compatibility_columns():
                    cur.execute(f"ALTER TABLE {documents} ADD COLUMN IF NOT EXISTS {column_sql}")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {chunks} (
                        id UUID PRIMARY KEY,
                        document_id UUID NOT NULL REFERENCES {documents}(id) ON DELETE CASCADE,
                        domain TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page_start INTEGER,
                        page_end INTEGER,
                        content TEXT NOT NULL,
                        token_count INTEGER,
                        embedding {vector_type}({self.embedding_dim}),
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        section_title TEXT,
                        section_number TEXT,
                        clause_ref TEXT,
                        requirement_type TEXT,
                        process_step TEXT,
                        risk_signal TEXT,
                        key_terms TEXT[],
                        detected_entities JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        UNIQUE (document_id, chunk_index)
                    )
                    """
                )
                for column_sql in chunk_compatibility_columns():
                    cur.execute(f"ALTER TABLE {chunks} ADD COLUMN IF NOT EXISTS {column_sql}")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_domain_idx ON {documents} (domain)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_current_idx ON {documents} (domain, is_current)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_document_type_idx ON {documents} (document_type_code)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_plant_process_idx ON {documents} (plant_code, process_code)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_product_customer_idx ON {documents} (product_code, customer_code)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_effective_date_idx ON {documents} (effective_date)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx ON {documents} USING gin (metadata)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS chunks_domain_idx ON {chunks} (domain)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON {chunks} USING gin (metadata)")
                self._create_audit_tables(cur)
            conn.commit()

    def _create_audit_tables(self, cur) -> None:
        """Create retrieval audit tables used by the Streamlit app."""

        retrieval_sessions = qname(self.schema, "retrieval_sessions")
        retrieval_evidence = qname(self.schema, "retrieval_evidence")
        decision_records = qname(self.schema, "decision_records")
        chunks = qname(self.schema, "chunks")
        documents = qname(self.schema, "documents")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {retrieval_sessions} (
                id BIGSERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                filters JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                prompt_profile TEXT,
                top_k INTEGER,
                answer TEXT,
                decision_intent TEXT,
                created_by TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {retrieval_evidence} (
                id BIGSERIAL PRIMARY KEY,
                session_id BIGINT NOT NULL REFERENCES {retrieval_sessions}(id) ON DELETE CASCADE,
                source_label TEXT NOT NULL,
                chunk_id UUID REFERENCES {chunks}(id) ON DELETE SET NULL,
                document_id UUID REFERENCES {documents}(id) ON DELETE SET NULL,
                score NUMERIC,
                page_start INTEGER,
                page_end INTEGER,
                evidence_role TEXT,
                quote_excerpt TEXT,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {decision_records} (
                id BIGSERIAL PRIMARY KEY,
                decision_code TEXT UNIQUE,
                title TEXT NOT NULL,
                decision_type TEXT,
                decision_summary TEXT NOT NULL,
                rationale TEXT,
                risk_assessment TEXT,
                owner_name TEXT,
                status TEXT,
                due_at DATE,
                closed_at DATE,
                retrieval_session_id BIGINT REFERENCES {retrieval_sessions}(id),
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS retrieval_sessions_created_at_idx ON {retrieval_sessions} (created_at DESC)")

    def find_document_id(self, domain: str, source_path: str, content_hash: str) -> str | None:
        """Find an already-indexed document.

        Parameters
        ----------
        domain
            Logical domain.
        source_path
            Absolute path of the PDF file.
        content_hash
            SHA-256 hash of the PDF contents.

        Returns
        -------
        str or None
            Existing document id when the same file content is already indexed.
        """

        logger.debug("Checking indexed document. domain='{}', source_path='{}'.", domain, source_path)
        documents = qname(self.schema, "documents")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id::text AS id
                    FROM {documents}
                    WHERE domain = %s AND source_path = %s AND content_hash = %s
                    """,
                    (domain, source_path, content_hash),
                )
                row = cur.fetchone()
        document_id = row["id"] if row else None
        logger.debug("Indexed document lookup result: {}.", "hit" if document_id else "miss")
        return document_id

    def delete_documents_by_source(self, domain: str, source_path: str) -> int:
        """Delete indexed rows for a source PDF.

        Parameters
        ----------
        domain
            Logical domain.
        source_path
            Absolute path of the PDF file.

        Returns
        -------
        int
            Number of deleted document rows.
        """

        logger.info("Deleting indexed document by source. domain='{}', source_path='{}'.", domain, source_path)
        documents = qname(self.schema, "documents")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {documents} WHERE domain = %s AND source_path = %s",
                    (domain, source_path),
                )
                deleted = cur.rowcount
            conn.commit()
        logger.info("Deleted {} document rows for source '{}'.", deleted, source_path)
        return deleted

    def mark_source_superseded(self, domain: str, source_path: str) -> str | None:
        """Mark current rows for one source path as no longer current."""

        logger.info("Marking previous document versions as superseded. domain='{}', source_path='{}'.", domain, source_path)
        documents = qname(self.schema, "documents")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id::text AS id
                    FROM {documents}
                    WHERE domain = %s AND source_path = %s AND COALESCE(is_current, TRUE) IS TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (domain, source_path),
                )
                row = cur.fetchone()
                cur.execute(
                    f"""
                    UPDATE {documents}
                    SET is_current = FALSE
                    WHERE domain = %s AND source_path = %s AND COALESCE(is_current, TRUE) IS TRUE
                    """,
                    (domain, source_path),
                )
            conn.commit()
        superseded_id = row["id"] if row else None
        logger.info("Superseded current document id: {}.", superseded_id or "none")
        return superseded_id

    def insert_document(
        self,
        domain: str,
        source_path: str,
        file_name: str,
        title: str | None,
        author: str | None,
        content_hash: str,
        metadata: dict[str, object],
    ) -> str:
        """Insert or update a document metadata row.

        Parameters
        ----------
        domain
            Logical domain.
        source_path
            Absolute file path.
        file_name
            Source PDF file name.
        title, author
            Optional PDF metadata.
        content_hash
            SHA-256 hash of the source file.
        metadata
            Additional metadata stored as JSONB.

        Returns
        -------
        str
            Document identifier.
        """

        logger.info("Upserting document metadata. file='{}', domain='{}'.", file_name, domain)
        documents = qname(self.schema, "documents")
        document_id = str(uuid4())
        typed = metadata_column_values(metadata)
        self._ensure_dimension_values(metadata)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {documents}
                        (
                            id, domain, source_path, file_name, title, author, content_hash, metadata,
                            is_current, supersedes_document_id, document_code, document_type_code,
                            revision, lifecycle_status, document_date, effective_date, review_due_date,
                            owner_area, plant_code, process_code, product_code, customer_code,
                            qms_process, source_system, source_record_id, confidentiality_level,
                            risk_level, approval_status, approved_by, approved_at
                        )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        TRUE, %s, %s, %s,
                        %s, %s, %s::date, %s::date, %s::date,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s::timestamptz
                    )
                    ON CONFLICT (domain, source_path, content_hash)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        metadata = EXCLUDED.metadata,
                        is_current = TRUE,
                        document_code = EXCLUDED.document_code,
                        document_type_code = EXCLUDED.document_type_code,
                        revision = EXCLUDED.revision,
                        lifecycle_status = EXCLUDED.lifecycle_status,
                        document_date = EXCLUDED.document_date,
                        effective_date = EXCLUDED.effective_date,
                        review_due_date = EXCLUDED.review_due_date,
                        owner_area = EXCLUDED.owner_area,
                        plant_code = EXCLUDED.plant_code,
                        process_code = EXCLUDED.process_code,
                        product_code = EXCLUDED.product_code,
                        customer_code = EXCLUDED.customer_code,
                        qms_process = EXCLUDED.qms_process,
                        source_system = EXCLUDED.source_system,
                        source_record_id = EXCLUDED.source_record_id,
                        confidentiality_level = EXCLUDED.confidentiality_level,
                        risk_level = EXCLUDED.risk_level,
                        approval_status = EXCLUDED.approval_status,
                        approved_by = EXCLUDED.approved_by,
                        approved_at = EXCLUDED.approved_at
                    RETURNING id::text AS id
                    """,
                    (
                        document_id,
                        domain,
                        source_path,
                        file_name,
                        title,
                        author,
                        content_hash,
                        Jsonb(metadata),
                        typed["supersedes_document_id"],
                        typed["document_code"],
                        typed["document_type_code"],
                        typed["revision"],
                        typed["lifecycle_status"],
                        typed["document_date"],
                        typed["effective_date"],
                        typed["review_due_date"],
                        typed["owner_area"],
                        typed["plant_code"],
                        typed["process_code"],
                        typed["product_code"],
                        typed["customer_code"],
                        typed["qms_process"],
                        typed["source_system"],
                        typed["source_record_id"],
                        typed["confidentiality_level"],
                        typed["risk_level"],
                        typed["approval_status"],
                        typed["approved_by"],
                        typed["approved_at"],
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        logger.debug("Document id ready: {}.", row["id"])
        return row["id"]

    def _ensure_dimension_values(self, metadata: Mapping[str, object]) -> None:
        """Upsert lightweight dimension rows before filling FK-backed columns."""

        dimensions = [
            ("plants", "plant_code", "plant_name", first_metadata_value(metadata, "plant_code", "plant")),
            ("processes", "process_code", "process_name", first_metadata_value(metadata, "process_code", "process")),
            ("products", "product_code", "product_name", first_metadata_value(metadata, "product_code", "product", "sku")),
            ("customers", "customer_code", "customer_name", first_metadata_value(metadata, "customer_code", "customer")),
        ]
        with self.connect() as conn:
            with conn.cursor() as cur:
                for table_name, code_column, name_column, value in dimensions:
                    if not value:
                        continue
                    table = qname(self.schema, table_name)
                    cur.execute(
                        f"""
                        INSERT INTO {table} ({qident(code_column)}, {qident(name_column)})
                        VALUES (%s, %s)
                        ON CONFLICT ({qident(code_column)}) DO UPDATE SET
                            {qident(name_column)} = EXCLUDED.{qident(name_column)}
                        """,
                        (value, value),
                    )
            conn.commit()

    def insert_chunks(
        self,
        document_id: str,
        domain: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Insert or update text chunks and their embeddings.

        Parameters
        ----------
        document_id
            Parent document id.
        domain
            Logical domain.
        chunks
            Text chunks to persist.
        embeddings
            Embedding vectors aligned one-to-one with ``chunks``.

        Returns
        -------
        int
            Number of chunks processed.
        """

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        logger.info("Upserting {} chunks for document '{}'.", len(chunks), document_id)
        table = qname(self.schema, "chunks")
        vector_type = qname(self.extensions_schema, "vector")
        inserted = 0
        with self.connect() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    chunk_typed = chunk_metadata_column_values(chunk.metadata)
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                            (
                                id, document_id, domain, chunk_index, page_start, page_end,
                                content, token_count, embedding, metadata, section_title,
                                section_number, clause_ref, requirement_type, process_step,
                                risk_signal, key_terms, detected_entities
                            )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::{vector_type}, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            token_count = EXCLUDED.token_count,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            section_title = EXCLUDED.section_title,
                            section_number = EXCLUDED.section_number,
                            clause_ref = EXCLUDED.clause_ref,
                            requirement_type = EXCLUDED.requirement_type,
                            process_step = EXCLUDED.process_step,
                            risk_signal = EXCLUDED.risk_signal,
                            key_terms = EXCLUDED.key_terms,
                            detected_entities = EXCLUDED.detected_entities
                        """,
                        (
                            str(uuid4()),
                            document_id,
                            domain,
                            chunk.index,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.content,
                            chunk.token_count,
                            vector_literal(embedding),
                            Jsonb(chunk.metadata),
                            chunk_typed["section_title"],
                            chunk_typed["section_number"],
                            chunk_typed["clause_ref"],
                            chunk_typed["requirement_type"],
                            chunk_typed["process_step"],
                            chunk_typed["risk_signal"],
                            chunk_typed["key_terms"],
                            Jsonb(chunk_typed["detected_entities"]),
                        ),
                    )
                    inserted += 1
            conn.commit()
        logger.success("Upserted {} chunks for document '{}'.", inserted, document_id)
        return inserted

    def search(
        self,
        domain: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        """Run a nearest-neighbor vector search.

        Parameters
        ----------
        domain
            Logical domain.
        query_embedding
            Query embedding vector.
        top_k
            Number of nearest chunks to return.
        filters
            Optional operational filters. Values are matched against document
            and chunk metadata keys such as plant, process, product, customer,
            document_type, audit, date_from, and date_to.

        Returns
        -------
        list[SearchResult]
            Search results ordered by vector distance.
        """

        logger.info(
            "Running vector search. schema='{}', domain='{}', top_k={}, filters={}.",
            self.schema,
            domain,
            top_k,
            sorted((filters or {}).keys()),
        )
        chunks = qname(self.schema, "chunks")
        documents = qname(self.schema, "documents")
        vector_type = qname(self.extensions_schema, "vector")
        query_vector = vector_literal(query_embedding)
        where_sql, where_params = build_filter_clause(domain, filters)
        document_metadata = document_metadata_json("d")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('search_path', %s, true)",
                    (index_search_path(self.schema, self.extensions_schema),),
                )
                cur.execute(
                    f"""
                    SELECT
                        c.id::text AS chunk_id,
                        d.id::text AS document_id,
                        d.file_name,
                        d.title,
                        c.chunk_index,
                        c.page_start,
                        c.page_end,
                        c.content,
                        {document_metadata} || COALESCE(d.metadata, '{{}}'::jsonb) || COALESCE(c.metadata, '{{}}'::jsonb) AS metadata,
                        1 - (c.embedding <=> %s::{vector_type}) AS score
                    FROM {chunks} c
                    JOIN {documents} d ON d.id = c.document_id
                    WHERE {where_sql}
                    ORDER BY c.embedding <=> %s::{vector_type}
                    LIMIT %s
                    """,
                    (query_vector, *where_params, query_vector, top_k),
                )
                rows = cur.fetchall()

        results = [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                file_name=row["file_name"],
                title=row["title"],
                chunk_index=row["chunk_index"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                content=row["content"],
                score=float(row["score"]),
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]
        logger.info("Vector search returned {} results.", len(results))
        return results

    def search_diverse(
        self,
        domain: str,
        query_embedding: Sequence[float],
        top_k: int,
        candidate_k: int,
        max_chunks_per_document: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        """Run vector search and diversify selected chunks by document.

        Parameters
        ----------
        domain
            Logical domain.
        query_embedding
            Query embedding vector.
        top_k
            Final number of chunks to return.
        candidate_k
            Number of nearest-neighbor candidates considered before filtering.
        max_chunks_per_document
            Maximum selected chunks per source document before fallback fill.
        filters
            Optional operational filters passed to vector search.

        Returns
        -------
        list[SearchResult]
            Diversified search results.
        """

        candidate_k = max(candidate_k, top_k)
        max_chunks_per_document = max(1, max_chunks_per_document)
        logger.info(
            "Running diverse vector search. schema='{}', domain='{}', top_k={}, candidate_k={}, max_chunks_per_document={}.",
            self.schema,
            domain,
            top_k,
            candidate_k,
            max_chunks_per_document,
        )
        candidates = self.search(
            domain=domain,
            query_embedding=query_embedding,
            top_k=candidate_k,
            filters=filters,
        )
        selected: list[SearchResult] = []
        counts_by_document: dict[str, int] = {}

        for result in candidates:
            count = counts_by_document.get(result.document_id, 0)
            if count >= max_chunks_per_document:
                continue
            selected.append(result)
            counts_by_document[result.document_id] = count + 1
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            selected_ids = {result.chunk_id for result in selected}
            for result in candidates:
                if result.chunk_id in selected_ids:
                    continue
                selected.append(result)
                if len(selected) >= top_k:
                    break

        logger.info(
            "Diverse vector search selected {} results from {} documents.",
            len(selected),
            len({result.document_id for result in selected}),
        )
        return selected

    def list_documents(self, domain: str | None = None) -> list[dict[str, object]]:
        """List indexed documents and chunk counts.

        Parameters
        ----------
        domain
            Optional domain filter.

        Returns
        -------
        list[dict[str, object]]
            Document rows suitable for Streamlit display.
        """

        logger.debug("Listing documents. schema='{}', domain='{}'.", self.schema, domain)
        documents = qname(self.schema, "documents")
        chunks = qname(self.schema, "chunks")
        if not self._table_exists("documents"):
            logger.info("Document table does not exist yet for schema '{}'.", self.schema)
            return []
        where = "WHERE d.domain = %s" if domain else ""
        params: tuple[object, ...] = (domain,) if domain else ()

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        d.id::text AS id,
                        d.domain,
                        d.file_name,
                        d.title,
                        d.author,
                        d.document_code,
                        d.document_type_code,
                        d.revision,
                        d.lifecycle_status,
                        d.effective_date,
                        d.approval_status,
                        d.is_current,
                        d.plant_code,
                        d.process_code,
                        d.product_code,
                        d.customer_code,
                        d.source_path,
                        d.created_at,
                        COUNT(c.id)::int AS chunks
                    FROM {documents} d
                    LEFT JOIN {chunks} c ON c.document_id = d.id
                    {where}
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                    """,
                    params,
                )
                rows = list(cur.fetchall())
        logger.debug("Listed {} documents.", len(rows))
        return rows

    def save_retrieval_session(
        self,
        question: str,
        filters: Mapping[str, object],
        prompt_profile: str,
        top_k: int,
        answer: str,
        contexts: Sequence[object],
    ) -> int:
        """Persist a RAG answer and the evidence chunks used to generate it."""

        retrieval_sessions = qname(self.schema, "retrieval_sessions")
        retrieval_evidence = qname(self.schema, "retrieval_evidence")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {retrieval_sessions}
                        (question, filters, prompt_profile, top_k, answer)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (question, Jsonb(dict(filters)), prompt_profile, top_k, answer),
                )
                session_id = int(cur.fetchone()["id"])
                for item in contexts:
                    result = item.result
                    cur.execute(
                        f"""
                        INSERT INTO {retrieval_evidence}
                            (
                                session_id, source_label, chunk_id, document_id, score,
                                page_start, page_end, evidence_role, quote_excerpt, metadata
                            )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            session_id,
                            item.source_id,
                            result.chunk_id,
                            result.document_id,
                            result.score,
                            result.page_start,
                            result.page_end,
                            "retrieved_context",
                            result.content[:1200],
                            Jsonb(result.metadata or {}),
                        ),
                    )
            conn.commit()
        logger.info("Saved retrieval session {} with {} evidence rows.", session_id, len(contexts))
        return session_id

    def list_recent_sessions(self, limit: int = 20) -> list[dict[str, object]]:
        """List recent retrieval sessions for UI traceability."""

        table = qname(self.schema, "retrieval_sessions")
        if not self._table_exists("retrieval_sessions"):
            return []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        created_at,
                        question,
                        filters,
                        prompt_profile,
                        top_k,
                        left(COALESCE(answer, ''), 900) AS answer_preview
                    FROM {table}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = list(cur.fetchall())
        return rows

    def list_document_gaps(self, domain: str, limit: int = 100) -> list[dict[str, object]]:
        """List indexed documents with missing operational/QMS metadata."""

        documents = qname(self.schema, "documents")
        chunks = qname(self.schema, "chunks")
        if not self._table_exists("documents"):
            return []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        d.file_name,
                        d.document_type_code,
                        d.revision,
                        d.effective_date,
                        d.approval_status,
                        d.is_current,
                        d.plant_code,
                        d.process_code,
                        d.product_code,
                        d.customer_code,
                        COUNT(c.id)::int AS chunks,
                        ARRAY_REMOVE(ARRAY[
                            CASE WHEN d.document_type_code IS NULL THEN 'tipo documental' END,
                            CASE WHEN d.revision IS NULL THEN 'revision' END,
                            CASE WHEN d.effective_date IS NULL THEN 'fecha efectiva' END,
                            CASE WHEN d.approval_status IS NULL THEN 'aprobacion' END,
                            CASE WHEN d.plant_code IS NULL THEN 'planta' END,
                            CASE WHEN d.process_code IS NULL THEN 'proceso' END
                        ], NULL) AS missing_metadata
                    FROM {documents} d
                    LEFT JOIN {chunks} c ON c.document_id = d.id
                    WHERE d.domain = %s
                    GROUP BY d.id
                    HAVING cardinality(ARRAY_REMOVE(ARRAY[
                        CASE WHEN d.document_type_code IS NULL THEN 'tipo documental' END,
                        CASE WHEN d.revision IS NULL THEN 'revision' END,
                        CASE WHEN d.effective_date IS NULL THEN 'fecha efectiva' END,
                        CASE WHEN d.approval_status IS NULL THEN 'aprobacion' END,
                        CASE WHEN d.plant_code IS NULL THEN 'planta' END,
                        CASE WHEN d.process_code IS NULL THEN 'proceso' END
                    ], NULL)) > 0
                    ORDER BY cardinality(ARRAY_REMOVE(ARRAY[
                        CASE WHEN d.document_type_code IS NULL THEN 'tipo documental' END,
                        CASE WHEN d.revision IS NULL THEN 'revision' END,
                        CASE WHEN d.effective_date IS NULL THEN 'fecha efectiva' END,
                        CASE WHEN d.approval_status IS NULL THEN 'aprobacion' END,
                        CASE WHEN d.plant_code IS NULL THEN 'planta' END,
                        CASE WHEN d.process_code IS NULL THEN 'proceso' END
                    ], NULL)) DESC, d.file_name
                    LIMIT %s
                    """,
                    (domain, limit),
                )
                rows = list(cur.fetchall())
        return rows

    def _try_create_vector_index(self) -> None:
        """Create an approximate vector index when pgvector supports it."""

        if self.embedding_dim > 2000:
            logger.warning(
                "Skipping approximate vector indexes because pgvector HNSW/IVFFLAT support up to 2000 dimensions; current dimension is {}.",
                self.embedding_dim,
            )
            logger.warning("Use OPENAI_EMBEDDING_DIM=2000 with text-embedding-3-large to keep approximate indexes.")
            return

        logger.info("Ensuring HNSW vector index for schema '{}'.", self.schema)
        chunks = qname(self.schema, "chunks")
        index_name = qident("chunks_embedding_hnsw_idx")
        try:
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT set_config('search_path', %s, true)",
                        (index_search_path(self.schema, self.extensions_schema),),
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {chunks}
                        USING hnsw (embedding vector_cosine_ops)
                        """
                    )
                conn.commit()
            logger.success("HNSW vector index is ready for schema '{}'.", self.schema)
        except Exception:
            # Older pgvector versions may not support HNSW. Similarity search still works.
            logger.warning("HNSW vector index creation skipped/failed: {}", format_db_error())
            self._try_create_ivfflat_index()
            return

    def _try_create_ivfflat_index(self) -> None:
        """Create an IVFFLAT fallback index when HNSW is unavailable."""

        logger.info("Trying IVFFLAT vector index fallback for schema '{}'.", self.schema)
        chunks = qname(self.schema, "chunks")
        index_name = qident("chunks_embedding_ivfflat_idx")
        try:
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT set_config('search_path', %s, true)",
                        (index_search_path(self.schema, self.extensions_schema),),
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {chunks}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                conn.commit()
            logger.success("IVFFLAT vector index fallback is ready for schema '{}'.", self.schema)
        except Exception:
            logger.warning("IVFFLAT vector index fallback also failed: {}", format_db_error())
            logger.warning("Continuing with exact vector search; this is correct but slower on large indexes.")
            return

    def _table_exists(self, table_name: str) -> bool:
        """Return whether a table exists in the domain schema."""

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = %s
                    ) AS exists
                    """,
                    (self.schema, table_name),
                )
                row = cur.fetchone()
        return bool(row["exists"])

    def _ensure_embedding_dimension(self) -> None:
        """Validate the stored embedding column dimension against settings."""

        chunks = qname(self.schema, "chunks")
        documents = qname(self.schema, "documents")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        format_type(a.atttypid, a.atttypmod) AS vector_type,
                        a.atttypmod AS type_modifier
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s
                      AND c.relname = 'chunks'
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                    """,
                    (self.schema,),
                )
                row = cur.fetchone()
                if not row:
                    return

                current_dim = vector_dimension_from_metadata(row["vector_type"], row["type_modifier"])
                if current_dim == self.embedding_dim:
                    logger.debug("Embedding dimension matches table definition: {}.", self.embedding_dim)
                    return

                if current_dim is None:
                    logger.warning(
                        "Could not infer embedding dimension from table metadata. vector_type='{}', type_modifier={}. "
                        "Skipping automatic table recreation.",
                        row["vector_type"],
                        row["type_modifier"],
                    )
                    return

                cur.execute(f"SELECT COUNT(*)::int AS count FROM {chunks}")
                chunk_count = cur.fetchone()["count"]

                if chunk_count == 0:
                    logger.warning(
                        "Embedding dimension changed from {} to {} and table is empty; recreating RAG tables.",
                        current_dim,
                        self.embedding_dim,
                    )
                    cur.execute(f"DROP TABLE IF EXISTS {chunks} CASCADE")
                    cur.execute(f"DROP TABLE IF EXISTS {documents} CASCADE")
                    conn.commit()
                    self._create_schema_tables()
                    return

        raise RuntimeError(
            f"El esquema '{self.schema}' tiene embeddings vector({current_dim}), "
            f"pero la configuracion actual espera vector({self.embedding_dim}). "
            "Crea un dominio/esquema nuevo o elimina y reingesta ese esquema para usar el modelo nuevo."
        )


def validate_identifier(value: str) -> str:
    """Validate a PostgreSQL identifier.

    Parameters
    ----------
    value
        Identifier to validate.

    Returns
    -------
    str
        The original identifier when valid.

    Raises
    ------
    ValueError
        If the identifier contains unsupported characters.
    """

    if not IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return value


def qident(identifier: str) -> str:
    """Quote a validated PostgreSQL identifier."""

    validate_identifier(identifier)
    return f'"{identifier}"'


def qname(schema: str, name: str) -> str:
    """Return a quoted schema-qualified object name."""

    return f"{qident(schema)}.{qident(name)}"


def index_search_path(schema: str, extensions_schema: str) -> str:
    """Build a safe search path for vector operators.

    Parameters
    ----------
    schema
        Domain schema.
    extensions_schema
        Schema containing pgvector extension objects.

    Returns
    -------
    str
        Comma-separated search path string.
    """

    validate_identifier(schema)
    validate_identifier(extensions_schema)
    parts = [qident(schema)]
    if extensions_schema != schema:
        parts.append(qident(extensions_schema))
    if extensions_schema != "public" and schema != "public":
        parts.append("public")
    return ", ".join(parts)


def parse_vector_dimension(vector_type: str | None) -> int | None:
    """Parse ``vector(N)`` dimension text returned by PostgreSQL."""

    match = re.match(r"vector\((\d+)\)", vector_type or "")
    if not match:
        return None
    return int(match.group(1))


def vector_dimension_from_metadata(vector_type: str | None, type_modifier: int | None) -> int | None:
    """Infer a pgvector column dimension from PostgreSQL metadata."""

    parsed = parse_vector_dimension(vector_type)
    if parsed is not None:
        return parsed
    if type_modifier is not None and type_modifier >= 0:
        return int(type_modifier)
    return None


def format_db_error() -> str:
    """Format the active PostgreSQL exception for useful logs."""

    import sys

    exc = sys.exc_info()[1]
    if exc is None:
        return "unknown error"

    parts = [f"{type(exc).__name__}: {exc}"]
    diag = getattr(exc, "diag", None)
    if diag is not None:
        for attr in ("severity", "sqlstate", "message_primary", "message_detail", "message_hint"):
            value = getattr(diag, attr, None)
            if value:
                parts.append(f"{attr}={value}")
    return " | ".join(parts)


def vector_literal(values: Sequence[float]) -> str:
    """Convert a Python vector into pgvector text literal syntax.

    Parameters
    ----------
    values
        Numeric vector values.

    Returns
    -------
    str
        Literal such as ``[0.1,-0.2]``.
    """

    if not values:
        raise ValueError("Vector cannot be empty")

    cleaned: list[str] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Vector contains non-finite values")
        cleaned.append(repr(number))
    return "[" + ",".join(cleaned) + "]"


def document_compatibility_columns() -> list[str]:
    """Columns expected by current app versions on existing document tables."""

    return [
        "is_current BOOLEAN NOT NULL DEFAULT TRUE",
        "supersedes_document_id UUID",
        "document_code TEXT",
        "document_type_code TEXT",
        "revision TEXT",
        "lifecycle_status TEXT",
        "document_date DATE",
        "effective_date DATE",
        "review_due_date DATE",
        "owner_area TEXT",
        "plant_code TEXT",
        "process_code TEXT",
        "product_code TEXT",
        "customer_code TEXT",
        "qms_process TEXT",
        "source_system TEXT",
        "source_record_id TEXT",
        "confidentiality_level TEXT",
        "risk_level TEXT",
        "approval_status TEXT",
        "approved_by TEXT",
        "approved_at TIMESTAMPTZ",
    ]


def chunk_compatibility_columns() -> list[str]:
    """Columns expected by current app versions on existing chunk tables."""

    return [
        "section_title TEXT",
        "section_number TEXT",
        "clause_ref TEXT",
        "requirement_type TEXT",
        "process_step TEXT",
        "risk_signal TEXT",
        "key_terms TEXT[]",
        "detected_entities JSONB NOT NULL DEFAULT '{}'::jsonb",
    ]


def metadata_column_values(metadata: Mapping[str, object]) -> dict[str, str | None]:
    """Map flexible metadata JSON to typed document columns."""

    document_type = first_metadata_value(metadata, "document_type_code", "document_type", "doc_type")
    if document_type:
        document_type = document_type.upper().replace(" ", "_")
    return {
        "supersedes_document_id": first_metadata_value(metadata, "supersedes_document_id"),
        "document_code": first_metadata_value(metadata, "document_code", "code"),
        "document_type_code": document_type,
        "revision": first_metadata_value(metadata, "revision", "rev", "version"),
        "lifecycle_status": first_metadata_value(metadata, "lifecycle_status", "status"),
        "document_date": iso_date_or_none(first_metadata_value(metadata, "document_date")),
        "effective_date": iso_date_or_none(first_metadata_value(metadata, "effective_date")),
        "review_due_date": iso_date_or_none(first_metadata_value(metadata, "review_due_date")),
        "owner_area": first_metadata_value(metadata, "owner_area", "owner"),
        "plant_code": first_metadata_value(metadata, "plant_code", "plant"),
        "process_code": first_metadata_value(metadata, "process_code", "process"),
        "product_code": first_metadata_value(metadata, "product_code", "product", "sku"),
        "customer_code": first_metadata_value(metadata, "customer_code", "customer"),
        "qms_process": first_metadata_value(metadata, "qms_process"),
        "source_system": first_metadata_value(metadata, "source_system"),
        "source_record_id": first_metadata_value(metadata, "source_record_id"),
        "confidentiality_level": first_metadata_value(metadata, "confidentiality_level"),
        "risk_level": first_metadata_value(metadata, "risk_level"),
        "approval_status": first_metadata_value(metadata, "approval_status"),
        "approved_by": first_metadata_value(metadata, "approved_by"),
        "approved_at": first_metadata_value(metadata, "approved_at"),
    }


def chunk_metadata_column_values(metadata: Mapping[str, object]) -> dict[str, object]:
    """Map chunk metadata JSON to typed chunk columns."""

    key_terms = metadata.get("key_terms")
    if not isinstance(key_terms, list):
        key_terms = None
    detected_entities = metadata.get("detected_entities")
    if not isinstance(detected_entities, dict):
        detected_entities = {}
    return {
        "section_title": first_metadata_value(metadata, "section_title"),
        "section_number": first_metadata_value(metadata, "section_number"),
        "clause_ref": first_metadata_value(metadata, "clause_ref"),
        "requirement_type": first_metadata_value(metadata, "requirement_type"),
        "process_step": first_metadata_value(metadata, "process_step"),
        "risk_signal": first_metadata_value(metadata, "risk_signal"),
        "key_terms": key_terms,
        "detected_entities": detected_entities,
    }


def first_metadata_value(metadata: Mapping[str, object], *keys: str) -> str | None:
    """Return the first non-empty metadata value for a set of aliases."""

    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def iso_date_or_none(value: str | None) -> str | None:
    """Return an ISO date string only when PostgreSQL can safely cast it."""

    if value and re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    return None


def document_metadata_json(alias: str = "d") -> str:
    """SQL expression that exposes typed document columns as JSON metadata."""

    return f"""
        jsonb_strip_nulls(jsonb_build_object(
            'document_code', {alias}.document_code,
            'document_type', {alias}.document_type_code,
            'document_type_code', {alias}.document_type_code,
            'revision', {alias}.revision,
            'lifecycle_status', {alias}.lifecycle_status,
            'document_date', {alias}.document_date,
            'effective_date', {alias}.effective_date,
            'review_due_date', {alias}.review_due_date,
            'owner_area', {alias}.owner_area,
            'plant', {alias}.plant_code,
            'plant_code', {alias}.plant_code,
            'process', {alias}.process_code,
            'process_code', {alias}.process_code,
            'product', {alias}.product_code,
            'product_code', {alias}.product_code,
            'customer', {alias}.customer_code,
            'customer_code', {alias}.customer_code,
            'qms_process', {alias}.qms_process,
            'confidentiality_level', {alias}.confidentiality_level,
            'risk_level', {alias}.risk_level,
            'approval_status', {alias}.approval_status,
            'approved_by', {alias}.approved_by,
            'is_current', {alias}.is_current
        ))
    """


def build_filter_clause(domain: str, filters: Mapping[str, object] | None = None) -> tuple[str, list[object]]:
    """Build a SQL WHERE clause for quality/operations metadata filters."""

    clauses = ["c.domain = %s", "c.embedding IS NOT NULL"]
    params: list[object] = [domain]
    if normalize_filter_value((filters or {}).get("include_obsolete")).lower() not in {"1", "true", "yes", "si"}:
        clauses.append("COALESCE(d.is_current, TRUE) IS TRUE")

    for logical_key, aliases in QUALITY_FILTER_ALIASES.items():
        value = normalize_filter_value((filters or {}).get(logical_key))
        if not value:
            continue
        pattern = f"%{value}%"
        alias_clauses: list[str] = []
        for alias in aliases:
            alias_clauses.append(
                """
                COALESCE(
                    NULLIF(d.metadata ->> %s, ''),
                    NULLIF(c.metadata ->> %s, ''),
                    NULLIF(to_jsonb(d) ->> %s, ''),
                    NULLIF(to_jsonb(c) ->> %s, ''),
                    ''
                ) ILIKE %s
                """
            )
            params.extend([alias, alias, alias, alias, pattern])
        clauses.append("(" + " OR ".join(alias_clauses) + ")")

    date_from = normalize_filter_value((filters or {}).get("date_from"))
    date_to = normalize_filter_value((filters or {}).get("date_to"))
    if date_from:
        clauses.append(f"{quality_date_expression()} >= %s::date")
        params.append(date_from)
    if date_to:
        clauses.append(f"{quality_date_expression()} <= %s::date")
        params.append(date_to)

    return " AND ".join(clauses), params


def quality_date_expression() -> str:
    """Return a safe SQL expression for document/effective dates."""

    raw_date = """
        COALESCE(
            NULLIF(d.metadata ->> 'document_date', ''),
            NULLIF(d.metadata ->> 'effective_date', ''),
            NULLIF(c.metadata ->> 'document_date', ''),
            NULLIF(c.metadata ->> 'effective_date', ''),
            NULLIF(to_jsonb(d) ->> 'document_date', ''),
            NULLIF(to_jsonb(d) ->> 'effective_date', ''),
            NULLIF(to_jsonb(c) ->> 'document_date', ''),
            NULLIF(to_jsonb(c) ->> 'effective_date', '')
        )
    """
    return f"""
        CASE
            WHEN ({raw_date}) ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
            THEN ({raw_date})::date
        END
    """


def normalize_filter_value(value: object) -> str:
    """Normalize Streamlit/CLI filter values into a compact string."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "todos", "todas"}:
        return ""
    return text
