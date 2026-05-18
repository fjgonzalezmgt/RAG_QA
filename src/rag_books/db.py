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
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

import psycopg
from loguru import logger
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import DatabaseSettings
from .text_splitter import TextChunk


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
                    CREATE TABLE IF NOT EXISTS {documents} (
                        id UUID PRIMARY KEY,
                        domain TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        title TEXT,
                        author TEXT,
                        content_hash TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (domain, source_path, content_hash)
                    )
                    """
                )
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
                        UNIQUE (document_id, chunk_index)
                    )
                    """
                )
                cur.execute(f"CREATE INDEX IF NOT EXISTS documents_domain_idx ON {documents} (domain)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS chunks_domain_idx ON {chunks} (domain)")
            conn.commit()

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
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {documents}
                        (id, domain, source_path, file_name, title, author, content_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (domain, source_path, content_hash)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        metadata = EXCLUDED.metadata
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
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        logger.debug("Document id ready: {}.", row["id"])
        return row["id"]

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
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                            (
                                id, document_id, domain, chunk_index, page_start, page_end,
                                content, token_count, embedding, metadata
                            )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::{vector_type}, %s)
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            token_count = EXCLUDED.token_count,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
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
                        ),
                    )
                    inserted += 1
            conn.commit()
        logger.success("Upserted {} chunks for document '{}'.", inserted, document_id)
        return inserted

    def search(self, domain: str, query_embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        """Run a nearest-neighbor vector search.

        Parameters
        ----------
        domain
            Logical domain.
        query_embedding
            Query embedding vector.
        top_k
            Number of nearest chunks to return.

        Returns
        -------
        list[SearchResult]
            Search results ordered by vector distance.
        """

        logger.info("Running vector search. schema='{}', domain='{}', top_k={}.", self.schema, domain, top_k)
        chunks = qname(self.schema, "chunks")
        documents = qname(self.schema, "documents")
        vector_type = qname(self.extensions_schema, "vector")
        query_vector = vector_literal(query_embedding)

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
                        c.metadata,
                        1 - (c.embedding <=> %s::{vector_type}) AS score
                    FROM {chunks} c
                    JOIN {documents} d ON d.id = c.document_id
                    WHERE c.domain = %s AND c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> %s::{vector_type}
                    LIMIT %s
                    """,
                    (query_vector, domain, query_vector, top_k),
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
        candidates = self.search(domain=domain, query_embedding=query_embedding, top_k=candidate_k)
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
