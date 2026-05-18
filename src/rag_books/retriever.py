"""Retrieval orchestration for RAG queries.

This module turns user questions into embeddings, runs vector search, and wraps
raw database rows with citation labels that the LLM can reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .db import SearchResult, VectorStore
from .embeddings import EmbeddingClient


@dataclass(frozen=True)
class RetrievedContext:
    """Context chunk selected for an answer.

    Attributes
    ----------
    source_id
        Citation id such as ``S1``.
    result
        Underlying database search result.
    """

    source_id: str
    result: SearchResult

    @property
    def citation(self) -> str:
        """Return a human-readable citation string."""

        page = ""
        if self.result.page_start and self.result.page_end:
            if self.result.page_start == self.result.page_end:
                page = f", p. {self.result.page_start}"
            else:
                page = f", pp. {self.result.page_start}-{self.result.page_end}"
        title = self.result.title or self.result.file_name
        return f"[{self.source_id}] {title}{page}"


class RAGRetriever:
    """Retrieve relevant chunks for a user question.

    Parameters
    ----------
    store
        Vector store used for search.
    embeddings
        Embedding client used to embed the question.
    """

    def __init__(self, store: VectorStore, embeddings: EmbeddingClient):
        """Initialize retrieval dependencies.

        Parameters
        ----------
        store
            Vector store.
        embeddings
            Embedding client.
        """

        self.store = store
        self.embeddings = embeddings

    def retrieve(
        self,
        question: str,
        domain: str,
        top_k: int,
        candidate_k: int | None = None,
        max_chunks_per_document: int = 2,
    ) -> list[RetrievedContext]:
        """Retrieve and label context chunks.

        Parameters
        ----------
        question
            User question.
        domain
            Logical domain/schema to search.
        top_k
            Number of final chunks to return.
        candidate_k
            Optional larger candidate pool used for document diversification.
        max_chunks_per_document
            Maximum chunks selected per document before fallback fill.

        Returns
        -------
        list[RetrievedContext]
            Retrieved contexts labeled as ``S1``, ``S2``, etc.
        """

        logger.info(
            "Retriever started. domain='{}', top_k={}, candidate_k={}, max_chunks_per_document={}, question_chars={}.",
            domain,
            top_k,
            candidate_k,
            max_chunks_per_document,
            len(question),
        )
        query_embedding = self.embeddings.embed_query(question)
        if candidate_k and candidate_k > top_k:
            results = self.store.search_diverse(
                domain=domain,
                query_embedding=query_embedding,
                top_k=top_k,
                candidate_k=candidate_k,
                max_chunks_per_document=max_chunks_per_document,
            )
        else:
            results = self.store.search(domain=domain, query_embedding=query_embedding, top_k=top_k)
        contexts = [
            RetrievedContext(source_id=f"S{index}", result=result)
            for index, result in enumerate(results, start=1)
        ]
        logger.info("Retriever finished. contexts={}.", len(contexts))
        return contexts
