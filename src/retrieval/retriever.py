"""
src/retrieval/retriever.py
───────────────────────────
Semantic retrieval engine combining:
  1. FAISS ANN search (dense retrieval)
  2. Cross-encoder re-ranking (ms-marco-MiniLM)
  3. HyDE query expansion (when initial results are sparse)

Corresponds to Section 5.3 of the paper.
"""

from __future__ import annotations

import structlog

from src.indexing.embedder import Embedder
from src.indexing.vector_store import FAISSVectorStore, SearchResult
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.query_expander import HyDEQueryExpander

logger = structlog.get_logger(__name__)


class SemanticRetriever:
    """
    Main retrieval engine as described in the paper.

    Pipeline per query:
      1. Embed query → query vector
      2. FAISS top-k ANN search (with optional metadata filters)
      3. Cross-encoder re-ranking
      4. Threshold filtering
      5. Optional HyDE expansion if results < min_docs threshold
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: Embedder,
        reranker: CrossEncoderReranker,
        query_expander: HyDEQueryExpander | None = None,
        config: dict | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.query_expander = query_expander
        cfg = config or {}
        self.top_k = cfg.get("top_k", 10)
        self.rerank_top_k = cfg.get("rerank_top_k", 5)
        self.rerank_threshold = cfg.get("rerank_threshold", 0.45)
        self.min_docs_before_expansion = cfg.get("min_docs_before_expansion", 3)
        self.hyde_enabled = cfg.get("hyde_enabled", True)

    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Full retrieval pipeline for a query string.
        Returns re-ranked, threshold-filtered results.
        """
        logger.info("Starting retrieval", query_preview=query[:100])

        # Step 1: Embed query
        query_embedding = self.embedder.embed_text(query)

        # Step 2: ANN search
        candidates = self.vector_store.search(
            query_embedding, top_k=self.top_k, filters=filters
        )
        logger.debug("ANN candidates", count=len(candidates))

        # Step 3: Re-rank
        if candidates:
            candidates = self.reranker.rerank(query, candidates)

        # Step 4: Threshold filter
        passing = [r for r in candidates if r.score >= self.rerank_threshold]
        logger.debug("After threshold filter", count=len(passing), threshold=self.rerank_threshold)

        # Step 5: HyDE expansion if sparse results
        if (
            len(passing) < self.min_docs_before_expansion
            and self.hyde_enabled
            and self.query_expander is not None
        ):
            logger.info(
                "Sparse results — triggering HyDE expansion",
                passing=len(passing),
                min_required=self.min_docs_before_expansion,
            )
            passing = self._hyde_expand(query, filters, existing=passing)

        # Assign final ranks
        for i, result in enumerate(passing[: self.rerank_top_k]):
            result.rank = i + 1

        final = passing[: self.rerank_top_k]
        logger.info("Retrieval complete", results_returned=len(final))
        return final

    def _hyde_expand(
        self,
        query: str,
        filters: dict | None,
        existing: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Hypothetical Document Embedding (HyDE) expansion.
        Generates a synthetic exemplar document and retrieves against it.
        """
        existing_ids = {r.chunk.chunk_id for r in existing}
        expanded_query = self.query_expander.expand(query)
        expanded_embedding = self.embedder.embed_text(expanded_query)

        new_candidates = self.vector_store.search(
            expanded_embedding, top_k=self.top_k, filters=filters
        )
        # Merge, avoiding duplicates
        merged = list(existing)
        for result in new_candidates:
            if result.chunk.chunk_id not in existing_ids:
                merged.append(result)
                existing_ids.add(result.chunk.chunk_id)

        if merged:
            merged = self.reranker.rerank(query, merged)

        return [r for r in merged if r.score >= self.rerank_threshold]
