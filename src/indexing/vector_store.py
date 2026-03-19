"""
src/indexing/vector_store.py
─────────────────────────────
FAISS-based vector store with HNSW index.
Implements the blue-green atomic index swap described in the paper (Section 4.4).
Supports hybrid retrieval: dense ANN + structured metadata filters.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import structlog

from src.indexing.chunker import DocumentChunk

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single retrieval result from the vector store."""
    chunk: DocumentChunk
    score: float          # Cosine similarity (0-1, higher = more relevant)
    rank: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "document_type": self.chunk.document_type.value,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "rank": self.rank,
            "metadata": self.chunk.metadata,
        }


class FAISSVectorStore:
    """
    FAISS HNSW vector store for semantic similarity search.

    Index configuration (from paper, Section 4.4):
      - Index type: HNSW (Hierarchical Navigable Small World)
      - M = 32 neighbours per node
      - ef_construction = 400
      - Similarity metric: cosine (inner product on L2-normalised vectors)

    Blue-green swap ensures zero-downtime re-indexing.
    """

    def __init__(self, config: dict):
        self.config = config
        self.index_path = Path(config.get("index_path", "data/faiss_index"))
        self.dimensions = config.get("dimensions", 1536)
        self.hnsw_m = config.get("hnsw_m", 32)
        self.hnsw_ef_construction = config.get("hnsw_ef_construction", 400)
        self.hnsw_ef_search = config.get("hnsw_ef_search", 64)

        self._lock = threading.RLock()
        self._index: faiss.IndexHNSWFlat | None = None
        self._chunks: list[DocumentChunk] = []  # parallel array to FAISS index
        self._id_to_position: dict[str, int] = {}

        os.makedirs(self.index_path, exist_ok=True)

        # Load existing index if present
        if self._index_files_exist():
            self._load_index()
        else:
            self._build_empty_index()

    # ── Index Construction ─────────────────────────────────────

    def _build_empty_index(self) -> None:
        """Create a fresh HNSW index configured as per the paper."""
        # HNSW index using Inner Product (for cosine after L2 normalisation)
        index = faiss.IndexHNSWFlat(self.dimensions, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search
        self._index = index
        self._chunks = []
        self._id_to_position = {}
        logger.info("Created empty FAISS HNSW index", dims=self.dimensions, M=self.hnsw_m)

    def _index_files_exist(self) -> bool:
        return (
            (self.index_path / "index.faiss").exists()
            and (self.index_path / "chunks.pkl").exists()
        )

    def _load_index(self) -> None:
        """Load existing index from disk."""
        self._index = faiss.read_index(str(self.index_path / "index.faiss"))
        self._index.hnsw.efSearch = self.hnsw_ef_search
        with open(self.index_path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        self._id_to_position = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        logger.info(
            "Loaded FAISS index",
            vectors=self._index.ntotal,
            chunks=len(self._chunks),
        )

    def _save_index(self, path: Path) -> None:
        """Persist index to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    # ── Blue-Green Atomic Index Swap ──────────────────────────

    def rebuild_index(
        self,
        chunk_embedding_pairs: list[tuple[DocumentChunk, np.ndarray]],
    ) -> None:
        """
        Build a new index from scratch and atomically hot-swap it.
        Zero-downtime: existing index serves queries until swap completes.
        """
        logger.info("Starting index rebuild", total_chunks=len(chunk_embedding_pairs))
        green_path = self.index_path / "green"

        # Build new index in background directory
        new_index = faiss.IndexHNSWFlat(self.dimensions, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        new_index.hnsw.efConstruction = self.hnsw_ef_construction
        new_index.hnsw.efSearch = self.hnsw_ef_search

        new_chunks = []
        vectors = []

        for chunk, embedding in chunk_embedding_pairs:
            normalised = self._normalise(embedding)
            vectors.append(normalised)
            new_chunks.append(chunk)

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            new_index.add(matrix)

        # Save to green directory
        self._save_index_raw(new_index, new_chunks, green_path)

        # Atomic swap under lock
        with self._lock:
            self._index = new_index
            self._chunks = new_chunks
            self._id_to_position = {c.chunk_id: i for i, c in enumerate(new_chunks)}
            # Promote green → live
            self._save_index(self.index_path)

        # Cleanup
        shutil.rmtree(green_path, ignore_errors=True)
        logger.info("Index rebuild complete", total_vectors=new_index.ntotal)

    def add_chunks(
        self,
        chunk_embedding_pairs: list[tuple[DocumentChunk, np.ndarray]],
    ) -> None:
        """Incrementally add chunks to an existing live index."""
        with self._lock:
            for chunk, embedding in chunk_embedding_pairs:
                if chunk.chunk_id in self._id_to_position:
                    continue  # Skip duplicates
                normalised = self._normalise(embedding).reshape(1, -1).astype(np.float32)
                self._index.add(normalised)
                self._id_to_position[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)

            self._save_index(self.index_path)

        logger.info(
            "Incremental index update",
            added=len(chunk_embedding_pairs),
            total=self._index.ntotal,
        )

    # ── Search ─────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Perform ANN search. Optionally filter results by metadata predicates.

        filters example: {"service": "payments-service", "priority": "P1"}
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                logger.warning("Search on empty index")
                return []

            query = self._normalise(query_embedding).reshape(1, -1).astype(np.float32)
            # Fetch extra candidates if filtering is active
            fetch_k = min(top_k * 3 if filters else top_k, self._index.ntotal)
            scores, indices = self._index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if filters and not self._matches_filters(chunk, filters):
                continue
            results.append(SearchResult(chunk=chunk, score=float(score), rank=len(results) + 1))
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: dict) -> bool:
        """Check if a chunk's metadata satisfies all filter predicates."""
        for key, value in filters.items():
            chunk_val = chunk.metadata.get(key)
            if isinstance(value, list):
                if chunk_val not in value:
                    return False
            elif chunk_val != value:
                return False
        return True

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a vector for cosine similarity via inner product."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _save_index_raw(
        index: faiss.IndexHNSWFlat,
        chunks: list[DocumentChunk],
        path: Path,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

    # ── Stats ──────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def stats(self) -> dict:
        return {
            "total_vectors": self.total_vectors,
            "total_chunks": self.total_chunks,
            "index_type": "HNSW",
            "dimensions": self.dimensions,
            "hnsw_m": self.hnsw_m,
        }
