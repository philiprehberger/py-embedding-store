"""In-memory vector store with multi-metric similarity search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = ["VectorStore", "SearchResult"]

_VALID_METRICS = ("cosine", "dot", "euclidean", "manhattan")


@dataclass
class SearchResult:
    """A single search result."""

    id: str
    score: float
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id!r}, score={self.score:.4f})"


@dataclass
class _Entry:
    id: str
    embedding: np.ndarray
    metadata: dict[str, Any]


class VectorStore:
    """In-memory vector store with multi-metric similarity search.

    Supports cosine, dot-product, euclidean, and manhattan distance metrics.
    Useful for prototyping RAG, semantic search, and similar applications
    without needing an external vector database.
    """

    def __init__(
        self,
        dimensions: int | None = None,
        metric: str = "cosine",
    ) -> None:
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                f"Valid metrics: {', '.join(_VALID_METRICS)}"
            )
        self._dimensions = dimensions
        self._metric = metric
        self._entries: dict[str, _Entry] = {}

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(
        self,
        id: str,
        embedding: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector with metadata.

        Args:
            id: Unique identifier.
            embedding: Vector as a list of floats or numpy array.
            metadata: Optional metadata dict.
        """
        vec = np.asarray(embedding, dtype=np.float32)
        if self._dimensions is None:
            self._dimensions = len(vec)
        elif len(vec) != self._dimensions:
            raise ValueError(
                f"Expected {self._dimensions} dimensions, got {len(vec)}"
            )

        self._entries[id] = _Entry(
            id=id,
            embedding=vec,
            metadata=metadata or {},
        )

    def add_many(
        self,
        items: list[tuple[str, list[float] | np.ndarray, dict[str, Any] | None]],
    ) -> None:
        """Add multiple vectors at once.

        Args:
            items: List of (id, embedding, metadata) tuples. Metadata may be
                ``None`` or omitted (2-element tuples are accepted).
        """
        for item in items:
            if len(item) == 3:
                id, embedding, metadata = item
            else:
                id, embedding = item[0], item[1]
                metadata = None
            self.add(id, embedding, metadata)

    def get(self, id: str) -> _Entry | None:
        """Get an entry by ID."""
        return self._entries.get(id)

    def delete(self, id: str) -> bool:
        """Delete an entry by ID.

        Returns:
            ``True`` if the entry existed and was removed.
        """
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    def remove(self, id: str) -> bool:
        """Remove an entry by ID. Alias for :meth:`delete`.

        Returns:
            ``True`` if the entry existed and was removed.
        """
        return self.delete(id)

    def update_metadata(self, id: str, metadata: dict[str, Any]) -> bool:
        """Update metadata for an existing entry."""
        entry = self._entries.get(id)
        if entry:
            entry.metadata.update(metadata)
            return True
        return False

    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
        metric: str | None = None,
        filter: Callable[[dict[str, Any]], bool] | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            metric: Distance metric override. Defaults to the store-level
                metric set in the constructor.
            filter: Optional metadata filter function.
            min_score: Minimum similarity score threshold.

        Returns:
            List of SearchResult objects sorted by score (highest first).
        """
        if not self._entries:
            return []

        effective_metric = metric or self._metric
        if effective_metric not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric: {effective_metric!r}. "
                f"Valid metrics: {', '.join(_VALID_METRICS)}"
            )

        query = np.asarray(query_embedding, dtype=np.float32)

        candidates = list(self._entries.values())
        if filter:
            candidates = [e for e in candidates if filter(e.metadata)]

        if not candidates:
            return []

        # Build matrix for vectorized computation
        matrix = np.stack([e.embedding for e in candidates])
        scores = self._compute_scores(query, matrix, effective_metric)

        results: list[SearchResult] = []
        indices = np.argsort(scores)[::-1][:top_k]

        for idx in indices:
            score = float(scores[idx])
            if min_score is not None and score < min_score:
                continue
            entry = candidates[idx]
            results.append(SearchResult(
                id=entry.id,
                score=score,
                metadata=dict(entry.metadata),
            ))

        return results

    def search_many(
        self,
        query_embeddings: list[list[float] | np.ndarray],
        top_k: int = 5,
        metric: str | None = None,
        filter: Callable[[dict[str, Any]], bool] | None = None,
        min_score: float | None = None,
    ) -> list[list[SearchResult]]:
        """Search for similar vectors for multiple queries at once.

        Args:
            query_embeddings: List of query vectors.
            top_k: Number of results per query.
            metric: Distance metric override.
            filter: Optional metadata filter function.
            min_score: Minimum similarity score threshold.

        Returns:
            List of result lists, one per query embedding.
        """
        return [
            self.search(
                query_embedding=q,
                top_k=top_k,
                metric=metric,
                filter=filter,
                min_score=min_score,
            )
            for q in query_embeddings
        ]

    @staticmethod
    def _compute_scores(
        query: np.ndarray,
        matrix: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """Compute similarity scores between a query and a matrix of vectors."""
        if metric == "cosine":
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return np.zeros(len(matrix))
            matrix_norms = np.linalg.norm(matrix, axis=1)
            matrix_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)
            return (matrix @ query) / (matrix_norms * query_norm)
        elif metric == "dot":
            return matrix @ query
        elif metric == "euclidean":
            distances = np.linalg.norm(matrix - query, axis=1)
            return 1.0 / (1.0 + distances)
        elif metric == "manhattan":
            distances = np.sum(np.abs(matrix - query), axis=1)
            return 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

    def save(self, path: str | Path) -> None:
        """Save the vector store to a JSON file.

        Args:
            path: File path to write. Parent directories must exist.
        """
        data = {
            "dimensions": self._dimensions,
            "metric": self._metric,
            "entries": [
                {
                    "id": e.id,
                    "embedding": e.embedding.tolist(),
                    "metadata": e.metadata,
                }
                for e in self._entries.values()
            ],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """Load a vector store from a JSON file.

        Args:
            path: File path to read.

        Returns:
            A new VectorStore populated with the saved entries.
        """
        data = json.loads(Path(path).read_text())
        store = cls(
            dimensions=data.get("dimensions"),
            metric=data.get("metric", "cosine"),
        )
        for entry in data.get("entries", []):
            store.add(
                id=entry["id"],
                embedding=entry["embedding"],
                metadata=entry.get("metadata", {}),
            )
        return store

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def ids(self) -> list[str]:
        """Get all entry IDs."""
        return list(self._entries.keys())

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def __contains__(self, id: str) -> bool:
        """Check if an ID exists in the store."""
        return id in self._entries
