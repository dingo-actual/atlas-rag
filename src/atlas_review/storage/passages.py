"""Passage and accepted-context persistence over a workspace database."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from atlas_review.contracts import ContextualizationResult, Passage
from atlas_review.errors import STAGE_STORAGE, AtlasReviewError
from atlas_review.storage.codec import decode_model, encode_model
from atlas_review.storage.database import WorkspaceDatabase

_INSERT_PASSAGE = """
INSERT INTO passages (
    passage_id, generation_id, entity_id, edge_id, section, ordinal,
    bm25_ordinal, source_text, context_text, retrieval_text, context_cache_key,
    source_sha256, content_sha256, embedding_tokens, reranker_tokens,
    vector_f32_le, payload_msgpack
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_CONTEXT = """
INSERT INTO context_cache (
    cache_key, created_at, document_sha256, chunk_sha256,
    contextualizer_fingerprint, context_text, payload_msgpack
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (cache_key) DO NOTHING
"""


@dataclass(frozen=True, slots=True)
class StoredIndexRow:
    """One stored passage as the raw input of a derived index.

    Instance Attributes:
        passage_id (int): SQLite passage identifier used as the dense key.
        bm25_ordinal (int): Contiguous ordinal used as the keyword position.
        retrieval_text (str): Augmented text indexed by BM25.
        vector (bytes): Explicit little-endian float32 vector blob.
    """

    passage_id: int
    bm25_ordinal: int
    retrieval_text: str
    vector: bytes


class PassageRepository:
    """Passage rows, vectors, and the immutable chunk-context cache.

    Instance Attributes:
        _database (WorkspaceDatabase): Open workspace connection factory.
    """

    def __init__(self, database: WorkspaceDatabase) -> None:
        """Bind the repository to one workspace database.

        Args:
            database (WorkspaceDatabase): Workspace connection factory.
        """
        self._database = database

    def next_passage_id(self, connection: sqlite3.Connection) -> int:
        """Return the next free passage identifier for explicit assignment."""
        row = connection.execute(
            "SELECT coalesce(max(passage_id), 0) FROM passages"
        ).fetchone()
        return int(row[0]) + 1

    def insert_passages(
        self,
        connection: sqlite3.Connection,
        passages: Sequence[Passage],
        vectors: Sequence[bytes],
    ) -> None:
        """Insert passage rows and their explicit vector blobs.

        Args:
            connection (sqlite3.Connection): Open immediate transaction.
            passages (Sequence[Passage]): Validated passage records.
            vectors (Sequence[bytes]): Little-endian float32 blobs of 3072 bytes.

        Raises:
            AtlasReviewError: The rows and vectors disagree in count.
        """
        if len(passages) != len(vectors):
            raise _storage_error(
                "invalid_passage_inputs", "passage rows and vector blobs disagree"
            )
        connection.executemany(
            _INSERT_PASSAGE,
            [
                (
                    passage.passage_id,
                    passage.generation_id,
                    passage.entity_id,
                    passage.edge_id,
                    passage.section,
                    passage.ordinal,
                    passage.bm25_ordinal,
                    passage.source_text,
                    passage.context_text,
                    passage.retrieval_text,
                    passage.context_cache_key,
                    passage.source_sha256,
                    passage.content_sha256,
                    passage.embedding_tokens,
                    passage.reranker_tokens,
                    vector,
                    encode_model(passage),
                )
                for passage, vector in zip(passages, vectors, strict=True)
            ],
        )

    def passages(self, generation_id: str) -> tuple[Passage, ...]:
        """Return every passage of a generation in BM25 ordinal order."""
        rows = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM passages WHERE generation_id = ? "
                "ORDER BY bm25_ordinal",
                (generation_id,),
            )
            .fetchall()
        )
        return tuple(decode_model(row["payload_msgpack"], Passage) for row in rows)

    def index_rows(self, generation_id: str) -> tuple[StoredIndexRow, ...]:
        """Return raw index inputs of one generation in BM25 ordinal order.

        Args:
            generation_id (str): Stored generation identifier.

        Returns:
            (tuple[StoredIndexRow, ...]) Passage identifiers, ordinals, text,
                and vectors in the order both indexes must use.
        """
        rows = (
            self._database.connect()
            .execute(
                "SELECT passage_id, bm25_ordinal, retrieval_text, vector_f32_le "
                "FROM passages WHERE generation_id = ? ORDER BY bm25_ordinal",
                (generation_id,),
            )
            .fetchall()
        )
        return tuple(
            StoredIndexRow(
                passage_id=int(row["passage_id"]),
                bm25_ordinal=int(row["bm25_ordinal"]),
                retrieval_text=str(row["retrieval_text"]),
                vector=bytes(row["vector_f32_le"]),
            )
            for row in rows
        )

    def passage(self, generation_id: str, passage_id: int) -> Passage:
        """Decode one passage or fail with a storage error."""
        row = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM passages "
                "WHERE generation_id = ? AND passage_id = ?",
                (generation_id, passage_id),
            )
            .fetchone()
        )
        if row is None:
            raise _storage_error(
                "unknown_passage",
                f"passage {passage_id} is not stored in this generation",
            )
        return decode_model(row["payload_msgpack"], Passage)

    def save_context_entry(
        self,
        result: ContextualizationResult,
        *,
        document_sha256: str,
        chunk_sha256: str,
    ) -> None:
        """Persist one accepted context entry in its own transaction.

        Args:
            result (ContextualizationResult): Validated cache entry.
            document_sha256 (str): Digest of the owner document.
            chunk_sha256 (str): Digest of the target excerpt text.
        """
        with self._database.transaction() as connection:
            connection.execute(
                _INSERT_CONTEXT,
                (
                    result.cache_key,
                    result.created_at,
                    document_sha256,
                    chunk_sha256,
                    result.request_fingerprint,
                    result.context_text,
                    encode_model(result),
                ),
            )

    def context_entry(self, cache_key: str) -> ContextualizationResult | None:
        """Decode and revalidate one cached context entry."""
        row = (
            self._database.connect()
            .execute(
                "SELECT payload_msgpack FROM context_cache WHERE cache_key = ?",
                (cache_key,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return decode_model(row["payload_msgpack"], ContextualizationResult)


def _storage_error(code: str, message: str) -> AtlasReviewError:
    """Build a storage error with the documented recovery action."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_STORAGE,
        message=message,
        recovery="Rebuild the generation explicitly or inspect the workspace.",
    )
