"""bm25s keyword index construction, persistence, and mapped search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import bm25s
import numpy as np

from atlas_review.config import KeywordIndexConfig
from atlas_review.errors import STAGE_INDEX, AtlasReviewError

Hits = tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class KeywordSearch:
    """Positive-score keyword hits and the zero-score results dropped.

    Instance Attributes:
        hits (Hits): ``(passage_id, score)`` in descending score order.
        zero_score_exclusions (int): Returned filler results with score zero.
    """

    hits: Hits
    zero_score_exclusions: int


def lexical_tokens(texts: Sequence[str], config: KeywordIndexConfig) -> list[list[str]]:
    """Tokenize text with the generation's versioned lexical analyzer.

    Args:
        texts (Sequence[str]): Augmented retrieval texts or one query.
        config (KeywordIndexConfig): Analyzer settings.

    Returns:
        (list[list[str]]) String tokens per input text.
    """
    return cast(
        "list[list[str]]",
        bm25s.tokenize(
            list(texts),
            lower=config.lowercase,
            token_pattern=config.token_pattern,
            stopwords=list(config.stopwords),
            stemmer=cast("Any", None),
            return_ids=False,
            show_progress=False,
        ),
    )


def build_keyword_index(texts: Sequence[str], config: KeywordIndexConfig) -> bm25s.BM25:
    """Build a BM25 index over the same passage set as the dense index.

    Args:
        texts (Sequence[str]): Augmented retrieval texts in passage order.
        config (KeywordIndexConfig): Scoring and analyzer settings.

    Returns:
        (bm25s.BM25) Populated keyword index.
    """
    index = bm25s.BM25(
        method=config.method,
        k1=config.k1,
        b=config.b,
        dtype=config.dtype,
        backend=config.backend,
    )
    index.index(lexical_tokens(texts, config), show_progress=False)
    return index


def save_keyword_index(
    index: bm25s.BM25, directory: Path, config: KeywordIndexConfig
) -> None:
    """Persist the keyword index without pickle and without its corpus.

    Args:
        index (bm25s.BM25): Populated keyword index.
        directory (Path): Bundle subdirectory for the generated files.
        config (KeywordIndexConfig): Settings the index was built with.
    """
    directory.mkdir(parents=True, exist_ok=True)
    index.save(
        str(directory),
        allow_pickle=False,
        show_progress=False,
    )
    verify_saved_settings(directory, config)


def load_keyword_index(directory: Path) -> bm25s.BM25:
    """Load a keyword index without an embedded corpus.

    Args:
        directory (Path): Bundle subdirectory holding the generated files.

    Returns:
        (bm25s.BM25) Keyword index returning numeric document positions.

    Raises:
        AtlasReviewError: The files are missing, unreadable, or mismatched.
    """
    try:
        index = bm25s.BM25.load(
            str(directory),
            load_corpus=False,
            mmap=False,
            allow_pickle=False,
            show_progress=False,
            verbosity=0,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise _index_error(
            "unreadable_keyword_index", f"cannot load {directory.name}"
        ) from exc
    if index.corpus is not None:
        raise _index_error(
            "keyword_index_has_corpus",
            "the stored keyword index embeds a corpus instead of passage positions",
        )
    return index


def keyword_search(
    index: bm25s.BM25,
    focus: str,
    *,
    limit: int,
    passage_ids_by_ordinal: Sequence[int],
    config: KeywordIndexConfig,
) -> KeywordSearch:
    """Search BM25 and resolve document positions through the stored mapping.

    Args:
        index (bm25s.BM25): Loaded keyword index.
        focus (str): Ordinary retrieval focus without the embedding instruction.
        limit (int): Maximum hits, bounded by the document count.
        passage_ids_by_ordinal (Sequence[int]): Ordinal to passage-ID mapping.
        config (KeywordIndexConfig): Analyzer settings for the query.

    Returns:
        (KeywordSearch) Positive-score hits, best first, and the dropped count.

    Raises:
        AtlasReviewError: The mapping, ordinals, or scores are invalid.
    """
    mapping = _checked_mapping(passage_ids_by_ordinal, index)
    tokens = lexical_tokens([focus], config)
    count = min(limit, len(mapping))
    if count <= 0 or not tokens[0]:
        return KeywordSearch((), 0)
    results = index.retrieve(
        tokens,
        k=count,
        sorted=True,
        show_progress=False,
        n_threads=1,
    )
    documents, scores = _result_arrays(results)
    if documents.shape != (1, count) or scores.shape != (1, count):
        raise _index_error(
            "invalid_keyword_result", "keyword search returned an unexpected shape"
        )
    hits: list[tuple[int, float]] = []
    dropped = 0
    for ordinal, score in zip(documents[0], scores[0], strict=True):
        position = int(ordinal)
        if not 0 <= position < len(mapping):
            raise _index_error(
                "invalid_keyword_ordinal",
                "keyword search returned an ordinal outside the mapping",
            )
        value = float(score)
        if not math.isfinite(value):
            raise _index_error(
                "invalid_keyword_score", "keyword search returned a non-finite score"
            )
        if value > 0.0:
            hits.append((mapping[position], value))
        else:
            dropped += 1
    return KeywordSearch(
        tuple(sorted(hits, key=lambda hit: (-hit[1], hit[0]))), dropped
    )


def verify_saved_settings(directory: Path, config: KeywordIndexConfig) -> None:
    """Check the saved scoring parameters match the recorded settings.

    Args:
        directory (Path): Bundle subdirectory holding the generated files.
        config (KeywordIndexConfig): Settings the index was built with.

    Raises:
        AtlasReviewError: The saved parameters disagree with the configuration.
    """
    index = load_keyword_index(directory)
    if index.method != config.method or index.k1 != config.k1 or index.b != config.b:
        raise _index_error(
            "keyword_settings_mismatch",
            "the saved keyword index does not match the configured settings",
        )


def _result_arrays(results: object) -> tuple[np.ndarray, np.ndarray]:
    """Read the bm25s result arrays as plain NumPy arrays."""
    typed = cast("Any", results)
    return (
        np.asarray(typed.documents, dtype=np.int64),
        np.asarray(typed.scores, dtype=np.float64),
    )


def _checked_mapping(
    passage_ids_by_ordinal: Sequence[int], index: bm25s.BM25
) -> tuple[int, ...]:
    """Validate the ordinal mapping against the loaded index."""
    document_count = int(index.scores["num_docs"])
    mapping = tuple(passage_ids_by_ordinal)
    if len(mapping) != document_count:
        raise _index_error(
            "invalid_keyword_mapping",
            "the passage-ID mapping does not cover the keyword index",
        )
    if any(type(value) is not int or not 0 < value < 2**63 for value in mapping):
        raise _index_error(
            "invalid_keyword_mapping", "the mapping contains an unusable passage ID"
        )
    if len(set(mapping)) != len(mapping):
        raise _index_error(
            "invalid_keyword_mapping", "the mapping repeats a passage ID"
        )
    return mapping


def _index_error(code: str, message: str) -> AtlasReviewError:
    """Build an index-integrity error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_INDEX,
        message=message,
        recovery="Rebuild the index bundle explicitly from stored vectors and text.",
    )
