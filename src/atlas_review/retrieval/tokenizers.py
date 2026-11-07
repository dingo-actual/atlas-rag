"""Token counting adapters for the two retrieval model tokenizers.

The counters load the tokenizer artifacts that belong to the configured
FastEmbed models and measure with truncation disabled. They are separate from
the lexical analyzer and from the inference provider's own tokenizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from fastembed import TextEmbedding
from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from tokenizers import Tokenizer

from atlas_review.config import AppConfig
from atlas_review.errors import STAGE_MODEL_ACQUISITION, AtlasReviewError
from atlas_review.hashing import sha256_hex

TOKENIZER_FILE = "tokenizer.json"


class TokenCounter(Protocol):
    """Measure text with one retrieval model tokenizer."""

    @property
    def name(self) -> str:
        """Return the model and tokenizer-artifact identity."""
        ...

    def count(self, text: str) -> int:
        """Return the token count of ``text``."""
        ...

    def count_pair(self, first: str, second: str) -> int:
        """Return the token count of a paired input."""
        ...

    def prefix_offset(self, text: str, tokens: int) -> int:
        """Return the character offset after the first ``tokens`` tokens."""
        ...

    def tail_offset(self, text: str, tokens: int) -> int:
        """Return the character offset where the last ``tokens`` tokens begin."""
        ...


@dataclass(frozen=True, slots=True)
class ModelTokenCounter:
    """Count tokens with a tokenizer loaded from a model artifact directory.

    Instance Attributes:
        name (str): Model and tokenizer-artifact identity for the manifest.
        tokenizer (Tokenizer): Truncation-free tokenizer.
    """

    name: str
    tokenizer: Tokenizer

    def count(self, text: str) -> int:
        """Return the token count, including single-input special tokens."""
        return len(self.tokenizer.encode(text).ids)

    def count_pair(self, first: str, second: str) -> int:
        """Return the paired-input token count."""
        return len(self.tokenizer.encode(first, second).ids)

    def prefix_offset(self, text: str, tokens: int) -> int:
        """Return the offset after the first ``tokens`` content tokens."""
        return _content_offsets(self.tokenizer, text, tokens, from_tail=False)

    def tail_offset(self, text: str, tokens: int) -> int:
        """Return the offset where the last ``tokens`` content tokens begin."""
        return _content_offsets(self.tokenizer, text, tokens, from_tail=True)


@dataclass(frozen=True, slots=True)
class TokenCounters:
    """The embedding and reranker counters used for one generation.

    Instance Attributes:
        embedding (TokenCounter): Counter built from the embedding artifacts.
        reranker (TokenCounter): Counter built from the reranker artifacts.
    """

    embedding: TokenCounter
    reranker: TokenCounter


def load_token_counters(config: AppConfig, cache_dir: Path) -> TokenCounters:
    """Load both tokenizers from the configured FastEmbed model artifacts.

    Args:
        config (AppConfig): Stored configuration naming the fixed models.
        cache_dir (Path): FastEmbed model cache directory for the workspace.

    Returns:
        (TokenCounters) Counters for the embedding and reranker tokenizers.

    Raises:
        AtlasReviewError: A model artifact or its tokenizer file is unavailable.
    """
    return TokenCounters(
        embedding=load_model_token_counter(
            TextEmbedding, config.embedding.model_name, cache_dir
        ),
        reranker=load_model_token_counter(
            TextCrossEncoder, config.reranker.model_name, cache_dir
        ),
    )


def load_model_token_counter(
    model_class: type[TextEmbedding] | type[TextCrossEncoder],
    model_name: str,
    cache_dir: Path,
) -> ModelTokenCounter:
    """Acquire or reuse one model directory and load its tokenizer artifact.

    Args:
        model_class (type[TextEmbedding] | type[TextCrossEncoder]): FastEmbed
            class whose artifacts define the tokenizer.
        model_name (str): Configured logical model name.
        cache_dir (Path): Model cache directory for the workspace.

    Returns:
        (ModelTokenCounter) Truncation-free tokenizer with recorded identity.

    Raises:
        AtlasReviewError: The model is unsupported, its artifacts cannot be
            acquired, or the tokenizer file is missing.
    """
    description = _model_description(model_class, model_name)
    try:
        # Each FastEmbed class narrows the generic description type it accepts.
        model_dir = Path(
            model_class.download_model(cast("Any", description), str(cache_dir))
        )
    except (OSError, ValueError) as exc:
        raise _model_error(
            "model_acquisition_failed",
            f"cannot acquire model artifacts for {model_name}: {exc}",
        ) from exc
    tokenizer_path = model_dir / TOKENIZER_FILE
    if not tokenizer_path.exists():
        raise _model_error(
            "missing_tokenizer",
            f"model artifact directory for {model_name} has no {TOKENIZER_FILE}",
        )
    return ModelTokenCounter(
        name=f"{model_name}@{sha256_hex(tokenizer_path.read_bytes())[:16]}",
        tokenizer=Tokenizer.from_file(str(tokenizer_path)),
    )


def _model_description(
    model_class: type[TextEmbedding] | type[TextCrossEncoder], model_name: str
) -> BaseModelDescription:
    """Rebuild the download description FastEmbed requires from its public listing.

    ``list_supported_models`` returns plain dictionaries while ``download_model``
    expects the description dataclass, so the adapter converts between them.
    """
    entry = next(
        (
            candidate
            for candidate in model_class.list_supported_models()
            if candidate["model"] == model_name
        ),
        None,
    )
    if entry is None:
        raise _model_error("unsupported_model", f"model {model_name} is not supported")
    sources = entry.get("sources")
    if not isinstance(sources, Mapping):
        raise _model_error(
            "unsupported_model", f"model {model_name} has no recorded artifact source"
        )
    return BaseModelDescription(
        model=model_name,
        sources=ModelSource(
            hf=cast("str | None", sources.get("hf")),
            url=cast("str | None", sources.get("url")),
            _deprecated_tar_struct=bool(sources.get("_deprecated_tar_struct", False)),
        ),
        model_file=str(entry["model_file"]),
        description=str(entry.get("description", "")),
        license=str(entry.get("license", "")),
        size_in_GB=float(cast("float", entry.get("size_in_GB", 0.0))),
        additional_files=[str(name) for name in entry.get("additional_files") or ()],
    )


def _content_offsets(
    tokenizer: Tokenizer, text: str, tokens: int, *, from_tail: bool
) -> int:
    """Map a content-token position to a character offset in ``text``."""
    offsets = [
        (start, end) for start, end in tokenizer.encode(text).offsets if end > start
    ]
    if not offsets:
        return len(text)
    if tokens <= 0:
        return len(text) if from_tail else 0
    if from_tail:
        if tokens >= len(offsets):
            return 0
        return offsets[len(offsets) - tokens][0]
    if tokens >= len(offsets):
        return len(text)
    return offsets[tokens - 1][1]


def _model_error(code: str, message: str) -> AtlasReviewError:
    """Build a model-acquisition error with the documented recovery action."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_MODEL_ACQUISITION,
        message=message,
        recovery=(
            "Check the model cache directory and network access, then retry the "
            "explicit ingestion build."
        ),
    )
