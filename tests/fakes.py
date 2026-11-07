"""Deterministic stand-ins for the model-backed collaborators.

These are hand-written fakes, not mocks: each one implements the real
interface with simple, inspectable behaviour so control flow can be tested
without model downloads or provider requests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from atlas_review.inference.client import ProviderReply

_TOKEN = re.compile(r"\S+")
DIMENSION = 768
FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


@dataclass(frozen=True, slots=True)
class WhitespaceTokenCounter:
    """Count whitespace-separated words as tokens, with real character offsets."""

    name: str = "whitespace-test"

    def count(self, text: str) -> int:
        """Return the number of whitespace tokens in ``text``."""
        return len(_TOKEN.findall(text))

    def count_pair(self, first: str, second: str) -> int:
        """Return the number of whitespace tokens in the paired input."""
        return self.count(first) + self.count(second)

    def prefix_offset(self, text: str, tokens: int) -> int:
        """Return the character offset after the first ``tokens`` tokens."""
        matches = list(_TOKEN.finditer(text))
        if tokens >= len(matches):
            return len(text)
        if tokens <= 0:
            return 0
        return matches[tokens].start()

    def tail_offset(self, text: str, tokens: int) -> int:
        """Return the character offset where the last ``tokens`` tokens begin."""
        matches = list(_TOKEN.finditer(text))
        if tokens <= 0 or not matches:
            return len(text)
        if tokens >= len(matches):
            return 0
        return matches[len(matches) - tokens].start()


def bag_of_words_vector(text: str) -> np.ndarray:
    """Return a deterministic hashed bag-of-words vector, unnormalized."""
    vector = np.zeros(DIMENSION, dtype=np.float32)
    for word in re.findall(r"[a-z0-9.]+", text.lower()):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % DIMENSION
        vector[index] += 1.0
    if not vector.any():
        vector[0] = 1.0
    return vector


@dataclass(slots=True)
class RecordingEmbedder:
    """Embed hashed bag-of-words vectors and record every input text."""

    inputs: list[tuple[str, ...]] = field(default_factory=list)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unnormalized vector per input text."""
        self.inputs.append(tuple(texts))
        return np.vstack([bag_of_words_vector(text) for text in texts])

    @property
    def query_inputs(self) -> list[str]:
        """Return the texts that were embedded as queries."""
        return [batch[0] for batch in self.inputs if len(batch) == 1]


@dataclass(slots=True)
class FixedContextualizer:
    """Return a fixed JSON context note that cites the owner document."""

    note: str = "This excerpt is part of the owner record."
    reply_builder: Callable[[str], str] | None = None
    calls: list[str] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)

    def __call__(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        output_tokens: int,
        temperature: float | None,
    ) -> ProviderReply:
        """Answer one contextualization request with a validated JSON object."""
        del system_prompt, temperature
        self.calls.append(user_payload)
        self.output_tokens.append(output_tokens)
        payload = json.loads(user_payload)
        locator = payload["document"]["description"]["locator"]
        owner = payload["document"]["owner"]["id"]
        text = (
            self.reply_builder(locator)
            if self.reply_builder is not None
            else json.dumps(
                {
                    "context": f"{self.note} Owner {owner}.",
                    "support_locators": [locator],
                }
            )
        )
        return ProviderReply(
            text=text,
            model="fake-model-1",
            response_id=f"resp-{len(self.calls)}",
            input_tokens=100,
            output_tokens=20,
            duration_seconds=0.01,
        )


def stream_chunks(
    pieces: Sequence[str],
    *,
    finish_reason: FinishReason = "stop",
    usage: tuple[int, int] | None = None,
    model: str = "fake-model-1",
    response_id: str = "chatcmpl-1",
) -> tuple[ChatCompletionChunk, ...]:
    """Build a realistic streamed completion from textual deltas."""
    chunks = [
        ChatCompletionChunk(
            id=response_id,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=piece),
                    finish_reason=None,
                )
            ],
            created=0,
            model=model,
            object="chat.completion.chunk",
        )
        for piece in pieces
    ]
    chunks.append(
        ChatCompletionChunk(
            id=response_id,
            choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason=finish_reason)],
            created=0,
            model=model,
            object="chat.completion.chunk",
        )
    )
    if usage is not None:
        chunks.append(
            ChatCompletionChunk(
                id=response_id,
                choices=[],
                created=0,
                model=model,
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    total_tokens=usage[0] + usage[1],
                ),
            )
        )
    return tuple(chunks)


@dataclass(slots=True)
class FakeCompletions:
    """Return a prepared stream and record every request keyword argument."""

    chunks: tuple[ChatCompletionChunk, ...]
    calls: list[dict[str, object]] = field(default_factory=list)

    def create(self, **kwargs: object) -> Iterator[ChatCompletionChunk]:
        """Return the prepared stream for this request."""
        self.calls.append(dict(kwargs))
        return iter(self.chunks)


@dataclass(slots=True)
class FakeChatNamespace:
    """Expose the completions resource the way the SDK client does."""

    completions: FakeCompletions


@dataclass(slots=True)
class FakeChatClient:
    """Stand in for the SDK client with a recording streamed resource."""

    completions: FakeCompletions
    closed: bool = False

    @property
    def chat(self) -> FakeChatNamespace:
        """Return the chat resource holding the completions endpoint."""
        return FakeChatNamespace(self.completions)

    def close(self) -> None:
        """Record that the client was closed."""
        self.closed = True
