"""OpenAI-compatible client, credential resolution, and streaming collection.

The client is built from stored configuration. The API key is resolved from
the named environment variable at call time and is never stored or logged.
Responses are streamed because at least one supported provider only allows
long outputs in streaming mode. No tools or provider response format are
sent, SDK retries are disabled, and a null temperature is omitted.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from openai import Omit, OpenAI, OpenAIError, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)

from atlas_review.config import LLMConfig
from atlas_review.errors import (
    STAGE_LLM_AUTHENTICATION,
    STAGE_LLM_PROVIDER,
    AtlasReviewError,
)
from atlas_review.inference.validate import MAX_REPORT_BYTES

MAX_CONTEXT_RESPONSE_BYTES = 16 * 1024
MAX_REVIEW_RESPONSE_BYTES = MAX_REPORT_BYTES
PLACEHOLDER_API_KEY = "not-required"
MAX_TOKENS_PARAMETER = "max_tokens"
JSON_RESPONSE_FORMAT = "json_object"


class MissingProviderCredentialError(ValueError):
    """Raised when the configured credential environment variable is unusable."""


@dataclass(frozen=True, slots=True)
class ProviderReply:
    """Sanitized result of one provider request.

    Instance Attributes:
        text (str): Assembled response text.
        model (str | None): Provider-reported model identifier.
        response_id (str | None): Provider response identifier.
        input_tokens (int | None): Reported prompt tokens, when present.
        output_tokens (int | None): Reported completion tokens, when present.
        duration_seconds (float): Monotonic wall time of the request.
    """

    text: str
    model: str | None
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float


class Contextualizer(Protocol):
    """Answer one chunk-contextualization request."""

    def __call__(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        output_tokens: int,
        temperature: float | None,
    ) -> ProviderReply:
        """Return the provider reply for one chunk."""
        ...


def resolve_api_key(config: LLMConfig, environ: Mapping[str, str] | None = None) -> str:
    """Resolve the configured credential without falling back to another name.

    Args:
        config (LLMConfig): Provider settings naming the credential policy.
        environ (Mapping[str, str] | None): Test seam; the process environment
            is used when omitted.

    Returns:
        (str) The resolved key, or the fixed placeholder in unauthenticated mode.

    Raises:
        MissingProviderCredentialError: The named variable is missing or blank.
    """
    if config.api_key_mode == "none":
        return PLACEHOLDER_API_KEY
    if config.api_key_env is None:
        raise MissingProviderCredentialError(
            "environment authentication requires a configured variable name"
        )
    values = os.environ if environ is None else environ
    value = values.get(config.api_key_env)
    if value is None or not value.strip():
        raise MissingProviderCredentialError(
            f"required environment variable {config.api_key_env} is missing or empty"
        )
    return value


def build_contextualizer(config: LLMConfig) -> ChatContextualizer:
    """Build the contextualizer, resolving the credential before any request.

    Contextualization never asks for a provider response format: its output is
    validated locally against the `ChunkContext` contract.

    Args:
        config (LLMConfig): Provider settings.

    Returns:
        (ChatContextualizer) Adapter bound to the configured endpoint.

    Raises:
        AtlasReviewError: The configured credential is unusable.
    """
    return _chat_adapter(
        config,
        max_response_bytes=MAX_CONTEXT_RESPONSE_BYTES,
        request_json_mode=False,
    )


def build_reviewer(config: LLMConfig) -> ChatContextualizer:
    """Build the review-synthesis adapter on the shared Chat Completions client.

    Review synthesis accepts the larger report response bound and sends the
    configured JSON response format, which contextualization never does.

    Args:
        config (LLMConfig): Provider settings.

    Returns:
        (ChatContextualizer) Adapter bound to the configured endpoint.

    Raises:
        AtlasReviewError: The configured credential is unusable.
    """
    return _chat_adapter(
        config,
        max_response_bytes=MAX_REVIEW_RESPONSE_BYTES,
        request_json_mode=True,
    )


def _chat_adapter(
    config: LLMConfig, *, max_response_bytes: int, request_json_mode: bool
) -> ChatContextualizer:
    """Resolve the credential and build one configured adapter."""
    try:
        api_key = resolve_api_key(config)
    except MissingProviderCredentialError as exc:
        raise AtlasReviewError(
            code="missing_provider_credential",
            stage=STAGE_LLM_AUTHENTICATION,
            message=str(exc),
            recovery=(
                "Set the configured environment variable in the shell that "
                "launches the application, then retry or restart it."
            ),
        ) from exc
    return ChatContextualizer(
        config,
        OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        ),
        max_response_bytes=max_response_bytes,
        request_json_mode=request_json_mode,
    )


@dataclass(slots=True)
class ChatContextualizer:
    """Send one streamed Chat Completions request through the configured client.

    Instance Attributes:
        config (LLMConfig): Provider settings used for the request.
        client (OpenAI): Client constructed with SDK retries disabled.
        max_response_bytes (int): Bound on the assembled response text.
        request_json_mode (bool): Whether the configured JSON response format
            is sent with the request (review synthesis only).
    """

    config: LLMConfig
    client: OpenAI
    max_response_bytes: int = MAX_CONTEXT_RESPONSE_BYTES
    request_json_mode: bool = False

    def __call__(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        output_tokens: int,
        temperature: float | None,
    ) -> ProviderReply:
        """Stream one request and return its assembled, bounded reply.

        Args:
            system_prompt (str): System instructions.
            user_payload (str): JSON request payload.
            output_tokens (int): Output allowance for this request.
            temperature (float | None): Optional sampling temperature; a null
                value is omitted from the request because some models reject it.

        Returns:
            (ProviderReply) Nonempty, untruncated response text and metadata.

        Raises:
            AtlasReviewError: The provider failed or returned unusable output.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        started = time.monotonic()
        try:
            stream = self._create(messages, output_tokens, temperature)
            reply = collect_stream(
                stream,
                duration=time.monotonic() - started,
                max_bytes=self.max_response_bytes,
            )
        except OpenAIError as exc:
            raise _provider_error(exc) from exc
        return reply

    def _create(
        self,
        messages: list[ChatCompletionMessageParam],
        output_tokens: int,
        temperature: float | None,
    ) -> Iterator[ChatCompletionChunk]:
        """Open the streamed completion with the configured token parameter."""
        response_format: ResponseFormatJSONObject | Omit = self._response_format()
        if self.config.output_token_parameter == MAX_TOKENS_PARAMETER:
            return self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                response_format=response_format,
                max_tokens=output_tokens,
                temperature=self._sampling(temperature),
            )
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            response_format=response_format,
            max_completion_tokens=output_tokens,
            temperature=self._sampling(temperature),
        )

    def _response_format(self) -> ResponseFormatJSONObject | Omit:
        """Return the configured JSON response format, or omit it."""
        if (
            not self.request_json_mode
            or self.config.response_format_mode != JSON_RESPONSE_FORMAT
        ):
            return omit
        return ResponseFormatJSONObject(type="json_object")

    @staticmethod
    def _sampling(temperature: float | None) -> float | Omit:
        """Omit a null temperature because some compatible models reject it."""
        return omit if temperature is None else temperature

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()


def collect_stream(
    chunks: Iterator[ChatCompletionChunk],
    *,
    duration: float,
    max_bytes: int = MAX_CONTEXT_RESPONSE_BYTES,
) -> ProviderReply:
    """Assemble one streamed completion and reject unusable output.

    Args:
        chunks (Iterator[ChatCompletionChunk]): Streamed deltas in order.
        duration (float): Wall time already spent on the request.
        max_bytes (int): Largest accepted response size in UTF-8 bytes.

    Returns:
        (ProviderReply) Assembled text and reportable metadata.

    Raises:
        AtlasReviewError: The stream is empty, refused, truncated, tool-calling,
            filtered, or oversized.
    """
    parts: list[str] = []
    size = 0
    finish_reason: str | None = None
    model: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    for chunk in chunks:
        model = _optional_text(chunk.model) or model
        response_id = _optional_text(chunk.id) or response_id
        usage = chunk.usage
        if usage is not None:
            input_tokens = _optional_count(usage.prompt_tokens) or input_tokens
            output_tokens = _optional_count(usage.completion_tokens) or output_tokens
        for choice in chunk.choices:
            finish_reason = choice.finish_reason or finish_reason
            if choice.delta.tool_calls or choice.delta.refusal:
                raise _output_error(
                    "unsupported_provider_response",
                    "provider refused or answered with a tool call instead of text",
                )
            piece = choice.delta.content
            if not piece:
                continue
            size += len(piece.encode("utf-8"))
            if size > max_bytes:
                raise _output_error(
                    "oversized_provider_response",
                    f"provider response exceeds {max_bytes} bytes",
                )
            parts.append(piece)
    if finish_reason == "length":
        raise _output_error(
            "truncated_provider_response",
            "provider response was terminated for length",
        )
    if finish_reason == "content_filter":
        raise _output_error(
            "filtered_provider_response",
            "provider response was withheld by a content filter",
        )
    text = "".join(parts)
    if not text.strip():
        raise _output_error("empty_provider_response", "provider returned no text")
    return ProviderReply(
        text=text,
        model=model,
        response_id=response_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration,
    )


def _optional_text(value: object) -> str | None:
    """Return non-blank text or ``None``."""
    return value if isinstance(value, str) and value.strip() else None


def _optional_count(value: object) -> int | None:
    """Return a reported token count or ``None`` when the provider omitted it."""
    return value if type(value) is int and value >= 0 else None


def _provider_error(exc: OpenAIError) -> AtlasReviewError:
    """Map a provider failure to a sanitized application error."""
    return AtlasReviewError(
        code="provider_request_failed",
        stage=STAGE_LLM_PROVIDER,
        message=f"the provider request failed: {type(exc).__name__}",
        recovery="Check the endpoint, model, and credentials, then retry explicitly.",
    )


def _output_error(code: str, message: str) -> AtlasReviewError:
    """Build an unusable-response error."""
    return AtlasReviewError(
        code=code,
        stage=STAGE_LLM_PROVIDER,
        message=message,
        recovery="Inspect the provider and model settings, then retry explicitly.",
    )
