"""Inference adapter behaviour: credentials and streamed response assembly."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import pytest
from freezegun import freeze_time
from openai import OpenAI, OpenAIError, omit
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from atlas_review.config import LLMConfig
from atlas_review.errors import AtlasReviewError
from atlas_review.inference.client import (
    MAX_CONTEXT_RESPONSE_BYTES,
    MAX_REVIEW_RESPONSE_BYTES,
    PLACEHOLDER_API_KEY,
    RETRY_BASE_DELAY_SECONDS,
    ChatContextualizer,
    MissingProviderCredentialError,
    build_contextualizer,
    build_reviewer,
    collect_stream,
    is_transient_provider_error,
    resolve_api_key,
)
from fakes import FakeChatClient, FakeCompletions, FinishReason, stream_chunks

CONFIG = LLMConfig(base_url="https://provider.example/v1", model="fake-model-1")


def provider(**overrides: object) -> LLMConfig:
    """Return a provider configuration with explicit overrides."""
    return LLMConfig.model_validate(
        {**CONFIG.model_dump(mode="json"), **overrides}, strict=True
    )


def contextualizer(
    config: LLMConfig, chunks: tuple[ChatCompletionChunk, ...]
) -> tuple[ChatContextualizer, FakeCompletions]:
    """Return a contextualizer bound to a recording fake client."""
    return adapter(config, chunks)


@dataclass(slots=True)
class Retries:
    """Record retry notices and backoff waits for one adapter.

    Instance Attributes:
        limit (int): Configured automatic retry count.
        notices (list[str]): Notices reported before each retry.
        sleeps (list[float]): Backoff waits in order.
    """

    limit: int = 0
    notices: list[str] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)


def adapter(
    config: LLMConfig,
    chunks: tuple[ChatCompletionChunk, ...],
    *,
    max_response_bytes: int = MAX_CONTEXT_RESPONSE_BYTES,
    request_json_mode: bool = False,
    errors: tuple[OpenAIError | None, ...] = (),
    retries: Retries | None = None,
) -> tuple[ChatContextualizer, FakeCompletions]:
    """Return an adapter bound to a recording fake client."""
    recording = Retries() if retries is None else retries
    completions = FakeCompletions(chunks=chunks, errors=errors)
    client = cast("OpenAI", FakeChatClient(completions=completions))
    return (
        ChatContextualizer(
            config,
            client,
            max_response_bytes=max_response_bytes,
            request_json_mode=request_json_mode,
            automatic_retries=recording.limit,
            notify=recording.notices.append,
            sleep=recording.sleeps.append,
        ),
        completions,
    )


def refusal_chunk() -> ChatCompletionChunk:
    """Return a streamed delta that refuses the request."""
    return ChatCompletionChunk(
        id="chatcmpl-1",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(refusal="I cannot help with that."),
                finish_reason=None,
            )
        ],
        created=0,
        model="fake-model-1",
        object="chat.completion.chunk",
    )


def tool_call_chunk() -> ChatCompletionChunk:
    """Return a streamed delta that asks for a tool call."""
    return ChatCompletionChunk(
        id="chatcmpl-1",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call-1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name="lookup", arguments="{}"
                            ),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
        created=0,
        model="fake-model-1",
        object="chat.completion.chunk",
    )


class TestCredentials:
    """The credential comes only from the configured variable name."""

    @pytest.mark.parametrize(
        ("environment", "expected"),
        [
            pytest.param({"ATLAS_REVIEW_API_KEY": "value"}, "value", id="present"),
            pytest.param({}, "error", id="missing"),
            pytest.param({"ATLAS_REVIEW_API_KEY": "  "}, "error", id="blank"),
            pytest.param(
                {"OPENAI_API_KEY": "other"}, "error", id="no-fallback-variable"
            ),
            pytest.param(
                {"atlas_review_api_key": "lower"}, "error", id="case-sensitive"
            ),
        ],
    )
    def test_named_variable_is_the_only_source(
        self, environment: dict[str, str], expected: str
    ) -> None:
        if expected == "error":
            with pytest.raises(MissingProviderCredentialError):
                resolve_api_key(CONFIG, environment)
            return
        assert resolve_api_key(CONFIG, environment) == expected

    def test_unauthenticated_mode_uses_a_placeholder(self) -> None:
        config = LLMConfig.model_validate(
            {
                **CONFIG.model_dump(mode="json"),
                "api_key_mode": "none",
                "api_key_env": None,
            },
            strict=True,
        )
        assert resolve_api_key(config, {}) == PLACEHOLDER_API_KEY

    def test_build_reports_the_missing_variable_as_an_application_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ATLAS_REVIEW_API_KEY", raising=False)
        with pytest.raises(AtlasReviewError) as caught:
            build_contextualizer(CONFIG)
        assert caught.value.code == "missing_provider_credential"
        assert caught.value.stage == "llm_authentication"


class TestStreamCollection:
    """Streamed deltas assemble into one validated reply."""

    def test_deltas_are_concatenated_in_order(self) -> None:
        reply = collect_stream(
            iter(stream_chunks(['{"context": ', '"note"}', ""])),
            started=time.monotonic(),
        )
        assert reply.text == '{"context": "note"}'
        assert reply.model == "fake-model-1"
        assert reply.response_id == "chatcmpl-1"

    def test_duration_includes_the_time_spent_receiving_the_stream(self) -> None:
        chunks = stream_chunks(["first", "second"])

        with freeze_time("2026-01-01T00:00:00+00:00") as frozen:

            def delayed() -> Iterator[ChatCompletionChunk]:
                for chunk in chunks:
                    frozen.tick(0.5)
                    yield chunk

            reply = collect_stream(delayed(), started=time.monotonic())

        assert reply.duration_seconds == pytest.approx(0.5 * len(chunks))

    @pytest.mark.parametrize(
        ("usage", "expected"), [((120, 30), (120, 30)), (None, (None, None))]
    )
    def test_reported_usage_is_kept_or_left_null(
        self, usage: tuple[int, int] | None, expected: tuple[int | None, int | None]
    ) -> None:
        reply = collect_stream(
            iter(stream_chunks(["text"], usage=usage)), started=time.monotonic()
        )
        assert (reply.input_tokens, reply.output_tokens) == expected

    @pytest.mark.parametrize(
        ("pieces", "finish_reason", "code"),
        [
            pytest.param(
                ["partial"], "length", "truncated_provider_response", id="length"
            ),
            pytest.param([], "stop", "empty_provider_response", id="empty"),
            pytest.param(["  ", "\n"], "stop", "empty_provider_response", id="blank"),
            pytest.param(
                ["x" * (MAX_CONTEXT_RESPONSE_BYTES // 2)] * 3,
                "stop",
                "oversized_provider_response",
                id="oversized",
            ),
            pytest.param(
                ["partial"],
                "content_filter",
                "filtered_provider_response",
                id="filtered",
            ),
        ],
    )
    def test_unusable_streams_are_rejected(
        self, pieces: list[str], finish_reason: FinishReason, code: str
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            collect_stream(
                iter(stream_chunks(pieces, finish_reason=finish_reason)),
                started=time.monotonic(),
            )
        assert caught.value.code == code

    def test_tool_call_delta_is_rejected(self) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            collect_stream(iter((tool_call_chunk(),)), started=time.monotonic())
        assert caught.value.code == "unsupported_provider_response"

    def test_refusal_delta_is_rejected(self) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            collect_stream(iter((refusal_chunk(),)), started=time.monotonic())
        assert caught.value.code == "unsupported_provider_response"

    def test_review_bound_accepts_a_larger_report_than_the_context_bound(
        self,
    ) -> None:
        report = '{"summary": "' + "x" * MAX_CONTEXT_RESPONSE_BYTES + '"}'
        reply = collect_stream(
            iter(stream_chunks([report])),
            started=time.monotonic(),
            max_bytes=MAX_REVIEW_RESPONSE_BYTES,
        )
        assert len(reply.text.encode("utf-8")) > MAX_CONTEXT_RESPONSE_BYTES

    def test_review_bound_is_still_enforced(self) -> None:
        oversized = "x" * (MAX_REVIEW_RESPONSE_BYTES + 1)
        with pytest.raises(AtlasReviewError) as caught:
            collect_stream(
                iter(stream_chunks([oversized])),
                started=time.monotonic(),
                max_bytes=MAX_REVIEW_RESPONSE_BYTES,
            )
        assert caught.value.code == "oversized_provider_response"


class TestStreamedRequest:
    """Every request is a streamed completion with the configured token limit."""

    @pytest.mark.parametrize("parameter", ["max_tokens", "max_completion_tokens"])
    def test_request_is_streamed_with_the_configured_token_parameter(
        self, parameter: str
    ) -> None:
        config = LLMConfig.model_validate(
            {
                **CONFIG.model_dump(mode="json"),
                "output_token_parameter": parameter,
            },
            strict=True,
        )
        client, completions = contextualizer(config, stream_chunks(["ok"]))
        client("system", "user", output_tokens=64, temperature=0.2)
        call = completions.calls[0]
        assert call["stream"] is True
        assert call["stream_options"] == {"include_usage": True}
        assert call[parameter] == 64
        assert call["temperature"] == 0.2
        assert call["reasoning_effort"] == "medium"
        assert call["model"] == "fake-model-1"
        assert "tools" not in call
        assert call["response_format"] is omit

    def test_null_temperature_is_omitted_from_the_request(self) -> None:
        client, completions = contextualizer(CONFIG, stream_chunks(["ok"]))
        client("system", "user", output_tokens=32, temperature=None)
        assert completions.calls[0]["temperature"] is omit

    @pytest.mark.parametrize(
        "effort",
        [
            pytest.param(None, id="provider-default"),
            pytest.param("medium", id="medium"),
        ],
    )
    def test_reasoning_effort_is_sent_only_when_configured(
        self, effort: str | None
    ) -> None:
        config = provider(reasoning_effort=effort)
        client, completions = contextualizer(config, stream_chunks(["ok"]))
        client("system", "user", output_tokens=32, temperature=None)
        sent = completions.calls[0]["reasoning_effort"]
        if effort is None:
            assert sent is omit
        else:
            assert sent == effort

    def test_assembled_stream_becomes_the_reply(self) -> None:
        client, _ = contextualizer(CONFIG, stream_chunks(["a", "b"], usage=(9, 2)))
        reply = client("system", "user", output_tokens=32, temperature=None)
        assert reply.text == "ab"
        assert reply.input_tokens == 9
        assert reply.output_tokens == 2

    def test_each_call_opens_its_own_stream(self) -> None:
        client, completions = contextualizer(CONFIG, stream_chunks(["once"]))
        client("system", "user", output_tokens=8, temperature=None)
        client("system", "user", output_tokens=8, temperature=None)
        assert len(completions.calls) == 2


class TestReviewRequestOptions:
    """Review synthesis sends the configured output budget and JSON mode."""

    @pytest.mark.parametrize(
        ("request_json_mode", "mode", "sent"),
        [
            pytest.param(True, "json_object", True, id="review-json-mode"),
            pytest.param(True, "text", False, id="review-text-mode"),
            pytest.param(False, "json_object", False, id="contextualization-json-mode"),
            pytest.param(False, "text", False, id="contextualization-text-mode"),
        ],
    )
    def test_json_mode_is_sent_only_for_review_synthesis(
        self, request_json_mode: bool, mode: str, sent: bool
    ) -> None:
        config = provider(response_format_mode=mode)
        client, completions = adapter(
            config,
            stream_chunks(["ok"]),
            max_response_bytes=MAX_REVIEW_RESPONSE_BYTES,
            request_json_mode=request_json_mode,
        )
        client("system", "user", output_tokens=64, temperature=None)
        call = completions.calls[0]
        if sent:
            assert call["response_format"] == {"type": "json_object"}
        else:
            assert call["response_format"] is omit

    def test_contextualization_bound_rejects_an_oversized_reply(self) -> None:
        oversized = ["x" * (MAX_CONTEXT_RESPONSE_BYTES + 1)]
        client, _ = contextualizer(CONFIG, stream_chunks(oversized))
        with pytest.raises(AtlasReviewError) as caught:
            client("system", "user", output_tokens=64, temperature=None)
        assert caught.value.code == "oversized_provider_response"

    def test_built_reviewer_uses_the_report_identity(self) -> None:
        config = provider(api_key_mode="none", api_key_env=None)
        reviewer = build_reviewer(config)
        try:
            assert reviewer.max_response_bytes == MAX_REVIEW_RESPONSE_BYTES
            assert reviewer.request_json_mode is True
        finally:
            reviewer.close()

    def test_built_contextualizer_keeps_the_context_identity(self) -> None:
        config = provider(api_key_mode="none", api_key_env=None)
        contextualizer_adapter = build_contextualizer(config)
        try:
            assert contextualizer_adapter.max_response_bytes == (
                MAX_CONTEXT_RESPONSE_BYTES
            )
            assert contextualizer_adapter.request_json_mode is False
        finally:
            contextualizer_adapter.close()


class ProviderStatusError(OpenAIError):
    """Stand-in for a provider failure that carries an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        """Record the status code the SDK would report."""
        super().__init__(f"provider returned HTTP {status_code}")
        self.status_code = status_code


class TestTransientRetries:
    """Only transient provider failures are repeated, and only within the limit."""

    @pytest.mark.parametrize("status_code", [429, 500, 503])
    def test_a_retryable_failure_is_retried_before_succeeding(
        self, status_code: int
    ) -> None:
        retries = Retries(limit=1)
        client, completions = adapter(
            CONFIG,
            stream_chunks(["a supported contextual note"]),
            errors=(ProviderStatusError(status_code),),
            retries=retries,
        )
        reply = client("system", "user", output_tokens=64, temperature=None)
        assert reply.text == "a supported contextual note"
        assert len(completions.calls) == 2
        assert retries.sleeps == [RETRY_BASE_DELAY_SECONDS]
        assert retries.notices == [
            f"provider ProviderStatusError from {CONFIG.model}; retry 1/1 in 2s"
        ]

    @pytest.mark.parametrize("status_code", [400, 401, 404, 422])
    def test_a_rejected_request_is_not_retried(self, status_code: int) -> None:
        retries = Retries(limit=2)
        client, completions = adapter(
            CONFIG,
            stream_chunks(["unused"]),
            errors=(ProviderStatusError(status_code),),
            retries=retries,
        )
        with pytest.raises(AtlasReviewError) as caught:
            client("system", "user", output_tokens=64, temperature=None)
        assert caught.value.code == "provider_request_failed"
        assert len(completions.calls) == 1
        assert retries.notices == []

    def test_the_retry_limit_bounds_the_repeated_requests(self) -> None:
        retries = Retries(limit=2)
        failures = tuple(ProviderStatusError(500) for _ in range(4))
        client, completions = adapter(
            CONFIG,
            stream_chunks(["unused"]),
            errors=failures,
            retries=retries,
        )
        with pytest.raises(AtlasReviewError) as caught:
            client("system", "user", output_tokens=64, temperature=None)
        assert caught.value.code == "provider_request_failed"
        assert len(completions.calls) == 3
        assert retries.sleeps == [
            RETRY_BASE_DELAY_SECONDS,
            RETRY_BASE_DELAY_SECONDS * 2,
        ]

    def test_zero_retries_sends_exactly_one_request(self) -> None:
        client, completions = adapter(
            CONFIG,
            stream_chunks(["unused"]),
            errors=(ProviderStatusError(503),),
            retries=Retries(limit=0),
        )
        with pytest.raises(AtlasReviewError):
            client("system", "user", output_tokens=64, temperature=None)
        assert len(completions.calls) == 1

    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [
            pytest.param(429, True, id="rate-limit"),
            pytest.param(500, True, id="server-error"),
            pytest.param(503, True, id="unavailable"),
            pytest.param(400, False, id="bad-request"),
            pytest.param(401, False, id="unauthenticated"),
            pytest.param(404, False, id="not-found"),
        ],
    )
    def test_transient_classification_follows_the_status_code(
        self, status_code: int, expected: bool
    ) -> None:
        assert is_transient_provider_error(ProviderStatusError(status_code)) is (
            expected
        )

    def test_built_adapters_carry_their_retry_counts(self) -> None:
        notices: list[str] = []
        config = provider(api_key_mode="none", api_key_env=None)
        reviewer = build_reviewer(config)
        context = build_contextualizer(
            config, automatic_retries=4, notify=notices.append
        )
        try:
            assert reviewer.automatic_retries == config.automatic_retries
            assert context.automatic_retries == 4
            assert context.notify == notices.append
        finally:
            reviewer.close()
            context.close()
