"""Inference adapter behaviour: credentials and streamed response assembly."""

from __future__ import annotations

from typing import cast

import pytest
from openai import OpenAI, omit
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
    MAX_RESPONSE_BYTES,
    PLACEHOLDER_API_KEY,
    ChatContextualizer,
    MissingProviderCredentialError,
    build_contextualizer,
    collect_stream,
    resolve_api_key,
)
from fakes import FakeChatClient, FakeCompletions, FinishReason, stream_chunks

CONFIG = LLMConfig(base_url="https://provider.example/v1", model="fake-model-1")


def contextualizer(
    config: LLMConfig, chunks: tuple[ChatCompletionChunk, ...]
) -> tuple[ChatContextualizer, FakeCompletions]:
    """Return a contextualizer bound to a recording fake client."""
    completions = FakeCompletions(chunks=chunks)
    client = cast("OpenAI", FakeChatClient(completions=completions))
    return ChatContextualizer(config, client), completions


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
            iter(stream_chunks(['{"context": ', '"note"}', ""])), duration=0.5
        )
        assert reply.text == '{"context": "note"}'
        assert reply.duration_seconds == 0.5
        assert reply.model == "fake-model-1"
        assert reply.response_id == "chatcmpl-1"

    @pytest.mark.parametrize(
        ("usage", "expected"), [((120, 30), (120, 30)), (None, (None, None))]
    )
    def test_reported_usage_is_kept_or_left_null(
        self, usage: tuple[int, int] | None, expected: tuple[int | None, int | None]
    ) -> None:
        reply = collect_stream(iter(stream_chunks(["text"], usage=usage)), duration=0.1)
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
                ["x" * (MAX_RESPONSE_BYTES // 2)] * 3,
                "stop",
                "oversized_provider_response",
                id="oversized",
            ),
        ],
    )
    def test_unusable_streams_are_rejected(
        self, pieces: list[str], finish_reason: FinishReason, code: str
    ) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            collect_stream(
                iter(stream_chunks(pieces, finish_reason=finish_reason)), duration=0.1
            )
        assert caught.value.code == code

    def test_tool_call_delta_is_rejected(self) -> None:
        with pytest.raises(AtlasReviewError) as caught:
            collect_stream(iter((tool_call_chunk(),)), duration=0.1)
        assert caught.value.code == "unsupported_provider_response"


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
        assert call["model"] == "fake-model-1"
        assert "tools" not in call
        assert "response_format" not in call

    def test_null_temperature_is_omitted_from_the_request(self) -> None:
        client, completions = contextualizer(CONFIG, stream_chunks(["ok"]))
        client("system", "user", output_tokens=32, temperature=None)
        assert completions.calls[0]["temperature"] is omit

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
