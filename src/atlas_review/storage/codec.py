"""MessagePack storage adapter for validated Pydantic payloads.

The codec stays a plain-value binary encoder. It never receives a model class as
a decoder type, so Pydantic remains the only validation authority.
"""

from __future__ import annotations

import msgspec
from pydantic import ValidationError

from atlas_review.contracts import AppModel
from atlas_review.errors import STAGE_STORAGE, AtlasReviewError

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder()


def encode_model(value: AppModel) -> bytes:
    """Validate a model, then encode its plain JSON-shaped values.

    Args:
        value (AppModel): Validated or draft Pydantic payload.

    Returns:
        (bytes) Versioned named-field MessagePack bytes.

    Raises:
        AtlasReviewError: The payload fails strict revalidation.
    """
    try:
        validated = type(value).model_validate(value, strict=True)
    except ValidationError as exc:
        raise _invalid_payload(exc) from exc
    plain = validated.model_dump(
        mode="json",
        by_alias=False,
        exclude_none=False,
        exclude_defaults=False,
        exclude_unset=False,
    )
    return _encoder.encode(plain)


def decode_model[ModelT: AppModel](payload: bytes, model_type: type[ModelT]) -> ModelT:
    """Decode MessagePack values and revalidate them as a model.

    Args:
        payload (bytes): Stored MessagePack bytes.
        model_type (type[ModelT]): Trusted model chosen by the repository
            operation, never by stored data.

    Returns:
        (ModelT) Revalidated model.

    Raises:
        AtlasReviewError: The bytes are not MessagePack, or the decoded values
            fail validation.
    """
    try:
        plain = _decoder.decode(payload)
    except msgspec.DecodeError as exc:
        raise AtlasReviewError(
            code="undecodable_payload",
            stage=STAGE_STORAGE,
            message="stored payload is not valid MessagePack",
            recovery="Rebuild the workspace from a known-good source file.",
        ) from exc
    try:
        return model_type.model_validate(plain, strict=True)
    except ValidationError as exc:
        raise _invalid_payload(exc) from exc


def encode_plain(value: object) -> bytes:
    """Encode a plain storage value as MessagePack.

    Args:
        value (object): JSON-compatible value such as a manifest mapping.

    Returns:
        (bytes) Encoded bytes.
    """
    return _encoder.encode(value)


def decode_plain(payload: bytes) -> object:
    """Decode MessagePack bytes to plain values.

    Args:
        payload (bytes): Stored MessagePack bytes.

    Returns:
        (object) Decoded plain value.

    Raises:
        AtlasReviewError: The bytes are not MessagePack.
    """
    try:
        return _decoder.decode(payload)
    except msgspec.DecodeError as exc:
        raise AtlasReviewError(
            code="undecodable_payload",
            stage=STAGE_STORAGE,
            message="stored payload is not valid MessagePack",
            recovery="Rebuild the workspace from a known-good source file.",
        ) from exc


def _invalid_payload(exc: ValidationError) -> AtlasReviewError:
    """Map a validation failure to a sanitized storage error."""
    details = tuple(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(
            include_input=False, include_context=False, include_url=False
        )
    )
    return AtlasReviewError(
        code="invalid_stored_payload",
        stage=STAGE_STORAGE,
        message="stored payload failed validation",
        details=details,
        recovery="Rebuild the workspace from a known-good source file.",
    )
