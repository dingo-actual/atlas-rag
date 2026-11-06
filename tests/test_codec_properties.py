"""Round-trip properties for the MessagePack payload codec."""

from __future__ import annotations

from typing import Literal

from hypothesis import given
from hypothesis import strategies as st

from atlas_review.contracts import AtlasEntity, AtlasRelationship
from atlas_review.storage.codec import decode_model, encode_model

_RelationName = Literal["employs", "mitigates", "specializes", "achieves", "sequences"]
_EntityKind = Literal[
    "collection", "matrix", "tactic", "technique", "mitigation", "case_study"
]

_HEX: st.SearchStrategy[str] = st.text(
    alphabet="0123456789abcdef", min_size=64, max_size=64
)
_TEXT = st.text(min_size=1)
_OPTIONAL_TEXT = st.none() | st.text(min_size=1)
_LOCATOR = st.builds(lambda body: "/" + body, st.text(min_size=0, max_size=40))
_RELATION: st.SearchStrategy[_RelationName] = st.sampled_from(
    ["employs", "mitigates", "specializes", "achieves", "sequences"]
)
_KIND: st.SearchStrategy[_EntityKind] = st.sampled_from(
    ["collection", "matrix", "tactic", "technique", "mitigation", "case_study"]
)


@st.composite
def relationships(draw: st.DrawFn) -> AtlasRelationship:
    """Build a valid relationship record with arbitrary text fields."""
    return AtlasRelationship(
        generation_id=draw(_TEXT),
        edge_id=draw(_HEX),
        relation=draw(_RELATION),
        source_id=draw(_TEXT),
        target_id=draw(_TEXT),
        description=draw(_OPTIONAL_TEXT),
        step_id=draw(_OPTIONAL_TEXT),
        tactic_id=draw(_OPTIONAL_TEXT),
        position=draw(st.none() | st.integers(min_value=0, max_value=10_000)),
        leads_to=draw(st.lists(_TEXT, max_size=5)),
        source_locator=draw(_LOCATOR),
    )


@st.composite
def entities(draw: st.DrawFn) -> AtlasEntity:
    """Build a valid entity record with arbitrary text fields."""
    return AtlasEntity(
        generation_id=draw(_TEXT),
        entity_id=draw(_TEXT),
        kind=draw(_KIND),
        name=draw(_TEXT),
        display_name=draw(_TEXT),
        description=draw(_TEXT),
        source_locator=draw(_LOCATOR),
        uuid=draw(_OPTIONAL_TEXT),
        maturity=draw(_OPTIONAL_TEXT),
        platforms=draw(st.lists(_TEXT, max_size=3)),
        lifecycle_phases=draw(st.lists(_TEXT, max_size=3)),
        categories=draw(st.lists(_TEXT, max_size=3)),
    )


class TestPayloadRoundTrip:
    """Encoding then decoding restores an equal validated model."""

    @given(relationships())
    def test_relationship_survives_encode_decode(
        self, value: AtlasRelationship
    ) -> None:
        assert decode_model(encode_model(value), AtlasRelationship) == value

    @given(entities())
    def test_entity_survives_encode_decode(self, value: AtlasEntity) -> None:
        assert decode_model(encode_model(value), AtlasEntity) == value
