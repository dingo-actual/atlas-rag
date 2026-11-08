"""Synthetic evidence packets shared by the review-inference tests.

The real retrieval fixture produces a packet whose ranking decides which
evidence identifiers exist. These builders fix that layout so prompt budgets
and report validation can be checked against known identifiers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from atlas_review.contracts import (
    AllowedAssociation,
    ContextualizationResult,
    EvidenceChannel,
    EvidenceItem,
    EvidencePacket,
    OwnerKind,
    Passage,
    RetrievalTrace,
    ReviewReport,
)

CASE = "AML.CS0035"
TECHNIQUE = "AML.T0051.001"
PARENT = "AML.T0051"
MITIGATION = "AML.M0033"
COLLECTION = "2026.09"
EXERCISE = "Exercise"
CREATED_AT = "2026-11-07T12:00:00+00:00"

KINDS: dict[str, OwnerKind] = {
    CASE: "case_study",
    TECHNIQUE: "technique",
    PARENT: "technique",
    MITIGATION: "mitigation",
}


def digest(label: str) -> str:
    """Return the stable 64-character identifier for one synthetic edge."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sample_trace(generation_id: str) -> RetrievalTrace:
    """Return a stored retrieval trace for the review fixtures."""
    return RetrievalTrace(
        schema_version=1,
        generation_id=generation_id,
        collection_version=COLLECTION,
        settings_fingerprint="a" * 64,
        started_at=CREATED_AT,
        focus="indirect prompt injection",
        query_tokens=3,
    )


def passage(
    evidence_id: str,
    *,
    entities: Sequence[str],
    passage_id: int,
    text: str,
    case_type: str | None = None,
    channels: Sequence[EvidenceChannel] = ("dense",),
) -> EvidenceItem:
    """Build one original-source passage item."""
    return EvidenceItem(
        evidence_id=evidence_id,
        item_kind="passage",
        entity_ids=list(entities),
        passage_id=passage_id,
        text=text,
        source_locator=f"/entities/{entities[0]}/section",
        collection_version=COLLECTION,
        case_type=case_type,
        retrieval_channels=list(channels),
    )


def card(
    evidence_id: str,
    *,
    entities: Sequence[str],
    label: str,
    text: str,
    case_type: str | None = None,
) -> EvidenceItem:
    """Build one structural relationship card."""
    return EvidenceItem(
        evidence_id=evidence_id,
        item_kind="relationship",
        entity_ids=list(entities),
        edge_id=digest(label),
        text=text,
        source_locator=f"/relationships/{entities[0]}/{label}",
        collection_version=COLLECTION,
        case_type=case_type,
        retrieval_channels=["graph_connector"],
    )


def packet(
    items: Sequence[EvidenceItem],
    associations: Sequence[AllowedAssociation] = (),
    kinds: Mapping[str, OwnerKind] | None = None,
) -> EvidencePacket:
    """Validate items and derive the entity allowlist from them."""
    known = KINDS if kinds is None else kinds
    named = {entity for item in items for entity in item.entity_ids}
    allowlist: dict[OwnerKind, list[str]] = {}
    for entity_id in sorted(named):
        kind = known.get(entity_id)
        if kind is not None:
            allowlist.setdefault(kind, []).append(entity_id)
    return EvidencePacket(
        schema_version=1,
        generation_id="generation-1",
        collection_version=COLLECTION,
        items=list(items),
        allowed_entity_ids_by_kind=allowlist,
        allowed_associations=list(associations),
    )


def case_association(*, case_evidence: Sequence[str]) -> AllowedAssociation:
    """Return the direct case-to-technique `employs` association."""
    return AllowedAssociation(
        source_id=CASE,
        source_kind="case_study",
        technique_id=TECHNIQUE,
        edge_ids=[digest("employs")],
        evidence_ids=list(case_evidence),
    )


def mitigation_association(*, mitigation_evidence: Sequence[str]) -> AllowedAssociation:
    """Return the parent-linked mitigation-to-technique association."""
    return AllowedAssociation(
        source_id=MITIGATION,
        source_kind="mitigation",
        technique_id=TECHNIQUE,
        parent_technique_id=PARENT,
        edge_ids=[digest("mitigates"), digest("specializes")],
        evidence_ids=list(mitigation_evidence),
    )


def default_items() -> list[EvidenceItem]:
    """Return the fixture packet's evidence items in rank order."""
    return [
        passage(
            "E001",
            entities=[TECHNIQUE],
            passage_id=1,
            text="Indirect prompt injection places instructions in retrieved data.",
        ),
        card(
            "E002",
            entities=[CASE, TECHNIQUE],
            label="employs",
            text="Data Exfiltration from Slack AI employs Indirect | step S01",
            case_type=EXERCISE,
        ),
        passage(
            "E003",
            entities=[MITIGATION],
            passage_id=2,
            text="Input Validation: validate untrusted text before it reaches a model.",
        ),
        card(
            "E004",
            entities=[MITIGATION, PARENT],
            label="mitigates",
            text="Input Validation mitigates LLM Prompt Injection",
        ),
        card(
            "E005",
            entities=[TECHNIQUE, PARENT],
            label="specializes",
            text="Indirect specializes LLM Prompt Injection",
        ),
        passage(
            "E006",
            entities=[CASE],
            passage_id=3,
            text="Researchers demonstrated exfiltration through indirect injection.",
            case_type=EXERCISE,
        ),
        passage(
            "E007",
            entities=[PARENT],
            passage_id=4,
            text="LLM Prompt Injection covers direct and indirect instruction abuse.",
        ),
    ]


def default_associations() -> list[AllowedAssociation]:
    """Return both associations supported by the default packet."""
    return [
        case_association(case_evidence=["E002", "E006"]),
        mitigation_association(mitigation_evidence=["E003", "E004", "E005"]),
    ]


def default_packet() -> EvidencePacket:
    """Return the default packet with its direct and parent-linked links."""
    return packet(default_items(), default_associations())


def padded_packet(count: int) -> EvidencePacket:
    """Return a packet with extra low-ranked technique passages for trimming."""
    items = default_items()
    items.extend(
        passage(
            f"E{index:03d}",
            entities=[TECHNIQUE],
            passage_id=index,
            text="Additional technique detail " * 12,
        )
        for index in range(8, 8 + count)
    )
    return packet(items, default_associations())


def generated_passage() -> Passage:
    """Return one stored passage with its generated retrieval note."""
    return Passage(
        schema_version=1,
        passage_id=1,
        generation_id="generation-1",
        entity_id=TECHNIQUE,
        owner_kind="technique",
        section="description",
        ordinal=0,
        bm25_ordinal=0,
        header=f"Technique {TECHNIQUE} | LLM Prompt Injection / Indirect",
        source_text="Indirect prompt injection places instructions in retrieved data.",
        context_text=(
            "This excerpt introduces the indirect variant of prompt injection."
        ),
        retrieval_text="Technique header\nContext (generated): note\nSource excerpt: text",
        context_cache_key=digest("context"),
        source_sha256=digest("source"),
        content_sha256=digest("content"),
        embedding_tokens=12,
        reranker_tokens=14,
        source_locator=f"/techniques/{TECHNIQUE}/description",
        source_start=0,
        source_end=60,
    )


def context_entry() -> ContextualizationResult:
    """Return the accepted cache entry that produced the generated note."""
    return ContextualizationResult(
        schema_version=1,
        cache_key=digest("context"),
        prompt_version="chunk-context-v1",
        model="fake-model-1",
        response_model="fake-model-1",
        response_id="chatcmpl-1",
        context_text=(
            "This excerpt introduces the indirect variant of prompt injection."
        ),
        support_locators=[f"/techniques/{TECHNIQUE}/description"],
        embedding_tokens=12,
        reranker_tokens=14,
        input_tokens=900,
        output_tokens=40,
        duration_seconds=0.5,
        request_fingerprint=digest("request"),
        created_at=CREATED_AT,
    )


CONTROL: dict[str, Any] = {
    "action": "Validate untrusted text before it reaches the model.",
    "rationale": "The mitigation is linked to the technique in this packet.",
    "verification": "Identify the component that validates retrieved text.",
    "mitigation_ids": [MITIGATION],
    "evidence_ids": ["E003", "E004"],
}

FINDING: dict[str, Any] = {
    "finding_id": "F001",
    "title": "Untrusted retrieved instructions may cross the trust boundary",
    "technique_ids": [TECHNIQUE],
    "priority": "review_next",
    "priority_rationale": "The design combines untrusted content with internal data.",
    "applicability": "Applies if retrieved text can alter the assistant's behaviour.",
    "input_facts": ["The assistant retrieves workspace messages."],
    "assumptions_or_unknowns": ["The validation boundary is unspecified."],
    "atlas_basis": "ATLAS describes indirect prompt injection and a Slack AI exercise.",
    "case_study_ids": [CASE],
    "evidence_ids": ["E001", "E002", "E006"],
    "controls": [CONTROL],
    "review_questions": ["Can retrieved content choose an output destination?"],
}


def finding(**overrides: Any) -> dict[str, Any]:
    """Return one finding mapping with explicit overrides."""
    return {**FINDING, **overrides}


def valid_report(*findings: dict[str, Any], **overrides: Any) -> ReviewReport:
    """Return a validated report holding the supplied findings."""
    body: dict[str, Any] = {
        "schema_version": 1,
        "summary": "Review untrusted retrieved content and its output handling.",
        "findings": list(findings) or [FINDING],
        "unanswered_questions": ["Which actions need human approval?"],
        "evidence_gaps": [],
        **overrides,
    }
    return ReviewReport.model_validate(body, strict=True)


def second_finding() -> dict[str, Any]:
    """Return an independent finding with its own identifier."""
    return finding(
        finding_id="F002",
        technique_ids=[PARENT],
        case_study_ids=[],
        evidence_ids=["E007"],
        controls=[],
    )
