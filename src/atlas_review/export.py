"""Markdown export of a saved review.

The export is self-contained: brief, report, citation index, and the evidence
appendix the model actually saw. Generated chunk context is never included.
"""

from __future__ import annotations

import re
from pathlib import Path

from atlas_review.citations import atlas_routes, case_label, mitigation_labels
from atlas_review.contracts import (
    ControlRecommendation,
    EvidenceItem,
    EvidencePacket,
    ReviewFinding,
    ReviewReport,
)
from atlas_review.errors import STAGE_EXPORT, AtlasReviewError
from atlas_review.sanitize import sanitize_text
from atlas_review.storage.reviews import StoredReview

DRAFT_NOTICE = "Draft review based on the supplied application brief."
ATTRIBUTION = (
    "ATLAS material: MITRE ATLAS data repository, Apache License 2.0, "
    "collection {version}."
)
_SLUG = re.compile(r"[^a-z0-9]+")
_BRIEF_FIELDS = (
    ("title", "Title"),
    ("review_question", "Review question"),
    ("retrieval_focus", "Retrieval focus"),
    ("system_description", "System description"),
    ("users_and_roles", "Users and roles"),
    ("data_sources", "Data sources"),
    ("untrusted_inputs", "Untrusted inputs"),
    ("tools_and_permissions", "Tools and permissions"),
    ("sensitive_assets", "Sensitive assets"),
    ("existing_controls", "Existing controls"),
    ("constraints", "Constraints"),
    ("unknowns", "Unknowns"),
)


def render_markdown(stored: StoredReview) -> str:
    """Render one saved review as a self-contained Markdown document.

    Args:
        stored (StoredReview): Saved review with its frozen evidence.

    Returns:
        (str) Markdown report, or a notice when no report is accepted yet.
    """
    if stored.report is None:
        return _lines(
            [
                f"# {_inline(stored.title)}",
                "",
                f"{DRAFT_NOTICE} No validated report is stored for this review yet.",
                "",
                f"- Review: {_inline(stored.review_id)}",
                f"- State: {stored.state}",
                f"- Collection: {_inline(stored.evidence.collection_version)}",
            ]
        )
    return _lines(
        [
            *_header(stored, stored.report),
            *_finding_sections(stored, stored.report),
            *_question_sections(stored.report),
            *_citation_index(stored.evidence),
            *_brief_section(stored),
            *_evidence_appendix(stored.evidence),
        ]
    )


def default_filename(stored: StoredReview) -> str:
    """Return a filename derived from the brief title and review identifier.

    Args:
        stored (StoredReview): Saved review.

    Returns:
        (str) Filesystem-safe Markdown filename.
    """
    slug = _SLUG.sub("-", sanitize_text(stored.title).lower()).strip("-")
    return f"{slug or 'review'}-{stored.review_id}.md"


def export_markdown(
    stored: StoredReview, directory: Path, *, overwrite: bool = False
) -> Path:
    """Write the report to one Markdown file and return its path.

    Args:
        stored (StoredReview): Saved review to export.
        directory (Path): Destination directory.
        overwrite (bool): Replace an existing file instead of choosing a new name.

    Returns:
        (Path) Written file.

    Raises:
        AtlasReviewError: The destination is unusable or has no free name.
    """
    temporary: Path | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = _free_path(directory, default_filename(stored), overwrite=overwrite)
        temporary = directory / f".{target.name}.tmp"
        temporary.write_text(render_markdown(stored), encoding="utf-8")
        temporary.replace(target)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise AtlasReviewError(
            code="export_failed",
            stage=STAGE_EXPORT,
            message=f"cannot write this report in {directory}",
            recovery="Choose a writable destination and export again.",
        ) from exc
    return target


def _free_path(directory: Path, name: str, *, overwrite: bool) -> Path:
    """Return a target path, numbering it when the name is already taken."""
    candidate = directory / name
    if overwrite or not candidate.exists():
        return candidate
    stem = candidate.stem
    for suffix in range(2, 100):
        alternative = directory / f"{stem}-{suffix}.md"
        if not alternative.exists():
            return alternative
    raise AtlasReviewError(
        code="export_failed",
        stage=STAGE_EXPORT,
        message=f"cannot find a free export name for {name}",
        recovery="Remove old exports from the directory or choose another one.",
    )


def _header(stored: StoredReview, report: ReviewReport) -> list[str]:
    """Render the report header with trusted application metadata."""
    provider = stored.config.llm
    return [
        f"# {_inline(stored.title)}",
        "",
        f"*{DRAFT_NOTICE}*",
        "",
        f"- Review: {_inline(stored.review_id)}",
        f"- Created: {_inline(stored.created_at)}",
        f"- Collection: {_inline(stored.evidence.collection_version)}",
        f"- Generation: {_inline(stored.generation_id)}",
        (
            f"- Retrieval-time provider: {_inline(provider.provider_name)} "
            f"({_inline(provider.model)})"
        ),
        f"- Findings: {len(report.findings)}",
        "",
        "## Summary",
        "",
        _inline(report.summary),
        "",
    ]


def _finding_sections(stored: StoredReview, report: ReviewReport) -> list[str]:
    """Render every finding in the order the model returned it."""
    lines: list[str] = ["## Findings", ""]
    if not report.findings:
        lines.extend(
            [
                (
                    "No findings: the supplied evidence did not support a "
                    "review concern. The questions below describe what to "
                    "clarify."
                ),
                "",
            ]
        )
        return lines
    for finding in report.findings:
        lines.extend(_finding(stored, finding))
    return lines


def _finding(stored: StoredReview, finding: ReviewFinding) -> list[str]:
    """Render one finding with its facts, cases, controls, and citations."""
    return [
        f"### {_inline(finding.finding_id)}: {_inline(finding.title)}",
        "",
        f"- Priority: {finding.priority}",
        f"- Priority rationale: {_inline(finding.priority_rationale)}",
        f"- Techniques: {_inline(', '.join(finding.technique_ids))}",
        f"- ATLAS basis: {_inline(finding.atlas_basis)}",
        "",
        f"**Applicability.** {_inline(finding.applicability)}",
        "",
        *_listed("Source facts from the brief", finding.input_facts),
        *_listed("Assumptions and unknowns", finding.assumptions_or_unknowns),
        *_cases(finding, stored.evidence),
        *_controls(finding, stored),
        *_listed("Review questions", finding.review_questions),
        f"**Evidence.** {_inline(', '.join(finding.evidence_ids))}",
        "",
    ]


def _cases(finding: ReviewFinding, packet: EvidencePacket) -> list[str]:
    """Render the case studies with the classification stored in the packet."""
    if not finding.case_study_ids:
        return []
    labels = [case_label(case_id, packet.items) for case_id in finding.case_study_ids]
    return [f"**Case studies.** {_inline('; '.join(labels))}", ""]


def _controls(finding: ReviewFinding, stored: StoredReview) -> list[str]:
    """Render controls with direct or parent-linked ATLAS grounding."""
    if not finding.controls:
        return []
    lines = [
        "**Controls**",
        "",
        (
            "*An ATLAS mitigation association is not evidence that the "
            "mitigation is present or effective in this application.*"
        ),
        "",
    ]
    for position, control in enumerate(finding.controls, start=1):
        lines.extend(
            [
                f"{position}. {_inline(control.action)}",
                f"   - Rationale: {_inline(control.rationale)}",
                f"   - Verification: {_inline(control.verification)}",
                f"   - ATLAS mitigations: {_inline(_mitigation_labels(control, finding, stored))}",
                f"   - Evidence: {_inline(', '.join(control.evidence_ids))}",
            ]
        )
    return [*lines, ""]


def _mitigation_labels(
    control: ControlRecommendation, finding: ReviewFinding, stored: StoredReview
) -> str:
    """Label each named mitigation as direct, parent-linked, or suggested."""
    return mitigation_labels(
        control.mitigation_ids,
        finding.technique_ids,
        stored.evidence.allowed_associations,
    )


def _question_sections(report: ReviewReport) -> list[str]:
    """Render the global questions and coverage gaps."""
    return [
        *_listed("Unanswered questions", report.unanswered_questions),
        *_listed("Evidence gaps", report.evidence_gaps),
    ]


def _citation_index(packet: EvidencePacket) -> list[str]:
    """Render the citation index with retained locators and ATLAS pages."""
    routes = atlas_routes(packet)
    lines = ["## Citations", ""]
    for item in packet.items:
        case = f", {_inline(item.case_type)}" if item.case_type else ""
        line = (
            f"- {_inline(item.evidence_id)} — "
            f"{_inline(', '.join(item.entity_ids))}{case} "
            f"({item.item_kind}, {_inline(item.source_locator)})"
        )
        links = [routes[entity] for entity in item.entity_ids if entity in routes]
        if links:
            line = f"{line}. ATLAS: {', '.join(links)}"
        lines.append(line)
    return [*lines, ""]


def _brief_section(stored: StoredReview) -> list[str]:
    """Render the application brief that produced the review."""
    brief = stored.request.model_dump(mode="json")
    lines = ["## Application brief", "", "| Field | Value |", "|---|---|"]
    for field, label in _BRIEF_FIELDS:
        lines.append(f"| {label} | {_cell(_joined(brief.get(field)))} |")
    return [*lines, ""]


def _evidence_appendix(packet: EvidencePacket) -> list[str]:
    """Render the original-source excerpts the model saw."""
    lines = [
        "## Evidence appendix",
        "",
        ATTRIBUTION.format(version=_inline(packet.collection_version)),
        (
            "Generated retrieval context is excluded; every excerpt below is "
            "stored ATLAS source text."
        ),
        "",
        "| ID | Kind | Entities | Case type | Source locator | Excerpt |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_appendix_row(item) for item in packet.items)
    return [*lines, ""]


def _appendix_row(item: EvidenceItem) -> str:
    """Render one evidence row with escaped cells."""
    return (
        f"| {_cell(item.evidence_id)} | {item.item_kind} "
        f"| {_cell(', '.join(item.entity_ids))} "
        f"| {_cell(item.case_type or '-')} "
        f"| {_cell(item.source_locator)} | {_cell(item.text)} |"
    )


def _listed(title: str, values: list[str]) -> list[str]:
    """Render an optional bullet list, or nothing when it is empty."""
    if not values:
        return []
    return [f"**{title}**", *[f"- {_inline(value)}" for value in values], ""]


def _joined(value: object) -> str:
    """Render a brief value as one display line."""
    if isinstance(value, list):
        return "; ".join(str(entry) for entry in value) or "Unspecified"
    if value is None:
        return "Unspecified"
    return str(value)


def _inline(value: str) -> str:
    """Sanitize one multi-line string for inline Markdown."""
    return " ".join(sanitize_text(value).split())


def _cell(value: str) -> str:
    """Sanitize and escape one table cell."""
    return _inline(value).replace("|", "\\|")


def _lines(parts: list[str]) -> str:
    """Join Markdown lines and end with exactly one newline."""
    return "\n".join(parts).rstrip() + "\n"
