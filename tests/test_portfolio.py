"""Portfolio deliverables: example briefs, recorded review, docs, notices.

These checks keep the committed examples and documentation valid as the
application changes. They do not call a model or a provider.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atlas_review.config import PINNED_SOURCE_URL, AppConfig
from atlas_review.contracts import ReviewRequest
from atlas_review.corpus.atlas_v6 import parse_source
from atlas_review.corpus.download import LATEST_POINTER_URL

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.json"))
REPORTS = sorted(
    path
    for path in (ROOT / "examples" / "reports").glob("*.md")
    if path.name != "README.md"
)
MANIFEST = ROOT / "docs" / "source_manifest.json"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
NOTICE = ROOT / "NOTICE"
GITIGNORE = ROOT / ".gitignore"
FIXTURE = ROOT / "fixtures" / "mini_atlas_v6.yaml"
ACQUIRED_AT = "2026-11-01T00:00:00+00:00"
SHA256 = re.compile(r"[0-9a-f]{64}")


def test_three_example_briefs_are_committed() -> None:
    """The specification asks for three review briefs."""
    assert [path.name for path in EXAMPLES] == [
        "email_assistant.json",
        "financial_operations_copilot.json",
        "internal_knowledge_assistant.json",
    ]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_example_brief_validates_and_fits_the_query_budget(example: Path) -> None:
    """Each committed brief is a strict request whose focus fits the budget."""
    brief = ReviewRequest.model_validate_json(example.read_text("utf-8"), strict=True)
    assert brief.title
    assert brief.review_question
    assert brief.system_description
    assert len(brief.retrieval_focus.split()) <= AppConfig().retrieval.query_max_tokens
    assert brief.unknowns


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_example_briefs_have_unique_titles(example: Path) -> None:
    """Distinct scenarios help the reviewer choose the right example."""
    titles = [
        ReviewRequest.model_validate_json(path.read_text("utf-8"), strict=True).title
        for path in EXAMPLES
    ]
    assert len(titles) == len(set(titles))
    assert example.exists()


def test_recorded_review_is_a_labelled_self_contained_export() -> None:
    """One recorded Markdown review carries its provenance and no secrets."""
    assert len(REPORTS) == 1
    markdown = REPORTS[0].read_text("utf-8")
    assert markdown.startswith("# ")
    assert "*Draft review based on the supplied application brief.*" in markdown
    assert "## Summary" in markdown
    assert "## Findings" in markdown
    assert "## Citations" in markdown
    assert "## Evidence appendix" in markdown
    assert "Apache License 2.0" in markdown
    assert "Retrieval-time provider: fireworks" in markdown
    assert "Context (generated):" not in markdown
    assert "FIREWORKS_API_KEY" not in markdown


def test_source_manifest_pins_the_configured_snapshot() -> None:
    """The documented identity matches the configured source and the fixture."""
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    assert manifest["default_url"] == PINNED_SOURCE_URL
    assert manifest["latest_pointer"] in LATEST_POINTER_URL
    assert manifest["resolved_commit"] in manifest["default_url"]
    assert SHA256.fullmatch(manifest["expected_sha256"])
    assert len(manifest["git_blob_sha1"]) == 40
    assert manifest["license"] == "Apache-2.0"
    snapshot = parse_source(
        FIXTURE.read_bytes(), origin=str(FIXTURE), acquired_at=ACQUIRED_AT
    ).snapshot
    assert snapshot.format_version == manifest["format_version"]
    assert snapshot.collection_version == manifest["collection_version"]
    counts = manifest["observed_counts"]
    assert counts["techniques"] >= counts["case_studies"] > 0
    assert manifest["observed_build_plan"]["source_chunks"] > 0


@pytest.mark.parametrize(
    "topic",
    [
        "Source: default",
        "Source: latest",
        "ATLAS-latest.yaml",
        "acquisition time",
    ],
)
def test_readme_documents_source_acquisition(topic: str) -> None:
    """The README explains both downloads and the preview identity."""
    assert topic in (ROOT / "README.md").read_text("utf-8")


@pytest.mark.parametrize(
    "topic",
    [
        "Context generation and cache",
        "BM25",
        "ordinal",
        "reciprocal rank fusion",
        "specializes",
        "SQLite is authoritative",
        "Model adapters",
    ],
)
def test_architecture_document_covers_the_required_topics(topic: str) -> None:
    """The architecture document addresses the specification's required areas."""
    assert topic.lower() in ARCHITECTURE.read_text("utf-8").lower()


@pytest.mark.parametrize(
    "notice",
    [
        "MITRE ATLAS",
        "Apache License 2.0",
        "BAAI/bge-base-en-v1.5",
        "Xenova/ms-marco-MiniLM-L-12-v2",
    ],
)
def test_notice_attributes_sources_and_models(notice: str) -> None:
    """Attribution covers the corpus and both fixed models."""
    assert notice in NOTICE.read_text("utf-8")


@pytest.mark.parametrize(
    "entry",
    [".atlas-review/", ".env", "*.sqlite3", ".venv"],
)
def test_gitignore_excludes_local_state(entry: str) -> None:
    """Workspace state, generated indexes, and secrets stay uncommitted."""
    assert entry in GITIGNORE.read_text("utf-8")
