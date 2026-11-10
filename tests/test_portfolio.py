"""Portfolio deliverables: example briefs, recorded review, and notices.

These checks keep the committed examples and notices valid as the application
changes. They do not call a model or a provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_review.config import AppConfig
from atlas_review.contracts import ReviewRequest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.json"))
REPORTS = sorted(
    path
    for path in (ROOT / "examples" / "reports").glob("*.md")
    if path.name != "README.md"
)
NOTICE = ROOT / "NOTICE"
GITIGNORE = ROOT / ".gitignore"


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
