# atlas-review

Local, offline-first MITRE ATLAS security review RAG for AI application designs.
The authoritative contract is `resources/atlas_security_review_rag_specification.md`.

## Status

Increment 1 is implemented: ATLAS v6 YAML source acquisition, preview,
normalized entity/relationship storage in SQLite, stored configuration, and
terminal inspection of techniques, mitigations, and case-study steps. Chunking,
embeddings, retrieval, and review synthesis belong to later increments.

## Setup

```bash
uv sync --group dev
```

## Use

```bash
# Ingest one ATLAS v6 source and print its preview
uv run atlas-review --workspace ./.atlas-review --source fixtures/mini_atlas_v6.yaml

# Open the terminal workspace menu (ingest, inspect, quit)
uv run atlas-review --workspace ./.atlas-review
```

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pyright
uv run pytest
uv run deptry .
```
