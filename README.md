# atlas-review

Local, offline-first MITRE ATLAS security review RAG for AI application designs.
The authoritative contract is `resources/atlas_security_review_rag_specification.md`.

## Status

Increment 2 is implemented: Increment 1 (ATLAS v6 source acquisition, preview,
normalized entity/relationship storage, stored configuration, and terminal
inspection) plus contextualized ingestion and the two initial search channels.
Ingestion builds owner documents per technique, mitigation, and case study,
reuses an immutable chunk-context cache, chunks source text on retrieval-model
token budgets, embeds the augmented retrieval text with FastEmbed
(`BAAI/bge-base-en-v1.5`), and publishes a USearch dense index and a `bm25s`
keyword index as one hashed bundle that is activated in a single transaction.
The workspace menu runs both channels and fuses their ranks with reciprocal
rank fusion. Reranking, exact-identifier and graph expansion, evidence
packing, and review synthesis belong to later increments.

## Setup

```bash
uv sync --group dev
```

Model artifacts download on first use into the workspace `models/` cache.
Treat that as an explicit, visible step.

## Use

```bash
# Ingest one ATLAS v6 source and print its preview (corpus rows only, no indexes)
uv run atlas-review --workspace ./.atlas-review --source fixtures/mini_atlas_v6.yaml

# Open the terminal workspace menu
uv run atlas-review --workspace ./.atlas-review
```

The menu offers `ingest` (plan, then build on confirmation), `search` (dense and
keyword channels with fused candidates), `inspect` (stored entity detail), and
`quit`. Only uncached source chunks need the configured provider; set the
configured credential variable (default `ATLAS_REVIEW_API_KEY`) before an
ingestion that reports provider requests.

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pyright
uv run pytest
uv run deptry .
uv run vulture
uv run interrogate

# Opt-in real-library check: downloads models and needs sockets
ATLAS_REVIEW_SMOKE=1 uv run pytest -m smoke --force-enable-socket
```
