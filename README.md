# atlas-review

`atlas-review` is a local terminal tool that reviews the security design of an
AI application against [MITRE ATLAS](https://atlas.mitre.org/). You describe
your system in a short brief. The tool finds relevant ATLAS techniques, case
studies, and mitigations, then writes a cited draft report you can read,
export, and discuss with your team.

The tool runs on your machine and keeps corpus data, cached model output, and
saved reviews in a workspace directory that you choose.

## What you get

- A review brief that you can fill in interactively or load from a JSON file.
- Retrieved ATLAS evidence with stable evidence IDs.
- A draft report where each finding cites the evidence it uses.
- An exported Markdown file containing the brief, the report, and the full
  evidence appendix, so a reviewer can check every claim.
- Saved reviews that you can reopen, re-export, or regenerate without
  repeating the retrieval step.

Each report is a draft for human review, not a certification or a test result.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI-compatible chat-completions endpoint and an API key for it.
  You configure the base URL, model name, and key variable yourself.
- Disk space for the embedding and reranking models (downloaded on first use)
  and for the workspace.

## Install

```bash
uv sync --group dev
```

The first ingestion and the first review download the embedding and reranking
models into the workspace `models/` directory. This is a visible, one-time
step that needs network access.

## Quick start

Start the tool with a workspace path:

```bash
uv run atlas-review --workspace ./.atlas-review
```

You get a menu:

```
1  Ingest ATLAS data
2  Run inference / security review
3  Configure inference
4  Inspect corpus and saved reviews
0  Exit
```

### Configure the provider (menu 3)

Enter the provider name, the HTTP(S) API base URL, the model name, and the
name of the environment variable that holds the API key (default
`ATLAS_REVIEW_API_KEY`). The key itself is never stored in the workspace.

Before running a review, export the key in your shell:

```bash
export ATLAS_REVIEW_API_KEY='...'
```

Provider settings are saved in the workspace. Requests are retried
automatically for temporary transport failures (rate limits, server errors,
timeouts). Validation failures are reported, not retried.

### Ingest ATLAS data (menu 1)

Choose one of three sources:

| Option | Meaning |
|---|---|
| `default` | The pinned ATLAS release this version was tested with |
| `latest` | The newest compatible ATLAS 6.x release from the upstream repository |
| `file` | A local ATLAS v6 YAML file |

The tool shows a preview (entity counts, collection version, estimated
passages) and asks before building. Building is a separate, explicit step.
Only text without a cached response needs provider requests, so re-ingesting a
known release is cheap, and the preview shows the request count first.

### Run a review (menu 2)

Load a brief from JSON (`load`) or fill one in field by field (`new`). A brief
asks for:

- the review question and retrieval focus
- the system description
- users and roles
- data sources and untrusted inputs
- tools and permissions
- sensitive assets
- existing controls
- constraints and unknowns

Then choose to retrieve evidence, review the retrieved packet, and generate
the report. Example briefs are in [`examples/`](examples/):

- `internal_knowledge_assistant.json`
- `email_assistant.json`
- `financial_operations_copilot.json`

`examples/reports/` holds one real, unedited exported review along with the
provider settings that produced it.

### Read and export (menu 4)

Reopen any saved review by ID or unique ID prefix. From there you can export
it to Markdown or regenerate it from the same stored evidence. Exports are
written to `exports/` inside the workspace. If a report is unusable, regenerate
it; earlier attempts are kept.

## Command line

```bash
# Store one local ATLAS v6 YAML source and print its preview, without indexes
uv run atlas-review --workspace ./.atlas-review --source fixtures/mini_atlas_v6.yaml
```

Use `--workspace` to keep separate corpora, provider settings, and reviews
apart. If a workspace was created by an older release, the tool asks before
migrating it in place and keeps stored generations and reviews.

## Workspace contents

| Path | Contents |
|---|---|
| `atlas.sqlite3` | Corpus, configuration, generations, and saved reviews |
| `indexes/` | Search indexes built by ingestion |
| `models/` | Downloaded model files |
| `exports/` | Markdown reports you exported |

Do not commit a workspace. It can contain a private brief and full report
text. `./.atlas-review/` is ignored by Git in this repository.

## Cost and data handling

- Corpus ingestion sends ATLAS text excerpts to the configured provider on a
  cold cache. The preview shows the request count before you confirm.
- A review sends the brief and the retrieved ATLAS passages to the provider.
- The only other outbound requests download the ATLAS source release and model
  files. There is no telemetry.
- Generated retrieval context is never included in an exported report or its
  evidence appendix. The appendix quotes original ATLAS text only.

## Troubleshooting

- **Status shows `not configured`** — set the provider URL and model in menu 3.
- **`missing_provider_credential`** — export the environment variable named in
  the provider settings. The tool never reads `OPENAI_API_KEY` on its own.
- **Ingestion refuses a file** — only ATLAS v6 YAML is accepted. The error names
  the unsupported construct.
- **Report generation fails** — the retrieved evidence and the failed attempt
  are kept. Fix the provider or model setting, then regenerate the review.

## License

Copyright 2026 Ryan P. Taylor <ryan@beta-reduce.net>

This project is licensed under the Apache License, Version 2.0. The full text
is in [`LICENSE`](LICENSE). ATLAS material stays under its upstream Apache
License 2.0 terms, and the retrieval models keep their own licenses. See
[`NOTICE`](NOTICE) for sources, versions, and notices.

## Development

Setup, quality gates, and contribution rules are in [`AGENTS.md`](AGENTS.md).
The authoritative product specification is
`resources/atlas_security_review_rag_specification.md`.
