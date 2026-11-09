# Recorded example review

`internal-knowledge-assistant-design-review-*.md` is one real, unedited export
from this application. It is illustrative output, not a benchmark score, and
not a guarantee that another model run produces the same wording.

## What produced it

| Setting | Value |
|---|---|
| Brief | [`examples/internal_knowledge_assistant.json`](../internal_knowledge_assistant.json) |
| Corpus | [`fixtures/mini_atlas_v6.yaml`](../../fixtures/mini_atlas_v6.yaml), an attributed ATLAS v6 excerpt with 10 searchable passages |
| Provider | Fireworks AI, `https://api.fireworks.ai/inference/v1` |
| Model | `accounts/fireworks/models/deepseek-v4p1-flash-1017` |
| Contextualization allowance | 32,768 output tokens, temperature 0, reasoning effort `medium` |
| Review allowance | 32,768 output tokens, temperature 0, reasoning effort `medium`, JSON-object response mode |
| Recorded | 2026-11-08 |

The report header inside the file records the review ID, timestamp, collection
version, generation ID, and provider/model, all taken from the stored review.

## Reproducing it

```bash
export FIREWORKS_API_KEY='...'
uv run atlas-review --workspace ./.atlas-review
# 4 Configuration -> edit the provider entry to the Fireworks values above
# 1 Ingest ATLAS data -> file -> fixtures/mini_atlas_v6.yaml
#   (default or latest also work; the fixture keeps this recording cheap)
# 2 Run inference / security review -> Load JSON brief ->
#   examples/internal_knowledge_assistant.json -> Retrieve -> Generate
```

The bundled fixture is a small attributed excerpt, so the report can only use
the techniques, cases, and mitigations present in that excerpt. Ingesting the
full pinned ATLAS snapshot from
[`docs/source_manifest.json`](../../docs/source_manifest.json) uses the same
steps and produces a much larger passage set: the plan reported 321 owner
documents and 1,521 chunks, which means 1,521 contextualization requests on a
cold cache. The application shows that count before an explicit build.

## What to look for

- Every finding cites evidence IDs that exist in the packet.
- Exercise cases stay labelled Exercise; the fixture's Incident case stays
  Incident.
- The mitigation association is labelled parent-linked where the shipped path
  goes through a parent technique.
- The application's "no action tools beyond retrieval" constraint is honoured;
  the report does not claim the assistant can act.
- The evidence appendix contains only original ATLAS text. Generated
  retrieval context influenced ranking and is not part of the exported
  evidence.
