# Internal knowledge assistant design review

*Draft review based on the supplied application brief.*

- Review: cbbcea47390b4d50893393305ce10282
- Created: 2026-11-08T17:50:13.203869+00:00
- Collection: 2026.09
- Generation: 403f1663-5a47-5f53-b20f-4c63d100f0e8
- Retrieval-time provider: fireworks (accounts/fireworks/models/deepseek-v4p1-flash-1017)
- Findings: 3

## Summary

Review of the design brief and supplied ATLAS evidence. The evidence supports indirect prompt injection as a relevant technique through an Exercise case (AML.CS0035), not an observed incident in this application. Main review concerns are untrusted workspace messages, service-identity retrieval scope, and unvalidated output links. No application testing was performed.

## Findings

### F1: Indirect prompt injection through external collaborator workspace messages

- Priority: review_first
- Priority rationale: External collaborators can write in selected channels and the assistant retrieves workspace messages; indirect prompt injection is directly relevant to the review question. The case is an Exercise, so this is a review priority, not evidence of exploitability.
- Techniques: AML.T0051.001
- ATLAS basis: AML.CS0035 is an Exercise case, not an Incident. It describes researchers demonstrating exfiltration through indirect prompt injection in a workspace message and employs AML.T0051.001 at steps S01/S03. AML.T0051.001 specializes AML.T0051. This supports relevance to the design, not proof that this application is exploitable.

**Applicability.** The brief states external collaborators can write in selected channels and the assistant retrieves workspace messages. Retrieved messages are untrusted input. The initial application has no action tools beyond retrieval, but the answer itself and any links are outputs; injected instructions could influence answer content. This is an application-specific inference from the brief, not an observed exploit.

**Source facts from the brief**
- External collaborators can write in selected channels.
- Assistant retrieves workspace messages.
- Initial application has no action tools beyond retrieval.
- Employees authenticate before using the assistant.

**Assumptions and unknowns**
- Whether retrieved workspace messages include external collaborator content by default.
- Whether any input validation or instruction hierarchy is applied.
- Whether the model can be induced to ignore citations.

**Case studies.** AML.CS0035 (Exercise)

**Controls**

*An ATLAS mitigation association is not evidence that the mitigation is present or effective in this application.*

1. Treat retrieved workspace messages as untrusted data and apply input validation/spotlighting before they reach the model; do not let message text override system/developer instructions.
   - Rationale: ATLAS associates Input Validation (AML.M0033) with LLM Prompt Injection (AML.T0051), and AML.T0051.001 specializes AML.T0051. The evidence describes indirect injection through retrieved workspace content. This is a proposed control; the packet does not show it is present or effective.
   - Verification: Place a benign canary instruction in an external collaborator message in a test channel; confirm the assistant does not follow it and that the answer remains grounded in cited content. Review logs for instruction-like retrieved text.
   - ATLAS mitigations: AML.M0033 (parent-linked via AML.T0051)
   - Evidence: E005, E007, E008, E009, E011

**Review questions**
- Are external collaborator messages retrieved by default, or only selected channels?
- Is retrieved text delimited/labeled as untrusted before model consumption?
- Can the assistant follow instructions embedded in retrieved messages?

**Evidence.** E001, E002, E003, E004, E005, E006, E009, E010

### F2: Injected answers could expose service-identity-accessible documents beyond requester authorization

- Priority: review_first
- Priority rationale: Sensitive internal documents are in scope and retrieval uses a service identity; if requester-level authorization is absent, an injected answer could expose documents the employee should not see. The packet does not confirm this is absent, so it is a verification priority.
- Techniques: AML.T0051.001
- ATLAS basis: AML.CS0035 is an Exercise case, not an Incident. It describes exfiltration through indirect prompt injection in a workspace message and employs AML.T0051.001. AML.T0051.001 describes placing instructions in data the model later retrieves. This supports the injection path; the service-identity authorization concern is an application-specific inference from the brief, not an ATLAS claim.

**Applicability.** The brief states retrieval runs under an application service identity and lists internal documents accessible to that identity as sensitive assets. It does not state whether document authorization is checked for the requesting employee. If not, an injected answer could surface content beyond the requester's permissions. This is an application-specific inference and unknown, not an observed flaw.

**Source facts from the brief**
- Retrieval runs under an application service identity.
- Sensitive assets include internal documents accessible to the service identity.
- Unknown whether document authorization is checked for the requesting employee.

**Assumptions and unknowns**
- Whether document authorization is checked for the requester.
- Whether the service identity has broad repository access.
- Whether answers can quote or link to documents.

**Case studies.** AML.CS0035 (Exercise)

**Controls**

*An ATLAS mitigation association is not evidence that the mitigation is present or effective in this application.*

1. Enforce requester-level document authorization before retrieval and before including retrieved content in the answer; do not rely solely on the service identity's access.
   - Rationale: The brief states retrieval runs under an application service identity and does not state whether document authorization is checked for the requesting employee. This is a proposed control; no evidence shows it is present.
   - Verification: Test with two employees having different document permissions and an external collaborator message; confirm the assistant cannot return content from documents the requester cannot access.
   - ATLAS mitigations: none named (reviewer-suggested check, not an ATLAS-prescribed control)
   - Evidence: E001, E002, E005
2. Apply least privilege to the application service identity so it can read only documents needed for the assistant's function.
   - Rationale: The brief lists internal documents accessible to the service identity as sensitive assets; least privilege reduces blast radius if injected content steers retrieval. This is a proposed control, not evidence of current configuration.
   - Verification: Review service identity permissions against required document sets; confirm no broad repository access.
   - ATLAS mitigations: none named (reviewer-suggested check, not an ATLAS-prescribed control)
   - Evidence: E001, E005

**Review questions**
- Is document authorization checked against the requesting employee before retrieval and before answer generation?
- What is the service identity's document scope?
- Can the answer include document content the requester cannot open?

**Evidence.** E001, E002, E005

### F3: Model-generated output links are not confirmed to be validated outside the model

- Priority: review_next
- Priority rationale: The brief explicitly leaves returned-link validation unknown; links are an output channel that could be influenced by retrieved content. This is lower than the injection and authorization questions but still worth review before rollout.
- Techniques: AML.T0051.001
- ATLAS basis: AML.CS0035 is an Exercise case, not an Incident. It describes exfiltration through indirect prompt injection in a workspace message. AML.T0051.001 describes placing instructions in data the model later retrieves. This supports the injection path; the link-validation concern is an application-specific unknown from the brief, not an ATLAS claim.

**Applicability.** The assistant returns a cited answer. If answers can include links, retrieved content could influence link text or target. The brief does not state whether links are validated outside the model. The initial application has no action tools beyond retrieval, but links are still output.

**Source facts from the brief**
- Assistant returns a cited answer.
- Unknown whether returned links are validated outside the model.

**Assumptions and unknowns**
- Whether answers can contain links.
- Whether link validation exists outside the model.
- Whether links are auto-fetched or rendered as clickable.

**Case studies.** AML.CS0035 (Exercise)

**Controls**

*An ATLAS mitigation association is not evidence that the mitigation is present or effective in this application.*

1. Validate or allowlist outbound links outside the model before display; show the destination domain and do not auto-fetch links.
   - Rationale: The brief lists returned-link validation as unknown; ATLAS evidence describes indirect injection via retrieved workspace content. This is a proposed control, not evidence it is present.
   - Verification: Inject a benign instruction in an external collaborator message asking for a link to a test domain; confirm the assistant either refuses or the link is blocked/flagged by an external validator.
   - ATLAS mitigations: none named (reviewer-suggested check, not an ATLAS-prescribed control)
   - Evidence: E001, E005

**Review questions**
- Does the assistant return links?
- Are links validated/allowlisted outside the model?
- Are links auto-fetched or rendered as clickable?

**Evidence.** E001, E005

**Unanswered questions**
- Is document authorization checked against the requesting employee before retrieval and before answer generation?
- What is the service identity's document scope?
- Are external collaborator messages retrieved by default?
- Is retrieved text delimited/labeled as untrusted before model consumption?
- Does the assistant return links, and are they validated/allowlisted outside the model?
- Are links auto-fetched or rendered as clickable?

**Evidence gaps**
- No evidence about the application's actual retrieval authorization, input validation, link validation, logging, or red-team results.
- No evidence that AML.M0033 is implemented or effective in this application.
- No evidence of an incident; AML.CS0035 is an Exercise case.
- No evidence about whether external collaborator messages are included in retrieval by default.

## Citations

- E001 — AML.CS0035, Exercise (passage, /case-studies/AML.CS0035/description). ATLAS: https://atlas.mitre.org/studies/AML.CS0035
- E002 — AML.CS0035, AML.T0051.001, Exercise (passage, /relationships/AML.CS0035/employs/2). ATLAS: https://atlas.mitre.org/studies/AML.CS0035, https://atlas.mitre.org/techniques/AML.T0051.001
- E003 — AML.CS0035, AML.T0051.001, Exercise (relationship, /relationships/AML.CS0035/employs/2). ATLAS: https://atlas.mitre.org/studies/AML.CS0035, https://atlas.mitre.org/techniques/AML.T0051.001
- E004 — AML.CS0035, AML.T0051.001, Exercise (relationship, /relationships/AML.CS0035/employs/0). ATLAS: https://atlas.mitre.org/studies/AML.CS0035, https://atlas.mitre.org/techniques/AML.T0051.001
- E005 — AML.T0051.001 (passage, /techniques/AML.T0051.001/description). ATLAS: https://atlas.mitre.org/techniques/AML.T0051.001
- E006 — AML.CS0035, AML.T0051.001, Exercise (passage, /relationships/AML.CS0035/employs/0). ATLAS: https://atlas.mitre.org/studies/AML.CS0035, https://atlas.mitre.org/techniques/AML.T0051.001
- E007 — AML.M0033 (passage, /mitigations/AML.M0033/description). ATLAS: https://atlas.mitre.org/mitigations/AML.M0033
- E008 — AML.M0033, AML.T0051 (relationship, /relationships/AML.M0033/mitigates/0). ATLAS: https://atlas.mitre.org/mitigations/AML.M0033, https://atlas.mitre.org/techniques/AML.T0051
- E009 — AML.T0051.001, AML.T0051 (relationship, /relationships/AML.T0051.001/specializes/0). ATLAS: https://atlas.mitre.org/techniques/AML.T0051.001, https://atlas.mitre.org/techniques/AML.T0051
- E010 — AML.T0051 (passage, /techniques/AML.T0051/description). ATLAS: https://atlas.mitre.org/techniques/AML.T0051
- E011 — AML.M0033, AML.T0051 (passage, /relationships/AML.M0033/mitigates/0). ATLAS: https://atlas.mitre.org/mitigations/AML.M0033, https://atlas.mitre.org/techniques/AML.T0051
- E012 — AML.T0012 (passage, /techniques/AML.T0012/description). ATLAS: https://atlas.mitre.org/techniques/AML.T0012
- E013 — AML.CS0035, AML.T0012, Exercise (relationship, /relationships/AML.CS0035/employs/1). ATLAS: https://atlas.mitre.org/studies/AML.CS0035, https://atlas.mitre.org/techniques/AML.T0012
- E014 — AML.CS0024, Incident (passage, /case-studies/AML.CS0024/description). ATLAS: https://atlas.mitre.org/studies/AML.CS0024
- E015 — AML.CS0035, AML.T0012, Exercise (passage, /relationships/AML.CS0035/employs/1). ATLAS: https://atlas.mitre.org/studies/AML.CS0035, https://atlas.mitre.org/techniques/AML.T0012

## Application brief

| Field | Value |
|---|---|
| Title | Internal knowledge assistant design review |
| Review question | How could untrusted retrieved content influence sensitive answers, and what should we verify before rollout? |
| Retrieval focus | Indirect prompt injection through retrieved workspace messages influencing sensitive answers and output links |
| System description | An employee assistant retrieves internal documents and workspace messages, then returns a cited answer. Retrieval runs under an application service identity. |
| Users and roles | Employees; External collaborators who can write in selected channels |
| Data sources | Internal document repository; Workspace messages |
| Untrusted inputs | Messages written by external collaborators |
| Tools and permissions | Search documents; Read retrieved messages; Return an answer to the requesting employee |
| Sensitive assets | Internal documents accessible to the service identity |
| Existing controls | Employees authenticate before using the assistant |
| Constraints | The initial application has no action tools beyond retrieval |
| Unknowns | Whether document authorization is checked for the requesting employee; Whether returned links are validated outside the model |

## Evidence appendix

ATLAS material: MITRE ATLAS data repository, Apache License 2.0, collection 2026.09.
Generated retrieval context is excluded; every excerpt below is stored ATLAS source text.

| ID | Kind | Entities | Case type | Source locator | Excerpt |
|---|---|---|---|---|---|
| E001 | passage | AML.CS0035 | Exercise | /case-studies/AML.CS0035/description | Case AML.CS0035 \| Data Exfiltration from Slack AI... Type: Exercise Researchers demonstrated exfiltration through indirect prompt injection [Existing reference](https://example.com/report) in a workspace message. |
| E002 | passage | AML.CS0035, AML.T0051.001 | Exercise | /relationships/AML.CS0035/employs/2 | Case AML.CS0035 \| Step S03 \| employs AML.T0051.001 Technique: LLM Prompt Injection / Indirect Type: Exercise The victim queried the assistant and received injected output. |
| E003 | relationship | AML.CS0035, AML.T0051.001 | Exercise | /relationships/AML.CS0035/employs/2 | Data Exfiltration from Slack AI (AML.CS0035) employs LLM Prompt Injection / Indirect (AML.T0051.001) \| step S03 \| case type: Exercise |
| E004 | relationship | AML.CS0035, AML.T0051.001 | Exercise | /relationships/AML.CS0035/employs/0 | Data Exfiltration from Slack AI (AML.CS0035) employs LLM Prompt Injection / Indirect (AML.T0051.001) \| step S01 \| case type: Exercise |
| E005 | passage | AML.T0051.001 | - | /techniques/AML.T0051.001/description | Technique AML.T0051.001 \| LLM Prompt Injection / Indirect... An adversary may place instructions in data the model later retrieves, for example a payload like reset in a public document. |
| E006 | passage | AML.CS0035, AML.T0051.001 | Exercise | /relationships/AML.CS0035/employs/0 | Case AML.CS0035 \| Step S01 \| employs AML.T0051.001 Technique: LLM Prompt Injection / Indirect Type: Exercise The adversary planted instructions in a public channel. |
| E007 | passage | AML.M0033 | - | /mitigations/AML.M0033/description | Mitigation AML.M0033 \| Input Validation... Validate untrusted text before it reaches a model. |
| E008 | relationship | AML.M0033, AML.T0051 | - | /relationships/AML.M0033/mitigates/0 | Input Validation (AML.M0033) mitigates LLM Prompt Injection (AML.T0051) |
| E009 | relationship | AML.T0051.001, AML.T0051 | - | /relationships/AML.T0051.001/specializes/0 | LLM Prompt Injection / Indirect (AML.T0051.001) specializes LLM Prompt Injection (AML.T0051) |
| E010 | passage | AML.T0051 | - | /techniques/AML.T0051/description | Technique AML.T0051 \| LLM Prompt Injection... Adversaries may inject instructions & markup such as IGNORE PREVIOUS into text consumed by a language model. |
| E011 | passage | AML.M0033, AML.T0051 | - | /relationships/AML.M0033/mitigates/0 | Mitigation AML.M0033 \| mitigates AML.T0051 Technique: LLM Prompt Injection Validation prevents injected instructions from executing. |
| E012 | passage | AML.T0012 | - | /techniques/AML.T0012/description | Technique AML.T0012 \| Valid Accounts... Adversaries may obtain and abuse credentials of existing accounts. See [[missing-ref]] for related reporting. |
| E013 | relationship | AML.CS0035, AML.T0012 | Exercise | /relationships/AML.CS0035/employs/1 | Data Exfiltration from Slack AI (AML.CS0035) employs Valid Accounts (AML.T0012) \| step S02 \| case type: Exercise |
| E014 | passage | AML.CS0024 | Incident | /case-studies/AML.CS0024/description | Case AML.CS0024 \| Exploitation of a Chatbot... Type: Incident An incident case used only to check case-type handling. |
| E015 | passage | AML.CS0035, AML.T0012 | Exercise | /relationships/AML.CS0035/employs/1 | Case AML.CS0035 \| Step S02 \| employs AML.T0012 Technique: Valid Accounts Type: Exercise The researcher created a valid, non-admin account. |
