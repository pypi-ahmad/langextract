# TrafficMind Multimodal Review Workflow

## Purpose

`MultimodalReviewWorkflow` is an operator-assistance workflow for reviewing an
incident with structured metadata plus attached evidence references such as
frames, crops, and clips.

The workflow now supports privacy-aware evidence presentation so the active
review role can be given redacted assets by default while original assets stay
restricted to explicitly authorized roles.

It is designed for:
- evidence-aware review assistance
- structured incident explanation
- operator guidance
- escalation recommendation

It is **not** designed for:
- live perception
- signal arbitration
- deterministic rule evaluation
- automated enforcement decisions

## Boundary

The workflow sits after deterministic incident creation. The authoritative
incident signal remains the structured event metadata and deterministic rule
explanation supplied to the request.

The workflow may use LangGraph to orchestrate review steps, but it must never
change or replace the deterministic incident outcome on its own.

## Inputs

The workflow consumes a typed `MultimodalReviewRequest` made of:
- `ReviewEvent`: event or violation metadata
- `RuleExplanation`: deterministic rule explanation and triggered conditions
- `EvidenceManifest`: evidence references for frames, crops, clips, or other assets
- `OperatorNote`: optional operator notes
- `PriorReviewEntry`: optional prior review history

## Grounding Rules

Every output section is paired with explicit grounding references. A reference
states whether it comes from:
- event metadata
- rule explanation
- evidence metadata
- actual attached media
- operator notes
- prior review history

Evidence references also distinguish three access modes:
- `attached_media`: actual image or clip is available to the workflow
- `stored_reference`: durable URI or path exists, but the media was not attached to this invocation
- `metadata_only`: manifest record exists, but no media is available

Each evidence reference may also preserve separate `original_asset` and
`redacted_asset` variants plus an explicit `redaction_state`. This keeps
privacy handling and provenance separate rather than overwriting the original
asset URI when a masked variant is produced.

This distinction is important because the workflow must not imply that it has
visually inspected an asset when it only received metadata or a storage URI.

## Outputs

`MultimodalReviewResult` returns:
- `review_summary`
- `likely_cause`
- `confidence_caveats`
- `recommended_operator_action`
- `escalation_suggestion`
- `evidence_inventory`
- `playback_manifest`
- `audit_log`

Each narrative output carries the references that support it.

## Safe Usage

- Treat the workflow as advisory.
- Use attached media when possible for operator review quality.
- Set `viewer_role` and `redaction_policy` on the request when the workflow is
    being used for operator playback or evidence export planning.
- If the workflow only received stored references or metadata-only evidence,
  keep the disposition provisional until an operator checks the assets.
- If the active role receives redacted assets, do not treat hidden masked areas
    as visually verified.
- When conflict or stale telemetry metadata appears, compare timestamps and
  deterministic incident logs before escalating.
- Keep the output with the original evidence manifest so the review remains auditable.

## Privacy Boundary

TrafficMind's current privacy layer is a masking and access-policy foundation.
It does **not** perform full legal/compliance enforcement for any particular
jurisdiction, and it does **not** automatically generate blurred media on its
own. It models:

- sensitive detail kinds such as faces and plates
- separate original versus redacted asset references
- role-based playback / export selection
- explicit warnings when a redacted variant is missing

See [privacy_redaction.md](privacy_redaction.md) for the full behavior and
current boundaries.

## Assistant Output Normalization

When a model-backed `ReviewAssistant` is used, the workflow post-processes every
narrative section before finalization:

1. **Reference validation**: Each `GroundingReference` returned by the assistant
   is checked against the known `source_id` values from the grounding context.
   References with unrecognized source IDs are dropped.
2. **Fallback injection**: If a section ends up with zero valid references (either
   because the assistant provided none or all were unrecognized), the workflow
   injects up to two contextually appropriate fallback references.
3. **Audit trail**: Every repair—dropped references and fallback injections—is
   recorded in the `audit_log` with the affected section name, so downstream
   consumers can distinguish assistant-provided grounding from workflow-injected
   grounding.

## Example

```python
from trafficmind.review import (
    EvidenceAccessMode,
    EvidenceManifest,
    EvidenceMediaKind,
    EvidenceReference,
    MultimodalReviewRequest,
    MultimodalReviewWorkflow,
    ReviewEvent,
    RuleExplanation,
)

workflow = MultimodalReviewWorkflow()

request = MultimodalReviewRequest(
    event=ReviewEvent(
        incident_id="INC-42",
        event_type="red_light_review",
        occurred_at=1712345678.0,
        junction_id="J-42",
        phase_id="P-1",
        metadata={"signal_conflict": True},
    ),
    rule_explanation=RuleExplanation(
        rule_id="rule-red-1",
        explanation="Vehicle crossed the stop line while the signal was restrictive.",
        triggered_conditions=(
            "vehicle entered stop-line zone",
            "signal state was restrictive",
        ),
    ),
    evidence_manifest=EvidenceManifest(
        manifest_id="MAN-42",
        incident_id="INC-42",
        references=(
            EvidenceReference(
                evidence_id="frame-1",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-1.jpg",
            ),
            EvidenceReference(
                evidence_id="clip-1",
                media_kind=EvidenceMediaKind.CLIP,
                access_mode=EvidenceAccessMode.STORED_REFERENCE,
                storage_uri="s3://bucket/clip-1.mp4",
            ),
        ),
    ),
)

result = workflow.invoke(request)
print(result.review_summary.text)
print(result.recommended_operator_action.text)
```