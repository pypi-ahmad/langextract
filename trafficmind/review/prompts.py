"""Prompt helpers for TrafficMind operator review workflows.

This module keeps prompt construction isolated from graph orchestration so that
prompt changes stay reviewable and testable.
"""

from __future__ import annotations

from trafficmind.review.models import EvidenceAccessMode
from trafficmind.review.models import EvidenceAssetReference
from trafficmind.review.models import EvidencePlaybackManifest
from trafficmind.review.models import EvidenceReference
from trafficmind.review.models import GroundedReviewContext
from trafficmind.review.models import MultimodalReviewRequest
from trafficmind.review.models import PresentedEvidenceReference
from trafficmind.review.models import ReviewPromptBundle
from trafficmind.review.privacy import build_playback_manifest
from trafficmind.review.privacy import DEFAULT_REDACTION_POLICY

SYSTEM_PROMPT = """You are assisting a traffic operations operator with incident review.

Hard boundaries:
- Deterministic violation logic is authoritative; this workflow does not replace it.
- Use only the supplied event metadata, rule explanation, evidence manifest references,
  operator notes, and prior review history.
- Never claim to have visually inspected an image or clip unless the evidence item is
  explicitly marked as attached media.
- If an evidence item is a stored reference or metadata-only record, say so plainly.
- If a role receives a redacted asset, do not infer hidden face, plate, or identifying
    details behind the masking.
- If original assets are restricted for the active role, state that boundary explicitly.
- Prefer short, operational language over speculative narration.
- When evidence is insufficient, say what is missing and recommend a review action.
"""


def build_prompt_bundle(
    request: MultimodalReviewRequest,
    context: GroundedReviewContext,
    *,
    playback_manifest: EvidencePlaybackManifest | None = None,
) -> ReviewPromptBundle:
  """Build a stable prompt bundle for an external multimodal assistant."""
  active_policy = request.redaction_policy or DEFAULT_REDACTION_POLICY
  playback_manifest = playback_manifest or build_playback_manifest(
      request.evidence_manifest,
      request.viewer_role,
      active_policy,
  )
  attachment_views = tuple(
      entry
      for entry in playback_manifest.entries
      if entry.selected_access_mode == EvidenceAccessMode.ATTACHED_MEDIA
  )
  attachment_ids = {entry.evidence_id for entry in attachment_views}
  attachment_evidence = tuple(
      reference
      for reference in request.evidence_manifest.references
      if reference.evidence_id in attachment_ids
  )
  return ReviewPromptBundle(
      system_prompt=SYSTEM_PROMPT,
      user_prompt=build_user_prompt(request, context, playback_manifest),
      attachment_evidence=attachment_evidence,
      attachment_views=attachment_views,
      playback_manifest=playback_manifest,
      viewer_role=request.viewer_role,
      policy_id=active_policy.policy_id,
      context=context,
  )


def build_user_prompt(
    request: MultimodalReviewRequest,
    context: GroundedReviewContext,
    playback_manifest: EvidencePlaybackManifest,
) -> str:
  """Render a structured user prompt from request and grounding context."""
  event = request.event
  rule = request.rule_explanation
  inventory = context.evidence_inventory
  presented_by_id = {
      entry.evidence_id: entry for entry in playback_manifest.entries
  }

  metadata_lines = []
  for key in sorted(event.metadata):
    metadata_lines.append(f"- {key}: {event.metadata[key]!r}")
  if not metadata_lines:
    metadata_lines.append("- No extra event metadata supplied")

  condition_lines = [f"- {item}" for item in rule.triggered_conditions]
  if not condition_lines:
    condition_lines.append("- No triggered condition list supplied")

  unresolved_lines = [f"- {item}" for item in rule.unresolved_conditions]
  if not unresolved_lines:
    unresolved_lines.append("- No unresolved conditions supplied")

  basis_lines = [f"- {item}" for item in rule.deterministic_basis]
  if not basis_lines:
    basis_lines.append("- No deterministic basis items supplied")

  evidence_lines = [
      _format_evidence_reference(
          reference,
          presented_by_id[reference.evidence_id],
      )
      for reference in request.evidence_manifest.references
  ]
  if not evidence_lines:
    evidence_lines.append("- No evidence references supplied")

  note_lines = [
      f"- {note.author}: {note.note}" for note in request.operator_notes
  ]
  if not note_lines:
    note_lines.append("- No operator notes supplied")

  history_lines = [
      (
          f"- {entry.reviewer} at {entry.reviewed_at}: {entry.decision}; "
          f"summary={entry.summary!r}; action={entry.action_taken!r}; "
          f"escalation={entry.escalation_outcome!r}"
      )
      for entry in request.prior_review_history
  ]
  if not history_lines:
    history_lines.append("- No prior review history supplied")

  return "\n".join([
      "Prepare an advisory incident review with these sections:",
      "1. review summary",
      "2. likely cause",
      "3. confidence caveats",
      "4. recommended operator action",
      "5. escalation suggestion",
      "",
      "State clearly whether each claim is grounded in:",
      "- event metadata",
      "- rule explanation",
      "- evidence metadata",
      "- actual attached media visible to the active role",
      "- operator notes",
      "- prior review history",
      "",
      f"Viewer role: {request.viewer_role.value}",
      f"Privacy policy: {playback_manifest.policy_id}",
      f"Playback default view: {playback_manifest.default_view.value}",
      f"Incident id: {event.incident_id}",
      f"Event type: {event.event_type}",
      f"Violation type: {event.violation_type!r}",
      f"Occurred at: {event.occurred_at}",
      f"Junction id: {event.junction_id}",
      f"Phase id: {event.phase_id!r}",
      f"Title: {event.title!r}",
      "",
      "Event / violation metadata:",
      *metadata_lines,
      "",
      f"Rule id: {rule.rule_id}",
      f"Rule title: {rule.title!r}",
      f"Rule explanation: {rule.explanation}",
      "Triggered conditions:",
      *condition_lines,
      "Unresolved conditions:",
      *unresolved_lines,
      "Deterministic basis:",
      *basis_lines,
      "",
      f"Evidence manifest id: {request.evidence_manifest.manifest_id}",
      (
          f"Evidence counts: total={inventory.total_references},"
          f" attached_media={inventory.attached_media_count},"
          f" stored_reference={inventory.stored_reference_count},"
          f" metadata_only={inventory.metadata_only_count}"
      ),
      "Evidence references:",
      *evidence_lines,
      "",
      "Operator notes:",
      *note_lines,
      "",
      "Prior review history:",
      *history_lines,
      "",
      (
          "Do not replace deterministic rule decisions. If evidence is weak or"
          " only reference-level, say so explicitly."
      ),
  ])


def _format_evidence_reference(
    reference: EvidenceReference,
    presented: PresentedEvidenceReference,
) -> str:
  access_detail = {
      EvidenceAccessMode.ATTACHED_MEDIA: (
          "actual attached media available to the workflow"
      ),
      EvidenceAccessMode.STORED_REFERENCE: (
          "stored media reference only; not attached to this invocation"
      ),
      EvidenceAccessMode.METADATA_ONLY: (
          "metadata only; no attached media available"
      ),
  }[reference.best_available_access_mode]
  parts = [
      f"- {reference.evidence_id}: {reference.media_kind.value}",
      f"mode={reference.best_available_access_mode.value}",
      f"detail={access_detail}",
      f"redaction_state={reference.redaction_state.value}",
      f"original={_format_asset(reference.original_asset)}",
      f"redacted={_format_asset(reference.redacted_asset)}",
      (
          f"role_view={presented.selected_view.value if presented.selected_view else 'metadata_only'}"
      ),
      f"role_access={presented.selected_access_mode.value}",
      f"role_uri={presented.selected_storage_uri!r}",
      (
          f"sensitive={', '.join(kind.value for kind in presented.sensitive_kinds) if presented.sensitive_kinds else 'none'}"
      ),
      f"boundary={presented.access_boundary!r}",
  ]
  if reference.label:
    parts.append(f"label={reference.label!r}")
  if reference.description:
    parts.append(f"description={reference.description!r}")
  if reference.observed_at is not None:
    parts.append(f"observed_at={reference.observed_at}")
  if reference.clip_start is not None or reference.clip_end is not None:
    parts.append(
        f"clip_window=({reference.clip_start!r}, {reference.clip_end!r})"
    )
  if presented.warnings:
    parts.append(f"warnings={list(presented.warnings)!r}")
  return "; ".join(parts)


def _format_asset(asset: EvidenceAssetReference | None) -> str:
  if asset is None:
    return "none"
  return (
      f"mode={asset.access_mode.value}, uri={asset.storage_uri!r}, "
      f"created_at={asset.created_at!r}, created_by={asset.created_by!r}"
  )
