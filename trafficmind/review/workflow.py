"""Multimodal operator review workflow for TrafficMind.

This workflow is intentionally review-only. It does not participate in live
signal perception, arbitration, or deterministic rule enforcement.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, runtime_checkable

from trafficmind.review.access import resolve_audit_trail
from trafficmind.review.models import EvidenceAccessMode
from trafficmind.review.models import EvidenceInventory
from trafficmind.review.models import EvidenceManifest
from trafficmind.review.models import EvidenceReference
from trafficmind.review.models import GroundedNarrative
from trafficmind.review.models import GroundedReviewContext
from trafficmind.review.models import GroundingReference
from trafficmind.review.models import GroundingSourceKind
from trafficmind.review.models import MultimodalReviewRequest
from trafficmind.review.models import MultimodalReviewResult
from trafficmind.review.models import ReviewDraft
from trafficmind.review.models import ReviewPromptBundle
from trafficmind.review.models import ReviewWorkflowState
from trafficmind.review.privacy import build_playback_manifest
from trafficmind.review.privacy import DEFAULT_REDACTION_POLICY
from trafficmind.review.prompts import build_prompt_bundle

DEFAULT_WORKFLOW_VERSION = "2026-04-06"


@runtime_checkable
class ReviewAssistant(Protocol):
  """Protocol for optional model-backed review assistants.

  Implementations may call a multimodal model or another external system, but
  must return a typed `ReviewDraft`. The workflow will validate that the
  resulting narratives still carry explicit grounding references.
  """

  def review(self, prompt_bundle: ReviewPromptBundle) -> ReviewDraft:
    """Generate a structured review draft from a prompt bundle."""


class MultimodalReviewWorkflow:
  """Evidence-aware incident review workflow for operators.

  The workflow consumes structured event metadata, rule explanation, evidence
  manifest references, operator notes, and prior review history. LangGraph is
  used only to orchestrate the review process. It is not involved in the live
  perception or deterministic rule hot path.
  """

  def __init__(
      self,
      assistant: ReviewAssistant | None = None,
      *,
      use_langgraph: bool = True,
      workflow_version: str = DEFAULT_WORKFLOW_VERSION,
  ) -> None:
    self.assistant = assistant
    self.use_langgraph = use_langgraph
    self.workflow_version = workflow_version
    self._compiled_graph = None

  @property
  def langgraph_available(self) -> bool:
    return _get_langgraph_components() is not None

  def invoke(
      self,
      request: MultimodalReviewRequest,
  ) -> MultimodalReviewResult:
    """Run the review workflow and return a typed result."""
    state: ReviewWorkflowState = {"request": request, "audit_log": []}

    if self.use_langgraph and self.langgraph_available:
      graph = self._get_or_build_graph()
      final_state = graph.invoke(state)
      return final_state["result"]

    state["audit_log"] = list(state.get("audit_log", [])) + [
        "LangGraph unavailable or disabled; used linear workflow execution",
    ]
    final_state = self._run_linear(state)
    return final_state["result"]

  def build_graph(self):
    """Compile and return the LangGraph state graph.

    Raises:
        ImportError: If LangGraph is not installed in the current runtime.
    """
    components = _get_langgraph_components()
    if components is None:
      raise ImportError(
          "LangGraph is required to build the review graph. Install the "
          'optional review dependency with: pip install "langextract[review]"'
      )

    StateGraph, start_token, end_token = components
    graph = StateGraph(ReviewWorkflowState)
    graph.add_node("ground_inputs", self._ground_inputs)
    graph.add_node("prepare_prompt", self._prepare_prompt)
    graph.add_node("draft_review", self._draft_review)
    graph.add_node("finalize_result", self._finalize_result)
    graph.add_edge(start_token, "ground_inputs")
    graph.add_edge("ground_inputs", "prepare_prompt")
    graph.add_edge("prepare_prompt", "draft_review")
    graph.add_edge("draft_review", "finalize_result")
    graph.add_edge("finalize_result", end_token)
    return graph.compile()

  def _get_or_build_graph(self):
    if self._compiled_graph is None:
      self._compiled_graph = self.build_graph()
    return self._compiled_graph

  def _run_linear(self, state: ReviewWorkflowState) -> ReviewWorkflowState:
    grounded = _merge_state(state, self._ground_inputs(state))
    prompted = _merge_state(grounded, self._prepare_prompt(grounded))
    drafted = _merge_state(prompted, self._draft_review(prompted))
    return _merge_state(drafted, self._finalize_result(drafted))

  def _ground_inputs(self, state: ReviewWorkflowState) -> ReviewWorkflowState:
    request = state["request"]
    evidence_inventory = _build_evidence_inventory(request.evidence_manifest)
    grounding_context = _build_grounding_context(request, evidence_inventory)
    audit_log = list(state.get("audit_log", []))
    audit_log.append(
        "Grounded request inputs into event, rule, evidence, note, and"
        " prior-review references"
    )
    audit_log.append(
        "Evidence inventory: "
        f"{evidence_inventory.total_references} refs, "
        f"{evidence_inventory.attached_media_count} attached media, "
        f"{evidence_inventory.stored_reference_count} stored refs, "
        f"{evidence_inventory.metadata_only_count} metadata-only"
    )
    return {
        "evidence_inventory": evidence_inventory,
        "grounding_context": grounding_context,
        "audit_log": audit_log,
    }

  def _prepare_prompt(self, state: ReviewWorkflowState) -> ReviewWorkflowState:
    request = state["request"]
    grounding_context = state["grounding_context"]
    active_policy = request.redaction_policy or DEFAULT_REDACTION_POLICY
    playback_manifest = build_playback_manifest(
        request.evidence_manifest,
        request.viewer_role,
        active_policy,
    )
    prompt_bundle = build_prompt_bundle(
        request,
        grounding_context,
        playback_manifest=playback_manifest,
    )
    audit_log = list(state.get("audit_log", []))
    audit_log.append(
        "Prepared prompt bundle with explicit separation between metadata and"
        " attached media references"
    )
    audit_log.append(
        "Resolved privacy-aware playback manifest for"
        f" role={request.viewer_role.value} under"
        f" policy={active_policy.policy_id};"
        f" default_view={playback_manifest.default_view.value}"
    )
    if playback_manifest.access_audit:
      audit_log.extend(playback_manifest.access_audit)
    return {
        "prompt_bundle": prompt_bundle,
        "playback_manifest": playback_manifest,
        "audit_log": audit_log,
    }

  def _draft_review(self, state: ReviewWorkflowState) -> ReviewWorkflowState:
    prompt_bundle = state["prompt_bundle"]
    request = state["request"]
    context = state["grounding_context"]

    if self.assistant is None:
      draft = self._build_deterministic_draft(request, context)
      used_assistant_model = False
      normalization_repairs: list[str] = []
    else:
      draft, normalization_repairs = self._normalize_assistant_draft(
          self.assistant.review(prompt_bundle), context
      )
      used_assistant_model = True

    audit_log = list(state.get("audit_log", []))
    audit_log.append(
        "Generated structured review draft "
        "using"
        f" {'assistant model' if used_assistant_model else 'deterministic workflow logic'}"
    )
    if normalization_repairs:
      audit_log.extend(normalization_repairs)
    return {
        "draft": draft,
        "used_assistant_model": used_assistant_model,
        "audit_log": audit_log,
    }

  def _finalize_result(self, state: ReviewWorkflowState) -> ReviewWorkflowState:
    request = state["request"]
    draft = state["draft"]
    evidence_inventory = state["evidence_inventory"]
    playback_manifest = state.get("playback_manifest")
    used_model = bool(state.get("used_assistant_model", False))
    audit_log = list(state.get("audit_log", []))
    audit_log.append(
        "Finalized advisory review result. Evidence modes:"
        f" {evidence_inventory.attached_media_count} attached,"
        f" {evidence_inventory.stored_reference_count} stored-ref,"
        f" {evidence_inventory.metadata_only_count} metadata-only. Asset"
        f" variants: {evidence_inventory.original_asset_count} original,"
        f" {evidence_inventory.redacted_asset_count} redacted. Draft source:"
        f" {'assistant model' if used_model else 'deterministic logic'}."
        " Deterministic rule logic remains authoritative."
    )
    audit_trail = resolve_audit_trail(tuple(audit_log), request.viewer_role)
    result = MultimodalReviewResult(
        incident_id=request.event.incident_id,
        event_type=request.event.event_type,
        workflow_version=self.workflow_version,
        used_langgraph=self.use_langgraph and self.langgraph_available,
        used_assistant_model=bool(state.get("used_assistant_model", False)),
        evidence_inventory=evidence_inventory,
        review_summary=draft.review_summary,
        likely_cause=draft.likely_cause,
        confidence_caveats=draft.confidence_caveats,
        recommended_operator_action=draft.recommended_operator_action,
        escalation_suggestion=draft.escalation_suggestion,
        playback_manifest=playback_manifest,
        audit_trail=audit_trail,
        audit_log=audit_trail.entries,
    )
    return {"result": result, "audit_log": audit_log}

  def _normalize_assistant_draft(
      self,
      draft: ReviewDraft,
      context: GroundedReviewContext,
  ) -> tuple[ReviewDraft, list[str]]:
    """Validate and repair assistant-produced references.

    Returns the normalized draft and a list of audit repair messages.
    """
    known_source_ids = frozenset(
        ref.source_id for ref in context.all_references
    )
    repairs: list[str] = []

    def _validate_and_ensure(
        narrative: GroundedNarrative,
        section_name: str,
        fallback: tuple[GroundingReference, ...],
    ) -> GroundedNarrative:
      if narrative.references:
        valid = tuple(
            ref
            for ref in narrative.references
            if ref.source_id in known_source_ids
        )
        dropped = len(narrative.references) - len(valid)
        if dropped:
          repairs.append(
              f"Dropped {dropped} unrecognized reference(s) from assistant"
              f" '{section_name}' section"
          )
        if valid:
          return GroundedNarrative(text=narrative.text, references=valid)
        repairs.append(
            f"Injected fallback references into assistant '{section_name}'"
            " section (all original refs were unrecognized)"
        )
        return GroundedNarrative(text=narrative.text, references=fallback[:2])
      repairs.append(
          f"Injected fallback references into assistant '{section_name}'"
          " section (no refs provided)"
      )
      return GroundedNarrative(text=narrative.text, references=fallback[:2])

    normalized = ReviewDraft(
        review_summary=_validate_and_ensure(
            draft.review_summary,
            "review_summary",
            context.event_references + context.rule_references,
        ),
        likely_cause=_validate_and_ensure(
            draft.likely_cause,
            "likely_cause",
            context.event_references
            + context.rule_references
            + context.evidence_references,
        ),
        confidence_caveats=tuple(
            _validate_and_ensure(
                caveat,
                f"confidence_caveats[{i}]",
                context.evidence_references
                + context.operator_note_references
                + context.prior_review_references
                + context.event_references,
            )
            for i, caveat in enumerate(draft.confidence_caveats)
        ),
        recommended_operator_action=_validate_and_ensure(
            draft.recommended_operator_action,
            "recommended_operator_action",
            context.rule_references
            + context.evidence_references
            + context.operator_note_references,
        ),
        escalation_suggestion=_validate_and_ensure(
            draft.escalation_suggestion,
            "escalation_suggestion",
            context.prior_review_references
            + context.event_references
            + context.evidence_references,
        ),
    )
    return normalized, repairs

  def _build_deterministic_draft(
      self,
      request: MultimodalReviewRequest,
      context: GroundedReviewContext,
  ) -> ReviewDraft:
    event = request.event
    inventory = context.evidence_inventory
    event_metadata = event.metadata
    attached_evidence = context.attached_media_references
    evidence_refs = context.evidence_references
    note_refs = context.operator_note_references
    history_refs = context.prior_review_references

    review_summary = GroundedNarrative(
        text=(
            f"Incident {event.incident_id} is a {event.event_type} review for"
            f" junction {event.junction_id}. The review package contains"
            f" {inventory.total_references} evidence references:"
            f" {inventory.attached_media_count} attached media items,"
            f" {inventory.stored_reference_count} stored-but-unattached"
            f" references, and {inventory.metadata_only_count} metadata-only"
            " records. The manifest preserves"
            f" {inventory.original_asset_count} original asset lineage(s) and"
            f" {inventory.redacted_asset_count} redacted variant(s). This"
            " workflow is advisory and must be checked against the"
            " deterministic rule explanation before final disposition."
        ),
        references=context.event_references
        + context.rule_references
        + evidence_refs[:2],
    )

    likely_cause = GroundedNarrative(
        text=_likely_cause_text(request, inventory),
        references=_likely_cause_references(request, context),
    )

    confidence_caveats = _confidence_caveats(request, context)

    recommended_operator_action = GroundedNarrative(
        text=_recommended_action_text(request, inventory),
        references=(
            context.rule_references + evidence_refs[:2] + note_refs[:1]
        ),
    )

    escalation_suggestion = GroundedNarrative(
        text=_escalation_text(request, inventory),
        references=(
            history_refs[:2]
            + context.event_references
            + attached_evidence[:1]
            + evidence_refs[:1]
        ),
    )

    return ReviewDraft(
        review_summary=review_summary,
        likely_cause=likely_cause,
        confidence_caveats=confidence_caveats,
        recommended_operator_action=recommended_operator_action,
        escalation_suggestion=escalation_suggestion,
    )


def _build_evidence_inventory(manifest: EvidenceManifest) -> EvidenceInventory:
  access_counts = Counter(
      reference.best_available_access_mode.value
      for reference in manifest.references
  )
  kind_counts = Counter(
      reference.media_kind.value for reference in manifest.references
  )
  redaction_state_counts = Counter(
      reference.redaction_state.value for reference in manifest.references
  )
  sensitive_kind_counts = Counter(
      detail.kind.value
      for reference in manifest.references
      for detail in reference.sensitive_details
  )
  return EvidenceInventory(
      total_references=len(manifest.references),
      attached_media_count=access_counts.get(
          EvidenceAccessMode.ATTACHED_MEDIA.value, 0
      ),
      stored_reference_count=access_counts.get(
          EvidenceAccessMode.STORED_REFERENCE.value, 0
      ),
      metadata_only_count=access_counts.get(
          EvidenceAccessMode.METADATA_ONLY.value, 0
      ),
      original_asset_count=sum(
          reference.original_asset is not None
          for reference in manifest.references
      ),
      redacted_asset_count=sum(
          reference.redacted_asset is not None
          for reference in manifest.references
      ),
      media_kind_counts=dict(kind_counts),
      redaction_state_counts=dict(redaction_state_counts),
      sensitive_kind_counts=dict(sensitive_kind_counts),
  )


def _build_grounding_context(
    request: MultimodalReviewRequest,
    evidence_inventory: EvidenceInventory,
) -> GroundedReviewContext:
  event = request.event
  rule = request.rule_explanation
  event_refs = (
      GroundingReference(
          source_kind=GroundingSourceKind.EVENT_METADATA,
          source_id=event.incident_id,
          label=event.title or event.event_type,
          detail=(
              f"event_type={event.event_type!r},"
              f" violation_type={event.violation_type!r},"
              f" occurred_at={event.occurred_at},"
              f" junction_id={event.junction_id!r},"
              f" phase_id={event.phase_id!r},"
              f" metadata_keys={sorted(event.metadata)}"
          ),
      ),
  )
  rule_refs = (
      GroundingReference(
          source_kind=GroundingSourceKind.RULE_EXPLANATION,
          source_id=rule.rule_id,
          label=rule.title or rule.rule_id,
          detail=(
              f"explanation={rule.explanation!r};"
              f" triggered_conditions={list(rule.triggered_conditions)!r};"
              f" unresolved_conditions={list(rule.unresolved_conditions)!r};"
              f" deterministic_basis={list(rule.deterministic_basis)!r}"
          ),
      ),
  )

  evidence_refs: list[GroundingReference] = []
  attached_media_refs: list[GroundingReference] = []
  for reference in request.evidence_manifest.references:
    grounding = GroundingReference(
        source_kind=(
            GroundingSourceKind.ATTACHED_MEDIA
            if reference.best_available_access_mode
            == EvidenceAccessMode.ATTACHED_MEDIA
            else GroundingSourceKind.EVIDENCE_METADATA
        ),
        source_id=reference.evidence_id,
        label=reference.label or reference.evidence_id,
        detail=_describe_evidence_reference(reference),
        access_mode=reference.best_available_access_mode,
        media_kind=reference.media_kind,
        redaction_state=reference.redaction_state,
    )
    evidence_refs.append(grounding)
    if (
        reference.best_available_access_mode
        == EvidenceAccessMode.ATTACHED_MEDIA
    ):
      attached_media_refs.append(grounding)

  operator_note_refs = tuple(
      GroundingReference(
          source_kind=GroundingSourceKind.OPERATOR_NOTE,
          source_id=note.note_id or f"note-{index}",
          label=note.author,
          detail=note.note,
      )
      for index, note in enumerate(request.operator_notes, start=1)
  )
  prior_review_refs = tuple(
      GroundingReference(
          source_kind=GroundingSourceKind.PRIOR_REVIEW,
          source_id=f"prior-review-{index}",
          label=entry.reviewer,
          detail=(
              f"decision={entry.decision!r}; summary={entry.summary!r};"
              f" action_taken={entry.action_taken!r};"
              f" escalation_outcome={entry.escalation_outcome!r}"
          ),
      )
      for index, entry in enumerate(request.prior_review_history, start=1)
  )
  return GroundedReviewContext(
      evidence_inventory=evidence_inventory,
      event_references=event_refs,
      rule_references=rule_refs,
      evidence_references=tuple(evidence_refs),
      attached_media_references=tuple(attached_media_refs),
      operator_note_references=operator_note_refs,
      prior_review_references=prior_review_refs,
  )


def _describe_evidence_reference(reference: EvidenceReference) -> str:
  if reference.best_available_access_mode == EvidenceAccessMode.ATTACHED_MEDIA:
    availability = "actual attached media available"
  elif (
      reference.best_available_access_mode
      == EvidenceAccessMode.STORED_REFERENCE
  ):
    availability = (
        "stored media reference only; not attached to this invocation"
    )
  else:
    availability = "metadata only; no media available"
  return (
      f"media_kind={reference.media_kind.value!r};"
      f" access={reference.best_available_access_mode.value!r};"
      f" availability={availability};"
      f" original_asset={_describe_asset(reference.original_asset)!r};"
      f" redacted_asset={_describe_asset(reference.redacted_asset)!r};"
      f" redaction_state={reference.redaction_state.value!r};"
      f" sensitive_kinds={[detail.kind.value for detail in reference.sensitive_details]!r};"
      f" observed_at={reference.observed_at!r};"
      f" clip_window=({reference.clip_start!r}, {reference.clip_end!r});"
      f" description={reference.description!r};"
      f" metadata_keys={sorted(reference.metadata)}"
  )


def _describe_asset(asset) -> str:
  if asset is None:
    return "none"
  return (
      f"mode={asset.access_mode.value}, storage_uri={asset.storage_uri!r}, "
      f"created_at={asset.created_at!r}, created_by={asset.created_by!r}, "
      f"metadata_keys={sorted(asset.metadata)}"
  )


def _likely_cause_text(
    request: MultimodalReviewRequest,
    inventory: EvidenceInventory,
) -> str:
  metadata = request.event.metadata
  if _metadata_truthy(
      metadata, "signal_conflict", "conflict", "source_conflict"
  ):
    return (
        "Available metadata suggests the incident is most likely tied to a"
        " controller-versus-vision disagreement around the rule evaluation"
        " window."
    )
  if _metadata_truthy(metadata, "stale_signal", "stale", "telemetry_delay"):
    return (
        "Available metadata suggests stale or delayed signal telemetry during"
        " the incident window, which may have reduced confidence in the review"
        " context."
    )
  if inventory.attached_media_count == 0 and inventory.total_references > 0:
    return (
        "The manifest is reference-heavy and does not attach actual media to"
        " this invocation, so the workflow can only identify likely cause from"
        " metadata and rule context."
    )
  if len(request.prior_review_history) >= 2:
    return (
        "Repeated prior reviews at the same location suggest a recurring"
        " operational pattern rather than a one-off incident."
    )
  return (
      "The available event metadata and rule explanation align with the"
      " incident, but they do not isolate a single root cause beyond the"
      " deterministic rule trigger."
  )


def _likely_cause_references(
    request: MultimodalReviewRequest,
    context: GroundedReviewContext,
) -> tuple[GroundingReference, ...]:
  metadata = request.event.metadata
  if _metadata_truthy(
      metadata, "signal_conflict", "conflict", "source_conflict"
  ):
    return context.event_references + context.evidence_references[:1]
  if _metadata_truthy(metadata, "stale_signal", "stale", "telemetry_delay"):
    return context.event_references + context.evidence_references[:1]
  if len(request.prior_review_history) >= 2:
    return context.prior_review_references[:2] + context.event_references
  return context.event_references + context.rule_references


def _confidence_caveats(
    request: MultimodalReviewRequest,
    context: GroundedReviewContext,
) -> tuple[GroundedNarrative, ...]:
  inventory = context.evidence_inventory
  active_policy = request.redaction_policy or DEFAULT_REDACTION_POLICY
  caveats: list[GroundedNarrative] = []
  if request.viewer_role in active_policy.mask_by_default_for_roles and any(
      _reference_requires_redaction(reference, active_policy)
      for reference in request.evidence_manifest.references
  ):
    caveats.append(
        GroundedNarrative(
            text=(
                f"Viewer role {request.viewer_role.value} receives redacted"
                " playback by default under privacy policy"
                f" {active_policy.policy_id}; masked faces, plates, or other"
                " identifying details may be unavailable in the visible asset"
                " set."
            ),
            references=context.evidence_references[:2],
        )
    )
  if inventory.attached_media_count == 0 and inventory.total_references > 0:
    caveats.append(
        GroundedNarrative(
            text=(
                "This review did not receive attached images or clips;"
                " conclusions are based on metadata, stored references, rule"
                " context, and notes rather than direct visual inspection."
            ),
            references=context.evidence_references[:2]
            + context.rule_references,
        )
    )
  if inventory.stored_reference_count > 0:
    caveats.append(
        GroundedNarrative(
            text=(
                "Some evidence is available only as stored references, so the"
                " workflow cannot confirm whether those media assets visually"
                " support or contradict the incident."
            ),
            references=context.evidence_references[:2],
        )
    )
  if inventory.metadata_only_count > 0:
    caveats.append(
        GroundedNarrative(
            text=(
                "At least one evidence item is metadata-only, which limits"
                " confidence in asset-level interpretation."
            ),
            references=context.evidence_references[:2],
        )
    )
  if _metadata_truthy(
      request.event.metadata, "stale_signal", "stale", "telemetry_delay"
  ):
    caveats.append(
        GroundedNarrative(
            text=(
                "Stale or delayed telemetry appears in the supplied metadata,"
                " so timestamp alignment should be checked before acting on the"
                " review."
            ),
            references=context.event_references,
        )
    )
  if not request.prior_review_history:
    caveats.append(
        GroundedNarrative(
            text=(
                "No prior review history was supplied, so recurrence and"
                " earlier operator decisions could not be compared."
            ),
            references=context.event_references,
        )
    )
  if not caveats:
    caveats.append(
        GroundedNarrative(
            text=(
                "The supplied package appears internally consistent, but the"
                " output remains advisory and should be checked against"
                " deterministic incident logs."
            ),
            references=context.event_references + context.rule_references,
        )
    )
  return tuple(caveats)


def _recommended_action_text(
    request: MultimodalReviewRequest,
    inventory: EvidenceInventory,
) -> str:
  metadata = request.event.metadata
  stale_signal = _metadata_truthy(
      metadata, "stale_signal", "stale", "telemetry_delay"
  )
  signal_conflict = _metadata_truthy(
      metadata, "signal_conflict", "conflict", "source_conflict"
  )
  if (
      stale_signal
      and inventory.attached_media_count == 0
      and inventory.total_references > 0
  ):
    return (
        "Verify signal telemetry freshness and attach or open the referenced "
        "frame, crop, or clip assets before final disposition so the operator "
        "can compare them with the deterministic rule explanation."
    )
  if inventory.attached_media_count == 0 and inventory.total_references > 0:
    return (
        "Attach or open the referenced frame, crop, or clip assets before final"
        " disposition so the operator can compare them with the deterministic"
        " rule explanation."
    )
  if signal_conflict:
    return (
        "Compare controller logs, vision timeline, and evidence timestamps"
        " before confirming the incident."
    )
  if stale_signal:
    return (
        "Verify signal telemetry freshness and timestamp alignment before"
        " treating the advisory review as actionable."
    )
  return (
      "Use the deterministic rule explanation together with the attached"
      " evidence package to confirm, dismiss, or request follow-up review of"
      " the incident."
  )


def _escalation_text(
    request: MultimodalReviewRequest,
    inventory: EvidenceInventory,
) -> str:
  metadata = request.event.metadata
  repeat_count = len(request.prior_review_history)
  if repeat_count >= 2:
    return (
        "Escalate to operations or maintenance review because similar incidents"
        " have already appeared in prior review history."
    )
  if _metadata_truthy(
      metadata, "signal_conflict", "conflict", "source_conflict"
  ):
    return (
        "Escalate to supervisor or engineering review if controller and vision"
        " disagreement persists after timestamp checks."
    )
  if _metadata_truthy(metadata, "stale_signal", "stale", "telemetry_delay"):
    return (
        "Escalate to telemetry or controller maintenance if stale signal"
        " metadata continues across incidents."
    )
  if (
      inventory.attached_media_count == 0
      and inventory.stored_reference_count > 0
  ):
    return (
        "No immediate escalation is required, but a supervisor review is"
        " appropriate if the stored media cannot be retrieved for operator"
        " confirmation."
    )
  return (
      "No immediate escalation is suggested from the supplied review package."
  )


def _metadata_truthy(metadata: dict[str, Any], *keys: str) -> bool:
  for key in keys:
    if bool(metadata.get(key)):
      return True
  return False


def _reference_requires_redaction(
    reference: EvidenceReference,
    policy,
) -> bool:
  if reference.redaction_state != reference.redaction_state.NOT_REQUIRED:
    return True
  return any(
      policy.requires_redaction(detail.kind)
      for detail in reference.sensitive_details
  )


def _merge_state(
    base: ReviewWorkflowState,
    update: ReviewWorkflowState,
) -> ReviewWorkflowState:
  merged = dict(base)
  merged.update(update)
  return merged


def _get_langgraph_components():
  try:
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph
  except ImportError:
    return None
  return StateGraph, START, END
