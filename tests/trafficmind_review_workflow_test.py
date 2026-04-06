"""Tests for trafficmind.review.workflow."""

import unittest

from trafficmind.review.models import AssetViewKind
from trafficmind.review.models import EvidenceAccessMode
from trafficmind.review.models import EvidenceAssetReference
from trafficmind.review.models import EvidenceManifest
from trafficmind.review.models import EvidenceMediaKind
from trafficmind.review.models import EvidenceReference
from trafficmind.review.models import GroundedNarrative
from trafficmind.review.models import GroundingReference
from trafficmind.review.models import GroundingSourceKind
from trafficmind.review.models import MaskingOperation
from trafficmind.review.models import MultimodalReviewRequest
from trafficmind.review.models import OperatorNote
from trafficmind.review.models import PriorReviewEntry
from trafficmind.review.models import RedactionPolicy
from trafficmind.review.models import RedactionState
from trafficmind.review.models import ReviewDraft
from trafficmind.review.models import ReviewEvent
from trafficmind.review.models import ReviewRole
from trafficmind.review.models import RuleExplanation
from trafficmind.review.models import SensitiveVisualDetail
from trafficmind.review.models import SensitiveVisualKind
from trafficmind.review.prompts import build_prompt_bundle
from trafficmind.review.workflow import MultimodalReviewWorkflow

T0 = 1000.0


def _request(
    *,
    event_metadata=None,
    evidence=None,
    operator_notes=(),
    prior_review_history=(),
    viewer_role=ReviewRole.OPERATOR,
    redaction_policy=None,
):
  event = ReviewEvent(
      incident_id="INC-1",
      event_type="red_light_review",
      violation_type="red_light",
      occurred_at=T0,
      junction_id="J1",
      phase_id="P1",
      title="Potential red-light incident",
      metadata={} if event_metadata is None else dict(event_metadata),
  )
  rule = RuleExplanation(
      rule_id="rule-red-1",
      title="Red-light rule",
      explanation=(
          "Vehicle crossed the stop line while the signal was restrictive."
      ),
      triggered_conditions=(
          "vehicle entered stop-line zone",
          "signal state was restrictive",
      ),
      unresolved_conditions=("operator should verify evidence timestamps",),
      deterministic_basis=(
          "Signal state comes from deterministic incident metadata",
      ),
  )
  manifest = EvidenceManifest(
      manifest_id="MAN-1",
      incident_id="INC-1",
      references=tuple(evidence or ()),
  )
  return MultimodalReviewRequest(
      event=event,
      rule_explanation=rule,
      evidence_manifest=manifest,
      operator_notes=tuple(operator_notes),
      prior_review_history=tuple(prior_review_history),
      viewer_role=viewer_role,
      redaction_policy=redaction_policy,
  )


def _asset(access_mode, storage_uri=None, **metadata):
  return EvidenceAssetReference(
      access_mode=access_mode,
      storage_uri=storage_uri,
      metadata=dict(metadata),
  )


def _sensitive(detail_id, kind, masking_operation):
  return SensitiveVisualDetail(
      detail_id=detail_id,
      kind=kind,
      masking_operation=masking_operation,
  )


class EmptyReferenceAssistant:
  """Assistant stub that intentionally omits grounding references."""

  def review(self, prompt_bundle):
    del prompt_bundle
    empty = GroundedNarrative(text="Assistant draft without explicit refs")
    return ReviewDraft(
        review_summary=empty,
        likely_cause=empty,
        confidence_caveats=(empty,),
        recommended_operator_action=empty,
        escalation_suggestion=empty,
    )


class FabricatedRefsAssistant:
  """Assistant stub that returns references with unknown source_ids."""

  def review(self, prompt_bundle):
    del prompt_bundle
    fake_ref = GroundingReference(
        source_kind=GroundingSourceKind.EVENT_METADATA,
        source_id="FABRICATED-999",
        label="Fake source",
        detail="This source does not exist in the grounding context",
    )
    narrative = GroundedNarrative(
        text="Draft with fabricated refs",
        references=(fake_ref,),
    )
    return ReviewDraft(
        review_summary=narrative,
        likely_cause=narrative,
        confidence_caveats=(narrative,),
        recommended_operator_action=narrative,
        escalation_suggestion=narrative,
    )


class TestMultimodalReviewWorkflow(unittest.TestCase):

  def test_langgraph_workflow_returns_grounded_sections(self):
    workflow = MultimodalReviewWorkflow()
    request = _request(
        event_metadata={"signal_conflict": True},
        evidence=(
            EvidenceReference(
                evidence_id="frame-1",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-1.jpg",
                label="Overview frame",
                observed_at=T0,
            ),
            EvidenceReference(
                evidence_id="clip-1",
                media_kind=EvidenceMediaKind.CLIP,
                access_mode=EvidenceAccessMode.STORED_REFERENCE,
                storage_uri="s3://bucket/clip-1.mp4",
                label="Signal clip",
                clip_start=T0 - 5,
                clip_end=T0 + 5,
            ),
        ),
        operator_notes=(
            OperatorNote(author="op-a", note="Vehicle looked late on yellow."),
        ),
    )

    result = workflow.invoke(request)

    self.assertTrue(result.used_langgraph)
    self.assertFalse(result.used_assistant_model)
    self.assertIn("attached media", result.review_summary.text)
    self.assertGreaterEqual(len(result.review_summary.references), 2)
    self.assertIn(
        "controller-versus-vision disagreement", result.likely_cause.text
    )
    caveat_texts = [caveat.text for caveat in result.confidence_caveats]
    self.assertTrue(any("stored references" in text for text in caveat_texts))
    self.assertTrue(
        any("advisory" in item.lower() for item in result.audit_log)
    )

  def test_inventory_and_caveats_call_out_metadata_vs_attachments(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(
        event_metadata={"stale_signal": True},
        evidence=(
            EvidenceReference(
                evidence_id="crop-1",
                media_kind=EvidenceMediaKind.CROP,
                access_mode=EvidenceAccessMode.METADATA_ONLY,
                label="Plate crop",
                description="Metadata record only",
            ),
            EvidenceReference(
                evidence_id="clip-2",
                media_kind=EvidenceMediaKind.CLIP,
                access_mode=EvidenceAccessMode.STORED_REFERENCE,
                storage_uri="s3://bucket/clip-2.mp4",
                label="Approach clip",
            ),
        ),
    )

    result = workflow.invoke(request)

    self.assertFalse(result.used_langgraph)
    self.assertEqual(result.evidence_inventory.attached_media_count, 0)
    self.assertEqual(result.evidence_inventory.stored_reference_count, 1)
    self.assertEqual(result.evidence_inventory.metadata_only_count, 1)
    caveat_texts = [caveat.text for caveat in result.confidence_caveats]
    self.assertTrue(
        any(
            "did not receive attached images or clips" in text
            for text in caveat_texts
        )
    )
    self.assertTrue(any("metadata-only" in text for text in caveat_texts))
    self.assertIn(
        "Verify signal telemetry freshness",
        result.recommended_operator_action.text,
    )

  def test_prior_review_history_drives_escalation(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(
        evidence=(
            EvidenceReference(
                evidence_id="frame-2",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="file:///evidence/frame-2.jpg",
            ),
        ),
        prior_review_history=(
            PriorReviewEntry(
                reviewer="rev-a",
                reviewed_at=T0 - 3600,
                summary="Similar signal review occurred yesterday.",
                decision="escalated",
                action_taken="checked controller logs",
                escalation_outcome="maintenance follow-up requested",
            ),
            PriorReviewEntry(
                reviewer="rev-b",
                reviewed_at=T0 - 7200,
                summary="Another similar incident last week.",
                decision="confirmed",
                action_taken="escalated to operations",
            ),
        ),
    )

    result = workflow.invoke(request)

    self.assertIn(
        "Escalate to operations or maintenance review",
        result.escalation_suggestion.text,
    )
    self.assertGreaterEqual(len(result.escalation_suggestion.references), 2)

  def test_prompt_bundle_marks_access_modes_explicitly(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(
        evidence=(
            EvidenceReference(
                evidence_id="frame-3",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-3.jpg",
            ),
            EvidenceReference(
                evidence_id="clip-3",
                media_kind=EvidenceMediaKind.CLIP,
                access_mode=EvidenceAccessMode.STORED_REFERENCE,
                storage_uri="s3://bucket/clip-3.mp4",
            ),
            EvidenceReference(
                evidence_id="crop-3",
                media_kind=EvidenceMediaKind.CROP,
                access_mode=EvidenceAccessMode.METADATA_ONLY,
            ),
        ),
    )

    grounded_state = workflow._ground_inputs(
        {"request": request, "audit_log": []}
    )
    prompt_state = workflow._prepare_prompt({
        "request": request,
        "audit_log": grounded_state["audit_log"],
        "grounding_context": grounded_state["grounding_context"],
    })
    prompt_bundle = prompt_state["prompt_bundle"]

    self.assertEqual(len(prompt_bundle.attachment_evidence), 1)
    self.assertIn(
        "actual attached media available to the workflow",
        prompt_bundle.user_prompt,
    )
    self.assertIn(
        "stored media reference only; not attached to this invocation",
        prompt_bundle.user_prompt,
    )
    self.assertIn(
        "metadata only; no attached media available", prompt_bundle.user_prompt
    )
    self.assertIn(
        "Never claim to have visually inspected", prompt_bundle.system_prompt
    )

  def test_assistant_output_without_refs_is_repaired(self):
    workflow = MultimodalReviewWorkflow(
        assistant=EmptyReferenceAssistant(),
        use_langgraph=False,
    )
    request = _request(
        evidence=(
            EvidenceReference(
                evidence_id="frame-4",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-4.jpg",
            ),
        ),
    )

    result = workflow.invoke(request)

    self.assertTrue(result.used_assistant_model)
    self.assertTrue(result.review_summary.references)
    self.assertTrue(result.likely_cause.references)
    self.assertTrue(result.confidence_caveats[0].references)
    self.assertTrue(
        any(
            "Injected fallback references" in entry
            for entry in result.audit_log
        ),
        "Audit log should record that fallback references were injected",
    )

  def test_fabricated_assistant_refs_are_dropped_and_logged(self):
    workflow = MultimodalReviewWorkflow(
        assistant=FabricatedRefsAssistant(),
        use_langgraph=False,
    )
    request = _request(
        evidence=(
            EvidenceReference(
                evidence_id="frame-5",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-5.jpg",
            ),
        ),
    )

    result = workflow.invoke(request)

    self.assertTrue(result.used_assistant_model)
    for section in (
        result.review_summary,
        result.likely_cause,
        result.recommended_operator_action,
        result.escalation_suggestion,
    ):
      for ref in section.references:
        self.assertNotEqual(
            ref.source_id,
            "FABRICATED-999",
            f"Fabricated source_id should have been dropped from {section}",
        )
    self.assertTrue(
        any("unrecognized reference" in entry for entry in result.audit_log),
        "Audit log should record dropped unrecognized references",
    )
    self.assertTrue(
        any("Injected fallback" in entry for entry in result.audit_log),
        "Audit log should record fallback injection after all refs were"
        " unrecognized",
    )

  def test_finalize_audit_message_includes_evidence_modes(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(
        evidence=(
            EvidenceReference(
                evidence_id="frame-6",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-6.jpg",
            ),
            EvidenceReference(
                evidence_id="clip-6",
                media_kind=EvidenceMediaKind.CLIP,
                access_mode=EvidenceAccessMode.STORED_REFERENCE,
                storage_uri="s3://bucket/clip-6.mp4",
            ),
        ),
    )

    result = workflow.invoke(request)

    finalize_entries = [e for e in result.audit_log if "Finalized" in e]
    self.assertTrue(finalize_entries, "Expected a Finalized audit entry")
    entry = finalize_entries[0]
    self.assertIn("1 attached", entry)
    self.assertIn("1 stored-ref", entry)
    self.assertIn("0 metadata-only", entry)
    self.assertIn("deterministic logic", entry)

  def test_all_references_includes_attached_media(self):
    from trafficmind.review.workflow import _build_evidence_inventory
    from trafficmind.review.workflow import _build_grounding_context

    request = _request(
        evidence=(
            EvidenceReference(
                evidence_id="frame-7",
                media_kind=EvidenceMediaKind.FRAME,
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-7.jpg",
            ),
        ),
    )
    inventory = _build_evidence_inventory(request.evidence_manifest)
    context = _build_grounding_context(request, inventory)

    self.assertTrue(context.attached_media_references)
    all_ids = {ref.source_id for ref in context.all_references}
    attached_ids = {ref.source_id for ref in context.attached_media_references}
    self.assertTrue(
        attached_ids.issubset(all_ids),
        "all_references must include attached_media_references",
    )

  def test_prompt_bundle_uses_redacted_view_for_operator_role(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(
        viewer_role=ReviewRole.OPERATOR,
        redaction_policy=RedactionPolicy(policy_id="policy-redacted-test"),
        evidence=(
            EvidenceReference(
                evidence_id="frame-private-1",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-private-1-original.jpg",
                ),
                redacted_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-private-1-redacted.jpg",
                ),
                redaction_state=RedactionState.REDACTED_AVAILABLE,
                sensitive_details=(
                    _sensitive(
                        "face-1",
                        SensitiveVisualKind.FACE,
                        MaskingOperation.FACE_MASK,
                    ),
                    _sensitive(
                        "plate-1",
                        SensitiveVisualKind.PLATE,
                        MaskingOperation.PLATE_MASK,
                    ),
                ),
                label="Approach frame",
            ),
        ),
    )

    grounded_state = workflow._ground_inputs(
        {"request": request, "audit_log": []}
    )
    prompt_state = workflow._prepare_prompt({
        "request": request,
        "audit_log": grounded_state["audit_log"],
        "grounding_context": grounded_state["grounding_context"],
    })
    prompt_bundle = prompt_state["prompt_bundle"]

    self.assertEqual(prompt_bundle.viewer_role, ReviewRole.OPERATOR)
    self.assertEqual(prompt_bundle.policy_id, "policy-redacted-test")
    self.assertEqual(len(prompt_bundle.attachment_views), 1)
    self.assertEqual(
        prompt_bundle.attachment_views[0].selected_view,
        AssetViewKind.REDACTED,
    )
    self.assertEqual(
        prompt_bundle.attachment_views[0].selected_storage_uri,
        "s3://bucket/frame-private-1-redacted.jpg",
    )
    self.assertIn("Viewer role: operator", prompt_bundle.user_prompt)
    self.assertIn(
        "redaction_state=redacted_available", prompt_bundle.user_prompt
    )
    self.assertIn("role_view=redacted", prompt_bundle.user_prompt)
    self.assertIn(
        "original assets remain restricted", prompt_bundle.user_prompt
    )

  def test_result_exposes_playback_manifest(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(
        viewer_role=ReviewRole.OPERATOR,
        evidence=(
            EvidenceReference(
                evidence_id="frame-private-2",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-private-2-original.jpg",
                ),
                redacted_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-private-2-redacted.jpg",
                ),
                redaction_state=RedactionState.REDACTED_AVAILABLE,
                sensitive_details=(
                    _sensitive(
                        "plate-2",
                        SensitiveVisualKind.PLATE,
                        MaskingOperation.PLATE_MASK,
                    ),
                ),
                label="Plate frame",
            ),
        ),
    )

    result = workflow.invoke(request)

    self.assertIsNotNone(result.playback_manifest)
    self.assertIsNotNone(result.audit_trail)
    self.assertEqual(result.audit_trail.view_kind.value, "basic")
    self.assertEqual(result.audit_log, result.audit_trail.entries)
    self.assertEqual(result.playback_manifest.viewer_role, ReviewRole.OPERATOR)
    self.assertEqual(
        result.playback_manifest.entries[0].selected_view,
        AssetViewKind.REDACTED,
    )
    self.assertTrue(
        any("role=operator" in entry for entry in result.audit_log),
        "Audit log should capture the privacy-aware role resolution",
    )

  def test_auditor_receives_full_audit_trail_view(self):
    workflow = MultimodalReviewWorkflow(use_langgraph=False)
    request = _request(viewer_role=ReviewRole.AUDITOR)

    result = workflow.invoke(request)

    self.assertIsNotNone(result.audit_trail)
    self.assertEqual(result.audit_trail.view_kind.value, "full")


if __name__ == "__main__":
  unittest.main()
