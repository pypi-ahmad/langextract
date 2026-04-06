"""Tests for trafficmind.integrations foundations."""

from __future__ import annotations

import tempfile
import unittest

from trafficmind.integrations import build_case_update
from trafficmind.integrations import build_notification_message
from trafficmind.integrations import CaseSystemAdapter
from trafficmind.integrations import InMemoryCaseSystemAdapter
from trafficmind.integrations import LocalFilesystemObjectStore
from trafficmind.integrations import NotificationSeverity
from trafficmind.integrations import ObjectStorageAdapter
from trafficmind.integrations.models import CaseSyncAction
from trafficmind.integrations.models import ObjectPutRequest
from trafficmind.integrations.signals import adapt_signal_adapter
from trafficmind.integrations.signals import ExternalSignalAdapter
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.review.models import AssetViewKind
from trafficmind.review.models import EvidenceExportBundle
from trafficmind.review.models import EvidenceInventory
from trafficmind.review.models import GroundedNarrative
from trafficmind.review.models import MultimodalReviewResult
from trafficmind.review.models import ReviewEvent
from trafficmind.review.models import ReviewRole
from trafficmind.search.models import ReviewStatus


def _event(*, incident_id: str = "INC-42") -> ReviewEvent:
  return ReviewEvent(
      incident_id=incident_id,
      event_type="red_light_review",
      occurred_at=1000.0,
      junction_id="J-7",
      phase_id="P-2",
      violation_type="red_light",
      title="Potential red-light incident",
  )


def _review_result(*, incident_id: str = "INC-42") -> MultimodalReviewResult:
  summary = GroundedNarrative(text="Deterministic review summary")
  action = GroundedNarrative(text="Verify signal evidence and approve")
  return MultimodalReviewResult(
      incident_id=incident_id,
      event_type="red_light_review",
      workflow_version="2026-04-06",
      used_langgraph=False,
      used_assistant_model=False,
      evidence_inventory=EvidenceInventory(
          total_references=0,
          attached_media_count=0,
          stored_reference_count=0,
          metadata_only_count=0,
      ),
      review_summary=summary,
      likely_cause=GroundedNarrative(text="Likely stale controller signal"),
      confidence_caveats=(GroundedNarrative(text="No attached media"),),
      recommended_operator_action=action,
      escalation_suggestion=GroundedNarrative(
          text="Escalate if conflicts continue"
      ),
  )


def _export_bundle(*, incident_id: str = "INC-42") -> EvidenceExportBundle:
  return EvidenceExportBundle(
      manifest_id="MAN-1",
      incident_id=incident_id,
      viewer_role=ReviewRole.OPERATOR,
      policy_id="privacy-default-v1",
      export_view=AssetViewKind.REDACTED,
      includes_originals=False,
  )


def _state() -> SignalState:
  return SignalState(
      junction_id="J-7",
      controller_id="CTRL-1",
      phase_id="P-2",
      state=PhaseState.GREEN,
      timestamp=1000.0,
      source_type=SourceType.CONTROLLER,
  )


class TestIntegrationModels(unittest.TestCase):

  def test_build_case_update_uses_review_summary_by_default(self):
    update = build_case_update(
        _event(),
        review_status=ReviewStatus.IN_REVIEW,
        review_result=_review_result(),
        export_bundle=_export_bundle(),
        tags=("priority-lane",),
    )

    self.assertEqual(update.incident_id, "INC-42")
    self.assertEqual(update.title, "Potential red-light incident")
    self.assertEqual(update.summary, "Deterministic review summary")
    self.assertEqual(update.review_status, ReviewStatus.IN_REVIEW)
    self.assertEqual(update.tags, ("priority-lane",))

  def test_build_case_update_rejects_mismatched_incident_ids(self):
    with self.assertRaises(ValueError):
      build_case_update(
          _event(incident_id="INC-42"),
          review_result=_review_result(incident_id="INC-99"),
      )

  def test_build_notification_message_includes_recommended_action(self):
    message = build_notification_message(
        build_case_update(
            _event(),
            review_status=ReviewStatus.ESCALATED,
            review_result=_review_result(),
        ),
        severity=NotificationSeverity.CRITICAL,
    )

    self.assertEqual(message.incident_id, "INC-42")
    self.assertEqual(message.severity, NotificationSeverity.CRITICAL)
    self.assertIn("Escalated", message.title)
    self.assertIn("Recommended action", message.body)

  def test_build_notification_uses_summary_fallback_without_review_result(self):
    message = build_notification_message(
        build_case_update(_event(), review_status=ReviewStatus.PENDING),
    )

    self.assertEqual(message.incident_id, "INC-42")
    self.assertIn("Pending", message.title)
    self.assertIn("Potential red-light incident", message.body)
    self.assertNotIn("Recommended action", message.body)


class TestInMemoryCaseSystemAdapter(unittest.TestCase):

  def test_adapter_satisfies_case_system_protocol(self):
    adapter = InMemoryCaseSystemAdapter()
    self.assertIsInstance(adapter, CaseSystemAdapter)

  def test_upsert_returns_created_then_updated(self):
    adapter = InMemoryCaseSystemAdapter()
    update = build_case_update(_event(), review_result=_review_result())

    created = adapter.upsert_case(update)
    updated = adapter.upsert_case(update)

    self.assertEqual(created.action, CaseSyncAction.CREATED)
    self.assertEqual(updated.action, CaseSyncAction.UPDATED)
    self.assertEqual(created.external_case_id, updated.external_case_id)
    self.assertIn("INC-42", adapter.cases)


class TestLocalFilesystemObjectStore(unittest.TestCase):

  def test_put_and_get_round_trip(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      store = LocalFilesystemObjectStore(tmpdir)
      reference = store.put_object(
          ObjectPutRequest(
              object_name="exports/inc-42.json",
              content=b'{"incident_id": "INC-42"}',
              content_type="application/json",
              metadata={"kind": "case-export"},
          )
      )

      blob = store.get_object(reference.storage_uri)

      self.assertEqual(blob.content, b'{"incident_id": "INC-42"}')
      self.assertEqual(blob.reference.content_type, "application/json")
      self.assertEqual(blob.reference.metadata["kind"], "case-export")

  def test_adapter_satisfies_object_storage_protocol(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      store = LocalFilesystemObjectStore(tmpdir)
      self.assertIsInstance(store, ObjectStorageAdapter)

  def test_rejects_path_escape(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      store = LocalFilesystemObjectStore(tmpdir)
      with self.assertRaises(ValueError):
        store.put_object(
            ObjectPutRequest(object_name="../outside.txt", content=b"oops")
        )


class TestSignalAdapterBridge(unittest.TestCase):

  def test_bridge_wraps_external_signal_adapter(self):

    class StubAdapter:

      def adapter_name(self) -> str:
        return "stub-signal-adapter"

      def fetch_signal_states(self):
        return [_state()]

    bridge = adapt_signal_adapter(StubAdapter())
    states = bridge.fetch()

    self.assertEqual(bridge.source_name(), "stub-signal-adapter")
    self.assertEqual(len(states), 1)
    self.assertEqual(states[0].junction_id, "J-7")

  def test_stub_adapter_satisfies_external_signal_protocol(self):

    class StubAdapter:

      def adapter_name(self) -> str:
        return "stub"

      def fetch_signal_states(self):
        return []

    self.assertIsInstance(StubAdapter(), ExternalSignalAdapter)


if __name__ == "__main__":
  unittest.main()
