"""Tests for trafficmind.registry — model/config registry and provenance."""

import unittest

from trafficmind.registry.models import ConfigBundle
from trafficmind.registry.models import EntryStatus
from trafficmind.registry.models import ModelFamily
from trafficmind.registry.models import ModelRegistryEntry
from trafficmind.registry.models import PipelineSnapshot
from trafficmind.registry.models import PipelineSnapshotComparison
from trafficmind.registry.models import ProvenanceChain
from trafficmind.registry.models import ProvenanceStamp
from trafficmind.registry.models import RegistryEntryKind
from trafficmind.registry.models import RulesRegistryEntry
from trafficmind.registry.models import TaskType
from trafficmind.registry.registry import ModelConfigRegistry

# ---------------------------------------------------------------------------
# ConfigBundle
# ---------------------------------------------------------------------------


class TestConfigBundle(unittest.TestCase):

  def test_empty_config_gets_hash(self):
    cb = ConfigBundle()
    self.assertTrue(cb.config_hash.startswith("cfg-"))
    self.assertEqual(len(cb.config_hash), 20)  # "cfg-" + 16 hex chars

  def test_same_params_same_hash(self):
    a = ConfigBundle(parameters={"threshold": 0.5, "nms": 0.4})
    b = ConfigBundle(parameters={"nms": 0.4, "threshold": 0.5})
    self.assertEqual(a.config_hash, b.config_hash)

  def test_different_params_different_hash(self):
    a = ConfigBundle(parameters={"threshold": 0.5})
    b = ConfigBundle(parameters={"threshold": 0.6})
    self.assertNotEqual(a.config_hash, b.config_hash)

  def test_explicit_hash_preserved(self):
    cb = ConfigBundle(parameters={"x": 1}, config_hash="custom-hash")
    self.assertEqual(cb.config_hash, "custom-hash")


# ---------------------------------------------------------------------------
# ModelRegistryEntry
# ---------------------------------------------------------------------------


class TestModelRegistryEntry(unittest.TestCase):

  def test_valid_entry(self):
    entry = ModelRegistryEntry(
        entry_id="yolo-v8-det",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="YOLOv8n",
        version="8.0.1",
    )
    self.assertEqual(entry.entry_id, "yolo-v8-det")
    self.assertEqual(entry.status, EntryStatus.ACTIVE)

  def test_empty_entry_id_rejected(self):
    with self.assertRaises(ValueError):
      ModelRegistryEntry(
          entry_id="",
          family=ModelFamily.DETECTION,
          task_type=TaskType.OBJECT_DETECTION,
          name="YOLOv8n",
          version="8.0.1",
      )

  def test_empty_name_rejected(self):
    with self.assertRaises(ValueError):
      ModelRegistryEntry(
          entry_id="x",
          family=ModelFamily.DETECTION,
          task_type=TaskType.OBJECT_DETECTION,
          name="",
          version="1.0",
      )

  def test_empty_version_rejected(self):
    with self.assertRaises(ValueError):
      ModelRegistryEntry(
          entry_id="x",
          family=ModelFamily.DETECTION,
          task_type=TaskType.OBJECT_DETECTION,
          name="model",
          version="",
      )


# ---------------------------------------------------------------------------
# RulesRegistryEntry
# ---------------------------------------------------------------------------


class TestRulesRegistryEntry(unittest.TestCase):

  def test_valid_entry(self):
    entry = RulesRegistryEntry(
        entry_id="rules-v2",
        rule_set_id="traffic-violations",
        version="2.1",
        rule_ids=["red-light", "stop-line"],
    )
    self.assertEqual(entry.rule_ids, ("red-light", "stop-line"))

  def test_empty_rule_set_id_rejected(self):
    with self.assertRaises(ValueError):
      RulesRegistryEntry(
          entry_id="x",
          rule_set_id="",
          version="1.0",
      )


# ---------------------------------------------------------------------------
# ProvenanceStamp
# ---------------------------------------------------------------------------


class TestProvenanceStamp(unittest.TestCase):

  def test_valid_stamp(self):
    stamp = ProvenanceStamp(
        registry_entry_id="yolo-v8-det",
        entry_kind=RegistryEntryKind.MODEL,
        entry_version="8.0.1",
        task_type=TaskType.OBJECT_DETECTION,
        config_hash="cfg-detstamp000001",
        confidence=0.92,
    )
    self.assertTrue(stamp.stamp_id.startswith("ps-"))
    self.assertEqual(stamp.confidence, 0.92)
    self.assertEqual(stamp.entry_kind, RegistryEntryKind.MODEL)
    self.assertEqual(stamp.entry_label, "yolo-v8-det")

  def test_empty_registry_entry_id_rejected(self):
    with self.assertRaises(ValueError):
      ProvenanceStamp(registry_entry_id="")

  def test_empty_config_hash_rejected(self):
    with self.assertRaises(ValueError):
      ProvenanceStamp(
          registry_entry_id="x",
          entry_kind=RegistryEntryKind.MODEL,
          entry_version="1.0",
      )

  def test_confidence_out_of_range_rejected(self):
    with self.assertRaises(ValueError):
      ProvenanceStamp(
          registry_entry_id="x",
          entry_kind=RegistryEntryKind.MODEL,
          entry_version="1.0",
          config_hash="cfg-teststamp00001",
          confidence=1.5,
      )


# ---------------------------------------------------------------------------
# ProvenanceChain
# ---------------------------------------------------------------------------


class TestProvenanceChain(unittest.TestCase):

  def test_valid_chain(self):
    s1 = ProvenanceStamp(
        registry_entry_id="det-1",
        entry_kind=RegistryEntryKind.MODEL,
        entry_version="1.0",
        task_type=TaskType.OBJECT_DETECTION,
        config_hash="cfg-det1000000001",
    )
    s2 = ProvenanceStamp(
        registry_entry_id="ocr-1",
        entry_kind=RegistryEntryKind.MODEL,
        entry_version="2.0",
        task_type=TaskType.PLATE_RECOGNITION,
        config_hash="cfg-ocr1000000001",
    )
    chain = ProvenanceChain(
        output_id="plate-read-42",
        output_kind="plate_read",
        stamps=(s1, s2),
    )
    self.assertEqual(len(chain.stamps), 2)
    self.assertEqual(chain.stamps[0].task_type, TaskType.OBJECT_DETECTION)
    self.assertEqual(chain.stamps[1].task_type, TaskType.PLATE_RECOGNITION)

  def test_empty_output_id_rejected(self):
    with self.assertRaises(ValueError):
      ProvenanceChain(output_id="", output_kind="event")

  def test_empty_output_kind_rejected(self):
    with self.assertRaises(ValueError):
      ProvenanceChain(output_id="x", output_kind="")


# ---------------------------------------------------------------------------
# PipelineSnapshot
# ---------------------------------------------------------------------------


class TestPipelineSnapshot(unittest.TestCase):

  def test_snapshot_collects_entry_ids(self):
    m = ModelRegistryEntry(
        entry_id="m-1",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="M1",
        version="1.0",
    )
    r = RulesRegistryEntry(
        entry_id="r-1",
        rule_set_id="rs",
        version="1.0",
    )
    snap = PipelineSnapshot(
        snapshot_id="snap-1",
        model_entries=(m,),
        rules_entries=(r,),
    )
    self.assertEqual(snap.all_entry_ids, ("m-1", "r-1"))


# ---------------------------------------------------------------------------
# ModelConfigRegistry
# ---------------------------------------------------------------------------


class TestModelConfigRegistry(unittest.TestCase):

  def _registry(self) -> ModelConfigRegistry:
    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-yolo",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="YOLOv8n",
        version="8.0.1",
        config=ConfigBundle(parameters={"conf": 0.25, "iou": 0.45}),
    )
    reg.register_model(
        entry_id="ocr-paddle",
        family=ModelFamily.OCR,
        task_type=TaskType.PLATE_RECOGNITION,
        name="PaddleOCR",
        version="2.7",
        status=EntryStatus.EXPERIMENTAL,
    )
    reg.register_rules(
        entry_id="rules-traffic-v1",
        rule_set_id="traffic-violations",
        version="1.0",
        rule_ids=["red-light-running", "stop-line-violation"],
        config=ConfigBundle(parameters={"grace_period_s": 0.5}),
    )
    return reg

  def test_register_and_lookup_model(self):
    reg = self._registry()
    entry = reg.get_model("det-yolo")
    self.assertIsNotNone(entry)
    self.assertEqual(entry.name, "YOLOv8n")
    self.assertEqual(entry.config.parameters["conf"], 0.25)

  def test_register_duplicate_model_rejected(self):
    reg = self._registry()
    with self.assertRaises(ValueError):
      reg.register_model(
          entry_id="det-yolo",
          family=ModelFamily.DETECTION,
          task_type=TaskType.OBJECT_DETECTION,
          name="dup",
          version="1.0",
      )

  def test_register_and_lookup_rules(self):
    reg = self._registry()
    entry = reg.get_rules("rules-traffic-v1")
    self.assertIsNotNone(entry)
    self.assertEqual(entry.rule_set_id, "traffic-violations")

  def test_register_duplicate_rules_rejected(self):
    reg = self._registry()
    with self.assertRaises(ValueError):
      reg.register_rules(
          entry_id="rules-traffic-v1",
          rule_set_id="dup",
          version="1.0",
      )

  def test_register_duplicate_entry_id_across_types_rejected(self):
    reg = self._registry()
    with self.assertRaises(ValueError):
      reg.register_rules(
          entry_id="det-yolo",
          rule_set_id="traffic-violations-copy",
          version="1.0",
      )

  def test_get_entry_finds_both_types(self):
    reg = self._registry()
    self.assertIsNotNone(reg.get_entry("det-yolo"))
    self.assertIsNotNone(reg.get_entry("rules-traffic-v1"))
    self.assertIsNone(reg.get_entry("nonexistent"))

  def test_list_models_filtered_by_family(self):
    reg = self._registry()
    det = reg.list_models(family=ModelFamily.DETECTION)
    self.assertEqual(len(det), 1)
    self.assertEqual(det[0].entry_id, "det-yolo")

  def test_list_models_filtered_by_status(self):
    reg = self._registry()
    exp = reg.list_models(status=EntryStatus.EXPERIMENTAL)
    self.assertEqual(len(exp), 1)
    self.assertEqual(exp[0].entry_id, "ocr-paddle")

  def test_list_rules_filtered_by_status(self):
    reg = self._registry()
    active = reg.list_rules(status=EntryStatus.ACTIVE)
    self.assertEqual(len(active), 1)

  def test_set_model_status(self):
    reg = self._registry()
    updated = reg.set_model_status("det-yolo", EntryStatus.DEPRECATED)
    self.assertEqual(updated.status, EntryStatus.DEPRECATED)
    self.assertEqual(reg.get_model("det-yolo").status, EntryStatus.DEPRECATED)

  def test_set_model_status_missing_raises(self):
    reg = self._registry()
    with self.assertRaises(KeyError):
      reg.set_model_status("missing", EntryStatus.INACTIVE)

  def test_set_rules_status(self):
    reg = self._registry()
    updated = reg.set_rules_status("rules-traffic-v1", EntryStatus.INACTIVE)
    self.assertEqual(updated.status, EntryStatus.INACTIVE)

  def test_set_rules_status_missing_raises(self):
    reg = self._registry()
    with self.assertRaises(KeyError):
      reg.set_rules_status("missing", EntryStatus.INACTIVE)


class TestRegistrySnapshots(unittest.TestCase):

  def _registry(self) -> ModelConfigRegistry:
    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-1",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="Det",
        version="1.0",
    )
    reg.register_model(
        entry_id="det-exp",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="DetExp",
        version="0.1",
        status=EntryStatus.EXPERIMENTAL,
    )
    reg.register_rules(
        entry_id="rules-1",
        rule_set_id="rs",
        version="1.0",
    )
    return reg

  def test_snapshot_includes_only_active_by_default(self):
    reg = self._registry()
    snap = reg.take_snapshot(snapshot_id="s1")
    self.assertEqual(len(snap.model_entries), 1)
    self.assertEqual(snap.model_entries[0].entry_id, "det-1")
    self.assertEqual(len(snap.rules_entries), 1)

  def test_snapshot_with_custom_statuses(self):
    reg = self._registry()
    snap = reg.take_snapshot(
        snapshot_id="s2",
        include_statuses=frozenset(
            {EntryStatus.ACTIVE, EntryStatus.EXPERIMENTAL}
        ),
    )
    self.assertEqual(len(snap.model_entries), 2)

  def test_snapshot_stored_and_retrievable(self):
    reg = self._registry()
    reg.take_snapshot(snapshot_id="s3")
    self.assertIsNotNone(reg.get_snapshot("s3"))
    self.assertEqual(reg.snapshot_count, 1)

  def test_list_snapshots(self):
    reg = self._registry()
    reg.take_snapshot(snapshot_id="s4")
    reg.take_snapshot(snapshot_id="s5")
    self.assertEqual(len(reg.list_snapshots()), 2)

  def test_duplicate_snapshot_id_rejected(self):
    reg = self._registry()
    reg.take_snapshot(snapshot_id="dup")
    with self.assertRaises(ValueError):
      reg.take_snapshot(snapshot_id="dup")

  def test_compare_snapshots_reports_added_and_changed_entries(self):
    reg = self._registry()
    reg.take_snapshot(snapshot_id="base")
    reg.set_model_status("det-1", EntryStatus.DEPRECATED)
    reg.set_rules_status("rules-1", EntryStatus.INACTIVE)
    reg.register_model(
        entry_id="ocr-1",
        family=ModelFamily.OCR,
        task_type=TaskType.PLATE_RECOGNITION,
        name="OCR",
        version="1.0",
    )
    reg.take_snapshot(
        snapshot_id="next",
        include_statuses=frozenset({
            EntryStatus.ACTIVE,
            EntryStatus.EXPERIMENTAL,
            EntryStatus.DEPRECATED,
            EntryStatus.INACTIVE,
        }),
    )

    diff = reg.compare_snapshots("base", "next")

    self.assertIsInstance(diff, PipelineSnapshotComparison)
    self.assertEqual(diff.snapshot_a_id, "base")
    self.assertEqual(diff.snapshot_b_id, "next")
    self.assertIn("ocr-1", diff.added_entry_ids)
    self.assertIn("det-1", diff.changed_entry_ids)
    self.assertIn("rules-1", diff.changed_entry_ids)

  def test_compare_snapshots_missing_raises(self):
    reg = self._registry()
    reg.take_snapshot(snapshot_id="s6")
    with self.assertRaises(KeyError):
      reg.compare_snapshots("s6", "missing")


class TestRegistryProvenance(unittest.TestCase):

  def _registry(self) -> ModelConfigRegistry:
    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-yolo",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="YOLOv8n",
        version="8.0.1",
    )
    reg.register_rules(
        entry_id="rules-v1",
        rule_set_id="traffic",
        version="1.0",
    )
    return reg

  def test_stamp_from_model(self):
    reg = self._registry()
    stamp = reg.stamp("det-yolo", confidence=0.88)
    self.assertEqual(stamp.registry_entry_id, "det-yolo")
    self.assertEqual(stamp.entry_kind, RegistryEntryKind.MODEL)
    self.assertEqual(stamp.entry_version, "8.0.1")
    self.assertEqual(stamp.entry_label, "YOLOv8n")
    self.assertEqual(stamp.family, ModelFamily.DETECTION)
    self.assertEqual(stamp.task_type, TaskType.OBJECT_DETECTION)
    self.assertTrue(stamp.config_hash.startswith("cfg-"))
    self.assertEqual(stamp.confidence, 0.88)

  def test_stamp_from_rules(self):
    reg = self._registry()
    stamp = reg.stamp("rules-v1")
    self.assertEqual(stamp.entry_kind, RegistryEntryKind.RULES)
    self.assertEqual(stamp.family, ModelFamily.RULES)
    self.assertEqual(stamp.task_type, TaskType.RULE_EVALUATION)
    self.assertEqual(stamp.entry_version, "1.0")
    self.assertEqual(stamp.rule_set_id, "traffic")
    self.assertEqual(stamp.rule_ids, ())
    self.assertTrue(stamp.config_hash.startswith("cfg-"))

  def test_stamp_missing_entry_raises(self):
    reg = self._registry()
    with self.assertRaises(KeyError):
      reg.stamp("nonexistent")

  def test_build_provenance_chain(self):
    reg = self._registry()
    s1 = reg.stamp("det-yolo", confidence=0.9)
    s2 = reg.stamp("rules-v1")
    chain = reg.build_provenance_chain(
        output_id="EVT-42",
        output_kind="event",
        stamps=[s1, s2],
        pipeline_snapshot_id="snap-prod-a",
    )
    self.assertEqual(chain.output_id, "EVT-42")
    self.assertEqual(chain.pipeline_snapshot_id, "snap-prod-a")
    self.assertEqual(len(chain.stamps), 2)

  def test_counts(self):
    reg = self._registry()
    self.assertEqual(reg.model_count, 1)
    self.assertEqual(reg.rules_count, 1)
    self.assertEqual(reg.snapshot_count, 0)


# ---------------------------------------------------------------------------
# Integration: provenance on review models
# ---------------------------------------------------------------------------


class TestProvenanceOnReviewModels(unittest.TestCase):

  def test_review_event_with_provenance(self):
    from trafficmind.review.models import ReviewEvent

    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-1",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="YOLOv8",
        version="8.0.1",
    )
    reg.register_rules(
        entry_id="rules-1",
        rule_set_id="violations",
        version="1.0",
    )
    chain = reg.build_provenance_chain(
        output_id="INC-100",
        output_kind="event",
        stamps=[
            reg.stamp("det-1", confidence=0.95),
            reg.stamp("rules-1"),
        ],
    )
    event = ReviewEvent(
        incident_id="INC-100",
        event_type="red_light_violation",
        occurred_at=1700000000.0,
        junction_id="J-1",
        provenance=chain,
    )
    self.assertIsNotNone(event.provenance)
    self.assertEqual(len(event.provenance.stamps), 2)
    self.assertEqual(
        event.provenance.stamps[0].task_type, TaskType.OBJECT_DETECTION
    )

  def test_review_event_without_provenance(self):
    """Existing callers that omit provenance are unaffected."""
    from trafficmind.review.models import ReviewEvent

    event = ReviewEvent(
        incident_id="INC-101",
        event_type="speed_violation",
        occurred_at=1700000001.0,
        junction_id="J-2",
    )
    self.assertIsNone(event.provenance)

  def test_evidence_reference_with_provenance(self):
    from trafficmind.review.models import EvidenceAccessMode
    from trafficmind.review.models import EvidenceAssetReference
    from trafficmind.review.models import EvidenceMediaKind
    from trafficmind.review.models import EvidenceReference

    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-face",
        family=ModelFamily.DETECTION,
        task_type=TaskType.FACE_DETECTION,
        name="RetinaFace",
        version="0.3",
    )
    chain = reg.build_provenance_chain(
        output_id="frame-500",
        output_kind="evidence",
        stamps=[reg.stamp("det-face", confidence=0.97)],
    )
    ref = EvidenceReference(
        evidence_id="frame-500",
        media_kind=EvidenceMediaKind.FRAME,
        original_asset=EvidenceAssetReference(
            access_mode=EvidenceAccessMode.STORED_REFERENCE,
            storage_uri="s3://bucket/frame-500.jpg",
        ),
        provenance=chain,
    )
    self.assertEqual(ref.provenance.output_kind, "evidence")

  def test_review_result_with_snapshot_id(self):
    from trafficmind.review.models import EvidenceInventory
    from trafficmind.review.models import GroundedNarrative
    from trafficmind.review.models import MultimodalReviewResult

    result = MultimodalReviewResult(
        incident_id="INC-200",
        event_type="red_light",
        workflow_version="0.9.0",
        used_langgraph=False,
        used_assistant_model=False,
        evidence_inventory=EvidenceInventory(
            total_references=1,
            attached_media_count=1,
            stored_reference_count=0,
            metadata_only_count=0,
        ),
        review_summary=GroundedNarrative(text="summary"),
        likely_cause=GroundedNarrative(text="cause"),
        confidence_caveats=(GroundedNarrative(text="caveat"),),
        recommended_operator_action=GroundedNarrative(text="action"),
        escalation_suggestion=GroundedNarrative(text="none"),
        pipeline_snapshot_id="snap-abc123",
    )
    self.assertEqual(result.pipeline_snapshot_id, "snap-abc123")


class TestProvenanceOnSearchRecords(unittest.TestCase):

  def test_violation_record_can_carry_provenance(self):
    from trafficmind.search.models import RecordKind
    from trafficmind.search.models import SearchFilter
    from trafficmind.search.store import InMemorySearchStore
    from trafficmind.search.store import PlatformRecord

    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-yolo",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="YOLOv8n",
        version="8.0.1",
    )
    reg.register_rules(
        entry_id="rules-violations",
        rule_set_id="traffic-violations",
        version="2.0",
    )
    provenance = reg.build_provenance_chain(
        output_id="V-100",
        output_kind="violation",
        stamps=[
            reg.stamp("det-yolo", confidence=0.93),
            reg.stamp("rules-violations"),
        ],
    )
    store = InMemorySearchStore()
    store.add(
        PlatformRecord(
            record_kind=RecordKind.VIOLATION,
            record_id="V-100",
            label="Red light",
            detail="Vehicle crossed stop line.",
            event_type="red_light_violation",
            violation_type="red_light",
            provenance=provenance,
        )
    )

    hit = store.query(
        SearchFilter(
            violation_types=("red_light",),
            record_kinds=(RecordKind.VIOLATION,),
        )
    )[0]

    self.assertIsNotNone(hit.provenance)
    self.assertEqual(hit.provenance.output_kind, "violation")
    self.assertEqual(hit.reference.provenance.output_id, "V-100")
    self.assertEqual(
        hit.provenance.stamps[0].config_hash,
        reg.get_model("det-yolo").config.config_hash,
    )
    self.assertEqual(hit.provenance.stamps[1].rule_set_id, "traffic-violations")

  def test_plate_read_record_can_carry_provenance(self):
    from trafficmind.search.models import RecordKind
    from trafficmind.search.models import SearchFilter
    from trafficmind.search.store import InMemorySearchStore
    from trafficmind.search.store import PlatformRecord

    reg = ModelConfigRegistry()
    reg.register_model(
        entry_id="det-plate",
        family=ModelFamily.DETECTION,
        task_type=TaskType.OBJECT_DETECTION,
        name="PlateDetector",
        version="1.4",
    )
    reg.register_model(
        entry_id="ocr-plate",
        family=ModelFamily.OCR,
        task_type=TaskType.PLATE_RECOGNITION,
        name="PaddleOCR",
        version="2.7",
    )
    provenance = reg.build_provenance_chain(
        output_id="PR-200",
        output_kind="plate_read",
        stamps=[
            reg.stamp("det-plate", confidence=0.91),
            reg.stamp("ocr-plate", confidence=0.87),
        ],
    )
    store = InMemorySearchStore()
    store.add(
        PlatformRecord(
            record_kind=RecordKind.PLATE_READ,
            record_id="PR-200",
            label="Plate read AB12XYZ",
            detail="ANPR capture.",
            plate_text="AB12XYZ",
            provenance=provenance,
        )
    )

    hit = store.query(
        SearchFilter(
            plate_text="AB12XYZ",
            record_kinds=(RecordKind.PLATE_READ,),
        )
    )[0]

    self.assertIsNotNone(hit.provenance)
    self.assertEqual(hit.provenance.output_kind, "plate_read")
    self.assertEqual(len(hit.provenance.stamps), 2)
    self.assertTrue(
        all(
            stamp.config_hash.startswith("cfg-")
            for stamp in hit.provenance.stamps
        )
    )


if __name__ == "__main__":
  unittest.main()
