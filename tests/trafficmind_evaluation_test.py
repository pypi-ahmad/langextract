"""Tests for TrafficMind evaluation artifact loading and HTML report rendering."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from trafficmind.evaluation import load_evaluation_artifacts
from trafficmind.evaluation import render_evaluation_report
from trafficmind.evaluation import write_evaluation_report
from trafficmind.evaluation.models import EvaluationArtifact
from trafficmind.evaluation.models import EvaluationSample
from trafficmind.evaluation.models import EvaluationSection
from trafficmind.evaluation.models import EvaluationSectionKind
from trafficmind.evaluation.models import ManualReviewSummary
from trafficmind.evaluation.models import MeasuredMetric
from trafficmind.evaluation.models import PlaceholderNotice
from trafficmind.evaluation.models import RegistryBinding
from trafficmind.evaluation.models import RuleValidationScenario
from trafficmind.registry.models import ModelFamily
from trafficmind.registry.models import TaskType


class TestEvaluationArtifactLoading(unittest.TestCase):

  def test_loads_artifacts_from_directory(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      artifact_path = root / "eval.json"
      artifact_path.write_text(
          json.dumps({
              "artifact_id": "artifact-a",
              "title": "Night regression",
              "captured_at": 1743948000,
              "camera_id": "CAM-7",
              "scenario_id": "night-rain",
              "pipeline_snapshot_id": "snap-night-1",
              "registry_bindings": [{
                  "entry_id": "det-yolo",
                  "entry_label": "YOLOv8n",
                  "entry_version": "8.0.1",
                  "config_hash": "cfg-det1234567890ab",
                  "task_type": "object_detection",
                  "family": "detection",
              }],
              "sections": [
                  {
                      "section_id": "det-main",
                      "kind": "detection_sanity",
                      "measured_metrics": [{
                          "name": "precision_at_iou_50",
                          "value": 0.91,
                          "unit": "ratio",
                          "sample_size": 120,
                      }],
                  },
                  {
                      "section_id": "workflow-main",
                      "kind": "workflow_summary",
                      "manual_summaries": [{
                          "title": "Operator spot check",
                          "summary": "Reviewers confirmed alert ordering.",
                          "reviewer": "ops-a",
                          "status": "pass",
                          "reviewed_at": 1743948600,
                      }],
                  },
              ],
          }),
          encoding="utf-8",
      )

      artifacts = load_evaluation_artifacts(root)

    self.assertEqual(len(artifacts), 1)
    artifact = artifacts[0]
    self.assertEqual(artifact.artifact_id, "artifact-a")
    self.assertEqual(artifact.source_path, str(artifact_path))
    self.assertEqual(
        artifact.registry_bindings[0].display_name,
        "YOLOv8n v8.0.1 [cfg-det1234567890ab]",
    )
    self.assertEqual(
        artifact.sections[0].kind, EvaluationSectionKind.DETECTION_SANITY
    )
    self.assertEqual(artifact.sections[1].manual_summaries[0].reviewer, "ops-a")


class TestEvaluationReportRendering(unittest.TestCase):

  def _artifact_payload(self):
    return {
        "artifact_id": "artifact-b",
        "title": "Mixed camera regression",
        "captured_at": 1743948000,
        "camera_id": "CAM-3",
        "scenario_id": "junction-4-daylight",
        "pipeline_snapshot_id": "snap-prod-2",
        "registry_bindings": [
            {
                "entry_id": "det-yolo",
                "entry_label": "YOLOv8n",
                "entry_version": "8.0.1",
                "config_hash": "cfg-det1234567890ab",
                "task_type": "object_detection",
                "family": "detection",
            },
            {
                "entry_id": "ocr-paddle",
                "entry_label": "PaddleOCR",
                "entry_version": "2.7",
                "config_hash": "cfg-ocr1234567890ab",
                "task_type": "plate_recognition",
                "family": "ocr",
            },
        ],
        "sections": [
            {
                "section_id": "det-main",
                "kind": "detection_sanity",
                "summary_text": "Fixture-based daytime sanity set.",
                "measured_metrics": [{
                    "name": "recall_at_iou_50",
                    "value": 0.94,
                    "unit": "ratio",
                    "sample_size": 180,
                    "note": "Loaded from local regression export.",
                }],
            },
            {
                "section_id": "ocr-main",
                "kind": "ocr_quality",
                "task_type": "plate_recognition",
                "samples": [{
                    "sample_id": "plate-1",
                    "label": "Plate crop 001",
                    "expected_value": "AB12XYZ",
                    "observed_value": "AB12XY2",
                    "score": 0.86,
                    "passed": False,
                    "note": "Night glare on final character.",
                }],
                "manual_summaries": [{
                    "title": "Manual OCR review",
                    "summary": "Most failures came from reflective plates.",
                    "reviewer": "qa-1",
                    "status": "needs_review",
                }],
            },
            {
                "section_id": "rules-main",
                "kind": "rule_validation",
                "task_type": "rule_evaluation",
                "registry_bindings": [{
                    "entry_id": "rules-violations",
                    "entry_label": "traffic-violations",
                    "entry_version": "2.0",
                    "config_hash": "cfg-rule1234567890",
                    "task_type": "rule_evaluation",
                    "family": "rules",
                }],
                "validation_scenarios": [{
                    "scenario_id": "rl-001",
                    "title": "Red light baseline",
                    "expected_outcome": "violation",
                    "actual_outcome": "violation",
                    "passed": True,
                    "note": "Fixture scenario from deterministic rule suite.",
                }],
            },
        ],
    }

  def test_rendered_report_contains_filters_and_separation_labels(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      artifact_path = Path(tmp_dir) / "artifact.json"
      artifact_path.write_text(
          json.dumps(self._artifact_payload()), encoding="utf-8"
      )
      artifacts = load_evaluation_artifacts(artifact_path)

    html = render_evaluation_report(artifacts)

    self.assertIn("Model / Config Version", html)
    self.assertIn("Camera", html)
    self.assertIn("Test Scenario", html)
    self.assertIn("Task Type", html)
    self.assertIn("Measured Metrics", html)
    self.assertIn("Manual Review Summaries", html)
    self.assertIn("Not Yet Available", html)
    self.assertIn("YOLOv8n v8.0.1 [cfg-det1234567890ab]", html)
    self.assertIn("PaddleOCR v2.7 [cfg-ocr1234567890ab]", html)
    self.assertIn("traffic-violations v2.0 [cfg-rule1234567890]", html)
    self.assertIn("Plate crop 001", html)
    self.assertIn("Rule Validation Scenarios", html)
    self.assertIn(
        "This admin report renders only locally loaded evaluation artifacts",
        html,
    )
    self.assertIn(str(artifact_path), html)

  def test_missing_section_kinds_render_placeholder_cards(self):
    payload = self._artifact_payload()
    payload["sections"] = payload["sections"][:1]
    with tempfile.TemporaryDirectory() as tmp_dir:
      artifact_path = Path(tmp_dir) / "artifact.json"
      artifact_path.write_text(json.dumps(payload), encoding="utf-8")
      artifacts = load_evaluation_artifacts(artifact_path)

    html = render_evaluation_report(artifacts)

    self.assertIn("Tracking Consistency Checks", html)
    self.assertIn("OCR Quality Samples", html)
    self.assertIn("Workflow / Evaluation Summaries", html)
    self.assertIn("No local results loaded", html)
    self.assertIn(
        "This section has no measured artifact data or stored review summary",
        html,
    )


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


def _make_binding(**overrides: object) -> RegistryBinding:
  defaults = {
      "entry_id": "test-id",
      "entry_label": "TestModel",
      "entry_version": "1.0",
      "config_hash": "cfg-abc1234567890a",
      "task_type": TaskType.OBJECT_DETECTION,
      "family": ModelFamily.DETECTION,
  }
  return RegistryBinding(**(defaults | overrides))  # type: ignore[arg-type]


def _make_section(**overrides: object) -> EvaluationSection:
  defaults = {
      "section_id": "sec-1",
      "kind": EvaluationSectionKind.DETECTION_SANITY,
      "task_type": TaskType.OBJECT_DETECTION,
  }
  return EvaluationSection(**(defaults | overrides))  # type: ignore[arg-type]


class TestModelValidation(unittest.TestCase):
  """Validation rules encoded in __post_init__ of model dataclasses."""

  # ---- RegistryBinding ----

  def test_binding_display_name(self):
    binding = _make_binding()
    self.assertEqual(
        binding.display_name, "TestModel v1.0 [cfg-abc1234567890a]"
    )

  def test_binding_empty_entry_id_rejected(self):
    with self.assertRaises(ValueError, msg="entry_id must be non-empty"):
      _make_binding(entry_id="")

  def test_binding_empty_entry_label_rejected(self):
    with self.assertRaises(ValueError):
      _make_binding(entry_label="")

  def test_binding_empty_entry_version_rejected(self):
    with self.assertRaises(ValueError):
      _make_binding(entry_version="")

  def test_binding_empty_config_hash_rejected(self):
    with self.assertRaises(ValueError):
      _make_binding(config_hash="")

  # ---- MeasuredMetric ----

  def test_metric_valid_construction(self):
    metric = MeasuredMetric(
        name="precision", value=0.95, unit="ratio", sample_size=200
    )
    self.assertEqual(metric.value, 0.95)
    self.assertEqual(metric.sample_size, 200)

  def test_metric_empty_name_rejected(self):
    with self.assertRaises(ValueError):
      MeasuredMetric(name="", value=0.5)

  def test_metric_boolean_value_rejected(self):
    with self.assertRaises(TypeError):
      MeasuredMetric(name="flag", value=True)  # type: ignore[arg-type]

  def test_metric_negative_sample_size_rejected(self):
    with self.assertRaises(ValueError):
      MeasuredMetric(name="recall", value=0.8, sample_size=-1)

  def test_metric_zero_sample_size_accepted(self):
    metric = MeasuredMetric(name="recall", value=0.8, sample_size=0)
    self.assertEqual(metric.sample_size, 0)

  # ---- ManualReviewSummary ----

  def test_manual_summary_empty_title_rejected(self):
    with self.assertRaises(ValueError):
      ManualReviewSummary(title="", summary="ok")

  def test_manual_summary_empty_summary_rejected(self):
    with self.assertRaises(ValueError):
      ManualReviewSummary(title="Check", summary="")

  # ---- EvaluationSample ----

  def test_sample_score_below_zero_rejected(self):
    with self.assertRaises(ValueError):
      EvaluationSample(sample_id="s1", label="test", score=-0.01)

  def test_sample_score_above_one_rejected(self):
    with self.assertRaises(ValueError):
      EvaluationSample(sample_id="s1", label="test", score=1.01)

  def test_sample_score_bounds_accepted(self):
    s_low = EvaluationSample(sample_id="s1", label="test", score=0.0)
    s_high = EvaluationSample(sample_id="s2", label="test", score=1.0)
    self.assertEqual(s_low.score, 0.0)
    self.assertEqual(s_high.score, 1.0)

  def test_sample_empty_id_rejected(self):
    with self.assertRaises(ValueError):
      EvaluationSample(sample_id="", label="test")

  def test_sample_empty_label_rejected(self):
    with self.assertRaises(ValueError):
      EvaluationSample(sample_id="s1", label="")

  # ---- RuleValidationScenario ----

  def test_scenario_empty_scenario_id_rejected(self):
    with self.assertRaises(ValueError):
      RuleValidationScenario(
          scenario_id="",
          title="t",
          expected_outcome="a",
          actual_outcome="a",
          passed=True,
      )

  def test_scenario_empty_title_rejected(self):
    with self.assertRaises(ValueError):
      RuleValidationScenario(
          scenario_id="sc",
          title="",
          expected_outcome="a",
          actual_outcome="a",
          passed=True,
      )

  def test_scenario_empty_expected_rejected(self):
    with self.assertRaises(ValueError):
      RuleValidationScenario(
          scenario_id="sc",
          title="t",
          expected_outcome="",
          actual_outcome="a",
          passed=True,
      )

  def test_scenario_empty_actual_rejected(self):
    with self.assertRaises(ValueError):
      RuleValidationScenario(
          scenario_id="sc",
          title="t",
          expected_outcome="a",
          actual_outcome="",
          passed=True,
      )

  # ---- PlaceholderNotice ----

  def test_placeholder_empty_title_rejected(self):
    with self.assertRaises(ValueError):
      PlaceholderNotice(title="", detail="some detail")

  def test_placeholder_empty_detail_rejected(self):
    with self.assertRaises(ValueError):
      PlaceholderNotice(title="heading", detail="")

  # ---- EvaluationSection ----

  def test_section_empty_id_rejected(self):
    with self.assertRaises(ValueError):
      _make_section(section_id="")

  def test_section_auto_title_from_kind(self):
    section = _make_section(
        title="", placeholder=PlaceholderNotice(title="P", detail="D")
    )
    self.assertEqual(section.title, "Detection Sanity Metrics")

  def test_section_explicit_title_preserved(self):
    section = _make_section(
        title="Custom Title",
        placeholder=PlaceholderNotice(title="P", detail="D"),
    )
    self.assertEqual(section.title, "Custom Title")

  def test_section_tuples_coerced_from_lists(self):
    section = _make_section(
        measured_metrics=[MeasuredMetric(name="x", value=1)],
        samples=[EvaluationSample(sample_id="s1", label="l")],
    )
    self.assertIsInstance(section.measured_metrics, tuple)
    self.assertIsInstance(section.samples, tuple)


# ---------------------------------------------------------------------------
# IO error-path tests
# ---------------------------------------------------------------------------


class TestIOErrorPaths(unittest.TestCase):

  def _write_json(self, tmp_dir: Path, name: str, payload: object) -> Path:
    path = tmp_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

  def _minimal_artifact(self, **overrides: object) -> dict:
    defaults: dict = {
        "artifact_id": "a1",
        "title": "Test",
        "captured_at": 1700000000,
        "sections": [],
    }
    return {**defaults, **overrides}

  def test_missing_path_raises_file_not_found(self):
    with self.assertRaises(FileNotFoundError):
      load_evaluation_artifacts(Path("/nonexistent/path/does_not_exist.json"))

  def test_invalid_json_raises_error(self):
    with tempfile.TemporaryDirectory() as tmp:
      bad = Path(tmp) / "bad.json"
      bad.write_text("{not valid json !!!}", encoding="utf-8")
      with self.assertRaises(json.JSONDecodeError):
        load_evaluation_artifacts(bad)

  def test_non_object_entry_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      p = self._write_json(Path(tmp), "bad.json", ["not an object"])
      with self.assertRaises(TypeError):
        load_evaluation_artifacts(p)

  def test_missing_required_field_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      p = self._write_json(Path(tmp), "bad.json", {"title": "T"})
      with self.assertRaises(TypeError):
        load_evaluation_artifacts(p)

  def test_boolean_as_captured_at_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      payload = self._minimal_artifact(captured_at=True)
      p = self._write_json(Path(tmp), "bad.json", payload)
      with self.assertRaises(TypeError):
        load_evaluation_artifacts(p)

  def test_invalid_section_kind_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      payload = self._minimal_artifact(
          sections=[{"section_id": "s1", "kind": "not_a_real_kind"}]
      )
      p = self._write_json(Path(tmp), "bad.json", payload)
      with self.assertRaises(ValueError):
        load_evaluation_artifacts(p)

  def test_empty_directory_returns_empty_tuple(self):
    with tempfile.TemporaryDirectory() as tmp:
      artifacts = load_evaluation_artifacts(Path(tmp))
    self.assertEqual(artifacts, ())

  def test_list_payload_loads_multiple_artifacts(self):
    with tempfile.TemporaryDirectory() as tmp:
      a1 = self._minimal_artifact(artifact_id="a1")
      a2 = self._minimal_artifact(artifact_id="a2")
      p = self._write_json(Path(tmp), "multi.json", [a1, a2])
      artifacts = load_evaluation_artifacts(p)
    self.assertEqual(len(artifacts), 2)
    ids = {a.artifact_id for a in artifacts}
    self.assertEqual(ids, {"a1", "a2"})

  def test_placeholder_parsing_from_json(self):
    with tempfile.TemporaryDirectory() as tmp:
      payload = self._minimal_artifact(
          sections=[{
              "section_id": "s1",
              "kind": "detection_sanity",
              "placeholder": {
                  "title": "Pending",
                  "detail": "Awaiting test run.",
              },
          }]
      )
      p = self._write_json(Path(tmp), "ph.json", payload)
      artifacts = load_evaluation_artifacts(p)
    ph = artifacts[0].sections[0].placeholder
    self.assertIsNotNone(ph)
    self.assertEqual(ph.title, "Pending")

  def test_invalid_placeholder_type_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      payload = self._minimal_artifact(
          sections=[{
              "section_id": "s1",
              "kind": "detection_sanity",
              "placeholder": "not_an_object",
          }]
      )
      p = self._write_json(Path(tmp), "bad.json", payload)
      with self.assertRaises(TypeError):
        load_evaluation_artifacts(p)

  def test_validation_scenario_non_boolean_passed_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      payload = self._minimal_artifact(
          sections=[{
              "section_id": "s1",
              "kind": "rule_validation",
              "task_type": "rule_evaluation",
              "validation_scenarios": [{
                  "scenario_id": "sc1",
                  "title": "Test",
                  "expected_outcome": "a",
                  "actual_outcome": "a",
                  "passed": "yes",
              }],
          }]
      )
      p = self._write_json(Path(tmp), "bad.json", payload)
      with self.assertRaises(TypeError):
        load_evaluation_artifacts(p)

  def test_source_path_set_on_loaded_artifact(self):
    with tempfile.TemporaryDirectory() as tmp:
      payload = self._minimal_artifact()
      p = self._write_json(Path(tmp), "eval.json", payload)
      artifacts = load_evaluation_artifacts(p)
    self.assertEqual(artifacts[0].source_path, str(p))

  def test_artifacts_sorted_newest_first(self):
    with tempfile.TemporaryDirectory() as tmp:
      a_old = self._minimal_artifact(artifact_id="old", captured_at=1000)
      a_new = self._minimal_artifact(artifact_id="new", captured_at=2000)
      self._write_json(Path(tmp), "old.json", a_old)
      self._write_json(Path(tmp), "new.json", a_new)
      artifacts = load_evaluation_artifacts(Path(tmp))
    self.assertEqual(artifacts[0].artifact_id, "new")
    self.assertEqual(artifacts[1].artifact_id, "old")


# ---------------------------------------------------------------------------
# Report edge-case tests
# ---------------------------------------------------------------------------


class TestReportEdgeCases(unittest.TestCase):

  def _load_from_payload(self, payload: dict) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
      p = Path(tmp) / "a.json"
      p.write_text(json.dumps(payload), encoding="utf-8")
      return load_evaluation_artifacts(p)

  def _minimal_payload(self, **overrides: object) -> dict:
    defaults: dict = {
        "artifact_id": "a1",
        "title": "Test Artifact",
        "captured_at": 1700000000,
        "sections": [],
    }
    return {**defaults, **overrides}

  def test_write_evaluation_report_writes_file(self):
    artifacts = self._load_from_payload(self._minimal_payload())
    with tempfile.TemporaryDirectory() as tmp:
      out = Path(tmp) / "sub" / "report.html"
      result = write_evaluation_report(artifacts, out)
      self.assertTrue(result.exists())
      content = result.read_text(encoding="utf-8")
      self.assertIn("<!doctype html", content)
      self.assertIn("TrafficMind Evaluation Report", content)

  def test_empty_artifacts_renders_no_artifacts_notice(self):
    html = render_evaluation_report(())
    self.assertIn("No local artifact files were loaded", html)

  def test_custom_title_appears_in_html(self):
    artifacts = self._load_from_payload(self._minimal_payload())
    html = render_evaluation_report(artifacts, title="Night Regression")
    self.assertIn("Night Regression", html)

  def test_json_payload_escapes_script_tags(self):
    artifacts = self._load_from_payload(
        self._minimal_payload(title="</script><h1>xss</h1>")
    )
    html = render_evaluation_report(artifacts)
    self.assertNotIn("</script><h1>xss</h1>", html)
    self.assertIn("<\\/script>", html)

  def test_tracking_consistency_section_renders(self):
    payload = self._minimal_payload(
        sections=[{
            "section_id": "track-1",
            "kind": "tracking_consistency",
            "task_type": "vehicle_tracking",
            "measured_metrics": [{
                "name": "id_switch_rate",
                "value": 0.02,
                "unit": "ratio",
                "sample_size": 50,
            }],
            "samples": [{
                "sample_id": "trk-s1",
                "label": "Track continuity clip A",
                "expected_value": "continuous",
                "observed_value": "1 break",
                "score": 0.9,
                "passed": False,
            }],
        }]
    )
    artifacts = self._load_from_payload(payload)
    html = render_evaluation_report(artifacts)
    self.assertIn("Tracking Consistency Checks", html)
    self.assertIn("id_switch_rate", html)
    self.assertIn("Track continuity clip A", html)

  def test_section_kind_filter_present(self):
    artifacts = self._load_from_payload(self._minimal_payload())
    html = render_evaluation_report(artifacts)
    self.assertIn("kindFilter", html)
    self.assertIn("Section Kind", html)

  def test_reset_filters_button_present(self):
    artifacts = self._load_from_payload(self._minimal_payload())
    html = render_evaluation_report(artifacts)
    self.assertIn("resetFilters", html)
    self.assertIn("Reset Filters", html)


if __name__ == "__main__":
  unittest.main()
