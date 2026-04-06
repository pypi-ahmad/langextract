"""Load TrafficMind evaluation artifacts from local JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from trafficmind.evaluation.models import EvaluationArtifact
from trafficmind.evaluation.models import EvaluationSample
from trafficmind.evaluation.models import EvaluationSection
from trafficmind.evaluation.models import EvaluationSectionKind
from trafficmind.evaluation.models import ManualReviewSummary
from trafficmind.evaluation.models import MeasuredMetric
from trafficmind.evaluation.models import PlaceholderNotice
from trafficmind.evaluation.models import RegistryBinding
from trafficmind.evaluation.models import RuleValidationScenario
from trafficmind.evaluation.models import SECTION_DEFAULT_TASK_TYPES
from trafficmind.registry.models import ModelFamily
from trafficmind.registry.models import TaskType


def load_evaluation_artifacts(
    paths: str | Path | Sequence[str | Path],
) -> tuple[EvaluationArtifact, ...]:
  """Load evaluation artifacts from files or directories.

  Each JSON file may contain either a single artifact object or a list of
  artifact objects.
  """
  if isinstance(paths, (str, Path)):
    raw_paths: list[str | Path] = [paths]
  else:
    raw_paths = list(paths)
  files = _expand_paths(raw_paths)
  artifacts: list[EvaluationArtifact] = []
  for file_path in files:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    payload_items = payload if isinstance(payload, list) else [payload]
    if not isinstance(payload_items, list):
      raise TypeError(
          f"Artifact file {file_path} must contain an object or list"
      )
    for item in payload_items:
      if not isinstance(item, dict):
        raise TypeError(f"Artifact entry in {file_path} must be a JSON object")
      artifacts.append(_parse_artifact(item, source_path=str(file_path)))
  artifacts.sort(
      key=lambda artifact: (artifact.captured_at, artifact.artifact_id),
      reverse=True,
  )
  return tuple(artifacts)


def _expand_paths(raw_paths: Sequence[str | Path]) -> list[Path]:
  files: list[Path] = []
  for raw in raw_paths:
    path = Path(raw)
    if not path.exists():
      raise FileNotFoundError(f"Evaluation artifact path not found: {path}")
    if path.is_dir():
      files.extend(sorted(path.glob("*.json")))
    else:
      files.append(path)
  return files


def _parse_artifact(
    raw: dict[str, Any], *, source_path: str
) -> EvaluationArtifact:
  sections = tuple(_parse_section(item) for item in _get_list(raw, "sections"))
  return EvaluationArtifact(
      artifact_id=_get_required_str(raw, "artifact_id"),
      title=_get_required_str(raw, "title"),
      captured_at=_get_required_number(raw, "captured_at"),
      source_path=source_path,
      camera_id=_get_optional_str(raw, "camera_id"),
      scenario_id=_get_optional_str(raw, "scenario_id"),
      pipeline_snapshot_id=_get_optional_str(raw, "pipeline_snapshot_id"),
      registry_bindings=tuple(
          _parse_registry_binding(item)
          for item in _get_list(raw, "registry_bindings")
      ),
      sections=sections,
      metadata=_get_optional_mapping(raw, "metadata"),
  )


def _parse_section(raw: dict[str, Any]) -> EvaluationSection:
  kind = EvaluationSectionKind(_get_required_str(raw, "kind"))
  task_type = TaskType(
      _get_optional_str(raw, "task_type")
      or SECTION_DEFAULT_TASK_TYPES[kind].value
  )
  return EvaluationSection(
      section_id=_get_required_str(raw, "section_id"),
      kind=kind,
      task_type=task_type,
      title=_get_optional_str(raw, "title") or "",
      summary_text=_get_optional_str(raw, "summary_text") or "",
      registry_bindings=tuple(
          _parse_registry_binding(item)
          for item in _get_list(raw, "registry_bindings")
      ),
      measured_metrics=tuple(
          _parse_measured_metric(item)
          for item in _get_list(raw, "measured_metrics")
      ),
      manual_summaries=tuple(
          _parse_manual_summary(item)
          for item in _get_list(raw, "manual_summaries")
      ),
      samples=tuple(_parse_sample(item) for item in _get_list(raw, "samples")),
      validation_scenarios=tuple(
          _parse_validation_scenario(item)
          for item in _get_list(raw, "validation_scenarios")
      ),
      placeholder=_parse_placeholder(raw.get("placeholder")),
      camera_id=_get_optional_str(raw, "camera_id"),
      scenario_id=_get_optional_str(raw, "scenario_id"),
      date_start=_get_optional_number(raw, "date_start"),
      date_end=_get_optional_number(raw, "date_end"),
      metadata=_get_optional_mapping(raw, "metadata"),
  )


def _parse_registry_binding(raw: dict[str, Any]) -> RegistryBinding:
  return RegistryBinding(
      entry_id=_get_required_str(raw, "entry_id"),
      entry_label=_get_required_str(raw, "entry_label"),
      entry_version=_get_required_str(raw, "entry_version"),
      config_hash=_get_required_str(raw, "config_hash"),
      task_type=TaskType(_get_required_str(raw, "task_type")),
      family=ModelFamily(_get_required_str(raw, "family")),
      metadata=_get_optional_mapping(raw, "metadata"),
  )


def _parse_measured_metric(raw: dict[str, Any]) -> MeasuredMetric:
  return MeasuredMetric(
      name=_get_required_str(raw, "name"),
      value=_get_required_number(raw, "value"),
      unit=_get_optional_str(raw, "unit") or "",
      sample_size=_get_optional_int(raw, "sample_size"),
      note=_get_optional_str(raw, "note") or "",
  )


def _parse_manual_summary(raw: dict[str, Any]) -> ManualReviewSummary:
  return ManualReviewSummary(
      title=_get_required_str(raw, "title"),
      summary=_get_required_str(raw, "summary"),
      reviewer=_get_optional_str(raw, "reviewer"),
      status=_get_optional_str(raw, "status") or "",
      reviewed_at=_get_optional_number(raw, "reviewed_at"),
  )


def _parse_sample(raw: dict[str, Any]) -> EvaluationSample:
  return EvaluationSample(
      sample_id=_get_required_str(raw, "sample_id"),
      label=_get_required_str(raw, "label"),
      expected_value=_get_optional_str(raw, "expected_value"),
      observed_value=_get_optional_str(raw, "observed_value"),
      score=_get_optional_number(raw, "score"),
      passed=_get_optional_bool(raw, "passed"),
      note=_get_optional_str(raw, "note") or "",
      media_reference=_get_optional_str(raw, "media_reference"),
  )


def _parse_validation_scenario(raw: dict[str, Any]) -> RuleValidationScenario:
  passed = raw.get("passed")
  if not isinstance(passed, bool):
    raise TypeError("passed must be a boolean for validation scenarios")
  return RuleValidationScenario(
      scenario_id=_get_required_str(raw, "scenario_id"),
      title=_get_required_str(raw, "title"),
      expected_outcome=_get_required_str(raw, "expected_outcome"),
      actual_outcome=_get_required_str(raw, "actual_outcome"),
      passed=passed,
      note=_get_optional_str(raw, "note") or "",
  )


def _parse_placeholder(raw: Any) -> PlaceholderNotice | None:
  if raw is None:
    return None
  if not isinstance(raw, dict):
    raise TypeError("placeholder must be an object when provided")
  return PlaceholderNotice(
      title=_get_required_str(raw, "title"),
      detail=_get_required_str(raw, "detail"),
  )


def _get_required_str(raw: dict[str, Any], key: str) -> str:
  value = raw.get(key)
  if not isinstance(value, str) or not value:
    raise TypeError(f"{key} must be a non-empty string")
  return value


def _get_optional_str(raw: dict[str, Any], key: str) -> str | None:
  value = raw.get(key)
  if value is None:
    return None
  if not isinstance(value, str):
    raise TypeError(f"{key} must be a string when provided")
  return value


def _get_required_number(raw: dict[str, Any], key: str) -> float | int:
  value = raw.get(key)
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{key} must be numeric")
  return value


def _get_optional_number(raw: dict[str, Any], key: str) -> float | int | None:
  value = raw.get(key)
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{key} must be numeric when provided")
  return value


def _get_optional_int(raw: dict[str, Any], key: str) -> int | None:
  value = raw.get(key)
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{key} must be an integer when provided")
  return value


def _get_optional_bool(raw: dict[str, Any], key: str) -> bool | None:
  value = raw.get(key)
  if value is None:
    return None
  if not isinstance(value, bool):
    raise TypeError(f"{key} must be a boolean when provided")
  return value


def _get_optional_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
  value = raw.get(key)
  if value is None:
    return {}
  if not isinstance(value, dict):
    raise TypeError(f"{key} must be an object when provided")
  return dict(value)


def _get_list(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
  value = raw.get(key)
  if value is None:
    return []
  if not isinstance(value, list):
    raise TypeError(f"{key} must be a list when provided")
  output: list[dict[str, Any]] = []
  for item in value:
    if not isinstance(item, dict):
      raise TypeError(f"{key} entries must be objects")
    output.append(item)
  return output
