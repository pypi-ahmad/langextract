"""Typed evaluation artifact models for TrafficMind benchmark and sanity views."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Any

from trafficmind.registry.models import ModelFamily
from trafficmind.registry.models import TaskType


class EvaluationSectionKind(str, enum.Enum):
  """Top-level sections shown in the evaluation admin view."""

  DETECTION_SANITY = "detection_sanity"
  TRACKING_CONSISTENCY = "tracking_consistency"
  OCR_QUALITY = "ocr_quality"
  RULE_VALIDATION = "rule_validation"
  WORKFLOW_SUMMARY = "workflow_summary"


SECTION_TITLES: dict[EvaluationSectionKind, str] = {
    EvaluationSectionKind.DETECTION_SANITY: "Detection Sanity Metrics",
    EvaluationSectionKind.TRACKING_CONSISTENCY: "Tracking Consistency Checks",
    EvaluationSectionKind.OCR_QUALITY: "OCR Quality Samples",
    EvaluationSectionKind.RULE_VALIDATION: "Rule Validation Scenarios",
    EvaluationSectionKind.WORKFLOW_SUMMARY: "Workflow / Evaluation Summaries",
}

SECTION_DEFAULT_TASK_TYPES: dict[EvaluationSectionKind, TaskType] = {
    EvaluationSectionKind.DETECTION_SANITY: TaskType.OBJECT_DETECTION,
    EvaluationSectionKind.TRACKING_CONSISTENCY: TaskType.VEHICLE_TRACKING,
    EvaluationSectionKind.OCR_QUALITY: TaskType.PLATE_RECOGNITION,
    EvaluationSectionKind.RULE_VALIDATION: TaskType.RULE_EVALUATION,
    EvaluationSectionKind.WORKFLOW_SUMMARY: TaskType.OTHER,
}


@dataclass(frozen=True)
class RegistryBinding:
  """Registry identity attached to an evaluation section or artifact."""

  entry_id: str
  entry_label: str
  entry_version: str
  config_hash: str
  task_type: TaskType
  family: ModelFamily
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.entry_id:
      raise ValueError("entry_id must be non-empty")
    if not self.entry_label:
      raise ValueError("entry_label must be non-empty")
    if not self.entry_version:
      raise ValueError("entry_version must be non-empty")
    if not self.config_hash:
      raise ValueError("config_hash must be non-empty")

  @property
  def display_name(self) -> str:
    return f"{self.entry_label} v{self.entry_version} [{self.config_hash}]"


@dataclass(frozen=True)
class MeasuredMetric:
  """A real measured metric loaded from an artifact."""

  name: str
  value: float | int
  unit: str = ""
  sample_size: int | None = None
  note: str = ""

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("name must be non-empty")
    if isinstance(self.value, bool):
      raise TypeError("value must be numeric, not boolean")
    if self.sample_size is not None and self.sample_size < 0:
      raise ValueError("sample_size must be >= 0")


@dataclass(frozen=True)
class ManualReviewSummary:
  """A human-authored qualitative summary shown separately from metrics."""

  title: str
  summary: str
  reviewer: str | None = None
  status: str = ""
  reviewed_at: float | None = None

  def __post_init__(self) -> None:
    if not self.title:
      raise ValueError("title must be non-empty")
    if not self.summary:
      raise ValueError("summary must be non-empty")


@dataclass(frozen=True)
class EvaluationSample:
  """A concrete sample row, typically used for OCR or tracking evidence."""

  sample_id: str
  label: str
  expected_value: str | None = None
  observed_value: str | None = None
  score: float | None = None
  passed: bool | None = None
  note: str = ""
  media_reference: str | None = None

  def __post_init__(self) -> None:
    if not self.sample_id:
      raise ValueError("sample_id must be non-empty")
    if not self.label:
      raise ValueError("label must be non-empty")
    if self.score is not None and not 0.0 <= self.score <= 1.0:
      raise ValueError("score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RuleValidationScenario:
  """A deterministic rule validation scenario."""

  scenario_id: str
  title: str
  expected_outcome: str
  actual_outcome: str
  passed: bool
  note: str = ""

  def __post_init__(self) -> None:
    if not self.scenario_id:
      raise ValueError("scenario_id must be non-empty")
    if not self.title:
      raise ValueError("title must be non-empty")
    if not self.expected_outcome:
      raise ValueError("expected_outcome must be non-empty")
    if not self.actual_outcome:
      raise ValueError("actual_outcome must be non-empty")


@dataclass(frozen=True)
class PlaceholderNotice:
  """Honest explanation for a section that is not yet backed by local data."""

  title: str
  detail: str

  def __post_init__(self) -> None:
    if not self.title:
      raise ValueError("title must be non-empty")
    if not self.detail:
      raise ValueError("detail must be non-empty")


@dataclass(frozen=True)
class EvaluationSection:
  """One reportable evaluation section within an artifact."""

  section_id: str
  kind: EvaluationSectionKind
  task_type: TaskType
  title: str = ""
  summary_text: str = ""
  registry_bindings: tuple[RegistryBinding, ...] = ()
  measured_metrics: tuple[MeasuredMetric, ...] = ()
  manual_summaries: tuple[ManualReviewSummary, ...] = ()
  samples: tuple[EvaluationSample, ...] = ()
  validation_scenarios: tuple[RuleValidationScenario, ...] = ()
  placeholder: PlaceholderNotice | None = None
  camera_id: str | None = None
  scenario_id: str | None = None
  date_start: float | None = None
  date_end: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.section_id:
      raise ValueError("section_id must be non-empty")
    if not self.title:
      object.__setattr__(self, "title", SECTION_TITLES[self.kind])
    object.__setattr__(self, "registry_bindings", tuple(self.registry_bindings))
    object.__setattr__(self, "measured_metrics", tuple(self.measured_metrics))
    object.__setattr__(self, "manual_summaries", tuple(self.manual_summaries))
    object.__setattr__(self, "samples", tuple(self.samples))
    object.__setattr__(
        self, "validation_scenarios", tuple(self.validation_scenarios)
    )
    if (
        self.date_start is not None
        and self.date_end is not None
        and self.date_end < self.date_start
    ):
      raise ValueError("date_end must be >= date_start")
    if not any((
        self.summary_text,
        self.measured_metrics,
        self.manual_summaries,
        self.samples,
        self.validation_scenarios,
        self.placeholder is not None,
    )):
      raise ValueError("section must contain data or a placeholder notice")

  @property
  def has_measured_data(self) -> bool:
    return bool(
        self.measured_metrics or self.samples or self.validation_scenarios
    )

  @property
  def has_manual_review(self) -> bool:
    return bool(self.manual_summaries)


@dataclass(frozen=True)
class EvaluationArtifact:
  """One local evaluation artifact or stored result export."""

  artifact_id: str
  title: str
  captured_at: float
  source_path: str = ""
  camera_id: str | None = None
  scenario_id: str | None = None
  pipeline_snapshot_id: str | None = None
  registry_bindings: tuple[RegistryBinding, ...] = ()
  sections: tuple[EvaluationSection, ...] = ()
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.artifact_id:
      raise ValueError("artifact_id must be non-empty")
    if not self.title:
      raise ValueError("title must be non-empty")
    object.__setattr__(self, "registry_bindings", tuple(self.registry_bindings))
    object.__setattr__(self, "sections", tuple(self.sections))
