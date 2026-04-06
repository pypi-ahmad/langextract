"""Typed models for model/config registration and provenance tracking.

This module provides a lightweight registry foundation for recording which
models, thresholds, rule configurations, and pipeline settings produced a
given output (event, violation, plate read, evidence frame). It is designed
for audit, debugging, and future experiment comparison — not as a full ML
platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
import hashlib
import json
import time
from typing import Any
import uuid

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelFamily(str, enum.Enum):
  """Broad category of a registered model or algorithm."""

  DETECTION = "detection"
  OCR = "ocr"
  TRACKING = "tracking"
  CLASSIFICATION = "classification"
  RULES = "rules"
  OTHER = "other"


class TaskType(str, enum.Enum):
  """Specific task a registered entry addresses."""

  OBJECT_DETECTION = "object_detection"
  PLATE_RECOGNITION = "plate_recognition"
  FACE_DETECTION = "face_detection"
  VEHICLE_TRACKING = "vehicle_tracking"
  SIGNAL_STATE_CLASSIFICATION = "signal_state_classification"
  RULE_EVALUATION = "rule_evaluation"
  SPEED_ESTIMATION = "speed_estimation"
  OTHER = "other"


class EntryStatus(str, enum.Enum):
  """Lifecycle status of a registry entry."""

  ACTIVE = "active"
  INACTIVE = "inactive"
  EXPERIMENTAL = "experimental"
  DEPRECATED = "deprecated"


class RegistryEntryKind(str, enum.Enum):
  """Concrete registry entry type referenced by provenance."""

  MODEL = "model"
  RULES = "rules"


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------


def _stable_json(value: Any) -> str:
  """Deterministic JSON for hashing."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ConfigBundle:
  """Immutable snapshot of a configuration / threshold / parameter set.

  ``config_hash`` is derived automatically from ``parameters`` so two
  bundles with identical parameters produce the same hash regardless of
  insertion order.
  """

  parameters: dict[str, Any] = field(default_factory=dict)
  config_hash: str = ""

  def __post_init__(self) -> None:
    if not self.config_hash:
      digest = hashlib.sha256(
          _stable_json(self.parameters).encode()
      ).hexdigest()[:16]
      object.__setattr__(self, "config_hash", f"cfg-{digest}")


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRegistryEntry:
  """One registered model or algorithm version.

  This is a provenance record, not a runtime model loader. It captures
  enough to answer *"which model and thresholds produced this output?"*
  """

  entry_id: str
  family: ModelFamily
  task_type: TaskType
  name: str
  version: str
  config: ConfigBundle = field(default_factory=ConfigBundle)
  status: EntryStatus = EntryStatus.ACTIVE
  notes: str = ""
  registered_at: float = field(default_factory=time.time)
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.entry_id:
      raise ValueError("entry_id must be non-empty")
    if not self.name:
      raise ValueError("name must be non-empty")
    if not self.version:
      raise ValueError("version must be non-empty")


@dataclass(frozen=True)
class RulesRegistryEntry:
  """One registered rule-configuration version.

  Separate from ``ModelRegistryEntry`` because rules are deterministic
  logic, not learned models — but they still need version tracking for
  audit and reproducibility.
  """

  entry_id: str
  rule_set_id: str
  version: str
  description: str = ""
  rule_ids: tuple[str, ...] = ()
  config: ConfigBundle = field(default_factory=ConfigBundle)
  status: EntryStatus = EntryStatus.ACTIVE
  registered_at: float = field(default_factory=time.time)
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.entry_id:
      raise ValueError("entry_id must be non-empty")
    if not self.rule_set_id:
      raise ValueError("rule_set_id must be non-empty")
    if not self.version:
      raise ValueError("version must be non-empty")
    object.__setattr__(self, "rule_ids", tuple(self.rule_ids))


# ---------------------------------------------------------------------------
# Provenance stamps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceStamp:
  """Links an output (event, evidence, plate read, etc.) to its producing
  registry entry.

  A single output can carry multiple stamps when it was produced by a
  pipeline of models (e.g. detection → OCR → rule evaluation).
  """

  stamp_id: str = field(default_factory=lambda: f"ps-{uuid.uuid4().hex[:12]}")
  registry_entry_id: str = ""
  entry_kind: RegistryEntryKind | None = None
  entry_version: str = ""
  task_type: TaskType = TaskType.OTHER
  entry_label: str = ""
  family: ModelFamily = ModelFamily.OTHER
  config_hash: str = ""
  rule_set_id: str | None = None
  rule_ids: tuple[str, ...] = ()
  produced_at: float = field(default_factory=time.time)
  pipeline_run_id: str = ""
  confidence: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.registry_entry_id:
      raise ValueError("registry_entry_id must be non-empty")
    if self.entry_kind is None:
      raise ValueError("entry_kind must be set")
    if not self.entry_version:
      raise ValueError("entry_version must be non-empty")
    if not self.config_hash:
      raise ValueError("config_hash must be non-empty")
    if not self.entry_label:
      object.__setattr__(self, "entry_label", self.registry_entry_id)
    object.__setattr__(self, "rule_ids", tuple(self.rule_ids))
    if self.entry_kind == RegistryEntryKind.RULES and not self.rule_set_id:
      raise ValueError("rule_set_id must be non-empty for rules provenance")
    if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
      raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ProvenanceChain:
  """Ordered sequence of provenance stamps for a single output.

  Stamps appear in pipeline order: the first stamp is the earliest
  processing stage (e.g. detection), later stamps represent downstream
  stages (e.g. OCR, rule evaluation).
  """

  output_id: str
  output_kind: str
  stamps: tuple[ProvenanceStamp, ...] = ()
  pipeline_snapshot_id: str | None = None
  created_at: float = field(default_factory=time.time)

  def __post_init__(self) -> None:
    if not self.output_id:
      raise ValueError("output_id must be non-empty")
    if not self.output_kind:
      raise ValueError("output_kind must be non-empty")
    if self.pipeline_snapshot_id == "":
      raise ValueError("pipeline_snapshot_id must be non-empty when provided")
    object.__setattr__(self, "stamps", tuple(self.stamps))


# ---------------------------------------------------------------------------
# Pipeline snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineSnapshot:
  """Point-in-time snapshot of all active registry entries.

  Useful for recording the exact configuration that was running when a
  batch of detections or events was produced. Supports future experiment
  comparisons: *"snapshot A vs snapshot B"*.
  """

  snapshot_id: str
  model_entries: tuple[ModelRegistryEntry, ...] = ()
  rules_entries: tuple[RulesRegistryEntry, ...] = ()
  created_at: float = field(default_factory=time.time)
  description: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.snapshot_id:
      raise ValueError("snapshot_id must be non-empty")
    object.__setattr__(self, "model_entries", tuple(self.model_entries))
    object.__setattr__(self, "rules_entries", tuple(self.rules_entries))

  @property
  def all_entry_ids(self) -> tuple[str, ...]:
    return tuple(e.entry_id for e in self.model_entries) + tuple(
        e.entry_id for e in self.rules_entries
    )


@dataclass(frozen=True)
class PipelineSnapshotComparison:
  """Lightweight diff between two pipeline snapshots.

  This stays intentionally narrow: it highlights which registered entries
  were added, removed, or materially changed between two runs without
  pretending to be a full experiment-tracking platform.
  """

  snapshot_a_id: str
  snapshot_b_id: str
  added_entry_ids: tuple[str, ...] = ()
  removed_entry_ids: tuple[str, ...] = ()
  changed_entry_ids: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if not self.snapshot_a_id:
      raise ValueError("snapshot_a_id must be non-empty")
    if not self.snapshot_b_id:
      raise ValueError("snapshot_b_id must be non-empty")
    object.__setattr__(self, "added_entry_ids", tuple(self.added_entry_ids))
    object.__setattr__(self, "removed_entry_ids", tuple(self.removed_entry_ids))
    object.__setattr__(self, "changed_entry_ids", tuple(self.changed_entry_ids))
