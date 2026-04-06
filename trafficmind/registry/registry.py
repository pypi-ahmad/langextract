"""In-memory model and configuration registry.

Provides registration, lookup, deactivation, and snapshot operations for
model and rules configuration entries. Thread-safe via a simple lock so
the registry can be shared across workflow nodes.
"""

from __future__ import annotations

import threading
import time
from typing import Sequence
import uuid

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


class ModelConfigRegistry:
  """Lightweight in-memory registry for model and rules configurations.

  Not a database — designed for single-process provenance tracking,
  debugging, and audit. Snapshots can be serialized if persistence is
  needed later.
  """

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._models: dict[str, ModelRegistryEntry] = {}
    self._rules: dict[str, RulesRegistryEntry] = {}
    self._snapshots: dict[str, PipelineSnapshot] = {}

  # ------------------------------------------------------------------
  # Model entries
  # ------------------------------------------------------------------

  def register_model(
      self,
      *,
      family: ModelFamily,
      task_type: TaskType,
      name: str,
      version: str,
      config: ConfigBundle | None = None,
      status: EntryStatus = EntryStatus.ACTIVE,
      notes: str = "",
      entry_id: str = "",
      metadata: dict | None = None,
  ) -> ModelRegistryEntry:
    """Register a model entry and return it."""
    eid = entry_id or f"model-{uuid.uuid4().hex[:12]}"
    entry = ModelRegistryEntry(
        entry_id=eid,
        family=family,
        task_type=task_type,
        name=name,
        version=version,
        config=config or ConfigBundle(),
        status=status,
        notes=notes,
        metadata=metadata or {},
    )
    with self._lock:
      self._ensure_entry_id_available(eid)
      self._models[eid] = entry
    return entry

  def register_rules(
      self,
      *,
      rule_set_id: str,
      version: str,
      description: str = "",
      rule_ids: Sequence[str] = (),
      config: ConfigBundle | None = None,
      status: EntryStatus = EntryStatus.ACTIVE,
      entry_id: str = "",
      metadata: dict | None = None,
  ) -> RulesRegistryEntry:
    """Register a rules-configuration entry and return it."""
    eid = entry_id or f"rules-{uuid.uuid4().hex[:12]}"
    entry = RulesRegistryEntry(
        entry_id=eid,
        rule_set_id=rule_set_id,
        version=version,
        description=description,
        rule_ids=tuple(rule_ids),
        config=config or ConfigBundle(),
        status=status,
        metadata=metadata or {},
    )
    with self._lock:
      self._ensure_entry_id_available(eid)
      self._rules[eid] = entry
    return entry

  # ------------------------------------------------------------------
  # Lookup
  # ------------------------------------------------------------------

  def get_model(self, entry_id: str) -> ModelRegistryEntry | None:
    with self._lock:
      return self._models.get(entry_id)

  def get_rules(self, entry_id: str) -> RulesRegistryEntry | None:
    with self._lock:
      return self._rules.get(entry_id)

  def get_entry(
      self,
      entry_id: str,
  ) -> ModelRegistryEntry | RulesRegistryEntry | None:
    """Look up any entry by id (model or rules)."""
    return self.get_model(entry_id) or self.get_rules(entry_id)

  def list_models(
      self,
      *,
      family: ModelFamily | None = None,
      task_type: TaskType | None = None,
      status: EntryStatus | None = None,
  ) -> tuple[ModelRegistryEntry, ...]:
    """Return model entries, optionally filtered."""
    with self._lock:
      entries = list(self._models.values())
    if family is not None:
      entries = [e for e in entries if e.family == family]
    if task_type is not None:
      entries = [e for e in entries if e.task_type == task_type]
    if status is not None:
      entries = [e for e in entries if e.status == status]
    return tuple(entries)

  def list_rules(
      self,
      *,
      status: EntryStatus | None = None,
  ) -> tuple[RulesRegistryEntry, ...]:
    """Return rules entries, optionally filtered."""
    with self._lock:
      entries = list(self._rules.values())
    if status is not None:
      entries = [e for e in entries if e.status == status]
    return tuple(entries)

  # ------------------------------------------------------------------
  # Status changes
  # ------------------------------------------------------------------

  def set_model_status(
      self,
      entry_id: str,
      status: EntryStatus,
  ) -> ModelRegistryEntry:
    """Return a new entry with updated status, replacing the old one."""
    with self._lock:
      old = self._models.get(entry_id)
      if old is None:
        raise KeyError(f"Model entry {entry_id!r} not found")
      updated = ModelRegistryEntry(
          entry_id=old.entry_id,
          family=old.family,
          task_type=old.task_type,
          name=old.name,
          version=old.version,
          config=old.config,
          status=status,
          notes=old.notes,
          registered_at=old.registered_at,
          metadata=old.metadata,
      )
      self._models[entry_id] = updated
    return updated

  def set_rules_status(
      self,
      entry_id: str,
      status: EntryStatus,
  ) -> RulesRegistryEntry:
    with self._lock:
      old = self._rules.get(entry_id)
      if old is None:
        raise KeyError(f"Rules entry {entry_id!r} not found")
      updated = RulesRegistryEntry(
          entry_id=old.entry_id,
          rule_set_id=old.rule_set_id,
          version=old.version,
          description=old.description,
          rule_ids=old.rule_ids,
          config=old.config,
          status=status,
          registered_at=old.registered_at,
          metadata=old.metadata,
      )
      self._rules[entry_id] = updated
    return updated

  # ------------------------------------------------------------------
  # Snapshots
  # ------------------------------------------------------------------

  def take_snapshot(
      self,
      *,
      description: str = "",
      snapshot_id: str = "",
      include_statuses: frozenset[EntryStatus] | None = None,
      metadata: dict | None = None,
  ) -> PipelineSnapshot:
    """Capture a point-in-time snapshot of (optionally filtered) entries.

    By default, only ``ACTIVE`` entries are included. Pass
    ``include_statuses`` to override.
    """
    allowed = include_statuses or frozenset({EntryStatus.ACTIVE})
    sid = snapshot_id or f"snap-{uuid.uuid4().hex[:12]}"
    with self._lock:
      if sid in self._snapshots:
        raise ValueError(f"Snapshot {sid!r} already exists")
      model_entries = tuple(
          sorted(
              (e for e in self._models.values() if e.status in allowed),
              key=lambda entry: entry.entry_id,
          )
      )
      rules_entries = tuple(
          sorted(
              (e for e in self._rules.values() if e.status in allowed),
              key=lambda entry: entry.entry_id,
          )
      )
      snapshot = PipelineSnapshot(
          snapshot_id=sid,
          model_entries=model_entries,
          rules_entries=rules_entries,
          description=description,
          metadata=metadata or {},
      )
      self._snapshots[sid] = snapshot
    return snapshot

  def get_snapshot(self, snapshot_id: str) -> PipelineSnapshot | None:
    with self._lock:
      return self._snapshots.get(snapshot_id)

  def list_snapshots(self) -> tuple[PipelineSnapshot, ...]:
    with self._lock:
      return tuple(self._snapshots.values())

  def compare_snapshots(
      self,
      snapshot_a_id: str,
      snapshot_b_id: str,
  ) -> PipelineSnapshotComparison:
    """Return a lightweight diff between two stored snapshots."""
    snapshot_a = self.get_snapshot(snapshot_a_id)
    if snapshot_a is None:
      raise KeyError(f"Snapshot {snapshot_a_id!r} not found")
    snapshot_b = self.get_snapshot(snapshot_b_id)
    if snapshot_b is None:
      raise KeyError(f"Snapshot {snapshot_b_id!r} not found")

    entries_a = {
        entry.entry_id: self._entry_signature(entry)
        for entry in (*snapshot_a.model_entries, *snapshot_a.rules_entries)
    }
    entries_b = {
        entry.entry_id: self._entry_signature(entry)
        for entry in (*snapshot_b.model_entries, *snapshot_b.rules_entries)
    }
    ids_a = set(entries_a)
    ids_b = set(entries_b)
    shared_ids = ids_a & ids_b

    return PipelineSnapshotComparison(
        snapshot_a_id=snapshot_a_id,
        snapshot_b_id=snapshot_b_id,
        added_entry_ids=tuple(sorted(ids_b - ids_a)),
        removed_entry_ids=tuple(sorted(ids_a - ids_b)),
        changed_entry_ids=tuple(
            sorted(
                entry_id
                for entry_id in shared_ids
                if entries_a[entry_id] != entries_b[entry_id]
            )
        ),
    )

  # ------------------------------------------------------------------
  # Provenance helpers
  # ------------------------------------------------------------------

  def stamp(
      self,
      entry_id: str,
      *,
      pipeline_run_id: str = "",
      confidence: float | None = None,
      metadata: dict | None = None,
  ) -> ProvenanceStamp:
    """Create a provenance stamp from a registered entry.

    Raises ``KeyError`` if the entry does not exist.
    """
    entry = self.get_entry(entry_id)
    if entry is None:
      raise KeyError(f"Registry entry {entry_id!r} not found")
    if isinstance(entry, ModelRegistryEntry):
      return ProvenanceStamp(
          registry_entry_id=entry_id,
          entry_kind=RegistryEntryKind.MODEL,
          entry_version=entry.version,
          task_type=entry.task_type,
          entry_label=entry.name,
          family=entry.family,
          config_hash=entry.config.config_hash,
          pipeline_run_id=pipeline_run_id,
          confidence=confidence,
          metadata=metadata or {},
      )
    else:
      return ProvenanceStamp(
          registry_entry_id=entry_id,
          entry_kind=RegistryEntryKind.RULES,
          entry_version=entry.version,
          task_type=TaskType.RULE_EVALUATION,
          entry_label=entry.rule_set_id,
          family=ModelFamily.RULES,
          config_hash=entry.config.config_hash,
          rule_set_id=entry.rule_set_id,
          rule_ids=entry.rule_ids,
          pipeline_run_id=pipeline_run_id,
          confidence=confidence,
          metadata=metadata or {},
      )

  def build_provenance_chain(
      self,
      output_id: str,
      output_kind: str,
      stamps: Sequence[ProvenanceStamp],
      *,
      pipeline_snapshot_id: str | None = None,
  ) -> ProvenanceChain:
    """Build a provenance chain from an ordered sequence of stamps."""
    return ProvenanceChain(
        output_id=output_id,
        output_kind=output_kind,
        stamps=tuple(stamps),
        pipeline_snapshot_id=pipeline_snapshot_id,
    )

  # ------------------------------------------------------------------
  # Introspection
  # ------------------------------------------------------------------

  @property
  def model_count(self) -> int:
    with self._lock:
      return len(self._models)

  @property
  def rules_count(self) -> int:
    with self._lock:
      return len(self._rules)

  @property
  def snapshot_count(self) -> int:
    with self._lock:
      return len(self._snapshots)

  @staticmethod
  def _entry_signature(
      entry: ModelRegistryEntry | RulesRegistryEntry,
  ) -> tuple[str, str, str, str]:
    if isinstance(entry, ModelRegistryEntry):
      return (
          "model",
          entry.version,
          entry.status.value,
          entry.config.config_hash,
      )
    return (
        "rules",
        entry.version,
        entry.status.value,
        entry.config.config_hash,
    )

  def _ensure_entry_id_available(self, entry_id: str) -> None:
    if entry_id in self._models or entry_id in self._rules:
      raise ValueError(
          f"Registry entry {entry_id!r} is already registered; entry ids must"
          " be globally unique across models and rules"
      )
