# TrafficMind Model Registry and Provenance Tracking

## Purpose

The model registry tracks which models, thresholds, rule configurations, and
pipeline settings produced a given event, violation, plate read, or evidence
frame. It supports:

- **Audit**: answering *"which model version and thresholds produced this
  detection?"*
- **Debugging**: comparing outputs from different model/config versions
- **Reproducibility**: snapshotting the complete active configuration at any
  point in time
- **Future experiment comparison**: comparing snapshot A vs snapshot B without
  pretending a full ML platform exists

This is a **provenance and configuration tracking layer**, not a training
pipeline or model serving framework.

## Concepts

### Registry Entries

Two kinds of entries can be registered:

| Type | Class | Use |
|---|---|---|
| ML model / algorithm | `ModelRegistryEntry` | Detection, OCR, tracking, classification |
| Deterministic rules | `RulesRegistryEntry` | Violation rules, signal-state rules |

Each entry records:

- **identity**: `entry_id`, `name`, `version`
- **classification**: `family` (detection / ocr / tracking / classification / rules), `task_type` (object_detection / plate_recognition / face_detection / vehicle_tracking / signal_state_classification / rule_evaluation / speed_estimation)
- **config snapshot**: `ConfigBundle` with arbitrary parameters and a deterministic `config_hash`
- **lifecycle**: `status` (active / inactive / experimental / deprecated)
- **notes / metadata**: free-form fields for context

### ConfigBundle

An immutable snapshot of parameters and thresholds. The `config_hash` is
computed deterministically from the parameters so identical configs always
produce the same hash, regardless of key insertion order.

```python
from trafficmind.registry import ConfigBundle

cfg = ConfigBundle(parameters={
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "img_size": 640,
})
print(cfg.config_hash)  # "cfg-<16 hex chars>"
```

### Provenance Stamps

A `ProvenanceStamp` links one processing stage to a registry entry:

```python
stamp = registry.stamp("det-yolo", confidence=0.92, pipeline_run_id="run-42")
```

Each stamp carries enough stable identity to remain readable even if the
in-memory registry is no longer available at inspection time:

- `registry_entry_id`
- `entry_kind` (`model` or `rules`)
- `entry_label` (`name` for models, `rule_set_id` for rules)
- `entry_version`
- `family` and `task_type`
- `config_hash`
- `rule_set_id` / `rule_ids` for rules provenance

### Provenance Chains

A `ProvenanceChain` records the full pipeline that produced an output. Stamps
appear in pipeline order (detection → OCR → rule evaluation). It can also
carry the active `pipeline_snapshot_id` when the caller wants a direct link to
the broader runtime configuration:

```python
chain = registry.build_provenance_chain(
    output_id="plate-read-100",
    output_kind="plate_read",
    stamps=[det_stamp, ocr_stamp, rule_stamp],
  pipeline_snapshot_id="prod-2026-04-06",
)
```

### Pipeline Snapshots

A `PipelineSnapshot` captures all active (or filtered) registry entries at a
point in time:

```python
snapshot = registry.take_snapshot(description="Production config 2026-04-06")
```

Snapshots are stored in the registry and can be retrieved later for comparison.

## Integration with Review Models

Provenance is wired into the existing review models as optional fields so
existing callers are unaffected:

| Model | Field | Type |
|---|---|---|
| `ReviewEvent` | `provenance` | `ProvenanceChain \| None` |
| `EvidenceReference` | `provenance` | `ProvenanceChain \| None` |
| `PlatformRecord` / `SearchHit` | `provenance` | `ProvenanceChain \| None` |
| `MultimodalReviewResult` | `pipeline_snapshot_id` | `str \| None` |

This means:

- An event submitted for review can declare which detection model and rule
  version produced it.
- A violation record or plate-read record in the runtime/search layer can
  declare which detection, OCR, tracking, and rules stages produced it.
- An evidence frame can declare which model produced its detections or crops.
- A provenance chain can carry the pipeline snapshot id for the runtime config
  that produced the output, and a review result can separately reference the
  snapshot active when the workflow ran.
- Because each stamp stores `config_hash` and rules identity directly, outputs
  remain auditable even when a live in-memory registry instance is unavailable.

### Snapshot Comparison

The registry also supports a lightweight snapshot diff:

```python
diff = registry.compare_snapshots("prod-2026-04-06", "exp-2026-04-07")
print(diff.added_entry_ids)
print(diff.removed_entry_ids)
print(diff.changed_entry_ids)
```

This is intentionally practical rather than overbuilt. It highlights which
registered entries changed between two runs so teams can compare benchmark or
debugging outcomes without introducing a full experiment platform.

## Registry API

### ModelConfigRegistry

```python
from trafficmind.registry import (
    ModelConfigRegistry, ModelFamily, TaskType, EntryStatus, ConfigBundle,
)

registry = ModelConfigRegistry()

# Register a detection model
det = registry.register_model(
    entry_id="det-yolov8n",
    family=ModelFamily.DETECTION,
    task_type=TaskType.OBJECT_DETECTION,
    name="YOLOv8n",
    version="8.0.1",
    config=ConfigBundle(parameters={"conf": 0.25, "iou": 0.45}),
)

# Register an OCR model
ocr = registry.register_model(
    entry_id="ocr-paddle",
    family=ModelFamily.OCR,
    task_type=TaskType.PLATE_RECOGNITION,
    name="PaddleOCR",
    version="2.7",
    status=EntryStatus.EXPERIMENTAL,
)

# Register a rule configuration
rules = registry.register_rules(
    entry_id="rules-violations-v2",
    rule_set_id="traffic-violations",
    version="2.0",
    rule_ids=["red-light-running", "stop-line-violation", "wrong-way"],
    config=ConfigBundle(parameters={"grace_period_s": 0.5}),
)

# Look up entries
entry = registry.get_model("det-yolov8n")
entry = registry.get_rules("rules-violations-v2")
entry = registry.get_entry("det-yolov8n")  # finds either type

# Filter entries
active_detectors = registry.list_models(
    family=ModelFamily.DETECTION, status=EntryStatus.ACTIVE,
)

# Change status
registry.set_model_status("ocr-paddle", EntryStatus.ACTIVE)
registry.set_rules_status("rules-violations-v2", EntryStatus.DEPRECATED)

# Take a snapshot
snapshot = registry.take_snapshot(
    description="Production config 2026-04-06",
)

# Create provenance
det_stamp = registry.stamp("det-yolov8n", confidence=0.92)
rule_stamp = registry.stamp("rules-violations-v2")
chain = registry.build_provenance_chain(
  output_id="INC-42",
  output_kind="event",
  stamps=[det_stamp, rule_stamp],
  pipeline_snapshot_id=snapshot.snapshot_id,
)
```

## What This Does NOT Do

- **No model loading or inference** — the registry tracks metadata, not
  weights or runtime objects.
- **No training pipelines** — TrafficMind does not contain training code.
- **No A/B testing framework** — snapshots support future comparisons, but
  there is no experiment orchestrator yet.
- **No persistent storage** — the registry is in-memory. Serialization to
  JSON/database can be added when needed.
- **No authentication** — entry registration is not gated by permissions.
  This may be added when the registry is exposed via an API.

These capabilities may be added later, but they are not present today and
should not be implied.

## Design Decisions

1. **Separate ModelRegistryEntry and RulesRegistryEntry** — rules are
   deterministic logic, not learned models. Keeping them separate makes the
   distinction explicit while still allowing both to appear in provenance
   chains and snapshots.

2. **ConfigBundle with deterministic hashing** — enables comparing configs
   by hash without deep equality checks. Useful for detecting config drift.

3. **ProvenanceChain as ordered stamps** — models real pipelines where
  detection feeds OCR feeds rule evaluation. Each stage gets its own stamp,
  and the chain can optionally link to a broader pipeline snapshot.

4. **Self-describing provenance stamps** — stamps include `config_hash`,
  entry kind, entry label, and rules identifiers where relevant so stored
  outputs still carry meaningful audit context without requiring a live
  registry lookup.

5. **Optional integration** — provenance fields on ReviewEvent,
   EvidenceReference, and MultimodalReviewResult are all optional (None by
   default). Existing code is unaffected.

6. **Thread-safe registry** — the in-memory store uses a lock so it can
   be shared across workflow nodes without races.

7. **Audit-friendly identifiers** — registry entry ids are globally unique
  across models and rules, and snapshot ids cannot be silently overwritten.
