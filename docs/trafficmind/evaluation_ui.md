# TrafficMind Evaluation UI Foundation

## Purpose

This foundation provides a practical, honest evaluation/admin view for
TrafficMind. It is designed to improve trust and debugging by rendering only
**local evaluation artifacts** or **stored result exports** that actually exist
 on disk.

It does **not** invent benchmark numbers, run hidden calculations in the UI, or
pretend to be a leaderboard or experiment platform.

## What The UI Shows

The page renders one card per artifact section and always keeps these classes of
information visually separate:

- **Measured metrics**: numeric values loaded from artifact JSON
- **Manual review summaries**: human-authored notes and verdicts
- **Not yet available**: explicit placeholders for section kinds that are not
  present in the loaded artifacts

Supported section kinds:

- `detection_sanity`
- `tracking_consistency`
- `ocr_quality`
- `rule_validation`
- `workflow_summary`

The report includes filters for:

- model/config version
- section kind (detection, tracking, OCR, rules, workflow)
- camera
- test scenario
- date/time window
- task type

A **Reset Filters** button clears all active filters in one click.

## Artifact-Backed Only

The HTML report is built from JSON artifacts loaded with
`trafficmind.evaluation.load_evaluation_artifacts(...)`.

Each artifact contains:

- artifact metadata such as `artifact_id`, `title`, `captured_at`, camera,
  scenario, and optional `pipeline_snapshot_id`
- registry bindings linking the result to model/config versions or rules configs
- one or more evaluation sections

Sections may include:

- `measured_metrics`
- `manual_summaries`
- `samples`
- `validation_scenarios`
- `placeholder`

The page never backfills missing metrics. If a category is missing from an
artifact, the UI shows a placeholder explaining that no local results were
loaded for that section.

## Example Artifact Shape

```json
{
  "artifact_id": "eval-night-cam7",
  "title": "Night rain regression set",
  "captured_at": 1743948000,
  "camera_id": "CAM-7",
  "scenario_id": "night-rain",
  "pipeline_snapshot_id": "snap-prod-2026-04-06",
  "registry_bindings": [
    {
      "entry_id": "det-yolo",
      "entry_label": "YOLOv8n",
      "entry_version": "8.0.1",
      "config_hash": "cfg-a1b2c3d4e5f6a7b8",
      "task_type": "object_detection",
      "family": "detection"
    }
  ],
  "sections": [
    {
      "section_id": "det-sanity-main",
      "kind": "detection_sanity",
      "task_type": "object_detection",
      "summary_text": "Fixture-based night regression sanity pass.",
      "measured_metrics": [
        {
          "name": "precision_at_iou_50",
          "value": 0.91,
          "unit": "ratio",
          "sample_size": 120,
          "note": "Derived from local fixture frames."
        }
      ]
    },
    {
      "section_id": "ocr-samples-main",
      "kind": "ocr_quality",
      "task_type": "plate_recognition",
      "samples": [
        {
          "sample_id": "plate-001",
          "label": "Plate crop 001",
          "expected_value": "AB12XYZ",
          "observed_value": "AB12XY2",
          "score": 0.86,
          "passed": false,
          "note": "Night glare on final character."
        }
      ]
    }
  ]
}
```

## Rendering The Report

Use the helper script:

```bash
python scripts/render_trafficmind_evaluation_report.py \
  trafficmind_eval_results/ \
  --output reports/trafficmind_eval.html
```

Or call the APIs directly:

```python
from trafficmind.evaluation import (
    load_evaluation_artifacts,
    write_evaluation_report,
)

artifacts = load_evaluation_artifacts("trafficmind_eval_results")
write_evaluation_report(artifacts, "reports/trafficmind_eval.html")
```

## How Results Should Be Generated

This UI layer does not define a single benchmark runner. It is meant to render
outputs from real local evaluation sources such as:

- fixture-based sanity checks in tests
- stored benchmark JSON exports
- manual review exports from workflow investigations
- rule validation batches from deterministic scenario suites

That keeps the page implementation-backed and honest.

## Interpretation Guidance

- Treat **measured metrics** as exact only for the dataset and fixture set that
  produced the artifact.
- Treat **manual review summaries** as operator or reviewer observations, not
  numeric performance claims.
- Treat **placeholder sections** as missing coverage, not zero performance.
- Use registry bindings and `pipeline_snapshot_id` to trace a visible result
  back to the model/config or rules version that produced it.

## Non-Goals

- No leaderboard ranking
- No fabricated benchmark aggregates
- No live metric recomputation in the browser
- No hidden evaluation database
- No claim that missing categories are healthy or passing