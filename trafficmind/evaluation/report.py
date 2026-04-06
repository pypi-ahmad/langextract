"""Render TrafficMind evaluation artifacts as a standalone filterable HTML page."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Sequence

from trafficmind.evaluation.models import EvaluationArtifact
from trafficmind.evaluation.models import EvaluationSection
from trafficmind.evaluation.models import EvaluationSectionKind
from trafficmind.evaluation.models import PlaceholderNotice
from trafficmind.evaluation.models import SECTION_DEFAULT_TASK_TYPES
from trafficmind.evaluation.models import SECTION_TITLES

_SECTION_ORDER = (
    EvaluationSectionKind.DETECTION_SANITY,
    EvaluationSectionKind.TRACKING_CONSISTENCY,
    EvaluationSectionKind.OCR_QUALITY,
    EvaluationSectionKind.RULE_VALIDATION,
    EvaluationSectionKind.WORKFLOW_SUMMARY,
)


def render_evaluation_report(
    artifacts: Sequence[EvaluationArtifact],
    *,
    title: str = "TrafficMind Evaluation Report",
) -> str:
  """Render a standalone HTML page backed by local evaluation artifacts."""
  payload = {
      "cards": _build_cards(artifacts),
      "artifactCount": len(artifacts),
  }
  data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
  artifact_inventory_html = _render_artifact_inventory(artifacts)
  template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f4efe5;
      --panel: rgba(255, 251, 245, 0.94);
      --ink: #172127;
      --muted: #5f6a70;
      --line: rgba(23, 33, 39, 0.14);
      --accent: #0b6e5d;
      --warning: #c27b16;
      --danger: #a3442e;
      --shadow: 0 16px 40px rgba(23, 33, 39, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(11, 110, 93, 0.10), transparent 32%),
        radial-gradient(circle at top right, rgba(194, 123, 22, 0.10), transparent 28%),
        var(--bg);
    }
    .shell {
      width: min(1400px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }
    .hero, .filters, .summary, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 24px 28px;
      margin-bottom: 18px;
    }
    .hero h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .hero p {
      margin: 0;
      max-width: 76ch;
      color: var(--muted);
      font-size: 0.98rem;
      line-height: 1.6;
    }
    .filters {
      padding: 18px 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .field label {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .field select, .field input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.85);
      color: var(--ink);
      font: inherit;
    }
    .summary {
      padding: 14px 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .artifact-inventory {
      padding: 16px 18px;
      margin-bottom: 18px;
    }
    .artifact-inventory h2 {
      margin: 0 0 10px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    .artifact-inventory ul {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }
    .artifact-inventory li {
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid rgba(23, 33, 39, 0.08);
    }
    .artifact-inventory .path {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.92rem;
      word-break: break-all;
    }
    .stat {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid rgba(23, 33, 39, 0.08);
    }
    .stat .label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .stat .value {
      font-size: 1.6rem;
      font-weight: 700;
    }
    .cards {
      display: grid;
      gap: 16px;
    }
    .card {
      padding: 18px 20px;
    }
    .card-top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .card h2 {
      margin: 0;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
    }
    .subhead {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .badge {
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      background: rgba(11, 110, 93, 0.10);
      color: var(--accent);
      border: 1px solid rgba(11, 110, 93, 0.16);
    }
    .badge.placeholder {
      background: rgba(194, 123, 22, 0.12);
      color: var(--warning);
      border-color: rgba(194, 123, 22, 0.22);
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .meta {
      font-size: 0.92rem;
      color: var(--muted);
    }
    .meta strong {
      color: var(--ink);
      display: block;
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .section-block {
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .section-block h3 {
      margin: 0 0 10px;
      font-size: 0.96rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }
    th, td {
      border-bottom: 1px solid rgba(23, 33, 39, 0.08);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
    }
    ul.clean {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }
    .note {
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(23, 33, 39, 0.08);
    }
    .note .title {
      font-weight: 700;
      margin-bottom: 4px;
    }
    .placeholder-callout {
      border-radius: 14px;
      padding: 14px 16px;
      background: rgba(194, 123, 22, 0.10);
      color: #72460b;
      border: 1px solid rgba(194, 123, 22, 0.18);
    }
    .binding-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .binding {
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(23, 33, 39, 0.05);
      border: 1px solid rgba(23, 33, 39, 0.08);
      font-size: 0.84rem;
    }
    .empty {
      padding: 28px;
      text-align: center;
      color: var(--muted);
      background: var(--panel);
      border: 1px dashed rgba(23, 33, 39, 0.18);
      border-radius: 20px;
    }
    .pass { color: var(--accent); font-weight: 700; }
    .fail { color: var(--danger); font-weight: 700; }
    @media (max-width: 720px) {
      .shell { width: min(100vw - 18px, 1400px); margin-top: 12px; }
      .hero, .filters, .summary, .card { border-radius: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>__TITLE__</h1>
      <p>
        This admin report renders only locally loaded evaluation artifacts or stored result exports.
        Real measured metrics, manual review summaries, and not-yet-available sections are shown separately so
        missing data is visible instead of being backfilled.
      </p>
    </section>

    <section class="filters">
      <div class="field">
        <label for="modelFilter">Model / Config Version</label>
        <select id="modelFilter"><option value="">All loaded bindings</option></select>
      </div>
      <div class="field">
        <label for="kindFilter">Section Kind</label>
        <select id="kindFilter"><option value="">All section kinds</option></select>
      </div>
      <div class="field">
        <label for="cameraFilter">Camera</label>
        <select id="cameraFilter"><option value="">All cameras</option></select>
      </div>
      <div class="field">
        <label for="scenarioFilter">Test Scenario</label>
        <select id="scenarioFilter"><option value="">All scenarios</option></select>
      </div>
      <div class="field">
        <label for="taskFilter">Task Type</label>
        <select id="taskFilter"><option value="">All task types</option></select>
      </div>
      <div class="field">
        <label for="fromFilter">From</label>
        <input id="fromFilter" type="datetime-local">
      </div>
      <div class="field">
        <label for="toFilter">To</label>
        <input id="toFilter" type="datetime-local">
      </div>
      <div class="field" style="align-self: end;">
        <button id="resetFilters" type="button" style="cursor:pointer; padding:6px 16px; border-radius:8px; border:1px solid rgba(23,33,39,0.15); background:var(--panel); color:var(--ink); font-size:0.92rem;">Reset Filters</button>
      </div>
    </section>

    <section class="artifact-inventory hero">
      <h2>Loaded Artifacts</h2>
      __ARTIFACT_INVENTORY__
    </section>

    <section class="summary" id="summary"></section>
    <section class="cards" id="cards"></section>
  </div>

  <script id="trafficmind-eval-data" type="application/json">__DATA__</script>
  <script>
    const payload = JSON.parse(document.getElementById("trafficmind-eval-data").textContent);
    const cards = payload.cards;

    const controls = {
      modelFilter: document.getElementById("modelFilter"),
      kindFilter: document.getElementById("kindFilter"),
      cameraFilter: document.getElementById("cameraFilter"),
      scenarioFilter: document.getElementById("scenarioFilter"),
      taskFilter: document.getElementById("taskFilter"),
      fromFilter: document.getElementById("fromFilter"),
      toFilter: document.getElementById("toFilter"),
    };

    function uniqueSorted(values) {
      return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
    }

    function fillSelect(select, values) {
      const original = select.value;
      values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
      if (values.includes(original)) {
        select.value = original;
      }
    }

    fillSelect(controls.modelFilter, uniqueSorted(cards.flatMap((card) => card.bindingKeys)));
    (function () {
      const kindMap = {};
      cards.forEach(function (card) { if (card.kind) kindMap[card.kind] = card.kindLabel; });
      Object.keys(kindMap).sort().forEach(function (kind) {
        const option = document.createElement("option");
        option.value = kind;
        option.textContent = kindMap[kind];
        controls.kindFilter.appendChild(option);
      });
    })();
    fillSelect(controls.cameraFilter, uniqueSorted(cards.map((card) => card.cameraId)));
    fillSelect(controls.scenarioFilter, uniqueSorted(cards.map((card) => card.scenarioId)));
    fillSelect(controls.taskFilter, uniqueSorted(cards.map((card) => card.taskType)));

    Object.values(controls).forEach((control) => control.addEventListener("change", render));

    document.getElementById("resetFilters").addEventListener("click", function () {
      Object.values(controls).forEach(function (control) {
        if (control.tagName === "SELECT") { control.selectedIndex = 0; }
        else { control.value = ""; }
      });
      render();
    });

    function toEpoch(value) {
      if (!value) {
        return null;
      }
      const millis = Date.parse(value);
      return Number.isNaN(millis) ? null : millis / 1000;
    }

    function matches(card) {
      const fromTs = toEpoch(controls.fromFilter.value);
      const toTs = toEpoch(controls.toFilter.value);
      if (controls.modelFilter.value && !card.bindingKeys.includes(controls.modelFilter.value)) {
        return false;
      }
      if (controls.kindFilter.value && card.kind !== controls.kindFilter.value) {
        return false;
      }
      if (controls.cameraFilter.value && card.cameraId !== controls.cameraFilter.value) {
        return false;
      }
      if (controls.scenarioFilter.value && card.scenarioId !== controls.scenarioFilter.value) {
        return false;
      }
      if (controls.taskFilter.value && card.taskType !== controls.taskFilter.value) {
        return false;
      }
      if (fromTs !== null && card.capturedAt < fromTs) {
        return false;
      }
      if (toTs !== null && card.capturedAt > toTs) {
        return false;
      }
      return true;
    }

    function formatDate(epochSeconds) {
      return epochSeconds ? new Date(epochSeconds * 1000).toLocaleString() : "n/a";
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function renderSummary(filtered) {
      const artifactCount = new Set(filtered.map((card) => card.artifactId)).size;
      const dataBackedCount = filtered.filter((card) => card.measuredMetrics.length || card.samples.length || card.validationScenarios.length).length;
      const manualCount = filtered.filter((card) => card.manualSummaries.length).length;
      const placeholderCount = filtered.filter((card) => card.placeholder).length;
      document.getElementById("summary").innerHTML = [
        ["Artifacts", artifactCount],
        ["Sections Shown", filtered.length],
        ["Data-Backed Sections", dataBackedCount],
        ["Manual Review Sections", manualCount],
        ["Placeholder Sections", placeholderCount],
      ].map(([label, value]) => `
        <div class="stat">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
        </div>`).join("");
    }

    function renderCards(filtered) {
      const container = document.getElementById("cards");
      if (!filtered.length) {
        container.innerHTML = `
          <div class="empty">
            No evaluation sections match the current filters. Load local artifact JSON files or widen the filter window.
          </div>`;
        return;
      }
      container.innerHTML = filtered.map((card) => {
        const metricsBlock = card.measuredMetrics.length ? `
          <div class="section-block">
            <h3>Measured Metrics</h3>
            <table>
              <thead><tr><th>Metric</th><th>Value</th><th>Sample Size</th><th>Notes</th></tr></thead>
              <tbody>
                ${card.measuredMetrics.map((metric) => `
                  <tr>
                    <td>${escapeHtml(metric.name)}</td>
                    <td>${escapeHtml(metric.unit ? `${metric.value} ${metric.unit}` : metric.value)}</td>
                    <td>${escapeHtml(metric.sampleSize ?? "n/a")}</td>
                    <td>${escapeHtml(metric.note || "")}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>` : "";

        const samplesBlock = card.samples.length ? `
          <div class="section-block">
            <h3>Quality / Consistency Samples</h3>
            <table>
              <thead><tr><th>Sample</th><th>Expected</th><th>Observed</th><th>Score</th><th>Status</th><th>Notes</th></tr></thead>
              <tbody>
                ${card.samples.map((sample) => `
                  <tr>
                    <td>${escapeHtml(sample.label)}</td>
                    <td>${escapeHtml(sample.expectedValue || "n/a")}</td>
                    <td>${escapeHtml(sample.observedValue || "n/a")}</td>
                    <td>${escapeHtml(sample.score ?? "n/a")}</td>
                    <td class="${sample.passed === true ? "pass" : sample.passed === false ? "fail" : ""}">${escapeHtml(sample.passed === null ? "n/a" : sample.passed ? "pass" : "fail")}</td>
                    <td>${escapeHtml(sample.note || "")}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>` : "";

        const scenariosBlock = card.validationScenarios.length ? `
          <div class="section-block">
            <h3>Rule Validation Scenarios</h3>
            <table>
              <thead><tr><th>Scenario</th><th>Expected</th><th>Actual</th><th>Status</th><th>Notes</th></tr></thead>
              <tbody>
                ${card.validationScenarios.map((scenario) => `
                  <tr>
                    <td>${escapeHtml(scenario.title)}</td>
                    <td>${escapeHtml(scenario.expectedOutcome)}</td>
                    <td>${escapeHtml(scenario.actualOutcome)}</td>
                    <td class="${scenario.passed ? "pass" : "fail"}">${scenario.passed ? "pass" : "fail"}</td>
                    <td>${escapeHtml(scenario.note || "")}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>` : "";

        const manualBlock = card.manualSummaries.length ? `
          <div class="section-block">
            <h3>Manual Review Summaries</h3>
            <ul class="clean">
              ${card.manualSummaries.map((summary) => `
                <li class="note">
                  <div class="title">${escapeHtml(summary.title)}</div>
                  <div>${escapeHtml(summary.summary)}</div>
                  <div class="subhead">${escapeHtml(summary.reviewer || "unattributed review")}${summary.status ? ` · ${escapeHtml(summary.status)}` : ""}${summary.reviewedAt ? ` · ${escapeHtml(formatDate(summary.reviewedAt))}` : ""}</div>
                </li>`).join("")}
            </ul>
          </div>` : "";

        const placeholderBlock = card.placeholder ? `
          <div class="section-block">
            <h3>Not Yet Available</h3>
            <div class="placeholder-callout">
              <strong>${escapeHtml(card.placeholder.title)}</strong><br>
              ${escapeHtml(card.placeholder.detail)}
            </div>
          </div>` : "";

        const bindingBlock = card.bindings.length ? `
          <div class="section-block">
            <h3>Model / Config Bindings</h3>
            <div class="binding-list">
              ${card.bindings.map((binding) => `<div class="binding">${escapeHtml(binding.display)}</div>`).join("")}
            </div>
          </div>` : "";

        const summaryTextBlock = card.summaryText ? `
          <div class="section-block">
            <h3>Section Summary</h3>
            <div class="note">${escapeHtml(card.summaryText)}</div>
          </div>` : "";

        return `
          <article class="card">
            <div class="card-top">
              <div>
                <h2>${escapeHtml(card.title)}</h2>
                <div class="subhead">${escapeHtml(card.artifactTitle)} · ${escapeHtml(formatDate(card.capturedAt))}</div>
              </div>
              <div class="badges">
                <span class="badge">${escapeHtml(card.kindLabel)}</span>
                <span class="badge">${escapeHtml(card.taskType)}</span>
                ${card.placeholder ? '<span class="badge placeholder">no local data</span>' : ''}
              </div>
            </div>
            <div class="meta-grid">
              <div class="meta"><strong>Artifact</strong>${escapeHtml(card.artifactId)}</div>
              <div class="meta"><strong>Source File</strong>${escapeHtml(card.sourcePath || "n/a")}</div>
              <div class="meta"><strong>Camera</strong>${escapeHtml(card.cameraId || "n/a")}</div>
              <div class="meta"><strong>Scenario</strong>${escapeHtml(card.scenarioId || "n/a")}</div>
              <div class="meta"><strong>Snapshot</strong>${escapeHtml(card.pipelineSnapshotId || "n/a")}</div>
            </div>
            ${summaryTextBlock}
            ${metricsBlock}
            ${samplesBlock}
            ${scenariosBlock}
            ${manualBlock}
            ${placeholderBlock}
            ${bindingBlock}
          </article>`;
      }).join("");
    }

    function render() {
      const filtered = cards.filter(matches);
      renderSummary(filtered);
      renderCards(filtered);
    }

    render();
  </script>
</body>
</html>
"""
  return (
      template.replace("__TITLE__", escape(title))
      .replace("__ARTIFACT_INVENTORY__", artifact_inventory_html)
      .replace("__DATA__", data_json)
  )


def write_evaluation_report(
    artifacts: Sequence[EvaluationArtifact],
    output_path: str | Path,
    *,
    title: str = "TrafficMind Evaluation Report",
) -> Path:
  """Write the rendered evaluation report to disk."""
  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      render_evaluation_report(artifacts, title=title), encoding="utf-8"
  )
  return output


def _build_cards(
    artifacts: Sequence[EvaluationArtifact],
) -> list[dict[str, Any]]:
  cards: list[dict[str, Any]] = []
  for artifact in artifacts:
    present_kinds = {section.kind for section in artifact.sections}
    for section in artifact.sections:
      cards.append(_serialize_card(artifact, section))
    for missing_kind in _SECTION_ORDER:
      if missing_kind in present_kinds:
        continue
      placeholder_section = EvaluationSection(
          section_id=f"{artifact.artifact_id}-{missing_kind.value}-placeholder",
          kind=missing_kind,
          task_type=SECTION_DEFAULT_TASK_TYPES[missing_kind],
          placeholder=PlaceholderNotice(
              title="No local results loaded",
              detail=(
                  "This section has no measured artifact data or stored review"
                  " summary in the loaded files for this artifact."
              ),
          ),
      )
      cards.append(_serialize_card(artifact, placeholder_section))
  return cards


def _render_artifact_inventory(artifacts: Sequence[EvaluationArtifact]) -> str:
  if not artifacts:
    return (
        '<div class="note">No local artifact files were loaded. Render the page'
        " again after exporting evaluation JSON files.</div>"
    )
  items = "".join(
      (
          "<li>"
          f"<strong>{escape(artifact.title)}</strong>"
          f"<span class=\"path\">{escape(artifact.source_path or 'n/a')}</span>"
          "</li>"
      )
      for artifact in artifacts
  )
  return f"<ul>{items}</ul>"


def _serialize_card(
    artifact: EvaluationArtifact, section: EvaluationSection
) -> dict[str, Any]:
  bindings = section.registry_bindings or artifact.registry_bindings
  return {
      "artifactId": artifact.artifact_id,
      "artifactTitle": artifact.title,
      "capturedAt": artifact.captured_at,
      "sourcePath": artifact.source_path,
      "cameraId": section.camera_id or artifact.camera_id,
      "scenarioId": section.scenario_id or artifact.scenario_id,
      "pipelineSnapshotId": artifact.pipeline_snapshot_id,
      "kind": section.kind.value,
      "kindLabel": SECTION_TITLES[section.kind],
      "taskType": section.task_type.value,
      "title": section.title,
      "summaryText": section.summary_text,
      "measuredMetrics": [
          {
              "name": metric.name,
              "value": metric.value,
              "unit": metric.unit,
              "sampleSize": metric.sample_size,
              "note": metric.note,
          }
          for metric in section.measured_metrics
      ],
      "manualSummaries": [
          {
              "title": summary.title,
              "summary": summary.summary,
              "reviewer": summary.reviewer,
              "status": summary.status,
              "reviewedAt": summary.reviewed_at,
          }
          for summary in section.manual_summaries
      ],
      "samples": [
          {
              "sampleId": sample.sample_id,
              "label": sample.label,
              "expectedValue": sample.expected_value,
              "observedValue": sample.observed_value,
              "score": sample.score,
              "passed": sample.passed,
              "note": sample.note,
              "mediaReference": sample.media_reference,
          }
          for sample in section.samples
      ],
      "validationScenarios": [
          {
              "scenarioId": scenario.scenario_id,
              "title": scenario.title,
              "expectedOutcome": scenario.expected_outcome,
              "actualOutcome": scenario.actual_outcome,
              "passed": scenario.passed,
              "note": scenario.note,
          }
          for scenario in section.validation_scenarios
      ],
      "placeholder": (
          {
              "title": section.placeholder.title,
              "detail": section.placeholder.detail,
          }
          if section.placeholder is not None
          else None
      ),
      "bindings": [
          {
              "entryId": binding.entry_id,
              "display": binding.display_name,
              "taskType": binding.task_type.value,
              "family": binding.family.value,
          }
          for binding in bindings
      ],
      "bindingKeys": [binding.display_name for binding in bindings],
  }
