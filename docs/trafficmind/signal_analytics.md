# Signal & Phase Analytics

## What This Is

Descriptive, operational analytics derived from signal-state history.
These analytics help **operators and planners** understand junction
behaviour and phase efficiency.  They do **not** implement adaptive
signal control or automated optimisation.

## Available Analytics

### 1. Phase Duration Summary

**What it measures:** Total and average time a phase spends in each
state (RED, GREEN, AMBER, etc.) over a time window.

**How to interpret:**
- Large totals in `state_durations["red"]` or long average red runs in
  `mean_durations["red"]` indicate the phase spent a substantial share
  of the observed window in a restrictive state.
- Uneven `mean_durations` across phases at the same junction highlight
  phasing imbalances.
- `coverage_ratio < 1.0` means signal data has gaps in the window —
  treat total durations as lower-bound estimates and mean durations as
  approximate summaries of the observed samples.

**Assumptions:** Durations are computed from consecutive observations.
If observations arrive at irregular intervals the duration of the
*last* state in a gap is attributed to the gap length, which may
overcount that state.

```python
from trafficmind.analytics import AnalyticsEngine, TimeWindow
from trafficmind.store import SignalStore

engine = AnalyticsEngine(SignalStore())
result = engine.phase_durations("J-42", "P-1", TimeWindow(t0, t1))
# result.state_durations  →  {"red": 120.0, "green": 80.0, ...}
# result.mean_durations   →  {"red": 40.0, "green": 26.7}
# result.coverage_ratio   →  0.93
```

---

### 2. Red/Green Occupancy Profile

**What it measures:** Fraction of total observed time the phase
spends in a restrictive (red/stop) vs permissive (green) state,
plus red→green transition counts.

**How to interpret:**
- `red_fraction` and `green_fraction` are proportions of *total
  observed time*, so they may sum to less than 1.0 when the phase
  spends time in AMBER, UNKNOWN, or OFF states.
- `red_fraction` close to 1.0 means the phase is almost always
  stopped — potential capacity starvation.
- `estimated_queue_events` counts red→green transitions. A high count
  with short greens suggests frequent but brief service cycles.
- `mean_red_before_green` is a proxy for average wait time.

**Queue proxy caveat:** This is *not* a direct queue length or vehicle
count measurement.  It counts signal-state transitions.  Use it as a
relative indicator between phases or time periods, not an absolute
queue metric.

---

### 3. Queue Discharge Profile

**What it measures:** How long the green phase lasts after each
red→green transition.  This is the green-phase duration per cycle,
which correlates with but is not equivalent to queue clearance time.

**How to interpret:**
- Short `mean_discharge` with many transitions suggests brief green
  service windows.
- Long `max_discharge` outliers show unusually long green periods in
  the observed data.
- Compare `median_discharge` across time windows to see whether typical
  green service length is changing.

**Assumptions:** "Discharge duration" is measured from the start of
green to the first non-green state.  This measures the full green-phase
duration, which includes time beyond actual queue clearance.  Use
it as a relative indicator, not a direct queue-clearance measurement.

---

### 4. Oversaturation Indicator

**What it measures:** How often the green phase is shorter than a
configurable threshold, based on the fraction of cycles where green
duration falls below that threshold.

**How to interpret:**
- `recurring = True` (≥ 50 % short-green cycles) indicates a
  pattern worth investigating — it *may* reflect demand exceeding
  capacity, or a timing plan that deliberately favours other phases.
- `short_green_ratio` below 0.5 but above 0 suggests occasional
  short greens.
- Adjust `short_green_threshold` based on local knowledge of
  expected green service lengths for that phase.

**NOT adaptive control:** This flags a pattern for a human to
investigate.  It does not automatically extend green times or
diagnose the root cause.

---

### 5. Phase Violation Trend

**What it measures:** Time-bucketed counts of signal-data anomalies:
conflicts between controller and vision sources, stale observations,
and UNKNOWN states.

**How to interpret:**
- Rising `conflict_count` over successive buckets may indicate a
  sensor drift or controller misconfiguration.
- Persistent `stale_count` suggests a connectivity issue with a
  source.
- `unknown_count` spikes may correlate with controller outages.

**"Violation" ≠ traffic law violation.** These are *data quality*
anomalies, not vehicle infractions.

---

## Time Windows & Comparison

All analytics accept a `TimeWindow(start, end)` specified in Unix
epoch seconds.  Either bound may be `None` for an open-ended query.

To compare two periods (e.g. morning peak vs evening peak):

```python
morning = TimeWindow(t_morning_start, t_morning_end)
evening = TimeWindow(t_evening_start, t_evening_end)
comparison = engine.compare_windows("J-42", "P-1", morning, evening)
# comparison.phase_durations → (morning_result, evening_result)
```

---

## Junction-Level Summary

`engine.junction_summary("J-42", window)` runs all five analytics for
every phase at the junction and returns a single
`JunctionAnalyticsSummary` with roll-up lists.

---

## Per-Camera / Per-Source Views

Signal observations carry a `source_type` field, and vision records may
also carry `metadata["camera_id"]`. The analytics engine now supports
three common operational views directly:

- **Per-junction view**: `engine.junction_summary(...)`
- **Per-phase filtered by source type**:
  `engine.phase_durations(..., source_types=frozenset({SourceType.CONTROLLER}))`
- **Per-camera view**: `engine.camera_summary("J-42", "CAM-7", window)`

When `camera_id` is provided, the engine automatically uses vision
observations only. Camera id is taken from `metadata["camera_id"]`
when present, with a legacy fallback to `controller_id` for older
vision records.

If no source filter is supplied, the default phase/junction view may
mix raw controller-fed and vision-derived observations. That is useful
for broad operational summaries, but source-specific investigation is
usually clearer with `source_types` or `camera_id` filters.

---

## `partial_data` Flag

Every analytics result carries a `partial_data: bool` field.  It is
`True` when the data covered less than 95 % of the requested time
window. Results also carry an `assumptions` tuple that explains how to
interpret partial, camera-scoped, or proxy-based outputs. When
`partial_data` is set:

- Totals are *lower-bound estimates*.
- Ratios may be biased by non-random data gaps and should not be
  treated as lower-bound estimates.
- Comparisons between windows with different coverage ratios should
  be normalised by `coverage_ratio` before drawing conclusions.
- The flag does **not** mean the results are wrong — just that they
  may be incomplete.

---

## Limitations

- **No real queue-length data.** All queue-related analytics are
  proxies derived from phase-state transitions.
- **No vehicle counts.** The analytics layer operates on signal
  state only.
- **No predictive modelling.** All outputs are descriptive summaries
  of historical data.
- **Irregular observation intervals** can skew duration measurements.
  Higher-frequency data gives more accurate results.
- **No automated signal optimisation.** These analytics identify
  patterns — they do not change signal timings.
