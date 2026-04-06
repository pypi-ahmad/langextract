"""Phase-specific violation / anomaly trend analysis.

Buckets signal-state history into fixed-width time intervals and
counts signal-data anomalies (conflicts between sources, stale
observations, unknown states) per bucket.

"Violations" here means *signal-data anomalies*, not traffic-law
violations.
"""

from __future__ import annotations

import math

from trafficmind.analytics.models import PhaseViolationTrend
from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.models import ViolationTrendPoint
from trafficmind.models import PhaseState
from trafficmind.models import SignalState

# Default bucket width for trend bucketing.
DEFAULT_BUCKET_SECONDS: float = 300.0  # 5 minutes


def compute_violation_trend(
    history: list[SignalState],
    window: TimeWindow,
    *,
    bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
) -> PhaseViolationTrend:
  """Build a time-bucketed anomaly trend from signal history.

  Each observation is classified as:
  - **conflict**: the observation's ``metadata`` contains a truthy
    ``"conflict"`` key (set by upstream arbitration logging).
  - **stale**: the observation's ``metadata`` contains a truthy
    ``"stale"`` key.
  - **unknown**: the observation has ``state == UNKNOWN``.

  These are counted per bucket to show trends over time.
  """
  if not history:
    return PhaseViolationTrend(
        junction_id="",
        phase_id="",
        window=window,
        bucket_seconds=bucket_seconds,
        buckets=[],
        total_conflicts=0,
        total_stale=0,
        total_unknown=0,
        partial_data=True,
    )

  junction_id = history[0].junction_id
  phase_id = history[0].phase_id

  # Determine actual time span.
  ts_min = history[0].timestamp
  ts_max = history[-1].timestamp
  w_start = window.start if window.start is not None else ts_min
  w_end = window.end if window.end is not None else ts_max + bucket_seconds

  n_buckets = max(1, math.ceil((w_end - w_start) / bucket_seconds))

  # Initialise buckets.
  buckets: list[dict[str, int]] = [
      {"conflict": 0, "stale": 0, "unknown": 0, "total": 0}
      for _ in range(n_buckets)
  ]

  for obs in history:
    idx = int((obs.timestamp - w_start) / bucket_seconds)
    if idx < 0 or idx >= n_buckets:
      continue
    b = buckets[idx]
    b["total"] += 1
    if obs.metadata.get("conflict"):
      b["conflict"] += 1
    if obs.metadata.get("stale"):
      b["stale"] += 1
    if obs.state == PhaseState.UNKNOWN:
      b["unknown"] += 1

  points: list[ViolationTrendPoint] = []
  for i, b in enumerate(buckets):
    points.append(
        ViolationTrendPoint(
            bucket_start=w_start + i * bucket_seconds,
            bucket_end=w_start + (i + 1) * bucket_seconds,
            conflict_count=b["conflict"],
            stale_count=b["stale"],
            unknown_count=b["unknown"],
            total_observations=b["total"],
        )
    )

  total_conflicts = sum(b["conflict"] for b in buckets)
  total_stale = sum(b["stale"] for b in buckets)
  total_unknown = sum(b["unknown"] for b in buckets)

  covered = ts_max - ts_min if len(history) > 1 else 0.0
  win_dur = window.duration or 1.0

  return PhaseViolationTrend(
      junction_id=junction_id,
      phase_id=phase_id,
      window=window,
      bucket_seconds=bucket_seconds,
      buckets=points,
      total_conflicts=total_conflicts,
      total_stale=total_stale,
      total_unknown=total_unknown,
      partial_data=(covered / win_dur) < 0.95,
  )
