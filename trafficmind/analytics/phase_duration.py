"""Phase duration analysis.

Computes how long a phase spends in each state over a time window
by walking consecutive signal-state observations and measuring the
gaps between transitions.
"""

from __future__ import annotations

from collections import defaultdict

from trafficmind.analytics.models import PhaseDurationSummary
from trafficmind.analytics.models import TimeWindow
from trafficmind.models import SignalState


def compute_phase_durations(
    history: list[SignalState],
    window: TimeWindow,
) -> PhaseDurationSummary:
  """Compute a ``PhaseDurationSummary`` from ordered history.

  *history* must be sorted by timestamp. Durations are inferred from
  consecutive timestamps, so sparse or irregular observations can
  over-attribute time to the previously observed state.
  """
  if not history:
    return PhaseDurationSummary(
        junction_id="",
        phase_id="",
        window=window,
        state_durations={},
        state_counts={},
        mean_durations={},
        total_observations=0,
        coverage_ratio=0.0,
        partial_data=True,
    )

  junction_id = history[0].junction_id
  phase_id = history[0].phase_id

  durations: dict[str, float] = defaultdict(float)
  counts: dict[str, int] = defaultdict(int)
  covered = 0.0

  # Walk consecutive pairs.
  prev = history[0]
  counts[prev.state.value] += 1
  for cur in history[1:]:
    dt = cur.timestamp - prev.timestamp
    if dt >= 0:
      durations[prev.state.value] += dt
      covered += dt
    if cur.state != prev.state:
      counts[cur.state.value] += 1
    prev = cur

  total_time = window.duration
  coverage = covered / total_time if total_time and total_time > 0 else 0.0
  partial = coverage < 0.95

  mean_dur: dict[str, float] = {}
  for state_val, total_dur in durations.items():
    c = counts.get(state_val, 1)
    mean_dur[state_val] = total_dur / c if c > 0 else 0.0

  return PhaseDurationSummary(
      junction_id=junction_id,
      phase_id=phase_id,
      window=window,
      state_durations=dict(durations),
      state_counts=dict(counts),
      mean_durations=mean_dur,
      total_observations=len(history),
      coverage_ratio=min(coverage, 1.0),
      partial_data=partial,
  )
