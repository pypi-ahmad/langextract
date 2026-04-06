"""Green-phase duration per cycle analysis.

Measures the time from each red→green transition until the phase
next leaves green.  This is the green-phase duration per cycle,
which correlates with but is not equivalent to queue clearance time.
"""

from __future__ import annotations

import statistics

from trafficmind.analytics.models import QueueDischargeProfile
from trafficmind.analytics.models import TimeWindow
from trafficmind.models import PhaseState
from trafficmind.models import RESTRICTIVE_STATES
from trafficmind.models import SignalState

_GREEN_STATES = frozenset({PhaseState.GREEN})


def compute_queue_discharge(
    history: list[SignalState],
    window: TimeWindow,
) -> QueueDischargeProfile:
  """Compute green-phase durations from red→green transitions.

  For each red→green transition, the "discharge duration" is the
  time from the green onset until the phase next leaves green.
  This measures green-phase duration per cycle, which includes
  time beyond actual queue clearance.
  """
  if not history:
    return QueueDischargeProfile(
        junction_id="",
        phase_id="",
        window=window,
        transition_count=0,
        discharge_durations=[],
        mean_discharge=None,
        median_discharge=None,
        max_discharge=None,
        partial_data=True,
    )

  junction_id = history[0].junction_id
  phase_id = history[0].phase_id

  durations: list[float] = []
  green_start: float | None = None

  prev = history[0]
  for cur in history[1:]:
    # Detect red→green.
    if prev.state in RESTRICTIVE_STATES and cur.state in _GREEN_STATES:
      green_start = cur.timestamp

    # Detect end of green.
    if (
        green_start is not None
        and prev.state in _GREEN_STATES
        and cur.state not in _GREEN_STATES
    ):
      dur = cur.timestamp - green_start
      if dur >= 0:
        durations.append(dur)
      green_start = None

    prev = cur

  count = len(durations)
  covered = (
      (history[-1].timestamp - history[0].timestamp)
      if len(history) > 1
      else 0.0
  )
  win_dur = window.duration or 1.0

  return QueueDischargeProfile(
      junction_id=junction_id,
      phase_id=phase_id,
      window=window,
      transition_count=count,
      discharge_durations=durations,
      mean_discharge=statistics.mean(durations) if durations else None,
      median_discharge=statistics.median(durations) if durations else None,
      max_discharge=max(durations) if durations else None,
      partial_data=(covered / win_dur) < 0.95,
  )
