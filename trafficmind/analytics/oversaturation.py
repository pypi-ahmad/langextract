"""Recurring short-green cycle detection.

Identifies cycles where the green phase was shorter than a
configurable threshold.  Recurring short greens *may* indicate
demand exceeding capacity, but can also reflect deliberate
timing plans that allocate green to other phases.
"""

from __future__ import annotations

import statistics

from trafficmind.analytics.models import OversaturationIndicator
from trafficmind.analytics.models import TimeWindow
from trafficmind.models import PhaseState
from trafficmind.models import RESTRICTIVE_STATES
from trafficmind.models import SignalState

_GREEN_STATES = frozenset({PhaseState.GREEN})

# Default threshold below which a green phase is considered "short".
DEFAULT_SHORT_GREEN_SECONDS: float = 10.0


def compute_oversaturation(
    history: list[SignalState],
    window: TimeWindow,
    *,
    short_green_threshold: float = DEFAULT_SHORT_GREEN_SECONDS,
) -> OversaturationIndicator:
  """Detect recurring short-green cycles from signal history.

  A "cycle" is a RED→GREEN→(non-green) sequence.  A cycle is
  "short-green" when the green duration is below
  *short_green_threshold*.  If ≥ 50 % of cycles are short-green
  the phase is flagged with ``recurring=True``, indicating a
  pattern that warrants investigation.
  """
  if not history:
    return OversaturationIndicator(
        junction_id="",
        phase_id="",
        window=window,
        cycle_count=0,
        short_green_count=0,
        short_green_threshold_seconds=short_green_threshold,
        short_green_ratio=0.0,
        mean_green_duration=None,
        recurring=False,
        partial_data=True,
    )

  junction_id = history[0].junction_id
  phase_id = history[0].phase_id

  green_durations: list[float] = []
  green_start: float | None = None

  prev = history[0]
  for cur in history[1:]:
    if prev.state in RESTRICTIVE_STATES and cur.state in _GREEN_STATES:
      green_start = cur.timestamp

    if (
        green_start is not None
        and prev.state in _GREEN_STATES
        and cur.state not in _GREEN_STATES
    ):
      dur = cur.timestamp - green_start
      if dur >= 0:
        green_durations.append(dur)
      green_start = None

    prev = cur

  cycle_count = len(green_durations)
  short_count = sum(1 for d in green_durations if d < short_green_threshold)

  covered = (
      (history[-1].timestamp - history[0].timestamp)
      if len(history) > 1
      else 0.0
  )
  win_dur = window.duration or 1.0

  return OversaturationIndicator(
      junction_id=junction_id,
      phase_id=phase_id,
      window=window,
      cycle_count=cycle_count,
      short_green_count=short_count,
      short_green_threshold_seconds=short_green_threshold,
      short_green_ratio=short_count / cycle_count if cycle_count else 0.0,
      mean_green_duration=(
          statistics.mean(green_durations) if green_durations else None
      ),
      recurring=cycle_count > 0 and (short_count / cycle_count) >= 0.5,
      partial_data=(covered / win_dur) < 0.95,
  )
