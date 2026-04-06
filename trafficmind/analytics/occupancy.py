"""Red/green occupancy correlation analysis.

Measures the fraction of time a phase spends in restrictive vs
permissive states and estimates queue pressure from the length of
consecutive red runs.
"""

from __future__ import annotations

from trafficmind.analytics.models import OccupancyCorrelation
from trafficmind.analytics.models import TimeWindow
from trafficmind.models import PhaseState
from trafficmind.models import RESTRICTIVE_STATES
from trafficmind.models import SignalState

# States considered "green" / permissive for occupancy purposes.
_GREEN_STATES = frozenset({PhaseState.GREEN})


def compute_occupancy_correlation(
    history: list[SignalState],
    window: TimeWindow,
) -> OccupancyCorrelation:
  """Compute red/green occupancy fractions and queue-pressure proxy.

  A "queue event" is any contiguous run of restrictive states that
  is followed by a green.
  """
  if not history:
    return OccupancyCorrelation(
        junction_id="",
        phase_id="",
        window=window,
        red_total_seconds=0.0,
        green_total_seconds=0.0,
        red_fraction=0.0,
        green_fraction=0.0,
        estimated_queue_events=0,
        mean_red_before_green=0.0,
        partial_data=True,
    )

  junction_id = history[0].junction_id
  phase_id = history[0].phase_id

  red_total = 0.0
  green_total = 0.0
  covered = 0.0

  # Track contiguous red runs to estimate queue events.
  queue_events = 0
  red_before_green_sums = 0.0
  current_red_run = 0.0
  in_red_run = history[0].state in RESTRICTIVE_STATES

  prev = history[0]
  for cur in history[1:]:
    dt = cur.timestamp - prev.timestamp
    if dt < 0:
      prev = cur
      continue
    covered += dt

    if prev.state in RESTRICTIVE_STATES:
      red_total += dt
      current_red_run += dt
    elif prev.state in _GREEN_STATES:
      green_total += dt

    # Detect red→green transition.
    if prev.state in RESTRICTIVE_STATES and cur.state in _GREEN_STATES:
      queue_events += 1
      red_before_green_sums += current_red_run
      current_red_run = 0.0
      in_red_run = False
    elif cur.state in RESTRICTIVE_STATES:
      if not in_red_run:
        current_red_run = 0.0
        in_red_run = True
    else:
      in_red_run = False
      current_red_run = 0.0

    prev = cur

  total = covered or 1.0

  return OccupancyCorrelation(
      junction_id=junction_id,
      phase_id=phase_id,
      window=window,
      red_total_seconds=red_total,
      green_total_seconds=green_total,
      red_fraction=red_total / total,
      green_fraction=green_total / total,
      estimated_queue_events=queue_events,
      mean_red_before_green=(
          red_before_green_sums / queue_events if queue_events else 0.0
      ),
      partial_data=(covered / (window.duration or 1.0)) < 0.95,
  )
