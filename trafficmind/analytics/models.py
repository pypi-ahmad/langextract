"""Typed output models for signal and phase analytics.

All analytics results are frozen dataclasses suitable for
JSON serialisation, dashboard rendering, or report generation.
"""

from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------------
# Time-window specification
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TimeWindow:
  """Half-open interval [start, end) in Unix epoch seconds.

  Either bound may be ``None`` to leave that side open.
  """

  start: float | None = None
  end: float | None = None

  def contains(self, ts: float) -> bool:
    if self.start is not None and ts < self.start:
      return False
    if self.end is not None and ts >= self.end:
      return False
    return True

  @property
  def duration(self) -> float | None:
    """Window duration in seconds, or None if unbounded."""
    if self.start is not None and self.end is not None:
      return self.end - self.start
    return None


# ------------------------------------------------------------------
# Phase duration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseDurationSummary:
  """Statistics for how long a phase spent in each state."""

  junction_id: str
  phase_id: str
  window: TimeWindow
  state_durations: dict[str, float]  # state value → total seconds
  state_counts: dict[str, int]  # state value → transition count
  mean_durations: dict[str, float]  # state value → mean seconds per occurrence
  total_observations: int
  coverage_ratio: float  # fraction of window covered by data
  partial_data: bool  # True when data gaps may affect accuracy
  source_types_seen: tuple[str, ...] = ()
  camera_id: str | None = None
  assumptions: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Occupancy correlation
# ------------------------------------------------------------------


@dataclass(frozen=True)
class OccupancyCorrelation:
  """Red/green occupancy profile with transition-derived queue proxy.

  ``red_fraction`` and ``green_fraction`` are proportions of total
  observed time (not just red+green time) so they may sum to less
  than 1.0 when AMBER, UNKNOWN, or OFF states are present.

  ``estimated_queue_events`` counts red→green transitions — a proxy
  for queue-service cycles, **not** a direct queue-length or vehicle
  count.
  """

  junction_id: str
  phase_id: str
  window: TimeWindow
  red_total_seconds: float
  green_total_seconds: float
  red_fraction: float
  green_fraction: float
  estimated_queue_events: int
  mean_red_before_green: float  # avg red duration preceding each green
  partial_data: bool
  source_types_seen: tuple[str, ...] = ()
  camera_id: str | None = None
  assumptions: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Queue discharge
# ------------------------------------------------------------------


@dataclass(frozen=True)
class QueueDischargeProfile:
  """Green-phase duration per red→green cycle.

  ``discharge_durations`` lists the time (seconds) from each
  red→green transition to the next non-green state.  This measures
  green-phase duration per cycle — a proxy that correlates with
  queue service time but includes time after the queue clears.
  """

  junction_id: str
  phase_id: str
  window: TimeWindow
  transition_count: int
  discharge_durations: list[float]
  mean_discharge: float | None  # None if no transitions
  median_discharge: float | None
  max_discharge: float | None
  partial_data: bool
  source_types_seen: tuple[str, ...] = ()
  camera_id: str | None = None
  assumptions: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Oversaturation
# ------------------------------------------------------------------


@dataclass(frozen=True)
class OversaturationIndicator:
  """Flags recurring short green cycles at a phase.

  A short-green event is detected when a phase cycles
  RED → GREEN → (non-green) and the green duration falls below
  ``short_green_threshold_seconds``.  This pattern *may* indicate
  demand exceeding capacity, but it can also result from
  controller timing plans that deliberately favour other phases.
  """

  junction_id: str
  phase_id: str
  window: TimeWindow
  cycle_count: int
  short_green_count: int
  short_green_threshold_seconds: float
  short_green_ratio: float  # short_green_count / cycle_count
  mean_green_duration: float | None
  recurring: bool  # True when short_green_ratio ≥ 0.5
  partial_data: bool
  source_types_seen: tuple[str, ...] = ()
  camera_id: str | None = None
  assumptions: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Phase violation trends
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ViolationTrendPoint:
  """A single data-point in a violation trend series."""

  bucket_start: float
  bucket_end: float
  conflict_count: int
  stale_count: int
  unknown_count: int
  total_observations: int


@dataclass(frozen=True)
class PhaseViolationTrend:
  """Time-series of signal anomalies (conflicts, staleness, unknowns).

  Each bucket covers a fixed-width interval.  "Violations" here
  means *signal-data anomalies*, not traffic-law violations.
  """

  junction_id: str
  phase_id: str
  window: TimeWindow
  bucket_seconds: float
  buckets: list[ViolationTrendPoint]
  total_conflicts: int
  total_stale: int
  total_unknown: int
  partial_data: bool
  source_types_seen: tuple[str, ...] = ()
  camera_id: str | None = None
  assumptions: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Junction summary
# ------------------------------------------------------------------


@dataclass(frozen=True)
class JunctionAnalyticsSummary:
  """Roll-up of all analytics for one junction across all phases."""

  junction_id: str
  window: TimeWindow
  phase_durations: list[PhaseDurationSummary]
  occupancy_correlations: list[OccupancyCorrelation]
  queue_discharges: list[QueueDischargeProfile]
  oversaturation_indicators: list[OversaturationIndicator]
  violation_trends: list[PhaseViolationTrend]
  partial_data: bool
  source_types_seen: tuple[str, ...] = ()
  assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CameraAnalyticsSummary:
  """Roll-up of all analytics for one camera across all phases."""

  junction_id: str
  camera_id: str
  window: TimeWindow
  phase_durations: list[PhaseDurationSummary]
  occupancy_correlations: list[OccupancyCorrelation]
  queue_discharges: list[QueueDischargeProfile]
  oversaturation_indicators: list[OversaturationIndicator]
  violation_trends: list[PhaseViolationTrend]
  partial_data: bool
  source_types_seen: tuple[str, ...] = ("vision",)
  assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseAnalyticsComparison:
  """Typed side-by-side comparison for two time windows."""

  junction_id: str
  phase_id: str
  window_a: TimeWindow
  window_b: TimeWindow
  phase_durations: tuple[PhaseDurationSummary, PhaseDurationSummary]
  occupancy_correlations: tuple[OccupancyCorrelation, OccupancyCorrelation]
  queue_discharges: tuple[QueueDischargeProfile, QueueDischargeProfile]
  oversaturation_indicators: tuple[
      OversaturationIndicator, OversaturationIndicator
  ]
  violation_trends: tuple[PhaseViolationTrend, PhaseViolationTrend]
  camera_id: str | None = None
  source_types_seen: tuple[str, ...] = ()
  assumptions: tuple[str, ...] = ()
