"""Analytics engine — facade for all signal/phase analytics.

Pulls history from a ``SignalStore`` and runs the requested analytics
over configurable time windows.
"""

from __future__ import annotations

from dataclasses import replace

from trafficmind.analytics.models import CameraAnalyticsSummary
from trafficmind.analytics.models import JunctionAnalyticsSummary
from trafficmind.analytics.models import OccupancyCorrelation
from trafficmind.analytics.models import OversaturationIndicator
from trafficmind.analytics.models import PhaseAnalyticsComparison
from trafficmind.analytics.models import PhaseDurationSummary
from trafficmind.analytics.models import PhaseViolationTrend
from trafficmind.analytics.models import QueueDischargeProfile
from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.occupancy import compute_occupancy_correlation
from trafficmind.analytics.oversaturation import compute_oversaturation
from trafficmind.analytics.oversaturation import DEFAULT_SHORT_GREEN_SECONDS
from trafficmind.analytics.phase_duration import compute_phase_durations
from trafficmind.analytics.queue_discharge import compute_queue_discharge
from trafficmind.analytics.violation_trend import compute_violation_trend
from trafficmind.analytics.violation_trend import DEFAULT_BUCKET_SECONDS
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.store import SignalStore


class AnalyticsEngine:
  """High-level facade for running analytics against stored signal data.

  Parameters:
      store: The ``SignalStore`` to read history from.
  """

  def __init__(self, store: SignalStore) -> None:
    self.store = store

  # ------------------------------------------------------------------
  # Per-phase analytics
  # ------------------------------------------------------------------

  def phase_durations(
      self,
      junction_id: str,
      phase_id: str,
      window: TimeWindow,
      *,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> PhaseDurationSummary:
    history = self._history(
        junction_id,
        phase_id,
        window,
        source_types=source_types,
        camera_id=camera_id,
    )
    result = compute_phase_durations(history, window)
    return self._contextualize(
        result,
        history,
        camera_id=camera_id,
        source_types=source_types,
    )

  def occupancy_correlation(
      self,
      junction_id: str,
      phase_id: str,
      window: TimeWindow,
      *,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> OccupancyCorrelation:
    history = self._history(
        junction_id,
        phase_id,
        window,
        source_types=source_types,
        camera_id=camera_id,
    )
    result = compute_occupancy_correlation(history, window)
    return self._contextualize(
        result,
        history,
        camera_id=camera_id,
        source_types=source_types,
        proxy_note=(
            "Queue-event metrics are signal-state proxies, not direct "
            "queue-length measurements."
        ),
    )

  def queue_discharge(
      self,
      junction_id: str,
      phase_id: str,
      window: TimeWindow,
      *,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> QueueDischargeProfile:
    history = self._history(
        junction_id,
        phase_id,
        window,
        source_types=source_types,
        camera_id=camera_id,
    )
    result = compute_queue_discharge(history, window)
    return self._contextualize(
        result,
        history,
        camera_id=camera_id,
        source_types=source_types,
        proxy_note=(
            "Queue-discharge durations approximate green service time and "
            "are not direct queue-clearance measurements."
        ),
    )

  def oversaturation(
      self,
      junction_id: str,
      phase_id: str,
      window: TimeWindow,
      *,
      short_green_threshold: float = DEFAULT_SHORT_GREEN_SECONDS,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> OversaturationIndicator:
    history = self._history(
        junction_id,
        phase_id,
        window,
        source_types=source_types,
        camera_id=camera_id,
    )
    result = compute_oversaturation(
        history,
        window,
        short_green_threshold=short_green_threshold,
    )
    return self._contextualize(
        result,
        history,
        camera_id=camera_id,
        source_types=source_types,
        proxy_note=(
            "Short-green detection is based on cycle timing patterns only; "
            "it does not measure actual traffic demand or queue length."
        ),
    )

  def violation_trend(
      self,
      junction_id: str,
      phase_id: str,
      window: TimeWindow,
      *,
      bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> PhaseViolationTrend:
    history = self._history(
        junction_id,
        phase_id,
        window,
        source_types=source_types,
        camera_id=camera_id,
    )
    result = compute_violation_trend(
        history,
        window,
        bucket_seconds=bucket_seconds,
    )
    return self._contextualize(
        result,
        history,
        camera_id=camera_id,
        source_types=source_types,
        proxy_note=(
            "Violation trends represent signal-data anomalies, not traffic-law"
            " violations."
        ),
    )

  # ------------------------------------------------------------------
  # Junction-level roll-up
  # ------------------------------------------------------------------

  def junction_summary(
      self,
      junction_id: str,
      window: TimeWindow,
      *,
      short_green_threshold: float = DEFAULT_SHORT_GREEN_SECONDS,
      bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
      source_types: frozenset[SourceType] | None = None,
  ) -> JunctionAnalyticsSummary:
    """Run all analytics for every phase in *junction_id*."""
    phases = sorted(self.store.phase_ids_for_junction(junction_id))
    dur_list: list[PhaseDurationSummary] = []
    occ_list: list[OccupancyCorrelation] = []
    qd_list: list[QueueDischargeProfile] = []
    os_list: list[OversaturationIndicator] = []
    vt_list: list[PhaseViolationTrend] = []

    any_partial = len(phases) == 0
    for pid in phases:
      dur = self.phase_durations(
          junction_id,
          pid,
          window,
          source_types=source_types,
      )
      occ = self.occupancy_correlation(
          junction_id,
          pid,
          window,
          source_types=source_types,
      )
      qd = self.queue_discharge(
          junction_id,
          pid,
          window,
          source_types=source_types,
      )
      ov = self.oversaturation(
          junction_id,
          pid,
          window,
          short_green_threshold=short_green_threshold,
          source_types=source_types,
      )
      vt = self.violation_trend(
          junction_id,
          pid,
          window,
          bucket_seconds=bucket_seconds,
          source_types=source_types,
      )
      dur_list.append(dur)
      occ_list.append(occ)
      qd_list.append(qd)
      os_list.append(ov)
      vt_list.append(vt)
      if any(r.partial_data for r in (dur, occ, qd, ov, vt)):
        any_partial = True

    return JunctionAnalyticsSummary(
        junction_id=junction_id,
        window=window,
        phase_durations=dur_list,
        occupancy_correlations=occ_list,
        queue_discharges=qd_list,
        oversaturation_indicators=os_list,
        violation_trends=vt_list,
        partial_data=any_partial,
        source_types_seen=self._selected_source_types(
            source_types, dur_list + occ_list + qd_list + os_list + vt_list
        ),
        assumptions=self._summary_assumptions(
            any_partial, source_types=source_types
        ),
    )

  def camera_summary(
      self,
      junction_id: str,
      camera_id: str,
      window: TimeWindow,
      *,
      short_green_threshold: float = DEFAULT_SHORT_GREEN_SECONDS,
      bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
  ) -> CameraAnalyticsSummary:
    """Run all analytics for a single camera across visible phases."""
    phases = sorted(self.store.phase_ids_for_camera(junction_id, camera_id))
    dur_list: list[PhaseDurationSummary] = []
    occ_list: list[OccupancyCorrelation] = []
    qd_list: list[QueueDischargeProfile] = []
    os_list: list[OversaturationIndicator] = []
    vt_list: list[PhaseViolationTrend] = []

    any_partial = len(phases) == 0
    source_types = frozenset({SourceType.VISION})
    for pid in phases:
      history = self._history(
          junction_id,
          pid,
          window,
          source_types=source_types,
          camera_id=camera_id,
      )
      if not history:
        continue
      dur = self._contextualize(
          compute_phase_durations(history, window),
          history,
          camera_id=camera_id,
          source_types=source_types,
      )
      occ = self._contextualize(
          compute_occupancy_correlation(history, window),
          history,
          camera_id=camera_id,
          source_types=source_types,
          proxy_note=(
              "Queue-event metrics are signal-state proxies, not direct "
              "queue-length measurements."
          ),
      )
      qd = self._contextualize(
          compute_queue_discharge(history, window),
          history,
          camera_id=camera_id,
          source_types=source_types,
          proxy_note=(
              "Queue-discharge durations approximate green service time and "
              "are not direct queue-clearance measurements."
          ),
      )
      ov = self._contextualize(
          compute_oversaturation(
              history,
              window,
              short_green_threshold=short_green_threshold,
          ),
          history,
          camera_id=camera_id,
          source_types=source_types,
          proxy_note=(
              "Short-green detection is based on cycle timing patterns only; "
              "it does not measure actual traffic demand or queue length."
          ),
      )
      vt = self._contextualize(
          compute_violation_trend(
              history,
              window,
              bucket_seconds=bucket_seconds,
          ),
          history,
          camera_id=camera_id,
          source_types=source_types,
          proxy_note=(
              "Violation trends represent signal-data anomalies, not"
              " traffic-law violations."
          ),
      )
      dur_list.append(dur)
      occ_list.append(occ)
      qd_list.append(qd)
      os_list.append(ov)
      vt_list.append(vt)
      if any(r.partial_data for r in (dur, occ, qd, ov, vt)):
        any_partial = True

    return CameraAnalyticsSummary(
        junction_id=junction_id,
        camera_id=camera_id,
        window=window,
        phase_durations=dur_list,
        occupancy_correlations=occ_list,
        queue_discharges=qd_list,
        oversaturation_indicators=os_list,
        violation_trends=vt_list,
        partial_data=any_partial,
        assumptions=self._summary_assumptions(
            any_partial,
            source_types=source_types,
            camera_id=camera_id,
        ),
    )

  # ------------------------------------------------------------------
  # Comparison helper
  # ------------------------------------------------------------------

  def compare_windows(
      self,
      junction_id: str,
      phase_id: str,
      window_a: TimeWindow,
      window_b: TimeWindow,
      *,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> PhaseAnalyticsComparison:
    """Return typed side-by-side analytics for two time windows."""
    phase_durations = (
        self.phase_durations(
            junction_id,
            phase_id,
            window_a,
            source_types=source_types,
            camera_id=camera_id,
        ),
        self.phase_durations(
            junction_id,
            phase_id,
            window_b,
            source_types=source_types,
            camera_id=camera_id,
        ),
    )
    occupancy_correlations = (
        self.occupancy_correlation(
            junction_id,
            phase_id,
            window_a,
            source_types=source_types,
            camera_id=camera_id,
        ),
        self.occupancy_correlation(
            junction_id,
            phase_id,
            window_b,
            source_types=source_types,
            camera_id=camera_id,
        ),
    )
    queue_discharges = (
        self.queue_discharge(
            junction_id,
            phase_id,
            window_a,
            source_types=source_types,
            camera_id=camera_id,
        ),
        self.queue_discharge(
            junction_id,
            phase_id,
            window_b,
            source_types=source_types,
            camera_id=camera_id,
        ),
    )
    oversaturation_indicators = (
        self.oversaturation(
            junction_id,
            phase_id,
            window_a,
            source_types=source_types,
            camera_id=camera_id,
        ),
        self.oversaturation(
            junction_id,
            phase_id,
            window_b,
            source_types=source_types,
            camera_id=camera_id,
        ),
    )
    violation_trends = (
        self.violation_trend(
            junction_id,
            phase_id,
            window_a,
            source_types=source_types,
            camera_id=camera_id,
        ),
        self.violation_trend(
            junction_id,
            phase_id,
            window_b,
            source_types=source_types,
            camera_id=camera_id,
        ),
    )
    source_types_seen = self._selected_source_types(
        source_types,
        [
            *phase_durations,
            *occupancy_correlations,
            *queue_discharges,
            *oversaturation_indicators,
            *violation_trends,
        ],
    )
    return PhaseAnalyticsComparison(
        junction_id=junction_id,
        phase_id=phase_id,
        window_a=window_a,
        window_b=window_b,
        phase_durations=phase_durations,
        occupancy_correlations=occupancy_correlations,
        queue_discharges=queue_discharges,
        oversaturation_indicators=oversaturation_indicators,
        violation_trends=violation_trends,
        camera_id=camera_id,
        source_types_seen=source_types_seen,
        assumptions=self._summary_assumptions(
            any(
                result.partial_data
                for result in [
                    *phase_durations,
                    *occupancy_correlations,
                    *queue_discharges,
                    *oversaturation_indicators,
                    *violation_trends,
                ]
            ),
            source_types=source_types,
            camera_id=camera_id,
        ),
    )

  def _history(
      self,
      junction_id: str,
      phase_id: str,
      window: TimeWindow,
      *,
      source_types: frozenset[SourceType] | None,
      camera_id: str | None,
  ) -> list[SignalState]:
    if camera_id is not None and source_types is None:
      source_types = frozenset({SourceType.VISION})
    return self.store.all_history_window(
        junction_id,
        phase_id,
        start=window.start,
        end=window.end,
        source_types=source_types,
        camera_id=camera_id,
    )

  def _contextualize(
      self,
      result,
      history: list[SignalState],
      *,
      camera_id: str | None,
      source_types: frozenset[SourceType] | None,
      proxy_note: str | None = None,
  ):
    assumptions = list(
        self._assumptions(
            history,
            result.partial_data,
            source_types=source_types,
            camera_id=camera_id,
        )
    )
    if proxy_note is not None:
      assumptions.append(proxy_note)
    return replace(
        result,
        source_types_seen=self._source_types_seen(history, source_types),
        camera_id=camera_id,
        assumptions=tuple(assumptions),
    )

  @staticmethod
  def _source_types_seen(
      history: list[SignalState],
      source_types: frozenset[SourceType] | None,
  ) -> tuple[str, ...]:
    if source_types is not None:
      return tuple(sorted(source_type.value for source_type in source_types))
    return tuple(sorted({state.source_type.value for state in history}))

  def _selected_source_types(
      self,
      source_types: frozenset[SourceType] | None,
      results: list,
  ) -> tuple[str, ...]:
    if source_types is not None:
      return tuple(sorted(source_type.value for source_type in source_types))
    return tuple(
        sorted({
            source_type
            for result in results
            for source_type in getattr(result, "source_types_seen", ())
        })
    )

  @staticmethod
  def _assumptions(
      history: list[SignalState],
      partial_data: bool,
      *,
      source_types: frozenset[SourceType] | None,
      camera_id: str | None,
  ) -> tuple[str, ...]:
    assumptions: list[str] = []
    if not history:
      assumptions.append(
          "No signal observations were available inside the requested window."
      )
    if partial_data:
      assumptions.append(
          "Partial coverage: observations cover less than 95% of the requested"
          " window; totals should be treated as lower-bound estimates."
      )
    if camera_id is not None:
      assumptions.append(
          "Camera view uses vision observations only; camera id is taken from"
          " metadata['camera_id'] or a legacy vision controller_id fallback."
      )
    elif (
        source_types is None
        and len({state.source_type for state in history}) > 1
    ):
      assumptions.append(
          "This view mixes raw observations from multiple source types. Use"
          " source_types or camera_id filters for a source-specific view."
      )
    return tuple(assumptions)

  @staticmethod
  def _summary_assumptions(
      partial_data: bool,
      *,
      source_types: frozenset[SourceType] | None,
      camera_id: str | None = None,
  ) -> tuple[str, ...]:
    assumptions: list[str] = []
    if partial_data:
      assumptions.append(
          "At least one requested phase had partial or missing signal coverage"
          " in the selected window."
      )
    if camera_id is not None:
      assumptions.append(
          "Camera summaries are derived from vision observations only and"
          " reflect the selected camera's field of view."
      )
    elif source_types is None:
      assumptions.append(
          "Junction summaries may mix observations from controller-fed and"
          " vision-derived sources unless a source_types filter is supplied."
      )
    return tuple(assumptions)
