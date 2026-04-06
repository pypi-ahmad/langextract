"""Integration tests for trafficmind.analytics.engine."""

import unittest

from trafficmind.analytics.engine import AnalyticsEngine
from trafficmind.analytics.models import PhaseAnalyticsComparison
from trafficmind.analytics.models import TimeWindow
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.store import SignalStore

T0 = 1000.0


def _s(
    state,
    ts,
    junction="J1",
    phase="P1",
    source=SourceType.CONTROLLER,
    controller_id="C1",
    **meta,
):
  return SignalState(
      junction_id=junction,
      controller_id=controller_id,
      phase_id=phase,
      state=state,
      timestamp=ts,
      source_type=source,
      metadata=meta,
  )


def _make_typical_history():
  """A complete cycle: RED→GREEN→AMBER→RED across 60 seconds."""
  return [
      _s(PhaseState.RED, T0),
      _s(PhaseState.GREEN, T0 + 15),
      _s(PhaseState.AMBER, T0 + 45),
      _s(PhaseState.RED, T0 + 50),
  ]


class TestAnalyticsEngine(unittest.TestCase):

  def _engine_with_history(self, history):
    store = SignalStore(stale_after_seconds=60, history_size=200)
    store.update_batch(history)
    return AnalyticsEngine(store)

  def test_phase_durations(self):
    eng = self._engine_with_history(_make_typical_history())
    r = eng.phase_durations("J1", "P1", TimeWindow(T0, T0 + 50))
    self.assertEqual(r.junction_id, "J1")
    self.assertIn("red", r.state_durations)
    self.assertIn("green", r.state_durations)

  def test_occupancy_correlation(self):
    eng = self._engine_with_history(_make_typical_history())
    r = eng.occupancy_correlation("J1", "P1", TimeWindow(T0, T0 + 50))
    self.assertGreater(r.red_total_seconds, 0)
    self.assertGreater(r.green_total_seconds, 0)
    self.assertEqual(r.estimated_queue_events, 1)

  def test_queue_discharge(self):
    eng = self._engine_with_history(_make_typical_history())
    r = eng.queue_discharge("J1", "P1", TimeWindow(T0, T0 + 50))
    self.assertEqual(r.transition_count, 1)
    self.assertAlmostEqual(r.discharge_durations[0], 30.0)  # green T0+15→T0+45

  def test_oversaturation_not_recurring(self):
    eng = self._engine_with_history(_make_typical_history())
    r = eng.oversaturation(
        "J1", "P1", TimeWindow(T0, T0 + 50), short_green_threshold=10.0
    )
    self.assertEqual(r.cycle_count, 1)
    self.assertEqual(r.short_green_count, 0)
    self.assertFalse(r.recurring)

  def test_violation_trend(self):
    history = [
        _s(PhaseState.RED, T0, conflict=True),
        _s(PhaseState.UNKNOWN, T0 + 10),
        _s(PhaseState.GREEN, T0 + 20),
    ]
    eng = self._engine_with_history(history)
    r = eng.violation_trend(
        "J1", "P1", TimeWindow(T0, T0 + 300), bucket_seconds=300
    )
    self.assertEqual(r.total_conflicts, 1)
    self.assertEqual(r.total_unknown, 1)

  def test_junction_summary(self):
    history = [
        _s(PhaseState.RED, T0, phase="P1"),
        _s(PhaseState.GREEN, T0 + 20, phase="P1"),
        _s(PhaseState.RED, T0 + 50, phase="P1"),
        _s(PhaseState.GREEN, T0, phase="P2"),
        _s(PhaseState.RED, T0 + 30, phase="P2"),
    ]
    eng = self._engine_with_history(history)
    summary = eng.junction_summary("J1", TimeWindow(T0, T0 + 50))
    self.assertEqual(summary.junction_id, "J1")
    self.assertEqual(len(summary.phase_durations), 2)
    self.assertEqual(len(summary.occupancy_correlations), 2)
    phases = {pd.phase_id for pd in summary.phase_durations}
    self.assertEqual(phases, {"P1", "P2"})

  def test_compare_windows(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 10),
        _s(PhaseState.RED, T0 + 30),
        _s(PhaseState.GREEN, T0 + 60),
        _s(PhaseState.RED, T0 + 90),
    ]
    eng = self._engine_with_history(history)
    window_a = TimeWindow(T0, T0 + 30)
    window_b = TimeWindow(T0 + 30, T0 + 90)
    cmp = eng.compare_windows("J1", "P1", window_a, window_b)
    self.assertIsInstance(cmp, PhaseAnalyticsComparison)
    a, b = cmp.phase_durations
    self.assertEqual(a.window, window_a)
    self.assertEqual(b.window, window_b)

  def test_empty_junction(self):
    store = SignalStore(history_size=200)
    eng = AnalyticsEngine(store)
    summary = eng.junction_summary("MISSING", TimeWindow(T0, T0 + 60))
    self.assertEqual(summary.phase_durations, [])
    self.assertTrue(summary.partial_data)

  def test_mixed_source_types(self):
    """Analytics should merge history across source types."""
    history = [
        _s(PhaseState.RED, T0, source=SourceType.CONTROLLER),
        _s(PhaseState.GREEN, T0 + 10, source=SourceType.VISION),
        _s(PhaseState.RED, T0 + 30, source=SourceType.CONTROLLER),
    ]
    eng = self._engine_with_history(history)
    r = eng.phase_durations("J1", "P1", TimeWindow(T0, T0 + 30))
    # Should see transitions from combined timeline.
    self.assertGreater(r.total_observations, 1)
    self.assertIn("controller", r.source_types_seen)
    self.assertIn("vision", r.source_types_seen)
    self.assertTrue(
        any("mixes raw observations" in item for item in r.assumptions)
    )

  def test_source_filter_prevents_cross_source_merge(self):
    history = [
        _s(PhaseState.RED, T0, source=SourceType.CONTROLLER),
        _s(
            PhaseState.GREEN,
            T0 + 10,
            source=SourceType.VISION,
            controller_id="CAM-1",
        ),
        _s(PhaseState.RED, T0 + 30, source=SourceType.CONTROLLER),
    ]
    eng = self._engine_with_history(history)
    r = eng.phase_durations(
        "J1",
        "P1",
        TimeWindow(T0, T0 + 40),
        source_types=frozenset({SourceType.CONTROLLER}),
    )
    self.assertEqual(r.total_observations, 2)
    self.assertEqual(r.source_types_seen, ("controller",))

  def test_camera_summary_filters_to_selected_camera(self):
    history = [
        _s(
            PhaseState.RED,
            T0,
            source=SourceType.VISION,
            controller_id="CAM-1",
            camera_id="CAM-1",
        ),
        _s(
            PhaseState.GREEN,
            T0 + 20,
            source=SourceType.VISION,
            controller_id="CAM-1",
            camera_id="CAM-1",
        ),
        _s(
            PhaseState.RED,
            T0 + 40,
            source=SourceType.VISION,
            controller_id="CAM-1",
            camera_id="CAM-1",
        ),
        _s(
            PhaseState.RED,
            T0,
            source=SourceType.VISION,
            phase="P2",
            controller_id="CAM-2",
            camera_id="CAM-2",
        ),
        _s(
            PhaseState.GREEN,
            T0 + 15,
            source=SourceType.VISION,
            phase="P2",
            controller_id="CAM-2",
            camera_id="CAM-2",
        ),
    ]
    eng = self._engine_with_history(history)
    summary = eng.camera_summary("J1", "CAM-1", TimeWindow(T0, T0 + 50))
    self.assertEqual(summary.camera_id, "CAM-1")
    self.assertEqual(len(summary.phase_durations), 1)
    self.assertEqual(summary.phase_durations[0].phase_id, "P1")
    self.assertEqual(summary.phase_durations[0].camera_id, "CAM-1")

  def test_phase_durations_camera_view_uses_vision_only(self):
    history = [
        _s(PhaseState.RED, T0, source=SourceType.CONTROLLER),
        _s(
            PhaseState.RED,
            T0,
            source=SourceType.VISION,
            controller_id="CAM-1",
            camera_id="CAM-1",
        ),
        _s(
            PhaseState.GREEN,
            T0 + 10,
            source=SourceType.VISION,
            controller_id="CAM-1",
            camera_id="CAM-1",
        ),
        _s(
            PhaseState.RED,
            T0 + 30,
            source=SourceType.VISION,
            controller_id="CAM-1",
            camera_id="CAM-1",
        ),
    ]
    eng = self._engine_with_history(history)
    r = eng.phase_durations(
        "J1",
        "P1",
        TimeWindow(T0, T0 + 31),
        camera_id="CAM-1",
    )
    self.assertEqual(r.camera_id, "CAM-1")
    self.assertEqual(r.source_types_seen, ("vision",))
    self.assertEqual(r.total_observations, 3)
    self.assertTrue(
        any(
            "Camera view uses vision observations only" in item
            for item in r.assumptions
        )
    )

  def test_time_window_filtering(self):
    """Data outside the window should be excluded."""
    history = [
        _s(PhaseState.RED, T0),  # inside
        _s(PhaseState.GREEN, T0 + 10),  # inside
        _s(PhaseState.RED, T0 + 100),  # outside
    ]
    eng = self._engine_with_history(history)
    r = eng.phase_durations("J1", "P1", TimeWindow(T0, T0 + 50))
    self.assertEqual(r.total_observations, 2)


if __name__ == "__main__":
  unittest.main()
