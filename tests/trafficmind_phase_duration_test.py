"""Tests for trafficmind.analytics.phase_duration."""

import unittest

from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.phase_duration import compute_phase_durations
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType

T0 = 1000.0


def _s(state, ts, phase="P1"):
  return SignalState(
      junction_id="J1",
      controller_id="C1",
      phase_id=phase,
      state=state,
      timestamp=ts,
      source_type=SourceType.CONTROLLER,
  )


class TestPhaseDurations(unittest.TestCase):

  def test_empty_history(self):
    r = compute_phase_durations([], TimeWindow(T0, T0 + 60))
    self.assertEqual(r.total_observations, 0)
    self.assertTrue(r.partial_data)
    self.assertEqual(r.coverage_ratio, 0.0)

  def test_single_observation(self):
    r = compute_phase_durations(
        [_s(PhaseState.RED, T0)], TimeWindow(T0, T0 + 60)
    )
    self.assertEqual(r.total_observations, 1)
    self.assertEqual(r.state_counts.get("red", 0), 1)
    # No pairs → no duration accumulated.
    self.assertEqual(r.state_durations.get("red", 0.0), 0.0)

  def test_two_states(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 30),
        _s(PhaseState.RED, T0 + 60),
    ]
    r = compute_phase_durations(history, TimeWindow(T0, T0 + 60))
    self.assertAlmostEqual(r.state_durations["red"], 30.0)
    self.assertAlmostEqual(r.state_durations["green"], 30.0)
    self.assertEqual(r.state_counts["red"], 2)  # R at T0 and T0+60
    self.assertEqual(r.state_counts["green"], 1)

  def test_mean_durations(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 10),
        _s(PhaseState.RED, T0 + 20),
        _s(PhaseState.GREEN, T0 + 40),
    ]
    r = compute_phase_durations(history, TimeWindow(T0, T0 + 40))
    # Red occupies T0→T0+10 (10s) and T0+20→T0+40 (20s) = 30s.
    self.assertAlmostEqual(r.state_durations["red"], 30.0)
    # Red count = 2 (initial + second), so mean = 30/2 = 15.
    self.assertAlmostEqual(r.mean_durations["red"], 15.0)

  def test_coverage_ratio(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 30),
    ]
    r = compute_phase_durations(history, TimeWindow(T0, T0 + 60))
    # 30s covered out of 60s window.
    self.assertAlmostEqual(r.coverage_ratio, 0.5)
    self.assertTrue(r.partial_data)

  def test_full_coverage(self):
    history = [
        _s(PhaseState.GREEN, T0),
        _s(PhaseState.RED, T0 + 60),
    ]
    r = compute_phase_durations(history, TimeWindow(T0, T0 + 60))
    self.assertAlmostEqual(r.coverage_ratio, 1.0)
    self.assertFalse(r.partial_data)

  def test_junction_and_phase_propagated(self):
    r = compute_phase_durations(
        [_s(PhaseState.RED, T0), _s(PhaseState.RED, T0 + 10)],
        TimeWindow(T0, T0 + 10),
    )
    self.assertEqual(r.junction_id, "J1")
    self.assertEqual(r.phase_id, "P1")


if __name__ == "__main__":
  unittest.main()
