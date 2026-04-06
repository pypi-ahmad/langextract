"""Tests for trafficmind.analytics.occupancy."""

import unittest

from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.occupancy import compute_occupancy_correlation
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType

T0 = 1000.0


def _s(state, ts):
  return SignalState(
      junction_id="J1",
      controller_id="C1",
      phase_id="P1",
      state=state,
      timestamp=ts,
      source_type=SourceType.CONTROLLER,
  )


class TestOccupancyCorrelation(unittest.TestCase):

  def test_empty_history(self):
    r = compute_occupancy_correlation([], TimeWindow(T0, T0 + 60))
    self.assertTrue(r.partial_data)
    self.assertEqual(r.estimated_queue_events, 0)

  def test_all_red(self):
    history = [_s(PhaseState.RED, T0), _s(PhaseState.RED, T0 + 60)]
    r = compute_occupancy_correlation(history, TimeWindow(T0, T0 + 60))
    self.assertAlmostEqual(r.red_total_seconds, 60.0)
    self.assertAlmostEqual(r.green_total_seconds, 0.0)
    self.assertAlmostEqual(r.red_fraction, 1.0)
    self.assertEqual(r.estimated_queue_events, 0)

  def test_all_green(self):
    history = [_s(PhaseState.GREEN, T0), _s(PhaseState.GREEN, T0 + 60)]
    r = compute_occupancy_correlation(history, TimeWindow(T0, T0 + 60))
    self.assertAlmostEqual(r.green_total_seconds, 60.0)
    self.assertAlmostEqual(r.green_fraction, 1.0)

  def test_red_green_transition_queue_event(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 20),
        _s(PhaseState.RED, T0 + 50),
    ]
    r = compute_occupancy_correlation(history, TimeWindow(T0, T0 + 50))
    self.assertEqual(r.estimated_queue_events, 1)
    self.assertAlmostEqual(r.mean_red_before_green, 20.0)
    self.assertAlmostEqual(r.red_total_seconds, 20.0)
    self.assertAlmostEqual(r.green_total_seconds, 30.0)

  def test_multiple_queue_events(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 10),
        _s(PhaseState.RED, T0 + 30),
        _s(PhaseState.GREEN, T0 + 50),
        _s(PhaseState.RED, T0 + 60),
    ]
    r = compute_occupancy_correlation(history, TimeWindow(T0, T0 + 60))
    self.assertEqual(r.estimated_queue_events, 2)

  def test_red_green_fraction_uses_total_observed_time(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.AMBER, T0 + 10),
        _s(PhaseState.GREEN, T0 + 20),
        _s(PhaseState.RED, T0 + 40),
    ]
    r = compute_occupancy_correlation(history, TimeWindow(T0, T0 + 40))
    # Covered time = 40s total; amber occupies 10s and should stay in
    # the denominator rather than being discarded.
    self.assertAlmostEqual(r.red_total_seconds, 10.0)
    self.assertAlmostEqual(r.green_total_seconds, 20.0)
    self.assertAlmostEqual(r.red_fraction, 0.25)
    self.assertAlmostEqual(r.green_fraction, 0.5)

  def test_junction_phase_propagated(self):
    r = compute_occupancy_correlation(
        [_s(PhaseState.RED, T0), _s(PhaseState.RED, T0 + 10)],
        TimeWindow(T0, T0 + 10),
    )
    self.assertEqual(r.junction_id, "J1")
    self.assertEqual(r.phase_id, "P1")


if __name__ == "__main__":
  unittest.main()
