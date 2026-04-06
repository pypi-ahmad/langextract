"""Tests for trafficmind.analytics.queue_discharge."""

import unittest

from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.queue_discharge import compute_queue_discharge
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


class TestQueueDischarge(unittest.TestCase):

  def test_empty_history(self):
    r = compute_queue_discharge([], TimeWindow(T0, T0 + 60))
    self.assertEqual(r.transition_count, 0)
    self.assertIsNone(r.mean_discharge)
    self.assertTrue(r.partial_data)

  def test_no_transitions(self):
    history = [_s(PhaseState.GREEN, T0), _s(PhaseState.GREEN, T0 + 60)]
    r = compute_queue_discharge(history, TimeWindow(T0, T0 + 60))
    self.assertEqual(r.transition_count, 0)

  def test_single_discharge(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 10),
        _s(PhaseState.AMBER, T0 + 35),
    ]
    r = compute_queue_discharge(history, TimeWindow(T0, T0 + 35))
    self.assertEqual(r.transition_count, 1)
    self.assertAlmostEqual(r.discharge_durations[0], 25.0)
    self.assertAlmostEqual(r.mean_discharge, 25.0)

  def test_multiple_discharges(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 15),  # discharge = 10
        _s(PhaseState.GREEN, T0 + 25),
        _s(PhaseState.AMBER, T0 + 45),  # discharge = 20
    ]
    r = compute_queue_discharge(history, TimeWindow(T0, T0 + 45))
    self.assertEqual(r.transition_count, 2)
    self.assertAlmostEqual(r.discharge_durations[0], 10.0)
    self.assertAlmostEqual(r.discharge_durations[1], 20.0)
    self.assertAlmostEqual(r.mean_discharge, 15.0)
    self.assertAlmostEqual(r.median_discharge, 15.0)
    self.assertAlmostEqual(r.max_discharge, 20.0)

  def test_red_amber_counts_as_restrictive(self):
    """RED_AMBER → GREEN should also trigger a discharge."""
    history = [
        _s(PhaseState.RED_AMBER, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.AMBER, T0 + 30),
    ]
    r = compute_queue_discharge(history, TimeWindow(T0, T0 + 30))
    self.assertEqual(r.transition_count, 1)
    self.assertAlmostEqual(r.discharge_durations[0], 25.0)


if __name__ == "__main__":
  unittest.main()
