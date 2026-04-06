"""Tests for trafficmind.analytics.oversaturation."""

import unittest

from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.oversaturation import compute_oversaturation
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


class TestOversaturation(unittest.TestCase):

  def test_empty_history(self):
    r = compute_oversaturation([], TimeWindow(T0, T0 + 60))
    self.assertEqual(r.cycle_count, 0)
    self.assertFalse(r.recurring)
    self.assertTrue(r.partial_data)

  def test_no_short_greens(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 30),  # green = 25s > 10s threshold
        _s(PhaseState.GREEN, T0 + 35),
        _s(PhaseState.RED, T0 + 60),  # green = 25s
    ]
    r = compute_oversaturation(
        history, TimeWindow(T0, T0 + 60), short_green_threshold=10.0
    )
    self.assertEqual(r.cycle_count, 2)
    self.assertEqual(r.short_green_count, 0)
    self.assertFalse(r.recurring)

  def test_all_short_greens(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 10),  # green = 5s < 10s
        _s(PhaseState.GREEN, T0 + 20),
        _s(PhaseState.RED, T0 + 25),  # green = 5s
    ]
    r = compute_oversaturation(
        history, TimeWindow(T0, T0 + 25), short_green_threshold=10.0
    )
    self.assertEqual(r.cycle_count, 2)
    self.assertEqual(r.short_green_count, 2)
    self.assertAlmostEqual(r.short_green_ratio, 1.0)
    self.assertTrue(r.recurring)

  def test_mixed_cycles(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 10),  # 5s — short
        _s(PhaseState.GREEN, T0 + 15),
        _s(PhaseState.RED, T0 + 30),  # 15s — not short
    ]
    r = compute_oversaturation(
        history, TimeWindow(T0, T0 + 30), short_green_threshold=10.0
    )
    self.assertEqual(r.cycle_count, 2)
    self.assertEqual(r.short_green_count, 1)
    self.assertAlmostEqual(r.short_green_ratio, 0.5)
    self.assertTrue(r.recurring)  # 50% threshold

  def test_below_recurring_threshold(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 8),  # 3s — short
        _s(PhaseState.GREEN, T0 + 15),
        _s(PhaseState.RED, T0 + 30),  # 15s — not short
        _s(PhaseState.GREEN, T0 + 35),
        _s(PhaseState.RED, T0 + 55),  # 20s — not short
    ]
    r = compute_oversaturation(
        history, TimeWindow(T0, T0 + 55), short_green_threshold=10.0
    )
    self.assertEqual(r.cycle_count, 3)
    self.assertEqual(r.short_green_count, 1)
    self.assertFalse(r.recurring)  # 1/3 < 0.5

  def test_mean_green_duration(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 15),  # 10s
        _s(PhaseState.GREEN, T0 + 20),
        _s(PhaseState.RED, T0 + 40),  # 20s
    ]
    r = compute_oversaturation(history, TimeWindow(T0, T0 + 40))
    self.assertAlmostEqual(r.mean_green_duration, 15.0)

  def test_custom_threshold(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 5),
        _s(PhaseState.RED, T0 + 20),  # 15s
    ]
    # With threshold=20, 15s is short.
    r1 = compute_oversaturation(
        history, TimeWindow(T0, T0 + 20), short_green_threshold=20.0
    )
    self.assertEqual(r1.short_green_count, 1)
    # With threshold=10, 15s is not short.
    r2 = compute_oversaturation(
        history, TimeWindow(T0, T0 + 20), short_green_threshold=10.0
    )
    self.assertEqual(r2.short_green_count, 0)


if __name__ == "__main__":
  unittest.main()
