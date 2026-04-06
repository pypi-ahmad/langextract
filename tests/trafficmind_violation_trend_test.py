"""Tests for trafficmind.analytics.violation_trend."""

import unittest

from trafficmind.analytics.models import TimeWindow
from trafficmind.analytics.violation_trend import compute_violation_trend
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType

T0 = 1000.0


def _s(state, ts, *, conflict=False, stale=False):
  meta = {}
  if conflict:
    meta["conflict"] = True
  if stale:
    meta["stale"] = True
  return SignalState(
      junction_id="J1",
      controller_id="C1",
      phase_id="P1",
      state=state,
      timestamp=ts,
      source_type=SourceType.CONTROLLER,
      metadata=meta,
  )


class TestViolationTrend(unittest.TestCase):

  def test_empty_history(self):
    r = compute_violation_trend([], TimeWindow(T0, T0 + 300))
    self.assertTrue(r.partial_data)
    self.assertEqual(r.buckets, [])

  def test_no_anomalies(self):
    history = [
        _s(PhaseState.RED, T0),
        _s(PhaseState.GREEN, T0 + 30),
    ]
    r = compute_violation_trend(
        history, TimeWindow(T0, T0 + 300), bucket_seconds=300
    )
    self.assertEqual(len(r.buckets), 1)
    self.assertEqual(r.total_conflicts, 0)
    self.assertEqual(r.total_stale, 0)
    self.assertEqual(r.total_unknown, 0)
    self.assertEqual(r.buckets[0].total_observations, 2)

  def test_conflict_counted(self):
    history = [
        _s(PhaseState.RED, T0, conflict=True),
        _s(PhaseState.GREEN, T0 + 10),
    ]
    r = compute_violation_trend(
        history, TimeWindow(T0, T0 + 300), bucket_seconds=300
    )
    self.assertEqual(r.total_conflicts, 1)
    self.assertEqual(r.buckets[0].conflict_count, 1)

  def test_stale_counted(self):
    history = [
        _s(PhaseState.RED, T0, stale=True),
        _s(PhaseState.GREEN, T0 + 10, stale=True),
    ]
    r = compute_violation_trend(
        history, TimeWindow(T0, T0 + 300), bucket_seconds=300
    )
    self.assertEqual(r.total_stale, 2)

  def test_unknown_counted(self):
    history = [
        _s(PhaseState.UNKNOWN, T0),
        _s(PhaseState.UNKNOWN, T0 + 30),
        _s(PhaseState.RED, T0 + 60),
    ]
    r = compute_violation_trend(
        history, TimeWindow(T0, T0 + 300), bucket_seconds=300
    )
    self.assertEqual(r.total_unknown, 2)

  def test_multiple_buckets(self):
    history = [
        _s(PhaseState.RED, T0, conflict=True),  # bucket 0
        _s(PhaseState.GREEN, T0 + 100),  # bucket 0
        _s(PhaseState.UNKNOWN, T0 + 200, stale=True),  # bucket 0
        _s(PhaseState.RED, T0 + 350, conflict=True),  # bucket 1
    ]
    r = compute_violation_trend(
        history, TimeWindow(T0, T0 + 600), bucket_seconds=300
    )
    self.assertEqual(len(r.buckets), 2)
    self.assertEqual(r.buckets[0].conflict_count, 1)
    self.assertEqual(r.buckets[0].unknown_count, 1)
    self.assertEqual(r.buckets[0].stale_count, 1)
    self.assertEqual(r.buckets[0].total_observations, 3)
    self.assertEqual(r.buckets[1].conflict_count, 1)
    self.assertEqual(r.buckets[1].total_observations, 1)

  def test_bucket_boundaries(self):
    r = compute_violation_trend(
        [_s(PhaseState.RED, T0)],
        TimeWindow(T0, T0 + 600),
        bucket_seconds=300,
    )
    self.assertEqual(len(r.buckets), 2)
    self.assertAlmostEqual(r.buckets[0].bucket_start, T0)
    self.assertAlmostEqual(r.buckets[0].bucket_end, T0 + 300)
    self.assertAlmostEqual(r.buckets[1].bucket_start, T0 + 300)
    self.assertAlmostEqual(r.buckets[1].bucket_end, T0 + 600)


if __name__ == "__main__":
  unittest.main()
