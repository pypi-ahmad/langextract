"""Tests for trafficmind.store."""

import time
import unittest

from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.store import SignalStore


def _state(
    source_type=SourceType.CONTROLLER,
    state=PhaseState.RED,
    ts=1000.0,
    junction="J1",
    phase="P1",
    confidence=1.0,
    controller_id="C1",
    metadata=None,
):
  return SignalState(
      junction_id=junction,
      controller_id=controller_id,
      phase_id=phase,
      state=state,
      timestamp=ts,
      source_type=source_type,
      confidence=confidence,
      metadata={} if metadata is None else metadata,
  )


class TestSignalStore(unittest.TestCase):

  def test_update_and_get_latest(self):
    store = SignalStore()
    s = _state()
    store.update(s)
    got = store.get_latest("J1", "P1", SourceType.CONTROLLER)
    self.assertEqual(got, s)

  def test_get_controller_state(self):
    store = SignalStore()
    store.update(_state(source_type=SourceType.CONTROLLER, ts=100))
    store.update(_state(source_type=SourceType.POLLING, ts=200))
    # Polling is newer → chosen.
    ctrl = store.get_controller_state("J1", "P1")
    self.assertIsNotNone(ctrl)
    self.assertEqual(ctrl.source_type, SourceType.POLLING)

  def test_get_vision_state(self):
    store = SignalStore()
    store.update(_state(source_type=SourceType.VISION, state=PhaseState.GREEN))
    vis = store.get_vision_state("J1", "P1")
    self.assertIsNotNone(vis)
    self.assertEqual(vis.state, PhaseState.GREEN)

  def test_vision_none_when_only_controller(self):
    store = SignalStore()
    store.update(_state(source_type=SourceType.CONTROLLER))
    self.assertIsNone(store.get_vision_state("J1", "P1"))

  def test_is_stale(self):
    store = SignalStore(stale_after_seconds=10)
    s = _state(ts=100.0)
    self.assertTrue(store.is_stale(s, now=200.0))
    self.assertFalse(store.is_stale(s, now=105.0))

  def test_history(self):
    store = SignalStore()
    store.update(_state(ts=1))
    store.update(_state(ts=2))
    store.update(_state(ts=3))
    hist = store.get_history("J1", "P1", SourceType.CONTROLLER)
    self.assertEqual(len(hist), 3)

  def test_all_junctions(self):
    store = SignalStore()
    store.update(_state(junction="A"))
    store.update(_state(junction="B"))
    self.assertEqual(store.all_junctions(), {"A", "B"})

  def test_clear(self):
    store = SignalStore()
    store.update(_state())
    store.clear()
    self.assertIsNone(store.get_latest("J1", "P1", SourceType.CONTROLLER))

  def test_update_batch(self):
    store = SignalStore()
    store.update_batch([_state(ts=1), _state(ts=2, phase="P2")])
    self.assertIsNotNone(store.get_latest("J1", "P1", SourceType.CONTROLLER))
    self.assertIsNotNone(store.get_latest("J1", "P2", SourceType.CONTROLLER))

  def test_out_of_order_keeps_newest(self):
    """Late-arriving older observation must not overwrite a newer one."""
    store = SignalStore()
    store.update(_state(ts=200, state=PhaseState.GREEN))
    store.update(_state(ts=100, state=PhaseState.RED))  # older
    latest = store.get_latest("J1", "P1", SourceType.CONTROLLER)
    self.assertEqual(latest.timestamp, 200)
    self.assertEqual(latest.state, PhaseState.GREEN)
    # History should still record both.
    hist = store.get_history("J1", "P1", SourceType.CONTROLLER)
    self.assertEqual(len(hist), 2)

  def test_phase_ids_for_junction(self):
    store = SignalStore()
    store.update(_state(junction="J1", phase="P1"))
    store.update(_state(junction="J1", phase="P2"))
    store.update(_state(junction="J2", phase="P3"))
    self.assertEqual(store.phase_ids_for_junction("J1"), {"P1", "P2"})
    self.assertEqual(store.phase_ids_for_junction("J2"), {"P3"})
    self.assertEqual(store.phase_ids_for_junction("J99"), set())

  def test_latest_observation_count(self):
    store = SignalStore()
    store.update(_state(source_type=SourceType.CONTROLLER, ts=10))
    store.update(_state(source_type=SourceType.VISION, ts=11))
    self.assertEqual(store.latest_observation_count, 2)

  def test_get_history_window_is_end_exclusive(self):
    store = SignalStore()
    store.update(_state(ts=10))
    store.update(_state(ts=20))
    window = store.get_history_window(
        "J1",
        "P1",
        SourceType.CONTROLLER,
        start=10,
        end=20,
    )
    self.assertEqual([state.timestamp for state in window], [10])

  def test_all_history_window_can_filter_source_types(self):
    store = SignalStore()
    store.update(_state(ts=10, source_type=SourceType.CONTROLLER))
    store.update(
        _state(ts=11, source_type=SourceType.VISION, controller_id="CAM-1")
    )
    history = store.all_history_window(
        "J1",
        "P1",
        source_types=frozenset({SourceType.VISION}),
    )
    self.assertEqual(len(history), 1)
    self.assertEqual(history[0].source_type, SourceType.VISION)

  def test_camera_ids_and_phase_ids_for_camera(self):
    store = SignalStore()
    store.update(
        _state(
            ts=10,
            source_type=SourceType.VISION,
            phase="P1",
            controller_id="CAM-1",
        )
    )
    store.update(
        _state(
            ts=20,
            source_type=SourceType.VISION,
            phase="P2",
            controller_id="legacy-feed",
            metadata={"camera_id": "CAM-2"},
        )
    )
    self.assertEqual(store.camera_ids_for_junction("J1"), {"CAM-1", "CAM-2"})
    self.assertEqual(store.phase_ids_for_camera("J1", "CAM-1"), {"P1"})
    self.assertEqual(store.phase_ids_for_camera("J1", "CAM-2"), {"P2"})


if __name__ == "__main__":
  unittest.main()
