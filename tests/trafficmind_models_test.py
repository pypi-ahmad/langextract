"""Tests for trafficmind.models."""

import time
import unittest

from trafficmind.models import ArbitrationMode
from trafficmind.models import PhaseState
from trafficmind.models import RESTRICTIVE_STATES
from trafficmind.models import SignalReport
from trafficmind.models import SignalState
from trafficmind.models import SourceType


class TestPhaseState(unittest.TestCase):

  def test_all_values_are_strings(self):
    for ps in PhaseState:
      self.assertIsInstance(ps.value, str)

  def test_restrictive_states_subset(self):
    self.assertTrue(RESTRICTIVE_STATES.issubset(set(PhaseState)))
    self.assertIn(PhaseState.RED, RESTRICTIVE_STATES)
    self.assertNotIn(PhaseState.GREEN, RESTRICTIVE_STATES)


class TestSignalState(unittest.TestCase):

  def _make(self, **overrides):
    defaults = dict(
        junction_id="J1",
        controller_id="C1",
        phase_id="P1",
        state=PhaseState.RED,
        timestamp=1000.0,
        source_type=SourceType.CONTROLLER,
        confidence=1.0,
    )
    defaults.update(overrides)
    return SignalState(**defaults)

  def test_basic_creation(self):
    s = self._make()
    self.assertEqual(s.junction_id, "J1")
    self.assertEqual(s.state, PhaseState.RED)
    self.assertEqual(s.confidence, 1.0)

  def test_is_restrictive(self):
    self.assertTrue(self._make(state=PhaseState.RED).is_restrictive)
    self.assertTrue(self._make(state=PhaseState.FLASHING_RED).is_restrictive)
    self.assertFalse(self._make(state=PhaseState.GREEN).is_restrictive)

  def test_age(self):
    s = self._make(timestamp=100.0)
    self.assertAlmostEqual(s.age(now=130.0), 30.0)

  def test_age_with_zero_now(self):
    """now=0.0 must not fall through to time.time()."""
    s = self._make(timestamp=10.0)
    self.assertAlmostEqual(s.age(now=0.0), -10.0)

  def test_confidence_validation(self):
    with self.assertRaises(ValueError):
      self._make(confidence=1.5)
    with self.assertRaises(ValueError):
      self._make(confidence=-0.1)

  def test_metadata_default_empty(self):
    s = self._make()
    self.assertEqual(s.metadata, {})

  def test_camera_id_none_for_non_vision(self):
    s = self._make(source_type=SourceType.CONTROLLER)
    self.assertIsNone(s.camera_id)

  def test_camera_id_from_metadata_for_vision(self):
    s = self._make(
        source_type=SourceType.VISION,
        controller_id="legacy-camera",
        metadata={"camera_id": "CAM-7"},
    )
    self.assertEqual(s.camera_id, "CAM-7")

  def test_camera_id_falls_back_to_controller_for_vision(self):
    s = self._make(
        source_type=SourceType.VISION,
        controller_id="CAM-legacy",
    )
    self.assertEqual(s.camera_id, "CAM-legacy")

  def test_frozen(self):
    s = self._make()
    with self.assertRaises(AttributeError):
      s.junction_id = "X"  # type: ignore[misc]


class TestSignalReport(unittest.TestCase):

  def test_defaults(self):
    r = SignalReport(
        junction_id="J1",
        phase_id="P1",
        resolved_state=PhaseState.GREEN,
        mode=ArbitrationMode.HYBRID,
    )
    self.assertFalse(r.conflict)
    self.assertFalse(r.stale)
    self.assertIsNone(r.controller_state)
    self.assertIsNone(r.vision_state)


if __name__ == "__main__":
  unittest.main()
