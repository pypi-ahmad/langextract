"""Tests for trafficmind.arbitration."""

import unittest

from trafficmind.arbitration import Arbitrator
from trafficmind.models import ArbitrationMode
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.store import SignalStore

NOW = 1000.0


def _ctrl(state=PhaseState.RED, ts=NOW, confidence=1.0):
  return SignalState(
      junction_id="J1",
      controller_id="C1",
      phase_id="P1",
      state=state,
      timestamp=ts,
      source_type=SourceType.CONTROLLER,
      confidence=confidence,
  )


def _vision(state=PhaseState.GREEN, ts=NOW, confidence=0.85):
  return SignalState(
      junction_id="J1",
      controller_id="CAM1",
      phase_id="P1",
      state=state,
      timestamp=ts,
      source_type=SourceType.VISION,
      confidence=confidence,
  )


class TestArbitratorControllerOnly(unittest.TestCase):

  def test_uses_controller_state(self):
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.GREEN))
    arb = Arbitrator(store, mode=ArbitrationMode.CONTROLLER_ONLY)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.GREEN)
    self.assertFalse(report.conflict)

  def test_missing_controller_returns_unknown(self):
    store = SignalStore()
    arb = Arbitrator(store, mode=ArbitrationMode.CONTROLLER_ONLY)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.UNKNOWN)
    self.assertTrue(report.stale)

  def test_stale_controller_returns_unknown(self):
    store = SignalStore(stale_after_seconds=10)
    store.update(_ctrl(ts=NOW - 20))
    arb = Arbitrator(store, mode=ArbitrationMode.CONTROLLER_ONLY)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.UNKNOWN)
    self.assertTrue(report.stale)


class TestArbitratorVisionOnly(unittest.TestCase):

  def test_uses_vision_state(self):
    store = SignalStore(stale_after_seconds=30)
    store.update(_vision(PhaseState.RED))
    arb = Arbitrator(store, mode=ArbitrationMode.VISION_ONLY)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)

  def test_missing_vision_returns_unknown(self):
    store = SignalStore()
    arb = Arbitrator(store, mode=ArbitrationMode.VISION_ONLY)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.UNKNOWN)


class TestArbitratorHybrid(unittest.TestCase):

  def test_agreement_returns_shared_state(self):
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.RED))
    store.update(_vision(PhaseState.RED))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)
    self.assertFalse(report.conflict)
    self.assertIn("agree", report.reason.lower())

  def test_conflict_prefers_restrictive(self):
    """When controller says RED and vision says GREEN, choose RED."""
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.RED))
    store.update(_vision(PhaseState.GREEN))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)
    self.assertTrue(report.conflict)

  def test_conflict_vision_red_chosen(self):
    """Vision says RED, controller says GREEN → RED wins."""
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.GREEN))
    store.update(_vision(PhaseState.RED))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)
    self.assertTrue(report.conflict)

  def test_stale_controller_fresh_vision_uses_vision(self):
    store = SignalStore(stale_after_seconds=10)
    store.update(_ctrl(PhaseState.GREEN, ts=NOW - 20))
    store.update(_vision(PhaseState.RED, ts=NOW))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)
    self.assertTrue(report.stale)

  def test_both_missing_returns_unknown(self):
    store = SignalStore()
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.UNKNOWN)
    self.assertTrue(report.stale)

  def test_only_controller_degrades(self):
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.GREEN))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.GREEN)
    self.assertIn("only controller", report.reason.lower())

  def test_only_vision_degrades(self):
    store = SignalStore(stale_after_seconds=30)
    store.update(_vision(PhaseState.AMBER))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.AMBER)
    self.assertIn("only vision", report.reason.lower())

  def test_both_non_restrictive_conflict_uses_higher_confidence(self):
    """Both GREEN vs AMBER (non-restrictive) → higher confidence wins."""
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.GREEN, confidence=0.7))
    store.update(_vision(PhaseState.AMBER, confidence=0.9))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    # Vision has higher confidence.
    self.assertEqual(report.resolved_state, PhaseState.AMBER)
    self.assertTrue(report.conflict)

  def test_mode_override_per_call(self):
    store = SignalStore(stale_after_seconds=30)
    store.update(_ctrl(PhaseState.RED))
    store.update(_vision(PhaseState.GREEN))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", mode=ArbitrationMode.VISION_ONLY, now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.GREEN)

  def test_report_preserves_provenance(self):
    store = SignalStore(stale_after_seconds=30)
    ctrl = _ctrl(PhaseState.RED)
    vis = _vision(PhaseState.RED)
    store.update(ctrl)
    store.update(vis)
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.controller_state, ctrl)
    self.assertEqual(report.vision_state, vis)

  def test_both_stale_conflict_returns_unknown(self):
    """Both sources stale + disagreement → UNKNOWN (don't guess)."""
    store = SignalStore(stale_after_seconds=10)
    store.update(_ctrl(PhaseState.GREEN, ts=NOW - 20))
    store.update(_vision(PhaseState.RED, ts=NOW - 15))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.UNKNOWN)
    self.assertTrue(report.conflict)
    self.assertTrue(report.stale)

  def test_agreement_stale_still_uses_agreed_state(self):
    """Both agree on RED but one is stale → use RED (corroborated)."""
    store = SignalStore(stale_after_seconds=10)
    store.update(_ctrl(PhaseState.RED, ts=NOW - 20))
    store.update(_vision(PhaseState.RED, ts=NOW))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)
    self.assertFalse(report.conflict)
    self.assertTrue(report.stale)
    self.assertIn("agree", report.reason.lower())
    self.assertIn("stale", report.reason.lower())

  def test_both_stale_agreement_uses_agreed_state(self):
    """Both stale but agree → trust the agreed state."""
    store = SignalStore(stale_after_seconds=10)
    store.update(_ctrl(PhaseState.RED, ts=NOW - 20))
    store.update(_vision(PhaseState.RED, ts=NOW - 15))
    arb = Arbitrator(store, mode=ArbitrationMode.HYBRID)
    report = arb.resolve("J1", "P1", now=NOW)
    self.assertEqual(report.resolved_state, PhaseState.RED)
    self.assertFalse(report.conflict)
    self.assertTrue(report.stale)


if __name__ == "__main__":
  unittest.main()
