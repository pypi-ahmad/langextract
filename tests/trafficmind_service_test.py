"""Tests for trafficmind.service (SignalService facade)."""

import unittest

from trafficmind.models import ArbitrationMode
from trafficmind.models import PhaseState
from trafficmind.models import SourceType
from trafficmind.service import SignalService
from trafficmind.sources.simulator import SimulatorSource


class TestSignalService(unittest.TestCase):

  def test_register_and_ingest(self):
    svc = SignalService(mode=ArbitrationMode.CONTROLLER_ONLY)
    sim = SimulatorSource(cycle=[PhaseState.RED])
    svc.register_source(sim)
    count = svc.ingest()
    self.assertEqual(count, 1)

  def test_resolve_after_ingest(self):
    svc = SignalService(
        mode=ArbitrationMode.CONTROLLER_ONLY,
        stale_after_seconds=60,
    )
    sim = SimulatorSource(
        junction_id="J1",
        controller_id="C1",
        phase_id="P1",
        cycle=[PhaseState.GREEN],
    )
    svc.register_source(sim)
    svc.ingest()
    report = svc.resolve("J1", "P1")
    self.assertEqual(report.resolved_state, PhaseState.GREEN)

  def test_resolve_all(self):
    svc = SignalService(
        mode=ArbitrationMode.CONTROLLER_ONLY,
        stale_after_seconds=60,
    )
    s1 = SimulatorSource(junction_id="A", phase_id="P1", cycle=[PhaseState.RED])
    s2 = SimulatorSource(
        junction_id="B", phase_id="P1", cycle=[PhaseState.GREEN]
    )
    svc.register_source(s1)
    svc.register_source(s2)
    svc.ingest()
    reports = svc.resolve_all()
    self.assertEqual(len(reports), 2)
    junctions = {r.junction_id for r in reports}
    self.assertEqual(junctions, {"A", "B"})

  def test_sources_property(self):
    svc = SignalService()
    self.assertEqual(len(svc.sources), 0)
    svc.register_source(SimulatorSource())
    self.assertEqual(len(svc.sources), 1)

  def test_ingest_handles_source_error(self):
    """A failing source should not crash the ingest loop."""

    class FailSource:

      def source_name(self):
        return "fail"

      def fetch(self):
        raise RuntimeError("boom")

    svc = SignalService()
    svc.register_source(FailSource())
    svc.register_source(SimulatorSource(cycle=[PhaseState.RED]))
    count = svc.ingest()
    # Only the simulator contributed.
    self.assertEqual(count, 1)


if __name__ == "__main__":
  unittest.main()
