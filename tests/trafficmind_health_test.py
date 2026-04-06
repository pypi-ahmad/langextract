"""Tests for TrafficMind startup checks and health probes."""

from __future__ import annotations

import time
import unittest

from trafficmind.config import Profile
from trafficmind.config import ServiceConfig
from trafficmind.health import health_snapshot
from trafficmind.health import HealthSnapshot
from trafficmind.health import run_startup_checks
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.service import SignalService
from trafficmind.store import SignalStore


class TestRunStartupChecks(unittest.TestCase):

  def test_passes_with_valid_config(self):
    cfg = ServiceConfig(
        profile=Profile.LOCAL,
        stale_after_seconds=60,
        history_size=100,
        log_level="DEBUG",
    )
    problems = run_startup_checks(cfg)
    self.assertEqual(problems, [])

  def test_unreachable_polling_url_reported(self):
    cfg = ServiceConfig(
        profile=Profile.LOCAL,
        stale_after_seconds=60,
        history_size=100,
        log_level="DEBUG",
        polling_url="http://127.0.0.1:19999/nonexistent",
        polling_timeout_seconds=1.0,
    )
    problems = run_startup_checks(cfg)
    self.assertTrue(len(problems) >= 1)
    self.assertIn("unreachable", problems[0].lower())

  def test_no_polling_url_skips_connectivity(self):
    cfg = ServiceConfig(
        profile=Profile.LOCAL,
        stale_after_seconds=60,
        history_size=100,
        log_level="DEBUG",
        polling_url=None,
    )
    problems = run_startup_checks(cfg)
    self.assertEqual(problems, [])


class TestHealthSnapshot(unittest.TestCase):

  def test_snapshot_without_service(self):
    cfg = ServiceConfig(
        profile=Profile.DEV,
        stale_after_seconds=30,
        history_size=50,
        log_level="INFO",
    )
    snap = health_snapshot(config=cfg, start_time=time.time() - 5)
    self.assertTrue(snap.ok)
    self.assertEqual(snap.profile, "dev")
    self.assertGreaterEqual(snap.uptime_seconds, 4.0)
    self.assertEqual(snap.store_junctions, 0)

  def test_snapshot_with_service(self):
    service = SignalService(
        store=SignalStore(stale_after_seconds=60),
    )
    cfg = ServiceConfig(
        profile=Profile.LOCAL,
        stale_after_seconds=60,
        history_size=100,
        log_level="DEBUG",
    )
    snap = health_snapshot(config=cfg, service=service, start_time=time.time())
    self.assertTrue(snap.ok)
    self.assertEqual(snap.source_count, 0)

  def test_snapshot_counts_store_observations(self):
    service = SignalService(
        store=SignalStore(stale_after_seconds=60),
    )
    service.ingest_states([
        SignalState(
            junction_id="J1",
            controller_id="CTRL-1",
            phase_id="P1",
            state=PhaseState.RED,
            timestamp=1000.0,
            source_type=SourceType.CONTROLLER,
        ),
        SignalState(
            junction_id="J1",
            controller_id="CAM-1",
            phase_id="P1",
            state=PhaseState.GREEN,
            timestamp=1001.0,
            source_type=SourceType.VISION,
        ),
    ])
    cfg = ServiceConfig(
        profile=Profile.DEV,
        stale_after_seconds=60,
        history_size=100,
        log_level="INFO",
    )

    snap = health_snapshot(config=cfg, service=service)

    self.assertEqual(snap.store_junctions, 1)
    self.assertEqual(snap.store_observations, 2)

  def test_as_dict_serialises(self):
    snap = HealthSnapshot(
        ok=True,
        profile="local",
        uptime_seconds=12.345,
        store_junctions=3,
    )
    d = snap.as_dict()
    self.assertTrue(d["ok"])
    self.assertEqual(d["profile"], "local")
    self.assertEqual(d["uptime_seconds"], 12.35)
    self.assertEqual(d["store_junctions"], 3)
    self.assertIsInstance(d["problems"], list)

  def test_snapshot_no_config_no_service(self):
    snap = health_snapshot()
    self.assertTrue(snap.ok)
    self.assertEqual(snap.profile, "unknown")


if __name__ == "__main__":
  unittest.main()
