"""Tests for trafficmind.sources.*."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from trafficmind.exceptions import InvalidSignalDataError
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.sources.file_feed import FileFeedSource
from trafficmind.sources.simulator import SimulatorSource
from trafficmind.sources.webhook import WebhookReceiver


def _sample_record(**overrides):
  rec = {
      "junction_id": "J1",
      "controller_id": "C1",
      "phase_id": "P1",
      "state": "red",
      "timestamp": "1000.0",
  }
  rec.update(overrides)
  return rec


# ── File feed ─────────────────────────────────────────────────────────


class TestFileFeedJSON(unittest.TestCase):

  def test_reads_json(self):
    records = [_sample_record(), _sample_record(phase_id="P2", state="green")]
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as f:
      json.dump(records, f)
      path = f.name
    try:
      src = FileFeedSource(path)
      result = src.fetch()
      self.assertEqual(len(result), 2)
      self.assertEqual(result[0].state, PhaseState.RED)
      self.assertEqual(result[1].state, PhaseState.GREEN)
      self.assertEqual(result[0].source_type, SourceType.FILE_FEED)
    finally:
      os.unlink(path)

  def test_missing_file_returns_empty(self):
    src = FileFeedSource("/nonexistent/path.json")
    self.assertEqual(src.fetch(), [])

  def test_invalid_state_raises(self):
    records = [_sample_record(state="purple")]
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as f:
      json.dump(records, f)
      path = f.name
    try:
      src = FileFeedSource(path)
      with self.assertRaises(InvalidSignalDataError):
        src.fetch()
    finally:
      os.unlink(path)

  def test_missing_field_raises(self):
    records = [{"junction_id": "J1", "state": "red"}]  # incomplete
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as f:
      json.dump(records, f)
      path = f.name
    try:
      src = FileFeedSource(path)
      with self.assertRaises(InvalidSignalDataError):
        src.fetch()
    finally:
      os.unlink(path)


class TestFileFeedCSV(unittest.TestCase):

  def test_reads_csv(self):
    lines = [
        "junction_id,controller_id,phase_id,state,timestamp,confidence",
        "J1,C1,P1,red,1000.0,0.95",
        "J1,C1,P2,green,1001.0,1.0",
    ]
    with tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False, newline=""
    ) as f:
      f.write("\n".join(lines))
      path = f.name
    try:
      src = FileFeedSource(path)
      result = src.fetch()
      self.assertEqual(len(result), 2)
      self.assertAlmostEqual(result[0].confidence, 0.95)
    finally:
      os.unlink(path)


# ── Webhook ───────────────────────────────────────────────────────────


class TestWebhookReceiver(unittest.TestCase):

  def test_receive_and_fetch_drains(self):
    wh = WebhookReceiver()
    wh.receive(_sample_record())
    wh.receive(_sample_record(phase_id="P2", state="green"))
    self.assertEqual(wh.pending_count, 2)
    states = wh.fetch()
    self.assertEqual(len(states), 2)
    self.assertEqual(wh.pending_count, 0)
    self.assertEqual(wh.fetch(), [])

  def test_receive_batch(self):
    wh = WebhookReceiver()
    wh.receive_batch([_sample_record(), _sample_record(state="green")])
    self.assertEqual(wh.pending_count, 2)

  def test_invalid_record_raises(self):
    wh = WebhookReceiver()
    with self.assertRaises(InvalidSignalDataError):
      wh.receive({"junction_id": "J1"})


# ── Simulator ─────────────────────────────────────────────────────────


class TestSimulator(unittest.TestCase):

  def test_cycles_through_states(self):
    sim = SimulatorSource(cycle=[PhaseState.GREEN, PhaseState.RED])
    self.assertEqual(sim.fetch()[0].state, PhaseState.GREEN)
    self.assertEqual(sim.fetch()[0].state, PhaseState.RED)
    self.assertEqual(sim.fetch()[0].state, PhaseState.GREEN)

  def test_source_type_is_simulator(self):
    sim = SimulatorSource()
    state = sim.fetch()[0]
    self.assertEqual(state.source_type, SourceType.SIMULATOR)

  def test_source_name(self):
    sim = SimulatorSource(junction_id="X", controller_id="Y")
    self.assertIn("X", sim.source_name())


if __name__ == "__main__":
  unittest.main()
