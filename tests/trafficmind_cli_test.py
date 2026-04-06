"""Tests for TrafficMind CLI entry points."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _trafficmind_env(**overrides: str) -> dict[str, str]:
  env = os.environ.copy()
  env.pop("PYTHONPATH", None)
  for key in list(env):
    if key.startswith("TRAFFICMIND_"):
      del env[key]
  env.update(overrides)
  return env


class TestTrafficMindCli(unittest.TestCase):

  def test_module_entry_point_succeeds(self):
    result = subprocess.run(
        [sys.executable, "-m", "trafficmind.cli"],
        cwd=_REPO_ROOT,
        env=_trafficmind_env(TRAFFICMIND_PROFILE="dev"),
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0, msg=result.stderr)
    self.assertIn("Profile : dev", result.stdout)
    self.assertIn("All startup checks passed", result.stdout)

  def test_source_script_succeeds_without_pythonpath(self):
    result = subprocess.run(
        [sys.executable, "scripts/check_trafficmind.py"],
        cwd=_REPO_ROOT,
        env=_trafficmind_env(TRAFFICMIND_PROFILE="local"),
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0, msg=result.stderr)
    self.assertIn("Profile : local", result.stdout)
    self.assertIn("All startup checks passed", result.stdout)


if __name__ == "__main__":
  unittest.main()
