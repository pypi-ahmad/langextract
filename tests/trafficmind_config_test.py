"""Tests for TrafficMind environment config validation."""

from __future__ import annotations

import os
import unittest

from trafficmind.config import active_profile
from trafficmind.config import from_env
from trafficmind.config import Profile
from trafficmind.config import ServiceConfig


class TestActiveProfile(unittest.TestCase):

  def test_default_is_local(self):
    env = os.environ.copy()
    env.pop("TRAFFICMIND_PROFILE", None)
    with _patch_env(env):
      self.assertEqual(active_profile(), Profile.LOCAL)

  def test_explicit_profile_parsed(self):
    for name, expected in [
        ("local", Profile.LOCAL),
        ("dev", Profile.DEV),
        ("staging", Profile.STAGING),
        ("prod", Profile.PROD),
        ("PROD", Profile.PROD),
        ("  Dev  ", Profile.DEV),
    ]:
      with _patch_env({"TRAFFICMIND_PROFILE": name}):
        self.assertEqual(active_profile(), expected, msg=name)

  def test_unknown_profile_raises(self):
    with _patch_env({"TRAFFICMIND_PROFILE": "fantasy"}):
      with self.assertRaises(ValueError) as ctx:
        active_profile()
      self.assertIn("fantasy", str(ctx.exception))


class TestServiceConfig(unittest.TestCase):

  def test_valid_construction(self):
    cfg = ServiceConfig(
        profile=Profile.LOCAL,
        stale_after_seconds=30.0,
        history_size=50,
        log_level="INFO",
    )
    self.assertEqual(cfg.profile, Profile.LOCAL)
    self.assertEqual(cfg.stale_after_seconds, 30.0)

  def test_zero_staleness_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=0,
          history_size=50,
          log_level="INFO",
      )

  def test_negative_staleness_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=-1,
          history_size=50,
          log_level="INFO",
      )

  def test_zero_history_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=30,
          history_size=0,
          log_level="INFO",
      )

  def test_invalid_log_level_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=30,
          history_size=50,
          log_level="VERBOSE",
      )

  def test_prod_rejects_large_staleness(self):
    with self.assertRaises(ValueError) as ctx:
      ServiceConfig(
          profile=Profile.PROD,
          stale_after_seconds=60,
          history_size=50,
          log_level="WARNING",
      )
    self.assertIn("prod", str(ctx.exception))

  def test_staging_rejects_large_staleness(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.STAGING,
          stale_after_seconds=31,
          history_size=50,
          log_level="INFO",
          polling_url="http://localhost:9090/states",
      )

  def test_local_allows_large_staleness(self):
    cfg = ServiceConfig(
        profile=Profile.LOCAL,
        stale_after_seconds=120,
        history_size=50,
        log_level="DEBUG",
    )
    self.assertEqual(cfg.stale_after_seconds, 120)

  def test_zero_polling_timeout_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=30,
          history_size=50,
          log_level="INFO",
          polling_timeout_seconds=0,
      )

  def test_zero_webhook_buffer_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=30,
          history_size=50,
          log_level="INFO",
          webhook_max_buffer=0,
      )

  def test_staging_requires_polling_url(self):
    with self.assertRaises(ValueError) as ctx:
      ServiceConfig(
          profile=Profile.STAGING,
          stale_after_seconds=15,
          history_size=50,
          log_level="INFO",
      )
    self.assertIn("polling_url", str(ctx.exception))

  def test_prod_requires_polling_url(self):
    with self.assertRaises(ValueError) as ctx:
      ServiceConfig(
          profile=Profile.PROD,
          stale_after_seconds=15,
          history_size=50,
          log_level="WARNING",
      )
    self.assertIn("polling_url", str(ctx.exception))

  def test_invalid_polling_url_rejected(self):
    with self.assertRaises(ValueError):
      ServiceConfig(
          profile=Profile.LOCAL,
          stale_after_seconds=30,
          history_size=50,
          log_level="INFO",
          polling_url="ftp://localhost/states",
      )


class TestFromEnv(unittest.TestCase):

  def test_defaults_for_local(self):
    env = os.environ.copy()
    for key in list(env):
      if key.startswith("TRAFFICMIND_"):
        del env[key]
    with _patch_env(env):
      cfg = from_env()
    self.assertEqual(cfg.profile, Profile.LOCAL)
    self.assertEqual(cfg.stale_after_seconds, 60.0)
    self.assertEqual(cfg.log_level, "DEBUG")

  def test_override_via_env(self):
    env = {
        "TRAFFICMIND_PROFILE": "dev",
        "TRAFFICMIND_STALE_AFTER_SECONDS": "25",
        "TRAFFICMIND_HISTORY_SIZE": "75",
        "TRAFFICMIND_LOG_LEVEL": "error",
    }
    with _patch_env(env):
      cfg = from_env()
    self.assertEqual(cfg.profile, Profile.DEV)
    self.assertEqual(cfg.stale_after_seconds, 25.0)
    self.assertEqual(cfg.history_size, 75)
    self.assertEqual(cfg.log_level, "ERROR")

  def test_invalid_number_env_raises(self):
    with _patch_env({"TRAFFICMIND_STALE_AFTER_SECONDS": "abc"}):
      with self.assertRaises(ValueError):
        from_env()

  def test_invalid_int_env_raises(self):
    with _patch_env({"TRAFFICMIND_HISTORY_SIZE": "3.5"}):
      with self.assertRaises(ValueError):
        from_env()

  def test_polling_url_picked_up(self):
    url = "http://localhost:9090/states"
    with _patch_env({"TRAFFICMIND_POLLING_URL": url}):
      cfg = from_env()
    self.assertEqual(cfg.polling_url, url)

  def test_empty_polling_url_is_none(self):
    with _patch_env({"TRAFFICMIND_POLLING_URL": ""}):
      cfg = from_env()
    self.assertIsNone(cfg.polling_url)

  def test_blank_polling_url_is_none(self):
    with _patch_env({"TRAFFICMIND_POLLING_URL": "   "}):
      cfg = from_env()
    self.assertIsNone(cfg.polling_url)

  def test_staging_env_requires_polling_url(self):
    with _patch_env({"TRAFFICMIND_PROFILE": "staging"}):
      with self.assertRaises(ValueError) as ctx:
        from_env()
    self.assertIn("polling_url", str(ctx.exception))

  def test_staging_env_accepts_polling_url(self):
    env = {
        "TRAFFICMIND_PROFILE": "staging",
        "TRAFFICMIND_POLLING_URL": "https://example.test/states",
    }
    with _patch_env(env):
      cfg = from_env()
    self.assertEqual(cfg.profile, Profile.STAGING)
    self.assertEqual(cfg.polling_url, env["TRAFFICMIND_POLLING_URL"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _patch_env:
  """Context manager that temporarily replaces ``os.environ``."""

  def __init__(self, env: dict[str, str]):
    self._env = env

  def __enter__(self):
    self._orig = os.environ.copy()
    os.environ.clear()
    os.environ.update(self._env)
    return self

  def __exit__(self, *args):
    os.environ.clear()
    os.environ.update(self._orig)


if __name__ == "__main__":
  unittest.main()
