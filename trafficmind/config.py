"""Environment profile and configuration validation for TrafficMind.

Provides a lightweight :class:`ServiceConfig` that validates required
settings at construction time so deployment misconfigurations surface
immediately rather than at first use.

Supported profiles
------------------
The ``TRAFFICMIND_PROFILE`` environment variable selects a named profile
that adjusts default values.  Any setting can still be overridden via
explicit constructor arguments or per-setting environment variables.

``local`` (default)
    Suitable for single-machine development.  The signal store uses a
    generous staleness window and logging is verbose.

``dev``
    Shared development server.  Staleness window is tighter and debug
    logging is disabled.

``staging``
    Pre-production verification.  Uses the same conservative defaults
    as the strictest profile but is intended for integration rehearsal,
    not as a claim of production readiness.

``prod``
    Strictest runtime validation profile.  This is useful for
    production-like checks in CI or controlled environments, but does
    not imply that the subsystem is fully production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class Profile(str, enum.Enum):
  """Deployment profiles supported by TrafficMind."""

  LOCAL = "local"
  DEV = "dev"
  STAGING = "staging"
  PROD = "prod"


# Defaults keyed by profile.
_STALE_AFTER: dict[Profile, float] = {
    Profile.LOCAL: 60.0,
    Profile.DEV: 30.0,
    Profile.STAGING: 15.0,
    Profile.PROD: 15.0,
}

_HISTORY_SIZE: dict[Profile, int] = {
    Profile.LOCAL: 100,
    Profile.DEV: 50,
    Profile.STAGING: 50,
    Profile.PROD: 50,
}

_LOG_LEVEL: dict[Profile, str] = {
    Profile.LOCAL: "DEBUG",
    Profile.DEV: "INFO",
    Profile.STAGING: "INFO",
    Profile.PROD: "WARNING",
}


def active_profile() -> Profile:
  """Return the profile selected by ``TRAFFICMIND_PROFILE``.

  Falls back to :pyattr:`Profile.LOCAL` when the variable is unset or
  empty.  Raises :class:`ValueError` for unrecognised profile names.
  """
  raw = os.getenv("TRAFFICMIND_PROFILE", "").strip().lower()
  if not raw:
    return Profile.LOCAL
  try:
    return Profile(raw)
  except ValueError:
    allowed = ", ".join(p.value for p in Profile)
    raise ValueError(
        f"Unknown TRAFFICMIND_PROFILE={raw!r}; expected one of: {allowed}"
    ) from None


@dataclass(frozen=True)
class ServiceConfig:
  """Validated configuration bundle for :class:`SignalService`.

  Construction-time validation ensures the config is internally
  consistent.  Use :func:`from_env` to build a config from the
  current environment.
  """

  profile: Profile
  stale_after_seconds: float
  history_size: int
  log_level: str
  polling_url: str | None = None
  polling_timeout_seconds: float = 10.0
  webhook_max_buffer: int = 10_000

  def __post_init__(self) -> None:
    if self.stale_after_seconds <= 0:
      raise ValueError(
          "stale_after_seconds must be positive, "
          f"got {self.stale_after_seconds}"
      )
    if self.history_size < 1:
      raise ValueError(f"history_size must be >= 1, got {self.history_size}")
    if self.polling_timeout_seconds <= 0:
      raise ValueError(
          "polling_timeout_seconds must be positive, "
          f"got {self.polling_timeout_seconds}"
      )
    if self.webhook_max_buffer < 1:
      raise ValueError(
          f"webhook_max_buffer must be >= 1, got {self.webhook_max_buffer}"
      )
    if self.log_level not in (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ):
      raise ValueError(
          f"log_level must be a standard Python level, got {self.log_level!r}"
      )
    if self.profile in (Profile.STAGING, Profile.PROD):
      if self.stale_after_seconds > 30:
        raise ValueError(
            "stale_after_seconds must be <= 30 in "
            f"{self.profile.value} profile, "
            f"got {self.stale_after_seconds}"
        )
      if not self.polling_url:
        raise ValueError(
            f"polling_url is required in {self.profile.value} profile"
        )

    if self.polling_url:
      parsed = urlparse(self.polling_url)
      if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"polling_url must be a valid http(s) URL, got {self.polling_url!r}"
        )


def from_env() -> ServiceConfig:
  """Build a :class:`ServiceConfig` from environment variables.

  Reads ``TRAFFICMIND_PROFILE`` to select defaults, then allows
  per-setting overrides via individual env vars.
  """
  profile = active_profile()

  def _float(var: str, default: float) -> float:
    raw = os.getenv(var, "").strip()
    if not raw:
      return default
    try:
      return float(raw)
    except ValueError:
      raise ValueError(f"{var}={raw!r} is not a valid number") from None

  def _int(var: str, default: int) -> int:
    raw = os.getenv(var, "").strip()
    if not raw:
      return default
    try:
      return int(raw)
    except ValueError:
      raise ValueError(f"{var}={raw!r} is not a valid integer") from None

  polling_url = os.getenv("TRAFFICMIND_POLLING_URL", "").strip() or None

  return ServiceConfig(
      profile=profile,
      stale_after_seconds=_float(
          "TRAFFICMIND_STALE_AFTER_SECONDS",
          _STALE_AFTER[profile],
      ),
      history_size=_int(
          "TRAFFICMIND_HISTORY_SIZE",
          _HISTORY_SIZE[profile],
      ),
      log_level=os.getenv(
          "TRAFFICMIND_LOG_LEVEL",
          _LOG_LEVEL[profile],
      )
      .strip()
      .upper(),
      polling_url=polling_url,
      polling_timeout_seconds=_float(
          "TRAFFICMIND_POLLING_TIMEOUT",
          10.0,
      ),
      webhook_max_buffer=_int(
          "TRAFFICMIND_WEBHOOK_MAX_BUFFER",
          10_000,
      ),
  )
