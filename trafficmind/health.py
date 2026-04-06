"""Startup sanity checks and health probes for TrafficMind.

:func:`run_startup_checks` should be called once when the service boots.
It validates that configuration is loadable, the signal store can be
instantiated, and optional external dependencies (polling endpoints)
are reachable.  Failures are returned as a list of human-readable
strings rather than raised — callers decide whether to abort or warn.

:func:`health_snapshot` returns a lightweight readiness summary suitable
for a ``/healthz``-style endpoint or a periodic self-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
import time
from typing import Sequence, TYPE_CHECKING
import urllib.error
import urllib.request

from trafficmind.config import from_env
from trafficmind.config import ServiceConfig

if TYPE_CHECKING:
  from trafficmind.service import SignalService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Startup checks
# ------------------------------------------------------------------


def run_startup_checks(
    config: ServiceConfig | None = None,
) -> list[str]:
  """Run pre-flight validations and return any problems found.

  Returns an empty list when everything looks good.
  """
  problems: list[str] = []

  # 1. Config loads without error.
  if config is None:
    try:
      config = from_env()
    except (ValueError, TypeError) as exc:
      problems.append(f"Config validation failed: {exc}")
      return problems  # No point continuing.

  # 2. Signal store can be instantiated with the configured values.
  try:
    from trafficmind.store import SignalStore

    SignalStore(
        stale_after_seconds=config.stale_after_seconds,
        history_size=config.history_size,
    )
  except Exception as exc:  # noqa: BLE001
    problems.append(f"SignalStore instantiation failed: {exc}")

  # 3. If a polling URL is configured, check basic reachability.
  if config.polling_url:
    try:
      req = urllib.request.Request(config.polling_url, method="HEAD")
      with urllib.request.urlopen(req, timeout=config.polling_timeout_seconds):
        pass
    except (urllib.error.URLError, OSError) as exc:
      problems.append(
          f"Polling endpoint unreachable ({config.polling_url}): {exc}"
      )

  if problems:
    for p in problems:
      logger.warning("Startup check failed: %s", p)
  else:
    logger.info(
        "All startup checks passed (profile=%s)",
        config.profile.value,
    )

  return problems


# ------------------------------------------------------------------
# Health snapshot
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HealthSnapshot:
  """Lightweight readiness snapshot.

  Attributes:
      ok: ``True`` when the service is considered healthy.
      profile: Active deployment profile name.
      uptime_seconds: Seconds since the snapshot reference time.
      store_junctions: Number of junctions currently tracked.
      store_observations: Number of latest per-source observations retained.
      source_count: Number of registered signal sources.
      problems: Non-empty when ``ok`` is ``False``.
  """

  ok: bool
  profile: str
  uptime_seconds: float
  store_junctions: int = 0
  store_observations: int = 0
  source_count: int = 0
  problems: tuple[str, ...] = ()

  def as_dict(self) -> dict:
    """Serialise to a plain dict for JSON responses."""
    return {
        "ok": self.ok,
        "profile": self.profile,
        "uptime_seconds": round(self.uptime_seconds, 2),
        "store_junctions": self.store_junctions,
        "store_observations": self.store_observations,
        "source_count": self.source_count,
        "problems": list(self.problems),
    }


def health_snapshot(
    *,
    config: ServiceConfig | None = None,
    service: SignalService | None = None,
    start_time: float | None = None,
) -> HealthSnapshot:
  """Build a health snapshot from the current service state.

  Parameters:
      config: Resolved service configuration.
      service: A :class:`~trafficmind.service.SignalService` instance.
      start_time: Unix epoch when the service started, used to
          compute uptime.
  """
  problems: list[str] = []
  now = time.time()
  uptime = (now - start_time) if start_time is not None else 0.0

  profile_name = config.profile.value if config else "unknown"

  store_junctions = 0
  store_observations = 0
  source_count = 0

  if service is not None:
    try:
      store_junctions = len(service.store.all_junctions())
    except Exception:  # noqa: BLE001
      problems.append("Failed to read store junctions")

    try:
      store_observations = service.store.latest_observation_count
    except Exception:  # noqa: BLE001
      problems.append("Failed to read store observations")

    try:
      source_count = len(service.sources)
    except Exception:  # noqa: BLE001
      problems.append("Failed to read source count")

  return HealthSnapshot(
      ok=len(problems) == 0,
      profile=profile_name,
      uptime_seconds=uptime,
      store_junctions=store_junctions,
      store_observations=store_observations,
      source_count=source_count,
      problems=tuple(problems),
  )
