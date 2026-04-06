"""TrafficMind command-line entry points."""

from __future__ import annotations

import sys

from trafficmind.config import from_env
from trafficmind.health import run_startup_checks


def main() -> None:
  """Run startup sanity checks using environment-derived config."""
  try:
    config = from_env()
  except ValueError as exc:
    print(f"FAIL  Config error: {exc}", file=sys.stderr)
    raise SystemExit(1) from None

  print(f"Profile : {config.profile.value}")
  print(f"Stale   : {config.stale_after_seconds}s")
  print(f"History : {config.history_size}")
  print(f"LogLevel: {config.log_level}")
  if config.polling_url:
    print(f"Polling : {config.polling_url}")
  print()

  problems = run_startup_checks(config)
  if problems:
    for problem in problems:
      print(f"FAIL  {problem}", file=sys.stderr)
    raise SystemExit(1)

  print("OK  All startup checks passed.")


if __name__ == "__main__":
  main()
