"""Backend API / service layer for signal integration.

Provides a ``SignalService`` that wires sources, store, and arbitrator
together into a single facade for the rest of the platform.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from trafficmind.arbitration import Arbitrator
from trafficmind.models import ArbitrationMode
from trafficmind.models import SignalReport
from trafficmind.models import SignalState
from trafficmind.sources.base import SignalSource
from trafficmind.store import SignalStore

logger = logging.getLogger(__name__)


class SignalService:
  """High-level facade for the signal integration subsystem.

  Typical usage::

      store = SignalStore(stale_after_seconds=30)
      service = SignalService(store=store, mode=ArbitrationMode.HYBRID)
      service.register_source(my_polling_source)
      service.ingest()
      report = service.resolve("J-42", "P-1")
  """

  def __init__(
      self,
      store: SignalStore | None = None,
      *,
      mode: ArbitrationMode = ArbitrationMode.HYBRID,
      stale_after_seconds: float = 30.0,
  ) -> None:
    self.store = store or SignalStore(stale_after_seconds=stale_after_seconds)
    self.arbitrator = Arbitrator(self.store, mode=mode)
    self._sources: list[SignalSource] = []

  # ------------------------------------------------------------------
  # Source management
  # ------------------------------------------------------------------

  def register_source(self, source: SignalSource) -> None:
    logger.info("Registered signal source: %s", source.source_name())
    self._sources.append(source)

  @property
  def sources(self) -> Sequence[SignalSource]:
    return list(self._sources)

  # ------------------------------------------------------------------
  # Ingestion
  # ------------------------------------------------------------------

  def ingest(self) -> int:
    """Poll all registered sources and update the store.

    Returns the total number of new signal states ingested.
    """
    total = 0
    for source in self._sources:
      try:
        states = source.fetch()
      except Exception:
        logger.exception("Error fetching from source %s", source.source_name())
        continue
      self.store.update_batch(states)
      total += len(states)
    return total

  def ingest_states(self, states: Sequence[SignalState]) -> None:
    """Directly insert pre-built signal states into the store."""
    self.store.update_batch(states)

  # ------------------------------------------------------------------
  # Resolution
  # ------------------------------------------------------------------

  def resolve(
      self,
      junction_id: str,
      phase_id: str,
      *,
      mode: ArbitrationMode | None = None,
      now: float | None = None,
  ) -> SignalReport:
    """Arbitrate and return a ``SignalReport``."""
    return self.arbitrator.resolve(junction_id, phase_id, mode=mode, now=now)

  def resolve_all(
      self,
      *,
      mode: ArbitrationMode | None = None,
      now: float | None = None,
  ) -> list[SignalReport]:
    """Resolve every junction/phase combination known to the store."""
    reports: list[SignalReport] = []
    now = now or time.time()
    for junction_id in sorted(self.store.all_junctions()):
      for phase_id in sorted(self.store.phase_ids_for_junction(junction_id)):
        reports.append(
            self.arbitrator.resolve(junction_id, phase_id, mode=mode, now=now)
        )
    return reports
