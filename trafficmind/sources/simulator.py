"""Local mock signal simulator.

Generates deterministic or random signal-state sequences for testing
and development without a real controller connection.
"""

from __future__ import annotations

import itertools
import time
from typing import Sequence

from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.sources.base import SignalSource

_DEFAULT_CYCLE: list[PhaseState] = [
    PhaseState.GREEN,
    PhaseState.AMBER,
    PhaseState.RED,
    PhaseState.RED_AMBER,
]


class SimulatorSource(SignalSource):
  """Cycles through phase states for one junction/controller.

  Each call to ``fetch()`` advances the cycle by one step and returns
  a single ``SignalState``.  Useful for integration tests and demos.
  """

  def __init__(
      self,
      junction_id: str = "SIM-J1",
      controller_id: str = "SIM-C1",
      phase_id: str = "SIM-P1",
      *,
      cycle: list[PhaseState] | None = None,
      confidence: float = 0.9,
  ) -> None:
    self._junction_id = junction_id
    self._controller_id = controller_id
    self._phase_id = phase_id
    self._confidence = confidence
    self._cycle = itertools.cycle(cycle or _DEFAULT_CYCLE)

  def source_name(self) -> str:
    return f"simulator:{self._junction_id}/{self._controller_id}"

  def fetch(self) -> Sequence[SignalState]:
    state = next(self._cycle)
    return [
        SignalState(
            junction_id=self._junction_id,
            controller_id=self._controller_id,
            phase_id=self._phase_id,
            state=state,
            timestamp=time.time(),
            source_type=SourceType.SIMULATOR,
            confidence=self._confidence,
        )
    ]
