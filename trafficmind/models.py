"""Core data models for signal controller integration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
import time
from typing import Any


class SourceType(enum.Enum):
  """Origin of a signal-state observation."""

  CONTROLLER = "controller"
  VISION = "vision"
  FILE_FEED = "file_feed"
  POLLING = "polling"
  WEBHOOK = "webhook"
  SIMULATOR = "simulator"


class PhaseState(enum.Enum):
  """Possible traffic signal phase states."""

  RED = "red"
  AMBER = "amber"
  GREEN = "green"
  RED_AMBER = "red_amber"
  FLASHING_RED = "flashing_red"
  FLASHING_AMBER = "flashing_amber"
  OFF = "off"
  UNKNOWN = "unknown"


class ArbitrationMode(enum.Enum):
  """How to resolve between controller and vision signal state."""

  CONTROLLER_ONLY = "controller_only"
  VISION_ONLY = "vision_only"
  HYBRID = "hybrid"


# Phase states considered restrictive / stop-indicating.
RESTRICTIVE_STATES: frozenset[PhaseState] = frozenset({
    PhaseState.RED,
    PhaseState.RED_AMBER,
    PhaseState.FLASHING_RED,
})


@dataclass(frozen=True)
class SignalState:
  """A single signal-state observation from any source.

  Attributes:
      junction_id: Identifier for the junction / intersection.
      controller_id: Identifier for the signal controller.
      phase_id: The phase channel within the controller.
      state: The observed phase state.
      timestamp: Unix epoch seconds when the state was observed.
      source_type: Where this observation came from.
      confidence: Trust/confidence in the observation (0.0–1.0).
      metadata: Optional vendor- or source-specific extra data.
  """

  junction_id: str
  controller_id: str
  phase_id: str
  state: PhaseState
  timestamp: float
  source_type: SourceType
  confidence: float = 1.0
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not 0.0 <= self.confidence <= 1.0:
      raise ValueError(
          f"confidence must be between 0.0 and 1.0, got {self.confidence}"
      )

  @property
  def is_restrictive(self) -> bool:
    """True when the state indicates a stop / restriction."""
    return self.state in RESTRICTIVE_STATES

  def age(self, now: float | None = None) -> float:
    """Seconds elapsed since the observation timestamp."""
    return (time.time() if now is None else now) - self.timestamp

  @property
  def camera_id(self) -> str | None:
    """Camera identifier for vision observations, if available.

    Vision records may carry ``metadata["camera_id"]`` explicitly.
    For legacy observations where that metadata was not populated,
    ``controller_id`` is treated as the originating camera id.
    """
    if self.source_type != SourceType.VISION:
      return None
    camera_id = self.metadata.get("camera_id")
    if camera_id is not None:
      return str(camera_id)
    return self.controller_id or None


@dataclass(frozen=True)
class SignalReport:
  """Result of merging / arbitrating controller and vision signals.

  Attributes:
      junction_id: Junction this report covers.
      phase_id: Phase channel this report covers.
      resolved_state: The state chosen after arbitration.
      mode: The arbitration mode that was used.
      controller_state: The controller observation, if any.
      vision_state: The vision observation, if any.
      conflict: True when sources disagreed.
      stale: True when at least one source was stale.
      reason: Human-readable explanation of the resolution.
  """

  junction_id: str
  phase_id: str
  resolved_state: PhaseState
  mode: ArbitrationMode
  controller_state: SignalState | None = None
  vision_state: SignalState | None = None
  conflict: bool = False
  stale: bool = False
  reason: str = ""
