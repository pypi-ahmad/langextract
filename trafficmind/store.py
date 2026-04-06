"""In-memory signal state store with staleness tracking.

Keeps the most-recent ``SignalState`` per (junction, phase, source_type)
tuple and supports staleness detection, history buffering, and
separation of controller vs. vision observations.
"""

from __future__ import annotations

from collections import defaultdict
from collections import deque
import threading
from typing import Sequence

from trafficmind.models import SignalState
from trafficmind.models import SourceType

# Source types considered "controller-side" (external signal data).
CONTROLLER_SOURCE_TYPES: frozenset[SourceType] = frozenset({
    SourceType.CONTROLLER,
    SourceType.FILE_FEED,
    SourceType.POLLING,
    SourceType.WEBHOOK,
    SourceType.SIMULATOR,
})

# Source types considered "vision-side".
VISION_SOURCE_TYPES: frozenset[SourceType] = frozenset({
    SourceType.VISION,
})


class SignalStore:
  """Thread-safe store of the latest signal observations.

  Parameters:
      stale_after_seconds: Observations older than this are considered stale.
      history_size: Number of past observations to keep per key.
  """

  def __init__(
      self,
      *,
      stale_after_seconds: float = 30.0,
      history_size: int = 50,
  ) -> None:
    self.stale_after_seconds = stale_after_seconds
    self._history_size = history_size
    self._lock = threading.Lock()

    # Keyed by (junction_id, phase_id, source_type)
    self._latest: dict[tuple[str, str, SourceType], SignalState] = {}
    self._history: dict[tuple[str, str, SourceType], deque[SignalState]] = (
        defaultdict(lambda: deque(maxlen=self._history_size))
    )

  # ------------------------------------------------------------------
  # Write
  # ------------------------------------------------------------------

  def update(self, state: SignalState) -> None:
    """Insert or replace the latest observation for the given key.

    Only updates ``_latest`` if *state* is at least as recent as
    the currently stored observation.  History always receives the
    new entry regardless of ordering.
    """
    key = (state.junction_id, state.phase_id, state.source_type)
    with self._lock:
      existing = self._latest.get(key)
      if existing is None or state.timestamp >= existing.timestamp:
        self._latest[key] = state
      self._history[key].append(state)

  def update_batch(self, states: Sequence[SignalState]) -> None:
    with self._lock:
      for state in states:
        key = (state.junction_id, state.phase_id, state.source_type)
        existing = self._latest.get(key)
        if existing is None or state.timestamp >= existing.timestamp:
          self._latest[key] = state
        self._history[key].append(state)

  # ------------------------------------------------------------------
  # Read
  # ------------------------------------------------------------------

  def get_latest(
      self,
      junction_id: str,
      phase_id: str,
      source_type: SourceType,
  ) -> SignalState | None:
    key = (junction_id, phase_id, source_type)
    with self._lock:
      return self._latest.get(key)

  def get_controller_state(
      self, junction_id: str, phase_id: str
  ) -> SignalState | None:
    """Return the freshest controller-side observation."""
    with self._lock:
      candidates = [
          v
          for (j, p, s), v in self._latest.items()
          if j == junction_id and p == phase_id and s in CONTROLLER_SOURCE_TYPES
      ]
    if not candidates:
      return None
    return max(candidates, key=lambda s: s.timestamp)

  def get_vision_state(
      self, junction_id: str, phase_id: str
  ) -> SignalState | None:
    """Return the freshest vision-side observation."""
    with self._lock:
      candidates = [
          v
          for (j, p, s), v in self._latest.items()
          if j == junction_id and p == phase_id and s in VISION_SOURCE_TYPES
      ]
    if not candidates:
      return None
    return max(candidates, key=lambda s: s.timestamp)

  def is_stale(self, state: SignalState, now: float | None = None) -> bool:
    return state.age(now) > self.stale_after_seconds

  def get_history(
      self,
      junction_id: str,
      phase_id: str,
      source_type: SourceType,
  ) -> list[SignalState]:
    key = (junction_id, phase_id, source_type)
    with self._lock:
      return list(self._history.get(key, []))

  def get_history_window(
      self,
      junction_id: str,
      phase_id: str,
      source_type: SourceType,
      *,
      start: float | None = None,
      end: float | None = None,
      camera_id: str | None = None,
  ) -> list[SignalState]:
    """Return history entries whose timestamp falls in [start, end).

    Either bound may be ``None`` to leave that side open.
    """
    key = (junction_id, phase_id, source_type)
    with self._lock:
      raw = list(self._history.get(key, []))
    out: list[SignalState] = []
    for s in raw:
      if start is not None and s.timestamp < start:
        continue
      if end is not None and s.timestamp >= end:
        continue
      if camera_id is not None and s.camera_id != camera_id:
        continue
      out.append(s)
    return out

  def all_history_window(
      self,
      junction_id: str,
      phase_id: str,
      *,
      start: float | None = None,
      end: float | None = None,
      source_types: frozenset[SourceType] | None = None,
      camera_id: str | None = None,
  ) -> list[SignalState]:
    """Return history across *all* source types for a junction/phase."""
    with self._lock:
      keys = [
          k
          for k in self._history
          if k[0] == junction_id
          and k[1] == phase_id
          and (source_types is None or k[2] in source_types)
      ]
      raw: list[SignalState] = []
      for k in keys:
        raw.extend(self._history[k])
    out: list[SignalState] = []
    for s in raw:
      if start is not None and s.timestamp < start:
        continue
      if end is not None and s.timestamp >= end:
        continue
      if camera_id is not None and s.camera_id != camera_id:
        continue
      out.append(s)
    out.sort(key=lambda s: s.timestamp)
    return out

  def all_junctions(self) -> set[str]:
    with self._lock:
      return {j for j, _, _ in self._latest}

  def phase_ids_for_junction(self, junction_id: str) -> set[str]:
    """Return all phase IDs stored for *junction_id*."""
    with self._lock:
      return {p for j, p, _ in self._latest if j == junction_id}

  def camera_ids_for_junction(self, junction_id: str) -> set[str]:
    """Return camera ids seen in vision history for *junction_id*."""
    with self._lock:
      camera_ids = {
          state.camera_id
          for (j, _, _), history in self._history.items()
          if j == junction_id
          for state in history
          if state.camera_id is not None
      }
    return {camera_id for camera_id in camera_ids if camera_id is not None}

  def phase_ids_for_camera(self, junction_id: str, camera_id: str) -> set[str]:
    """Return phase ids observed by *camera_id* in *junction_id*."""
    with self._lock:
      return {
          state.phase_id
          for (j, _, _), history in self._history.items()
          if j == junction_id
          for state in history
          if state.camera_id == camera_id
      }

  @property
  def latest_observation_count(self) -> int:
    """Number of latest per-source observations currently retained."""
    with self._lock:
      return len(self._latest)

  def clear(self) -> None:
    with self._lock:
      self._latest.clear()
      self._history.clear()
