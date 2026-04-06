"""Arbitration layer: merge controller and vision signal states.

Supports three modes:
- **controller_only** – use only external signal data.
- **vision_only** – use only vision-derived data.
- **hybrid** – merge both sources with conflict detection.

Design principles:
- Conservative: when in doubt, assume the most restrictive state.
- Transparent: conflicts and staleness are always surfaced.
- Provenance-preserving: original observations are attached to the report.
"""

from __future__ import annotations

import time

from trafficmind.models import ArbitrationMode
from trafficmind.models import PhaseState
from trafficmind.models import RESTRICTIVE_STATES
from trafficmind.models import SignalReport
from trafficmind.models import SignalState
from trafficmind.store import SignalStore


class Arbitrator:
  """Resolves the effective signal state for a junction phase.

  Parameters:
      store: The ``SignalStore`` holding the latest observations.
      mode: Default arbitration mode.
      stale_threshold: Override the store's staleness window (seconds).
          ``None`` defers to the store's own ``stale_after_seconds``.
  """

  def __init__(
      self,
      store: SignalStore,
      *,
      mode: ArbitrationMode = ArbitrationMode.HYBRID,
      stale_threshold: float | None = None,
  ) -> None:
    self.store = store
    self.mode = mode
    if stale_threshold is not None:
      self.store.stale_after_seconds = stale_threshold

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def resolve(
      self,
      junction_id: str,
      phase_id: str,
      *,
      mode: ArbitrationMode | None = None,
      now: float | None = None,
  ) -> SignalReport:
    """Produce a ``SignalReport`` for the given junction/phase."""
    mode = mode or self.mode
    now = now or time.time()

    ctrl = self.store.get_controller_state(junction_id, phase_id)
    vision = self.store.get_vision_state(junction_id, phase_id)

    if mode == ArbitrationMode.CONTROLLER_ONLY:
      return self._resolve_single(
          junction_id, phase_id, ctrl, "controller", mode, now
      )
    if mode == ArbitrationMode.VISION_ONLY:
      return self._resolve_single(
          junction_id, phase_id, vision, "vision", mode, now
      )
    return self._resolve_hybrid(junction_id, phase_id, ctrl, vision, now)

  # ------------------------------------------------------------------
  # Internal
  # ------------------------------------------------------------------

  def _resolve_single(
      self,
      junction_id: str,
      phase_id: str,
      state: SignalState | None,
      label: str,
      mode: ArbitrationMode,
      now: float,
  ) -> SignalReport:
    if state is None:
      return SignalReport(
          junction_id=junction_id,
          phase_id=phase_id,
          resolved_state=PhaseState.UNKNOWN,
          mode=mode,
          stale=True,
          reason=f"No {label} data available",
      )

    stale = self.store.is_stale(state, now)
    resolved = PhaseState.UNKNOWN if stale else state.state
    return SignalReport(
        junction_id=junction_id,
        phase_id=phase_id,
        resolved_state=resolved,
        mode=mode,
        controller_state=state if label == "controller" else None,
        vision_state=state if label == "vision" else None,
        stale=stale,
        reason=(
            f"{label} stale ({state.age(now):.1f}s)"
            if stale
            else f"{label} data used"
        ),
    )

  def _resolve_hybrid(
      self,
      junction_id: str,
      phase_id: str,
      ctrl: SignalState | None,
      vision: SignalState | None,
      now: float,
  ) -> SignalReport:
    ctrl_stale = ctrl is not None and self.store.is_stale(ctrl, now)
    vision_stale = vision is not None and self.store.is_stale(vision, now)
    any_stale = ctrl_stale or vision_stale

    # Both missing → unknown.
    if ctrl is None and vision is None:
      return SignalReport(
          junction_id=junction_id,
          phase_id=phase_id,
          resolved_state=PhaseState.UNKNOWN,
          mode=ArbitrationMode.HYBRID,
          stale=True,
          reason="No controller or vision data available",
      )

    # Only one present → degrade to single-source.
    if ctrl is None:
      return self._degrade(
          junction_id, phase_id, vision, "vision", vision_stale, now
      )
    if vision is None:
      return self._degrade(
          junction_id, phase_id, ctrl, "controller", ctrl_stale, now
      )

    # Both present – check agreement.
    agree = ctrl.state == vision.state
    conflict = not agree

    if agree:
      # Sources agree on the state.  Use it even if stale —
      # corroboration from two independent sources is strong.
      return SignalReport(
          junction_id=junction_id,
          phase_id=phase_id,
          resolved_state=ctrl.state,
          mode=ArbitrationMode.HYBRID,
          controller_state=ctrl,
          vision_state=vision,
          conflict=False,
          stale=any_stale,
          reason=(
              "Controller and vision agree"
              + (
                  "; note: "
                  + self._stale_detail(
                      ctrl, vision, ctrl_stale, vision_stale, now
                  )
                  if any_stale
                  else ""
              )
          ),
      )

    # Conflict or staleness – conservative resolution.
    resolved = self._conservative_pick(ctrl, vision, ctrl_stale, vision_stale)

    parts: list[str] = []
    if conflict:
      parts.append(
          f"Conflict: controller={ctrl.state.value} vs"
          f" vision={vision.state.value}"
      )
    if ctrl_stale:
      parts.append(f"controller stale ({ctrl.age(now):.1f}s)")
    if vision_stale:
      parts.append(f"vision stale ({vision.age(now):.1f}s)")
    parts.append(f"resolved={resolved.value}")

    return SignalReport(
        junction_id=junction_id,
        phase_id=phase_id,
        resolved_state=resolved,
        mode=ArbitrationMode.HYBRID,
        controller_state=ctrl,
        vision_state=vision,
        conflict=conflict,
        stale=any_stale,
        reason="; ".join(parts),
    )

  @staticmethod
  def _conservative_pick(
      ctrl: SignalState,
      vision: SignalState,
      ctrl_stale: bool,
      vision_stale: bool,
  ) -> PhaseState:
    """Choose the most restrictive / safest state.

    Strategy:
    1. Both stale → UNKNOWN (don't trust either).
    2. If either non-stale source says RED, choose RED.
    3. Prefer the non-stale source over the stale one.
    4. Both fresh → prefer the more restrictive state; if
       equally restrictive, prefer higher confidence.
    """
    # 1. Both stale → refuse to guess.
    if ctrl_stale and vision_stale:
      return PhaseState.UNKNOWN

    # 2. If either non-stale source says stop → stop.
    if not ctrl_stale and ctrl.state in RESTRICTIVE_STATES:
      return ctrl.state
    if not vision_stale and vision.state in RESTRICTIVE_STATES:
      return vision.state

    # 3. One stale, one fresh → trust the fresh one.
    if ctrl_stale and not vision_stale:
      return vision.state
    if vision_stale and not ctrl_stale:
      return ctrl.state

    # 4. Both fresh, neither restrictive → prefer the more
    #    restrictive PhaseState, falling back to higher confidence.
    if (
        ctrl.state in RESTRICTIVE_STATES
        and vision.state not in RESTRICTIVE_STATES
    ):
      return ctrl.state
    if (
        vision.state in RESTRICTIVE_STATES
        and ctrl.state not in RESTRICTIVE_STATES
    ):
      return vision.state
    if ctrl.confidence >= vision.confidence:
      return ctrl.state
    return vision.state

  @staticmethod
  def _stale_detail(
      ctrl: SignalState,
      vision: SignalState,
      ctrl_stale: bool,
      vision_stale: bool,
      now: float,
  ) -> str:
    parts: list[str] = []
    if ctrl_stale:
      parts.append(f"controller stale ({ctrl.age(now):.1f}s)")
    if vision_stale:
      parts.append(f"vision stale ({vision.age(now):.1f}s)")
    return "; ".join(parts)

  def _degrade(
      self,
      junction_id: str,
      phase_id: str,
      state: SignalState,
      label: str,
      stale: bool,
      now: float,
  ) -> SignalReport:
    resolved = PhaseState.UNKNOWN if stale else state.state
    return SignalReport(
        junction_id=junction_id,
        phase_id=phase_id,
        resolved_state=resolved,
        mode=ArbitrationMode.HYBRID,
        controller_state=state if label == "controller" else None,
        vision_state=state if label == "vision" else None,
        stale=stale,
        reason=(
            f"Only {label} available"
            + (f", stale ({state.age(now):.1f}s)" if stale else "")
        ),
    )
