"""Optional bridge between enterprise signal adapters and SignalSource."""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Sequence

from trafficmind.models import SignalState
from trafficmind.sources.base import SignalSource


@runtime_checkable
class ExternalSignalAdapter(Protocol):
  """Vendor-neutral adapter contract for external signal systems.

  TrafficMind already supports external signal integration via
  :class:`trafficmind.sources.base.SignalSource`. This bridge exists only for
  teams that want a consistent "adapter" vocabulary across outbound and
  inbound integrations.
  """

  def adapter_name(self) -> str:
    """Return a human-readable adapter identifier."""

  def fetch_signal_states(self) -> Sequence[SignalState]:
    """Return the latest available signal observations."""


class SignalSourceBridge(SignalSource):
  """Wrap an ExternalSignalAdapter in the existing SignalSource contract."""

  def __init__(self, adapter: ExternalSignalAdapter) -> None:
    self._adapter = adapter

  def fetch(self) -> Sequence[SignalState]:
    return tuple(self._adapter.fetch_signal_states())

  def source_name(self) -> str:
    return self._adapter.adapter_name()


def adapt_signal_adapter(adapter: ExternalSignalAdapter) -> SignalSource:
  """Return a SignalSource wrapper for an ExternalSignalAdapter."""
  return SignalSourceBridge(adapter)
