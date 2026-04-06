"""Abstract base for all signal sources."""

from __future__ import annotations

import abc
from typing import Sequence

from trafficmind.models import SignalState


class SignalSource(abc.ABC):
  """Interface every signal-state source must implement."""

  @abc.abstractmethod
  def fetch(self) -> Sequence[SignalState]:
    """Return the latest signal states available from this source.

    Implementations should return an empty sequence when no data is
    available rather than raising.
    """

  @abc.abstractmethod
  def source_name(self) -> str:
    """Human-readable name identifying this source instance."""
