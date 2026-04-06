"""Webhook / push-event signal receiver.

Accepts signal-state records pushed from external systems and buffers
them for consumption by the arbitration layer.
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Sequence

from trafficmind.exceptions import InvalidSignalDataError
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.sources.base import SignalSource
from trafficmind.sources.file_feed import _parse_record


class WebhookReceiver(SignalSource):
  """Thread-safe buffer for push-based signal events.

  External code (e.g. an HTTP handler) calls ``receive()`` to push
  records into the buffer.  ``fetch()`` drains the buffer and returns
  all accumulated states since the last call.
  """

  def __init__(
      self,
      *,
      max_buffer: int = 10_000,
      source_type: SourceType = SourceType.WEBHOOK,
  ) -> None:
    self._source_type = source_type
    self._buffer: deque[SignalState] = deque(maxlen=max_buffer)
    self._lock = threading.Lock()

  def source_name(self) -> str:
    return "webhook"

  def receive(self, raw: dict) -> SignalState:
    """Validate and buffer one inbound signal record.

    Returns the parsed ``SignalState`` so the caller can acknowledge.
    Raises ``InvalidSignalDataError`` on bad input.
    """
    state = _parse_record(raw, self._source_type)
    with self._lock:
      self._buffer.append(state)
    return state

  def receive_batch(self, records: list[dict]) -> list[SignalState]:
    """Validate and buffer multiple records atomically."""
    states = [_parse_record(r, self._source_type) for r in records]
    with self._lock:
      self._buffer.extend(states)
    return states

  def fetch(self) -> Sequence[SignalState]:
    """Drain and return all buffered states."""
    with self._lock:
      result = list(self._buffer)
      self._buffer.clear()
    return result

  @property
  def pending_count(self) -> int:
    with self._lock:
      return len(self._buffer)
