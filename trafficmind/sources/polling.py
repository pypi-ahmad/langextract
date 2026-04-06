"""HTTP-polling signal source.

Periodically fetches signal states from an external HTTP endpoint
that returns a JSON array of signal records.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence
import urllib.error
import urllib.request

from trafficmind.exceptions import SourceUnavailableError
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.sources.base import SignalSource
from trafficmind.sources.file_feed import _parse_record

logger = logging.getLogger(__name__)


class PollingSource(SignalSource):
  """Fetches signal state from an HTTP JSON endpoint.

  Each call to ``fetch()`` performs one HTTP GET.  The endpoint must
  return a JSON array of signal-state records.
  """

  def __init__(
      self,
      url: str,
      *,
      timeout_seconds: float = 10.0,
      source_type: SourceType = SourceType.POLLING,
      headers: dict[str, str] | None = None,
  ) -> None:
    self._url = url
    self._timeout = timeout_seconds
    self._source_type = source_type
    self._headers = headers or {}

  def source_name(self) -> str:
    return f"polling:{self._url}"

  def fetch(self) -> Sequence[SignalState]:
    req = urllib.request.Request(self._url, method="GET")
    for k, v in self._headers.items():
      req.add_header(k, v)
    try:
      with urllib.request.urlopen(req, timeout=self._timeout) as resp:
        body = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
      raise SourceUnavailableError(
          f"Failed to poll {self._url}: {exc}"
      ) from exc

    data = json.loads(body)
    if not isinstance(data, list):
      logger.warning("Polling response is not a JSON list")
      return []
    return [_parse_record(r, self._source_type) for r in data]
