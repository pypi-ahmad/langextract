"""File-based signal data ingestion.

Reads signal states from JSON or CSV files on disk.  Each record maps to
a ``SignalState`` instance.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Sequence

from trafficmind.exceptions import InvalidSignalDataError
from trafficmind.models import PhaseState
from trafficmind.models import SignalState
from trafficmind.models import SourceType
from trafficmind.sources.base import SignalSource

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {
    "junction_id",
    "controller_id",
    "phase_id",
    "state",
    "timestamp",
}


def _parse_record(raw: dict, source_type: SourceType) -> SignalState:
  missing = _REQUIRED_FIELDS - raw.keys()
  if missing:
    raise InvalidSignalDataError(f"Missing fields: {sorted(missing)}")

  try:
    state = PhaseState(raw["state"])
  except ValueError:
    raise InvalidSignalDataError(f"Unknown phase state: {raw['state']!r}")

  return SignalState(
      junction_id=str(raw["junction_id"]),
      controller_id=str(raw["controller_id"]),
      phase_id=str(raw["phase_id"]),
      state=state,
      timestamp=float(raw["timestamp"]),
      source_type=source_type,
      confidence=float(raw.get("confidence", 1.0)),
      metadata={
          k: v
          for k, v in raw.items()
          if k not in _REQUIRED_FIELDS and k != "confidence"
      },
  )


class FileFeedSource(SignalSource):
  """Reads signal states from a JSON or CSV file.

  JSON files should contain a top-level list of record objects.
  CSV files should have a header row matching the required field names.
  """

  def __init__(
      self,
      path: str | Path,
      *,
      source_type: SourceType = SourceType.FILE_FEED,
  ) -> None:
    self._path = Path(path)
    self._source_type = source_type

  def source_name(self) -> str:
    return f"file_feed:{self._path.name}"

  def fetch(self) -> Sequence[SignalState]:
    if not self._path.exists():
      logger.warning("Signal file not found: %s", self._path)
      return []

    suffix = self._path.suffix.lower()
    if suffix == ".json":
      return self._read_json()
    elif suffix == ".csv":
      return self._read_csv()
    else:
      logger.warning("Unsupported file type: %s", suffix)
      return []

  def _read_json(self) -> list[SignalState]:
    text = self._path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
      raise InvalidSignalDataError("JSON root must be a list of records")
    return [_parse_record(r, self._source_type) for r in data]

  def _read_csv(self) -> list[SignalState]:
    states: list[SignalState] = []
    with self._path.open(newline="", encoding="utf-8") as fh:
      reader = csv.DictReader(fh)
      for row in reader:
        states.append(_parse_record(dict(row), self._source_type))
    return states
