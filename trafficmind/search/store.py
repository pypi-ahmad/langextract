"""Searchable store protocol and in-memory implementation.

The ``SearchableStore`` protocol defines the backend contract that the
search executor queries against.  ``InMemorySearchStore`` provides a
lightweight in-memory implementation useful for testing and small
deployments.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Protocol, runtime_checkable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
  from trafficmind.registry.models import ProvenanceChain

from trafficmind.search.models import MatchEvidence
from trafficmind.search.models import RecordKind
from trafficmind.search.models import RecordReference
from trafficmind.search.models import ReviewStatus
from trafficmind.search.models import SearchFilter
from trafficmind.search.models import SearchHit
from trafficmind.search.models import TimeRange

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchableStore(Protocol):
  """Backend contract for the search executor.

  Implementations must accept a ``SearchFilter`` and return matching hits.
  The store is responsible for applying all filter dimensions it supports
  and returning only matching records with explicit references.
  """

  def query(self, search_filter: SearchFilter) -> list[SearchHit]:
    ...

  def count(self, search_filter: SearchFilter) -> int:
    ...


# ---------------------------------------------------------------------------
# Platform record – lightweight internal representation
# ---------------------------------------------------------------------------


class PlatformRecord:
  """A single indexed record in the in-memory store.

  This is a mutable container used only inside ``InMemorySearchStore``.
  External callers interact with frozen ``SearchHit`` instances.
  """

  __slots__ = (
      "record_kind",
      "record_id",
      "label",
      "detail",
      "timestamp",
      "junction_id",
      "camera_id",
      "event_type",
      "violation_type",
      "plate_text",
      "review_status",
      "vehicle_type",
      "phase_state",
      "metadata",
      "provenance",
  )

  def __init__(
      self,
      *,
      record_kind: RecordKind,
      record_id: str,
      label: str = "",
      detail: str = "",
      timestamp: float | None = None,
      junction_id: str | None = None,
      camera_id: str | None = None,
      event_type: str | None = None,
      violation_type: str | None = None,
      plate_text: str | None = None,
      review_status: ReviewStatus | None = None,
      vehicle_type: str | None = None,
      phase_state: str | None = None,
      metadata: dict[str, Any] | None = None,
      provenance: ProvenanceChain | None = None,
  ) -> None:
    self.record_kind = record_kind
    self.record_id = record_id
    self.label = label
    self.detail = detail
    self.timestamp = timestamp
    self.junction_id = junction_id
    self.camera_id = camera_id
    self.event_type = event_type
    self.violation_type = violation_type
    self.plate_text = plate_text
    self.review_status = review_status
    self.vehicle_type = vehicle_type
    self.phase_state = phase_state
    self.metadata = metadata or {}
    self.provenance = provenance


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class InMemorySearchStore:
  """Thread-safe in-memory searchable store.

  Records are scanned linearly with per-dimension filtering.  This is
  appropriate for testing and small-to-medium record sets.  Production
  deployments should implement ``SearchableStore`` against a real
  database or search index.
  """

  def __init__(self) -> None:
    self._records: list[PlatformRecord] = []

  def add(self, record: PlatformRecord) -> None:
    self._records.append(record)

  def add_many(self, records: Sequence[PlatformRecord]) -> None:
    self._records.extend(records)

  @property
  def size(self) -> int:
    return len(self._records)

  def clear(self) -> None:
    self._records.clear()

  def query(self, search_filter: SearchFilter) -> list[SearchHit]:
    matched = self._apply_filter(search_filter)
    # Sort by timestamp descending (most recent first), None last
    matched.sort(
        key=lambda r: r.timestamp if r.timestamp is not None else float("-inf"),
        reverse=True,
    )
    limit = search_filter.max_results
    return [_record_to_hit(r, search_filter) for r in matched[:limit]]

  def count(self, search_filter: SearchFilter) -> int:
    return len(self._apply_filter(search_filter))

  def _apply_filter(self, sf: SearchFilter) -> list[PlatformRecord]:
    results: list[PlatformRecord] = []
    for rec in self._records:
      if not _matches(rec, sf):
        continue
      results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def _matches(rec: PlatformRecord, sf: SearchFilter) -> bool:
  if sf.record_kinds and rec.record_kind not in sf.record_kinds:
    return False
  if sf.time_range and not _time_matches(rec.timestamp, sf.time_range):
    return False
  if sf.junction_ids and not _id_matches(rec.junction_id, sf.junction_ids):
    return False
  if sf.camera_ids and not _id_matches(rec.camera_id, sf.camera_ids):
    return False
  if sf.event_types and not _str_in(rec.event_type, sf.event_types):
    return False
  if sf.violation_types and not _str_in(rec.violation_type, sf.violation_types):
    return False
  if sf.plate_text and not _plate_matches(
      rec.plate_text, sf.plate_text, sf.plate_partial
  ):
    return False
  if sf.review_statuses and not _status_matches(
      rec.review_status, sf.review_statuses
  ):
    return False
  if sf.vehicle_types and not _str_in(rec.vehicle_type, sf.vehicle_types):
    return False
  if sf.text_terms and not _text_terms_match(rec, sf.text_terms):
    return False
  if sf.phase_states and not _str_in(rec.phase_state, sf.phase_states):
    return False
  return True


def _time_matches(ts: float | None, tr: TimeRange) -> bool:
  if ts is None:
    return False
  return tr.contains(ts)


def _id_matches(value: str | None, candidates: tuple[str, ...]) -> bool:
  if value is None:
    return False
  v = value.upper()
  return any(v == c.upper() for c in candidates)


def _str_in(value: str | None, candidates: tuple[str, ...]) -> bool:
  if value is None:
    return False
  v = value.lower()
  return any(v == c.lower() for c in candidates)


def _plate_matches(
    record_plate: str | None, query_plate: str, partial: bool
) -> bool:
  if record_plate is None:
    return False
  rp = record_plate.upper()
  qp = query_plate.upper()
  if not partial:
    return rp == qp
  # Partial: treat query as a glob pattern or substring
  if "*" in qp or "?" in qp:
    return fnmatch.fnmatch(rp, qp)
  return qp in rp


def _status_matches(
    record_status: ReviewStatus | None,
    candidates: tuple[ReviewStatus, ...],
) -> bool:
  if record_status is None:
    return False
  return record_status in candidates


def _text_terms_match(rec: PlatformRecord, terms: tuple[str, ...]) -> bool:
  searchable_parts = [
      rec.label,
      rec.detail,
      *[str(value) for value in rec.metadata.values()],
  ]
  return all(
      any(_contains_phrase(part, term) for part in searchable_parts if part)
      for term in terms
  )


def _contains_phrase(text: str, phrase: str) -> bool:
  return (
      re.search(
          r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", text.lower()
      )
      is not None
  )


def _first_matching_text_basis(rec: PlatformRecord, term: str) -> str:
  if rec.label and _contains_phrase(rec.label, term):
    return "matched in record label"
  if rec.detail and _contains_phrase(rec.detail, term):
    return "matched in record detail"
  for key, value in rec.metadata.items():
    if _contains_phrase(str(value), term):
      return f"matched in metadata[{key!r}]"
  return "matched in indexed text"


def _first_requested_match(
    value: str | None,
    candidates: tuple[str, ...],
) -> str:
  if value is None:
    return candidates[0]
  lowered = value.lower()
  for candidate in candidates:
    if lowered == candidate.lower():
      return candidate
  return candidates[0]


# ---------------------------------------------------------------------------
# Record → SearchHit conversion
# ---------------------------------------------------------------------------


def _record_to_hit(rec: PlatformRecord, sf: SearchFilter) -> SearchHit:
  matched_filters: list[str] = []
  matched_evidence: list[MatchEvidence] = []
  if sf.time_range and rec.timestamp is not None:
    matched_filters.append("time_range")
    matched_evidence.append(
        MatchEvidence(
            filter_name="time_range",
            requested_value=f"[{sf.time_range.start}, {sf.time_range.end})",
            matched_value=str(rec.timestamp),
            basis="record timestamp falls inside the requested time range",
        )
    )
  if sf.junction_ids and rec.junction_id:
    matched_filters.append("junction_id")
    matched_evidence.append(
        MatchEvidence(
            filter_name="junction_id",
            requested_value=_first_requested_match(
                rec.junction_id, sf.junction_ids
            ),
            matched_value=rec.junction_id,
            basis="matched normalized junction identifier",
        )
    )
  if sf.camera_ids and rec.camera_id:
    matched_filters.append("camera_id")
    matched_evidence.append(
        MatchEvidence(
            filter_name="camera_id",
            requested_value=_first_requested_match(
                rec.camera_id, sf.camera_ids
            ),
            matched_value=rec.camera_id,
            basis="matched normalized camera identifier",
        )
    )
  if sf.event_types and rec.event_type:
    matched_filters.append("event_type")
    matched_evidence.append(
        MatchEvidence(
            filter_name="event_type",
            requested_value=_first_requested_match(
                rec.event_type, sf.event_types
            ),
            matched_value=rec.event_type,
            basis="matched record event_type field",
        )
    )
  if sf.violation_types and rec.violation_type:
    matched_filters.append("violation_type")
    matched_evidence.append(
        MatchEvidence(
            filter_name="violation_type",
            requested_value=_first_requested_match(
                rec.violation_type, sf.violation_types
            ),
            matched_value=rec.violation_type,
            basis="matched record violation_type field",
        )
    )
  if sf.plate_text and rec.plate_text:
    matched_filters.append("plate_text")
    matched_evidence.append(
        MatchEvidence(
            filter_name="plate_text",
            requested_value=sf.plate_text,
            matched_value=rec.plate_text,
            basis=(
                "partial plate match against indexed plate_text"
                if sf.plate_partial
                else "exact plate match against indexed plate_text"
            ),
        )
    )
  if sf.review_statuses and rec.review_status:
    matched_filters.append("review_status")
    matched_evidence.append(
        MatchEvidence(
            filter_name="review_status",
            requested_value=next(
                (
                    status.value
                    for status in sf.review_statuses
                    if status == rec.review_status
                ),
                rec.review_status.value,
            ),
            matched_value=rec.review_status.value,
            basis="matched record review_status field",
        )
    )
  if sf.vehicle_types and rec.vehicle_type:
    matched_filters.append("vehicle_type")
    matched_evidence.append(
        MatchEvidence(
            filter_name="vehicle_type",
            requested_value=_first_requested_match(
                rec.vehicle_type, sf.vehicle_types
            ),
            matched_value=rec.vehicle_type,
            basis="matched record vehicle_type field",
        )
    )
  if sf.text_terms:
    matched_filters.append("text_term")
    for term in sf.text_terms:
      matched_evidence.append(
          MatchEvidence(
              filter_name="text_term",
              requested_value=term,
              matched_value=term,
              basis=_first_matching_text_basis(rec, term),
          )
      )

  return SearchHit(
      reference=RecordReference(
          record_kind=rec.record_kind,
          record_id=rec.record_id,
          label=rec.label,
          detail=rec.detail,
          provenance=rec.provenance,
      ),
      matched_filters=tuple(matched_filters),
      matched_evidence=tuple(matched_evidence),
      timestamp=rec.timestamp,
      junction_id=rec.junction_id,
      camera_id=rec.camera_id,
      event_type=rec.event_type,
      violation_type=rec.violation_type,
      plate_text=rec.plate_text,
      review_status=rec.review_status,
      vehicle_type=rec.vehicle_type,
      metadata=dict(rec.metadata),
      provenance=rec.provenance,
  )
