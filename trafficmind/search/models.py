"""Typed models for the TrafficMind natural-language search layer.

Every search result carries an explicit reference chain back to the
underlying platform record so operators can verify what the query
matched and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from trafficmind.registry.models import ProvenanceChain


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecordKind(str, enum.Enum):
  """Category of a platform record that can appear in search results."""

  INCIDENT = "incident"
  VIOLATION = "violation"
  PLATE_READ = "plate_read"
  SIGNAL_STATE = "signal_state"
  EVIDENCE = "evidence"


class ReviewStatus(str, enum.Enum):
  """Operator review lifecycle stages."""

  PENDING = "pending"
  IN_REVIEW = "in_review"
  CONFIRMED = "confirmed"
  DISMISSED = "dismissed"
  ESCALATED = "escalated"


class ParseConfidence(str, enum.Enum):
  """How confidently the parser translated the natural-language query."""

  EXACT = "exact"
  HIGH = "high"
  PARTIAL = "partial"
  LOW = "low"


class SafeguardKind(str, enum.Enum):
  """Categories of safeguard triggers during query parsing or execution."""

  OVERLY_BROAD = "overly_broad"
  AMBIGUOUS_TIME = "ambiguous_time"
  AMBIGUOUS_ENTITY = "ambiguous_entity"
  NO_FILTERS = "no_filters"
  TOO_MANY_RESULTS = "too_many_results"
  UNRECOGNIZED_TOKENS = "unrecognized_tokens"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeRange:
  """Half-open time interval [start, end).  Either bound may be None."""

  start: float | None = None
  end: float | None = None

  def __post_init__(self) -> None:
    if (
        self.start is not None
        and self.end is not None
        and self.end < self.start
    ):
      raise ValueError("end must be >= start")

  def contains(self, timestamp: float) -> bool:
    if self.start is not None and timestamp < self.start:
      return False
    if self.end is not None and timestamp >= self.end:
      return False
    return True


@dataclass(frozen=True)
class SearchFilter:
  """Structured search filter extracted from a natural-language query.

  Every field is optional.  An absent field means "no constraint".
  """

  time_range: TimeRange | None = None
  junction_ids: tuple[str, ...] = ()
  camera_ids: tuple[str, ...] = ()
  event_types: tuple[str, ...] = ()
  violation_types: tuple[str, ...] = ()
  plate_text: str | None = None
  plate_partial: bool = False
  review_statuses: tuple[ReviewStatus, ...] = ()
  record_kinds: tuple[RecordKind, ...] = ()
  vehicle_types: tuple[str, ...] = ()
  text_terms: tuple[str, ...] = ()
  phase_states: tuple[str, ...] = ()
  max_results: int = 200

  def __post_init__(self) -> None:
    # Normalize tuples
    object.__setattr__(self, "junction_ids", tuple(self.junction_ids))
    object.__setattr__(self, "camera_ids", tuple(self.camera_ids))
    object.__setattr__(self, "event_types", tuple(self.event_types))
    object.__setattr__(self, "violation_types", tuple(self.violation_types))
    object.__setattr__(self, "review_statuses", tuple(self.review_statuses))
    object.__setattr__(self, "record_kinds", tuple(self.record_kinds))
    object.__setattr__(self, "vehicle_types", tuple(self.vehicle_types))
    object.__setattr__(self, "text_terms", tuple(self.text_terms))
    object.__setattr__(self, "phase_states", tuple(self.phase_states))
    if self.max_results < 1:
      raise ValueError("max_results must be >= 1")

  @property
  def filter_count(self) -> int:
    """Number of non-empty user-specified filter dimensions.

    ``record_kinds`` is excluded because it is auto-inferred.
    """
    count = 0
    if self.time_range is not None:
      count += 1
    if self.junction_ids:
      count += 1
    if self.camera_ids:
      count += 1
    if self.event_types:
      count += 1
    if self.violation_types:
      count += 1
    if self.plate_text:
      count += 1
    if self.review_statuses:
      count += 1
    if self.vehicle_types:
      count += 1
    if self.text_terms:
      count += 1
    if self.phase_states:
      count += 1
    return count


# ---------------------------------------------------------------------------
# Parse result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Safeguard:
  """One safeguard that fired during parsing or execution."""

  kind: SafeguardKind
  message: str


@dataclass(frozen=True)
class TokenExtraction:
  """One token from the query that was mapped to a filter dimension."""

  token: str
  dimension: str
  value: str
  confidence: ParseConfidence


@dataclass(frozen=True)
class ParsedQuery:
  """The result of parsing a natural-language search query.

  Carries the extracted structured filter, a human-readable explanation
  of how each portion of the query was interpreted, and any safeguards
  that were triggered.
  """

  raw_query: str
  normalized_query: str
  search_filter: SearchFilter
  extractions: tuple[TokenExtraction, ...] = ()
  unrecognized_tokens: tuple[str, ...] = ()
  confidence: ParseConfidence = ParseConfidence.LOW
  safeguards: tuple[Safeguard, ...] = ()
  explanation: str = ""

  def __post_init__(self) -> None:
    object.__setattr__(self, "extractions", tuple(self.extractions))
    object.__setattr__(
        self, "unrecognized_tokens", tuple(self.unrecognized_tokens)
    )
    object.__setattr__(self, "safeguards", tuple(self.safeguards))


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordReference:
  """Pointer to the specific platform record that matched."""

  record_kind: RecordKind
  record_id: str
  label: str
  detail: str
  provenance: ProvenanceChain | None = None


@dataclass(frozen=True)
class MatchEvidence:
  """One concrete reason a record matched a requested filter."""

  filter_name: str
  requested_value: str
  matched_value: str
  basis: str


@dataclass(frozen=True)
class SearchHit:
  """One search result with explicit references to underlying records."""

  reference: RecordReference
  matched_filters: tuple[str, ...] = ()
  matched_evidence: tuple[MatchEvidence, ...] = ()
  timestamp: float | None = None
  junction_id: str | None = None
  camera_id: str | None = None
  event_type: str | None = None
  violation_type: str | None = None
  plate_text: str | None = None
  review_status: ReviewStatus | None = None
  vehicle_type: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  provenance: ProvenanceChain | None = None

  def __post_init__(self) -> None:
    object.__setattr__(self, "matched_filters", tuple(self.matched_filters))
    object.__setattr__(self, "matched_evidence", tuple(self.matched_evidence))


@dataclass(frozen=True)
class SearchResult:
  """Typed output of a natural-language search execution."""

  query: ParsedQuery
  hits: tuple[SearchHit, ...] = ()
  total_candidates: int = 0
  truncated: bool = False
  safeguards: tuple[Safeguard, ...] = ()
  audit_log: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "hits", tuple(self.hits))
    object.__setattr__(self, "safeguards", tuple(self.safeguards))
    object.__setattr__(self, "audit_log", tuple(self.audit_log))
