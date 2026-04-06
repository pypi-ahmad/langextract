"""Deterministic natural-language query parser for TrafficMind search.

Translates operator queries into structured ``SearchFilter`` instances using
regex patterns and keyword tables.  The parser intentionally avoids LLM calls
so the translation is reproducible and auditable.

The parser is designed to be conservative: it only extracts filter dimensions
that it recognizes with reasonable confidence, and flags everything else as
unrecognized tokens so the caller can decide whether to proceed.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

from trafficmind.search.models import ParseConfidence
from trafficmind.search.models import ParsedQuery
from trafficmind.search.models import RecordKind
from trafficmind.search.models import ReviewStatus
from trafficmind.search.models import Safeguard
from trafficmind.search.models import SafeguardKind
from trafficmind.search.models import SearchFilter
from trafficmind.search.models import TimeRange
from trafficmind.search.models import TokenExtraction

# ---------------------------------------------------------------------------
# Time expression helpers
# ---------------------------------------------------------------------------

# Named relative windows mapped to (offset_seconds_before_now, offset_seconds_after_now | 0)
_RELATIVE_TIME_LABELS: dict[str, tuple[float, float]] = {
    "last 1 hour": (3600, 0),
    "last hour": (3600, 0),
    "past hour": (3600, 0),
    "last 2 hours": (7200, 0),
    "past 2 hours": (7200, 0),
    "last 4 hours": (14400, 0),
    "last 6 hours": (21600, 0),
    "last 8 hours": (28800, 0),
    "last 12 hours": (43200, 0),
    "last 24 hours": (86400, 0),
    "past 24 hours": (86400, 0),
    "last day": (86400, 0),
    "past day": (86400, 0),
    "last 48 hours": (172800, 0),
    "last 2 days": (172800, 0),
    "last 3 days": (259200, 0),
    "last 7 days": (604800, 0),
    "last week": (604800, 0),
    "past week": (604800, 0),
}

_RELATIVE_N_PATTERN = re.compile(
    r"\blast\s+(\d+)\s+(hours?|days?|minutes?|mins?)\b", re.IGNORECASE
)

# Named day periods relative to today midnight
# (start_hour, end_hour) in local-day hours since midnight
_DAY_PERIOD_LABELS: dict[str, tuple[int, int]] = {
    "this morning": (6, 12),
    "morning": (6, 12),
    "this afternoon": (12, 18),
    "afternoon": (12, 18),
    "this evening": (18, 22),
    "evening": (18, 22),
    "tonight": (18, 6),  # wraps to next day
    "last night": (-6, 6),  # previous day 18:00 to today 06:00
    "overnight": (-6, 6),
}


def _today_midnight(now: float) -> float:
  """Return Unix epoch for start of today (local time)."""
  import datetime

  dt = datetime.datetime.fromtimestamp(now)
  midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
  return midnight.timestamp()


def _parse_time_range(
    text: str, now: float
) -> tuple[TimeRange | None, list[TokenExtraction]]:
  """Extract a time range from the query text."""
  lower = text.lower()
  extractions: list[TokenExtraction] = []

  # Check named relative windows (longest match first)
  for label in sorted(_RELATIVE_TIME_LABELS, key=len, reverse=True):
    if label in lower:
      offset_before, _ = _RELATIVE_TIME_LABELS[label]
      tr = TimeRange(start=now - offset_before, end=now)
      extractions.append(
          TokenExtraction(
              token=label,
              dimension="time_range",
              value=f"last {offset_before / 3600:.0f}h",
              confidence=ParseConfidence.EXACT,
          )
      )
      return tr, extractions

  # Check "last N hours/days/minutes"
  match = _RELATIVE_N_PATTERN.search(lower)
  if match:
    n = int(match.group(1))
    unit = match.group(2).lower().rstrip("s")
    multiplier = {"hour": 3600, "day": 86400, "minute": 60, "min": 60}.get(
        unit, 3600
    )
    offset = n * multiplier
    tr = TimeRange(start=now - offset, end=now)
    extractions.append(
        TokenExtraction(
            token=match.group(0),
            dimension="time_range",
            value=f"last {n} {unit}(s)",
            confidence=ParseConfidence.EXACT,
        )
    )
    return tr, extractions

  # Check day-period labels (longest first)
  for label in sorted(_DAY_PERIOD_LABELS, key=len, reverse=True):
    if label in lower:
      start_hour, end_hour = _DAY_PERIOD_LABELS[label]
      midnight = _today_midnight(now)
      if start_hour < 0:
        start = midnight + start_hour * 3600
      else:
        start = midnight + start_hour * 3600
      if end_hour <= start_hour:
        end = midnight + (24 + end_hour) * 3600
      else:
        end = midnight + end_hour * 3600
      tr = TimeRange(start=start, end=end)
      extractions.append(
          TokenExtraction(
              token=label,
              dimension="time_range",
              value=f"day period {label}",
              confidence=ParseConfidence.HIGH,
          )
      )
      return tr, extractions

  # "yesterday"
  if "yesterday" in lower:
    midnight = _today_midnight(now)
    tr = TimeRange(start=midnight - 86400, end=midnight)
    extractions.append(
        TokenExtraction(
            token="yesterday",
            dimension="time_range",
            value="yesterday 00:00–24:00",
            confidence=ParseConfidence.EXACT,
        )
    )
    return tr, extractions

  # "today"
  if "today" in lower:
    midnight = _today_midnight(now)
    tr = TimeRange(start=midnight, end=now)
    extractions.append(
        TokenExtraction(
            token="today",
            dimension="time_range",
            value="today 00:00–now",
            confidence=ParseConfidence.EXACT,
        )
    )
    return tr, extractions

  return None, extractions


def _has_phrase(text: str, phrase: str) -> bool:
  """Return True when *phrase* appears as a whole phrase in *text*."""
  return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


# ---------------------------------------------------------------------------
# Junction / camera extraction
# ---------------------------------------------------------------------------

_JUNCTION_WORD_PATTERN = re.compile(
    r"\b(?:junction|junc|jct)\s*[-_]?([A-Za-z0-9][\w\-]*)\b",
    re.IGNORECASE,
)
_JUNCTION_CODE_PATTERN = re.compile(
    r"\bJ[-_\s]?(\d[\w\-]*)\b",
    re.IGNORECASE,
)
_CAMERA_WORD_PATTERN = re.compile(
    r"\b(?:camera|cam)\s*[-_]?([A-Za-z0-9][\w\-]*)\b",
    re.IGNORECASE,
)
_CAMERA_CODE_PATTERN = re.compile(
    r"\bCAM[-_\s]?(\d[\w\-]*)\b",
    re.IGNORECASE,
)


def _parse_junction_ids(
    text: str,
) -> tuple[tuple[str, ...], list[TokenExtraction]]:
  extractions: list[TokenExtraction] = []
  ids: list[str] = []
  for m in _JUNCTION_WORD_PATTERN.finditer(text):
    raw_id = m.group(1)
    # Normalize: "4" → "J-4", already prefixed → keep
    normalized = raw_id if not raw_id.isdigit() else f"J-{raw_id}"
    ids.append(normalized)
    extractions.append(
        TokenExtraction(
            token=m.group(0),
            dimension="junction_id",
            value=normalized,
            confidence=ParseConfidence.HIGH,
        )
    )
  for m in _JUNCTION_CODE_PATTERN.finditer(text):
    normalized = f"J-{m.group(1)}"
    ids.append(normalized)
    extractions.append(
        TokenExtraction(
            token=m.group(0),
            dimension="junction_id",
            value=normalized,
            confidence=ParseConfidence.HIGH,
        )
    )
  return tuple(dict.fromkeys(ids)), extractions


def _parse_camera_ids(
    text: str,
) -> tuple[tuple[str, ...], list[TokenExtraction]]:
  extractions: list[TokenExtraction] = []
  ids: list[str] = []
  for m in _CAMERA_WORD_PATTERN.finditer(text):
    raw_id = m.group(1)
    normalized = raw_id if not raw_id.isdigit() else f"CAM-{raw_id}"
    ids.append(normalized)
    extractions.append(
        TokenExtraction(
            token=m.group(0),
            dimension="camera_id",
            value=normalized,
            confidence=ParseConfidence.HIGH,
        )
    )
  for m in _CAMERA_CODE_PATTERN.finditer(text):
    normalized = f"CAM-{m.group(1)}"
    ids.append(normalized)
    extractions.append(
        TokenExtraction(
            token=m.group(0),
            dimension="camera_id",
            value=normalized,
            confidence=ParseConfidence.HIGH,
        )
    )
  return tuple(dict.fromkeys(ids)), extractions


# ---------------------------------------------------------------------------
# Event / violation type extraction
# ---------------------------------------------------------------------------

_EVENT_TYPE_MAP: dict[str, str] = {
    "red light": "red_light_violation",
    "red-light": "red_light_violation",
    "ran red": "red_light_violation",
    "running red": "red_light_violation",
    "signal violation": "signal_violation",
    "restricted zone": "restricted_zone_violation",
    "restricted area": "restricted_zone_violation",
    "no entry": "restricted_zone_violation",
    "stop violation": "stop_violation",
    "stop event": "stop_event",
    "speeding": "speeding_violation",
    "over speed": "speeding_violation",
    "overspeed": "speeding_violation",
    "illegal turn": "illegal_turn",
    "wrong way": "wrong_way_violation",
    "wrong-way": "wrong_way_violation",
    "amber running": "amber_running",
    "amber violation": "amber_running",
    "blocking": "intersection_blocking",
    "intersection blocking": "intersection_blocking",
    "u-turn": "illegal_uturn",
    "illegal u-turn": "illegal_uturn",
}

_VIOLATION_TYPE_MAP: dict[str, str] = {
    "red light": "red_light",
    "red-light": "red_light",
    "signal": "signal",
    "restricted zone": "restricted_zone",
    "restricted area": "restricted_zone",
    "stop": "stop",
    "speeding": "speeding",
    "speed": "speeding",
    "illegal turn": "illegal_turn",
    "wrong way": "wrong_way",
    "wrong-way": "wrong_way",
    "amber": "amber",
    "blocking": "blocking",
}


def _parse_event_types(
    text: str,
) -> tuple[tuple[str, ...], list[TokenExtraction]]:
  lower = text.lower()
  extractions: list[TokenExtraction] = []
  matched: list[str] = []
  for phrase in sorted(_EVENT_TYPE_MAP, key=len, reverse=True):
    if _has_phrase(lower, phrase):
      event_type = _EVENT_TYPE_MAP[phrase]
      if event_type not in matched:
        matched.append(event_type)
        extractions.append(
            TokenExtraction(
                token=phrase,
                dimension="event_type",
                value=event_type,
                confidence=ParseConfidence.HIGH,
            )
        )
  return tuple(matched), extractions


def _parse_violation_types(
    text: str,
) -> tuple[tuple[str, ...], list[TokenExtraction]]:
  lower = text.lower()
  extractions: list[TokenExtraction] = []
  matched: list[str] = []
  for phrase in sorted(_VIOLATION_TYPE_MAP, key=len, reverse=True):
    if _has_phrase(lower, phrase):
      vt = _VIOLATION_TYPE_MAP[phrase]
      if vt not in matched:
        matched.append(vt)
        extractions.append(
            TokenExtraction(
                token=phrase,
                dimension="violation_type",
                value=vt,
                confidence=ParseConfidence.HIGH,
            )
        )
  return tuple(matched), extractions


# ---------------------------------------------------------------------------
# Plate text extraction
# ---------------------------------------------------------------------------

_PLATE_PATTERN = re.compile(
    r"\b(?:plate|plates|anpr|registration|reg(?:istration)?)\s+(?:read(?:s|ing)?s?\s+)?(?:similar\s+to\s+|like\s+|matching\s+|resembling\s+)?([A-Z0-9][A-Z0-9\-*?]{1,14})\b",
    re.IGNORECASE,
)
_SIMILAR_PLATE_PATTERN = re.compile(
    r"\b(?:similar\s+to|like|matching|resembling)\s+([A-Z0-9][A-Z0-9\-*?]{1,14})\b",
    re.IGNORECASE,
)


def _has_plate_context(text: str) -> bool:
  lower = text.lower()
  return any(
      _has_phrase(lower, cue)
      for cue in (
          "plate",
          "plates",
          "plate read",
          "anpr",
          "registration",
          "reg",
      )
  )


def _parse_plate_text(
    text: str,
) -> tuple[str | None, bool, list[TokenExtraction]]:
  extractions: list[TokenExtraction] = []
  lower = text.lower()

  m = _PLATE_PATTERN.search(text)
  if m:
    plate = m.group(1).upper()
    partial = (
        bool(_SIMILAR_PLATE_PATTERN.search(text))
        or "*" in plate
        or "?" in plate
    )
    extractions.append(
        TokenExtraction(
            token=m.group(0),
            dimension="plate_text",
            value=plate,
            confidence=ParseConfidence.HIGH
            if not partial
            else ParseConfidence.PARTIAL,
        )
    )
    return plate, partial, extractions

  # Fallback: similarity phrases are accepted only when the query is clearly
  # about ANPR / plates. This avoids interpreting location phrases such as
  # "near Junction 4" as a plate query.
  m2 = (
      _SIMILAR_PLATE_PATTERN.search(text) if _has_plate_context(lower) else None
  )
  if m2:
    plate = m2.group(1).upper()
    extractions.append(
        TokenExtraction(
            token=m2.group(0),
            dimension="plate_text",
            value=plate,
            confidence=ParseConfidence.PARTIAL,
        )
    )
    return plate, True, extractions

  return None, False, extractions


# ---------------------------------------------------------------------------
# Review status extraction
# ---------------------------------------------------------------------------

_REVIEW_STATUS_MAP: dict[str, ReviewStatus] = {
    "pending": ReviewStatus.PENDING,
    "pending review": ReviewStatus.PENDING,
    "awaiting review": ReviewStatus.PENDING,
    "in review": ReviewStatus.IN_REVIEW,
    "under review": ReviewStatus.IN_REVIEW,
    "being reviewed": ReviewStatus.IN_REVIEW,
    "confirmed": ReviewStatus.CONFIRMED,
    "dismissed": ReviewStatus.DISMISSED,
    "rejected": ReviewStatus.DISMISSED,
    "escalated": ReviewStatus.ESCALATED,
}


def _parse_review_statuses(
    text: str,
) -> tuple[tuple[ReviewStatus, ...], list[TokenExtraction]]:
  lower = text.lower()
  extractions: list[TokenExtraction] = []
  matched: list[ReviewStatus] = []
  for phrase in sorted(_REVIEW_STATUS_MAP, key=len, reverse=True):
    if _has_phrase(lower, phrase):
      status = _REVIEW_STATUS_MAP[phrase]
      if status not in matched:
        matched.append(status)
        extractions.append(
            TokenExtraction(
                token=phrase,
                dimension="review_status",
                value=status.value,
                confidence=ParseConfidence.HIGH,
            )
        )
  return tuple(matched), extractions


# ---------------------------------------------------------------------------
# Vehicle type extraction
# ---------------------------------------------------------------------------

_VEHICLE_TYPES: dict[str, str] = {
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "hgv": "truck",
    "car": "car",
    "cars": "car",
    "van": "van",
    "vans": "van",
    "bus": "bus",
    "buses": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
    "taxi": "taxi",
    "ambulance": "emergency",
    "fire engine": "emergency",
    "emergency": "emergency",
}


def _parse_vehicle_types(
    text: str,
) -> tuple[tuple[str, ...], list[TokenExtraction]]:
  lower = text.lower()
  extractions: list[TokenExtraction] = []
  matched: list[str] = []
  for phrase in sorted(_VEHICLE_TYPES, key=len, reverse=True):
    if _has_phrase(lower, phrase):
      vt = _VEHICLE_TYPES[phrase]
      if vt not in matched:
        matched.append(vt)
        extractions.append(
            TokenExtraction(
                token=phrase,
                dimension="vehicle_type",
                value=vt,
                confidence=ParseConfidence.HIGH,
            )
        )
  return tuple(matched), extractions


_TEXT_TERM_MAP: dict[str, str] = {
    "stopped": "stopped",
    "parked": "parked",
    "queued": "queued",
    "queueing": "queued",
    "queuing": "queued",
    "lingering": "lingering",
    "loitering": "lingering",
}


def _parse_text_terms(
    text: str,
) -> tuple[tuple[str, ...], list[TokenExtraction]]:
  lower = text.lower()
  extractions: list[TokenExtraction] = []
  matched: list[str] = []
  for phrase in sorted(_TEXT_TERM_MAP, key=len, reverse=True):
    if _has_phrase(lower, phrase):
      term = _TEXT_TERM_MAP[phrase]
      if term not in matched:
        matched.append(term)
        extractions.append(
            TokenExtraction(
                token=phrase,
                dimension="text_term",
                value=term,
                confidence=ParseConfidence.HIGH,
            )
        )
  return tuple(matched), extractions


# ---------------------------------------------------------------------------
# Record kind inference
# ---------------------------------------------------------------------------


def _infer_record_kinds(
    event_types: tuple[str, ...],
    violation_types: tuple[str, ...],
    plate_text: str | None,
) -> tuple[RecordKind, ...]:
  """Infer which record kinds the query targets."""
  kinds: list[RecordKind] = []
  if plate_text:
    kinds.append(RecordKind.PLATE_READ)
  if violation_types:
    kinds.append(RecordKind.VIOLATION)
  if event_types:
    for et in event_types:
      if "violation" in et and RecordKind.VIOLATION not in kinds:
        kinds.append(RecordKind.VIOLATION)
      elif RecordKind.INCIDENT not in kinds:
        kinds.append(RecordKind.INCIDENT)
  if not kinds:
    kinds.extend(
        [RecordKind.INCIDENT, RecordKind.VIOLATION, RecordKind.PLATE_READ]
    )
  return tuple(dict.fromkeys(kinds))


# ---------------------------------------------------------------------------
# Unrecognized token detection
# ---------------------------------------------------------------------------

# Words that are structural / stop-words and should not be flagged
_STOP_WORDS = frozenset(
    "show find get list search all the a an in at on for from to of by "
    "with near around about is are was were do does did have has had "
    "and or not no this that these those my me i we us they them "
    "any some very most quite only just also please can could will would "
    "should shall may might let "
    "violation violations event events incident incidents reads reading "
    "entries entry records record captures capture plate plates anpr "
    "registration reg".split()
)


def _find_unrecognized_tokens(
    text: str,
    extractions: Sequence[TokenExtraction],
) -> tuple[str, ...]:
  """Find words in the query that were not mapped to any filter dimension."""
  remaining = text.lower()
  for ext in extractions:
    remaining = remaining.replace(ext.token.lower(), " ")
  words = re.findall(r"\b[a-z]{2,}\b", remaining)
  return tuple(w for w in words if w not in _STOP_WORDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_query(
    raw_query: str,
    *,
    now: float | None = None,
    max_results: int = 200,
) -> ParsedQuery:
  """Parse a natural-language search query into a structured filter.

  Parameters
  ----------
  raw_query:
      The operator's natural-language search string.
  now:
      Reference timestamp (Unix epoch). Defaults to current time.
  max_results:
      Upper bound on results to request.

  Returns
  -------
  ParsedQuery with the extracted filter, token-level explanations,
  any safeguards that triggered, and an overall confidence level.
  """
  if now is None:
    now = time.time()

  normalized = " ".join(raw_query.split())
  all_extractions: list[TokenExtraction] = []

  time_range, time_ext = _parse_time_range(normalized, now)
  all_extractions.extend(time_ext)

  junction_ids, junc_ext = _parse_junction_ids(normalized)
  all_extractions.extend(junc_ext)

  camera_ids, cam_ext = _parse_camera_ids(normalized)
  all_extractions.extend(cam_ext)

  event_types, event_ext = _parse_event_types(normalized)
  all_extractions.extend(event_ext)

  violation_types, viol_ext = _parse_violation_types(normalized)
  all_extractions.extend(viol_ext)

  plate_text, plate_partial, plate_ext = _parse_plate_text(normalized)
  all_extractions.extend(plate_ext)

  review_statuses, review_ext = _parse_review_statuses(normalized)
  all_extractions.extend(review_ext)

  vehicle_types, vehicle_ext = _parse_vehicle_types(normalized)
  all_extractions.extend(vehicle_ext)

  text_terms, text_term_ext = _parse_text_terms(normalized)
  all_extractions.extend(text_term_ext)

  record_kinds = _infer_record_kinds(event_types, violation_types, plate_text)

  search_filter = SearchFilter(
      time_range=time_range,
      junction_ids=junction_ids,
      camera_ids=camera_ids,
      event_types=event_types,
      violation_types=violation_types,
      plate_text=plate_text,
      plate_partial=plate_partial,
      review_statuses=review_statuses,
      record_kinds=record_kinds,
      vehicle_types=vehicle_types,
      text_terms=text_terms,
      max_results=max_results,
  )

  unrecognized = _find_unrecognized_tokens(normalized, all_extractions)
  safeguards = _check_safeguards(search_filter, unrecognized)
  confidence = _assess_confidence(search_filter, unrecognized)
  explanation = _build_explanation(search_filter, all_extractions, unrecognized)

  return ParsedQuery(
      raw_query=raw_query,
      normalized_query=normalized,
      search_filter=search_filter,
      extractions=tuple(all_extractions),
      unrecognized_tokens=unrecognized,
      confidence=confidence,
      safeguards=tuple(safeguards),
      explanation=explanation,
  )


# ---------------------------------------------------------------------------
# Safeguards
# ---------------------------------------------------------------------------


def _check_safeguards(
    search_filter: SearchFilter,
    unrecognized: tuple[str, ...],
) -> list[Safeguard]:
  safeguards: list[Safeguard] = []

  if search_filter.filter_count == 0:
    safeguards.append(
        Safeguard(
            kind=SafeguardKind.NO_FILTERS,
            message=(
                "No filter dimensions were extracted from the query. Results"
                " may be empty or overly broad."
            ),
        )
    )

  if search_filter.time_range is None:
    safeguards.append(
        Safeguard(
            kind=SafeguardKind.OVERLY_BROAD,
            message=(
                "No time range was extracted from the query. "
                "Consider adding a time constraint to narrow results."
            ),
        )
    )

  if (
      len(search_filter.event_types) > 1
      or len(search_filter.violation_types) > 1
  ):
    ambiguous_dimensions: list[str] = []
    if len(search_filter.event_types) > 1:
      ambiguous_dimensions.append("event type")
    if len(search_filter.violation_types) > 1:
      ambiguous_dimensions.append("violation type")
    safeguards.append(
        Safeguard(
            kind=SafeguardKind.AMBIGUOUS_ENTITY,
            message=(
                "Multiple values were extracted for "
                + " and ".join(ambiguous_dimensions)
                + ". Search uses OR semantics within a single filter dimension."
            ),
        )
    )

  if unrecognized:
    safeguards.append(
        Safeguard(
            kind=SafeguardKind.UNRECOGNIZED_TOKENS,
            message=(
                f"Unrecognized query tokens: {', '.join(unrecognized)}. These"
                " were ignored during parsing."
            ),
        )
    )

  return safeguards


def _assess_confidence(
    search_filter: SearchFilter,
    unrecognized: tuple[str, ...],
) -> ParseConfidence:
  fc = search_filter.filter_count
  if fc == 0:
    return ParseConfidence.LOW
  if (
      len(search_filter.event_types) > 1
      or len(search_filter.violation_types) > 1
  ):
    return ParseConfidence.PARTIAL
  if unrecognized and len(unrecognized) >= fc:
    return ParseConfidence.LOW
  if unrecognized:
    return ParseConfidence.PARTIAL
  if fc >= 2:
    return ParseConfidence.EXACT
  return ParseConfidence.HIGH


def _build_explanation(
    search_filter: SearchFilter,
    extractions: list[TokenExtraction],
    unrecognized: tuple[str, ...],
) -> str:
  parts: list[str] = []
  parts.append(
      f"Extracted {len(extractions)} filter dimension(s) from the query."
  )

  for ext in extractions:
    parts.append(
        f'  "{ext.token}" → {ext.dimension}={ext.value} (confidence:'
        f" {ext.confidence.value})"
    )

  if unrecognized:
    parts.append(f"Unrecognized tokens (ignored): {', '.join(unrecognized)}")

  constraints: list[str] = []
  if search_filter.time_range:
    constraints.append("time range")
  if search_filter.junction_ids:
    constraints.append(f"junction(s): {', '.join(search_filter.junction_ids)}")
  if search_filter.camera_ids:
    constraints.append(f"camera(s): {', '.join(search_filter.camera_ids)}")
  if search_filter.event_types:
    constraints.append(f"event type(s): {', '.join(search_filter.event_types)}")
  if search_filter.violation_types:
    constraints.append(
        f"violation type(s): {', '.join(search_filter.violation_types)}"
    )
  if search_filter.plate_text:
    mode = "partial" if search_filter.plate_partial else "exact"
    constraints.append(f"plate text: {search_filter.plate_text} ({mode})")
  if search_filter.review_statuses:
    constraints.append(
        "review status(es):"
        f" {', '.join(s.value for s in search_filter.review_statuses)}"
    )
  if search_filter.vehicle_types:
    constraints.append(
        f"vehicle type(s): {', '.join(search_filter.vehicle_types)}"
    )
  if search_filter.text_terms:
    constraints.append(f"text term(s): {', '.join(search_filter.text_terms)}")

  if constraints:
    parts.append("Active filters: " + "; ".join(constraints))
  else:
    parts.append("No active filters extracted.")

  if (
      len(search_filter.event_types) > 1
      or len(search_filter.violation_types) > 1
  ):
    parts.append(
        "Multiple values in the same filter dimension use OR semantics; narrow"
        " the query if you intended a single event or violation class."
    )

  return "\n".join(parts)
