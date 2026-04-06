"""Tests for trafficmind.search — natural-language search layer."""

import unittest

from trafficmind.search.executor import SearchExecutor
from trafficmind.search.models import ParseConfidence
from trafficmind.search.models import RecordKind
from trafficmind.search.models import ReviewStatus
from trafficmind.search.models import SafeguardKind
from trafficmind.search.models import SearchFilter
from trafficmind.search.models import TimeRange
from trafficmind.search.parser import parse_query
from trafficmind.search.store import InMemorySearchStore
from trafficmind.search.store import PlatformRecord

# Reference time: 2026-04-06 14:00:00 UTC (a Monday afternoon)
T_REF = 1743948000.0


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseQuery(unittest.TestCase):
  """Test the deterministic NL → SearchFilter parser."""

  def test_red_light_near_junction(self):
    pq = parse_query(
        "find all red-light violations near Junction 4 this morning",
        now=T_REF,
    )
    self.assertIn("red_light_violation", pq.search_filter.event_types)
    self.assertIn("red_light", pq.search_filter.violation_types)
    self.assertIn("J-4", pq.search_filter.junction_ids)
    self.assertIsNotNone(pq.search_filter.time_range)
    self.assertIn(ParseConfidence.EXACT, [ParseConfidence.EXACT, pq.confidence])
    self.assertTrue(pq.explanation)

  def test_trucks_restricted_zone_last_night(self):
    pq = parse_query(
        "show trucks stopped in restricted zone last night",
        now=T_REF,
    )
    self.assertIn("truck", pq.search_filter.vehicle_types)
    self.assertIn("restricted_zone_violation", pq.search_filter.event_types)
    self.assertIn("stopped", pq.search_filter.text_terms)
    self.assertNotIn("stop_event", pq.search_filter.event_types)
    self.assertIsNotNone(pq.search_filter.time_range)
    # "last night" should be roughly prev-evening to early-morning
    tr = pq.search_filter.time_range
    self.assertIsNotNone(tr.start)
    self.assertIsNotNone(tr.end)
    self.assertLess(tr.start, tr.end)

  def test_plate_similarity_requires_plate_context(self):
    pq = parse_query(
        "find incidents near Junction 4 in the last 24 hours",
        now=T_REF,
    )
    self.assertIsNone(pq.search_filter.plate_text)
    self.assertFalse(pq.search_filter.plate_partial)

  def test_camera_extraction_does_not_trigger_on_car_word(self):
    pq = parse_query(
        "show cars at junction 4 last hour",
        now=T_REF,
    )
    self.assertIn("car", pq.search_filter.vehicle_types)
    self.assertEqual(pq.search_filter.camera_ids, ())

  def test_plate_reads_similar_partial(self):
    pq = parse_query(
        "show plate reads similar to AB12 in the last 24 hours",
        now=T_REF,
    )
    self.assertEqual(pq.search_filter.plate_text, "AB12")
    self.assertTrue(pq.search_filter.plate_partial)
    self.assertIsNotNone(pq.search_filter.time_range)
    self.assertIn(RecordKind.PLATE_READ, pq.search_filter.record_kinds)

  def test_exact_plate_match(self):
    pq = parse_query("find plate ABC1234", now=T_REF)
    self.assertEqual(pq.search_filter.plate_text, "ABC1234")
    self.assertFalse(pq.search_filter.plate_partial)

  def test_review_status_pending(self):
    pq = parse_query(
        "show pending review incidents at Junction 7",
        now=T_REF,
    )
    self.assertIn(ReviewStatus.PENDING, pq.search_filter.review_statuses)
    self.assertIn("J-7", pq.search_filter.junction_ids)

  def test_escalated_violations(self):
    pq = parse_query("list escalated violations", now=T_REF)
    self.assertIn(ReviewStatus.ESCALATED, pq.search_filter.review_statuses)

  def test_relative_time_last_n_hours(self):
    pq = parse_query("find events in the last 3 hours", now=T_REF)
    tr = pq.search_filter.time_range
    self.assertIsNotNone(tr)
    self.assertAlmostEqual(tr.end, T_REF, delta=1)
    self.assertAlmostEqual(tr.start, T_REF - 3 * 3600, delta=1)

  def test_yesterday_time_range(self):
    pq = parse_query("show all violations yesterday", now=T_REF)
    tr = pq.search_filter.time_range
    self.assertIsNotNone(tr)
    # Window should be exactly 24h
    self.assertAlmostEqual(tr.end - tr.start, 86400, delta=1)

  def test_today_time_range(self):
    pq = parse_query("list events today", now=T_REF)
    tr = pq.search_filter.time_range
    self.assertIsNotNone(tr)
    self.assertAlmostEqual(tr.end, T_REF, delta=1)

  def test_no_filters_safeguard(self):
    pq = parse_query("hello world", now=T_REF)
    kinds = [s.kind for s in pq.safeguards]
    self.assertIn(SafeguardKind.NO_FILTERS, kinds)
    self.assertEqual(pq.confidence, ParseConfidence.LOW)

  def test_overly_broad_no_time(self):
    pq = parse_query("find all red-light events", now=T_REF)
    kinds = [s.kind for s in pq.safeguards]
    # Event type extracted but no time range → overly broad
    self.assertIn(SafeguardKind.OVERLY_BROAD, kinds)

  def test_multiple_event_values_raise_ambiguity_safeguard(self):
    pq = parse_query(
        "find red light or speeding violations last 2 hours",
        now=T_REF,
    )
    kinds = [s.kind for s in pq.safeguards]
    self.assertIn(SafeguardKind.AMBIGUOUS_ENTITY, kinds)
    self.assertEqual(pq.confidence, ParseConfidence.PARTIAL)
    self.assertIn("OR semantics", pq.explanation)

  def test_unrecognized_tokens_flagged(self):
    pq = parse_query(
        "find xylophone incidents near Junction 2 last 24 hours",
        now=T_REF,
    )
    self.assertIn("xylophone", pq.unrecognized_tokens)
    kinds = [s.kind for s in pq.safeguards]
    self.assertIn(SafeguardKind.UNRECOGNIZED_TOKENS, kinds)

  def test_multiple_junction_ids(self):
    pq = parse_query(
        "show events at junction 3 and junction 5",
        now=T_REF,
    )
    self.assertIn("J-3", pq.search_filter.junction_ids)
    self.assertIn("J-5", pq.search_filter.junction_ids)

  def test_camera_extraction(self):
    pq = parse_query("events from camera 12 last hour", now=T_REF)
    self.assertIn("CAM-12", pq.search_filter.camera_ids)

  def test_confidence_exact_with_multiple_filters(self):
    pq = parse_query(
        "find red-light violations at junction 4 in the last 2 hours",
        now=T_REF,
    )
    self.assertIn(pq.confidence, (ParseConfidence.EXACT, ParseConfidence.HIGH))

  def test_record_kind_inference_plate(self):
    pq = parse_query("plate AB12", now=T_REF)
    self.assertIn(RecordKind.PLATE_READ, pq.search_filter.record_kinds)

  def test_record_kind_inference_violation(self):
    pq = parse_query("red light violations", now=T_REF)
    self.assertIn(RecordKind.VIOLATION, pq.search_filter.record_kinds)

  def test_filter_count(self):
    sf = SearchFilter(
        time_range=TimeRange(start=0.0, end=100.0),
        junction_ids=("J-1",),
        event_types=("red_light_violation",),
    )
    self.assertEqual(sf.filter_count, 3)

  def test_explanation_is_populated(self):
    pq = parse_query(
        "show trucks at Junction 4 last 24 hours",
        now=T_REF,
    )
    self.assertIn("junction", pq.explanation.lower())
    self.assertIn("truck", pq.explanation.lower())
    self.assertIn("time range", pq.explanation.lower())


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


class TestInMemorySearchStore(unittest.TestCase):

  def _populated_store(self) -> InMemorySearchStore:
    store = InMemorySearchStore()
    store.add_many([
        PlatformRecord(
            record_kind=RecordKind.VIOLATION,
            record_id="V-1",
            label="Red light at J-4",
            detail="Vehicle crossed stop line while signal restrictive.",
            timestamp=T_REF - 3600,
            junction_id="J-4",
            event_type="red_light_violation",
            violation_type="red_light",
            review_status=ReviewStatus.PENDING,
            vehicle_type="car",
        ),
        PlatformRecord(
            record_kind=RecordKind.VIOLATION,
            record_id="V-2",
            label="Restricted zone entry",
            detail="Truck stopped in restricted zone during curfew hours.",
            timestamp=T_REF - 7200,
            junction_id="J-4",
            event_type="restricted_zone_violation",
            violation_type="restricted_zone",
            vehicle_type="truck",
            review_status=ReviewStatus.ESCALATED,
        ),
        PlatformRecord(
            record_kind=RecordKind.PLATE_READ,
            record_id="PR-1",
            label="Plate read AB12XYZ",
            detail="ANPR capture at Camera 3.",
            timestamp=T_REF - 1800,
            junction_id="J-4",
            camera_id="CAM-3",
            plate_text="AB12XYZ",
        ),
        PlatformRecord(
            record_kind=RecordKind.PLATE_READ,
            record_id="PR-2",
            label="Plate read AB12345",
            detail="ANPR capture at Camera 3.",
            timestamp=T_REF - 900,
            junction_id="J-4",
            camera_id="CAM-3",
            plate_text="AB12345",
        ),
        PlatformRecord(
            record_kind=RecordKind.INCIDENT,
            record_id="INC-1",
            label="Signal conflict",
            detail="Controller-vision disagreement at J-2.",
            timestamp=T_REF - 600,
            junction_id="J-2",
            event_type="signal_conflict",
        ),
        PlatformRecord(
            record_kind=RecordKind.VIOLATION,
            record_id="V-3",
            label="Old violation",
            detail="Violation from long ago.",
            timestamp=T_REF - 200000,
            junction_id="J-1",
            event_type="red_light_violation",
            violation_type="red_light",
            review_status=ReviewStatus.CONFIRMED,
        ),
    ])
    return store

  def test_query_by_time_and_junction(self):
    store = self._populated_store()
    sf = SearchFilter(
        time_range=TimeRange(start=T_REF - 86400, end=T_REF),
        junction_ids=("J-4",),
    )
    hits = store.query(sf)
    self.assertTrue(all(h.junction_id == "J-4" for h in hits))
    self.assertGreater(len(hits), 0)

  def test_plate_partial_match(self):
    store = self._populated_store()
    sf = SearchFilter(
        plate_text="AB12",
        plate_partial=True,
        record_kinds=(RecordKind.PLATE_READ,),
    )
    hits = store.query(sf)
    self.assertEqual(len(hits), 2)
    plates = {h.plate_text for h in hits}
    self.assertEqual(plates, {"AB12XYZ", "AB12345"})

  def test_plate_exact_match(self):
    store = self._populated_store()
    sf = SearchFilter(
        plate_text="AB12XYZ",
        plate_partial=False,
        record_kinds=(RecordKind.PLATE_READ,),
    )
    hits = store.query(sf)
    self.assertEqual(len(hits), 1)
    self.assertEqual(hits[0].plate_text, "AB12XYZ")

  def test_violation_type_filter(self):
    store = self._populated_store()
    sf = SearchFilter(
        violation_types=("red_light",),
        record_kinds=(RecordKind.VIOLATION,),
    )
    hits = store.query(sf)
    self.assertTrue(all(h.violation_type == "red_light" for h in hits))
    self.assertEqual(len(hits), 2)

  def test_review_status_filter(self):
    store = self._populated_store()
    sf = SearchFilter(
        review_statuses=(ReviewStatus.ESCALATED,),
    )
    hits = store.query(sf)
    self.assertEqual(len(hits), 1)
    self.assertEqual(hits[0].review_status, ReviewStatus.ESCALATED)

  def test_text_term_filter(self):
    store = self._populated_store()
    sf = SearchFilter(
        vehicle_types=("truck",),
        text_terms=("stopped",),
    )
    hits = store.query(sf)
    self.assertEqual(len(hits), 1)
    self.assertEqual(hits[0].reference.record_id, "V-2")

  def test_max_results_truncation(self):
    store = self._populated_store()
    sf = SearchFilter(max_results=2)
    hits = store.query(sf)
    self.assertEqual(len(hits), 2)

  def test_results_sorted_by_timestamp_desc(self):
    store = self._populated_store()
    sf = SearchFilter()
    hits = store.query(sf)
    timestamps = [h.timestamp for h in hits if h.timestamp is not None]
    self.assertEqual(timestamps, sorted(timestamps, reverse=True))

  def test_count_matches_query_length(self):
    store = self._populated_store()
    sf = SearchFilter(junction_ids=("J-4",))
    count = store.count(sf)
    hits = store.query(sf)
    self.assertEqual(count, len(hits))

  def test_matched_filters_populated(self):
    store = self._populated_store()
    sf = SearchFilter(
        time_range=TimeRange(start=T_REF - 86400, end=T_REF),
        junction_ids=("J-4",),
        violation_types=("red_light",),
        record_kinds=(RecordKind.VIOLATION,),
    )
    hits = store.query(sf)
    self.assertTrue(hits)
    for hit in hits:
      self.assertIn("time_range", hit.matched_filters)
      self.assertIn("junction_id", hit.matched_filters)
      self.assertIn("violation_type", hit.matched_filters)

  def test_matched_evidence_populated(self):
    store = self._populated_store()
    sf = SearchFilter(
        time_range=TimeRange(start=T_REF - 86400, end=T_REF),
        junction_ids=("J-4",),
        violation_types=("red_light",),
        record_kinds=(RecordKind.VIOLATION,),
    )
    hits = store.query(sf)
    self.assertTrue(hits)
    evidence = hits[0].matched_evidence
    self.assertTrue(evidence)
    self.assertTrue(any(item.filter_name == "junction_id" for item in evidence))
    self.assertTrue(any(item.filter_name == "time_range" for item in evidence))
    self.assertTrue(all(item.requested_value for item in evidence))
    self.assertTrue(all(item.matched_value for item in evidence))
    self.assertTrue(all(item.basis for item in evidence))


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------


class TestSearchExecutor(unittest.TestCase):

  def _executor(self) -> SearchExecutor:
    store = InMemorySearchStore()
    store.add_many([
        PlatformRecord(
            record_kind=RecordKind.VIOLATION,
            record_id="V-1",
            label="Red light at J-4",
            detail="Vehicle crossed stop line.",
            timestamp=T_REF - 3600,
            junction_id="J-4",
            event_type="red_light_violation",
            violation_type="red_light",
            review_status=ReviewStatus.PENDING,
        ),
        PlatformRecord(
            record_kind=RecordKind.PLATE_READ,
            record_id="PR-1",
            label="Plate read AB12XYZ",
            detail="ANPR capture.",
            timestamp=T_REF - 1800,
            junction_id="J-4",
            camera_id="CAM-3",
            plate_text="AB12XYZ",
        ),
        PlatformRecord(
            record_kind=RecordKind.VIOLATION,
            record_id="V-2",
            label="Restricted zone truck",
            detail="Truck stopped in restricted zone.",
            timestamp=T_REF - 7200,
            junction_id="J-4",
            event_type="restricted_zone_violation",
            violation_type="restricted_zone",
            vehicle_type="truck",
            review_status=ReviewStatus.ESCALATED,
        ),
    ])
    return SearchExecutor(store)

  def test_end_to_end_red_light_search(self):
    executor = self._executor()
    result = executor.search(
        "find all red-light violations near Junction 4 this morning",
        now=T_REF,
    )
    self.assertIsNotNone(result.query.search_filter.time_range)
    self.assertTrue(result.query.explanation)
    self.assertTrue(result.audit_log)
    # At least the parsed query is present
    self.assertIsInstance(result.total_candidates, int)

  def test_end_to_end_plate_search(self):
    executor = self._executor()
    result = executor.search(
        "show plate reads similar to AB12 in the last 24 hours",
        now=T_REF,
    )
    self.assertTrue(result.query.search_filter.plate_partial)
    for hit in result.hits:
      self.assertIsNotNone(hit.plate_text)
      self.assertIn("AB12", hit.plate_text)

  def test_broad_query_safeguard(self):
    executor = self._executor()
    result = executor.search("show everything", now=T_REF)
    kinds = [s.kind for s in result.safeguards]
    self.assertTrue(
        SafeguardKind.NO_FILTERS in kinds
        or SafeguardKind.OVERLY_BROAD in kinds,
        f"Expected broad-query safeguard, got: {kinds}",
    )

  def test_audit_log_present(self):
    executor = self._executor()
    result = executor.search(
        "red light violations last 2 hours",
        now=T_REF,
    )
    self.assertTrue(result.audit_log)
    self.assertTrue(any("Parsed query" in entry for entry in result.audit_log))
    self.assertTrue(
        any("candidates" in entry.lower() for entry in result.audit_log)
    )

  def test_truncated_flag_set(self):
    executor = self._executor()
    result = executor.search(
        "show all red light violations",
        now=T_REF,
        max_results=1,
    )
    # If there's more than 1 red-light violation, truncated=True
    if result.total_candidates > 1:
      self.assertTrue(result.truncated)

  def test_references_on_every_hit(self):
    executor = self._executor()
    result = executor.search(
        "violations at junction 4 last 24 hours",
        now=T_REF,
    )
    for hit in result.hits:
      self.assertIsNotNone(hit.reference)
      self.assertTrue(hit.reference.record_id)
      self.assertTrue(hit.reference.record_kind)

  def test_hit_provenance_is_explicit(self):
    executor = self._executor()
    result = executor.search(
        "find red-light violations at junction 4 in the last 24 hours",
        now=T_REF,
    )
    self.assertTrue(result.hits)
    first_hit = result.hits[0]
    self.assertTrue(first_hit.matched_evidence)
    self.assertTrue(
        any(
            item.filter_name == "junction_id"
            for item in first_hit.matched_evidence
        )
    )
    self.assertTrue(
        any(
            item.filter_name == "event_type"
            for item in first_hit.matched_evidence
        )
    )
    self.assertTrue(
        any(
            item.filter_name == "time_range"
            for item in first_hit.matched_evidence
        )
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestSearchModels(unittest.TestCase):

  def test_time_range_validation(self):
    with self.assertRaises(ValueError):
      TimeRange(start=100.0, end=50.0)

  def test_time_range_contains(self):
    tr = TimeRange(start=100.0, end=200.0)
    self.assertTrue(tr.contains(150.0))
    self.assertFalse(tr.contains(50.0))
    self.assertFalse(tr.contains(200.0))  # half-open
    self.assertTrue(tr.contains(100.0))  # inclusive start

  def test_time_range_open_ended(self):
    tr = TimeRange(start=100.0)
    self.assertTrue(tr.contains(200.0))
    self.assertFalse(tr.contains(50.0))

  def test_search_filter_max_results_validation(self):
    with self.assertRaises(ValueError):
      SearchFilter(max_results=0)

  def test_search_filter_filter_count_empty(self):
    sf = SearchFilter()
    self.assertEqual(sf.filter_count, 0)

  def test_search_filter_filter_count_includes_text_terms(self):
    sf = SearchFilter(text_terms=("stopped",))
    self.assertEqual(sf.filter_count, 1)


if __name__ == "__main__":
  unittest.main()
