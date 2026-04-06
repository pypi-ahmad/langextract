# TrafficMind Natural-Language Search

## Purpose

`trafficmind.search` provides a deterministic query translation layer that
lets operators search platform data using natural-language requests such as:

- "show trucks stopped in restricted zone last night"
- "find all red-light violations near Junction 4 this morning"
- "show plate reads similar to AB12 in the last 24 hours"

The layer translates each request into a typed `SearchFilter`, executes it
against a `SearchableStore`, and returns results with explicit references to
the underlying records.

## Design Principles

- **Deterministic parsing**: No LLM calls. All query translation is
  keyword- and regex-based so the same query always produces the same filter.
- **Grounded results**: Every `SearchHit` carries a `RecordReference` linking
  back to the exact platform record (incident, violation, plate read, etc.).
- **Concrete provenance**: Every `SearchHit.matched_evidence` entry states the
  requested filter, the matched record value, and the exact basis for that
  match.
- **Explainability**: `ParsedQuery.explanation` and `ParsedQuery.extractions`
  show exactly how each portion of the query was interpreted.
- **Safeguards**: Overly broad or ambiguous queries are flagged before
  execution, not silently widened.
- **Extensible store protocol**: Production deployments implement
  `SearchableStore` against a real database or search index.
  `InMemorySearchStore` is provided for testing.

## Supported Query Patterns

### Time expressions

| Pattern | Example | Resolution |
|---------|---------|------------|
| `last N hours/days/minutes` | "last 3 hours" | [now - 3h, now) |
| Named relative | "last 24 hours", "last week" | Fixed offset from now |
| Day period | "this morning", "last night", "this afternoon" | Clock-hour window |
| `yesterday` | "yesterday" | Previous calendar day |
| `today` | "today" | Midnight to now |

### Location filters

| Pattern | Example | Extracted as |
|---------|---------|-------------|
| Junction | "Junction 4", "J-4", "jct 12" | `junction_ids=("J-4",)` |
| Camera | "Camera 3", "cam 12" | `camera_ids=("CAM-3",)` |

### Event / violation types

| Phrase | Mapped event type |
|--------|------------------|
| "red light", "red-light", "ran red" | `red_light_violation` |
| "restricted zone", "restricted area", "no entry" | `restricted_zone_violation` |
| "speeding", "over speed" | `speeding_violation` |
| "stop event" | `stop_event` |
| "wrong way", "wrong-way" | `wrong_way_violation` |
| "illegal turn" | `illegal_turn` |
| "blocking", "intersection blocking" | `intersection_blocking` |
| "amber running" | `amber_running` |

### Text terms

Some operator phrases are treated as grounded text constraints instead of
inventing a structured event type. These terms must match record labels,
details, or metadata.

| Phrase | Extracted as |
|--------|--------------|
| "stopped", "parked", "queued", "lingering" | `text_terms=(...)` |

### Plate text

| Pattern | Example | Behaviour |
|---------|---------|-----------|
| Exact | "plate ABC1234" | Exact match |
| Partial / similar | "plate similar to AB12", "plate AB*" | Substring or glob |

Similarity-only phrases such as "similar to AB12" are only accepted when the
query already contains explicit plate or ANPR context. This avoids treating
location phrases such as "near Junction 4" as a plate filter.

### Review status

| Phrase | Status |
|--------|--------|
| "pending", "pending review", "awaiting review" | `PENDING` |
| "in review", "under review" | `IN_REVIEW` |
| "confirmed" | `CONFIRMED` |
| "dismissed", "rejected" | `DISMISSED` |
| "escalated" | `ESCALATED` |

### Vehicle types

"trucks", "cars", "vans", "buses", "motorcycles", "bicycles", "taxis",
"ambulances", "emergency".

## Safeguards

The search layer applies the following safeguards:

| Safeguard | Trigger | Effect |
|-----------|---------|--------|
| `NO_FILTERS` | No filter dimensions extracted | Warning in result |
| `OVERLY_BROAD` | No time range in query | Warning in result |
| `AMBIGUOUS_ENTITY` | Multiple values extracted for one filter dimension | Warning that OR semantics are in effect |
| `TOO_MANY_RESULTS` | Candidate count exceeds threshold | Warning + truncation |
| `UNRECOGNIZED_TOKENS` | Words not mapped to any filter | Warning listing ignored words |

Safeguards are informational — the search still executes. Callers should
inspect `SearchResult.safeguards` and decide whether to present a warning
to the operator.

## Limitations

- The parser uses English keyword tables only.
- Time expressions are relative to a reference timestamp (defaults to now).
  Absolute dates ("2026-04-01") are not yet supported.
- Vehicle type is an informational filter passed to the store. Whether results
  actually carry vehicle metadata depends on the platform's record population.
- The parser does not understand negation ("not Junction 4"). Multiple values
  within one filter dimension are treated with OR semantics and produce an
  ambiguity safeguard.
- Semantic / vector search is not included. The protocol is designed so a
  future `VectorSearchStore` implementation can be added alongside the
  existing keyword store without changing the executor or parser.

## Architecture

```
Operator query (string)
        │
        ▼
   parse_query()          ← deterministic regex/keyword extraction
        │
        ▼
   ParsedQuery            ← typed filter + extractions + safeguards
        │
        ▼
SearchExecutor.execute()  ← applies safeguards, queries store
        │
        ▼
   SearchResult           ← hits with RecordReferences + audit log
```

## Example

```python
from trafficmind.search import (
    InMemorySearchStore,
    PlatformRecord,
    RecordKind,
    ReviewStatus,
    SearchExecutor,
)

# Set up a store with some records
store = InMemorySearchStore()
store.add(PlatformRecord(
    record_kind=RecordKind.VIOLATION,
    record_id="V-42",
    label="Red light at J-4",
    detail="Vehicle crossed stop line while signal restrictive.",
    timestamp=1712345678.0,
    junction_id="J-4",
    event_type="red_light_violation",
    violation_type="red_light",
    review_status=ReviewStatus.PENDING,
))

# Create an executor
executor = SearchExecutor(store)

# Search with natural language
result = executor.search(
    "find red-light violations at junction 4 in the last 24 hours",
    now=1712360000.0,
)

# Inspect results
print(f"Confidence: {result.query.confidence.value}")
print(f"Explanation:\n{result.query.explanation}")
print(f"Hits: {len(result.hits)}")
for hit in result.hits:
    print(f"  {hit.reference.record_id}: {hit.reference.label}")
    print(f"    Matched filters: {hit.matched_filters}")
  for evidence in hit.matched_evidence:
    print(
      "    "
      f"[{evidence.filter_name}] requested={evidence.requested_value!r} "
      f"matched={evidence.matched_value!r} basis={evidence.basis}"
    )
if result.safeguards:
    print("Safeguards:")
    for sg in result.safeguards:
        print(f"  [{sg.kind.value}] {sg.message}")
```
