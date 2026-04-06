"""Search executor that ties the query parser to searchable stores.

The executor is the primary entry point for natural-language search.  It
parses the operator query, executes it against a ``SearchableStore``,
enforces safeguards, and returns an explainable ``SearchResult``.
"""

from __future__ import annotations

import time
from typing import Sequence

from trafficmind.search.models import ParseConfidence
from trafficmind.search.models import ParsedQuery
from trafficmind.search.models import Safeguard
from trafficmind.search.models import SafeguardKind
from trafficmind.search.models import SearchResult
from trafficmind.search.parser import parse_query
from trafficmind.search.store import SearchableStore

DEFAULT_MAX_RESULTS = 200
BROAD_QUERY_CANDIDATE_THRESHOLD = 5000


class SearchExecutor:
  """Execute natural-language searches against a ``SearchableStore``.

  The executor is stateless; all per-search context lives in the
  ``SearchResult`` returned from ``search()``.

  Parameters
  ----------
  store:
      Backend that implements the ``SearchableStore`` protocol.
  max_results:
      Default maximum hits to return per search.
  broad_threshold:
      If total candidates exceed this number, a ``TOO_MANY_RESULTS``
      safeguard is added.
  require_time_range:
      If True (default), searches without a time range receive a
      safeguard warning and are still executed.
  """

  def __init__(
      self,
      store: SearchableStore,
      *,
      max_results: int = DEFAULT_MAX_RESULTS,
      broad_threshold: int = BROAD_QUERY_CANDIDATE_THRESHOLD,
      require_time_range: bool = True,
  ) -> None:
    self.store = store
    self.max_results = max_results
    self.broad_threshold = broad_threshold
    self.require_time_range = require_time_range

  def search(
      self,
      raw_query: str,
      *,
      now: float | None = None,
      max_results: int | None = None,
  ) -> SearchResult:
    """Parse and execute a natural-language search.

    Parameters
    ----------
    raw_query:
        The operator's natural-language search string.
    now:
        Reference timestamp for relative time expressions.
    max_results:
        Override the default max_results for this single search.

    Returns
    -------
    A ``SearchResult`` with hits, the parsed query, safeguards,
    and an audit log.
    """
    if now is None:
      now = time.time()

    effective_max = max_results if max_results is not None else self.max_results
    parsed = parse_query(raw_query, now=now, max_results=effective_max)

    return self.execute(parsed)

  def execute(self, parsed: ParsedQuery) -> SearchResult:
    """Execute a pre-parsed query against the store.

    This is useful when the caller wants to inspect or modify the
    parsed query before execution.
    """
    audit: list[str] = []
    all_safeguards: list[Safeguard] = list(parsed.safeguards)

    audit.append(
        f"Parsed query with confidence={parsed.confidence.value}, "
        f"filter_count={parsed.search_filter.filter_count}, "
        f"unrecognized={len(parsed.unrecognized_tokens)}"
    )

    # Count total candidates before truncation
    total_candidates = self.store.count(parsed.search_filter)
    audit.append(f"Total matching candidates: {total_candidates}")

    if total_candidates > self.broad_threshold:
      all_safeguards.append(
          Safeguard(
              kind=SafeguardKind.TOO_MANY_RESULTS,
              message=(
                  f"Query matched {total_candidates} records, which exceeds "
                  f"the broad-query threshold of {self.broad_threshold}. "
                  "Results are truncated. Add more filters to narrow."
              ),
          )
      )
      audit.append(
          f"TOO_MANY_RESULTS safeguard: {total_candidates} >"
          f" {self.broad_threshold}"
      )

    if self.require_time_range and parsed.search_filter.time_range is None:
      if not any(s.kind == SafeguardKind.OVERLY_BROAD for s in all_safeguards):
        all_safeguards.append(
            Safeguard(
                kind=SafeguardKind.OVERLY_BROAD,
                message=(
                    "No time range was extracted. Consider adding a time"
                    " constraint."
                ),
            )
        )
      audit.append("No time range in query (require_time_range=True)")

    # Execute
    hits = self.store.query(parsed.search_filter)
    truncated = total_candidates > parsed.search_filter.max_results
    audit.append(f"Returned {len(hits)} hits (truncated={truncated})")

    return SearchResult(
        query=parsed,
        hits=tuple(hits),
        total_candidates=total_candidates,
        truncated=truncated,
        safeguards=tuple(all_safeguards),
        audit_log=tuple(audit),
    )
