"""Natural-language search layer for TrafficMind.

This package provides deterministic query translation from operator
natural-language requests into structured, auditable platform searches.
The search layer does not use LLM calls for query parsing — all
translation is keyword- and regex-based for reproducibility.
"""

from trafficmind.search.executor import SearchExecutor
from trafficmind.search.models import MatchEvidence
from trafficmind.search.models import ParseConfidence
from trafficmind.search.models import ParsedQuery
from trafficmind.search.models import RecordKind
from trafficmind.search.models import RecordReference
from trafficmind.search.models import ReviewStatus
from trafficmind.search.models import Safeguard
from trafficmind.search.models import SafeguardKind
from trafficmind.search.models import SearchFilter
from trafficmind.search.models import SearchHit
from trafficmind.search.models import SearchResult
from trafficmind.search.models import TimeRange
from trafficmind.search.models import TokenExtraction
from trafficmind.search.parser import parse_query
from trafficmind.search.store import InMemorySearchStore
from trafficmind.search.store import PlatformRecord
from trafficmind.search.store import SearchableStore

__all__ = [
    "InMemorySearchStore",
    "MatchEvidence",
    "ParseConfidence",
    "ParsedQuery",
    "PlatformRecord",
    "RecordKind",
    "RecordReference",
    "ReviewStatus",
    "Safeguard",
    "SafeguardKind",
    "SearchExecutor",
    "SearchFilter",
    "SearchHit",
    "SearchResult",
    "SearchableStore",
    "TimeRange",
    "TokenExtraction",
    "parse_query",
]
