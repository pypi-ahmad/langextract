"""Model and configuration registry for TrafficMind.

Tracks which models, thresholds, rule configurations, and pipeline settings
produced a given event, violation, plate read, or evidence frame. Designed for
audit, debugging, and future experiment comparison.
"""

from trafficmind.registry.models import ConfigBundle
from trafficmind.registry.models import EntryStatus
from trafficmind.registry.models import ModelFamily
from trafficmind.registry.models import ModelRegistryEntry
from trafficmind.registry.models import PipelineSnapshot
from trafficmind.registry.models import PipelineSnapshotComparison
from trafficmind.registry.models import ProvenanceChain
from trafficmind.registry.models import ProvenanceStamp
from trafficmind.registry.models import RegistryEntryKind
from trafficmind.registry.models import RulesRegistryEntry
from trafficmind.registry.models import TaskType
from trafficmind.registry.registry import ModelConfigRegistry

__all__ = [
    "ConfigBundle",
    "EntryStatus",
    "ModelConfigRegistry",
    "ModelFamily",
    "ModelRegistryEntry",
    "PipelineSnapshot",
    "PipelineSnapshotComparison",
    "ProvenanceChain",
    "ProvenanceStamp",
    "RegistryEntryKind",
    "RulesRegistryEntry",
    "TaskType",
]
