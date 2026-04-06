"""Enterprise integration adapter foundations for TrafficMind.

This package defines vendor-neutral contracts for connecting TrafficMind to
external case systems, notification channels, reporting pipelines, and object
stores. Only lightweight local reference adapters are included here; real
vendor connectors should live in deployer code or future dedicated packages.

An optional signal-adapter bridge is available in
``trafficmind.integrations.signals`` for teams that want a consistent
adapter vocabulary alongside the existing ``trafficmind.sources.SignalSource``
seam, but it is not re-exported here because the real inbound integration
point remains ``SignalSource``.
"""

from trafficmind.integrations.interfaces import CaseSystemAdapter
from trafficmind.integrations.interfaces import NotificationChannelAdapter
from trafficmind.integrations.interfaces import ObjectStorageAdapter
from trafficmind.integrations.interfaces import ReportingPipelineAdapter
from trafficmind.integrations.local import InMemoryCaseSystemAdapter
from trafficmind.integrations.local import LocalFilesystemObjectStore
from trafficmind.integrations.models import build_case_update
from trafficmind.integrations.models import build_notification_message
from trafficmind.integrations.models import CaseSyncAction
from trafficmind.integrations.models import CaseSyncResult
from trafficmind.integrations.models import CaseUpdate
from trafficmind.integrations.models import IntegrationPriority
from trafficmind.integrations.models import NotificationDelivery
from trafficmind.integrations.models import NotificationMessage
from trafficmind.integrations.models import NotificationSeverity
from trafficmind.integrations.models import ObjectBlob
from trafficmind.integrations.models import ObjectPutRequest
from trafficmind.integrations.models import ReportingBatch
from trafficmind.integrations.models import ReportingPublishResult
from trafficmind.integrations.models import StoredObjectReference

__all__ = [
    "CaseSyncAction",
    "CaseSyncResult",
    "CaseSystemAdapter",
    "CaseUpdate",
    "InMemoryCaseSystemAdapter",
    "IntegrationPriority",
    "LocalFilesystemObjectStore",
    "NotificationChannelAdapter",
    "NotificationDelivery",
    "NotificationMessage",
    "NotificationSeverity",
    "ObjectBlob",
    "ObjectPutRequest",
    "ObjectStorageAdapter",
    "ReportingBatch",
    "ReportingPipelineAdapter",
    "ReportingPublishResult",
    "StoredObjectReference",
    "build_case_update",
    "build_notification_message",
]
