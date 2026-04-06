"""Protocol-based adapter interfaces for TrafficMind enterprise integrations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trafficmind.integrations.models import CaseSyncResult
from trafficmind.integrations.models import CaseUpdate
from trafficmind.integrations.models import NotificationDelivery
from trafficmind.integrations.models import NotificationMessage
from trafficmind.integrations.models import ObjectBlob
from trafficmind.integrations.models import ObjectPutRequest
from trafficmind.integrations.models import ReportingBatch
from trafficmind.integrations.models import ReportingPublishResult
from trafficmind.integrations.models import StoredObjectReference


@runtime_checkable
class CaseSystemAdapter(Protocol):
  """Pluggable sink for external incident or case systems."""

  def adapter_name(self) -> str:
    """Return a human-readable adapter identifier."""

  def upsert_case(self, case_update: CaseUpdate) -> CaseSyncResult:
    """Create or update one external case from a TrafficMind payload."""


@runtime_checkable
class NotificationChannelAdapter(Protocol):
  """Pluggable outbound notification transport."""

  def adapter_name(self) -> str:
    """Return a human-readable adapter identifier."""

  def send_notification(
      self,
      message: NotificationMessage,
  ) -> NotificationDelivery:
    """Deliver one outbound notification."""


@runtime_checkable
class ReportingPipelineAdapter(Protocol):
  """Pluggable sink for BI, reporting, or downstream analytics batches."""

  def adapter_name(self) -> str:
    """Return a human-readable adapter identifier."""

  def publish_batch(
      self,
      batch: ReportingBatch,
  ) -> ReportingPublishResult:
    """Publish one tabular batch to a downstream reporting pipeline."""


@runtime_checkable
class ObjectStorageAdapter(Protocol):
  """Pluggable object storage provider for evidence and export assets."""

  def adapter_name(self) -> str:
    """Return a human-readable adapter identifier."""

  def put_object(
      self,
      request: ObjectPutRequest,
  ) -> StoredObjectReference:
    """Write an object and return its durable reference."""

  def get_object(self, storage_uri: str) -> ObjectBlob:
    """Read an object previously written to storage."""
