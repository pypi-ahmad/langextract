"""Typed outbound integration payloads for TrafficMind.

These models intentionally stay vendor-neutral. They carry TrafficMind domain
objects and adapter-ready payloads without claiming that any specific external
case system, notification product, reporting stack, or object store is already
integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Any, Mapping

from trafficmind.review.models import EvidenceExportBundle
from trafficmind.review.models import MultimodalReviewResult
from trafficmind.review.models import ReviewEvent
from trafficmind.search.models import ReviewStatus


class IntegrationPriority(str, enum.Enum):
  """Generic priority hint for outbound integrations."""

  LOW = "low"
  NORMAL = "normal"
  HIGH = "high"
  URGENT = "urgent"


class NotificationSeverity(str, enum.Enum):
  """Delivery severity for outbound notifications."""

  INFO = "info"
  WARNING = "warning"
  CRITICAL = "critical"


class CaseSyncAction(str, enum.Enum):
  """Result of synchronizing a TrafficMind incident to an external case."""

  CREATED = "created"
  UPDATED = "updated"
  NOOP = "noop"


@dataclass(frozen=True)
class CaseUpdate:
  """Vendor-neutral case payload built from TrafficMind review data."""

  incident_id: str
  event: ReviewEvent
  review_status: ReviewStatus = ReviewStatus.PENDING
  title: str | None = None
  summary: str | None = None
  external_case_id: str | None = None
  review_result: MultimodalReviewResult | None = None
  export_bundle: EvidenceExportBundle | None = None
  priority: IntegrationPriority = IntegrationPriority.NORMAL
  tags: tuple[str, ...] = ()
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.incident_id:
      raise ValueError("incident_id must be non-empty")
    if self.event.incident_id != self.incident_id:
      raise ValueError("event.incident_id must match incident_id")
    if (
        self.review_result is not None
        and self.review_result.incident_id != self.incident_id
    ):
      raise ValueError("review_result.incident_id must match incident_id")
    if (
        self.export_bundle is not None
        and self.export_bundle.incident_id != self.incident_id
    ):
      raise ValueError("export_bundle.incident_id must match incident_id")
    object.__setattr__(self, "tags", tuple(self.tags))
    object.__setattr__(self, "metadata", dict(self.metadata))

    resolved_title = self.title or self.event.title or self.event.event_type
    if not resolved_title:
      raise ValueError("title must be non-empty")
    object.__setattr__(self, "title", resolved_title)

    if self.summary is not None:
      resolved_summary = self.summary
    elif self.review_result is not None:
      resolved_summary = self.review_result.review_summary.text
    elif self.event.title:
      resolved_summary = self.event.title
    else:
      resolved_summary = (
          f"{self.event.event_type} at junction {self.event.junction_id}"
      )
    if not resolved_summary:
      raise ValueError("summary must be non-empty")
    object.__setattr__(self, "summary", resolved_summary)


@dataclass(frozen=True)
class CaseSyncResult:
  """Receipt returned by a case-system adapter."""

  adapter_name: str
  action: CaseSyncAction
  incident_id: str
  external_case_id: str
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.adapter_name:
      raise ValueError("adapter_name must be non-empty")
    if not self.incident_id:
      raise ValueError("incident_id must be non-empty")
    if not self.external_case_id:
      raise ValueError("external_case_id must be non-empty")
    object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class NotificationMessage:
  """Vendor-neutral notification payload."""

  notification_id: str
  title: str
  body: str
  severity: NotificationSeverity = NotificationSeverity.INFO
  incident_id: str | None = None
  priority: IntegrationPriority = IntegrationPriority.NORMAL
  tags: tuple[str, ...] = ()
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.notification_id:
      raise ValueError("notification_id must be non-empty")
    if not self.title:
      raise ValueError("title must be non-empty")
    if not self.body:
      raise ValueError("body must be non-empty")
    object.__setattr__(self, "tags", tuple(self.tags))
    object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class NotificationDelivery:
  """Receipt returned by a notification adapter."""

  adapter_name: str
  notification_id: str
  delivery_id: str
  destination: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.adapter_name:
      raise ValueError("adapter_name must be non-empty")
    if not self.notification_id:
      raise ValueError("notification_id must be non-empty")
    if not self.delivery_id:
      raise ValueError("delivery_id must be non-empty")
    object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ReportingBatch:
  """Tabular payload destined for a reporting or BI pipeline."""

  dataset: str
  generated_at: float
  rows: tuple[Mapping[str, Any], ...] = ()
  schema_version: str = "v1"
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.dataset:
      raise ValueError("dataset must be non-empty")
    object.__setattr__(
        self,
        "rows",
        tuple(dict(row) for row in self.rows),
    )
    object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ReportingPublishResult:
  """Receipt returned by a reporting adapter."""

  adapter_name: str
  dataset: str
  row_count: int
  location: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.adapter_name:
      raise ValueError("adapter_name must be non-empty")
    if not self.dataset:
      raise ValueError("dataset must be non-empty")
    if self.row_count < 0:
      raise ValueError("row_count must be >= 0")
    object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObjectPutRequest:
  """Write request for an object-store adapter."""

  object_name: str
  content: bytes
  content_type: str | None = None
  metadata: dict[str, str] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.object_name:
      raise ValueError("object_name must be non-empty")
    object.__setattr__(
        self,
        "metadata",
        {str(key): str(value) for key, value in self.metadata.items()},
    )


@dataclass(frozen=True)
class StoredObjectReference:
  """Descriptor returned after writing or locating an object."""

  object_name: str
  storage_uri: str
  size_bytes: int
  content_type: str | None = None
  metadata: dict[str, str] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.object_name:
      raise ValueError("object_name must be non-empty")
    if not self.storage_uri:
      raise ValueError("storage_uri must be non-empty")
    if self.size_bytes < 0:
      raise ValueError("size_bytes must be >= 0")
    object.__setattr__(
        self,
        "metadata",
        {str(key): str(value) for key, value in self.metadata.items()},
    )


@dataclass(frozen=True)
class ObjectBlob:
  """Object payload returned by a storage adapter read."""

  reference: StoredObjectReference
  content: bytes

  def __post_init__(self) -> None:
    if len(self.content) != self.reference.size_bytes:
      raise ValueError(
          "content length must match reference.size_bytes when reading objects"
      )


def build_case_update(
    event: ReviewEvent,
    *,
    review_status: ReviewStatus = ReviewStatus.PENDING,
    review_result: MultimodalReviewResult | None = None,
    export_bundle: EvidenceExportBundle | None = None,
    external_case_id: str | None = None,
    priority: IntegrationPriority = IntegrationPriority.NORMAL,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> CaseUpdate:
  """Build a case-sync payload from current TrafficMind review objects."""
  return CaseUpdate(
      incident_id=event.incident_id,
      event=event,
      review_status=review_status,
      external_case_id=external_case_id,
      review_result=review_result,
      export_bundle=export_bundle,
      priority=priority,
      tags=tags,
      metadata={} if metadata is None else dict(metadata),
  )


def build_notification_message(
    case_update: CaseUpdate,
    *,
    notification_id: str | None = None,
    severity: NotificationSeverity = NotificationSeverity.WARNING,
    title: str | None = None,
    body: str | None = None,
    priority: IntegrationPriority | None = None,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> NotificationMessage:
  """Build a vendor-neutral notification from a case-sync payload."""
  resolved_title = (
      title
      or f"{case_update.review_status.value.replace('_', ' ').title()}: {case_update.title}"
  )
  if body is not None:
    resolved_body = body
  elif case_update.review_result is not None:
    resolved_body = (
        f"{case_update.summary}\n\nRecommended action: "
        f"{case_update.review_result.recommended_operator_action.text}"
    )
  else:
    resolved_body = case_update.summary or case_update.title
  combined_tags = case_update.tags + tuple(tags)
  return NotificationMessage(
      notification_id=notification_id
      or f"{case_update.incident_id}:{case_update.review_status.value}",
      title=resolved_title,
      body=resolved_body,
      severity=severity,
      incident_id=case_update.incident_id,
      priority=priority or case_update.priority,
      tags=combined_tags,
      metadata={} if metadata is None else dict(metadata),
  )
