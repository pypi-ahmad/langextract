# TrafficMind Architecture & Integration Foundations

## Scope

TrafficMind is currently a **local-first library/subsystem**, not a fully
packaged network service. This document separates what is implemented today
from the adapter foundations that are in place for future enterprise
integrations.

## Core Implemented Product

| Area | Modules | Implemented Today |
|---|---|---|
| Signal ingestion and arbitration | `trafficmind.sources`, `trafficmind.store`, `trafficmind.arbitration`, `trafficmind.service` | In-process ingestion of external or simulated signal states with deterministic arbitration and conflict visibility |
| Traffic analytics | `trafficmind.analytics` | Deterministic analytics over signal-history windows, including occupancy, oversaturation, queue discharge, phase duration, and violation trends |
| Operator review workflow | `trafficmind.review` | Advisory multimodal review workflow with typed requests/results, grounding, privacy-aware playback manifests, and export planning |
| Natural-language search | `trafficmind.search` | Query parsing plus a pluggable search-store contract with an in-memory implementation |
| Evaluation reporting | `trafficmind.evaluation` | Local artifact-backed evaluation report loading, rendering, and export |

## Advanced Implemented Foundations

| Foundation | Modules | Current Status |
|---|---|---|
| Runtime profiles and startup validation | `trafficmind.config`, `trafficmind.health`, `trafficmind.cli` | Implemented for local/dev/staging/prod validation and startup sanity checks |
| Privacy and access boundaries | `trafficmind.review.access`, `trafficmind.review.privacy` | Implemented for evidence viewing/export policy, audit-trail visibility, and role-aware export planning |
| Model and rules provenance | `trafficmind.registry` | Implemented for model/rules registration and provenance stamps carried into review and search records |
| Integration adapter contracts | `trafficmind.integrations` | Vendor-neutral protocols for case systems, notification channels, reporting pipelines, and object storage. No real vendor connectors are shipped |
| Local reference adapters | `trafficmind.integrations.local` | `InMemoryCaseSystemAdapter` and `LocalFilesystemObjectStore` for local testing and wiring only |
| External signal adapter bridge | `trafficmind.integrations.signals` | Optional naming bridge (not re-exported by default). The real ingestion seam remains `trafficmind.sources.SignalSource` |

## Future Integration Points

| Need | Current Seam | Still Missing |
|---|---|---|
| External incident / case systems | `CaseSystemAdapter`, `CaseUpdate` | Vendor-specific API mapping, retries, auth, idempotency, and durable sync queues |
| Notification channels | `NotificationChannelAdapter`, `NotificationMessage` | Trigger policy, routing rules, delivery observability, and concrete transports |
| BI / reporting pipelines | `ReportingPipelineAdapter`, `ReportingBatch` | Warehouse connectors, durable export jobs, schema governance, and backfill tooling |
| Object storage providers | `ObjectStorageAdapter`, `ObjectPutRequest`, `StoredObjectReference` | Cloud/back-office storage connectors, lifecycle controls, and signed-access flows |
| External signal systems | `SignalSource` plus `ExternalSignalAdapter` bridge | Vendor protocol implementations such as NTCIP, UTMC, OCIT-C, or site-specific integrations |
| Persistence and external APIs | `SignalStore`, `SearchableStore`, `SignalService` | Durable databases, HTTP/gRPC APIs, authentication, metrics, tracing, and horizontal scaling |

## Example: Case-System Payload

```python
from trafficmind.integrations import InMemoryCaseSystemAdapter
from trafficmind.integrations import build_case_update
from trafficmind.search.models import ReviewStatus

adapter = InMemoryCaseSystemAdapter()
case_update = build_case_update(
    event=request.event,
    review_status=ReviewStatus.IN_REVIEW,
    review_result=result,
    export_bundle=export_bundle,
)
receipt = adapter.upsert_case(case_update)
print(receipt.external_case_id)
```

This example is intentionally local. It exercises the adapter contract without
pretending that TrafficMind already ships a ServiceNow, Jira, or custom case
connector.

## Example: Local Object Storage

```python
from trafficmind.integrations import LocalFilesystemObjectStore
from trafficmind.integrations.models import ObjectPutRequest

store = LocalFilesystemObjectStore("./trafficmind-artifacts")
reference = store.put_object(
    ObjectPutRequest(
        object_name="exports/inc-42.json",
        content=b'{"incident_id": "INC-42"}',
        content_type="application/json",
    )
)
```

The returned `storage_uri` can be threaded through review/export workflows as a
concrete local reference. Cloud or enterprise object stores should implement
the same interface outside this repo or in a future dedicated package.

## Boundaries Kept Explicit

- TrafficMind does **not** ship real vendor connectors for case systems, chat systems, BI platforms, or cloud object stores in this repository.
- Notification adapters do **not** imply a built-in alerting engine. Triggering and escalation policy remain caller-owned.
- Privacy-aware export planning remains separate from object storage. `build_export_bundle()` decides *what* may be exported; storage adapters decide *where* bytes live.
- Optional review assistance remains advisory. Deterministic incident logic and structured event metadata remain authoritative.