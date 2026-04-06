# TrafficMind Signal Controller Integration

## Overview

The signal integration layer enables TrafficMind to consume external
traffic-signal-controller state in addition to its existing vision-based
traffic-light detection. This makes red-light logic stronger where
external signal data exists while keeping source provenance explicit.

## Architecture

```
┌────────────────────────┐   ┌─────────────────────────┐
│   Signal Sources       │   │   Vision Pipeline       │
│  ┌──────────────────┐  │   │  (existing camera-based │
│  │ File Feed (.json/│  │   │   light detection)      │
│  │   .csv)          │  │   └──────────┬──────────────┘
│  ├──────────────────┤  │              │
│  │ Polling (HTTP)   │  │              │ SourceType.VISION
│  ├──────────────────┤  │              │
│  │ Webhook (push)   │  │              ▼
│  ├──────────────────┤  │   ┌─────────────────────────┐
│  │ Simulator (mock) │  │   │     Signal Store        │
│  └────────┬─────────┘  │   │  (per junction/phase/   │
│           │             │   │   source_type keying)   │
└───────────┼─────────────┘   └──────────┬──────────────┘
            │ SourceType.CONTROLLER       │
            │ SourceType.FILE_FEED        │
            │ SourceType.POLLING          ▼
            │ SourceType.WEBHOOK   ┌──────────────────┐
            └─────────────────────►│   Arbitrator     │
                                   │ (hybrid/ctrl/vis)│
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │  SignalReport     │
                                   │  resolved_state   │
                                   │  conflict flag    │
                                   │  stale flag       │
                                   │  reason string    │
                                   │  original states  │
                                   └──────────────────┘
```

## Supported Integration Patterns

### 1. File Feed
Read signal state from a local JSON or CSV file. Best for batch replay,
historical analysis, or air-gapped environments.

```python
from trafficmind.sources import FileFeedSource

source = FileFeedSource("signals.json")
states = source.fetch()
```

JSON format – top-level array of records:
```json
[
  {
    "junction_id": "J-42",
    "controller_id": "CTRL-7",
    "phase_id": "P-1",
    "state": "red",
    "timestamp": 1712345678.0,
    "confidence": 1.0
  }
]
```

### 2. HTTP Polling
Poll a REST endpoint that returns the same JSON array format.

```python
from trafficmind.sources import PollingSource

source = PollingSource(
    "https://signals.example.com/api/states",
    timeout_seconds=5,
    headers={"Authorization": "Bearer <token>"},
)
```

### 3. Webhook / Push Events
For event-driven architectures. The `WebhookReceiver` is a thread-safe
buffer that external HTTP handlers push records into.

```python
from trafficmind.sources import WebhookReceiver

receiver = WebhookReceiver(max_buffer=10_000)

# In your HTTP handler:
receiver.receive(request_json)

# Later, drain accumulated events:
states = receiver.fetch()
```

### 4. Mock Simulator
Cycles through a configurable phase sequence for testing.

```python
from trafficmind.sources import SimulatorSource
from trafficmind.models import PhaseState

sim = SimulatorSource(
    junction_id="SIM-J1",
    cycle=[PhaseState.GREEN, PhaseState.AMBER, PhaseState.RED],
)
```

## Trust / Priority Strategy

### Arbitration Modes

| Mode              | Behaviour |
|-------------------|-----------|
| `CONTROLLER_ONLY` | Use only external signal controller data. Vision is ignored. |
| `VISION_ONLY`     | Use only camera/vision-derived state. Controllers are ignored. |
| `HYBRID`          | Merge both sources with conflict detection (default). |

### Hybrid Resolution Rules

1. **Agreement** – when controller and vision report the same state,
   that state is used directly, even if one or both sources are stale
   (corroboration from two independent sources provides strong signal).
   The report still sets `stale=True` so consumers are aware.
2. **Both stale + conflict** – if both sources are stale and they
   disagree, the result is `UNKNOWN`. Stale, conflicting data cannot
   be trusted to make a safety decision.
3. **Restrictive wins** – if either *non-stale* source indicates a stop
   state (RED, RED_AMBER, FLASHING_RED), the restrictive state is chosen.
   This strengthens red-light logic.
4. **Fresh over stale** – when one source is stale and the other is
   fresh, the fresh source is preferred.
5. **Higher confidence** – when both sources are equally fresh and
   non-restrictive, the higher-confidence observation wins.

### Staleness

An observation is considered **stale** when its age exceeds the
configurable `stale_after_seconds` threshold (default: 30 s).

Staleness effects by scenario:
- **Single source, stale** → resolves to `UNKNOWN`.
- **Both stale, agreement** → uses the agreed state (corroboration is
  meaningful even from aging data).  Report flags `stale=True`.
- **Both stale, conflict** → resolves to `UNKNOWN` (cannot pick safely).
- **One stale, one fresh** → the fresh source is preferred.

Late-arriving out-of-order observations are also handled: the store
always keeps the *most-recent-by-timestamp* as the latest, so a delayed
old reading cannot mask a newer one.

### Conflict Transparency

Every `SignalReport` produced by the arbitrator carries:
- `conflict: bool` – True when sources disagreed.
- `stale: bool` – True when at least one source was stale.
- `reason: str` – Human-readable explanation of the resolution.
- `controller_state` / `vision_state` – the original observations, so
  downstream consumers can inspect raw provenance.

Conflicts are **never hidden**. The report always tells you why a
particular state was chosen.

## Limitations

- **No vendor-specific protocol support yet.** The integration layer is
  deliberately generic (JSON records via file / HTTP / push). Vendor
  adapters (e.g. NTCIP, UTMC, OCIT-C) should be added as new
  `SignalSource` sub-classes when needed.
- **No persistent storage.** The `SignalStore` is in-memory only. For
  durable history, plug in your own persistence behind the store.
- **Polling is synchronous.** Each `PollingSource.fetch()` call blocks on
  one HTTP round-trip. For high-frequency polling, wrap in an async loop
  or thread pool externally.
- **No authentication framework.** Webhook receivers should be fronted by
  your own auth middleware.
- **Single-process scope.** The store and arbitrator are not distributed.
  For multi-node deployments, share state through a message bus or
  database.

## Quick Start

```python
from trafficmind import SignalStore, Arbitrator, ArbitrationMode
from trafficmind.service import SignalService
from trafficmind.sources import SimulatorSource

service = SignalService(mode=ArbitrationMode.HYBRID, stale_after_seconds=30)
service.register_source(SimulatorSource())
service.ingest()

report = service.resolve("SIM-J1", "SIM-P1")
print(report.resolved_state, report.conflict, report.reason)
```
