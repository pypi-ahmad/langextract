# TrafficMind Deployment & Run Modes

## Current Status

TrafficMind is a **local-first** subsystem.  All components run in a
single process, with an in-memory signal store and threading-based
concurrency.  There is no distributed message bus, no required database,
and no orchestrator dependency.

This document describes what works today, what is partially ready, and
what is not yet production-grade.

For the broader module layout and integration-adapter boundaries, see
`docs/trafficmind/architecture.md`.

## Supported Run Modes

Examples below use POSIX shell syntax. In PowerShell, use
`$env:TRAFFICMIND_PROFILE = "dev"` before running `trafficmind-check`.

### 1. Local Development (default)

```bash
# Install with dev extras
pip install -e ".[dev,test]"

# Run startup checks
trafficmind-check
# or, from a source checkout without installing entry points:
python scripts/check_trafficmind.py

# Run tests
pytest tests/ -k trafficmind
```

The default profile (`TRAFFICMIND_PROFILE=local`) uses relaxed staleness
windows (60 s), verbose logging (DEBUG), and does not require any
external services.

### 2. Dev / Shared Server

```bash
export TRAFFICMIND_PROFILE=dev
trafficmind-check
```

Tighter staleness (30 s), INFO-level logging.  Useful when running the
service on a shared VM for integration testing.

### 3. Staging

```bash
export TRAFFICMIND_PROFILE=staging
export TRAFFICMIND_POLLING_URL=http://signal-controller.internal/api/states
trafficmind-check
```

Mirrors production defaults (15 s staleness, WARNING log level) and the
startup check verifies the polling endpoint is reachable. A polling URL
is required in this profile.

### 4. Prod Validation Profile

```bash
export TRAFFICMIND_PROFILE=prod
export TRAFFICMIND_POLLING_URL=http://signal-controller.internal/api/states
trafficmind-check
```

Same as staging with stricter validation enforced (e.g.
`stale_after_seconds` must be ≤ 30 and `TRAFFICMIND_POLLING_URL` is
required). This profile name exists for strict runtime validation, not
as a claim that the subsystem is fully production-ready.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAFFICMIND_PROFILE` | `local` | Profile name: `local`, `dev`, `staging`, `prod` |
| `TRAFFICMIND_STALE_AFTER_SECONDS` | *profile-dependent* | Seconds before a signal observation is considered stale |
| `TRAFFICMIND_HISTORY_SIZE` | *profile-dependent* | Number of past observations kept per key |
| `TRAFFICMIND_LOG_LEVEL` | *profile-dependent* | Python log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `TRAFFICMIND_POLLING_URL` | *(unset)* | HTTP endpoint for the polling source; required in `staging` and `prod` |
| `TRAFFICMIND_POLLING_TIMEOUT` | `10` | HTTP timeout in seconds for the polling source |
| `TRAFFICMIND_WEBHOOK_MAX_BUFFER` | `10000` | Maximum buffered webhook events before oldest are dropped |

Profile defaults:

| Setting | local | dev | staging | prod |
|---|---|---|---|---|
| `stale_after_seconds` | 60 | 30 | 15 | 15 |
| `history_size` | 100 | 50 | 50 | 50 |
| `log_level` | DEBUG | INFO | INFO | WARNING |

## Startup Checks

`trafficmind-check` (or `trafficmind.health.run_startup_checks`)
validates:

1. **Config loads** — all env vars parse to valid types and pass profile
   constraints.
2. **SignalStore instantiates** — the configured staleness and history
   values work.
3. **Polling URL shape** — when configured, it must be a valid `http` or
   `https` URL.
4. **Polling endpoint reachable** — in `staging` and `prod`, the polling
   URL is required and a HEAD request must succeed within the timeout.

The CLI exits 0 on success and 1 on any failure. Source checkouts can also
run `python scripts/check_trafficmind.py` directly.

## Health Probes

`trafficmind.health.health_snapshot()` returns a `HealthSnapshot`
dataclass with:

- `ok` — boolean readiness flag
- `profile` — active profile name
- `uptime_seconds` — since the reference start time
- `store_junctions` — number of junctions currently tracked
- `store_observations` — total observations in the store
- `source_count` — registered signal sources

Call `.as_dict()` to serialise for a `/healthz` JSON response.

## Docker

The root `Dockerfile` installs the published `langextract` wheel from PyPI.
It is designed for the extraction library, not specifically for running
TrafficMind services.

For a TrafficMind-specific container you would need to:

1. Copy the repo source into the image (the subsystem is not published
   separately on PyPI yet).
2. Set `TRAFFICMIND_PROFILE` and any required env vars.
3. Run `trafficmind-check` as a container health check.

**There is no Kubernetes manifest, Helm chart, or Terraform config in
this repo.**  Adding those is premature until the service has a proper
external API (HTTP/gRPC) and persistent storage layer.

## CI Validation

The CI workflow (`.github/workflows/ci.yaml`) runs:

- **Format check** — pyink + isort on `langextract`, `trafficmind`, and
  `tests`.
- **Import structure** — import-linter against the contracts in
   `pyproject.toml`.
- **Lint** — pylint on `langextract` and `trafficmind`.
- **Unit tests** — pytest across Python 3.10–3.12.
- **Build validation** — `python -m build` plus `twine check` on the wheel
   and sdist.
- **Startup sanity** — `trafficmind-check` under a non-default `dev`
   profile.

TrafficMind tests are included in the standard `pytest` run and require
no external services.

## What Is NOT Production-Ready

| Area | Status | Notes |
|---|---|---|
| Persistent storage | Not implemented | Signal store is in-memory only |
| External API server | Not implemented | No HTTP/gRPC server; `SignalService` is a library |
| Authentication / RBAC | Not implemented | Review workflow has access-control models but no auth integration |
| Horizontal scaling | Not supported | Single-process, thread-safe store only |
| Metrics / tracing | Not implemented | Standard Python logging only |
| TLS / mTLS | Not implemented | Polling source uses plain `urllib` |
| Secrets management | Not integrated | Env vars only; no vault or KMS |

These are listed honestly so that anyone evaluating the system knows
exactly where the boundaries are.

## Secrets & Sensitive Config

TrafficMind itself does not require API keys or secrets for its core
signal-integration workflow.  The broader `langextract` library uses:

| Secret | Used By | Storage |
|---|---|---|
| `GEMINI_API_KEY` / `LANGEXTRACT_API_KEY` | Gemini provider | Env var / `.env` file |
| `OPENAI_API_KEY` | OpenAI provider | Env var / `.env` file |
| `GLM_OCR_API_KEY` | GLM OCR backend | Env var |
| `ZENODO_TOKEN` | DOI publication | GitHub Actions secret |

**Never commit `.env` files or API keys to the repository.**  The
`.gitignore` already excludes `.env`. Start from `.env.example` when you
need a local config template.
