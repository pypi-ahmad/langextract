# TrafficMind Privacy Redaction Foundation

## Purpose

TrafficMind now includes a privacy-aware evidence foundation for handling
sensitive visual details such as:

- faces
- plates
- other personally identifying visual details

The foundation is designed to keep original and redacted asset lineage explicit
across manifests, review playback, and export planning.

It is a **policy and provenance layer**. It does **not** claim that TrafficMind
is fully compliant with every privacy or surveillance regulation in every
jurisdiction.

## What The Foundation Supports

### Sensitive-detail modeling

Each `EvidenceReference` can declare `SensitiveVisualDetail` entries for:

- `face`
- `plate`
- `personally_identifying_detail`

Each detail records the masking operation that should be applied when policy
requires redaction:

- `face_mask`
- `plate_mask`
- `detail_mask`

Bounding boxes are optional and are treated as a foundation for a later masking
engine, not as proof that masking has already been performed.

### Original versus redacted asset separation

Each logical evidence item can now preserve:

- `original_asset`
- `redacted_asset`

These remain separate fields so callers do not overwrite provenance when a
redacted variant is produced.

### Explicit redaction state

Each evidence item declares one of:

- `not_required`
- `pending`
- `redacted_available`

This state is visible in manifests and in the frontend-oriented playback view
models returned by the privacy helpers.

### Role-based policy

`RedactionPolicy` controls:

- which sensitive detail kinds require redaction
- which roles are masked by default
- which roles may retain access to originals
- whether export stays redacted by default
- whether restricted roles fall back to metadata-only access when a redacted
  asset is not yet available

Default policy behavior is intentionally conservative:

- operators, supervisors, and auditors are masked by default
- privacy reviewers and admins may retain original access
- export is redacted by default

## Playback And UI Support

This repository does not contain a full web playback UI. Instead, the privacy
layer exposes typed playback models intended for frontend consumption:

- `PresentedEvidenceReference`
- `EvidencePlaybackManifest`

These models make the following explicit for every visible evidence item:

- selected role-visible view (`original`, `redacted`, or metadata-only)
- selected storage URI
- selected access mode
- redaction state
- whether original and redacted variants exist
- sensitive detail kinds present
- the access boundary message shown to the caller
- warnings when a restricted role cannot receive the original asset

## Export Support

`build_export_bundle()` resolves which asset lineage a role may export.

Important defaults:

- exports remain redacted by default
- original export requires an explicit request and an authorized role
- unauthorized original-export requests are downgraded to redacted or
  metadata-only entries with warnings

This module returns an export manifest / bundle description. It does not write
files to disk or transfer media by itself.

## Permission-Based Access Control

The privacy layer is backed by a typed permission model in
`trafficmind.review.access`. Each `ReviewRole` maps to a set of `Permission`
values that govern what evidence and workflow actions the role can perform.

### Permissions

| Permission | Description |
|---|---|
| `view_redacted_evidence` | View evidence with masking applied |
| `view_unredacted_evidence` | View original, unmasked evidence |
| `export_redacted_evidence` | Export redacted evidence bundles |
| `export_unredacted_evidence` | Export original evidence bundles |
| `approve_incident` | Mark an incident as confirmed |
| `reject_incident` | Dismiss an incident |
| `escalate_incident` | Escalate an incident to a supervisor |
| `manage_watchlists` | Create and modify watchlists |
| `manage_policy` | Modify redaction and access policy |
| `view_audit_trail` | View basic audit logs |
| `view_full_audit_trail` | View full audit logs including access decisions |

### Default Role → Permission Mapping

| Permission | Operator | Supervisor | Auditor | Privacy Reviewer | Admin |
|---|:---:|:---:|:---:|:---:|:---:|
| view_redacted_evidence | ✓ | ✓ | ✓ | ✓ | ✓ |
| view_unredacted_evidence | | | | ✓ | ✓ |
| export_redacted_evidence | ✓ | ✓ | ✓ | ✓ | ✓ |
| export_unredacted_evidence | | | | ✓ | ✓ |
| approve_incident | ✓ | ✓ | | | ✓ |
| reject_incident | ✓ | ✓ | | | ✓ |
| escalate_incident | ✓ | ✓ | | | ✓ |
| manage_watchlists | | ✓ | | | ✓ |
| manage_policy | | | | | ✓ |
| view_audit_trail | ✓ | ✓ | ✓ | ✓ | ✓ |
| view_full_audit_trail | | | ✓ | ✓ | ✓ |

### Enforcement Points

Permission checks are enforced at the backend when an `AccessContext` is
supplied to `build_playback_manifest()` or `build_export_bundle()`. When the
caller lacks the required permission:

- The manifest or bundle is returned **empty** (no entries).
- A denial reason appears in `warnings`.
- An audit entry appears in `access_audit`.

When no `AccessContext` is supplied, the functions fall back to policy-only
resolution (the pre-existing behaviour) — so existing callers are unaffected.

`check_workflow_action()` gates review workflow actions (approve, reject,
escalate) and returns a typed `AccessDecision` with an audit-ready log entry.

`resolve_audit_trail()` applies the same role-permission model to workflow
audit logs. Roles with `view_full_audit_trail` receive the full entries,
roles with only `view_audit_trail` receive a basic view with access-decision
details redacted, and roles without audit permissions receive an empty trail.

### Audit Logging

Every `AccessDecision` carries an `audit_entry` string in a fixed format:

```
ACCESS GRANTED: caller=op-1 role=operator permission=view_redacted_evidence incident=INC-1
ACCESS DENIED:  caller=op-1 role=operator permission=view_unredacted_evidence incident=INC-1
```

When the review workflow builds a playback manifest, any access audit entries
from the manifest are propagated into the workflow's own `audit_log`.
`MultimodalReviewResult.audit_trail` exposes the resolved audit-trail view for
frontend or API consumers, while `audit_log` contains the visible entries for
backward-compatible callers.

### Customization

The default mapping can be overridden by constructing a custom
`RolePermissions` and passing it to check functions:

```python
from trafficmind.review import (
    Permission, ReviewRole, RolePermissions,
    AccessContext, check_access,
)

custom = RolePermissions(role_grants={
    ReviewRole.OPERATOR: frozenset({
        Permission.VIEW_REDACTED_EVIDENCE,
        Permission.MANAGE_WATCHLISTS,  # grant extra
    }),
})
ctx = AccessContext(role=ReviewRole.OPERATOR, caller_id="op-1")
decision = check_access(ctx, Permission.MANAGE_WATCHLISTS, role_permissions=custom)
assert decision.granted
```

## Review Workflow Integration

`MultimodalReviewRequest` now accepts:

- `viewer_role`
- `redaction_policy`

The review workflow resolves a role-aware `EvidencePlaybackManifest` before
building the prompt bundle and returns that manifest on
`MultimodalReviewResult.playback_manifest`.

This keeps access boundaries explicit for downstream playback or operator UI
surfaces.

## Current Boundaries

The current implementation does **not** do the following:

- perform computer-vision detection of faces or plates automatically
- generate blurred or pixelated images by itself
- enforce legal retention schedules
- certify compliance with GDPR, UK surveillance codes, CJIS, or other
  jurisdiction-specific requirements
- replace a full IAM system — the permission layer covers typed backend
  enforcement and audit-ready decisions, but does not provide authentication,
  token management, or session handling

Those capabilities may be added later, but they are not present today and
should not be implied in product or compliance documentation.

## Example

```python
from trafficmind.review import (
    DEFAULT_REDACTION_POLICY,
    EvidenceAccessMode,
    EvidenceAssetReference,
    EvidenceManifest,
    EvidenceMediaKind,
    EvidenceReference,
    MaskingOperation,
    RedactionState,
    ReviewRole,
    SensitiveVisualDetail,
    SensitiveVisualKind,
    build_export_bundle,
    build_playback_manifest,
)

manifest = EvidenceManifest(
    manifest_id="MAN-100",
    incident_id="INC-100",
    references=(
        EvidenceReference(
            evidence_id="frame-1",
            media_kind=EvidenceMediaKind.FRAME,
            original_asset=EvidenceAssetReference(
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-1-original.jpg",
            ),
            redacted_asset=EvidenceAssetReference(
                access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                storage_uri="s3://bucket/frame-1-redacted.jpg",
            ),
            redaction_state=RedactionState.REDACTED_AVAILABLE,
            sensitive_details=(
                SensitiveVisualDetail(
                    detail_id="plate-1",
                    kind=SensitiveVisualKind.PLATE,
                    masking_operation=MaskingOperation.PLATE_MASK,
                ),
            ),
        ),
    ),
)

playback = build_playback_manifest(
    manifest,
    ReviewRole.OPERATOR,
    DEFAULT_REDACTION_POLICY,
)

export_bundle = build_export_bundle(
    manifest,
    ReviewRole.OPERATOR,
    DEFAULT_REDACTION_POLICY,
)

print(playback.entries[0].selected_view.value)
print(export_bundle.export_view.value)
```