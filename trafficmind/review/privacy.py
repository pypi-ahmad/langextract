"""Privacy-aware evidence presentation and export helpers.

This module provides a policy-driven foundation for choosing original versus
redacted evidence variants for playback, review, and export. It does not
perform image processing itself; it models and enforces which variant should be
used for a given role.

When an ``AccessContext`` is supplied, the helpers perform backend permission
checks via the access control layer and attach audit entries to the returned
manifests. When omitted, behaviour is unchanged (policy-only resolution).
"""

from __future__ import annotations

from trafficmind.review.models import AssetViewKind
from trafficmind.review.models import EvidenceAccessMode
from trafficmind.review.models import EvidenceExportBundle
from trafficmind.review.models import EvidenceManifest
from trafficmind.review.models import EvidencePlaybackManifest
from trafficmind.review.models import EvidenceReference
from trafficmind.review.models import PresentedEvidenceReference
from trafficmind.review.models import RedactionPolicy
from trafficmind.review.models import RedactionState
from trafficmind.review.models import ReviewRole
from trafficmind.review.models import SensitiveVisualKind

DEFAULT_REDACTION_POLICY = RedactionPolicy(policy_id="privacy-default-v1")


def build_playback_manifest(
    manifest: EvidenceManifest,
    viewer_role: ReviewRole,
    policy: RedactionPolicy | None = None,
    *,
    access_context: "AccessContext | None" = None,
    role_permissions: "RolePermissions | None" = None,
) -> EvidencePlaybackManifest:
  """Resolve which evidence view a role should receive for playback.

  When *access_context* is supplied, a permission check is performed and
  the resulting audit entry is attached to the manifest. If the caller
  lacks ``VIEW_REDACTED_EVIDENCE`` the manifest is returned empty with a
  denial audit entry.
  """
  active_policy = policy or DEFAULT_REDACTION_POLICY
  access_audit: list[str] = []

  # Permission check when access context is provided.
  if access_context is not None:
    if access_context.role != viewer_role:
      raise ValueError(
          f"access_context.role ({access_context.role.value}) does not match "
          f"viewer_role ({viewer_role.value}); refusing to check permissions "
          "against a different identity than the evidence resolver uses"
      )
    from trafficmind.review.access import check_access
    from trafficmind.review.access import Permission

    wants_unredacted = viewer_role in active_policy.unredacted_roles
    base_permission = (
        Permission.VIEW_UNREDACTED_EVIDENCE
        if wants_unredacted
        else Permission.VIEW_REDACTED_EVIDENCE
    )
    decision = check_access(
        access_context,
        base_permission,
        role_permissions=role_permissions,
    )
    access_audit.append(decision.audit_entry)
    if not decision.granted:
      return EvidencePlaybackManifest(
          manifest_id=manifest.manifest_id,
          incident_id=manifest.incident_id,
          viewer_role=viewer_role,
          policy_id=active_policy.policy_id,
          default_view=AssetViewKind.REDACTED,
          entries=(),
          warnings=(decision.reason,),
          access_audit=tuple(access_audit),
      )

  entries = tuple(
      _present_reference(
          reference,
          viewer_role,
          active_policy,
          include_originals=(viewer_role in active_policy.unredacted_roles),
      )
      for reference in manifest.references
  )
  warnings = _collect_manifest_warnings(entries)
  if viewer_role in active_policy.mask_by_default_for_roles:
    warnings = (
        (
            f"Role {viewer_role.value} receives masked-by-default playback"
            f" under policy {active_policy.policy_id}."
        ),
        *warnings,
    )
    default_view = AssetViewKind.REDACTED
  else:
    default_view = AssetViewKind.ORIGINAL
  return EvidencePlaybackManifest(
      manifest_id=manifest.manifest_id,
      incident_id=manifest.incident_id,
      viewer_role=viewer_role,
      policy_id=active_policy.policy_id,
      default_view=default_view,
      entries=entries,
      warnings=warnings,
      access_audit=tuple(access_audit),
  )


def build_export_bundle(
    manifest: EvidenceManifest,
    viewer_role: ReviewRole,
    policy: RedactionPolicy | None = None,
    *,
    include_originals: bool | None = None,
    access_context: "AccessContext | None" = None,
    role_permissions: "RolePermissions | None" = None,
) -> EvidenceExportBundle:
  """Resolve which evidence view a role should receive for export.

  Exports remain redacted by default unless the caller explicitly requests
  originals and the role is authorized by policy.

  When *access_context* is supplied, a permission check is performed and
  the resulting audit entry is attached to the bundle. If the caller
  lacks the required export permission the bundle is returned empty with
  a denial audit entry.
  """
  active_policy = policy or DEFAULT_REDACTION_POLICY
  requested_originals = (
      False if include_originals is None else include_originals
  )
  allow_originals = (
      requested_originals and viewer_role in active_policy.unredacted_roles
  )
  access_audit: list[str] = []

  # Permission check when access context is provided.
  if access_context is not None:
    if access_context.role != viewer_role:
      raise ValueError(
          f"access_context.role ({access_context.role.value}) does not match "
          f"viewer_role ({viewer_role.value}); refusing to check permissions "
          "against a different identity than the evidence resolver uses"
      )
    from trafficmind.review.access import check_access
    from trafficmind.review.access import Permission

    export_permission = (
        Permission.EXPORT_UNREDACTED_EVIDENCE
        if allow_originals
        else Permission.EXPORT_REDACTED_EVIDENCE
    )
    decision = check_access(
        access_context,
        export_permission,
        role_permissions=role_permissions,
    )
    access_audit.append(decision.audit_entry)
    if not decision.granted:
      return EvidenceExportBundle(
          manifest_id=manifest.manifest_id,
          incident_id=manifest.incident_id,
          viewer_role=viewer_role,
          policy_id=active_policy.policy_id,
          export_view=AssetViewKind.REDACTED,
          includes_originals=False,
          entries=(),
          warnings=(decision.reason,),
          access_audit=tuple(access_audit),
      )

  entries = tuple(
      _present_reference(
          reference,
          viewer_role,
          active_policy,
          include_originals=allow_originals,
          prefer_redacted=(
              not allow_originals and active_policy.export_redacted_by_default
          ),
      )
      for reference in manifest.references
  )
  export_view = (
      AssetViewKind.ORIGINAL
      if allow_originals
      and any(
          entry.selected_view == AssetViewKind.ORIGINAL for entry in entries
      )
      else AssetViewKind.REDACTED
  )
  warnings = list(_collect_manifest_warnings(entries))
  if include_originals is None and active_policy.export_redacted_by_default:
    warnings.insert(
        0,
        "Export defaults to redacted assets under policy"
        f" {active_policy.policy_id}.",
    )
  elif requested_originals and not allow_originals:
    warnings.insert(
        0,
        f"Role {viewer_role.value} is not authorized to export original assets"
        f" under policy {active_policy.policy_id}; returned redacted or"
        " metadata-only entries instead.",
    )
  return EvidenceExportBundle(
      manifest_id=manifest.manifest_id,
      incident_id=manifest.incident_id,
      viewer_role=viewer_role,
      policy_id=active_policy.policy_id,
      export_view=export_view,
      includes_originals=allow_originals
      and export_view == AssetViewKind.ORIGINAL,
      entries=entries,
      warnings=tuple(warnings),
      access_audit=tuple(access_audit),
  )


def _present_reference(
    reference: EvidenceReference,
    viewer_role: ReviewRole,
    policy: RedactionPolicy,
    *,
    include_originals: bool,
    prefer_redacted: bool = False,
) -> PresentedEvidenceReference:
  warnings: list[str] = []
  requires_redaction = _requires_redaction(reference, policy)
  original_available = reference.original_asset is not None
  redacted_available = reference.redacted_asset is not None
  original_authorized = viewer_role in policy.unredacted_roles

  selected_view: AssetViewKind | None = None
  selected_access_mode = EvidenceAccessMode.METADATA_ONLY
  selected_storage_uri: str | None = None

  if (
      include_originals
      and original_authorized
      and reference.original_asset is not None
  ):
    selected_view = AssetViewKind.ORIGINAL
    selected_access_mode = reference.original_asset.access_mode
    selected_storage_uri = reference.original_asset.storage_uri
  elif (
      prefer_redacted
      and requires_redaction
      and reference.redacted_asset is not None
  ):
    selected_view = AssetViewKind.REDACTED
    selected_access_mode = reference.redacted_asset.access_mode
    selected_storage_uri = reference.redacted_asset.storage_uri
  elif requires_redaction and viewer_role in policy.mask_by_default_for_roles:
    if reference.redacted_asset is not None:
      selected_view = AssetViewKind.REDACTED
      selected_access_mode = reference.redacted_asset.access_mode
      selected_storage_uri = reference.redacted_asset.storage_uri
    elif policy.metadata_only_when_redaction_missing:
      warnings.append(
          "Sensitive evidence has no redacted asset yet; restricted roles"
          " receive metadata-only access until redaction is available."
      )
    else:
      warnings.append(
          "Sensitive evidence has no redacted asset; original asset remains"
          " restricted for this role."
      )
  elif requires_redaction:
    # Role is not in mask_by_default_for_roles or unredacted_roles but
    # evidence still requires redaction — default to redacted to avoid
    # leaking originals through a policy gap.
    if reference.redacted_asset is not None:
      selected_view = AssetViewKind.REDACTED
      selected_access_mode = reference.redacted_asset.access_mode
      selected_storage_uri = reference.redacted_asset.storage_uri
    elif policy.metadata_only_when_redaction_missing:
      warnings.append(
          "Sensitive evidence requires redaction and role is not authorized for"
          " originals; metadata-only access until redaction is available."
      )
    else:
      warnings.append(
          "Sensitive evidence requires redaction; original asset remains"
          " restricted for this role."
      )
  elif reference.original_asset is not None:
    selected_view = AssetViewKind.ORIGINAL
    selected_access_mode = reference.original_asset.access_mode
    selected_storage_uri = reference.original_asset.storage_uri
  elif reference.redacted_asset is not None:
    selected_view = AssetViewKind.REDACTED
    selected_access_mode = reference.redacted_asset.access_mode
    selected_storage_uri = reference.redacted_asset.storage_uri

  if requires_redaction:
    if original_authorized:
      access_boundary = (
          f"Role {viewer_role.value} may access originals under policy"
          f" {policy.policy_id}; non-authorized roles receive redacted or"
          " metadata-only assets."
      )
    else:
      access_boundary = (
          f"Role {viewer_role.value} is limited to redacted or metadata-only"
          f" assets under policy {policy.policy_id}; original assets remain"
          " restricted."
      )
  else:
    access_boundary = (
        f"No configured masking rule blocks role {viewer_role.value} from this"
        f" evidence under policy {policy.policy_id}."
    )

  return PresentedEvidenceReference(
      evidence_id=reference.evidence_id,
      media_kind=reference.media_kind,
      viewer_role=viewer_role,
      label=reference.label,
      description=reference.description,
      observed_at=reference.observed_at,
      clip_start=reference.clip_start,
      clip_end=reference.clip_end,
      selected_view=selected_view,
      selected_access_mode=selected_access_mode,
      selected_storage_uri=selected_storage_uri,
      redaction_state=reference.redaction_state,
      original_available=original_available,
      redacted_available=redacted_available,
      sensitive_kinds=_sensitive_kinds(reference),
      access_boundary=access_boundary,
      warnings=tuple(warnings),
  )


def _requires_redaction(
    reference: EvidenceReference,
    policy: RedactionPolicy,
) -> bool:
  if reference.redaction_state in (
      RedactionState.PENDING,
      RedactionState.REDACTED_AVAILABLE,
  ):
    return True
  return any(
      policy.requires_redaction(detail.kind)
      for detail in reference.sensitive_details
  )


def _sensitive_kinds(
    reference: EvidenceReference,
) -> tuple[SensitiveVisualKind, ...]:
  seen: list[SensitiveVisualKind] = []
  for detail in reference.sensitive_details:
    if detail.kind not in seen:
      seen.append(detail.kind)
  return tuple(seen)


def _collect_manifest_warnings(
    entries: tuple[PresentedEvidenceReference, ...],
) -> tuple[str, ...]:
  warnings: list[str] = []
  for entry in entries:
    warnings.extend(entry.warnings)
  deduped: list[str] = []
  for warning in warnings:
    if warning not in deduped:
      deduped.append(warning)
  return tuple(deduped)
