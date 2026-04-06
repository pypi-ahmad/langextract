"""Access control and permission enforcement for evidence and review actions.

This module provides a typed permission model layered on top of ``ReviewRole``.
It enforces backend access checks for evidence retrieval, export, workflow
actions, and policy management. Every check produces a typed ``AccessDecision``
with an audit-ready log entry.

This is intentionally **not** a full IAM system. It models the minimum
permission boundaries needed to enforce evidence access policy, and is designed
to be upgraded later without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
import re
import time
from typing import Any


class Permission(str, enum.Enum):
  """Named permissions that can be checked against a role."""

  # Evidence viewing
  VIEW_REDACTED_EVIDENCE = "view_redacted_evidence"
  VIEW_UNREDACTED_EVIDENCE = "view_unredacted_evidence"

  # Evidence export
  EXPORT_REDACTED_EVIDENCE = "export_redacted_evidence"
  EXPORT_UNREDACTED_EVIDENCE = "export_unredacted_evidence"

  # Review workflow actions
  APPROVE_INCIDENT = "approve_incident"
  REJECT_INCIDENT = "reject_incident"
  ESCALATE_INCIDENT = "escalate_incident"

  # Watchlist management
  MANAGE_WATCHLISTS = "manage_watchlists"

  # Policy and system settings
  MANAGE_POLICY = "manage_policy"

  # Audit trail visibility
  VIEW_AUDIT_TRAIL = "view_audit_trail"
  VIEW_FULL_AUDIT_TRAIL = "view_full_audit_trail"


class AuditTrailViewKind(str, enum.Enum):
  """Visibility level for audit-log presentation."""

  NONE = "none"
  BASIC = "basic"
  FULL = "full"


# ---------------------------------------------------------------------------
# Import ReviewRole here to avoid circular imports at module level.
# The review.models module does not import from this module.
# ---------------------------------------------------------------------------
from trafficmind.review.models import ReviewRole  # noqa: E402

# ---------------------------------------------------------------------------
# Default role → permission mapping
# ---------------------------------------------------------------------------

_DEFAULT_ROLE_PERMISSIONS: dict[ReviewRole, frozenset[Permission]] = {
    ReviewRole.OPERATOR: frozenset({
        Permission.VIEW_REDACTED_EVIDENCE,
        Permission.EXPORT_REDACTED_EVIDENCE,
        Permission.APPROVE_INCIDENT,
        Permission.REJECT_INCIDENT,
        Permission.ESCALATE_INCIDENT,
        Permission.VIEW_AUDIT_TRAIL,
    }),
    ReviewRole.SUPERVISOR: frozenset({
        Permission.VIEW_REDACTED_EVIDENCE,
        Permission.EXPORT_REDACTED_EVIDENCE,
        Permission.APPROVE_INCIDENT,
        Permission.REJECT_INCIDENT,
        Permission.ESCALATE_INCIDENT,
        Permission.MANAGE_WATCHLISTS,
        Permission.VIEW_AUDIT_TRAIL,
    }),
    ReviewRole.AUDITOR: frozenset({
        Permission.VIEW_REDACTED_EVIDENCE,
        Permission.EXPORT_REDACTED_EVIDENCE,
        Permission.VIEW_AUDIT_TRAIL,
        Permission.VIEW_FULL_AUDIT_TRAIL,
    }),
    ReviewRole.PRIVACY_REVIEWER: frozenset({
        Permission.VIEW_REDACTED_EVIDENCE,
        Permission.VIEW_UNREDACTED_EVIDENCE,
        Permission.EXPORT_REDACTED_EVIDENCE,
        Permission.EXPORT_UNREDACTED_EVIDENCE,
        Permission.VIEW_AUDIT_TRAIL,
        Permission.VIEW_FULL_AUDIT_TRAIL,
    }),
    ReviewRole.ADMIN: frozenset({
        Permission.VIEW_REDACTED_EVIDENCE,
        Permission.VIEW_UNREDACTED_EVIDENCE,
        Permission.EXPORT_REDACTED_EVIDENCE,
        Permission.EXPORT_UNREDACTED_EVIDENCE,
        Permission.APPROVE_INCIDENT,
        Permission.REJECT_INCIDENT,
        Permission.ESCALATE_INCIDENT,
        Permission.MANAGE_WATCHLISTS,
        Permission.MANAGE_POLICY,
        Permission.VIEW_AUDIT_TRAIL,
        Permission.VIEW_FULL_AUDIT_TRAIL,
    }),
}


@dataclass(frozen=True)
class RolePermissions:
  """Mapping from roles to their granted permissions.

  Callers can override the default mapping by supplying a custom
  ``role_grants`` dict. Unknown roles receive no permissions.
  """

  role_grants: dict[ReviewRole, frozenset[Permission]] = field(
      default_factory=lambda: dict(_DEFAULT_ROLE_PERMISSIONS),
  )

  def permissions_for(self, role: ReviewRole) -> frozenset[Permission]:
    """Return the permission set granted to *role*."""
    return self.role_grants.get(role, frozenset())

  def has_permission(self, role: ReviewRole, permission: Permission) -> bool:
    return permission in self.permissions_for(role)


DEFAULT_ROLE_PERMISSIONS = RolePermissions()


@dataclass(frozen=True)
class AccessContext:
  """Caller identity and context for a single access check.

  This is deliberately lightweight — it carries enough for the
  permission layer to make a decision and produce an audit entry,
  but does not model sessions, tokens, or authentication.
  """

  role: ReviewRole
  caller_id: str
  incident_id: str | None = None
  evidence_id: str | None = None
  reason: str = ""
  timestamp: float = field(default_factory=time.time)

  def __post_init__(self) -> None:
    if not self.caller_id:
      raise ValueError("caller_id must be non-empty")


@dataclass(frozen=True)
class AccessDecision:
  """Result of a permission check — granted or denied with audit context.

  ``permission`` is ``None`` when the check was denied before a valid
  permission could be resolved (e.g. unrecognized workflow action).
  """

  granted: bool
  permission: Permission | None
  role: ReviewRole
  caller_id: str
  reason: str
  audit_entry: str
  timestamp: float = field(default_factory=time.time)
  incident_id: str | None = None
  evidence_id: str | None = None


@dataclass(frozen=True)
class AuditTrailView:
  """Role-resolved audit trail for backend and UI consumption."""

  view_kind: AuditTrailViewKind
  entries: tuple[str, ...] = ()
  warnings: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "entries", tuple(self.entries))
    object.__setattr__(self, "warnings", tuple(self.warnings))


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_access(
    ctx: AccessContext,
    permission: Permission,
    *,
    role_permissions: RolePermissions | None = None,
) -> AccessDecision:
  """Check whether *ctx.role* holds *permission*.

  Returns an ``AccessDecision`` regardless of outcome. Callers should
  inspect ``decision.granted`` before proceeding.
  """
  rp = role_permissions or DEFAULT_ROLE_PERMISSIONS
  granted = rp.has_permission(ctx.role, permission)
  if granted:
    reason = f"Role {ctx.role.value} holds {permission.value}"
    audit_entry = (
        f"ACCESS GRANTED: caller={ctx.caller_id} role={ctx.role.value} "
        f"permission={permission.value}"
    )
  else:
    reason = f"Role {ctx.role.value} does not hold {permission.value}"
    audit_entry = (
        f"ACCESS DENIED: caller={ctx.caller_id} role={ctx.role.value} "
        f"permission={permission.value}"
    )
  if ctx.incident_id:
    audit_entry += f" incident={ctx.incident_id}"
  if ctx.evidence_id:
    audit_entry += f" evidence={ctx.evidence_id}"
  if ctx.reason:
    audit_entry += f" reason={ctx.reason!r}"
  return AccessDecision(
      granted=granted,
      permission=permission,
      role=ctx.role,
      caller_id=ctx.caller_id,
      reason=reason,
      audit_entry=audit_entry,
      timestamp=ctx.timestamp,
      incident_id=ctx.incident_id,
      evidence_id=ctx.evidence_id,
  )


def check_evidence_access(
    ctx: AccessContext,
    *,
    wants_unredacted: bool = False,
    is_export: bool = False,
    role_permissions: RolePermissions | None = None,
) -> AccessDecision:
  """Convenience check for evidence retrieval or export.

  Selects the appropriate permission based on the caller's intent
  (redacted vs unredacted, view vs export) and delegates to
  ``check_access``.
  """
  if is_export:
    permission = (
        Permission.EXPORT_UNREDACTED_EVIDENCE
        if wants_unredacted
        else Permission.EXPORT_REDACTED_EVIDENCE
    )
  else:
    permission = (
        Permission.VIEW_UNREDACTED_EVIDENCE
        if wants_unredacted
        else Permission.VIEW_REDACTED_EVIDENCE
    )
  return check_access(ctx, permission, role_permissions=role_permissions)


def check_workflow_action(
    ctx: AccessContext,
    action: str,
    *,
    role_permissions: RolePermissions | None = None,
) -> AccessDecision:
  """Check permission for a review workflow action by name.

  Supported action strings: ``approve``, ``reject``, ``escalate``.
  Unrecognized actions are denied by default.
  """
  permission_map: dict[str, Permission] = {
      "approve": Permission.APPROVE_INCIDENT,
      "reject": Permission.REJECT_INCIDENT,
      "escalate": Permission.ESCALATE_INCIDENT,
  }
  permission = permission_map.get(action.lower())
  if permission is None:
    return AccessDecision(
        granted=False,
        permission=None,
        role=ctx.role,
        caller_id=ctx.caller_id,
        reason=f"Unrecognized workflow action: {action!r}",
        audit_entry=(
            f"ACCESS DENIED: caller={ctx.caller_id} role={ctx.role.value} "
            f"action={action!r} (unrecognized)"
        ),
        timestamp=ctx.timestamp,
        incident_id=ctx.incident_id,
        evidence_id=ctx.evidence_id,
    )
  return check_access(ctx, permission, role_permissions=role_permissions)


def check_audit_trail_access(
    ctx: AccessContext,
    *,
    wants_full: bool = False,
    role_permissions: RolePermissions | None = None,
) -> AccessDecision:
  """Check permission for audit-trail visibility.

  ``wants_full=False`` checks basic audit visibility.
  ``wants_full=True`` checks full audit-trail visibility.
  """
  permission = (
      Permission.VIEW_FULL_AUDIT_TRAIL
      if wants_full
      else Permission.VIEW_AUDIT_TRAIL
  )
  return check_access(ctx, permission, role_permissions=role_permissions)


def audit_trail_view_for_role(
    role: ReviewRole,
    *,
    role_permissions: RolePermissions | None = None,
) -> AuditTrailViewKind:
  """Resolve the strongest audit-trail view a role may receive."""
  rp = role_permissions or DEFAULT_ROLE_PERMISSIONS
  if rp.has_permission(role, Permission.VIEW_FULL_AUDIT_TRAIL):
    return AuditTrailViewKind.FULL
  if rp.has_permission(role, Permission.VIEW_AUDIT_TRAIL):
    return AuditTrailViewKind.BASIC
  return AuditTrailViewKind.NONE


def resolve_audit_trail(
    audit_entries: tuple[str, ...] | list[str],
    viewer_role: ReviewRole,
    *,
    role_permissions: RolePermissions | None = None,
) -> AuditTrailView:
  """Return the audit-log view allowed for *viewer_role*.

  - ``FULL`` returns entries unchanged.
  - ``BASIC`` redacts sensitive access-decision details while preserving
    operational workflow steps.
  - ``NONE`` returns an empty entry set with a warning.
  """
  entries = tuple(audit_entries)
  view_kind = audit_trail_view_for_role(
      viewer_role,
      role_permissions=role_permissions,
  )
  if view_kind == AuditTrailViewKind.NONE:
    return AuditTrailView(
        view_kind=view_kind,
        entries=(),
        warnings=(
            f"Role {viewer_role.value} is not authorized to view audit trails.",
        ),
    )
  if view_kind == AuditTrailViewKind.FULL:
    return AuditTrailView(view_kind=view_kind, entries=entries)

  sanitized_entries = tuple(_sanitize_audit_entry(entry) for entry in entries)
  warnings: tuple[str, ...] = ()
  if sanitized_entries != entries:
    warnings = (
        "Access-decision details are redacted in the basic audit trail view.",
    )
  return AuditTrailView(
      view_kind=view_kind,
      entries=sanitized_entries,
      warnings=warnings,
  )


def _sanitize_audit_entry(entry: str) -> str:
  if not entry.startswith(("ACCESS GRANTED:", "ACCESS DENIED:")):
    return entry
  outcome = "GRANTED" if entry.startswith("ACCESS GRANTED:") else "DENIED"
  role = _extract_audit_field(entry, "role")
  permission = _extract_audit_field(entry, "permission")
  action = _extract_audit_field(entry, "action")
  details: list[str] = []
  if role:
    details.append(f"role={role}")
  if permission:
    details.append(f"permission={permission}")
  elif action:
    details.append(f"action={action}")
  if not details:
    details.append("details=redacted")
  return f"ACCESS {outcome}: {' '.join(details)}"


def _extract_audit_field(entry: str, field_name: str) -> str | None:
  match = re.search(rf"{field_name}=([^\s]+)", entry)
  if match is None:
    return None
  return match.group(1)
