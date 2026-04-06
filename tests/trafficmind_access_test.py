"""Tests for trafficmind.review.access — permission model and enforcement."""

import unittest

from trafficmind.review.access import AccessContext
from trafficmind.review.access import AccessDecision
from trafficmind.review.access import audit_trail_view_for_role
from trafficmind.review.access import AuditTrailViewKind
from trafficmind.review.access import check_access
from trafficmind.review.access import check_audit_trail_access
from trafficmind.review.access import check_evidence_access
from trafficmind.review.access import check_workflow_action
from trafficmind.review.access import DEFAULT_ROLE_PERMISSIONS
from trafficmind.review.access import Permission
from trafficmind.review.access import resolve_audit_trail
from trafficmind.review.access import RolePermissions
from trafficmind.review.models import ReviewRole


class TestDefaultRolePermissions(unittest.TestCase):
  """Verify the default permission mapping is sensible."""

  def test_operator_can_view_redacted(self):
    self.assertTrue(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.OPERATOR,
            Permission.VIEW_REDACTED_EVIDENCE,
        )
    )

  def test_operator_cannot_view_unredacted(self):
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.OPERATOR,
            Permission.VIEW_UNREDACTED_EVIDENCE,
        )
    )

  def test_operator_cannot_export_unredacted(self):
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.OPERATOR,
            Permission.EXPORT_UNREDACTED_EVIDENCE,
        )
    )

  def test_operator_can_approve_reject_escalate(self):
    for perm in (
        Permission.APPROVE_INCIDENT,
        Permission.REJECT_INCIDENT,
        Permission.ESCALATE_INCIDENT,
    ):
      with self.subTest(perm=perm):
        self.assertTrue(
            DEFAULT_ROLE_PERMISSIONS.has_permission(ReviewRole.OPERATOR, perm)
        )

  def test_operator_cannot_manage_watchlists_or_policy(self):
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.OPERATOR,
            Permission.MANAGE_WATCHLISTS,
        )
    )
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.OPERATOR,
            Permission.MANAGE_POLICY,
        )
    )

  def test_supervisor_can_manage_watchlists(self):
    self.assertTrue(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.SUPERVISOR,
            Permission.MANAGE_WATCHLISTS,
        )
    )

  def test_auditor_can_view_full_audit_trail(self):
    self.assertTrue(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.AUDITOR,
            Permission.VIEW_FULL_AUDIT_TRAIL,
        )
    )

  def test_auditor_cannot_approve_incidents(self):
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.AUDITOR,
            Permission.APPROVE_INCIDENT,
        )
    )

  def test_privacy_reviewer_can_view_and_export_unredacted(self):
    for perm in (
        Permission.VIEW_UNREDACTED_EVIDENCE,
        Permission.EXPORT_UNREDACTED_EVIDENCE,
    ):
      with self.subTest(perm=perm):
        self.assertTrue(
            DEFAULT_ROLE_PERMISSIONS.has_permission(
                ReviewRole.PRIVACY_REVIEWER,
                perm,
            )
        )

  def test_privacy_reviewer_cannot_approve_or_manage_policy(self):
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.PRIVACY_REVIEWER,
            Permission.APPROVE_INCIDENT,
        )
    )
    self.assertFalse(
        DEFAULT_ROLE_PERMISSIONS.has_permission(
            ReviewRole.PRIVACY_REVIEWER,
            Permission.MANAGE_POLICY,
        )
    )

  def test_admin_has_all_permissions(self):
    for perm in Permission:
      with self.subTest(perm=perm):
        self.assertTrue(
            DEFAULT_ROLE_PERMISSIONS.has_permission(ReviewRole.ADMIN, perm),
            f"Admin should have {perm.value}",
        )


class TestCheckAccess(unittest.TestCase):

  def _ctx(self, role=ReviewRole.OPERATOR, **kwargs):
    return AccessContext(
        role=role,
        caller_id="test-user",
        incident_id="INC-1",
        **kwargs,
    )

  def test_granted_decision(self):
    ctx = self._ctx()
    decision = check_access(ctx, Permission.VIEW_REDACTED_EVIDENCE)
    self.assertTrue(decision.granted)
    self.assertIn("GRANTED", decision.audit_entry)
    self.assertIn("test-user", decision.audit_entry)
    self.assertIn("INC-1", decision.audit_entry)

  def test_denied_decision(self):
    ctx = self._ctx()
    decision = check_access(ctx, Permission.VIEW_UNREDACTED_EVIDENCE)
    self.assertFalse(decision.granted)
    self.assertIn("DENIED", decision.audit_entry)

  def test_audit_entry_includes_evidence_id(self):
    ctx = self._ctx(evidence_id="frame-1")
    decision = check_access(ctx, Permission.VIEW_REDACTED_EVIDENCE)
    self.assertIn("frame-1", decision.audit_entry)

  def test_audit_entry_includes_reason(self):
    ctx = self._ctx(reason="routine playback")
    decision = check_access(ctx, Permission.VIEW_REDACTED_EVIDENCE)
    self.assertIn("routine playback", decision.audit_entry)

  def test_custom_role_permissions(self):
    custom = RolePermissions(
        role_grants={
            ReviewRole.OPERATOR: frozenset({Permission.MANAGE_POLICY}),
        }
    )
    ctx = self._ctx()
    decision = check_access(
        ctx,
        Permission.MANAGE_POLICY,
        role_permissions=custom,
    )
    self.assertTrue(decision.granted)
    # Default permission should be absent in custom mapping.
    decision2 = check_access(
        ctx,
        Permission.VIEW_REDACTED_EVIDENCE,
        role_permissions=custom,
    )
    self.assertFalse(decision2.granted)

  def test_access_context_requires_caller_id(self):
    with self.assertRaises(ValueError):
      AccessContext(role=ReviewRole.OPERATOR, caller_id="")


class TestCheckEvidenceAccess(unittest.TestCase):

  def _ctx(self, role=ReviewRole.OPERATOR):
    return AccessContext(role=role, caller_id="user-1", incident_id="INC-1")

  def test_view_redacted(self):
    decision = check_evidence_access(self._ctx())
    self.assertTrue(decision.granted)
    self.assertEqual(decision.permission, Permission.VIEW_REDACTED_EVIDENCE)

  def test_view_unredacted_as_operator_denied(self):
    decision = check_evidence_access(self._ctx(), wants_unredacted=True)
    self.assertFalse(decision.granted)
    self.assertEqual(decision.permission, Permission.VIEW_UNREDACTED_EVIDENCE)

  def test_export_redacted(self):
    decision = check_evidence_access(self._ctx(), is_export=True)
    self.assertTrue(decision.granted)
    self.assertEqual(decision.permission, Permission.EXPORT_REDACTED_EVIDENCE)

  def test_export_unredacted_as_operator_denied(self):
    decision = check_evidence_access(
        self._ctx(),
        wants_unredacted=True,
        is_export=True,
    )
    self.assertFalse(decision.granted)
    self.assertEqual(decision.permission, Permission.EXPORT_UNREDACTED_EVIDENCE)

  def test_export_unredacted_as_privacy_reviewer_granted(self):
    decision = check_evidence_access(
        self._ctx(ReviewRole.PRIVACY_REVIEWER),
        wants_unredacted=True,
        is_export=True,
    )
    self.assertTrue(decision.granted)


class TestCheckWorkflowAction(unittest.TestCase):

  def _ctx(self, role=ReviewRole.OPERATOR):
    return AccessContext(role=role, caller_id="user-1", incident_id="INC-2")

  def test_approve_granted_for_operator(self):
    decision = check_workflow_action(self._ctx(), "approve")
    self.assertTrue(decision.granted)

  def test_approve_denied_for_auditor(self):
    decision = check_workflow_action(self._ctx(ReviewRole.AUDITOR), "approve")
    self.assertFalse(decision.granted)

  def test_escalate(self):
    decision = check_workflow_action(self._ctx(), "escalate")
    self.assertTrue(decision.granted)

  def test_unrecognized_action_denied(self):
    decision = check_workflow_action(self._ctx(), "delete")
    self.assertFalse(decision.granted)
    self.assertIsNone(decision.permission)
    self.assertIn("unrecognized", decision.audit_entry.lower())

  def test_case_insensitive(self):
    decision = check_workflow_action(self._ctx(), "Approve")
    self.assertTrue(decision.granted)


class TestAuditTrailAccess(unittest.TestCase):

  def _ctx(self, role=ReviewRole.OPERATOR):
    return AccessContext(role=role, caller_id="auditor-1", incident_id="INC-3")

  def test_basic_audit_access_granted_for_operator(self):
    decision = check_audit_trail_access(self._ctx())
    self.assertTrue(decision.granted)
    self.assertEqual(decision.permission, Permission.VIEW_AUDIT_TRAIL)

  def test_full_audit_access_denied_for_operator(self):
    decision = check_audit_trail_access(self._ctx(), wants_full=True)
    self.assertFalse(decision.granted)
    self.assertEqual(decision.permission, Permission.VIEW_FULL_AUDIT_TRAIL)

  def test_full_audit_access_granted_for_auditor(self):
    decision = check_audit_trail_access(
        self._ctx(ReviewRole.AUDITOR),
        wants_full=True,
    )
    self.assertTrue(decision.granted)

  def test_audit_trail_view_for_role(self):
    self.assertEqual(
        audit_trail_view_for_role(ReviewRole.OPERATOR),
        AuditTrailViewKind.BASIC,
    )
    self.assertEqual(
        audit_trail_view_for_role(ReviewRole.AUDITOR),
        AuditTrailViewKind.FULL,
    )

  def test_resolve_basic_audit_trail_redacts_access_details(self):
    view = resolve_audit_trail(
        (
            (
                "Grounded request inputs into event, rule, evidence, note, and"
                " prior-review references"
            ),
            (
                "ACCESS GRANTED: caller=user-1 role=operator"
                " permission=view_redacted_evidence incident=INC-1"
                " evidence=frame-1 reason='playback'"
            ),
        ),
        ReviewRole.OPERATOR,
    )
    self.assertEqual(view.view_kind, AuditTrailViewKind.BASIC)
    self.assertEqual(len(view.entries), 2)
    self.assertEqual(
        view.entries[1],
        "ACCESS GRANTED: role=operator permission=view_redacted_evidence",
    )
    self.assertNotIn("caller=", view.entries[1])
    self.assertTrue(view.warnings)

  def test_resolve_full_audit_trail_preserves_entries(self):
    entry = (
        "ACCESS DENIED: caller=user-1 role=auditor"
        " permission=view_unredacted_evidence incident=INC-1"
    )
    view = resolve_audit_trail((entry,), ReviewRole.AUDITOR)
    self.assertEqual(view.view_kind, AuditTrailViewKind.FULL)
    self.assertEqual(view.entries, (entry,))

  def test_resolve_denied_audit_trail_returns_empty(self):
    custom = RolePermissions(role_grants={ReviewRole.OPERATOR: frozenset()})
    view = resolve_audit_trail(
        (
            (
                "Generated structured review draft using deterministic workflow"
                " logic"
            ),
        ),
        ReviewRole.OPERATOR,
        role_permissions=custom,
    )
    self.assertEqual(view.view_kind, AuditTrailViewKind.NONE)
    self.assertEqual(view.entries, ())
    self.assertTrue(view.warnings)


class TestPlaybackManifestWithAccessContext(unittest.TestCase):
  """Integration: permission checks wired into build_playback_manifest."""

  def _manifest(self):
    from trafficmind.review.models import EvidenceAccessMode
    from trafficmind.review.models import EvidenceAssetReference
    from trafficmind.review.models import EvidenceManifest
    from trafficmind.review.models import EvidenceMediaKind
    from trafficmind.review.models import EvidenceReference
    from trafficmind.review.models import MaskingOperation
    from trafficmind.review.models import RedactionState
    from trafficmind.review.models import SensitiveVisualDetail
    from trafficmind.review.models import SensitiveVisualKind

    return EvidenceManifest(
        manifest_id="MAN-AC-1",
        incident_id="INC-AC-1",
        references=(
            EvidenceReference(
                evidence_id="frame-ac-1",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=EvidenceAssetReference(
                    access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                    storage_uri="s3://bucket/frame-ac-1-original.jpg",
                ),
                redacted_asset=EvidenceAssetReference(
                    access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                    storage_uri="s3://bucket/frame-ac-1-redacted.jpg",
                ),
                redaction_state=RedactionState.REDACTED_AVAILABLE,
                sensitive_details=(
                    SensitiveVisualDetail(
                        detail_id="face-ac",
                        kind=SensitiveVisualKind.FACE,
                        masking_operation=MaskingOperation.FACE_MASK,
                    ),
                ),
            ),
        ),
    )

  def test_playback_with_context_granted_includes_audit(self):
    from trafficmind.review.privacy import build_playback_manifest

    manifest = self._manifest()
    ctx = AccessContext(
        role=ReviewRole.OPERATOR,
        caller_id="op-1",
        incident_id="INC-AC-1",
    )
    playback = build_playback_manifest(
        manifest,
        ReviewRole.OPERATOR,
        access_context=ctx,
    )
    self.assertTrue(playback.access_audit)
    self.assertIn("GRANTED", playback.access_audit[0])
    self.assertEqual(len(playback.entries), 1)

  def test_playback_without_context_has_no_audit(self):
    from trafficmind.review.privacy import build_playback_manifest

    manifest = self._manifest()
    playback = build_playback_manifest(manifest, ReviewRole.OPERATOR)
    self.assertEqual(playback.access_audit, ())

  def test_playback_denied_returns_empty_entries(self):
    from trafficmind.review.privacy import build_playback_manifest

    manifest = self._manifest()
    # Operator viewing unredacted is not allowed; however the manifest
    # function checks the base capability for the role, not unredacted
    # specifically. To trigger denial, use a custom permission map
    # that strips all permissions.
    empty_perms = RolePermissions(role_grants={})
    ctx = AccessContext(
        role=ReviewRole.OPERATOR,
        caller_id="denied-user",
        incident_id="INC-AC-1",
    )
    playback = build_playback_manifest(
        manifest,
        ReviewRole.OPERATOR,
        access_context=ctx,
        role_permissions=empty_perms,
    )
    self.assertEqual(len(playback.entries), 0)
    self.assertIn("DENIED", playback.access_audit[0])
    self.assertTrue(playback.warnings)


class TestExportBundleWithAccessContext(unittest.TestCase):
  """Integration: permission checks wired into build_export_bundle."""

  def _manifest(self):
    from trafficmind.review.models import EvidenceAccessMode
    from trafficmind.review.models import EvidenceAssetReference
    from trafficmind.review.models import EvidenceManifest
    from trafficmind.review.models import EvidenceMediaKind
    from trafficmind.review.models import EvidenceReference
    from trafficmind.review.models import RedactionState

    return EvidenceManifest(
        manifest_id="MAN-EX-1",
        incident_id="INC-EX-1",
        references=(
            EvidenceReference(
                evidence_id="clip-ex-1",
                media_kind=EvidenceMediaKind.CLIP,
                original_asset=EvidenceAssetReference(
                    access_mode=EvidenceAccessMode.STORED_REFERENCE,
                    storage_uri="s3://bucket/clip-ex-1.mp4",
                ),
                redacted_asset=EvidenceAssetReference(
                    access_mode=EvidenceAccessMode.STORED_REFERENCE,
                    storage_uri="s3://bucket/clip-ex-1-redacted.mp4",
                ),
                redaction_state=RedactionState.REDACTED_AVAILABLE,
            ),
        ),
    )

  def test_export_with_context_granted(self):
    from trafficmind.review.privacy import build_export_bundle

    ctx = AccessContext(
        role=ReviewRole.OPERATOR,
        caller_id="op-1",
        incident_id="INC-EX-1",
    )
    bundle = build_export_bundle(
        self._manifest(),
        ReviewRole.OPERATOR,
        access_context=ctx,
    )
    self.assertTrue(bundle.access_audit)
    self.assertIn("GRANTED", bundle.access_audit[0])
    self.assertEqual(len(bundle.entries), 1)

  def test_export_denied_returns_empty_entries(self):
    from trafficmind.review.privacy import build_export_bundle

    empty_perms = RolePermissions(role_grants={})
    ctx = AccessContext(
        role=ReviewRole.OPERATOR,
        caller_id="denied-user",
        incident_id="INC-EX-1",
    )
    bundle = build_export_bundle(
        self._manifest(),
        ReviewRole.OPERATOR,
        access_context=ctx,
        role_permissions=empty_perms,
    )
    self.assertEqual(len(bundle.entries), 0)
    self.assertIn("DENIED", bundle.access_audit[0])
    self.assertFalse(bundle.includes_originals)


class TestRoleMismatchGuard(unittest.TestCase):
  """Ensure access_context.role must match viewer_role."""

  def _manifest(self):
    from trafficmind.review.models import EvidenceAccessMode
    from trafficmind.review.models import EvidenceAssetReference
    from trafficmind.review.models import EvidenceManifest
    from trafficmind.review.models import EvidenceMediaKind
    from trafficmind.review.models import EvidenceReference
    from trafficmind.review.models import RedactionState

    return EvidenceManifest(
        manifest_id="MAN-MIS-1",
        incident_id="INC-MIS-1",
        references=(
            EvidenceReference(
                evidence_id="frame-mis-1",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=EvidenceAssetReference(
                    access_mode=EvidenceAccessMode.ATTACHED_MEDIA,
                    storage_uri="s3://bucket/frame-mis-1.jpg",
                ),
                redaction_state=RedactionState.NOT_REQUIRED,
            ),
        ),
    )

  def test_playback_rejects_role_mismatch(self):
    from trafficmind.review.privacy import build_playback_manifest

    ctx = AccessContext(
        role=ReviewRole.ADMIN,
        caller_id="admin-1",
        incident_id="INC-MIS-1",
    )
    with self.assertRaises(ValueError) as cm:
      build_playback_manifest(
          self._manifest(),
          ReviewRole.OPERATOR,
          access_context=ctx,
      )
    self.assertIn("does not match", str(cm.exception))

  def test_export_rejects_role_mismatch(self):
    from trafficmind.review.privacy import build_export_bundle

    ctx = AccessContext(
        role=ReviewRole.ADMIN,
        caller_id="admin-1",
        incident_id="INC-MIS-1",
    )
    with self.assertRaises(ValueError) as cm:
      build_export_bundle(
          self._manifest(),
          ReviewRole.OPERATOR,
          access_context=ctx,
      )
    self.assertIn("does not match", str(cm.exception))

  def test_matching_roles_accepted(self):
    from trafficmind.review.privacy import build_playback_manifest

    ctx = AccessContext(
        role=ReviewRole.OPERATOR,
        caller_id="op-1",
        incident_id="INC-MIS-1",
    )
    playback = build_playback_manifest(
        self._manifest(),
        ReviewRole.OPERATOR,
        access_context=ctx,
    )
    self.assertTrue(playback.access_audit)


if __name__ == "__main__":
  unittest.main()
