"""Tests for trafficmind.review.privacy."""

import unittest

from trafficmind.review import AssetViewKind
from trafficmind.review import build_export_bundle
from trafficmind.review import build_playback_manifest
from trafficmind.review import DEFAULT_REDACTION_POLICY
from trafficmind.review import EvidenceAccessMode
from trafficmind.review import EvidenceAssetReference
from trafficmind.review import EvidenceExportBundle
from trafficmind.review import EvidenceManifest
from trafficmind.review import EvidenceMediaKind
from trafficmind.review import EvidencePlaybackManifest
from trafficmind.review import EvidenceReference
from trafficmind.review import MaskingOperation
from trafficmind.review import RedactionPolicy
from trafficmind.review import RedactionState
from trafficmind.review import ReviewRole
from trafficmind.review import SensitiveVisualDetail
from trafficmind.review import SensitiveVisualKind


def _asset(access_mode, storage_uri=None, **metadata):
  return EvidenceAssetReference(
      access_mode=access_mode,
      storage_uri=storage_uri,
      metadata=dict(metadata),
  )


def _detail(detail_id, kind, masking_operation):
  return SensitiveVisualDetail(
      detail_id=detail_id,
      kind=kind,
      masking_operation=masking_operation,
  )


class TestPrivacyPlaybackAndExport(unittest.TestCase):

  def _manifest(self):
    return EvidenceManifest(
        manifest_id="MAN-PRIV-1",
        incident_id="INC-PRIV-1",
        references=(
            EvidenceReference(
                evidence_id="frame-1",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-1-original.jpg",
                ),
                redacted_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-1-redacted.jpg",
                ),
                redaction_state=RedactionState.REDACTED_AVAILABLE,
                sensitive_details=(
                    _detail(
                        "face-1",
                        SensitiveVisualKind.FACE,
                        MaskingOperation.FACE_MASK,
                    ),
                    _detail(
                        "plate-1",
                        SensitiveVisualKind.PLATE,
                        MaskingOperation.PLATE_MASK,
                    ),
                ),
                label="Wide frame",
            ),
            EvidenceReference(
                evidence_id="clip-1",
                media_kind=EvidenceMediaKind.CLIP,
                access_mode=EvidenceAccessMode.STORED_REFERENCE,
                storage_uri="s3://bucket/clip-1.mp4",
                label="Unsensitive clip",
            ),
        ),
    )

  def test_operator_gets_redacted_playback_by_default(self):
    manifest = self._manifest()

    playback = build_playback_manifest(
        manifest,
        ReviewRole.OPERATOR,
        DEFAULT_REDACTION_POLICY,
    )

    self.assertIsInstance(playback, EvidencePlaybackManifest)
    self.assertEqual(playback.default_view, AssetViewKind.REDACTED)
    self.assertEqual(playback.entries[0].selected_view, AssetViewKind.REDACTED)
    self.assertEqual(
        playback.entries[0].selected_storage_uri,
        "s3://bucket/frame-1-redacted.jpg",
    )
    self.assertIn("masked-by-default playback", playback.warnings[0])

  def test_authorized_role_can_view_originals(self):
    manifest = self._manifest()

    playback = build_playback_manifest(
        manifest,
        ReviewRole.PRIVACY_REVIEWER,
        DEFAULT_REDACTION_POLICY,
    )

    self.assertEqual(playback.default_view, AssetViewKind.ORIGINAL)
    self.assertEqual(playback.entries[0].selected_view, AssetViewKind.ORIGINAL)
    self.assertEqual(
        playback.entries[0].selected_storage_uri,
        "s3://bucket/frame-1-original.jpg",
    )

  def test_missing_redacted_variant_falls_back_to_metadata_only_for_operator(
      self,
  ):
    manifest = EvidenceManifest(
        manifest_id="MAN-PRIV-2",
        incident_id="INC-PRIV-2",
        references=(
            EvidenceReference(
                evidence_id="frame-2",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-2-original.jpg",
                ),
                redaction_state=RedactionState.PENDING,
                sensitive_details=(
                    _detail(
                        "face-2",
                        SensitiveVisualKind.FACE,
                        MaskingOperation.FACE_MASK,
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

    self.assertIsNone(playback.entries[0].selected_view)
    self.assertEqual(
        playback.entries[0].selected_access_mode,
        EvidenceAccessMode.METADATA_ONLY,
    )
    self.assertIsNone(playback.entries[0].selected_storage_uri)
    self.assertTrue(playback.entries[0].warnings)

  def test_export_bundle_defaults_to_redacted(self):
    manifest = self._manifest()

    bundle = build_export_bundle(
        manifest,
        ReviewRole.PRIVACY_REVIEWER,
        DEFAULT_REDACTION_POLICY,
    )

    self.assertIsInstance(bundle, EvidenceExportBundle)
    self.assertEqual(bundle.export_view, AssetViewKind.REDACTED)
    self.assertFalse(bundle.includes_originals)
    self.assertEqual(bundle.entries[0].selected_view, AssetViewKind.REDACTED)
    self.assertIn("Export defaults to redacted assets", bundle.warnings[0])

  def test_export_bundle_rejects_originals_for_unauthorized_role(self):
    manifest = self._manifest()

    bundle = build_export_bundle(
        manifest,
        ReviewRole.OPERATOR,
        DEFAULT_REDACTION_POLICY,
        include_originals=True,
    )

    self.assertEqual(bundle.export_view, AssetViewKind.REDACTED)
    self.assertFalse(bundle.includes_originals)
    self.assertEqual(bundle.entries[0].selected_view, AssetViewKind.REDACTED)
    self.assertIn(
        "not authorized to export original assets", bundle.warnings[0]
    )

  def test_policy_can_disable_plate_masking(self):
    manifest = self._manifest()
    policy = RedactionPolicy(
        policy_id="policy-no-plate-mask",
        mask_faces=True,
        mask_plates=False,
        mask_personally_identifying_details=True,
    )

    playback = build_playback_manifest(
        manifest,
        ReviewRole.OPERATOR,
        policy,
    )

    self.assertEqual(playback.entries[0].selected_view, AssetViewKind.REDACTED)
    self.assertIn(
        SensitiveVisualKind.PLATE, playback.entries[0].sensitive_kinds
    )

  def test_role_not_in_either_policy_list_gets_redacted_for_sensitive_evidence(
      self,
  ):
    """A role absent from both mask_by_default_for_roles and
    unredacted_roles must not silently receive originals when
    evidence requires redaction."""
    policy = RedactionPolicy(
        policy_id="policy-gap-test",
        mask_by_default_for_roles=(ReviewRole.OPERATOR,),
        unredacted_roles=(ReviewRole.ADMIN,),
    )
    manifest = self._manifest()

    playback = build_playback_manifest(
        manifest,
        ReviewRole.SUPERVISOR,  # not in either list
        policy,
    )

    sensitive_entry = playback.entries[0]
    self.assertEqual(
        sensitive_entry.selected_view,
        AssetViewKind.REDACTED,
        "Unlisted role must not silently receive originals for sensitive"
        " evidence",
    )
    self.assertEqual(
        sensitive_entry.selected_storage_uri,
        "s3://bucket/frame-1-redacted.jpg",
    )
    # Non-sensitive evidence should still resolve to original normally.
    non_sensitive_entry = playback.entries[1]
    self.assertEqual(non_sensitive_entry.selected_view, AssetViewKind.ORIGINAL)

  def test_role_not_in_either_list_gets_metadata_only_when_redacted_missing(
      self,
  ):
    """Unlisted role with sensitive evidence but no redacted asset
    falls back to metadata-only under default policy."""
    policy = RedactionPolicy(
        policy_id="policy-gap-pending",
        mask_by_default_for_roles=(ReviewRole.OPERATOR,),
        unredacted_roles=(ReviewRole.ADMIN,),
    )
    manifest = EvidenceManifest(
        manifest_id="MAN-GAP",
        incident_id="INC-GAP",
        references=(
            EvidenceReference(
                evidence_id="frame-gap",
                media_kind=EvidenceMediaKind.FRAME,
                original_asset=_asset(
                    EvidenceAccessMode.ATTACHED_MEDIA,
                    "s3://bucket/frame-gap-original.jpg",
                ),
                redaction_state=RedactionState.PENDING,
                sensitive_details=(
                    _detail(
                        "face-gap",
                        SensitiveVisualKind.FACE,
                        MaskingOperation.FACE_MASK,
                    ),
                ),
            ),
        ),
    )

    playback = build_playback_manifest(
        manifest,
        ReviewRole.SUPERVISOR,  # not in either list
        policy,
    )

    entry = playback.entries[0]
    self.assertIsNone(entry.selected_view)
    self.assertEqual(
        entry.selected_access_mode, EvidenceAccessMode.METADATA_ONLY
    )
    self.assertTrue(entry.warnings)


if __name__ == "__main__":
  unittest.main()
