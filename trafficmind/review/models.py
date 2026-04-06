"""Typed models for TrafficMind operator review workflows.

These models intentionally separate:
- event and rule metadata
- evidence manifest references
- operator-provided context
- workflow grounding references
- typed workflow outputs

The workflow can therefore stay auditable and explicit about whether a claim is
based on structured metadata, stored-but-unattached evidence references, or
actual attached media references.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Any, TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
  from trafficmind.registry.models import ProvenanceChain
  from trafficmind.review.access import AuditTrailView


class EvidenceMediaKind(str, enum.Enum):
  """Supported evidence asset categories."""

  FRAME = "frame"
  CROP = "crop"
  CLIP = "clip"
  IMAGE = "image"
  VIDEO = "video"
  DOCUMENT = "document"
  OTHER = "other"


class EvidenceAccessMode(str, enum.Enum):
  """How much of an evidence asset is available to the review workflow."""

  ATTACHED_MEDIA = "attached_media"
  STORED_REFERENCE = "stored_reference"
  METADATA_ONLY = "metadata_only"


class ReviewRole(str, enum.Enum):
  """Role consuming evidence during review, playback, or export."""

  OPERATOR = "operator"
  SUPERVISOR = "supervisor"
  PRIVACY_REVIEWER = "privacy_reviewer"
  ADMIN = "admin"
  AUDITOR = "auditor"


class SensitiveVisualKind(str, enum.Enum):
  """Sensitive visual details that may require masking or redaction."""

  FACE = "face"
  PLATE = "plate"
  PERSONALLY_IDENTIFYING_DETAIL = "personally_identifying_detail"


class MaskingOperation(str, enum.Enum):
  """Foundation masking operations supported by privacy-aware evidence flows."""

  FACE_MASK = "face_mask"
  PLATE_MASK = "plate_mask"
  DETAIL_MASK = "detail_mask"


class AssetViewKind(str, enum.Enum):
  """Which asset lineage a consumer is using."""

  ORIGINAL = "original"
  REDACTED = "redacted"


class RedactionState(str, enum.Enum):
  """Current masking / redaction state for one logical evidence item."""

  NOT_REQUIRED = "not_required"
  PENDING = "pending"
  REDACTED_AVAILABLE = "redacted_available"


class GroundingSourceKind(str, enum.Enum):
  """Source categories used in grounded workflow outputs."""

  EVENT_METADATA = "event_metadata"
  RULE_EXPLANATION = "rule_explanation"
  EVIDENCE_METADATA = "evidence_metadata"
  ATTACHED_MEDIA = "attached_media"
  OPERATOR_NOTE = "operator_note"
  PRIOR_REVIEW = "prior_review"


@dataclass(frozen=True)
class SensitiveVisualDetail:
  """One sensitive visual detail that may require masking.

  Bounding boxes are optional because this foundation models policy and
  provenance, not a full masking engine.
  """

  detail_id: str
  kind: SensitiveVisualKind
  masking_operation: MaskingOperation
  bounding_box: tuple[float, float, float, float] | None = None
  confidence: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.detail_id:
      raise ValueError("detail_id must be non-empty")
    if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
      raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class EvidenceAssetReference:
  """One concrete asset variant for a logical evidence item.

  Original and redacted assets stay in separate fields on `EvidenceReference`
  so callers never need to overwrite provenance when a redacted variant is
  produced.
  """

  access_mode: EvidenceAccessMode
  storage_uri: str | None = None
  created_at: float | None = None
  created_by: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if (
        self.access_mode != EvidenceAccessMode.METADATA_ONLY
        and not self.storage_uri
    ):
      raise ValueError(
          "storage_uri is required when asset access is more than metadata-only"
      )


@dataclass(frozen=True)
class RedactionPolicy:
  """Configurable role-based masking and export policy.

  This policy does not claim jurisdiction-specific compliance. It only models
  which sensitive detail kinds should be masked and which roles may retain
  access to original assets.
  """

  policy_id: str
  mask_faces: bool = True
  mask_plates: bool = True
  mask_personally_identifying_details: bool = True
  mask_by_default_for_roles: tuple[ReviewRole, ...] = (
      ReviewRole.OPERATOR,
      ReviewRole.SUPERVISOR,
      ReviewRole.AUDITOR,
  )
  unredacted_roles: tuple[ReviewRole, ...] = (
      ReviewRole.PRIVACY_REVIEWER,
      ReviewRole.ADMIN,
  )
  export_redacted_by_default: bool = True
  metadata_only_when_redaction_missing: bool = True

  def __post_init__(self) -> None:
    if not self.policy_id:
      raise ValueError("policy_id must be non-empty")
    object.__setattr__(
        self,
        "mask_by_default_for_roles",
        tuple(self.mask_by_default_for_roles),
    )
    object.__setattr__(
        self,
        "unredacted_roles",
        tuple(self.unredacted_roles),
    )
    overlapping_roles = set(self.mask_by_default_for_roles) & set(
        self.unredacted_roles
    )
    if overlapping_roles:
      overlap_text = ", ".join(
          role.value
          for role in sorted(overlapping_roles, key=lambda role: role.value)
      )
      raise ValueError(
          "mask_by_default_for_roles and unredacted_roles must be disjoint; "
          f"overlap={overlap_text}"
      )

  def requires_redaction(self, sensitive_kind: SensitiveVisualKind) -> bool:
    if sensitive_kind == SensitiveVisualKind.FACE:
      return self.mask_faces
    if sensitive_kind == SensitiveVisualKind.PLATE:
      return self.mask_plates
    return self.mask_personally_identifying_details


@dataclass(frozen=True)
class ReviewEvent:
  """Structured event or violation metadata submitted for operator review."""

  incident_id: str
  event_type: str
  occurred_at: float
  junction_id: str
  phase_id: str | None = None
  violation_type: str | None = None
  title: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  provenance: ProvenanceChain | None = None

  def __post_init__(self) -> None:
    if not self.incident_id:
      raise ValueError("incident_id must be non-empty")
    if not self.event_type:
      raise ValueError("event_type must be non-empty")
    if not self.junction_id:
      raise ValueError("junction_id must be non-empty")


@dataclass(frozen=True)
class RuleExplanation:
  """Deterministic rule context attached to a review request."""

  rule_id: str
  explanation: str
  title: str | None = None
  triggered_conditions: tuple[str, ...] = ()
  unresolved_conditions: tuple[str, ...] = ()
  deterministic_basis: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if not self.rule_id:
      raise ValueError("rule_id must be non-empty")
    if not self.explanation:
      raise ValueError("explanation must be non-empty")
    object.__setattr__(
        self, "triggered_conditions", tuple(self.triggered_conditions)
    )
    object.__setattr__(
        self, "unresolved_conditions", tuple(self.unresolved_conditions)
    )
    object.__setattr__(
        self, "deterministic_basis", tuple(self.deterministic_basis)
    )


@dataclass(frozen=True)
class EvidenceReference:
  """One evidence item from a manifest.

  `ATTACHED_MEDIA` means the actual image or clip is available to the review
  workflow or downstream multimodal assistant.
  `STORED_REFERENCE` means a durable URI/path exists but the media was not
  attached to this specific review invocation.
  `METADATA_ONLY` means the manifest contains a record of the asset without a
  retrievable media payload.
  """

  evidence_id: str
  media_kind: EvidenceMediaKind
  access_mode: EvidenceAccessMode | None = None
  storage_uri: str | None = None
  original_asset: EvidenceAssetReference | None = None
  redacted_asset: EvidenceAssetReference | None = None
  redaction_state: RedactionState = RedactionState.NOT_REQUIRED
  sensitive_details: tuple[SensitiveVisualDetail, ...] = ()
  label: str | None = None
  description: str | None = None
  observed_at: float | None = None
  clip_start: float | None = None
  clip_end: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  provenance: ProvenanceChain | None = None

  def __post_init__(self) -> None:
    if not self.evidence_id:
      raise ValueError("evidence_id must be non-empty")
    original_asset = self.original_asset
    if original_asset is None and self.access_mode is not None:
      original_asset = EvidenceAssetReference(
          access_mode=self.access_mode,
          storage_uri=self.storage_uri,
      )
      object.__setattr__(self, "original_asset", original_asset)
    elif original_asset is not None and self.access_mode is not None:
      if self.access_mode != original_asset.access_mode:
        raise ValueError(
            "legacy access_mode must match original_asset.access_mode when both"
            " are provided"
        )
      if (
          self.storage_uri is not None
          and self.storage_uri != original_asset.storage_uri
      ):
        raise ValueError(
            "legacy storage_uri must match original_asset.storage_uri when both"
            " are provided"
        )
    elif original_asset is not None and self.access_mode is None:
      object.__setattr__(self, "access_mode", original_asset.access_mode)
      object.__setattr__(self, "storage_uri", original_asset.storage_uri)

    if original_asset is None and self.redacted_asset is None:
      raise ValueError(
          "at least one of original_asset, redacted_asset, or legacy"
          " access_mode/storage_uri must be provided"
      )
    if self.access_mode is None:
      object.__setattr__(
          self,
          "access_mode",
          self.best_available_access_mode,
      )
    if self.storage_uri is None:
      object.__setattr__(
          self,
          "storage_uri",
          self.best_available_storage_uri,
      )
    object.__setattr__(self, "sensitive_details", tuple(self.sensitive_details))
    if (
        self.redaction_state == RedactionState.REDACTED_AVAILABLE
        and self.redacted_asset is None
    ):
      raise ValueError(
          "redacted_asset is required when redaction_state is"
          " REDACTED_AVAILABLE"
      )
    if (
        self.clip_start is not None
        and self.clip_end is not None
        and self.clip_end < self.clip_start
    ):
      raise ValueError("clip_end must be greater than or equal to clip_start")

  @property
  def has_sensitive_details(self) -> bool:
    return bool(self.sensitive_details)

  @property
  def best_available_access_mode(self) -> EvidenceAccessMode:
    # Reports the highest-capability access mode across ALL asset
    # variants without policy filtering.  Role-aware access resolution
    # happens in the privacy resolver, not here.
    candidate_modes = [
        asset.access_mode
        for asset in (self.original_asset, self.redacted_asset)
        if asset is not None
    ]
    if EvidenceAccessMode.ATTACHED_MEDIA in candidate_modes:
      return EvidenceAccessMode.ATTACHED_MEDIA
    if EvidenceAccessMode.STORED_REFERENCE in candidate_modes:
      return EvidenceAccessMode.STORED_REFERENCE
    return EvidenceAccessMode.METADATA_ONLY

  @property
  def best_available_storage_uri(self) -> str | None:
    # Prefer redacted URI when both exist so legacy consumers that read
    # ``storage_uri`` directly do not inadvertently surface originals.
    # Original URIs remain available via ``original_asset.storage_uri``.
    for asset in (self.redacted_asset, self.original_asset):
      if asset is not None and asset.storage_uri is not None:
        return asset.storage_uri
    return None

  def asset_for_view(
      self,
      view_kind: AssetViewKind,
  ) -> EvidenceAssetReference | None:
    if view_kind == AssetViewKind.ORIGINAL:
      return self.original_asset
    return self.redacted_asset


@dataclass(frozen=True)
class EvidenceManifest:
  """Evidence manifest references associated with an incident review."""

  manifest_id: str
  incident_id: str
  references: tuple[EvidenceReference, ...] = ()
  privacy_policy_id: str | None = None
  default_export_view: AssetViewKind = AssetViewKind.REDACTED
  generated_at: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.manifest_id:
      raise ValueError("manifest_id must be non-empty")
    if not self.incident_id:
      raise ValueError("incident_id must be non-empty")
    object.__setattr__(self, "references", tuple(self.references))

  @property
  def contains_sensitive_evidence(self) -> bool:
    return any(reference.has_sensitive_details for reference in self.references)


@dataclass(frozen=True)
class OperatorNote:
  """Free-form operator notes added to a review."""

  author: str
  note: str
  created_at: float | None = None
  note_id: str | None = None

  def __post_init__(self) -> None:
    if not self.author:
      raise ValueError("author must be non-empty")
    if not self.note:
      raise ValueError("note must be non-empty")


@dataclass(frozen=True)
class PriorReviewEntry:
  """Earlier review activity that should inform a current incident review."""

  reviewer: str
  reviewed_at: float
  summary: str
  decision: str
  action_taken: str | None = None
  escalation_outcome: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.reviewer:
      raise ValueError("reviewer must be non-empty")
    if not self.summary:
      raise ValueError("summary must be non-empty")
    if not self.decision:
      raise ValueError("decision must be non-empty")


@dataclass(frozen=True)
class GroundingReference:
  """One auditable reference used to support a workflow output claim."""

  source_kind: GroundingSourceKind
  source_id: str
  label: str
  detail: str
  access_mode: EvidenceAccessMode | None = None
  media_kind: EvidenceMediaKind | None = None
  selected_view: AssetViewKind | None = None
  redaction_state: RedactionState | None = None


@dataclass(frozen=True)
class GroundedNarrative:
  """A narrative section paired with explicit supporting references."""

  text: str
  references: tuple[GroundingReference, ...] = ()

  def __post_init__(self) -> None:
    if not self.text:
      raise ValueError("text must be non-empty")
    object.__setattr__(self, "references", tuple(self.references))


@dataclass(frozen=True)
class EvidenceInventory:
  """Count summary for a review request's evidence manifest."""

  total_references: int
  attached_media_count: int
  stored_reference_count: int
  metadata_only_count: int
  original_asset_count: int = 0
  redacted_asset_count: int = 0
  media_kind_counts: dict[str, int] = field(default_factory=dict)
  redaction_state_counts: dict[str, int] = field(default_factory=dict)
  sensitive_kind_counts: dict[str, int] = field(default_factory=dict)

  @property
  def has_attached_media(self) -> bool:
    return self.attached_media_count > 0

  @property
  def has_redacted_assets(self) -> bool:
    return self.redacted_asset_count > 0


@dataclass(frozen=True)
class PresentedEvidenceReference:
  """Role-resolved evidence view for playback, export, or assistant prompts."""

  evidence_id: str
  media_kind: EvidenceMediaKind
  viewer_role: ReviewRole
  label: str | None = None
  description: str | None = None
  observed_at: float | None = None
  clip_start: float | None = None
  clip_end: float | None = None
  selected_view: AssetViewKind | None = None
  selected_access_mode: EvidenceAccessMode = EvidenceAccessMode.METADATA_ONLY
  selected_storage_uri: str | None = None
  redaction_state: RedactionState = RedactionState.NOT_REQUIRED
  original_available: bool = False
  redacted_available: bool = False
  sensitive_kinds: tuple[SensitiveVisualKind, ...] = ()
  access_boundary: str = ""
  warnings: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "sensitive_kinds", tuple(self.sensitive_kinds))
    object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True)
class EvidencePlaybackManifest:
  """Frontend-friendly playback manifest with explicit masking state."""

  manifest_id: str
  incident_id: str
  viewer_role: ReviewRole
  policy_id: str
  default_view: AssetViewKind
  entries: tuple[PresentedEvidenceReference, ...] = ()
  warnings: tuple[str, ...] = ()
  access_audit: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "entries", tuple(self.entries))
    object.__setattr__(self, "warnings", tuple(self.warnings))
    object.__setattr__(self, "access_audit", tuple(self.access_audit))


@dataclass(frozen=True)
class EvidenceExportBundle:
  """Role-resolved export bundle description.

  This is an export plan / manifest, not a file writer. The foundation keeps
  export defaults explicit without claiming jurisdiction-specific compliance.
  """

  manifest_id: str
  incident_id: str
  viewer_role: ReviewRole
  policy_id: str
  export_view: AssetViewKind
  includes_originals: bool
  entries: tuple[PresentedEvidenceReference, ...] = ()
  warnings: tuple[str, ...] = ()
  access_audit: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "entries", tuple(self.entries))
    object.__setattr__(self, "warnings", tuple(self.warnings))
    object.__setattr__(self, "access_audit", tuple(self.access_audit))


@dataclass(frozen=True)
class GroundedReviewContext:
  """Normalized, auditable inputs prepared for the review workflow."""

  evidence_inventory: EvidenceInventory
  event_references: tuple[GroundingReference, ...] = ()
  rule_references: tuple[GroundingReference, ...] = ()
  evidence_references: tuple[GroundingReference, ...] = ()
  attached_media_references: tuple[GroundingReference, ...] = ()
  operator_note_references: tuple[GroundingReference, ...] = ()
  prior_review_references: tuple[GroundingReference, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "event_references", tuple(self.event_references))
    object.__setattr__(self, "rule_references", tuple(self.rule_references))
    object.__setattr__(
        self, "evidence_references", tuple(self.evidence_references)
    )
    object.__setattr__(
        self,
        "attached_media_references",
        tuple(self.attached_media_references),
    )
    object.__setattr__(
        self,
        "operator_note_references",
        tuple(self.operator_note_references),
    )
    object.__setattr__(
        self,
        "prior_review_references",
        tuple(self.prior_review_references),
    )

  @property
  def all_references(self) -> tuple[GroundingReference, ...]:
    return (
        self.event_references
        + self.rule_references
        + self.evidence_references
        + self.attached_media_references
        + self.operator_note_references
        + self.prior_review_references
    )


@dataclass(frozen=True)
class ReviewPromptBundle:
  """Prompt material prepared for a multimodal review assistant."""

  system_prompt: str
  user_prompt: str
  attachment_evidence: tuple[EvidenceReference, ...] = ()
  attachment_views: tuple[PresentedEvidenceReference, ...] = ()
  playback_manifest: EvidencePlaybackManifest | None = None
  viewer_role: ReviewRole | None = None
  policy_id: str | None = None
  context: GroundedReviewContext | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
        self, "attachment_evidence", tuple(self.attachment_evidence)
    )
    object.__setattr__(self, "attachment_views", tuple(self.attachment_views))


@dataclass(frozen=True)
class ReviewDraft:
  """Structured review draft produced by a deterministic or model-backed node."""

  review_summary: GroundedNarrative
  likely_cause: GroundedNarrative
  confidence_caveats: tuple[GroundedNarrative, ...]
  recommended_operator_action: GroundedNarrative
  escalation_suggestion: GroundedNarrative

  def __post_init__(self) -> None:
    object.__setattr__(
        self, "confidence_caveats", tuple(self.confidence_caveats)
    )


@dataclass(frozen=True)
class MultimodalReviewRequest:
  """Full input payload for an operator incident review."""

  event: ReviewEvent
  rule_explanation: RuleExplanation
  evidence_manifest: EvidenceManifest
  operator_notes: tuple[OperatorNote, ...] = ()
  prior_review_history: tuple[PriorReviewEntry, ...] = ()
  viewer_role: ReviewRole = ReviewRole.OPERATOR
  redaction_policy: RedactionPolicy | None = None

  def __post_init__(self) -> None:
    if self.evidence_manifest.incident_id != self.event.incident_id:
      raise ValueError(
          "evidence_manifest.incident_id must match event.incident_id"
      )
    object.__setattr__(self, "operator_notes", tuple(self.operator_notes))
    object.__setattr__(
        self, "prior_review_history", tuple(self.prior_review_history)
    )


@dataclass(frozen=True)
class MultimodalReviewResult:
  """Typed output of the operator review workflow."""

  incident_id: str
  event_type: str
  workflow_version: str
  used_langgraph: bool
  used_assistant_model: bool
  evidence_inventory: EvidenceInventory
  review_summary: GroundedNarrative
  likely_cause: GroundedNarrative
  confidence_caveats: tuple[GroundedNarrative, ...]
  recommended_operator_action: GroundedNarrative
  escalation_suggestion: GroundedNarrative
  playback_manifest: EvidencePlaybackManifest | None = None
  audit_trail: AuditTrailView | None = None
  audit_log: tuple[str, ...] = ()
  pipeline_snapshot_id: str | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
        self, "confidence_caveats", tuple(self.confidence_caveats)
    )
    object.__setattr__(self, "audit_log", tuple(self.audit_log))


class ReviewWorkflowState(TypedDict, total=False):
  """Typed LangGraph state for the operator review workflow."""

  request: MultimodalReviewRequest
  evidence_inventory: EvidenceInventory
  grounding_context: GroundedReviewContext
  playback_manifest: EvidencePlaybackManifest
  prompt_bundle: ReviewPromptBundle
  draft: ReviewDraft
  result: MultimodalReviewResult
  used_assistant_model: bool
  audit_log: list[str]
