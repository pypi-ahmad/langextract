"""Minimal local adapters for TrafficMind integration foundations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from trafficmind.integrations.models import CaseSyncAction
from trafficmind.integrations.models import CaseSyncResult
from trafficmind.integrations.models import CaseUpdate
from trafficmind.integrations.models import ObjectBlob
from trafficmind.integrations.models import ObjectPutRequest
from trafficmind.integrations.models import StoredObjectReference


class InMemoryCaseSystemAdapter:
  """Reference case adapter that stores the latest payload per incident.

  This adapter is intentionally simple and non-durable. It is useful for local
  demos, tests, and validating adapter wiring before a real external case
  system is chosen.
  """

  def __init__(self, *, name: str = "in_memory_case_system") -> None:
    self._name = name
    self._cases_by_incident: dict[str, CaseUpdate] = {}

  def adapter_name(self) -> str:
    return self._name

  @property
  def cases(self) -> dict[str, CaseUpdate]:
    return dict(self._cases_by_incident)

  def upsert_case(self, case_update: CaseUpdate) -> CaseSyncResult:
    existing = self._cases_by_incident.get(case_update.incident_id)
    external_case_id = (
        case_update.external_case_id
        or (existing.external_case_id if existing is not None else None)
        or f"local-case-{case_update.incident_id}"
    )
    stored = replace(case_update, external_case_id=external_case_id)
    self._cases_by_incident[case_update.incident_id] = stored
    return CaseSyncResult(
        adapter_name=self.adapter_name(),
        action=(
            CaseSyncAction.UPDATED
            if existing is not None
            else CaseSyncAction.CREATED
        ),
        incident_id=case_update.incident_id,
        external_case_id=external_case_id,
        metadata={"stored_case_count": len(self._cases_by_incident)},
    )


class LocalFilesystemObjectStore:
  """Reference object store rooted in one local filesystem directory.

  The file bytes are durable on disk. Content-type and metadata are kept in an
  in-memory catalog for convenience only.
  """

  def __init__(
      self,
      root_path: str | Path,
      *,
      name: str = "local_filesystem_object_store",
  ) -> None:
    self._name = name
    self._root_path = Path(root_path).resolve()
    self._root_path.mkdir(parents=True, exist_ok=True)
    self._catalog: dict[str, StoredObjectReference] = {}

  def adapter_name(self) -> str:
    return self._name

  @property
  def root_path(self) -> Path:
    return self._root_path

  def put_object(self, request: ObjectPutRequest) -> StoredObjectReference:
    target = self._resolve_object_name(request.object_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(request.content)
    reference = StoredObjectReference(
        object_name=target.relative_to(self._root_path).as_posix(),
        storage_uri=target.as_uri(),
        size_bytes=len(request.content),
        content_type=request.content_type,
        metadata=request.metadata,
    )
    self._catalog[reference.storage_uri] = reference
    return reference

  def get_object(self, storage_uri: str) -> ObjectBlob:
    path = self._resolve_storage_uri(storage_uri)
    if not path.exists():
      raise FileNotFoundError(path)
    content = path.read_bytes()
    storage_key = path.as_uri()
    known_reference = self._catalog.get(storage_key)
    if known_reference is None:
      reference = StoredObjectReference(
          object_name=path.relative_to(self._root_path).as_posix(),
          storage_uri=storage_key,
          size_bytes=len(content),
      )
    elif known_reference.size_bytes == len(content):
      reference = known_reference
    else:
      reference = replace(known_reference, size_bytes=len(content))
      self._catalog[storage_key] = reference
    return ObjectBlob(reference=reference, content=content)

  def _resolve_object_name(self, object_name: str) -> Path:
    candidate = Path(object_name)
    if candidate.is_absolute():
      raise ValueError("object_name must be relative to the store root")
    resolved = (self._root_path / candidate).resolve()
    if resolved != self._root_path and self._root_path not in resolved.parents:
      raise ValueError("object_name must stay within the store root")
    return resolved

  def _resolve_storage_uri(self, storage_uri: str) -> Path:
    parsed = urlparse(storage_uri)
    if parsed.scheme not in ("", "file"):
      raise ValueError("storage_uri must be a local path or file:// URI")
    if parsed.scheme == "file":
      raw_path = url2pathname(parsed.path)
      if parsed.netloc:
        raw_path = f"//{parsed.netloc}{raw_path}"
      candidate = Path(raw_path)
    else:
      candidate = Path(storage_uri)
    resolved = candidate.resolve()
    if resolved != self._root_path and self._root_path not in resolved.parents:
      raise ValueError("storage_uri must stay within the store root")
    return resolved
