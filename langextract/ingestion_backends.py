# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ingestion backend registry for LangExtract.

This module provides a registry of parsing/ingestion backends organized by
file or input category. Each category (e.g. ``readable_pdf``, ``csv``,
``image``) can have multiple registered backends, with exactly one marked as
the current default.

Backends are registered with their availability status:

- ``AVAILABLE``: Fully wired into the ingestion pipeline. The existing
  implementation in ``ingestion.py`` uses this backend when selected (or
  when no explicit backend is requested and it is the default).
- ``REGISTERED``: Known to the registry but **not yet wired** into the
  pipeline. Selecting a ``REGISTERED`` backend will raise a clear error.

Usage::

    from langextract.ingestion_backends import (
        FileCategory,
        get_default_backend,
        get_backends,
        is_backend_installed,
    )

    # List backends for readable PDFs
    backends = get_backends(FileCategory.READABLE_PDF)

    # Check whether the default backend's library is installed
    default = get_default_backend(FileCategory.READABLE_PDF)
    print(default.name, is_backend_installed(default))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import importlib
import shutil
from typing import Any, Callable

__all__ = [
    "BackendCategoryInfo",
    "BackendDiagnostics",
    "BackendOptionInfo",
    "BackendStatus",
    "FileCategory",
    "IngestionBackend",
    "ParserBackendOptions",
    "find_backend_categories",
    "inspect_backend_runtime",
    "get_backend",
    "get_backends",
    "get_default_backend",
    "get_handler",
    "is_backend_installed",
    "list_available_backends",
    "list_all_backends",
    "list_categories",
    "register_handler",
    "require_backend",
    "resolve_backend",
    "resolve_handler",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FileCategory(str, enum.Enum):
  """Logical input/file categories for ingestion."""

  READABLE_PDF = "readable_pdf"
  SCANNED_PDF = "scanned_pdf"
  IMAGE = "image"
  TABLE_PDF = "table_pdf"
  DOCX = "docx"
  DOC = "doc"
  TXT = "txt"
  CSV = "csv"
  EXCEL = "excel"
  PPT = "ppt"
  PPTX = "pptx"
  HTML = "html"
  URL = "url"

  def __str__(self) -> str:
    return self.value


class BackendStatus(str, enum.Enum):
  """Wiring status of an ingestion backend.

  - ``AVAILABLE``: The backend is fully wired into ``ingestion.py``.
  - ``REGISTERED``: The backend is known to the registry but not yet
    implemented in the ingestion pipeline. Attempting to select it at
    runtime will raise an error.
  """

  AVAILABLE = "available"
  REGISTERED = "registered"

  def __str__(self) -> str:
    return self.value


# ---------------------------------------------------------------------------
# Backend descriptor
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class IngestionBackend:
  """Metadata for a single ingestion/parsing backend.

  Attributes:
    category: The file/input category this backend handles.
    name: Short unique name (e.g. ``"pymupdf"``, ``"pandas"``).
    pypi_package: PyPI package name to install, if any.
    import_check: Top-level Python module used to verify the library is
      installed (e.g. ``"pymupdf"``). ``None`` means the backend is always
      importable (built-in or stdlib).
    status: Whether the backend is fully wired (``AVAILABLE``) or only
      registered in the catalogue (``REGISTERED``).
    is_default: Whether this is the default backend for its category.
    description: One-line human-readable description.
    extras: Optional ``pyproject.toml`` extras that install this backend's
      dependency family (for example ``("pdf",)`` or ``("office", "xlsx")``).
  """

  category: FileCategory
  name: str
  pypi_package: str | None = None
  import_check: str | None = None
  status: BackendStatus = BackendStatus.REGISTERED
  is_default: bool = False
  description: str = ""
  extras: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class BackendDiagnostics:
  """Runtime-oriented availability information for a backend.

  Attributes:
    backend: The backend descriptor being inspected.
    installed: Whether the backend's primary dependency appears present in the
      current environment. ``None`` means runtime inspection is not available
      for that backend shape.
    usable: Whether the backend is both implemented and appears usable in the
      current environment based on known runtime requirements.
    unavailable_reason: Human-readable reason when the backend is not usable.
  """

  backend: IngestionBackend
  installed: bool | None
  usable: bool
  unavailable_reason: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class BackendOptionInfo:
  """Developer-friendly backend introspection details for one backend.

  This is intended for lightweight discovery in Python, for example via
  ``langextract.list_available_backends()``.
  """

  name: str
  description: str
  implementation: str
  installed: bool | None
  usable: bool
  default: bool
  auto_preference_rank: int | None
  extras: tuple[str, ...] = ()
  install_commands: tuple[str, ...] = ()
  reason: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class BackendCategoryInfo:
  """Developer-friendly backend introspection details for one category."""

  category: FileCategory
  default: str | None
  auto_preference: tuple[str, ...] | None
  backends: tuple[BackendOptionInfo, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ParserBackendOptions:
  """Optional parser backend selections keyed by file/input category.

  All fields are optional. Leaving a field unset preserves the current default
  behavior for that category. For supported categories, setting a backend field
  to ``"auto"`` enables deterministic fallback through a per-category backend
  preference order. Preference order can be overridden with
  ``backend_preference_order``.
  """

  readable_pdf_backend: str | None = None
  scanned_pdf_backend: str | None = None
  image_backend: str | None = None
  table_pdf_backend: str | None = None
  docx_backend: str | None = None
  doc_backend: str | None = None
  txt_backend: str | None = None
  csv_backend: str | None = None
  excel_backend: str | None = None
  ppt_backend: str | None = None
  pptx_backend: str | None = None
  html_backend: str | None = None
  url_backend: str | None = None
  backend_preference_order: Mapping[FileCategory | str, Sequence[str]] = (
      dataclasses.field(default_factory=dict)
  )

  def __post_init__(self) -> None:
    normalized_preferences = _normalize_backend_preference_order(
        self.backend_preference_order
    )
    object.__setattr__(self, "backend_preference_order", normalized_preferences)

    for category, field_name in _PARSER_BACKEND_FIELDS.items():
      backend_name = getattr(self, field_name)
      if backend_name is None:
        continue
      if _is_auto_backend_name(backend_name):
        if _supports_auto_backend(category):
          continue
        raise ValueError(
            "Automatic backend selection is not supported for "
            f"{category.value!r}."
        )
      if get_backend(category, backend_name) is not None:
        continue
      allowed = [backend.name for backend in get_backends(category)]
      if _supports_auto_backend(category):
        allowed.append(_AUTO_BACKEND_NAME)
      raise ValueError(
          f"Invalid backend {backend_name!r} for {category.value!r}. "
          f"Allowed backends: {allowed}"
      )

  def backend_for_category(self, category: FileCategory) -> str | None:
    """Return the configured backend name for *category*, if any."""
    return getattr(self, _PARSER_BACKEND_FIELDS[category])

  def preference_order_for_category(
      self,
      category: FileCategory,
  ) -> tuple[str, ...] | None:
    """Return the merged auto-selection preference order for *category*.

    The result always starts with any user-configured preferences, followed
    by the curated defaults, followed by every registered backend name.
    Duplicates are removed while preserving first-occurrence order.
    """
    if not _supports_auto_backend(category):
      return None

    configured = self.backend_preference_order.get(category)
    merged = list(configured or ())
    merged.extend(_AUTO_BACKEND_PREFERENCE_DEFAULTS[category])
    merged.extend(backend.name for backend in get_backends(category))
    return tuple(_dedupe_preserving_order(merged))

  def selections(self) -> dict[FileCategory, str]:
    """Return the explicitly configured backend selections."""
    return {
        category: backend_name
        for category, field_name in _PARSER_BACKEND_FIELDS.items()
        if (backend_name := getattr(self, field_name)) is not None
    }


_PARSER_BACKEND_FIELDS: dict[FileCategory, str] = {
    FileCategory.READABLE_PDF: "readable_pdf_backend",
    FileCategory.SCANNED_PDF: "scanned_pdf_backend",
    FileCategory.IMAGE: "image_backend",
    FileCategory.TABLE_PDF: "table_pdf_backend",
    FileCategory.DOCX: "docx_backend",
    FileCategory.DOC: "doc_backend",
    FileCategory.TXT: "txt_backend",
    FileCategory.CSV: "csv_backend",
    FileCategory.EXCEL: "excel_backend",
    FileCategory.PPT: "ppt_backend",
    FileCategory.PPTX: "pptx_backend",
    FileCategory.HTML: "html_backend",
    FileCategory.URL: "url_backend",
}


_AUTO_BACKEND_NAME = "auto"

_AUTO_BACKEND_PREFERENCE_DEFAULTS: dict[FileCategory, tuple[str, ...]] = {
    FileCategory.READABLE_PDF: ("pymupdf", "pdfplumber", "pypdf"),
    FileCategory.SCANNED_PDF: ("ocrmypdf", "paddleocr", "tesseract"),
    FileCategory.IMAGE: ("paddleocr", "tesseract"),
    FileCategory.DOCX: ("python-docx", "docx2txt"),
    FileCategory.CSV: ("pandas", "builtin_csv"),
    FileCategory.EXCEL: ("pandas", "openpyxl"),
    FileCategory.PPTX: ("python-pptx",),
    FileCategory.HTML: ("trafilatura", "beautifulsoup", "builtin"),
    FileCategory.URL: ("trafilatura", "beautifulsoup"),
}


def _is_auto_backend_name(name: str | None) -> bool:
  return name == _AUTO_BACKEND_NAME


def _supports_auto_backend(category: FileCategory) -> bool:
  return category in _AUTO_BACKEND_PREFERENCE_DEFAULTS


def _normalize_backend_preference_order(
    preferences: Mapping[FileCategory | str, Sequence[str]],
) -> dict[FileCategory, tuple[str, ...]]:
  normalized: dict[FileCategory, tuple[str, ...]] = {}
  for raw_category, raw_order in preferences.items():
    category = _normalize_preference_category(raw_category)
    if not _supports_auto_backend(category):
      raise ValueError(
          "Automatic backend selection is not supported for "
          f"{category.value!r}."
      )

    if isinstance(raw_order, str):
      raise ValueError(
          f"Backend preference order for {category.value!r} must be a "
          "sequence of backend names, not a single string."
      )

    order = tuple(raw_order)
    if not order:
      raise ValueError(
          f"Backend preference order for {category.value!r} must not be empty."
      )

    seen: set[str] = set()
    for backend_name in order:
      if _is_auto_backend_name(backend_name):
        raise ValueError(
            f"Backend preference order for {category.value!r} cannot "
            f"include {_AUTO_BACKEND_NAME!r}."
        )
      if get_backend(category, backend_name) is None:
        allowed = [backend.name for backend in get_backends(category)]
        raise ValueError(
            f"Invalid backend {backend_name!r} in preference order for "
            f"{category.value!r}. Allowed backends: {allowed}"
        )
      if backend_name in seen:
        raise ValueError(
            f"Duplicate backend {backend_name!r} in preference order for "
            f"{category.value!r}."
        )
      seen.add(backend_name)

    normalized[category] = order

  return normalized


def _normalize_preference_category(
    raw_category: FileCategory | str,
) -> FileCategory:
  if isinstance(raw_category, FileCategory):
    return raw_category
  try:
    return FileCategory(raw_category)
  except ValueError as e:
    raise ValueError(
        f"Unknown backend preference category {raw_category!r}."
    ) from e


_EXTRA_INSTALL_COMMANDS: dict[str, str] = {
    "pdf": 'pip install "langextract[pdf]"',
    "ocr": 'pip install "langextract[ocr]"',
    "office": 'pip install "langextract[office]"',
    "tables": 'pip install "langextract[tables]"',
    "html": 'pip install "langextract[html]"',
    "xlsx": 'pip install "langextract[xlsx]"',
    "full": 'pip install "langextract[full]"',
    "all": 'pip install "langextract[all]"',
}


# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_BACKENDS: tuple[IngestionBackend, ...] = (
    # ── readable_pdf ──────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.READABLE_PDF,
        name="pymupdf",
        pypi_package="pymupdf",
        import_check="pymupdf",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="Fast PDF text extraction via PyMuPDF (fitz).",
        extras=("pdf",),
    ),
    IngestionBackend(
        category=FileCategory.READABLE_PDF,
        name="pdfplumber",
        pypi_package="pdfplumber",
        import_check="pdfplumber",
        status=BackendStatus.AVAILABLE,
        description="PDF text and layout extraction via pdfplumber.",
        extras=("pdf", "tables"),
    ),
    IngestionBackend(
        category=FileCategory.READABLE_PDF,
        name="pypdf",
        pypi_package="pypdf",
        import_check="pypdf",
        status=BackendStatus.AVAILABLE,
        description="Pure-Python PDF text extraction via pypdf.",
        extras=("pdf",),
    ),
    # ── scanned_pdf ───────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.SCANNED_PDF,
        name="ocrmypdf",
        pypi_package="ocrmypdf",
        import_check="ocrmypdf",
        status=BackendStatus.REGISTERED,
        description="OCR scanned PDFs via ocrmypdf (Tesseract wrapper).",
        extras=("ocr",),
    ),
    IngestionBackend(
        category=FileCategory.SCANNED_PDF,
        name="paddleocr",
        pypi_package="paddleocr",
        import_check="paddleocr",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="OCR scanned PDFs via PaddleOCR.",
        extras=("ocr",),
    ),
    IngestionBackend(
        category=FileCategory.SCANNED_PDF,
        name="tesseract",
        pypi_package="pytesseract",
        import_check="pytesseract",
        status=BackendStatus.AVAILABLE,
        description="OCR scanned PDFs via Tesseract (pytesseract).",
        extras=("ocr",),
    ),
    # ── image ─────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.IMAGE,
        name="paddleocr",
        pypi_package="paddleocr",
        import_check="paddleocr",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="OCR images via PaddleOCR.",
        extras=("ocr",),
    ),
    IngestionBackend(
        category=FileCategory.IMAGE,
        name="tesseract",
        pypi_package="pytesseract",
        import_check="pytesseract",
        status=BackendStatus.AVAILABLE,
        description="OCR images via Tesseract (pytesseract).",
        extras=("ocr",),
    ),
    IngestionBackend(
        category=FileCategory.IMAGE,
        name="easyocr",
        pypi_package="easyocr",
        import_check="easyocr",
        status=BackendStatus.REGISTERED,
        description="OCR images via EasyOCR.",
        extras=("ocr",),
    ),
    # ── table_pdf ─────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.TABLE_PDF,
        name="pdfplumber",
        pypi_package="pdfplumber",
        import_check="pdfplumber",
        status=BackendStatus.REGISTERED,
        description="Table extraction from PDFs via pdfplumber.",
        extras=("tables", "pdf"),
    ),
    IngestionBackend(
        category=FileCategory.TABLE_PDF,
        name="camelot",
        pypi_package="camelot-py",
        import_check="camelot",
        status=BackendStatus.REGISTERED,
        description="Table extraction from PDFs via Camelot.",
        extras=("tables",),
    ),
    IngestionBackend(
        category=FileCategory.TABLE_PDF,
        name="tabula",
        pypi_package="tabula-py",
        import_check="tabula",
        status=BackendStatus.REGISTERED,
        description="Table extraction from PDFs via tabula-py.",
        extras=("tables",),
    ),
    # ── docx ──────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.DOCX,
        name="python-docx",
        pypi_package="python-docx",
        import_check="docx",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="DOCX text extraction via python-docx.",
        extras=("office",),
    ),
    IngestionBackend(
        category=FileCategory.DOCX,
        name="docx2txt",
        pypi_package="docx2txt",
        import_check="docx2txt",
        status=BackendStatus.AVAILABLE,
        description="DOCX text extraction via docx2txt.",
        extras=("office",),
    ),
    IngestionBackend(
        category=FileCategory.DOCX,
        name="mammoth",
        pypi_package="mammoth",
        import_check="mammoth",
        status=BackendStatus.REGISTERED,
        description="DOCX to HTML/text conversion via mammoth.",
        extras=("office",),
    ),
    # ── doc (legacy) ──────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.DOC,
        name="libreoffice",
        description=(
            "DOC text extraction via LibreOffice CLI (soffice). "
            "Requires LibreOffice installed on the system."
        ),
        status=BackendStatus.REGISTERED,
    ),
    IngestionBackend(
        category=FileCategory.DOC,
        name="antiword",
        description=(
            "DOC text extraction via antiword CLI. "
            "Requires antiword installed on the system."
        ),
        status=BackendStatus.REGISTERED,
    ),
    IngestionBackend(
        category=FileCategory.DOC,
        name="tika",
        pypi_package="tika",
        import_check="tika",
        status=BackendStatus.REGISTERED,
        description="DOC text extraction via Apache Tika.",
        extras=("office",),
    ),
    # ── txt ───────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.TXT,
        name="builtin",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="Built-in UTF-8/UTF-16 text file reading.",
    ),
    IngestionBackend(
        category=FileCategory.TXT,
        name="charset-normalizer",
        pypi_package="charset-normalizer",
        import_check="charset_normalizer",
        status=BackendStatus.REGISTERED,
        description="Text decoding with automatic charset detection.",
    ),
    # ── csv ───────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.CSV,
        name="pandas",
        pypi_package="pandas",
        import_check="pandas",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="CSV reading via pandas.read_csv.",
    ),
    IngestionBackend(
        category=FileCategory.CSV,
        name="builtin_csv",
        status=BackendStatus.AVAILABLE,
        description="CSV reading via the stdlib csv module.",
    ),
    # ── excel ─────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.EXCEL,
        name="pandas",
        pypi_package="pandas",
        import_check="pandas",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="Excel reading via pandas.read_excel + openpyxl engine.",
    ),
    IngestionBackend(
        category=FileCategory.EXCEL,
        name="openpyxl",
        pypi_package="openpyxl",
        import_check="openpyxl",
        status=BackendStatus.AVAILABLE,
        description="Excel reading directly via openpyxl (no pandas).",
        extras=("office", "xlsx"),
    ),
    # ── ppt (legacy) ──────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.PPT,
        name="libreoffice",
        description=(
            "PPT text extraction via LibreOffice CLI (soffice). "
            "Requires LibreOffice installed on the system."
        ),
        status=BackendStatus.REGISTERED,
    ),
    IngestionBackend(
        category=FileCategory.PPT,
        name="tika",
        pypi_package="tika",
        import_check="tika",
        status=BackendStatus.REGISTERED,
        description="PPT text extraction via Apache Tika.",
        extras=("office",),
    ),
    # ── pptx ──────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.PPTX,
        name="python-pptx",
        pypi_package="python-pptx",
        import_check="pptx",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="PPTX text extraction via python-pptx.",
        extras=("office",),
    ),
    IngestionBackend(
        category=FileCategory.PPTX,
        name="libreoffice",
        description=(
            "PPTX text extraction via LibreOffice CLI (soffice). "
            "Requires LibreOffice installed on the system."
        ),
        status=BackendStatus.REGISTERED,
    ),
    IngestionBackend(
        category=FileCategory.PPTX,
        name="tika",
        pypi_package="tika",
        import_check="tika",
        status=BackendStatus.REGISTERED,
        description="PPTX text extraction via Apache Tika.",
        extras=("office",),
    ),
    # ── html ──────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.HTML,
        name="builtin",
        status=BackendStatus.AVAILABLE,
        description="Built-in raw text reading for local HTML files.",
    ),
    IngestionBackend(
        category=FileCategory.HTML,
        name="trafilatura",
        pypi_package="trafilatura",
        import_check="trafilatura",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="HTML main-content extraction via trafilatura.",
        extras=("html",),
    ),
    IngestionBackend(
        category=FileCategory.HTML,
        name="beautifulsoup",
        pypi_package="beautifulsoup4",
        import_check="bs4",
        status=BackendStatus.AVAILABLE,
        description="HTML parsing via Beautiful Soup 4.",
        extras=("html",),
    ),
    # ── url ───────────────────────────────────────────────────────────
    IngestionBackend(
        category=FileCategory.URL,
        name="trafilatura",
        pypi_package="trafilatura",
        import_check="trafilatura",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        description="Web page content extraction via trafilatura.",
        extras=("html",),
    ),
    IngestionBackend(
        category=FileCategory.URL,
        name="beautifulsoup",
        pypi_package="beautifulsoup4",
        import_check="bs4",
        status=BackendStatus.AVAILABLE,
        description="Web page parsing via Beautiful Soup 4.",
        extras=("html",),
    ),
)


# Build category → backends index once at import time.
_BY_CATEGORY: dict[FileCategory, list[IngestionBackend]] = {}
for _b in _BACKENDS:
  _BY_CATEGORY.setdefault(_b.category, []).append(_b)


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------


def list_categories() -> list[FileCategory]:
  """Return all registered file categories in definition order."""
  seen: dict[FileCategory, None] = {}
  for b in _BACKENDS:
    seen.setdefault(b.category, None)
  return list(seen)


def list_all_backends() -> Sequence[IngestionBackend]:
  """Return every registered backend (read-only snapshot)."""
  return _BACKENDS


def _normalize_backend_category_selection(
    category: FileCategory | str | None,
) -> tuple[FileCategory, ...]:
  if category is None:
    return tuple(list_categories())
  if isinstance(category, FileCategory):
    return (category,)
  try:
    return (FileCategory(category),)
  except ValueError as e:
    valid = ", ".join(c.value for c in FileCategory)
    raise ValueError(
        f"Unknown category {category!r}. Valid categories: {valid}"
    ) from e


def list_available_backends(
    category: FileCategory | str | None = None,
) -> tuple[BackendCategoryInfo, ...]:
  """Return structured backend capability reports for one or all categories.

  Each returned category report answers the common user questions:

  - which categories are supported
  - which backends can be selected
  - which backends are installed and usable right now
  - which backend is the default
  - which extra package or install command is relevant

  Args:
    category: Optional category filter. Accepts a ``FileCategory`` value or
      its string form such as ``"readable_pdf"`` or ``"csv"``.

  Returns:
    A tuple of ``BackendCategoryInfo`` reports in registry definition order.

  Raises:
    ValueError: If ``category`` is not a known file category.
  """
  selected_categories = _normalize_backend_category_selection(category)
  auto_preferences = ParserBackendOptions()
  reports: list[BackendCategoryInfo] = []

  for selected_category in selected_categories:
    default_backend = get_default_backend(selected_category)
    auto_preference = auto_preferences.preference_order_for_category(
        selected_category
    )
    auto_rank = {
        backend_name: index + 1
        for index, backend_name in enumerate(auto_preference or ())
    }
    backend_infos: list[BackendOptionInfo] = []
    for backend in get_backends(selected_category):
      diagnostics = inspect_backend_runtime(backend)
      backend_infos.append(
          BackendOptionInfo(
              name=backend.name,
              description=backend.description,
              implementation=(
                  "implemented"
                  if backend.status == BackendStatus.AVAILABLE
                  else "planned"
              ),
              installed=diagnostics.installed,
              usable=diagnostics.usable,
              default=backend is default_backend,
              auto_preference_rank=auto_rank.get(backend.name),
              extras=backend.extras,
              install_commands=tuple(_install_commands_for_backend(backend)),
              reason=diagnostics.unavailable_reason,
          )
      )

    reports.append(
        BackendCategoryInfo(
            category=selected_category,
            default=None if default_backend is None else default_backend.name,
            auto_preference=auto_preference,
            backends=tuple(backend_infos),
        )
    )

  return tuple(reports)


def get_backends(
    category: FileCategory,
    *,
    available_only: bool = False,
) -> list[IngestionBackend]:
  """Return backends registered for *category*.

  Args:
    category: The file category to query.
    available_only: If ``True``, exclude ``REGISTERED``-only backends.

  Returns:
    List of backends, possibly empty if the category has no entries.
  """
  backends = _BY_CATEGORY.get(category, [])
  if available_only:
    return [b for b in backends if b.status == BackendStatus.AVAILABLE]
  return list(backends)


def get_default_backend(category: FileCategory) -> IngestionBackend | None:
  """Return the default backend for *category*, or ``None``."""
  for b in _BY_CATEGORY.get(category, []):
    if b.is_default:
      return b
  return None


def get_backend(
    category: FileCategory,
    name: str,
) -> IngestionBackend | None:
  """Look up a specific backend by category and name."""
  for b in _BY_CATEGORY.get(category, []):
    if b.name == name:
      return b
  return None


def find_backend_categories(name: str) -> list[FileCategory]:
  """Return every category that registers *name* in definition order.

  This is primarily intended for future config-driven routing where a caller
  may choose a backend name first and then decide which input category it
  applies to.
  """
  categories: list[FileCategory] = []
  for category in list_categories():
    if get_backend(category, name) is not None:
      categories.append(category)
  return categories


def is_backend_installed(backend: IngestionBackend) -> bool:
  """Check whether the backend's library is importable.

  Returns ``True`` for backends with no ``import_check`` (built-in /
  stdlib) and for backends whose library can be imported.
  """
  if backend.import_check is None:
    return True
  try:
    importlib.import_module(backend.import_check)
    return True
  except ImportError:
    return False


def _dedupe_preserving_order(items: Sequence[str]) -> list[str]:
  unique: list[str] = []
  seen: set[str] = set()
  for item in items:
    if item in seen:
      continue
    seen.add(item)
    unique.append(item)
  return unique


def _resolve_requested_backend_name(
    category: FileCategory,
    *,
    backend: str | None = None,
    parser_backends: ParserBackendOptions | None = None,
) -> str | None:
  if backend is not None:
    return backend
  if parser_backends is None:
    return None
  return parser_backends.backend_for_category(category)


def _auto_backend_preference_order(
    category: FileCategory,
    *,
    parser_backends: ParserBackendOptions | None = None,
) -> tuple[str, ...]:
  if not _supports_auto_backend(category):
    raise ValueError(
        f"Automatic backend selection is not supported for {category.value!r}."
    )

  if parser_backends is not None:
    configured_order = parser_backends.preference_order_for_category(category)
    if configured_order is not None:
      return configured_order

  return _AUTO_BACKEND_PREFERENCE_DEFAULTS[category]


def _describe_auto_backend_unavailable(backend: IngestionBackend) -> str:
  if backend.status == BackendStatus.REGISTERED:
    return "registered but not yet implemented"
  if not is_backend_installed(backend):
    if backend.pypi_package is not None:
      return f"missing optional dependency {backend.pypi_package!r}"
    if backend.import_check is not None:
      return f"missing optional dependency {backend.import_check!r}"
    return "missing optional dependency"
  return "unavailable"


def _iter_auto_backend_candidates(
    category: FileCategory,
    *,
    parser_backends: ParserBackendOptions | None = None,
) -> tuple[IngestionBackend, ...]:
  preference_order = _auto_backend_preference_order(
      category,
      parser_backends=parser_backends,
  )
  candidates: list[IngestionBackend] = []
  skipped: list[str] = []

  for backend_name in preference_order:
    backend = get_backend(category, backend_name)
    if backend is None:
      continue
    if backend.status != BackendStatus.AVAILABLE or not is_backend_installed(
        backend
    ):
      skipped.append(
          f"{backend.name} ({_describe_auto_backend_unavailable(backend)})"
      )
      continue
    candidates.append(backend)

  if candidates:
    return tuple(candidates)

  details = ", ".join(skipped) if skipped else ", ".join(preference_order)
  raise ImportError(
      f"Automatic backend selection for category {category.value!r} could "
      f"not find an available installed backend. Tried: {details}"
  )


def _install_commands_for_backend(
    backend: IngestionBackend,
    *,
    missing_package: str | None = None,
    extras: Sequence[str] | None = None,
) -> list[str]:
  commands = [
      _EXTRA_INSTALL_COMMANDS[extra]
      for extra in (extras if extras is not None else backend.extras)
      if extra in _EXTRA_INSTALL_COMMANDS
  ]

  package_name = missing_package or backend.pypi_package
  if package_name:
    commands.append(f"pip install {package_name}")

  return _dedupe_preserving_order(commands)


def _format_install_hint(
    backend: IngestionBackend,
    *,
    missing_package: str | None = None,
    extras: Sequence[str] | None = None,
) -> str:
  commands = _install_commands_for_backend(
      backend,
      missing_package=missing_package,
      extras=extras,
  )
  if not commands:
    return (
        "Install the required system dependency for the selected backend and "
        "retry."
    )
  if len(commands) == 1:
    return f"Install with: {commands[0]}"
  return "Install with one of: " + " or ".join(commands)


def _build_missing_dependency_error(
    backend: IngestionBackend,
    *,
    missing_import: str | None = None,
    missing_package: str | None = None,
    extras: Sequence[str] | None = None,
) -> ImportError:
  package_name = missing_package or backend.pypi_package or missing_import
  import_name = missing_import or backend.import_check

  requirement = []
  if package_name is not None:
    requirement.append(f"package {package_name!r}")
  if import_name is not None and import_name != package_name:
    requirement.append(f"import name {import_name!r}")

  if requirement:
    dependency_text = " (" + ", ".join(requirement) + ")"
  else:
    dependency_text = ""

  return ImportError(
      f"Selected ingestion backend {backend.name!r} for category "
      f"{backend.category.value!r} requires an optional dependency"
      f"{dependency_text}, but it is not installed. "
      f"{_format_install_hint(backend, missing_package=missing_package, extras=extras)}"
  )


_SYSTEM_BACKEND_COMMANDS: dict[str, str] = {
    "antiword": "antiword",
    "libreoffice": "soffice",
}


def _can_import_module(import_name: str) -> bool:
  try:
    importlib.import_module(import_name)
    return True
  except ImportError:
    return False


def _missing_dependency_reason(
    backend: IngestionBackend,
    *,
    missing_import: str | None = None,
    missing_package: str | None = None,
    extras: Sequence[str] | None = None,
) -> str:
  package_name = missing_package or backend.pypi_package or missing_import
  import_name = missing_import or backend.import_check

  if package_name is not None:
    dependency_text = f"missing optional dependency {package_name!r}"
  elif import_name is not None:
    dependency_text = f"missing optional dependency {import_name!r}"
  else:
    dependency_text = "missing optional dependency"

  return (
      f"{dependency_text}. "
      f"{_format_install_hint(backend, missing_package=missing_package, extras=extras)}"
  )


def _primary_dependency_status(
    backend: IngestionBackend,
) -> tuple[bool | None, str | None]:
  if backend.import_check is not None:
    if is_backend_installed(backend):
      return True, None
    return False, _missing_dependency_reason(backend)

  command = _SYSTEM_BACKEND_COMMANDS.get(backend.name)
  if command is not None:
    if shutil.which(command) is not None:
      return True, None
    return False, f"missing system executable {command!r} on PATH"

  if backend.status == BackendStatus.AVAILABLE:
    return True, None

  return None, "runtime dependency inspection is not available for this backend"


def _additional_runtime_issues(backend: IngestionBackend) -> list[str]:
  issues: list[str] = []

  if backend.category == FileCategory.EXCEL and backend.name == "pandas":
    openpyxl_backend = get_backend(FileCategory.EXCEL, "openpyxl")
    if openpyxl_backend is not None and not is_backend_installed(
        openpyxl_backend
    ):
      issues.append(
          "requires optional dependency 'openpyxl' for Excel reading. "
          + _format_install_hint(
              openpyxl_backend,
              missing_package="openpyxl",
              extras=("office", "xlsx"),
          )
      )

  if backend.category == FileCategory.SCANNED_PDF:
    pymupdf_backend = get_backend(FileCategory.READABLE_PDF, "pymupdf")
    if pymupdf_backend is not None and not is_backend_installed(
        pymupdf_backend
    ):
      issues.append(
          "requires optional dependency 'pymupdf' for PDF page rendering. "
          + _format_install_hint(
              pymupdf_backend,
              missing_package="pymupdf",
              extras=("pdf",),
          )
      )

  if backend.name == "tesseract":
    if not _can_import_module("PIL"):
      issues.append(
          "requires optional dependency 'pillow'. Install with: pip install"
          " pillow"
      )
    if shutil.which("tesseract") is None:
      issues.append(
          "requires system 'tesseract' executable to be installed and available"
          " on PATH"
      )

  return issues


def inspect_backend_runtime(backend: IngestionBackend) -> BackendDiagnostics:
  """Inspect whether a backend appears usable in the current environment.

  This combines the registry's implementation status with dependency checks
  that are known to affect runtime behavior for certain backends.
  """
  installed, installed_reason = _primary_dependency_status(backend)
  reasons: list[str] = []

  if backend.status == BackendStatus.REGISTERED:
    reasons.append("planned backend (registered but not yet implemented)")

  if installed_reason is not None and installed is not True:
    reasons.append(installed_reason)

  runtime_issues = (
      _additional_runtime_issues(backend) if installed is True else []
  )
  reasons.extend(runtime_issues)

  usable = (
      backend.status == BackendStatus.AVAILABLE
      and installed is True
      and not runtime_issues
  )
  return BackendDiagnostics(
      backend=backend,
      installed=installed,
      usable=usable,
      unavailable_reason="; ".join(reasons) if reasons else None,
  )


def require_backend(
    category: FileCategory,
    name: str | None = None,
) -> IngestionBackend:
  """Resolve and validate a backend for *category*.

  If *name* is ``None``, returns the category's default backend.

  Raises:
    ValueError: If the backend is not found in the registry.
    NotImplementedError: If the backend is ``REGISTERED`` but not yet
      wired into the ingestion pipeline.
    ImportError: If the backend's library is not installed.
  """
  if name is None:
    backend = get_default_backend(category)
    if backend is None:
      raise ValueError(
          "No default ingestion backend registered for category "
          f"{category.value!r}."
      )
  else:
    backend = get_backend(category, name)
    if backend is None:
      known = [b.name for b in get_backends(category)]
      raise ValueError(
          f"Unknown ingestion backend {name!r} for category "
          f"{category.value!r}. Registered backends: {known}"
      )

  if backend.status == BackendStatus.REGISTERED:
    raise NotImplementedError(
        f"Ingestion backend {backend.name!r} for category "
        f"{backend.category.value!r} is registered but not yet implemented. "
        "Available (wired) backends for this category: "
        f"{[b.name for b in get_backends(backend.category, available_only=True)]}"
    )

  if not is_backend_installed(backend):
    raise _build_missing_dependency_error(backend)

  return backend


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def resolve_backend(
    category: FileCategory,
    *,
    backend: str | None = None,
    parser_backends: ParserBackendOptions | None = None,
) -> IngestionBackend | None:
  """Resolve and validate a backend for *category*.

  Resolution priority:
    1. Explicit *backend* argument (legacy per-call override).
    2. Per-category selection from *parser_backends* config.
    3. Default backend for the category.

  Returns ``None`` when no explicit selection exists and the category has
  no default backend.

  Raises:
    ValueError: Backend name is unknown for the category.
    NotImplementedError: Backend is registered but not yet wired.
    ImportError: Backend library is not installed.
  """
  name = _resolve_requested_backend_name(
      category,
      backend=backend,
      parser_backends=parser_backends,
  )
  if _is_auto_backend_name(name):
    return _iter_auto_backend_candidates(
        category,
        parser_backends=parser_backends,
    )[0]
  if name is not None:
    return require_backend(category, name)
  default_backend = get_default_backend(category)
  if default_backend is None:
    return None
  return require_backend(category)


# ---------------------------------------------------------------------------
# Handler dispatch registry
# ---------------------------------------------------------------------------

_HANDLERS: dict[tuple[FileCategory, str], Callable[..., Any]] = {}


def register_handler(
    category: FileCategory,
    backend_name: str,
) -> Callable:
  """Decorator to register a dispatch handler for a ``(category, backend)`` pair.

  The decorated function is called by :func:`resolve_handler` when the
  given pair is selected at runtime.  Handler functions should accept
  ``**kwargs`` so the dispatch site can pass a uniform context.

  Raises:
    ValueError: If the ``(category, backend_name)`` pair is not in the
      registry.
  """
  if get_backend(category, backend_name) is None:
    raise ValueError(
        "Cannot register handler for unknown backend "
        f"{backend_name!r} in category {category.value!r}."
    )

  def decorator(fn: Callable) -> Callable:
    _HANDLERS[(category, backend_name)] = fn
    return fn

  return decorator


def get_handler(
    category: FileCategory,
    backend_name: str,
) -> Callable[..., Any] | None:
  """Return the registered handler for a ``(category, backend)`` pair."""
  return _HANDLERS.get((category, backend_name))


def resolve_handler(
    category: FileCategory,
    *,
    backend: str | None = None,
    parser_backends: ParserBackendOptions | None = None,
) -> tuple[IngestionBackend, Callable[..., Any]] | None:
  """Resolve a backend and look up its dispatch handler.

  Returns a ``(backend_metadata, handler_fn)`` tuple when resolution
  succeeds and a handler is registered.  Returns ``None`` when no backend
  resolves or the resolved backend has no handler.

  Validation errors from :func:`resolve_backend` propagate as-is.
  """
  name = _resolve_requested_backend_name(
      category,
      backend=backend,
      parser_backends=parser_backends,
  )
  if _is_auto_backend_name(name):
    for candidate in _iter_auto_backend_candidates(
        category,
        parser_backends=parser_backends,
    ):
      handler = get_handler(category, candidate.name)
      if handler is not None:
        return candidate, handler
    return None

  resolved = resolve_backend(
      category,
      backend=backend,
      parser_backends=parser_backends,
  )
  if resolved is None:
    return None
  handler = get_handler(category, resolved.name)
  if handler is None:
    return None
  return resolved, handler
