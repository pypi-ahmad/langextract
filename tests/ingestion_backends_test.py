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

"""Tests for langextract.ingestion_backends."""

from __future__ import annotations

import pathlib
import tempfile
from unittest import mock

try:
  import tomllib
except ImportError:  # pragma: no cover
  import tomli as tomllib

from absl.testing import absltest

from langextract import ingestion_backends
from langextract.ingestion_backends import BackendCategoryInfo
from langextract.ingestion_backends import BackendDiagnostics
from langextract.ingestion_backends import BackendOptionInfo
from langextract.ingestion_backends import BackendStatus
from langextract.ingestion_backends import FileCategory
from langextract.ingestion_backends import find_backend_categories
from langextract.ingestion_backends import get_backend
from langextract.ingestion_backends import get_backends
from langextract.ingestion_backends import get_default_backend
from langextract.ingestion_backends import IngestionBackend
from langextract.ingestion_backends import inspect_backend_runtime
from langextract.ingestion_backends import is_backend_installed
from langextract.ingestion_backends import list_all_backends
from langextract.ingestion_backends import list_available_backends
from langextract.ingestion_backends import list_categories
from langextract.ingestion_backends import ParserBackendOptions
from langextract.ingestion_backends import require_backend


class FileCategoryTest(absltest.TestCase):
  """Tests for the FileCategory enum."""

  def test_all_thirteen_categories_exist(self):
    expected = {
        "readable_pdf",
        "scanned_pdf",
        "image",
        "table_pdf",
        "docx",
        "doc",
        "txt",
        "csv",
        "excel",
        "ppt",
        "pptx",
        "html",
        "url",
    }
    actual = {c.value for c in FileCategory}
    self.assertEqual(actual, expected)

  def test_str_is_value(self):
    self.assertEqual(str(FileCategory.READABLE_PDF), "readable_pdf")
    self.assertEqual(str(FileCategory.URL), "url")


class BackendStatusTest(absltest.TestCase):

  def test_available_and_registered(self):
    self.assertEqual(str(BackendStatus.AVAILABLE), "available")
    self.assertEqual(str(BackendStatus.REGISTERED), "registered")


class ListCategoriesTest(absltest.TestCase):

  def test_returns_all_categories(self):
    cats = list_categories()
    self.assertLen(cats, 13)
    # First should be readable_pdf (definition order)
    self.assertEqual(cats[0], FileCategory.READABLE_PDF)

  def test_preserves_definition_order(self):
    cats = list_categories()
    values = [c.value for c in cats]
    # readable_pdf should come before scanned_pdf, etc.
    self.assertLess(values.index("readable_pdf"), values.index("scanned_pdf"))
    self.assertLess(values.index("csv"), values.index("excel"))


class ListAllBackendsTest(absltest.TestCase):

  def test_returns_nonempty(self):
    backends = list_all_backends()
    self.assertGreater(len(backends), 0)

  def test_all_entries_are_ingestion_backend(self):
    for b in list_all_backends():
      self.assertIsInstance(b, IngestionBackend)


class ListAvailableBackendsTest(absltest.TestCase):

  def test_returns_structured_reports_for_all_categories(self):
    reports = list_available_backends()

    self.assertLen(reports, 13)
    self.assertIsInstance(reports[0], BackendCategoryInfo)
    self.assertEqual(reports[0].category, FileCategory.READABLE_PDF)

  def test_category_filter_accepts_string_name(self):
    reports = list_available_backends("csv")

    self.assertLen(reports, 1)
    self.assertEqual(reports[0].category, FileCategory.CSV)

  def test_invalid_category_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, r"Valid categories"):
      list_available_backends("not-a-category")

  def test_report_includes_defaults_and_install_commands(self):
    report = list_available_backends(FileCategory.READABLE_PDF)[0]
    backends = {backend.name: backend for backend in report.backends}

    self.assertEqual(report.default, "pymupdf")
    self.assertIsInstance(backends["pymupdf"], BackendOptionInfo)
    self.assertTrue(backends["pymupdf"].default)
    self.assertIn(
        'pip install "langextract[pdf]"', backends["pymupdf"].install_commands
    )
    self.assertEqual(backends["pymupdf"].implementation, "implemented")

  def test_report_includes_runtime_diagnostics(self):
    def _is_installed(backend: IngestionBackend) -> bool:
      return backend.name not in {"trafilatura", "beautifulsoup4"}

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      report = list_available_backends(FileCategory.HTML)[0]

    backends = {backend.name: backend for backend in report.backends}
    self.assertFalse(backends["trafilatura"].usable)
    self.assertIn("langextract[html]", backends["trafilatura"].reason)
    self.assertEqual(backends["builtin"].implementation, "implemented")


class GetBackendsTest(absltest.TestCase):

  def test_readable_pdf_has_three_backends(self):
    backends = get_backends(FileCategory.READABLE_PDF)
    names = [b.name for b in backends]
    self.assertLen(names, 3)
    self.assertIn("pymupdf", names)
    self.assertIn("pdfplumber", names)
    self.assertIn("pypdf", names)

  def test_scanned_pdf_has_three_backends(self):
    backends = get_backends(FileCategory.SCANNED_PDF)
    self.assertLen(backends, 3)

  def test_image_has_three_backends(self):
    backends = get_backends(FileCategory.IMAGE)
    self.assertLen(backends, 3)

  def test_csv_has_two_backends(self):
    backends = get_backends(FileCategory.CSV)
    self.assertLen(backends, 2)

  def test_txt_has_two_backends(self):
    backends = get_backends(FileCategory.TXT)
    self.assertLen(backends, 2)

  def test_available_only_filters_registered(self):
    all_backends = get_backends(FileCategory.SCANNED_PDF)
    available = get_backends(FileCategory.SCANNED_PDF, available_only=True)
    self.assertGreater(len(all_backends), len(available))
    for b in available:
      self.assertEqual(b.status, BackendStatus.AVAILABLE)

  def test_available_only_filters_to_wired_scanned_pdf_backends(self):
    available = get_backends(FileCategory.SCANNED_PDF, available_only=True)
    self.assertEqual(
        {backend.name for backend in available},
        {
            "paddleocr",
            "tesseract",
        },
    )


class GetDefaultBackendTest(absltest.TestCase):

  def test_readable_pdf_default_is_pymupdf(self):
    default = get_default_backend(FileCategory.READABLE_PDF)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "pymupdf")
    self.assertTrue(default.is_default)
    self.assertEqual(default.status, BackendStatus.AVAILABLE)

  def test_txt_default_is_builtin(self):
    default = get_default_backend(FileCategory.TXT)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "builtin")

  def test_csv_default_is_pandas(self):
    default = get_default_backend(FileCategory.CSV)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "pandas")

  def test_excel_default_is_pandas(self):
    default = get_default_backend(FileCategory.EXCEL)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "pandas")

  def test_scanned_pdf_default_is_paddleocr(self):
    default = get_default_backend(FileCategory.SCANNED_PDF)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "paddleocr")

  def test_image_default_is_paddleocr(self):
    default = get_default_backend(FileCategory.IMAGE)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "paddleocr")

  def test_docx_default_is_python_docx(self):
    default = get_default_backend(FileCategory.DOCX)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "python-docx")

  def test_pptx_default_is_python_pptx(self):
    default = get_default_backend(FileCategory.PPTX)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "python-pptx")

  def test_html_default_is_trafilatura(self):
    default = get_default_backend(FileCategory.HTML)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "trafilatura")

  def test_url_default_is_trafilatura(self):
    default = get_default_backend(FileCategory.URL)
    self.assertIsNotNone(default)
    self.assertEqual(default.name, "trafilatura")

  def test_no_default_for_table_pdf(self):
    self.assertIsNone(get_default_backend(FileCategory.TABLE_PDF))

  def test_no_default_for_doc(self):
    self.assertIsNone(get_default_backend(FileCategory.DOC))

  def test_no_default_for_ppt(self):
    self.assertIsNone(get_default_backend(FileCategory.PPT))


class GetBackendTest(absltest.TestCase):

  def test_lookup_by_name(self):
    b = get_backend(FileCategory.READABLE_PDF, "pdfplumber")
    self.assertIsNotNone(b)
    self.assertEqual(b.name, "pdfplumber")
    self.assertEqual(b.category, FileCategory.READABLE_PDF)

  def test_returns_none_for_unknown(self):
    self.assertIsNone(get_backend(FileCategory.READABLE_PDF, "nonexistent"))

  def test_each_category_backend_names(self):
    """Verify the backend names per-category match the spec."""
    expected = {
        FileCategory.READABLE_PDF: {"pymupdf", "pdfplumber", "pypdf"},
        FileCategory.SCANNED_PDF: {"ocrmypdf", "paddleocr", "tesseract"},
        FileCategory.IMAGE: {"paddleocr", "tesseract", "easyocr"},
        FileCategory.TABLE_PDF: {"pdfplumber", "camelot", "tabula"},
        FileCategory.DOCX: {"python-docx", "docx2txt", "mammoth"},
        FileCategory.DOC: {"libreoffice", "antiword", "tika"},
        FileCategory.TXT: {"builtin", "charset-normalizer"},
        FileCategory.CSV: {"pandas", "builtin_csv"},
        FileCategory.EXCEL: {"pandas", "openpyxl"},
        FileCategory.PPT: {"libreoffice", "tika"},
        FileCategory.PPTX: {"python-pptx", "libreoffice", "tika"},
        FileCategory.HTML: {"builtin", "trafilatura", "beautifulsoup"},
        FileCategory.URL: {"trafilatura", "beautifulsoup"},
    }
    for category, expected_names in expected.items():
      actual_names = {b.name for b in get_backends(category)}
      self.assertEqual(
          actual_names,
          expected_names,
          f"Mismatch for {category.value}",
      )


class FindBackendCategoriesTest(absltest.TestCase):

  def test_returns_all_categories_for_reused_backend_names(self):
    self.assertEqual(
        find_backend_categories("pdfplumber"),
        [FileCategory.READABLE_PDF, FileCategory.TABLE_PDF],
    )
    self.assertEqual(
        find_backend_categories("pandas"),
        [FileCategory.CSV, FileCategory.EXCEL],
    )

  def test_unknown_backend_name_returns_empty(self):
    self.assertEmpty(find_backend_categories("does-not-exist"))


class ParserBackendOptionsTest(absltest.TestCase):

  def test_valid_backend_selections_construct(self):
    options = ParserBackendOptions(
        readable_pdf_backend="pymupdf",
        csv_backend="pandas",
        html_backend="beautifulsoup",
    )

    self.assertEqual(
        options.backend_for_category(FileCategory.READABLE_PDF),
        "pymupdf",
    )
    self.assertEqual(
        options.backend_for_category(FileCategory.CSV),
        "pandas",
    )
    self.assertEqual(
        options.selections(),
        {
            FileCategory.READABLE_PDF: "pymupdf",
            FileCategory.CSV: "pandas",
            FileCategory.HTML: "beautifulsoup",
        },
    )

  def test_invalid_backend_name_raises(self):
    with self.assertRaises(ValueError) as cm:
      ParserBackendOptions(readable_pdf_backend="not-a-pdf-parser")

    self.assertIn("readable_pdf", str(cm.exception))
    self.assertIn("not-a-pdf-parser", str(cm.exception))

  def test_auto_backend_selection_is_allowed_for_supported_category(self):
    options = ParserBackendOptions(
        readable_pdf_backend="auto",
        backend_preference_order={
            "readable_pdf": ("pypdf", "pdfplumber"),
        },
    )

    self.assertEqual(
        options.backend_for_category(FileCategory.READABLE_PDF),
        "auto",
    )
    self.assertEqual(
        options.preference_order_for_category(FileCategory.READABLE_PDF),
        ("pypdf", "pdfplumber", "pymupdf"),
    )

  def test_auto_backend_selection_rejects_unsupported_category(self):
    with self.assertRaises(ValueError) as cm:
      ParserBackendOptions(table_pdf_backend="auto")

    self.assertIn("table_pdf", str(cm.exception))

  def test_invalid_backend_preference_order_raises(self):
    with self.assertRaises(ValueError) as cm:
      ParserBackendOptions(
          html_backend="auto",
          backend_preference_order={"html": ("not-a-real-backend",)},
      )

    self.assertIn("html", str(cm.exception))
    self.assertIn("not-a-real-backend", str(cm.exception))

  def test_preference_order_includes_all_backends_without_override(self):
    options = ParserBackendOptions(image_backend="auto")
    order = options.preference_order_for_category(
        ingestion_backends.FileCategory.IMAGE,
    )
    registered_names = {
        b.name for b in get_backends(ingestion_backends.FileCategory.IMAGE)
    }
    self.assertTrue(
        registered_names.issubset(set(order)),
        f"Preference order {order} is missing backends: "
        f"{registered_names - set(order)}",
    )


class IsBackendInstalledTest(absltest.TestCase):

  def test_builtin_is_always_installed(self):
    b = get_backend(FileCategory.TXT, "builtin")
    self.assertTrue(is_backend_installed(b))

  def test_pandas_is_installed(self):
    b = get_backend(FileCategory.CSV, "pandas")
    self.assertTrue(is_backend_installed(b))

  def test_uninstalled_library(self):
    b = IngestionBackend(
        category=FileCategory.TXT,
        name="fake",
        import_check="definitely_not_a_real_package_xyz",
        status=BackendStatus.AVAILABLE,
    )
    self.assertFalse(is_backend_installed(b))


class InspectBackendRuntimeTest(absltest.TestCase):

  def test_builtin_backend_is_usable(self):
    backend = get_backend(FileCategory.TXT, "builtin")

    diagnostics = inspect_backend_runtime(backend)

    self.assertIsInstance(diagnostics, BackendDiagnostics)
    self.assertTrue(diagnostics.installed)
    self.assertTrue(diagnostics.usable)
    self.assertIsNone(diagnostics.unavailable_reason)

  def test_registered_backend_reports_planned(self):
    backend = get_backend(FileCategory.SCANNED_PDF, "ocrmypdf")

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        return_value=True,
    ):
      diagnostics = inspect_backend_runtime(backend)

    self.assertTrue(diagnostics.installed)
    self.assertFalse(diagnostics.usable)
    self.assertIn(
        "registered but not yet implemented", diagnostics.unavailable_reason
    )

  def test_scanned_pdf_reports_missing_pymupdf_runtime_dependency(self):
    backend = get_backend(FileCategory.SCANNED_PDF, "paddleocr")

    def _is_installed(candidate: IngestionBackend) -> bool:
      return candidate.name != "pymupdf"

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      diagnostics = inspect_backend_runtime(backend)

    self.assertTrue(diagnostics.installed)
    self.assertFalse(diagnostics.usable)
    self.assertIn("pymupdf", diagnostics.unavailable_reason)

  def test_excel_pandas_reports_missing_openpyxl_runtime_dependency(self):
    backend = get_backend(FileCategory.EXCEL, "pandas")

    def _is_installed(candidate: IngestionBackend) -> bool:
      return candidate.name != "openpyxl"

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      diagnostics = inspect_backend_runtime(backend)

    self.assertTrue(diagnostics.installed)
    self.assertFalse(diagnostics.usable)
    self.assertIn("openpyxl", diagnostics.unavailable_reason)

  def test_tesseract_reports_missing_system_executable(self):
    backend = get_backend(FileCategory.IMAGE, "tesseract")

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        return_value=True,
    ):
      with mock.patch.object(
          ingestion_backends,
          "_can_import_module",
          return_value=True,
      ):
        with mock.patch(
            "langextract.ingestion_backends.shutil.which", return_value=None
        ):
          diagnostics = inspect_backend_runtime(backend)

    self.assertTrue(diagnostics.installed)
    self.assertFalse(diagnostics.usable)
    self.assertIn("tesseract", diagnostics.unavailable_reason)


class RequireBackendTest(absltest.TestCase):

  def test_default_available_backend(self):
    b = require_backend(FileCategory.TXT)
    self.assertEqual(b.name, "builtin")

  def test_explicit_available_backend(self):
    # builtin txt backend has no import_check — always passes
    b = require_backend(FileCategory.TXT, "builtin")
    self.assertEqual(b.name, "builtin")

  def test_raises_for_unknown_name(self):
    with self.assertRaises(ValueError) as cm:
      require_backend(FileCategory.CSV, "nonexistent")
    self.assertIn("nonexistent", str(cm.exception))
    self.assertIn("csv", str(cm.exception))

  def test_raises_not_implemented_for_registered_only(self):
    with self.assertRaises(NotImplementedError) as cm:
      require_backend(FileCategory.SCANNED_PDF, "ocrmypdf")
    self.assertIn("ocrmypdf", str(cm.exception))
    self.assertIn("not yet implemented", str(cm.exception))
    self.assertIn("tesseract", str(cm.exception))

  def test_raises_value_error_no_default(self):
    with self.assertRaises(ValueError) as cm:
      require_backend(FileCategory.TABLE_PDF)
    self.assertIn("No default", str(cm.exception))

  def test_raises_import_error_for_missing_library(self):
    fake_backend = IngestionBackend(
        category=FileCategory.TXT,
        name="fake_lib",
        import_check="definitely_not_real_xyz",
        pypi_package="fake-lib",
        status=BackendStatus.AVAILABLE,
        is_default=False,
        extras=("html",),
    )
    # Patch the registry temporarily
    original = ingestion_backends._BY_CATEGORY.get(FileCategory.TXT, [])
    ingestion_backends._BY_CATEGORY[FileCategory.TXT] = original + [
        fake_backend
    ]
    try:
      with self.assertRaises(ImportError) as cm:
        require_backend(FileCategory.TXT, "fake_lib")
      self.assertIn("fake_lib", str(cm.exception))
      self.assertIn("definitely_not_real_xyz", str(cm.exception))
      self.assertIn("fake-lib", str(cm.exception))
      self.assertIn("langextract[html]", str(cm.exception))
    finally:
      ingestion_backends._BY_CATEGORY[FileCategory.TXT] = original


class ResolveBackendTest(absltest.TestCase):
  """Tests for resolve_backend()."""

  def test_returns_default_when_no_selection(self):
    resolved = ingestion_backends.resolve_backend(FileCategory.TXT)
    self.assertIsNotNone(resolved)
    self.assertEqual(resolved.name, "builtin")

  def test_returns_none_for_category_without_default(self):
    self.assertIsNone(
        ingestion_backends.resolve_backend(FileCategory.TABLE_PDF)
    )

  def test_explicit_backend_overrides_default(self):
    resolved = ingestion_backends.resolve_backend(
        FileCategory.TXT,
        backend="builtin",
    )
    self.assertEqual(resolved.name, "builtin")

  def test_parser_backends_selection_used(self):
    opts = ParserBackendOptions(csv_backend="pandas")
    resolved = ingestion_backends.resolve_backend(
        FileCategory.CSV,
        parser_backends=opts,
    )
    self.assertEqual(resolved.name, "pandas")

  def test_explicit_backend_beats_parser_backends(self):
    opts = ParserBackendOptions(txt_backend="builtin")
    # backend= kwarg takes priority over parser_backends
    resolved = ingestion_backends.resolve_backend(
        FileCategory.TXT,
        backend="builtin",
        parser_backends=opts,
    )
    self.assertEqual(resolved.name, "builtin")

  def test_raises_for_unknown_backend(self):
    with self.assertRaises(ValueError):
      ingestion_backends.resolve_backend(FileCategory.CSV, backend="nope")

  def test_raises_not_implemented_for_registered(self):
    with self.assertRaises(NotImplementedError):
      ingestion_backends.resolve_backend(
          FileCategory.SCANNED_PDF,
          backend="ocrmypdf",
      )

  def test_html_default_is_trafilatura(self):
    with mock.patch.dict("sys.modules", {"trafilatura": mock.MagicMock()}):
      resolved = ingestion_backends.resolve_backend(FileCategory.HTML)
    self.assertIsNotNone(resolved)
    self.assertEqual(resolved.name, "trafilatura")

  def test_url_default_is_trafilatura(self):
    with mock.patch.dict("sys.modules", {"trafilatura": mock.MagicMock()}):
      resolved = ingestion_backends.resolve_backend(FileCategory.URL)
    self.assertIsNotNone(resolved)
    self.assertEqual(resolved.name, "trafilatura")

  def test_default_backend_missing_dependency_raises(self):
    fake_default = IngestionBackend(
        category=FileCategory.TXT,
        name="fake_default",
        import_check="definitely_not_real_xyz",
        pypi_package="fake-default",
        status=BackendStatus.AVAILABLE,
        is_default=True,
        extras=("html",),
    )
    original = ingestion_backends._BY_CATEGORY.get(FileCategory.TXT, [])
    ingestion_backends._BY_CATEGORY[FileCategory.TXT] = [
        fake_default,
        *[backend for backend in original if not backend.is_default],
    ]
    try:
      with self.assertRaises(ImportError) as cm:
        ingestion_backends.resolve_backend(FileCategory.TXT)
      self.assertIn("fake_default", str(cm.exception))
      self.assertIn("fake-default", str(cm.exception))
      self.assertIn("langextract[html]", str(cm.exception))
    finally:
      ingestion_backends._BY_CATEGORY[FileCategory.TXT] = original

  def test_auto_backend_uses_first_installed_backend(self):
    def _is_installed(backend: IngestionBackend) -> bool:
      return backend.name != "pymupdf"

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      resolved = ingestion_backends.resolve_backend(
          FileCategory.READABLE_PDF,
          backend="auto",
      )

    self.assertEqual(resolved.name, "pdfplumber")

  def test_auto_backend_honors_preference_override(self):
    options = ParserBackendOptions(
        readable_pdf_backend="auto",
        backend_preference_order={
            FileCategory.READABLE_PDF: ("pypdf", "pdfplumber"),
        },
    )

    def _is_installed(backend: IngestionBackend) -> bool:
      return backend.name in {"pdfplumber", "pypdf"}

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      resolved = ingestion_backends.resolve_backend(
          FileCategory.READABLE_PDF,
          parser_backends=options,
      )

    self.assertEqual(resolved.name, "pypdf")

  def test_auto_backend_skips_registered_only_backends(self):
    def _is_installed(backend: IngestionBackend) -> bool:
      return backend.name == "tesseract"

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      resolved = ingestion_backends.resolve_backend(
          FileCategory.SCANNED_PDF,
          backend="auto",
      )

    self.assertEqual(resolved.name, "tesseract")

  def test_auto_backend_all_candidates_exhausted_raises(self):
    def _none_installed(backend: IngestionBackend) -> bool:
      return False

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_none_installed,
    ):
      with self.assertRaises(ImportError) as cm:
        ingestion_backends.resolve_backend(
            FileCategory.READABLE_PDF,
            backend="auto",
        )

    self.assertIn("readable_pdf", str(cm.exception))
    self.assertIn("could not find", str(cm.exception).lower())

  def test_explicit_backend_overrides_auto_parser_backends(self):
    opts = ParserBackendOptions(html_backend="auto")
    resolved = ingestion_backends.resolve_backend(
        FileCategory.HTML,
        backend="builtin",
        parser_backends=opts,
    )
    self.assertEqual(resolved.name, "builtin")

  def test_auto_csv_falls_back_to_builtin_csv(self):
    def _is_installed(backend: IngestionBackend) -> bool:
      return backend.name != "pandas"

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      resolved = ingestion_backends.resolve_backend(
          FileCategory.CSV,
          backend="auto",
      )

    self.assertEqual(resolved.name, "builtin_csv")


class HandlerRegistryTest(absltest.TestCase):
  """Tests for register_handler / get_handler / resolve_handler."""

  def setUp(self):
    super().setUp()
    # Handler registrations live in ingestion.py; importing it triggers them.
    import langextract.ingestion  # noqa: F401

  def test_txt_builtin_handler_registered(self):
    handler = ingestion_backends.get_handler(FileCategory.TXT, "builtin")
    self.assertIsNotNone(handler)

  def test_html_builtin_handler_registered(self):
    handler = ingestion_backends.get_handler(FileCategory.HTML, "builtin")
    self.assertIsNotNone(handler)

  def test_docx_handler_registered(self):
    handler = ingestion_backends.get_handler(FileCategory.DOCX, "python-docx")
    self.assertIsNotNone(handler)

  def test_readable_pdf_handler_registered(self):
    handler = ingestion_backends.get_handler(
        FileCategory.READABLE_PDF, "pdfplumber"
    )
    self.assertIsNotNone(handler)

  def test_csv_pandas_handler_registered(self):
    handler = ingestion_backends.get_handler(FileCategory.CSV, "pandas")
    self.assertIsNotNone(handler)

  def test_excel_pandas_handler_registered(self):
    handler = ingestion_backends.get_handler(FileCategory.EXCEL, "pandas")
    self.assertIsNotNone(handler)

  def test_get_handler_returns_none_for_unregistered(self):
    self.assertIsNone(
        ingestion_backends.get_handler(FileCategory.SCANNED_PDF, "ocrmypdf")
    )

  def test_resolve_handler_dispatches(self):
    result = ingestion_backends.resolve_handler(FileCategory.TXT)
    self.assertIsNotNone(result)
    backend, handler = result
    self.assertEqual(backend.name, "builtin")
    self.assertTrue(callable(handler))

  def test_resolve_handler_returns_none_without_handler(self):
    self.assertIsNone(
        ingestion_backends.resolve_handler(FileCategory.TABLE_PDF)
    )

  def test_resolve_handler_propagates_errors(self):
    with self.assertRaises(NotImplementedError):
      ingestion_backends.resolve_handler(
          FileCategory.SCANNED_PDF,
          backend="ocrmypdf",
      )

  def test_register_handler_rejects_unknown_backend(self):
    with self.assertRaises(ValueError):

      @ingestion_backends.register_handler(FileCategory.TXT, "nonexistent")
      def noop(**kwargs):
        pass

  def test_resolve_handler_auto_selects_first_available_with_handler(self):
    def _is_installed(backend: IngestionBackend) -> bool:
      return backend.name in {"pdfplumber", "pypdf"}

    with mock.patch.object(
        ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      result = ingestion_backends.resolve_handler(
          FileCategory.READABLE_PDF,
          backend="auto",
      )

    self.assertIsNotNone(result)
    backend, handler = result
    self.assertEqual(backend.name, "pdfplumber")
    self.assertTrue(callable(handler))


class PackagingExtrasTest(absltest.TestCase):
  """Tests for pyproject optional dependency groups."""

  def test_backend_family_extras_exist(self):
    pyproject_path = (
        pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    extras = data["project"]["optional-dependencies"]

    for extra_name in (
        "pdf",
        "ocr",
        "office",
        "html",
        "full",
        "xlsx",
        "all",
    ):
      self.assertIn(extra_name, extras)

  def test_extras_include_expected_backend_packages(self):
    pyproject_path = (
        pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    self.assertTrue(any(dep.startswith("pymupdf") for dep in extras["pdf"]))
    self.assertTrue(any(dep.startswith("openpyxl") for dep in extras["office"]))
    self.assertTrue(
        any(dep.startswith("trafilatura") for dep in extras["html"])
    )
    self.assertTrue(any(dep.startswith("pytesseract") for dep in extras["ocr"]))


class RegistryConsistencyTest(absltest.TestCase):
  """Registry-wide invariant checks."""

  def test_each_default_is_available(self):
    for category in FileCategory:
      default = get_default_backend(category)
      if default is not None:
        self.assertEqual(
            default.status,
            BackendStatus.AVAILABLE,
            f"Default backend {default.name!r} for {category.value} "
            f"is not AVAILABLE (status={default.status}).",
        )

  def test_at_most_one_default_per_category(self):
    for category in FileCategory:
      defaults = [b for b in get_backends(category) if b.is_default]
      self.assertLessEqual(
          len(defaults),
          1,
          f"Category {category.value} has multiple defaults: "
          f"{[b.name for b in defaults]}",
      )

  def test_no_duplicate_names_per_category(self):
    for category in FileCategory:
      names = [b.name for b in get_backends(category)]
      self.assertEqual(
          len(names),
          len(set(names)),
          f"Duplicate backend names in {category.value}: {names}",
      )

  def test_all_backends_have_description(self):
    for b in list_all_backends():
      self.assertTrue(
          b.description,
          f"Backend {b.name!r} in {b.category.value} has no description.",
      )


class IngestionBackendWiringTest(absltest.TestCase):
  """Test that ingestion.py dispatches through the backend registry."""

  def test_txt_file_default_backend_succeeds(self):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False
    ) as f:
      f.write("hello world")
      f.flush()
      result = ingestion.normalize_input(pathlib.Path(f.name))
    self.assertTrue(result.is_text)
    self.assertIn("hello world", result.text)

  def test_explicit_registered_only_backend_raises(self):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False
    ) as f:
      f.write("data")
      f.flush()
      with self.assertRaises(NotImplementedError) as cm:
        ingestion.normalize_input(
            pathlib.Path(f.name),
            backend="charset-normalizer",
        )
      self.assertIn("charset-normalizer", str(cm.exception))

  def test_unknown_backend_raises_value_error(self):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False
    ) as f:
      f.write("col1,col2\na,b\n")
      f.flush()
      with self.assertRaises(ValueError) as cm:
        ingestion.normalize_input(
            pathlib.Path(f.name),
            backend="nonexistent_csv_lib",
        )
      self.assertIn("nonexistent_csv_lib", str(cm.exception))

  def test_default_none_on_unsupported_category_falls_through(self):
    """When backend=None and category has no default, existing error wins."""
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
      f.write(b"fake")
      f.flush()
      with self.assertRaises(ingestion.UnsupportedIngestionError):
        ingestion.normalize_input(pathlib.Path(f.name))

  def test_explicit_docx_backend_dispatches(self):
    from langextract import ingestion

    mock_document = mock.MagicMock()
    mock_document.paragraphs = [mock.MagicMock(text="Paragraph text")]
    mock_document.tables = []
    docx_module = mock.MagicMock()
    docx_module.Document.return_value = mock_document

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
      f.write(b"fake")
      f.flush()
      with mock.patch.dict("sys.modules", {"docx": docx_module}):
        result = ingestion.normalize_input(
            pathlib.Path(f.name),
            backend="python-docx",
        )
      self.assertEqual(result.metadata["backend"], "python-docx")
      self.assertIn("Paragraph text", result.text)

  def test_local_html_uses_default_trafilatura_backend(self):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(
        suffix=".html", mode="w", delete=False, encoding="utf-8"
    ) as f:
      f.write("<html><body>Hello</body></html>")
      f.flush()
      with mock.patch.dict("sys.modules", {"trafilatura": mock.MagicMock()}):
        with mock.patch(
            "langextract.ingestion._extract_html_with_trafilatura",
            return_value="Hello",
        ):
          result = ingestion.normalize_input(pathlib.Path(f.name))
    self.assertTrue(result.is_text)
    self.assertEqual(result.metadata["backend"], "trafilatura")
    self.assertEqual(result.text, "Hello")

  def test_explicit_html_backend_dispatches_for_local_html(self):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(
        suffix=".html", mode="w", delete=False, encoding="utf-8"
    ) as f:
      f.write("<html><body>Hello</body></html>")
      f.flush()
      with mock.patch.dict("sys.modules", {"bs4": mock.MagicMock()}):
        with mock.patch(
            "langextract.ingestion._extract_html_with_beautifulsoup",
            return_value="Hello",
        ):
          result = ingestion.normalize_input(
              pathlib.Path(f.name),
              backend="beautifulsoup",
          )
      self.assertEqual(result.metadata["backend"], "beautifulsoup")
      self.assertEqual(result.text, "Hello")

  @mock.patch("langextract.ingestion.requests.get")
  def test_text_url_can_use_txt_backend_selection(self, mock_get):
    from langextract import ingestion

    response = mock.MagicMock()
    response.content = b"downloaded text"
    response.headers = {"Content-Type": "text/plain"}
    response.raise_for_status = mock.MagicMock()
    mock_get.return_value = response

    result = ingestion.normalize_input(
        "https://example.com/article.txt",
        backend="builtin",
    )

    self.assertTrue(result.is_text)
    self.assertEqual(result.text, "downloaded text")

  @mock.patch("langextract.ingestion.requests.get")
  def test_explicit_url_backend_dispatches_for_html_page(self, mock_get):
    from langextract import ingestion

    response = mock.MagicMock()
    response.content = b"<html><body>content</body></html>"
    response.headers = {"Content-Type": "text/html"}
    response.raise_for_status = mock.MagicMock()
    mock_get.return_value = response

    with mock.patch.dict("sys.modules", {"trafilatura": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._extract_html_with_trafilatura",
          return_value="content",
      ):
        result = ingestion.normalize_input(
            "https://example.com/page",
            backend="trafilatura",
        )
    self.assertEqual(result.metadata["backend"], "trafilatura")
    self.assertEqual(result.text, "content")

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_parser_backend_options_for_table_pdf_raise_before_pdf_open(
      self, mock_import_pymupdf
  ):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
      f.write(b"%PDF-1.4")
      f.flush()
      with self.assertRaises(NotImplementedError) as cm:
        ingestion.normalize_input(
            pathlib.Path(f.name),
            parser_backends=ParserBackendOptions(
                table_pdf_backend="camelot",
            ),
        )
      self.assertIn("camelot", str(cm.exception))
    mock_import_pymupdf.assert_not_called()

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_parser_backend_options_for_scanned_pdf_override_ocr_fallback(
      self, mock_import_pymupdf
  ):
    from langextract import ingestion

    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = ""

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import_pymupdf.return_value = mock_pymupdf

    fake_ocr_engine = mock.MagicMock()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
      f.write(b"%PDF-1.4")
      f.flush()
      with mock.patch.dict(
          "sys.modules",
          {"pymupdf": mock.MagicMock(), "pytesseract": mock.MagicMock()},
      ):
        with mock.patch(
            "langextract.ingestion._extract_text_with_tesseract",
            return_value="Scanned text",
        ):
          result = ingestion.normalize_input(
              pathlib.Path(f.name),
              ocr_engine=fake_ocr_engine,
              parser_backends=ParserBackendOptions(
                  scanned_pdf_backend="tesseract",
              ),
          )
      self.assertIn("Scanned text", result.text)
      self.assertEqual(result.metadata["backend"], "tesseract")
    fake_ocr_engine.run_ocr_pdf_page.assert_not_called()

  def test_parser_backend_options_for_local_html_dispatch(self):
    from langextract import ingestion

    with tempfile.NamedTemporaryFile(
        suffix=".html", mode="w", delete=False, encoding="utf-8"
    ) as f:
      f.write("<html><body>Hello</body></html>")
      f.flush()
      with mock.patch.dict("sys.modules", {"bs4": mock.MagicMock()}):
        with mock.patch(
            "langextract.ingestion._extract_html_with_beautifulsoup",
            return_value="Hello",
        ):
          result = ingestion.normalize_input(
              pathlib.Path(f.name),
              parser_backends=ParserBackendOptions(
                  html_backend="beautifulsoup",
              ),
          )
      self.assertEqual(result.metadata["backend"], "beautifulsoup")
      self.assertEqual(result.text, "Hello")


if __name__ == "__main__":
  absltest.main()
