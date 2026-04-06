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

"""Tests for the digital ingestion layer (langextract.ingestion)."""

from __future__ import annotations

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
import pandas as pd
import requests

from langextract import ingestion
from langextract.core import data


class PassthroughTest(absltest.TestCase):

  def test_plain_string_passthrough(self):
    text = "Patient takes Aspirin 100mg."
    result = ingestion.normalize(text)
    self.assertIs(result, text)

  def test_plain_string_not_fetched_when_disabled(self):
    result = ingestion.normalize("http://example.com", fetch_urls=False)
    self.assertEqual(result, "http://example.com")

  def test_document_wrapped_in_list(self):
    doc = data.Document(text="hello")
    result = ingestion.normalize(doc)
    self.assertIsInstance(result, list)
    self.assertLen(result, 1)
    self.assertIs(result[0], doc)

  def test_list_of_documents_passthrough(self):
    docs = [data.Document(text="a"), data.Document(text="b")]
    result = ingestion.normalize(docs)
    self.assertIs(result, docs)

  def test_unknown_type_raises_clear_error(self):
    with self.assertRaisesRegex(
        ingestion.IngestionError, "Unsupported input type"
    ):
      ingestion.normalize(object())

  def test_mixed_iterables_raise_clear_error(self):
    with self.assertRaisesRegex(ingestion.IngestionError, "Mixed iterables"):
      ingestion.normalize([{"text": "a"}, data.Document(text="b")])


class NormalizedInputModelTest(absltest.TestCase):

  def test_normalize_input_raw_text_returns_model(self):
    normalized = ingestion.normalize_input("Patient takes Aspirin 100mg.")

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.RAW_TEXT)
    self.assertEqual(
        normalized.as_extract_input(), "Patient takes Aspirin 100mg."
    )

  def test_table_like_records_normalize_to_text(self):
    normalized = ingestion.normalize_input(
        [{"text": "row1", "id": "a", "ctx": "extra"}],
        id_column="id",
        additional_context_column="ctx",
    )

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.TABLE)
    self.assertIn("Columns: text | id | ctx", normalized.text)
    self.assertIn('Row 1: text="row1" | id="a" | ctx="extra"', normalized.text)

  def test_single_mapping_normalize_input_returns_table_model(self):
    normalized = ingestion.normalize_input(
        {"text": "row1", "id": "a"},
        id_column="id",
    )

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.TABLE)
    self.assertIn("Columns: text | id", normalized.text)
    self.assertIn('Row 1: text="row1" | id="a"', normalized.text)

  def test_invalid_normalized_input_state_raises(self):
    with self.assertRaisesRegex(ValueError, "exactly one"):
      ingestion.NormalizedInput(
          source_type=ingestion.InputSourceType.RAW_TEXT,
          content_kind=ingestion.NormalizedContentKind.TEXT,
      )


class TextFileTest(absltest.TestCase):

  def test_txt_file(self):
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as handle:
      handle.write("Hello from a file.")
      path = pathlib.Path(handle.name)

    try:
      result = ingestion.normalize(path)
      self.assertEqual(result, "Hello from a file.")
    finally:
      path.unlink()

  def test_md_file(self):
    with tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", delete=False, encoding="utf-8"
    ) as handle:
      handle.write("# Heading")
      path = pathlib.Path(handle.name)

    try:
      result = ingestion.normalize(path)
      self.assertEqual(result, "# Heading")
    finally:
      path.unlink()

  def test_missing_file_raises(self):
    with self.assertRaises(FileNotFoundError):
      ingestion.normalize(pathlib.Path("/nonexistent/file.txt"))

  def test_unsupported_extension_raises(self):
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as handle:
      path = pathlib.Path(handle.name)

    try:
      with self.assertRaises(ingestion.UnsupportedIngestionError):
        ingestion.normalize(path)
    finally:
      path.unlink()


class CsvTest(absltest.TestCase):

  def _write_csv(self, content: str) -> pathlib.Path:
    handle = tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False, encoding="utf-8"
    )
    handle.write(content)
    handle.flush()
    handle.close()
    return pathlib.Path(handle.name)

  def test_csv_serializes_to_text(self):
    path = self._write_csv("text,id\nhello,1\nworld,2\n")

    try:
      result = ingestion.normalize(path, id_column="id")
      self.assertIsInstance(result, str)
      self.assertIn("Format: csv", result)
      self.assertIn('Row 1: text="hello" | id="1"', result)
      self.assertIn('Row 2: text="world" | id="2"', result)
    finally:
      path.unlink()

  def test_csv_custom_column_order(self):
    path = self._write_csv("note,uid,ctx\nalpha,a1,extra\n")

    try:
      result = ingestion.normalize(
          path,
          text_column="note",
          id_column="uid",
          additional_context_column="ctx",
      )
      self.assertIn("Columns: note | uid | ctx", result)
      self.assertIn('Row 1: note="alpha" | uid="a1" | ctx="extra"', result)
    finally:
      path.unlink()

  def test_empty_csv_raises(self):
    path = self._write_csv("")

    try:
      with self.assertRaisesRegex(
          ingestion.IngestionError, "CSV file is empty"
      ):
        ingestion.normalize(path)
    finally:
      path.unlink()

  def test_builtin_csv_backend_serializes_to_text(self):
    path = self._write_csv("text,id\nhello,1\n")

    try:
      result = ingestion.normalize(path, backend="builtin_csv", id_column="id")
      self.assertIn("Format: csv", result)
      self.assertIn('Row 1: text="hello" | id="1"', result)
    finally:
      path.unlink()

  def test_csv_normalize_input_records_selected_backend(self):
    path = self._write_csv("text,id\nhello,1\n")

    try:
      normalized = ingestion.normalize_input(path, id_column="id")
      self.assertEqual(normalized.metadata["backend"], "pandas")
    finally:
      path.unlink()


class DataFrameTest(absltest.TestCase):

  def test_basic_dataframe_serializes_to_text(self):
    df = pd.DataFrame({"text": ["row1", "row2"], "id": ["a", "b"]})
    result = ingestion.normalize(df, id_column="id")

    self.assertIn("Format: table", result)
    self.assertIn('Row 1: text="row1" | id="a"', result)
    self.assertIn('Row 2: text="row2" | id="b"', result)

  def test_custom_columns_are_prioritized_when_present(self):
    df = pd.DataFrame({"note": ["x"], "uid": ["u1"], "ctx": ["extra"]})
    result = ingestion.normalize(
        df,
        text_column="note",
        id_column="uid",
        additional_context_column="ctx",
    )

    self.assertIn("Columns: note | uid | ctx", result)
    self.assertIn('Row 1: note="x" | uid="u1" | ctx="extra"', result)

  def test_empty_dataframe_raises(self):
    df = pd.DataFrame({"text": []})
    with self.assertRaisesRegex(ingestion.IngestionError, "contains no rows"):
      ingestion.normalize(df)

  def test_dataframe_without_columns_raises(self):
    df = pd.DataFrame()
    with self.assertRaisesRegex(ingestion.IngestionError, "has no columns"):
      ingestion.normalize(df)


class XlsxTest(absltest.TestCase):

  @mock.patch("langextract.ingestion.pd.read_excel")
  @mock.patch("langextract.ingestion.pathlib.Path.exists", return_value=True)
  def test_xlsx_serializes_to_text(self, _exists, mock_read_excel):
    mock_read_excel.return_value = pd.DataFrame({"text": ["row"], "id": ["1"]})

    path = pathlib.Path("data.xlsx")
    with mock.patch.dict("sys.modules", {"openpyxl": mock.MagicMock()}):
      result = ingestion.normalize(path, id_column="id")

    self.assertIn("Format: xlsx", result)
    self.assertIn('Row 1: text="row" | id="1"', result)

  @mock.patch("langextract.ingestion.pathlib.Path.exists", return_value=True)
  def test_xlsx_missing_openpyxl_raises(self, _exists):
    with mock.patch.dict("sys.modules", {"openpyxl": None}):
      with self.assertRaises(ImportError) as context:
        ingestion.normalize(pathlib.Path("data.xlsx"))
      self.assertIn("pandas", str(context.exception))
      self.assertIn("openpyxl", str(context.exception))
      self.assertIn("langextract[office]", str(context.exception))
      self.assertIn("langextract[xlsx]", str(context.exception))

  @mock.patch("langextract.ingestion.pd.read_excel")
  @mock.patch("langextract.ingestion.pathlib.Path.exists", return_value=True)
  def test_xlsx_normalize_input_returns_model(self, _exists, mock_read_excel):
    mock_read_excel.return_value = pd.DataFrame({"text": ["row"], "id": ["1"]})

    with mock.patch.dict("sys.modules", {"openpyxl": mock.MagicMock()}):
      normalized = ingestion.normalize_input(
          pathlib.Path("data.xlsx"), id_column="id"
      )

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.XLSX)
    self.assertEqual(normalized.metadata["row_count"], 1)
    self.assertIn("Format: xlsx", normalized.text)

  @mock.patch("langextract.ingestion.pathlib.Path.exists", return_value=True)
  def test_explicit_openpyxl_backend_dispatches(self, _exists):
    workbook = mock.MagicMock()
    workbook.active.iter_rows.return_value = [
        ("text", "id"),
        ("row", "1"),
    ]
    openpyxl_module = mock.MagicMock()
    openpyxl_module.load_workbook.return_value = workbook

    with mock.patch.dict("sys.modules", {"openpyxl": openpyxl_module}):
      normalized = ingestion.normalize_input(
          pathlib.Path("data.xlsx"),
          backend="openpyxl",
          id_column="id",
      )

    self.assertEqual(normalized.metadata["backend"], "openpyxl")
    self.assertIn('Row 1: text="row" | id="1"', normalized.text)


class PdfTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._pymupdf_patch = mock.patch.dict(
        "sys.modules", {"pymupdf": mock.MagicMock()}
    )
    self._pymupdf_patch.start()
    self.addCleanup(self._pymupdf_patch.stop)

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_digital_pdf_extracts_text(self, mock_import):
    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = "Digital PDF text"

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    with mock.patch(
        "langextract.ingestion.pathlib.Path.exists", return_value=True
    ):
      result = ingestion.normalize(pathlib.Path("report.pdf"))

    self.assertIn("[Page 1]", result)
    self.assertIn("Digital PDF text", result)

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_scanned_pdf_without_text_raises_missing_default_backend_dependency(
      self, mock_import
  ):
    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = ""

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    with mock.patch(
        "langextract.ingestion.pathlib.Path.exists", return_value=True
    ):
      with self.assertRaises(ImportError) as context:
        ingestion.normalize(pathlib.Path("scan.pdf"))

    self.assertIn("paddleocr", str(context.exception))
    self.assertIn("langextract[ocr]", str(context.exception))

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_mixed_pdf_tracks_empty_pages(self, mock_import):
    text_page = mock.MagicMock()
    text_page.get_text.return_value = "Page 1 text"
    empty_page = mock.MagicMock()
    empty_page.get_text.return_value = ""

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = [text_page, empty_page]

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    with mock.patch(
        "langextract.ingestion.pathlib.Path.exists", return_value=True
    ):
      normalized = ingestion.normalize_input(pathlib.Path("mixed.pdf"))

    self.assertEqual(normalized.source_type, ingestion.InputSourceType.PDF)
    self.assertEqual(normalized.metadata["empty_pages"], [2])
    self.assertIn("Page 1 text", normalized.text)

  def test_pdf_missing_pymupdf_raises(self):
    with mock.patch.dict("sys.modules", {"pymupdf": None}):
      with self.assertRaises(ImportError) as context:
        ingestion._import_pymupdf()
      self.assertIn("pymupdf", str(context.exception))
      self.assertIn("langextract[pdf]", str(context.exception))

  def test_explicit_pypdf_backend_extracts_text(self):
    fake_page = mock.MagicMock()
    fake_page.extract_text.return_value = "Pypdf text"
    pypdf_module = mock.MagicMock()
    pypdf_module.PdfReader.return_value.pages = [fake_page]

    with mock.patch.dict("sys.modules", {"pypdf": pypdf_module}):
      with mock.patch(
          "langextract.ingestion.pathlib.Path.exists", return_value=True
      ):
        result = ingestion.normalize(
            pathlib.Path("report.pdf"), backend="pypdf"
        )

    self.assertIn("Pypdf text", result)

  def test_pdf_auto_backend_falls_back_to_available_reader(self):
    fake_page = mock.MagicMock()
    fake_page.extract_text.return_value = "Pypdf text"
    pypdf_module = mock.MagicMock()
    pypdf_module.PdfReader.return_value.pages = [fake_page]

    def _is_installed(backend):
      return backend.name == "pypdf"

    with mock.patch.object(
        ingestion.ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      with mock.patch.dict("sys.modules", {"pypdf": pypdf_module}):
        with mock.patch(
            "langextract.ingestion.pathlib.Path.exists",
            return_value=True,
        ):
          normalized = ingestion.normalize_input(
              pathlib.Path("report.pdf"),
              backend="auto",
          )

    self.assertEqual(normalized.metadata["backend"], "pypdf")
    self.assertIn("Pypdf text", normalized.text)


class ImageTest(absltest.TestCase):

  def test_image_path_uses_default_backend(self):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
      handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
      path = pathlib.Path(handle.name)

    try:
      with mock.patch.dict("sys.modules", {"paddleocr": mock.MagicMock()}):
        with mock.patch(
            "langextract.ingestion._extract_text_with_paddleocr_path",
            return_value="Image OCR",
        ):
          result = ingestion.normalize(path)
      self.assertEqual(result, "Image OCR")
    finally:
      path.unlink()

  def test_explicit_tesseract_backend_supports_image_path(self):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
      handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
      path = pathlib.Path(handle.name)

    try:
      with mock.patch.dict("sys.modules", {"pytesseract": mock.MagicMock()}):
        with mock.patch(
            "langextract.ingestion._extract_text_with_tesseract",
            return_value="Image OCR",
        ):
          normalized = ingestion.normalize_input(path, backend="tesseract")
      self.assertEqual(normalized.metadata["backend"], "tesseract")
      self.assertEqual(normalized.text, "Image OCR")
    finally:
      path.unlink()


class BytesTest(absltest.TestCase):

  @mock.patch("langextract.ingestion._from_pdf_bytes")
  def test_pdf_magic_bytes(self, mock_pdf):
    mock_pdf.return_value = ingestion.NormalizedInput.from_text(
        "pdf text",
        source_type=ingestion.InputSourceType.PDF,
    )

    result = ingestion.normalize(b"%PDF-1.4 rest of file")
    self.assertEqual(result, "pdf text")
    mock_pdf.assert_called_once()

  def test_utf8_text_bytes(self):
    result = ingestion.normalize("Plain text bytes".encode("utf-8"))
    self.assertEqual(result, "Plain text bytes")

  def test_bytearray_text_bytes_return_bytes_source_type(self):
    normalized = ingestion.normalize_input(bytearray(b"Plain text bytes"))

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.BYTES)
    self.assertEqual(normalized.text, "Plain text bytes")
    self.assertEqual(normalized.metadata["origin"], "bytes")

  def test_memoryview_text_bytes_return_bytes_source_type(self):
    normalized = ingestion.normalize_input(memoryview(b"Plain text bytes"))

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.BYTES)
    self.assertEqual(normalized.text, "Plain text bytes")
    self.assertEqual(normalized.metadata["origin"], "bytes")

  def test_image_bytes_use_default_backend(self):
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    with mock.patch.dict("sys.modules", {"paddleocr": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._extract_text_with_paddleocr_bytes",
          return_value="Bytes OCR",
      ):
        result = ingestion.normalize(raw)
    self.assertEqual(result, "Bytes OCR")

  def test_unknown_bytes_raise(self):
    with self.assertRaises(ingestion.IngestionError):
      ingestion.normalize(b"\x00\x01\x02\x03")

  def test_image_bytes_use_selected_backend(self):
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    with mock.patch.dict("sys.modules", {"pytesseract": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._extract_text_with_tesseract",
          return_value="Bytes OCR",
      ):
        normalized = ingestion.normalize_input(raw, backend="tesseract")

    self.assertEqual(normalized.source_type, ingestion.InputSourceType.IMAGE)
    self.assertEqual(normalized.metadata["backend"], "tesseract")
    self.assertEqual(normalized.text, "Bytes OCR")


class UrlTest(absltest.TestCase):

  @mock.patch("langextract.ingestion.requests.get")
  def test_text_url_returns_text(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"downloaded text"
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    result = ingestion.normalize("https://example.com/article.txt")
    self.assertEqual(result, "downloaded text")

  @mock.patch("langextract.ingestion.requests.get")
  def test_csv_url_serializes_to_text(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"text,id\nhello,1\nworld,2\n"
    mock_response.headers = {"Content-Type": "text/csv"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    result = ingestion.normalize(
        "https://example.com/report.csv", id_column="id"
    )
    self.assertIn("Format: csv", result)
    self.assertIn('Row 1: text="hello" | id="1"', result)

  @mock.patch("langextract.ingestion.requests.get")
  def test_csv_url_normalize_input_returns_model(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"text,id\nhello,1\n"
    mock_response.headers = {"Content-Type": "text/csv"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    normalized = ingestion.normalize_input(
        "https://example.com/report.csv",
        id_column="id",
    )

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.CSV)
    self.assertEqual(normalized.metadata["backend"], "pandas")
    self.assertEqual(
        normalized.metadata["url"], "https://example.com/report.csv"
    )
    self.assertIn('Row 1: text="hello" | id="1"', normalized.text)

  @mock.patch("langextract.ingestion.requests.get")
  def test_html_url_uses_default_trafilatura_backend(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"<html><body>downloaded text</body></html>"
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    with mock.patch.dict("sys.modules", {"trafilatura": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._extract_html_with_trafilatura",
          return_value="downloaded text",
      ):
        normalized = ingestion.normalize_input("https://example.com/article")

    self.assertEqual(normalized.metadata["backend"], "trafilatura")
    self.assertEqual(normalized.text, "downloaded text")

  @mock.patch("langextract.ingestion.requests.get")
  def test_html_url_auto_backend_uses_first_available_parser(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"<html><body>downloaded text</body></html>"
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    def _is_installed(backend):
      return backend.name == "beautifulsoup"

    with mock.patch.object(
        ingestion.ingestion_backends,
        "is_backend_installed",
        side_effect=_is_installed,
    ):
      with mock.patch.dict("sys.modules", {"bs4": mock.MagicMock()}):
        with mock.patch(
            "langextract.ingestion._extract_html_with_beautifulsoup",
            return_value="downloaded text",
        ):
          normalized = ingestion.normalize_input(
              "https://example.com/article",
              backend="auto",
          )

    self.assertEqual(normalized.metadata["backend"], "beautifulsoup")
    self.assertEqual(normalized.text, "downloaded text")

  @mock.patch("langextract.ingestion.requests.get")
  def test_remote_pdf_is_placeholder_only(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"%PDF-1.4"
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    with self.assertRaisesRegex(
        ingestion.UnsupportedIngestionError, "Download the file locally"
    ):
      ingestion.normalize("https://example.com/report.pdf")

  @mock.patch(
      "langextract.ingestion.requests.get",
      side_effect=requests.RequestException("boom"),
  )
  def test_request_failure_raises_url_fetch_error(self, _mock_get):
    with self.assertRaisesRegex(ingestion.UrlFetchError, "Failed to fetch URL"):
      ingestion.normalize("https://example.com/article")

  @mock.patch("langextract.ingestion.requests.get")
  def test_html_url_with_beautifulsoup_backend(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.content = b"<html><body>downloaded text</body></html>"
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.raise_for_status = mock.MagicMock()
    mock_get.return_value = mock_response

    with mock.patch.dict("sys.modules", {"bs4": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._extract_html_with_beautifulsoup",
          return_value="downloaded text",
      ):
        normalized = ingestion.normalize_input(
            "https://example.com/article",
            backend="beautifulsoup",
        )

    self.assertEqual(normalized.metadata["backend"], "beautifulsoup")
    self.assertEqual(normalized.text, "downloaded text")


class ExceptionTest(absltest.TestCase):

  def test_exception_hierarchy(self):
    from langextract.core import exceptions as core_exceptions

    self.assertTrue(
        issubclass(ingestion.IngestionError, core_exceptions.LangExtractError)
    )
    self.assertTrue(
        issubclass(ingestion.UrlFetchError, ingestion.IngestionError)
    )
    self.assertTrue(
        issubclass(
            ingestion.UnsupportedIngestionError, ingestion.IngestionError
        )
    )
    self.assertTrue(
        issubclass(ingestion.PdfTextExtractionError, ingestion.IngestionError)
    )


# ---------------------------------------------------------------------------
# OCR integration
# ---------------------------------------------------------------------------


class _FakeOcrEngine:
  """Minimal stub satisfying the OcrEngine duck-type."""

  def __init__(self, text="OCR result text"):
    self._text = text
    self.calls = []

  def run_ocr(self, image_data, *, prompt=None):
    from langextract.ocr import OcrResult

    self.calls.append(("run_ocr", len(image_data)))
    return OcrResult(text=self._text, metadata={"engine": "fake"})

  def run_ocr_pdf_page(self, page_image, *, page_number, prompt=None):
    from langextract.ocr import OcrResult

    self.calls.append(("run_ocr_pdf_page", page_number))
    return OcrResult(
        text=f"OCR page {page_number}",
        metadata={"engine": "fake"},
    )


class OcrImageIngestionTest(absltest.TestCase):
  """Test that images are routed through OCR when an engine is provided."""

  def test_image_file_dispatches_to_ocr(self):
    engine = _FakeOcrEngine(text="Scanned text from image.")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
      handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
      path = pathlib.Path(handle.name)

    try:
      result = ingestion.normalize(path, ocr_engine=engine)
      self.assertEqual(result, "Scanned text from image.")
      self.assertLen(engine.calls, 1)
    finally:
      path.unlink()

  def test_image_file_normalize_input_returns_model(self):
    engine = _FakeOcrEngine(text="OCR output")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
      handle.write(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
      path = pathlib.Path(handle.name)

    try:
      normalized = ingestion.normalize_input(path, ocr_engine=engine)
      self.assertTrue(normalized.is_text)
      self.assertEqual(normalized.source_type, ingestion.InputSourceType.IMAGE)
      self.assertEqual(normalized.text, "OCR output")
      self.assertEqual(normalized.metadata["format"], "ocr")
    finally:
      path.unlink()

  def test_image_bytes_dispatches_to_ocr(self):
    engine = _FakeOcrEngine(text="Bytes OCR output")
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    normalized = ingestion.normalize_input(raw, ocr_engine=engine)
    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.IMAGE)
    self.assertEqual(normalized.text, "Bytes OCR output")

  def test_image_bytes_use_default_backend_without_engine(self):
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    with mock.patch.dict("sys.modules", {"paddleocr": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._extract_text_with_paddleocr_bytes",
          return_value="Bytes OCR output",
      ):
        normalized = ingestion.normalize_input(raw)
    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.IMAGE)
    self.assertEqual(normalized.metadata["backend"], "paddleocr")
    self.assertEqual(normalized.text, "Bytes OCR output")


class OcrPdfIngestionTest(absltest.TestCase):
  """Test that scanned PDFs are OCR'd page-by-page when engine provided."""

  def setUp(self):
    super().setUp()
    self._pymupdf_patch = mock.patch.dict(
        "sys.modules", {"pymupdf": mock.MagicMock()}
    )
    self._pymupdf_patch.start()
    self.addCleanup(self._pymupdf_patch.stop)

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_scanned_pdf_dispatches_to_ocr(self, mock_import):
    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = ""
    mock_pixmap = mock.MagicMock()
    mock_pixmap.tobytes.return_value = b"fake-png-data"
    mock_page.get_pixmap.return_value = mock_pixmap

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    engine = _FakeOcrEngine()

    with mock.patch(
        "langextract.ingestion.pathlib.Path.exists", return_value=True
    ):
      normalized = ingestion.normalize_input(
          pathlib.Path("scan.pdf"),
          ocr_engine=engine,
      )

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.PDF)
    self.assertIn("[Page 1]", normalized.text)
    self.assertIn("[Page 2]", normalized.text)
    self.assertIn("OCR page 1", normalized.text)
    self.assertIn("OCR page 2", normalized.text)
    self.assertEqual(normalized.metadata["ocr_pages"], [1, 2])
    self.assertLen(engine.calls, 2)

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_digital_pdf_ignores_ocr_engine(self, mock_import):
    """Digital PDFs should use text extraction even when engine provided."""
    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = "Digital text"

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    engine = _FakeOcrEngine()

    with mock.patch(
        "langextract.ingestion.pathlib.Path.exists", return_value=True
    ):
      normalized = ingestion.normalize_input(
          pathlib.Path("report.pdf"),
          ocr_engine=engine,
      )

    self.assertIn("Digital text", normalized.text)
    self.assertLen(engine.calls, 0)  # OCR engine was NOT called

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_scanned_pdf_uses_default_backend_without_engine(self, mock_import):
    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = ""

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    with mock.patch.dict("sys.modules", {"paddleocr": mock.MagicMock()}):
      with mock.patch(
          "langextract.ingestion._ocr_pdf_with_backend",
          return_value=(
              "Default scanned text",
              {"backend": "paddleocr", "page_count": 1, "ocr_pages": [1]},
          ),
      ):
        with mock.patch(
            "langextract.ingestion.pathlib.Path.exists", return_value=True
        ):
          normalized = ingestion.normalize_input(pathlib.Path("scan.pdf"))

    self.assertTrue(normalized.is_text)
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.PDF)
    self.assertEqual(normalized.metadata["backend"], "paddleocr")
    self.assertIn("Default scanned text", normalized.text)

  def test_input_source_type_has_image(self):
    self.assertEqual(ingestion.InputSourceType.IMAGE.value, "image")


class MultiPagePdfTest(absltest.TestCase):
  """Test multi-page PDF text concatenation and metadata."""

  def setUp(self):
    super().setUp()
    self._pymupdf_patch = mock.patch.dict(
        "sys.modules", {"pymupdf": mock.MagicMock()}
    )
    self._pymupdf_patch.start()
    self.addCleanup(self._pymupdf_patch.stop)

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_multi_page_pdf_joins_pages_with_headers(self, mock_import):
    pages = ["First page text.", "Second page text.", "Third page text."]
    mock_pages = []
    for page_text in pages:
      page = mock.MagicMock()
      page.get_text.return_value = page_text
      mock_pages.append(page)

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 3
    mock_doc.__getitem__.side_effect = mock_pages

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    with mock.patch(
        "langextract.ingestion.pathlib.Path.exists", return_value=True
    ):
      normalized = ingestion.normalize_input(pathlib.Path("multi.pdf"))

    self.assertEqual(normalized.source_type, ingestion.InputSourceType.PDF)
    self.assertIn("[Page 1]", normalized.text)
    self.assertIn("[Page 2]", normalized.text)
    self.assertIn("[Page 3]", normalized.text)
    # Pages joined with double newline separator
    self.assertIn("First page text.", normalized.text)
    self.assertIn("Third page text.", normalized.text)
    self.assertEqual(normalized.metadata["text_pages"], [1, 2, 3])
    self.assertEqual(normalized.metadata["page_count"], 3)
    self.assertEqual(normalized.metadata["empty_pages"], [])

  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_pdf_bytes_also_extracts_text(self, mock_import):
    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = "Bytes PDF text"

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import.return_value = mock_pymupdf

    normalized = ingestion.normalize_input(b"%PDF-1.4 rest of bytes")
    self.assertEqual(normalized.source_type, ingestion.InputSourceType.PDF)
    self.assertIn("Bytes PDF text", normalized.text)
    self.assertEqual(normalized.metadata["origin"], "bytes")


class IngestionOptionsValidationTest(absltest.TestCase):
  """Test IngestionOptions edge cases."""

  def test_empty_text_column_raises(self):
    from langextract.extraction import IngestionOptions

    with self.assertRaisesRegex(ValueError, "non-empty string"):
      IngestionOptions(text_column="")

  def test_default_ingestion_options_are_valid(self):
    from langextract.extraction import IngestionOptions

    opts = IngestionOptions()
    self.assertEqual(opts.text_column, "text")
    self.assertTrue(opts.fetch_urls)
    self.assertIsNone(opts.id_column)

  def test_table_with_nan_values_quoted_as_empty(self):
    """NaN values in table cells should be serialized as empty strings."""
    import numpy as np

    df = pd.DataFrame({"text": ["hello", np.nan], "id": ["1", "2"]})
    result = ingestion.normalize(df)
    self.assertIn('text="hello"', result)
    self.assertIn('text=""', result)  # NaN → empty string

  def test_table_with_non_string_types_coerced(self):
    """Numeric and mixed types in table cells should be stringified."""
    df = pd.DataFrame({"text": [42, 3.14], "id": ["a", "b"]})
    result = ingestion.normalize(df)
    # pandas promotes int+float to float, so 42 → "42.0"
    self.assertIn('text="42.0"', result)
    self.assertIn('text="3.14"', result)


if __name__ == "__main__":
  absltest.main()
