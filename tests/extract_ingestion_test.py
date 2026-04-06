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

"""Integration tests for the extract() to ingestion.normalize_input() seam."""

from __future__ import annotations

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest

import langextract as lx
from langextract.core import data
from langextract.core import types

_MINIMAL_EXAMPLES = [
    lx.data.ExampleData(
        text="Example text",
        extractions=[
            lx.data.Extraction(
                extraction_class="entity",
                extraction_text="example",
            ),
        ],
    )
]


def _mock_model():
  model = mock.MagicMock()
  model.infer.return_value = [
      [types.ScoredOutput(output='{"extractions": []}')]
  ]
  model.requires_fence_output = False
  model.schema = None
  return model


class ExtractNormalizeSeamTest(absltest.TestCase):

  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch(
      "langextract.ingestion.normalize_input",
      wraps=lx.ingestion.normalize_input,
  )
  def test_plain_string_passes_through_normalize(
      self, mock_normalize, mock_create_model
  ):
    mock_create_model.return_value = _mock_model()

    lx.extract(
        text_or_documents="Patient takes Aspirin.",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    args, kwargs = mock_normalize.call_args
    self.assertEqual(args[0], "Patient takes Aspirin.")
    self.assertTrue(kwargs.get("fetch_urls", True))

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_plain_string_default_flow_reaches_annotate_text(
      self, mock_create_model, mock_annotate_text
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="Patient takes Aspirin.",
        extractions=[],
    )

    result = lx.extract(
        text_or_documents="Patient takes Aspirin.",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    mock_annotate_text.assert_called_once()
    _, kwargs = mock_annotate_text.call_args
    self.assertEqual(kwargs["text"], "Patient takes Aspirin.")
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_documents")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch(
      "langextract.ingestion.normalize_input",
      wraps=lx.ingestion.normalize_input,
  )
  def test_document_list_reaches_annotate_documents(
      self, mock_normalize, mock_create_model, mock_annotate_docs
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_docs.return_value = []

    docs = [data.Document(text="doc1"), data.Document(text="doc2")]

    lx.extract(
        text_or_documents=docs,
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    mock_normalize.assert_called_once()
    mock_annotate_docs.assert_called_once()
    _, kwargs = mock_annotate_docs.call_args
    self.assertIs(kwargs["documents"], docs)

  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch(
      "langextract.ingestion.normalize_input",
      wraps=lx.ingestion.normalize_input,
  )
  def test_fetch_urls_false_forwarded(self, mock_normalize, mock_create_model):
    mock_create_model.return_value = _mock_model()

    lx.extract(
        text_or_documents="https://example.com",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        fetch_urls=False,
    )

    _, kwargs = mock_normalize.call_args
    self.assertFalse(kwargs["fetch_urls"])

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_record_list_goes_to_annotate_text(
      self, mock_create_model, mock_annotate_text
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="serialized table",
        extractions=[],
    )

    result = lx.extract(
        text_or_documents=[{"text": "doc1", "id": "1"}],
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    mock_annotate_text.assert_called_once()
    _, kwargs = mock_annotate_text.call_args
    self.assertIn('Row 1: text="doc1" | id="1"', kwargs["text"])
    self.assertIsInstance(result, data.AnnotatedDocument)


class NormalizePathExtractionTest(absltest.TestCase):

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_text_file_goes_to_annotate_text(
      self, mock_create_model, mock_annotate_text
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="File content.",
        extractions=[],
    )

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as handle:
      handle.write("File content.")
      path = pathlib.Path(handle.name)

    try:
      result = lx.extract(
          text_or_documents=path,
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          api_key="k",
      )
    finally:
      path.unlink()

    _, kwargs = mock_annotate_text.call_args
    self.assertEqual(kwargs["text"], "File content.")
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_csv_path_goes_to_annotate_text(
      self, mock_create_model, mock_annotate_text
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="serialized csv",
        extractions=[],
    )

    with tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False, encoding="utf-8"
    ) as handle:
      handle.write("text,id\nhello,1\nworld,2\n")
      path = pathlib.Path(handle.name)

    try:
      lx.extract(
          text_or_documents=path,
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          api_key="k",
      )
    finally:
      path.unlink()

    _, kwargs = mock_annotate_text.call_args
    self.assertIn("Format: csv", kwargs["text"])
    self.assertIn('Row 1: text="hello" | id="1"', kwargs["text"])

  @mock.patch("langextract.annotation.Annotator.annotate_documents")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_single_document_goes_to_annotate_documents(
      self, mock_create_model, mock_annotate_docs
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_docs.return_value = []

    lx.extract(
        text_or_documents=data.Document(text="single doc"),
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    mock_annotate_docs.assert_called_once()

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_pdf_path_goes_to_annotate_text(
      self,
      mock_import_pymupdf,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="Digital PDF text",
        extractions=[],
    )

    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = "Digital PDF text"
    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import_pymupdf.return_value = mock_pymupdf

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
      handle.write(b"%PDF-1.4")
      path = pathlib.Path(handle.name)

    try:
      with mock.patch.dict("sys.modules", {"pymupdf": mock.MagicMock()}):
        result = lx.extract(
            text_or_documents=path,
            prompt_description="desc",
            examples=_MINIMAL_EXAMPLES,
            api_key="k",
        )
    finally:
      path.unlink()

    _, kwargs = mock_annotate_text.call_args
    self.assertIn("Digital PDF text", kwargs["text"])
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.pd.read_excel")
  def test_xlsx_path_goes_to_annotate_text(
      self,
      mock_read_excel,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="serialized xlsx",
        extractions=[],
    )
    mock_read_excel.return_value = lx.ingestion.pd.DataFrame(
        {"text": ["hello"], "id": ["1"]}
    )

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as handle:
      path = pathlib.Path(handle.name)

    try:
      with mock.patch.dict("sys.modules", {"openpyxl": mock.MagicMock()}):
        lx.extract(
            text_or_documents=path,
            prompt_description="desc",
            examples=_MINIMAL_EXAMPLES,
            api_key="k",
        )
    finally:
      path.unlink()

    _, kwargs = mock_annotate_text.call_args
    self.assertIn("Format: xlsx", kwargs["text"])
    self.assertIn('Row 1: text="hello" | id="1"', kwargs["text"])


class ExtractBytesInputTest(absltest.TestCase):
  """Test bytes input through extract()."""

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_utf8_bytes_go_to_annotate_text(
      self, mock_create_model, mock_annotate_text
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="byte text",
        extractions=[],
    )

    result = lx.extract(
        text_or_documents=b"Content from bytes.",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    _, kwargs = mock_annotate_text.call_args
    self.assertEqual(kwargs["text"], "Content from bytes.")
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_pdf_bytes_go_to_annotate_text(
      self,
      mock_import_pymupdf,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="PDF bytes text",
        extractions=[],
    )

    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = "PDF bytes text"
    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import_pymupdf.return_value = mock_pymupdf

    with mock.patch.dict("sys.modules", {"pymupdf": mock.MagicMock()}):
      result = lx.extract(
          text_or_documents=b"%PDF-1.4 rest of pdf",
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          api_key="k",
      )

    _, kwargs = mock_annotate_text.call_args
    self.assertIn("PDF bytes text", kwargs["text"])
    self.assertIsInstance(result, data.AnnotatedDocument)


class ExtractOcrSeamTest(absltest.TestCase):

  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_extract_resolves_and_forwards_ocr_engine(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
  ):
    mock_create_model.return_value = _mock_model()
    resolved_engine = object()
    mock_resolve_ocr_engine.return_value = resolved_engine
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="scan.png",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        ocr_engine="deepseek-ocr",
        ocr_config={"timeout": 7},
    )

    mock_resolve_ocr_engine.assert_called_once_with(
        "deepseek-ocr",
        config={"timeout": 7},
    )
    self.assertIs(
        mock_normalize_input.call_args[1]["ocr_engine"], resolved_engine
    )

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_image_path_can_flow_through_extract_with_ocr_engine(
      self,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="OCR text from image",
        extractions=[],
    )

    class _FakeOcrEngine:

      def run_ocr(self, image_data, *, prompt=None):
        del image_data, prompt
        return lx.ocr.OcrResult(
            text="OCR text from image",
            metadata={"engine": "fake"},
        )

      def run_ocr_pdf_page(self, page_image, *, page_number, prompt=None):
        del page_image, page_number, prompt
        return lx.ocr.OcrResult(
            text="OCR text from PDF page",
            metadata={"engine": "fake"},
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
      handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
      path = pathlib.Path(handle.name)

    try:
      result = lx.extract(
          text_or_documents=path,
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          api_key="k",
          ocr_engine=_FakeOcrEngine(),
      )
    finally:
      path.unlink()

    _, kwargs = mock_annotate_text.call_args
    self.assertEqual(kwargs["text"], "OCR text from image")
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion._import_pymupdf")
  def test_scanned_pdf_can_flow_through_extract_with_ocr_engine(
      self,
      mock_import_pymupdf,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="OCR text from PDF page",
        extractions=[],
    )

    mock_page = mock.MagicMock()
    mock_page.get_text.return_value = ""
    mock_pixmap = mock.MagicMock()
    mock_pixmap.tobytes.return_value = b"fake-png-data"
    mock_page.get_pixmap.return_value = mock_pixmap

    mock_doc = mock.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page

    mock_pymupdf = mock.MagicMock()
    mock_pymupdf.open.return_value = mock_doc
    mock_import_pymupdf.return_value = mock_pymupdf

    class _FakeOcrEngine:

      def run_ocr(self, image_data, *, prompt=None):
        del image_data, prompt
        return lx.ocr.OcrResult(text="unused", metadata={"engine": "fake"})

      def run_ocr_pdf_page(self, page_image, *, page_number, prompt=None):
        del page_image, page_number, prompt
        return lx.ocr.OcrResult(
            text="OCR text from PDF page",
            metadata={"engine": "fake"},
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
      handle.write(b"%PDF-1.4")
      path = pathlib.Path(handle.name)

    try:
      with mock.patch.dict("sys.modules", {"pymupdf": mock.MagicMock()}):
        result = lx.extract(
            text_or_documents=path,
            prompt_description="desc",
            examples=_MINIMAL_EXAMPLES,
            api_key="k",
            ocr_engine=_FakeOcrEngine(),
        )
    finally:
      path.unlink()

    _, kwargs = mock_annotate_text.call_args
    self.assertIn("OCR text from PDF page", kwargs["text"])
    self.assertIsInstance(result, data.AnnotatedDocument)


if __name__ == "__main__":
  absltest.main()
