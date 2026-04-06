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

"""Regression tests for the user-facing backend selection system.

Covers:
  1. Backend registry resolution
  2. Config validation (parse_extraction_config / load_extraction_config)
  3. Config precedence (direct kwargs > config file > defaults)
  4. Backend listing command
  5. CLI extract argument parsing
  6. Missing optional dependency errors
  7. Auto backend fallback behavior
  8. Backward compatibility for default usage
"""

from __future__ import annotations

import dataclasses
from io import StringIO
import json
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest

from langextract import cli
from langextract import extraction as extraction_mod
import langextract as lx
from langextract.core import data
from langextract.ingestion_backends import BackendStatus
from langextract.ingestion_backends import FileCategory
from langextract.ingestion_backends import get_backend
from langextract.ingestion_backends import get_backends
from langextract.ingestion_backends import get_default_backend
from langextract.ingestion_backends import IngestionBackend
from langextract.ingestion_backends import is_backend_installed
from langextract.ingestion_backends import ParserBackendOptions
from langextract.ingestion_backends import require_backend
from langextract.ingestion_backends import resolve_backend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLES = [
    data.ExampleData(
        text="Ada wrote the first algorithm.",
        extractions=[
            data.Extraction(
                extraction_class="person",
                extraction_text="Ada",
            )
        ],
    )
]

_DESCRIPTION = "Extract people"


# ===================================================================
# 1. Backend registry resolution
# ===================================================================


class ResolveBackendPriorityTest(absltest.TestCase):
  """Test resolve_backend() priority: explicit > parser_backends > default."""

  @mock.patch(
      "langextract.ingestion_backends.is_backend_installed",
      return_value=True,
  )
  def test_explicit_backend_overrides_parser_backends(self, _):
    opts = ParserBackendOptions(readable_pdf_backend="pypdf")
    result = resolve_backend(
        FileCategory.READABLE_PDF,
        backend="pdfplumber",
        parser_backends=opts,
    )
    self.assertEqual(result.name, "pdfplumber")

  @mock.patch(
      "langextract.ingestion_backends.is_backend_installed",
      return_value=True,
  )
  def test_parser_backends_overrides_default(self, _):
    opts = ParserBackendOptions(readable_pdf_backend="pypdf")
    result = resolve_backend(
        FileCategory.READABLE_PDF,
        parser_backends=opts,
    )
    self.assertEqual(result.name, "pypdf")

  @mock.patch(
      "langextract.ingestion_backends.is_backend_installed",
      return_value=True,
  )
  def test_falls_back_to_default_when_nothing_set(self, _):
    result = resolve_backend(FileCategory.READABLE_PDF)
    default = get_default_backend(FileCategory.READABLE_PDF)
    self.assertIsNotNone(result)
    self.assertEqual(result.name, default.name)

  def test_returns_none_for_category_without_default(self):
    result = resolve_backend(FileCategory.DOC)
    self.assertIsNone(result)

  @mock.patch(
      "langextract.ingestion_backends.is_backend_installed",
      return_value=True,
  )
  def test_auto_resolves_to_first_installed_backend(self, _):
    result = resolve_backend(
        FileCategory.READABLE_PDF,
        backend="auto",
    )
    self.assertIsNotNone(result)
    self.assertIn(result.name, ["pymupdf", "pdfplumber", "pypdf"])

  def test_unknown_backend_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, r"Unknown ingestion backend"):
      resolve_backend(FileCategory.CSV, backend="nonexistent")


class RequireBackendErrorsTest(absltest.TestCase):
  """Test require_backend() raises on registered-only or missing-dep backends."""

  def test_registered_only_raises_not_implemented(self):
    with self.assertRaisesRegex(NotImplementedError, r"registered but not yet"):
      require_backend(FileCategory.SCANNED_PDF, "ocrmypdf")

  def test_unavailable_library_raises_import_error(self):
    fake = IngestionBackend(
        category=FileCategory.TXT,
        name="fake_txt_backend",
        pypi_package="some-nonexistent-lib",
        import_check="some_nonexistent_lib",
        status=BackendStatus.AVAILABLE,
    )
    with (
        mock.patch(
            "langextract.ingestion_backends.get_backend",
            return_value=fake,
        ),
        mock.patch(
            "langextract.ingestion_backends.is_backend_installed",
            return_value=False,
        ),
    ):
      with self.assertRaises(ImportError):
        require_backend(FileCategory.TXT, "fake_txt_backend")

  def test_no_default_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, r"No default"):
      require_backend(FileCategory.DOC)


# ===================================================================
# 2. Config validation (parse_extraction_config)
# ===================================================================


class ParseExtractionConfigValidationTest(absltest.TestCase):
  """Test parse_extraction_config validates aliases, types, and conflicts."""

  def test_none_returns_default_options(self):
    result = lx.parse_extraction_config(None)
    self.assertIsInstance(result, lx.ExtractionOptions)
    self.assertIsNone(result.model_config)

  def test_passthrough_existing_options(self):
    original = lx.ExtractionOptions.for_model(model_id="gemini-2.5-flash")
    result = lx.parse_extraction_config(original)
    self.assertIs(result, original)

  def test_simple_alias_config(self):
    raw = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "fetch_urls": False,
        "ocr_engine": "deepseek-ocr",
        "backends": {
            "readable_pdf": "pymupdf",
            "url": "beautifulsoup",
        },
    }
    config = lx.parse_extraction_config(raw)
    self.assertEqual(config.model_config.provider, "openai")
    self.assertEqual(config.model_config.model_id, "gpt-4o-mini")
    self.assertFalse(config.ingestion.fetch_urls)
    self.assertEqual(config.ocr.engine, "deepseek-ocr")
    self.assertEqual(
        config.ingestion.parser_backends.readable_pdf_backend, "pymupdf"
    )
    self.assertEqual(
        config.ingestion.parser_backends.url_backend, "beautifulsoup"
    )

  def test_model_and_model_id_conflict_raises(self):
    with self.assertRaisesRegex(ValueError, r"[Cc]onflicting"):
      lx.parse_extraction_config({
          "model": "gpt-4o-mini",
          "model_id": "gemini-2.5-flash",
      })

  def test_model_and_model_id_same_value_deduplicates(self):
    config = lx.parse_extraction_config({
        "model": "gemini-2.5-flash",
        "model_id": "gemini-2.5-flash",
    })
    self.assertEqual(config.model_config.model_id, "gemini-2.5-flash")

  def test_unknown_top_level_key_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, r"Unknown extraction config"):
      lx.parse_extraction_config({"temperature": 0.5})

  def test_unknown_backend_alias_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, r"Unknown backend config"):
      lx.parse_extraction_config({
          "backends": {"nonexistent_cat": "some_backend"},
      })

  def test_invalid_backend_name_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, r"Invalid backend"):
      lx.parse_extraction_config({
          "backends": {"readable_pdf": "not-a-real-parser"},
      })

  def test_non_string_model_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, r"string"):
      lx.parse_extraction_config({"model": 42})

  def test_non_bool_fetch_urls_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, r"boolean"):
      lx.parse_extraction_config({"fetch_urls": "yes"})

  def test_non_mapping_raw_raises_type_error(self):
    with self.assertRaisesRegex(TypeError, r"Expected an object"):
      lx.parse_extraction_config("not-a-dict")

  def test_backend_short_name_aliases(self):
    """Backend aliases without '_backend' suffix should work."""
    config = lx.parse_extraction_config({
        "backends": {
            "csv": "builtin_csv",
            "excel": "openpyxl",
            "html": "builtin",
        },
    })
    pb = config.ingestion.parser_backends
    self.assertEqual(pb.csv_backend, "builtin_csv")
    self.assertEqual(pb.excel_backend, "openpyxl")
    self.assertEqual(pb.html_backend, "builtin")

  def test_backend_default_string_is_skipped(self):
    """The string 'default' in backends config should be treated as unset."""
    config = lx.parse_extraction_config({
        "backends": {
            "readable_pdf": "default",
            "csv": "builtin_csv",
        },
    })
    self.assertIsNone(config.ingestion.parser_backends.readable_pdf_backend)
    self.assertEqual(
        config.ingestion.parser_backends.csv_backend, "builtin_csv"
    )

  def test_backend_none_value_is_skipped(self):
    config = lx.parse_extraction_config({
        "backends": {"readable_pdf": None},
    })
    self.assertIsNone(config.ingestion.parser_backends.readable_pdf_backend)

  def test_ingestion_aliases_forwarded(self):
    config = lx.parse_extraction_config({
        "text_column": "body",
        "id_column": "doc_id",
        "additional_context_column": "notes",
    })
    self.assertEqual(config.ingestion.text_column, "body")
    self.assertEqual(config.ingestion.id_column, "doc_id")
    self.assertEqual(config.ingestion.additional_context_column, "notes")


class LoadExtractionConfigTest(absltest.TestCase):
  """Test load_extraction_config for different file formats."""

  def _write_temp(self, content: str, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    f.write(content)
    f.flush()
    f.close()
    return f.name

  def test_yaml_file(self):
    path = self._write_temp(
        "provider: gemini\nmodel: gemini-2.5-flash\n", ".yaml"
    )
    try:
      config = lx.load_extraction_config(path)
      self.assertEqual(config.model_config.model_id, "gemini-2.5-flash")
    finally:
      pathlib.Path(path).unlink()

  def test_json_file(self):
    path = self._write_temp(
        json.dumps({"provider": "openai", "model": "gpt-4o"}), ".json"
    )
    try:
      config = lx.load_extraction_config(path)
      self.assertEqual(config.model_config.model_id, "gpt-4o")
      self.assertEqual(config.model_config.provider, "openai")
    finally:
      pathlib.Path(path).unlink()

  def test_toml_file(self):
    path = self._write_temp(
        'provider = "gemini"\nmodel = "gemini-2.5-flash"\n', ".toml"
    )
    try:
      config = lx.load_extraction_config(path)
      self.assertEqual(config.model_config.model_id, "gemini-2.5-flash")
    finally:
      pathlib.Path(path).unlink()

  def test_unsupported_suffix_raises(self):
    path = self._write_temp("data", ".ini")
    try:
      with self.assertRaisesRegex(ValueError, r"Unsupported config file"):
        lx.load_extraction_config(path)
    finally:
      pathlib.Path(path).unlink()

  def test_malformed_json_raises(self):
    path = self._write_temp("{bad json", ".json")
    try:
      with self.assertRaises(json.JSONDecodeError):
        lx.load_extraction_config(path)
    finally:
      pathlib.Path(path).unlink()

  def test_non_mapping_top_level_raises(self):
    path = self._write_temp(json.dumps(["a", "b"]), ".json")
    try:
      with self.assertRaisesRegex(TypeError, r"object/table"):
        lx.load_extraction_config(path)
    finally:
      pathlib.Path(path).unlink()

  def test_empty_file_returns_defaults(self):
    path = self._write_temp("{}", ".json")
    try:
      config = lx.load_extraction_config(path)
      self.assertIsNone(config.model_config)
    finally:
      pathlib.Path(path).unlink()

  def test_yaml_with_backends_section(self):
    content = (
        "provider: gemini\n"
        "model: gemini-2.5-flash\n"
        "ocr_engine: deepseek-ocr\n"
        "fetch_urls: true\n"
        "backends:\n"
        "  readable_pdf: auto\n"
        "  scanned_pdf: paddleocr\n"
        "  url: trafilatura\n"
    )
    path = self._write_temp(content, ".yaml")
    try:
      config = lx.load_extraction_config(path)
      self.assertEqual(config.model_config.provider, "gemini")
      self.assertTrue(config.ingestion.fetch_urls)
      self.assertEqual(config.ocr.engine, "deepseek-ocr")
      pb = config.ingestion.parser_backends
      self.assertEqual(pb.readable_pdf_backend, "auto")
      self.assertEqual(pb.scanned_pdf_backend, "paddleocr")
      self.assertEqual(pb.url_backend, "trafilatura")
    finally:
      pathlib.Path(path).unlink()


# ===================================================================
# 3. Config precedence: direct args > config > defaults
# ===================================================================


class BackendKwargPrecedenceTest(absltest.TestCase):
  """Test that convenience kwargs > parser_backends > config defaults."""

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_convenience_kwarg_beats_options_parser_backends(
      self, mock_create_model, mock_annotator_cls
  ):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.schema = None
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = (
        mock.MagicMock()
    )

    with mock.patch(
        "langextract.extraction._resolve_ingestion_options",
        wraps=extraction_mod._resolve_ingestion_options,
    ) as spy:
      lx.extract(
          text_or_documents="test",
          prompt_description=_DESCRIPTION,
          examples=_EXAMPLES,
          use_schema_constraints=False,
          options=lx.ExtractionOptions.for_model(
              ingestion=lx.IngestionOptions(
                  parser_backends=lx.ParserBackendOptions(
                      readable_pdf_backend="pypdf"
                  ),
              )
          ),
          readable_pdf_backend="pdfplumber",
          prompt_validation_level=lx.extraction.pv.PromptValidationLevel.OFF,
      )
      call_kwargs = spy.call_args.kwargs
      self.assertEqual(call_kwargs["readable_pdf_backend"], "pdfplumber")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_parser_backends_kwarg_beats_options(
      self, mock_create_model, mock_annotator_cls
  ):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.schema = None
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = (
        mock.MagicMock()
    )

    with mock.patch(
        "langextract.extraction._resolve_ingestion_options",
        wraps=extraction_mod._resolve_ingestion_options,
    ) as spy:
      lx.extract(
          text_or_documents="test",
          prompt_description=_DESCRIPTION,
          examples=_EXAMPLES,
          use_schema_constraints=False,
          options=lx.ExtractionOptions.for_model(
              ingestion=lx.IngestionOptions(
                  parser_backends=lx.ParserBackendOptions(
                      html_backend="builtin",
                  ),
              )
          ),
          parser_backends=lx.ParserBackendOptions(
              html_backend="beautifulsoup",
          ),
          prompt_validation_level=lx.extraction.pv.PromptValidationLevel.OFF,
      )
      call_kwargs = spy.call_args.kwargs
      resolved_override = call_kwargs["parser_backends"]
      self.assertEqual(resolved_override.html_backend, "beautifulsoup")


class BuildParserBackendsMergeTest(absltest.TestCase):
  """Test _build_parser_backends merge logic directly."""

  def test_convenience_kwarg_overrides_base(self):
    base = ParserBackendOptions(readable_pdf_backend="pymupdf")
    result = extraction_mod._build_parser_backends(
        base=base,
        override=None,
        readable_pdf_backend="pdfplumber",
        scanned_pdf_backend=None,
        image_backend=None,
        docx_backend=None,
        html_backend=None,
        url_backend=None,
    )
    self.assertEqual(result.readable_pdf_backend, "pdfplumber")

  def test_override_replaces_base(self):
    base = ParserBackendOptions(csv_backend="pandas")
    override = ParserBackendOptions(csv_backend="builtin_csv")
    result = extraction_mod._build_parser_backends(
        base=base,
        override=override,
        readable_pdf_backend=None,
        scanned_pdf_backend=None,
        image_backend=None,
        docx_backend=None,
        html_backend=None,
        url_backend=None,
    )
    self.assertEqual(result.csv_backend, "builtin_csv")

  def test_convenience_kwarg_overrides_override(self):
    base = ParserBackendOptions()
    override = ParserBackendOptions(html_backend="builtin")
    result = extraction_mod._build_parser_backends(
        base=base,
        override=override,
        readable_pdf_backend=None,
        scanned_pdf_backend=None,
        image_backend=None,
        docx_backend=None,
        html_backend="trafilatura",
        url_backend=None,
    )
    self.assertEqual(result.html_backend, "trafilatura")

  def test_no_overrides_preserves_base(self):
    base = ParserBackendOptions(excel_backend="openpyxl")
    result = extraction_mod._build_parser_backends(
        base=base,
        override=None,
        readable_pdf_backend=None,
        scanned_pdf_backend=None,
        image_backend=None,
        docx_backend=None,
        html_backend=None,
        url_backend=None,
    )
    self.assertEqual(result.excel_backend, "openpyxl")

  def test_multiple_convenience_kwargs_combined(self):
    base = ParserBackendOptions()
    result = extraction_mod._build_parser_backends(
        base=base,
        override=None,
        readable_pdf_backend="pypdf",
        scanned_pdf_backend="tesseract",
        image_backend="tesseract",
        docx_backend="docx2txt",
        html_backend="beautifulsoup",
        url_backend="beautifulsoup",
    )
    self.assertEqual(result.readable_pdf_backend, "pypdf")
    self.assertEqual(result.scanned_pdf_backend, "tesseract")
    self.assertEqual(result.image_backend, "tesseract")
    self.assertEqual(result.docx_backend, "docx2txt")
    self.assertEqual(result.html_backend, "beautifulsoup")
    self.assertEqual(result.url_backend, "beautifulsoup")


class CLIBackendPrecedenceOverConfigTest(absltest.TestCase):
  """CLI --backend flags become direct kwargs, which beat config kwargs."""

  def _make_examples_file(self):
    raw = [{
        "text": "Hello world.",
        "extractions": [
            {"extraction_class": "greeting", "extraction_text": "Hello"}
        ],
    }]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(raw, f)
    f.flush()
    f.close()
    return f.name

  def test_direct_flag_and_config_both_forwarded(self):
    """Direct CLI backend flag appears as top-level kwarg alongside config."""
    examples_path = self._make_examples_file()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as config_file:
      config_file.write(
          "backends:\n  readable_pdf: pymupdf\n  url: trafilatura\n"
      )
      config_file.flush()
      config_path = config_file.name

    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "text",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--config",
            config_path,
            "--url-backend",
            "beautifulsoup",
        ])
        call_kwargs = m.call_args.kwargs
        # Config has readable_pdf=pymupdf and url=trafilatura
        self.assertEqual(
            call_kwargs["config"].ingestion.parser_backends.url_backend,
            "trafilatura",
        )
        # Direct flag overrides at the top-level extract() kwarg layer
        self.assertEqual(call_kwargs["url_backend"], "beautifulsoup")
    finally:
      pathlib.Path(examples_path).unlink()
      pathlib.Path(config_path).unlink()


# ===================================================================
# 4. Backend listing command
# ===================================================================


class BackendsListingContentTest(absltest.TestCase):
  """Verify backends command output contains expected structural data."""

  def test_json_output_includes_all_field_keys(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--json"])
    reports = json.loads(buf.getvalue())
    for report in reports:
      self.assertIn("category", report)
      self.assertIn("default", report)
      self.assertIn("auto_preference", report)
      self.assertIn("backends", report)
      for backend in report["backends"]:
        for key in (
            "name",
            "implementation",
            "installed",
            "usable",
            "default",
            "auto_preference_rank",
            "reason",
        ):
          self.assertIn(key, backend)

  def test_json_readable_pdf_shows_three_backends(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--category", "readable_pdf", "--json"])
    reports = json.loads(buf.getvalue())
    self.assertLen(reports, 1)
    self.assertEqual(reports[0]["category"], "readable_pdf")
    names = [b["name"] for b in reports[0]["backends"]]
    self.assertIn("pymupdf", names)
    self.assertIn("pdfplumber", names)
    self.assertIn("pypdf", names)
    self.assertEqual(reports[0]["default"], "pymupdf")

  def test_json_registered_backend_shows_planned(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--category", "scanned_pdf", "--json"])
    reports = json.loads(buf.getvalue())
    backends_by_name = {b["name"]: b for b in reports[0]["backends"]}
    self.assertEqual(backends_by_name["ocrmypdf"]["implementation"], "planned")
    self.assertEqual(
        backends_by_name["paddleocr"]["implementation"], "implemented"
    )

  def test_text_output_contains_category_names(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends"])
    output = buf.getvalue()
    for cat in ("readable_pdf", "scanned_pdf", "image", "csv", "html", "url"):
      self.assertIn(f"Category: {cat}", output)

  def test_auto_preference_shown_for_supported_categories(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--category", "readable_pdf"])
    output = buf.getvalue()
    self.assertIn("Auto preference:", output)
    self.assertNotIn("not supported", output)

  def test_auto_preference_not_supported_for_txt(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--category", "txt"])
    output = buf.getvalue()
    self.assertIn("not supported", output)


# ===================================================================
# 5. CLI extract argument parsing
# ===================================================================


class CLIExtractModelFlagsTest(absltest.TestCase):
  """Verify model-related CLI flags are forwarded to extract()."""

  def _make_examples_file(self):
    raw = [{
        "text": "Hello.",
        "extractions": [{"extraction_class": "g", "extraction_text": "Hello"}],
    }]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(raw, f)
    f.flush()
    f.close()
    return f.name

  def test_model_id_forwarded(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "text",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--model-id",
            "gemini-2.5-flash",
        ])
        self.assertEqual(m.call_args.kwargs["model_id"], "gemini-2.5-flash")
    finally:
      pathlib.Path(examples_path).unlink()

  def test_provider_forwarded(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "text",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--provider",
            "openai",
        ])
        self.assertEqual(m.call_args.kwargs["provider"], "openai")
    finally:
      pathlib.Path(examples_path).unlink()

  def test_ocr_engine_forwarded(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "text",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--ocr-engine",
            "glm-ocr",
        ])
        self.assertEqual(m.call_args.kwargs["ocr_engine"], "glm-ocr")
    finally:
      pathlib.Path(examples_path).unlink()

  def test_omitted_optional_flags_not_in_kwargs(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "text",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
        ])
        kw = m.call_args.kwargs
        self.assertNotIn("model_id", kw)
        self.assertNotIn("provider", kw)
        self.assertNotIn("api_key", kw)
        self.assertNotIn("model_url", kw)
        self.assertNotIn("ocr_engine", kw)
        self.assertNotIn("config", kw)
    finally:
      pathlib.Path(examples_path).unlink()

  def test_file_path_input_resolved_to_path_object(self):
    examples_path = self._make_examples_file()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as input_file:
      input_file.write("Some document text.")
      input_file.flush()
      input_path = input_file.name

    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            input_path,
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
        ])
        source = m.call_args.kwargs["text_or_documents"]
        self.assertIsInstance(source, pathlib.Path)
    finally:
      pathlib.Path(examples_path).unlink()
      pathlib.Path(input_path).unlink()


# ===================================================================
# 6. Missing optional dependency errors
# ===================================================================


class MissingDependencyErrorTest(absltest.TestCase):
  """Selecting a backend whose library is missing produces actionable errors."""

  def test_error_mentions_backend_name(self):
    backend = get_backend(FileCategory.READABLE_PDF, "pymupdf")
    self.assertIsNotNone(backend)
    with mock.patch(
        "langextract.ingestion_backends.is_backend_installed",
        return_value=False,
    ):
      with self.assertRaises(ImportError) as cm:
        require_backend(FileCategory.READABLE_PDF, "pymupdf")
      self.assertIn("pymupdf", str(cm.exception))

  def test_error_mentions_install_command(self):
    backend = get_backend(FileCategory.READABLE_PDF, "pymupdf")
    self.assertIsNotNone(backend)
    with mock.patch(
        "langextract.ingestion_backends.is_backend_installed",
        return_value=False,
    ):
      with self.assertRaises(ImportError) as cm:
        require_backend(FileCategory.READABLE_PDF, "pymupdf")
      self.assertIn("langextract[pdf]", str(cm.exception))

  def test_error_mentions_category(self):
    with mock.patch(
        "langextract.ingestion_backends.is_backend_installed",
        return_value=False,
    ):
      with self.assertRaises(ImportError) as cm:
        require_backend(FileCategory.HTML, "trafilatura")
      self.assertIn("html", str(cm.exception))

  def test_registered_only_error_lists_available_alternatives(self):
    with self.assertRaises(NotImplementedError) as cm:
      require_backend(FileCategory.SCANNED_PDF, "ocrmypdf")
    msg = str(cm.exception)
    self.assertIn("paddleocr", msg)
    self.assertIn("tesseract", msg)

  def test_cli_validates_before_calling_extract(self):
    examples_path = self._make_examples_file()
    try:
      with mock.patch("langextract.extract") as m:
        with self.assertRaises(SystemExit):
          cli.main([
              "extract",
              "text",
              "--prompt",
              "test",
              "--examples",
              examples_path,
              "--readable-pdf-backend",
              "nonexistent",
          ])
        m.assert_not_called()
    finally:
      pathlib.Path(examples_path).unlink()

  def _make_examples_file(self):
    raw = [{
        "text": "Hello.",
        "extractions": [{"extraction_class": "g", "extraction_text": "Hello"}],
    }]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(raw, f)
    f.flush()
    f.close()
    return f.name


# ===================================================================
# 7. Auto backend fallback behavior
# ===================================================================


class AutoBackendFallbackTest(absltest.TestCase):
  """auto mode skips REGISTERED and uninstalled backends."""

  @mock.patch(
      "langextract.ingestion_backends.is_backend_installed",
      return_value=True,
  )
  def test_auto_skips_registered_backends(self, _):
    """ocrmypdf is REGISTERED for scanned_pdf; auto should skip it."""
    result = resolve_backend(
        FileCategory.SCANNED_PDF,
        backend="auto",
    )
    self.assertIsNotNone(result)
    self.assertNotEqual(result.name, "ocrmypdf")
    self.assertEqual(result.status, BackendStatus.AVAILABLE)

  def test_auto_skips_uninstalled_backends(self):
    """Simulate all backends uninstalled; auto should raise ImportError."""
    with mock.patch(
        "langextract.ingestion_backends.is_backend_installed",
        return_value=False,
    ):
      with self.assertRaisesRegex(ImportError, r"could not find"):
        resolve_backend(FileCategory.READABLE_PDF, backend="auto")

  def test_auto_returns_first_installed_per_preference(self):
    """If first preference is not installed, fall through to next."""

    def selective_installed(backend):
      # Pretend only pypdf is installed
      return backend.name == "pypdf"

    with mock.patch(
        "langextract.ingestion_backends.is_backend_installed",
        side_effect=selective_installed,
    ):
      result = resolve_backend(FileCategory.READABLE_PDF, backend="auto")
      self.assertEqual(result.name, "pypdf")

  def test_auto_unsupported_category_raises(self):
    with self.assertRaisesRegex(ValueError, r"not supported"):
      ParserBackendOptions(table_pdf_backend="auto")

  def test_auto_respects_custom_preference_order(self):
    """User-configured preference_order should be honored."""
    opts = ParserBackendOptions(
        readable_pdf_backend="auto",
        backend_preference_order={
            "readable_pdf": ("pypdf", "pdfplumber", "pymupdf"),
        },
    )
    # Must be constructible and have the right preference order
    order = opts.preference_order_for_category(FileCategory.READABLE_PDF)
    self.assertEqual(order[0], "pypdf")
    self.assertEqual(order[1], "pdfplumber")
    self.assertEqual(order[2], "pymupdf")

  def test_auto_preference_deduplicates(self):
    """User preferences + defaults should not have duplicates."""
    opts = ParserBackendOptions(
        readable_pdf_backend="auto",
        backend_preference_order={
            "readable_pdf": ("pymupdf",),
        },
    )
    order = opts.preference_order_for_category(FileCategory.READABLE_PDF)
    self.assertEqual(order.count("pymupdf"), 1)


class AutoBackendImportErrorContentTest(absltest.TestCase):
  """When all candidates are unavailable, ImportError explains what was tried."""

  def test_import_error_lists_skipped_backends(self):
    with mock.patch(
        "langextract.ingestion_backends.is_backend_installed",
        return_value=False,
    ):
      with self.assertRaises(ImportError) as cm:
        resolve_backend(FileCategory.HTML, backend="auto")
      msg = str(cm.exception)
      self.assertIn("html", msg)
      # Should mention at least one of the backends it tried
      self.assertTrue(
          "trafilatura" in msg or "beautifulsoup" in msg or "builtin" in msg
      )


# ===================================================================
# 8. Backward compatibility for default usage
# ===================================================================


class BackwardCompatibilityTest(absltest.TestCase):
  """Ensure default usage paths work without explicit backend selection."""

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_without_backend_kwargs_uses_defaults(
      self, mock_create_model, mock_annotator_cls
  ):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.schema = None
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = (
        mock.MagicMock()
    )

    # Should not raise — default backends resolve without explicit selection
    lx.extract(
        text_or_documents="some text",
        prompt_description=_DESCRIPTION,
        examples=_EXAMPLES,
        use_schema_constraints=False,
        prompt_validation_level=lx.extraction.pv.PromptValidationLevel.OFF,
    )
    mock_create_model.assert_called_once()

  def test_options_alias_still_accepted(self):
    options = lx.ExtractionOptions.for_model(
        model_id="gemini-2.5-flash",
        provider="gemini",
    )
    # The 'options' alias should be accepted without error
    with (
        mock.patch("langextract.annotation.Annotator") as mock_ann,
        mock.patch("langextract.extraction.factory.create_model") as mock_cm,
    ):
      mock_model = mock.MagicMock()
      mock_model.requires_fence_output = True
      mock_model.schema = None
      mock_cm.return_value = mock_model
      mock_ann.return_value.annotate_text.return_value = mock.MagicMock()

      lx.extract(
          text_or_documents="some text",
          prompt_description=_DESCRIPTION,
          examples=_EXAMPLES,
          options=options,
          use_schema_constraints=False,
          prompt_validation_level=lx.extraction.pv.PromptValidationLevel.OFF,
      )

  def test_config_alias_also_accepted_with_extraction_options(self):
    config = lx.ExtractionOptions.for_model(
        model_id="gemini-2.5-flash",
    )
    with (
        mock.patch("langextract.annotation.Annotator") as mock_ann,
        mock.patch("langextract.extraction.factory.create_model") as mock_cm,
    ):
      mock_model = mock.MagicMock()
      mock_model.requires_fence_output = True
      mock_model.schema = None
      mock_cm.return_value = mock_model
      mock_ann.return_value.annotate_text.return_value = mock.MagicMock()

      lx.extract(
          text_or_documents="some text",
          prompt_description=_DESCRIPTION,
          examples=_EXAMPLES,
          config=config,
          use_schema_constraints=False,
          prompt_validation_level=lx.extraction.pv.PromptValidationLevel.OFF,
      )

  def test_default_parser_backend_options_all_none(self):
    opts = ParserBackendOptions()
    for cat in FileCategory:
      self.assertIsNone(
          opts.backend_for_category(cat),
          f"Expected None for {cat.value} but got"
          f" {opts.backend_for_category(cat)!r}",
      )

  def test_default_ingestion_options_have_sensible_values(self):
    opts = lx.IngestionOptions()
    self.assertIsInstance(opts.fetch_urls, bool)
    self.assertIsInstance(opts.text_column, str)
    self.assertTrue(len(opts.text_column) > 0)
    self.assertIsNone(opts.id_column)
    self.assertIsInstance(opts.parser_backends, ParserBackendOptions)

  def test_default_extraction_options_have_sensible_values(self):
    opts = lx.ExtractionOptions()
    self.assertIsNone(opts.model)
    self.assertIsNone(opts.model_config)
    self.assertIsInstance(opts.ingestion, lx.IngestionOptions)
    self.assertIsInstance(opts.ocr, lx.OcrOptions)
    self.assertIsNone(opts.ocr.engine)
    self.assertIsNone(opts.ocr.config)


class ParserBackendOptionsSelectionsTest(absltest.TestCase):
  """Test the selections() convenience method."""

  def test_empty_options_returns_empty_dict(self):
    opts = ParserBackendOptions()
    self.assertEqual(opts.selections(), {})

  def test_explicit_selections_returned(self):
    opts = ParserBackendOptions(
        readable_pdf_backend="pymupdf",
        html_backend="trafilatura",
    )
    selections = opts.selections()
    self.assertEqual(selections[FileCategory.READABLE_PDF], "pymupdf")
    self.assertEqual(selections[FileCategory.HTML], "trafilatura")
    self.assertNotIn(FileCategory.CSV, selections)


class IngestionOptionsValidationTest(absltest.TestCase):
  """IngestionOptions constructor validation."""

  def test_empty_text_column_raises(self):
    with self.assertRaisesRegex(ValueError, r"non-empty string"):
      lx.IngestionOptions(text_column="")

  def test_non_parser_backends_type_raises(self):
    with self.assertRaisesRegex(TypeError, r"ParserBackendOptions"):
      lx.IngestionOptions(parser_backends="not-valid")


if __name__ == "__main__":
  absltest.main()
