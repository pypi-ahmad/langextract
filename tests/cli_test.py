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

"""Tests for the langextract CLI entry point."""

from __future__ import annotations

from io import StringIO
import json
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest

from langextract import cli


class VersionTest(absltest.TestCase):

  def test_prints_version(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["version"])
    self.assertRegex(buf.getvalue().strip(), r"^langextract \S+")


class BackendsTest(absltest.TestCase):

  def test_lists_all_backends(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends"])
    output = buf.getvalue()
    self.assertIn("Category: readable_pdf", output)
    self.assertIn("Default:", output)
    self.assertIn("Auto preference:", output)

  def test_filter_by_category(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--category", "csv"])
    output = buf.getvalue()
    self.assertIn("Category: csv", output)
    self.assertNotIn("Category: readable_pdf", output)

  def test_json_output(self):
    buf = StringIO()
    with mock.patch("sys.stdout", buf):
      cli.main(["backends", "--json"])
    data = json.loads(buf.getvalue())
    self.assertIsInstance(data, list)
    self.assertTrue(len(data) > 0)
    self.assertIn("category", data[0])
    self.assertIn("default", data[0])
    self.assertIn("auto_preference", data[0])
    self.assertIn("backends", data[0])
    self.assertTrue(len(data[0]["backends"]) > 0)
    self.assertIn("name", data[0]["backends"][0])
    self.assertIn("implementation", data[0]["backends"][0])
    self.assertIn("installed", data[0]["backends"][0])
    self.assertIn("usable", data[0]["backends"][0])
    self.assertIn("default", data[0]["backends"][0])
    self.assertIn("auto_preference_rank", data[0]["backends"][0])
    self.assertIn("description", data[0]["backends"][0])
    self.assertIn("extras", data[0]["backends"][0])
    self.assertIn("install_commands", data[0]["backends"][0])
    self.assertIn("reason", data[0]["backends"][0])

  def test_invalid_category_exits(self):
    with self.assertRaises(SystemExit) as cm:
      cli.main(["backends", "--category", "nonexistent"])
    self.assertEqual(cm.exception.code, 1)


class ExtractArgParsingTest(absltest.TestCase):

  def test_missing_prompt_exits(self):
    with self.assertRaises(SystemExit):
      cli.main(["extract", "input.txt"])

  def test_missing_examples_exits(self):
    with self.assertRaises(SystemExit):
      cli.main(["extract", "input.txt", "--prompt", "Extract people"])

  def test_all_backend_flags_parsed(self):
    parser = cli._build_parser()
    args = parser.parse_args([
        "extract",
        "input.txt",
        "--prompt",
        "test",
        "--examples",
        "ex.json",
        "--readable-pdf-backend",
        "pymupdf",
        "--scanned-pdf-backend",
        "paddleocr",
        "--image-backend",
        "paddleocr",
        "--docx-backend",
        "python-docx",
        "--html-backend",
        "trafilatura",
        "--url-backend",
        "beautifulsoup",
    ])
    self.assertEqual(args.readable_pdf_backend, "pymupdf")
    self.assertEqual(args.scanned_pdf_backend, "paddleocr")
    self.assertEqual(args.image_backend, "paddleocr")
    self.assertEqual(args.docx_backend, "python-docx")
    self.assertEqual(args.html_backend, "trafilatura")
    self.assertEqual(args.url_backend, "beautifulsoup")

  def test_config_flag_parsed(self):
    parser = cli._build_parser()
    args = parser.parse_args([
        "extract",
        "input.txt",
        "--prompt",
        "test",
        "--examples",
        "ex.yaml",
        "--config",
        "config.yaml",
    ])
    self.assertEqual(args.config, "config.yaml")

  def test_output_flag_parsed(self):
    parser = cli._build_parser()
    args = parser.parse_args([
        "extract",
        "input.txt",
        "--prompt",
        "test",
        "--examples",
        "ex.json",
        "--output",
        "out.json",
    ])
    self.assertEqual(args.output, "out.json")


class ValidateBackendFlagTest(absltest.TestCase):

  def test_valid_backend_name(self):
    cli._validate_backend_flag("readable_pdf_backend", "pymupdf")

  def test_auto_where_supported(self):
    cli._validate_backend_flag("readable_pdf_backend", "auto")

  def test_auto_where_unsupported(self):
    with self.assertRaises(SystemExit) as cm:
      cli._validate_backend_flag("table_pdf_backend", "auto")
    self.assertEqual(cm.exception.code, 1)

  def test_unknown_backend_name(self):
    with self.assertRaises(SystemExit) as cm:
      cli._validate_backend_flag("readable_pdf_backend", "nonexistent")
    self.assertEqual(cm.exception.code, 1)


class LoadExamplesTest(absltest.TestCase):

  def test_loads_valid_examples(self):
    data = [{
        "text": "Ada Lovelace wrote the first algorithm.",
        "extractions": [{
            "extraction_class": "person",
            "extraction_text": "Ada Lovelace",
        }],
    }]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
      json.dump(data, f)
      f.flush()
      path = f.name

    try:
      examples = cli._load_examples(path)
      self.assertLen(examples, 1)
      self.assertEqual(
          examples[0].text, "Ada Lovelace wrote the first algorithm."
      )
      self.assertLen(examples[0].extractions, 1)
      self.assertEqual(examples[0].extractions[0].extraction_class, "person")
    finally:
      pathlib.Path(path).unlink()

  def test_forwards_optional_extraction_fields(self):
    data = [{
        "text": "Dr. Smith prescribed ibuprofen.",
        "extractions": [{
            "extraction_class": "medication",
            "extraction_text": "ibuprofen",
            "description": "NSAID pain reliever",
            "attributes": {"dosage": "200mg"},
            "group_index": 0,
            "extraction_index": 1,
        }],
    }]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
      json.dump(data, f)
      f.flush()
      path = f.name

    try:
      examples = cli._load_examples(path)
      ext = examples[0].extractions[0]
      self.assertEqual(ext.extraction_class, "medication")
      self.assertEqual(ext.extraction_text, "ibuprofen")
      self.assertEqual(ext.description, "NSAID pain reliever")
      self.assertEqual(ext.attributes, {"dosage": "200mg"})
      self.assertEqual(ext.group_index, 0)
      self.assertEqual(ext.extraction_index, 1)
    finally:
      pathlib.Path(path).unlink()

  def test_rejects_non_list_json(self):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
      json.dump({"text": "not a list"}, f)
      f.flush()
      path = f.name

    try:
      with self.assertRaises(SystemExit):
        cli._load_examples(path)
    finally:
      pathlib.Path(path).unlink()


class ExtractCommandWiringTest(absltest.TestCase):
  """Verify _cmd_extract builds correct kwargs for lx.extract()."""

  def _make_examples_file(self):
    data = [{
        "text": "Hello world.",
        "extractions": [
            {"extraction_class": "greeting", "extraction_text": "Hello"}
        ],
    }]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, f)
    f.flush()
    f.close()
    return f.name

  def test_basic_text_input_calls_extract(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={"extractions": []},
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "some raw text",
            "--prompt",
            "Extract greetings",
            "--examples",
            examples_path,
            "--quiet",
        ])
        m.assert_called_once()
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs["text_or_documents"], "some raw text")
    finally:
      pathlib.Path(examples_path).unlink()

  def test_direct_backend_kwarg_forwarded(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict", return_value={}
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
            "--readable-pdf-backend",
            "pymupdf",
        ])
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs.get("readable_pdf_backend"), "pymupdf")
        self.assertNotIn("parser_backends", call_kwargs)
    finally:
      pathlib.Path(examples_path).unlink()

  def test_scanned_pdf_backend_forwarded_directly(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict", return_value={}
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
            "--scanned-pdf-backend",
            "paddleocr",
        ])
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs.get("scanned_pdf_backend"), "paddleocr")
        self.assertNotIn("parser_backends", call_kwargs)
    finally:
      pathlib.Path(examples_path).unlink()

  def test_mixed_backend_flags_forward_directly(self):
    examples_path = self._make_examples_file()
    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict", return_value={}
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
            "--image-backend",
            "paddleocr",
            "--url-backend",
            "beautifulsoup",
        ])
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs.get("image_backend"), "paddleocr")
        self.assertEqual(call_kwargs.get("url_backend"), "beautifulsoup")
        self.assertNotIn("parser_backends", call_kwargs)
    finally:
      pathlib.Path(examples_path).unlink()

  def test_config_file_loads_simple_yaml_into_core_options(self):
    examples_path = self._make_examples_file()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as config_file:
      config_file.write(
          "provider: gemini\n"
          "model: gemini-2.5-flash\n"
          "ocr_engine: deepseek-ocr\n"
          "backends:\n"
          "  url: beautifulsoup\n"
          "  readable_pdf: auto\n"
      )
      config_file.flush()
      config_path = config_file.name

    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict", return_value={}
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "https://example.com",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--config",
            config_path,
        ])
        call_kwargs = m.call_args.kwargs
        config = call_kwargs.get("config")
        self.assertEqual(config.model_config.model_id, "gemini-2.5-flash")
        self.assertEqual(config.model_config.provider, "gemini")
        self.assertEqual(
            config.ingestion.parser_backends.url_backend,
            "beautifulsoup",
        )
        self.assertEqual(
            config.ingestion.parser_backends.readable_pdf_backend,
            "auto",
        )
        self.assertEqual(config.ocr.engine, "deepseek-ocr")
    finally:
      pathlib.Path(examples_path).unlink()
      pathlib.Path(config_path).unlink()

  def test_cli_backend_overrides_take_precedence_over_config(self):
    examples_path = self._make_examples_file()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as config_file:
      config_file.write(
          "backends:\n  url: trafilatura\n  scanned_pdf: tesseract\n"
      )
      config_file.flush()
      config_path = config_file.name

    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result) as m,
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict", return_value={}
          ),
          mock.patch("sys.stdout", new_callable=StringIO),
      ):
        cli.main([
            "extract",
            "https://example.com",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--config",
            config_path,
            "--url-backend",
            "beautifulsoup",
            "--html-backend",
            "builtin",
            "--scanned-pdf-backend",
            "auto",
        ])
        call_kwargs = m.call_args.kwargs
        self.assertEqual(
            call_kwargs["config"].ingestion.parser_backends.url_backend,
            "trafilatura",
        )
        self.assertEqual(call_kwargs["url_backend"], "beautifulsoup")
        self.assertEqual(call_kwargs["html_backend"], "builtin")
        self.assertEqual(call_kwargs["scanned_pdf_backend"], "auto")
    finally:
      pathlib.Path(examples_path).unlink()
      pathlib.Path(config_path).unlink()

  def test_output_to_file(self):
    examples_path = self._make_examples_file()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
      out_path = out_f.name

    try:
      fake_result = mock.MagicMock()
      with (
          mock.patch("langextract.extract", return_value=fake_result),
          mock.patch(
              "langextract.data_lib.annotated_document_to_dict",
              return_value={"extractions": []},
          ),
      ):
        cli.main([
            "extract",
            "text",
            "--prompt",
            "test",
            "--examples",
            examples_path,
            "--quiet",
            "--output",
            out_path,
        ])
      written = pathlib.Path(out_path).read_text(encoding="utf-8")
      parsed = json.loads(written)
      self.assertIn("extractions", parsed)
    finally:
      pathlib.Path(examples_path).unlink()
      pathlib.Path(out_path).unlink()

  def test_invalid_backend_exits_before_extract(self):
    examples_path = self._make_examples_file()
    try:
      with mock.patch("langextract.extract") as m:
        with self.assertRaises(SystemExit) as cm:
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
        self.assertEqual(cm.exception.code, 1)
        m.assert_not_called()
    finally:
      pathlib.Path(examples_path).unlink()

  def test_invalid_config_exits_before_extract(self):
    examples_path = self._make_examples_file()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as config_file:
      config_file.write("backends:\n  url: not-a-backend\n")
      config_file.flush()
      config_path = config_file.name

    try:
      with mock.patch("langextract.extract") as m:
        with self.assertRaises(SystemExit) as cm:
          cli.main([
              "extract",
              "text",
              "--prompt",
              "test",
              "--examples",
              examples_path,
              "--config",
              config_path,
          ])
        self.assertEqual(cm.exception.code, 1)
        m.assert_not_called()
    finally:
      pathlib.Path(examples_path).unlink()
      pathlib.Path(config_path).unlink()


class HelpTest(absltest.TestCase):

  def test_no_command_shows_help(self):
    with self.assertRaises(SystemExit) as cm:
      cli.main([])
    self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
  absltest.main()
