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

"""Tests for the high-level public extraction API."""

from __future__ import annotations

import inspect
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest

import langextract as lx
from langextract.core import data
from langextract.core import types
import langextract.extraction as extraction_api
import langextract.ingestion_backends as ingestion_backends_api

_MINIMAL_EXAMPLES = [
    lx.ExampleData(
        text="Example text",
        extractions=[
            lx.Extraction(
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


class PublicApiTest(absltest.TestCase):

  def test_top_level_exports_unified_api_types(self):
    self.assertEqual(
        str(inspect.signature(lx.extract)),
        str(inspect.signature(extraction_api.extract)),
    )
    self.assertIs(lx.ExtractionOptions, extraction_api.ExtractionOptions)
    self.assertIs(lx.IngestionOptions, extraction_api.IngestionOptions)
    self.assertIs(lx.OcrOptions, extraction_api.OcrOptions)
    self.assertIs(
        lx.ParserBackendOptions,
        extraction_api.ParserBackendOptions,
    )
    self.assertIs(
        lx.load_extraction_config,
        extraction_api.load_extraction_config,
    )
    self.assertIs(
        lx.parse_extraction_config,
        extraction_api.parse_extraction_config,
    )
    self.assertIs(
        lx.list_available_backends,
        ingestion_backends_api.list_available_backends,
    )
    self.assertIs(
        lx.BackendCategoryInfo,
        ingestion_backends_api.BackendCategoryInfo,
    )
    self.assertIs(
        lx.BackendOptionInfo,
        ingestion_backends_api.BackendOptionInfo,
    )
    self.assertIs(lx.ExampleData, data.ExampleData)
    self.assertIs(lx.Extraction, data.Extraction)
    self.assertIs(lx.Document, data.Document)
    self.assertIs(lx.AnnotatedDocument, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_for_model_without_explicit_model_preserves_default_text_flow(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="normalized text",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    result = lx.extract(
        text_or_documents="text",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="direct-key",
        options=lx.ExtractionOptions.for_model(),
        use_schema_constraints=False,
    )

    called_config = mock_create_model.call_args.kwargs["config"]
    self.assertEqual(called_config.model_id, lx.factory.DEFAULT_MODEL_ID)
    self.assertEqual(called_config.provider_kwargs["api_key"], "direct-key")
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_extraction_options_bundle_model_ocr_and_ingestion(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="normalized text",
        extractions=[],
    )
    resolved_engine = object()
    mock_resolve_ocr_engine.return_value = resolved_engine
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    options = lx.ExtractionOptions.for_model(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        provider_kwargs={"api_key": "test-key"},
        ingestion=lx.IngestionOptions(
            fetch_urls=False,
            text_column="body",
            id_column="row_id",
            additional_context_column="context",
        ),
        ocr=lx.OcrOptions(engine="deepseek-ocr", config={"timeout": 9}),
    )

    lx.extract(
        text_or_documents="https://example.com/data.csv",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        options=options,
        use_schema_constraints=False,
    )

    mock_resolve_ocr_engine.assert_called_once_with(
        "deepseek-ocr",
        config={"timeout": 9},
    )
    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertEqual(normalize_kwargs["text_column"], "body")
    self.assertEqual(normalize_kwargs["id_column"], "row_id")
    self.assertEqual(
        normalize_kwargs["additional_context_column"],
        "context",
    )
    self.assertFalse(normalize_kwargs["fetch_urls"])
    self.assertIs(normalize_kwargs["ocr_engine"], resolved_engine)
    self.assertEqual(
        normalize_kwargs["parser_backends"],
        lx.ParserBackendOptions(),
    )

    called_config = mock_create_model.call_args.kwargs["config"]
    self.assertEqual(called_config.model_id, "gemini-3-flash-preview")
    self.assertEqual(called_config.provider, "gemini")
    self.assertEqual(called_config.provider_kwargs["api_key"], "test-key")

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_explicit_ocr_and_fetch_urls_override_options(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="normalized text",
        extractions=[],
    )
    resolved_engine = object()
    mock_resolve_ocr_engine.return_value = resolved_engine
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="https://example.com/data.csv",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="direct-key",
        fetch_urls=True,
        ocr_engine="glm-ocr",
        ocr_config={"timeout": 7},
        options=lx.ExtractionOptions.for_model(
            model_id="gemini-3-flash-preview",
            provider="gemini",
            provider_kwargs={"api_key": "option-key"},
            ingestion=lx.IngestionOptions(fetch_urls=False),
            ocr=lx.OcrOptions(
                engine="deepseek-ocr",
                config={"timeout": 9},
            ),
        ),
        use_schema_constraints=False,
    )

    mock_resolve_ocr_engine.assert_called_once_with(
        "glm-ocr",
        config={"timeout": 7},
    )
    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertTrue(normalize_kwargs["fetch_urls"])

    called_config = mock_create_model.call_args.kwargs["config"]
    self.assertEqual(called_config.provider_kwargs["api_key"], "direct-key")

  @mock.patch("langextract.annotation.Annotator.annotate_documents")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_single_document_input_still_returns_list(
      self, mock_create_model, mock_annotate_documents
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_documents.return_value = [
        data.AnnotatedDocument(text="doc", extractions=[])
    ]

    result = lx.extract(
        text_or_documents=lx.Document(text="single doc"),
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
    )

    self.assertIsInstance(result, list)


class ExtractionOptionsEdgeCases(absltest.TestCase):
  """Edge cases for ExtractionOptions and related config types."""

  def test_for_model_with_only_ingestion_options(self):
    """for_model() used purely for ingestion config still produces valid options."""
    options = lx.ExtractionOptions.for_model(
        ingestion=lx.IngestionOptions(
            fetch_urls=False,
            text_column="body",
            parser_backends=lx.ParserBackendOptions(
                csv_backend="pandas",
            ),
        ),
    )

    self.assertIsNotNone(options.model_config)
    self.assertIsNone(options.model_config.model_id)
    self.assertIsNone(options.model_config.provider)
    self.assertEqual(options.ingestion.text_column, "body")
    self.assertFalse(options.ingestion.fetch_urls)
    self.assertEqual(options.ingestion.parser_backends.csv_backend, "pandas")

  def test_invalid_parser_backend_option_raises(self):
    with self.assertRaises(ValueError):
      lx.ParserBackendOptions(url_backend="not-a-real-backend")

  def test_for_model_with_only_ocr_options(self):
    """for_model() used purely for OCR config still produces valid options."""
    options = lx.ExtractionOptions.for_model(
        ocr=lx.OcrOptions(engine="deepseek-ocr", config={"timeout": 5}),
    )

    self.assertIsNone(options.model_config.model_id)
    self.assertEqual(options.ocr.engine, "deepseek-ocr")
    self.assertEqual(options.ocr.config, {"timeout": 5})

  def test_extraction_options_default_factory_creates_defaults(self):
    """Default ExtractionOptions should have sensible defaults."""
    options = lx.ExtractionOptions()
    self.assertIsNone(options.model)
    self.assertIsNone(options.model_config)
    self.assertTrue(options.ingestion.fetch_urls)
    self.assertEqual(options.ingestion.text_column, "text")
    self.assertEqual(
        options.ingestion.parser_backends, lx.ParserBackendOptions()
    )
    self.assertIsNone(options.ocr.engine)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_for_model_ingestion_only_flows_through_extract(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    """options= with only ingestion config should not break extract()."""
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    result = lx.extract(
        text_or_documents="text",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        options=lx.ExtractionOptions.for_model(
            ingestion=lx.IngestionOptions(fetch_urls=False),
        ),
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertFalse(normalize_kwargs["fetch_urls"])
    self.assertEqual(
        normalize_kwargs["parser_backends"],
        lx.ParserBackendOptions(),
    )
    called_config = mock_create_model.call_args.kwargs["config"]
    self.assertEqual(called_config.model_id, lx.factory.DEFAULT_MODEL_ID)
    self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_parser_backend_config_is_forwarded_to_runtime(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="normalized text",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    parser_backends = lx.ParserBackendOptions(
        readable_pdf_backend="auto",
        url_backend="auto",
        backend_preference_order={
            "readable_pdf": ("pypdf", "pdfplumber"),
            "url": ("beautifulsoup",),
        },
    )

    lx.extract(
        text_or_documents="https://example.com",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        options=lx.ExtractionOptions.for_model(
            ingestion=lx.IngestionOptions(
                parser_backends=parser_backends,
            )
        ),
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertEqual(normalize_kwargs["parser_backends"], parser_backends)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_high_level_config_argument_is_accepted_via_config(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="normalized text",
        extractions=[],
    )
    resolved_engine = object()
    mock_resolve_ocr_engine.return_value = resolved_engine
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="https://example.com/data.csv",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        config=lx.ExtractionOptions.for_model(
            model_id="gemini-3-flash-preview",
            provider="gemini",
            provider_kwargs={"api_key": "test-key"},
            ingestion=lx.IngestionOptions(
                fetch_urls=False,
                parser_backends=lx.ParserBackendOptions(
                    html_backend="trafilatura",
                ),
            ),
            ocr=lx.OcrOptions(engine="deepseek-ocr", config={"timeout": 9}),
        ),
        use_schema_constraints=False,
    )

    mock_resolve_ocr_engine.assert_called_once_with(
        "deepseek-ocr",
        config={"timeout": 9},
    )
    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertFalse(normalize_kwargs["fetch_urls"])
    self.assertEqual(
        normalize_kwargs["parser_backends"].html_backend,
        "trafilatura",
    )
    self.assertIs(normalize_kwargs["ocr_engine"], resolved_engine)

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_config_model_config_still_preserves_existing_behavior(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    low_level_config = lx.factory.ModelConfig(
        model_id="gemini-3-flash-preview",
        provider_kwargs={"api_key": "config-key"},
    )
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="normalized text",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "normalized text",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="text",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        config=low_level_config,
        use_schema_constraints=False,
    )

    self.assertIs(
        mock_create_model.call_args.kwargs["config"],
        low_level_config,
    )

  def test_config_extraction_options_conflicts_with_options_alias(self):
    with self.assertRaises(ValueError) as cm:
      lx.extract(
          text_or_documents="text",
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          api_key="k",
          config=lx.ExtractionOptions(),
          options=lx.ExtractionOptions(),
          use_schema_constraints=False,
      )

    self.assertIn("either 'config' or 'options'", str(cm.exception))


class ExtractionConfigFileSupportTest(absltest.TestCase):

  def test_parse_simple_alias_config(self):
    options = lx.parse_extraction_config({
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "ocr_engine": "deepseek-ocr",
        "fetch_urls": False,
        "backends": {
            "readable_pdf": "auto",
            "url": "beautifulsoup",
            "html": "default",
        },
    })

    self.assertEqual(options.model_config.provider, "gemini")
    self.assertEqual(options.model_config.model_id, "gemini-2.5-flash")
    self.assertEqual(options.ocr.engine, "deepseek-ocr")
    self.assertFalse(options.ingestion.fetch_urls)
    self.assertEqual(
        options.ingestion.parser_backends.readable_pdf_backend,
        "auto",
    )
    self.assertEqual(
        options.ingestion.parser_backends.url_backend,
        "beautifulsoup",
    )
    self.assertIsNone(options.ingestion.parser_backends.html_backend)

  def test_loads_yaml_config_file(self):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as config_file:
      config_file.write(
          "provider: gemini\n"
          "model: gemini-2.5-flash\n"
          "ocr_engine: deepseek-ocr\n"
          "backends:\n"
          "  scanned_pdf: auto\n"
          "  url: beautifulsoup\n"
      )
      config_file.flush()
      config_path = config_file.name

    try:
      options = lx.load_extraction_config(config_path)
      self.assertEqual(options.model_config.provider, "gemini")
      self.assertEqual(options.model_config.model_id, "gemini-2.5-flash")
      self.assertEqual(options.ocr.engine, "deepseek-ocr")
      self.assertEqual(
          options.ingestion.parser_backends.scanned_pdf_backend,
          "auto",
      )
      self.assertEqual(
          options.ingestion.parser_backends.url_backend,
          "beautifulsoup",
      )
    finally:
      pathlib.Path(config_path).unlink()

  def test_loads_toml_config_file(self):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    ) as config_file:
      config_file.write(
          'provider = "gemini"\n'
          'model = "gemini-2.5-flash"\n'
          'ocr_engine = "deepseek-ocr"\n\n'
          "[backends]\n"
          'readable_pdf = "auto"\n'
          'url = "trafilatura"\n'
      )
      config_file.flush()
      config_path = config_file.name

    try:
      options = lx.load_extraction_config(config_path)
      self.assertEqual(options.model_config.provider, "gemini")
      self.assertEqual(options.model_config.model_id, "gemini-2.5-flash")
      self.assertEqual(
          options.ingestion.parser_backends.readable_pdf_backend,
          "auto",
      )
      self.assertEqual(
          options.ingestion.parser_backends.url_backend,
          "trafilatura",
      )
    finally:
      pathlib.Path(config_path).unlink()

  def test_invalid_simple_config_type_raises(self):
    with self.assertRaises(TypeError):
      lx.parse_extraction_config({"fetch_urls": "yes"})

  def test_invalid_backend_alias_raises(self):
    with self.assertRaises(ValueError):
      lx.parse_extraction_config({
          "backends": {
              "readable_pdf": "not-a-backend",
          },
      })


class ProviderSelectionThroughExtractTest(absltest.TestCase):
  """Test provider selection scenarios through the full extract() entry point."""

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_provider_model_id_mismatch_raises_at_factory(
      self, mock_create_model, mock_annotator_cls
  ):
    """Mismatched provider and model_id should fail at the factory level."""
    from langextract.core import exceptions

    mock_create_model.side_effect = exceptions.InferenceConfigError(
        "not valid for provider"
    )

    with self.assertRaises(exceptions.InferenceConfigError) as cm:
      lx.extract(
          text_or_documents="text",
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          model_id="gpt-4o",
          provider="gemini",
          use_schema_constraints=False,
      )

    self.assertIn("not valid for provider", str(cm.exception))

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_provider_only_without_model_id_defers_to_provider_default(
      self, mock_create_model, mock_annotator_cls
  ):
    """provider= without model_id should not force the Gemini default."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = "ok"
    mock_config = mock.MagicMock()

    with mock.patch(
        "langextract.extraction.factory.ModelConfig", return_value=mock_config
    ) as mock_model_config:
      lx.extract(
          text_or_documents="text",
          prompt_description="desc",
          examples=_MINIMAL_EXAMPLES,
          provider="openai",
          use_schema_constraints=False,
      )

    _, kwargs = mock_model_config.call_args
    self.assertEqual(kwargs["provider"], "openai")
    self.assertIsNone(kwargs["model_id"])


class BackendKwargsOnExtractTest(absltest.TestCase):
  """Tests for convenience backend kwargs on extract()."""

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_direct_readable_pdf_backend_kwarg(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="report.pdf",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        readable_pdf_backend="pdfplumber",
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertEqual(
        normalize_kwargs["parser_backends"].readable_pdf_backend,
        "pdfplumber",
    )

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_direct_html_backend_kwarg(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="page.html",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        html_backend="trafilatura",
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertEqual(
        normalize_kwargs["parser_backends"].html_backend,
        "trafilatura",
    )

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_parser_backends_kwarg_overrides_options(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    base_backends = lx.ParserBackendOptions(csv_backend="pandas")
    override_backends = lx.ParserBackendOptions(csv_backend="builtin_csv")

    lx.extract(
        text_or_documents="data.csv",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        parser_backends=override_backends,
        options=lx.ExtractionOptions.for_model(
            ingestion=lx.IngestionOptions(parser_backends=base_backends),
        ),
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertEqual(
        normalize_kwargs["parser_backends"].csv_backend,
        "builtin_csv",
    )

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_convenience_kwarg_overrides_parser_backends_config(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="doc.docx",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        parser_backends=lx.ParserBackendOptions(
            docx_backend="python-docx",
            csv_backend="pandas",
        ),
        docx_backend="docx2txt",
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    backends = normalize_kwargs["parser_backends"]
    # convenience kwarg wins over parser_backends config
    self.assertEqual(backends.docx_backend, "docx2txt")
    # other fields from parser_backends config are preserved
    self.assertEqual(backends.csv_backend, "pandas")

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_convenience_kwarg_overrides_high_level_config_argument(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="doc.docx",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        config=lx.ExtractionOptions.for_model(
            ingestion=lx.IngestionOptions(
                parser_backends=lx.ParserBackendOptions(
                    readable_pdf_backend="pymupdf",
                    csv_backend="pandas",
                ),
            ),
        ),
        readable_pdf_backend="pdfplumber",
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    backends = normalize_kwargs["parser_backends"]
    self.assertEqual(backends.readable_pdf_backend, "pdfplumber")
    self.assertEqual(backends.csv_backend, "pandas")

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_multiple_convenience_kwargs_combined(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="text",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        readable_pdf_backend="pdfplumber",
        image_backend="paddleocr",
        docx_backend="docx2txt",
        html_backend="trafilatura",
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    backends = normalize_kwargs["parser_backends"]
    self.assertEqual(backends.readable_pdf_backend, "pdfplumber")
    self.assertEqual(backends.image_backend, "paddleocr")
    self.assertEqual(backends.docx_backend, "docx2txt")
    self.assertEqual(backends.html_backend, "trafilatura")

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.factory.create_model")
  @mock.patch("langextract.ingestion.normalize_input")
  @mock.patch("langextract.ocr.resolve_ocr_engine")
  def test_no_backend_kwargs_preserves_existing_behavior(
      self,
      mock_resolve_ocr_engine,
      mock_normalize_input,
      mock_create_model,
      mock_annotate_text,
  ):
    mock_create_model.return_value = _mock_model()
    mock_annotate_text.return_value = data.AnnotatedDocument(
        text="t",
        extractions=[],
    )
    mock_resolve_ocr_engine.return_value = None
    mock_normalize_input.return_value = lx.ingestion.NormalizedInput.from_text(
        "t",
        source_type=lx.ingestion.InputSourceType.RAW_TEXT,
    )

    lx.extract(
        text_or_documents="text",
        prompt_description="desc",
        examples=_MINIMAL_EXAMPLES,
        api_key="k",
        use_schema_constraints=False,
    )

    _, normalize_kwargs = mock_normalize_input.call_args
    self.assertEqual(
        normalize_kwargs["parser_backends"],
        lx.ParserBackendOptions(),
    )


if __name__ == "__main__":
  absltest.main()
