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

"""Tests for the OCR abstraction layer (langextract.ocr)."""

from __future__ import annotations

import base64
import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import requests

from langextract import ocr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # minimal PNG-like bytes


def _ok_response(text: str = "Extracted text here.") -> mock.MagicMock:
  """Build a fake ``requests.Response`` with 200 and valid JSON."""
  resp = mock.MagicMock(spec=requests.Response)
  resp.status_code = 200
  resp.json.return_value = {"response": text}
  return resp


def _error_response(status_code: int) -> mock.MagicMock:
  resp = mock.MagicMock(spec=requests.Response)
  resp.status_code = status_code
  resp.json.return_value = {}
  return resp


def _glm_ok_response(text: str = "GLM extracted text.") -> mock.MagicMock:
  resp = mock.MagicMock(spec=requests.Response)
  resp.status_code = 200
  resp.json.return_value = {"result": {"md_results": text}}
  return resp


# ---------------------------------------------------------------------------
# OcrResult
# ---------------------------------------------------------------------------


class OcrResultTest(absltest.TestCase):

  def test_basic_construction(self):
    result = ocr.OcrResult(text="hello")
    self.assertEqual(result.text, "hello")
    self.assertEqual(result.metadata, {})

  def test_metadata_preserved(self):
    result = ocr.OcrResult(text="x", metadata={"model": "deepseek-ocr"})
    self.assertEqual(result.metadata["model"], "deepseek-ocr")

  def test_immutable(self):
    result = ocr.OcrResult(text="x")
    with self.assertRaises(AttributeError):
      result.text = "y"


# ---------------------------------------------------------------------------
# OcrEngine ABC
# ---------------------------------------------------------------------------


class OcrEngineAbcTest(absltest.TestCase):

  def test_cannot_instantiate_abstract(self):
    with self.assertRaises(TypeError):
      ocr.OcrEngine()

  def test_run_ocr_pdf_page_delegates_to_run_ocr(self):
    class _Stub(ocr.OcrEngine):

      def run_ocr(self, image_data, *, prompt=None):
        return ocr.OcrResult(text="page text")

    engine = _Stub()
    result = engine.run_ocr_pdf_page(b"img", page_number=1)
    self.assertEqual(result.text, "page text")


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class ValidateModelIdTest(absltest.TestCase):

  def test_accepts_deepseek_ocr(self):
    ocr._validate_model_id("deepseek-ocr")  # should not raise

  def test_accepts_glm_ocr(self):
    ocr._validate_model_id("glm-ocr")  # should not raise

  def test_rejects_unknown_model(self):
    with self.assertRaisesRegex(ValueError, "Unsupported OCR model"):
      ocr._validate_model_id("llama3:8b")

  def test_rejects_empty_string(self):
    with self.assertRaisesRegex(ValueError, "Unsupported OCR model"):
      ocr._validate_model_id("")


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------


class ResolveBaseUrlTest(absltest.TestCase):

  def test_explicit_url_takes_precedence(self):
    with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env:1234"}):
      self.assertEqual(
          ocr._resolve_base_url("http://explicit:5678"),
          "http://explicit:5678",
      )

  def test_falls_back_to_env_var(self):
    with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env:1234"}):
      self.assertEqual(ocr._resolve_base_url(None), "http://env:1234")

  def test_falls_back_to_default(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(ocr._resolve_base_url(None), "http://localhost:11434")


# ---------------------------------------------------------------------------
# ocr_image – payload construction
# ---------------------------------------------------------------------------


class OcrImagePayloadTest(absltest.TestCase):
  """Verify the exact request sent to Ollama."""

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_payload_has_images_field(self, mock_post):
    ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]

    self.assertIn("images", payload)
    self.assertIsInstance(payload["images"], list)
    self.assertLen(payload["images"], 1)

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_image_is_base64_encoded(self, mock_post):
    ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

    payload = mock_post.call_args[1]["json"]
    decoded = base64.b64decode(payload["images"][0])
    self.assertEqual(decoded, _FAKE_IMAGE)

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_model_id_in_payload(self, mock_post):
    ocr.ocr_image(_FAKE_IMAGE, model_id="glm-ocr")

    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["model"], "glm-ocr")

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_stream_is_false(self, mock_post):
    ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

    payload = mock_post.call_args[1]["json"]
    self.assertFalse(payload["stream"])

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_custom_prompt(self, mock_post):
    ocr.ocr_image(
        _FAKE_IMAGE, model_id="deepseek-ocr", prompt="Read this receipt."
    )

    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["prompt"], "Read this receipt.")

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_api_url_construction(self, mock_post):
    ocr.ocr_image(
        _FAKE_IMAGE,
        model_id="deepseek-ocr",
        base_url="http://myhost:9999",
    )

    url = (
        mock_post.call_args[0][0]
        if mock_post.call_args[0]
        else mock_post.call_args[1].get(
            "url", mock_post.call_args[0][0] if mock_post.call_args[0] else None
        )
    )
    # The first positional arg to requests.post is the URL
    called_url = mock_post.call_args[0][0]
    self.assertEqual(called_url, "http://myhost:9999/api/generate")

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_custom_timeout(self, mock_post):
    ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr", timeout=30)

    self.assertEqual(mock_post.call_args[1]["timeout"], 30)

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_default_timeout(self, mock_post):
    ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

    self.assertEqual(mock_post.call_args[1]["timeout"], 120)


# ---------------------------------------------------------------------------
# ocr_image – input validation
# ---------------------------------------------------------------------------


class OcrImageValidationTest(absltest.TestCase):

  def test_rejects_unsupported_model(self):
    with self.assertRaises(ValueError):
      ocr.ocr_image(_FAKE_IMAGE, model_id="llama3:8b")

  def test_rejects_empty_image_data(self):
    with self.assertRaisesRegex(ValueError, "non-empty bytes"):
      ocr.ocr_image(b"", model_id="deepseek-ocr")


# ---------------------------------------------------------------------------
# ocr_image – error handling
# ---------------------------------------------------------------------------


class OcrImageErrorHandlingTest(absltest.TestCase):

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.ReadTimeout("timed out"),
  )
  def test_timeout_raises_ocr_timeout_error(self, _):
    with self.assertRaises(ocr.OcrTimeoutError) as cm:
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr", timeout=5)
    self.assertIn("timed out", str(cm.exception))

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.ConnectionError("refused"),
  )
  def test_connection_error_raises_ocr_connection_error(self, _):
    with self.assertRaises(ocr.OcrConnectionError) as cm:
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")
    self.assertIn("Cannot connect", str(cm.exception))

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.RequestException("something broke"),
  )
  def test_generic_request_error_raises_ocr_error(self, _):
    with self.assertRaises(ocr.OcrError):
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_error_response(404),
  )
  def test_404_raises_model_not_found(self, _):
    with self.assertRaises(ocr.OcrModelNotFoundError) as cm:
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")
    self.assertIn("ollama pull", str(cm.exception))

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_error_response(500),
  )
  def test_500_raises_ocr_error(self, _):
    with self.assertRaises(ocr.OcrError) as cm:
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")
    self.assertIn("500", str(cm.exception))


# ---------------------------------------------------------------------------
# ocr_image – response parsing
# ---------------------------------------------------------------------------


class OcrImageResponseParsingTest(absltest.TestCase):

  @mock.patch("langextract.ocr.requests.post")
  def test_returns_response_text(self, mock_post):
    mock_post.return_value = _ok_response("Hello World")
    result = ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")
    self.assertEqual(result, "Hello World")

  @mock.patch("langextract.ocr.requests.post")
  def test_returns_empty_string_if_model_returns_empty(self, mock_post):
    mock_post.return_value = _ok_response("")
    result = ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")
    self.assertEqual(result, "")

  @mock.patch("langextract.ocr.requests.post")
  def test_non_json_response_raises(self, mock_post):
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.side_effect = ValueError("not json")
    mock_post.return_value = resp

    with self.assertRaises(ocr.OcrResponseError):
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

  @mock.patch("langextract.ocr.requests.post")
  def test_missing_response_field_raises(self, mock_post):
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"done": True}
    mock_post.return_value = resp

    with self.assertRaises(ocr.OcrResponseError) as cm:
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")
    self.assertIn("missing 'response' field", str(cm.exception))

  @mock.patch("langextract.ocr.requests.post")
  def test_non_string_response_field_raises(self, mock_post):
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"response": 42}
    mock_post.return_value = resp

    with self.assertRaises(ocr.OcrResponseError):
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")

  @mock.patch("langextract.ocr.requests.post")
  def test_list_body_raises(self, mock_post):
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = ["not", "a", "dict"]
    mock_post.return_value = resp

    with self.assertRaises(ocr.OcrResponseError):
      ocr.ocr_image(_FAKE_IMAGE, model_id="deepseek-ocr")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class OcrExceptionHierarchyTest(absltest.TestCase):
  """Verify OCR exceptions inherit from LangExtractError."""

  def test_ocr_error_is_langextract_error(self):
    from langextract.core import exceptions as core_exc

    self.assertTrue(issubclass(ocr.OcrError, core_exc.LangExtractError))

  def test_all_specific_errors_are_ocr_error(self):
    for cls in (
        ocr.OcrTimeoutError,
        ocr.OcrConnectionError,
        ocr.OcrModelNotFoundError,
        ocr.OcrResponseError,
    ):
      with self.subTest(cls=cls.__name__):
        self.assertTrue(issubclass(cls, ocr.OcrError))


# ---------------------------------------------------------------------------
# OllamaOcrEngine
# ---------------------------------------------------------------------------


class OllamaOcrEngineConstructionTest(absltest.TestCase):

  def test_default_model(self):
    engine = ocr.OllamaOcrEngine()
    self.assertEqual(engine.model_id, "deepseek-ocr")

  def test_custom_model(self):
    engine = ocr.OllamaOcrEngine(model_id="glm-ocr")
    self.assertEqual(engine.model_id, "glm-ocr")

  def test_rejects_unsupported_model(self):
    with self.assertRaises(ValueError):
      ocr.OllamaOcrEngine(model_id="llama3:8b")


class OllamaOcrEngineRunOcrTest(absltest.TestCase):

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_returns_ocr_result(self, _):
    engine = ocr.OllamaOcrEngine(model_id="deepseek-ocr")
    result = engine.run_ocr(_FAKE_IMAGE)

    self.assertIsInstance(result, ocr.OcrResult)
    self.assertEqual(result.text, "Extracted text here.")
    self.assertEqual(result.metadata["engine"], "ollama")
    self.assertEqual(result.metadata["model"], "deepseek-ocr")

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_custom_prompt(self, mock_post):
    engine = ocr.OllamaOcrEngine(model_id="deepseek-ocr")
    engine.run_ocr(_FAKE_IMAGE, prompt="Read receipt.")

    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["prompt"], "Read receipt.")

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_constructor_prompt_used_as_default(self, mock_post):
    engine = ocr.OllamaOcrEngine(
        model_id="deepseek-ocr", prompt="Custom default"
    )
    engine.run_ocr(_FAKE_IMAGE)

    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["prompt"], "Custom default")

  @mock.patch("langextract.ocr.requests.post", return_value=_ok_response())
  def test_per_call_prompt_overrides_constructor(self, mock_post):
    engine = ocr.OllamaOcrEngine(model_id="deepseek-ocr", prompt="Default")
    engine.run_ocr(_FAKE_IMAGE, prompt="Override")

    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["prompt"], "Override")

  def test_empty_image_raises(self):
    engine = ocr.OllamaOcrEngine()
    with self.assertRaisesRegex(ValueError, "non-empty bytes"):
      engine.run_ocr(b"")

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.ReadTimeout("timed out"),
  )
  def test_timeout_raises(self, _):
    engine = ocr.OllamaOcrEngine(timeout=5)
    with self.assertRaises(ocr.OcrTimeoutError):
      engine.run_ocr(_FAKE_IMAGE)

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.ConnectionError("refused"),
  )
  def test_connection_error_raises(self, _):
    engine = ocr.OllamaOcrEngine()
    with self.assertRaises(ocr.OcrConnectionError):
      engine.run_ocr(_FAKE_IMAGE)

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_error_response(404),
  )
  def test_404_raises_model_not_found(self, _):
    engine = ocr.OllamaOcrEngine()
    with self.assertRaises(ocr.OcrModelNotFoundError):
      engine.run_ocr(_FAKE_IMAGE)

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_error_response(500),
  )
  def test_500_raises_ocr_error(self, _):
    engine = ocr.OllamaOcrEngine()
    with self.assertRaises(ocr.OcrError):
      engine.run_ocr(_FAKE_IMAGE)


class OllamaOcrEngineRunOcrPdfPageTest(absltest.TestCase):

  @mock.patch(
      "langextract.ocr.requests.post", return_value=_ok_response("Page 3 text")
  )
  def test_delegates_to_run_ocr(self, _):
    engine = ocr.OllamaOcrEngine()
    result = engine.run_ocr_pdf_page(b"\x89PNG...", page_number=3)
    self.assertEqual(result.text, "Page 3 text")


# ---------------------------------------------------------------------------
# GlmMaasOcrEngine
# ---------------------------------------------------------------------------


class GlmMaasOcrEngineConstructionTest(absltest.TestCase):

  def test_requires_api_key(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertRaisesRegex(ValueError, "API key is required"):
        ocr.GlmMaasOcrEngine()

  def test_accepts_explicit_key(self):
    engine = ocr.GlmMaasOcrEngine(api_key="test-key")
    # Should not raise
    self.assertIsNotNone(engine)

  def test_reads_env_key(self):
    with mock.patch.dict(os.environ, {"GLM_OCR_API_KEY": "env-key"}):
      engine = ocr.GlmMaasOcrEngine()
      self.assertIsNotNone(engine)


class GlmMaasOcrEngineRunOcrTest(absltest.TestCase):

  @mock.patch("langextract.ocr.requests.post", return_value=_glm_ok_response())
  def test_returns_ocr_result(self, _):
    engine = ocr.GlmMaasOcrEngine(api_key="key")
    result = engine.run_ocr(_FAKE_IMAGE)

    self.assertIsInstance(result, ocr.OcrResult)
    self.assertEqual(result.text, "GLM extracted text.")
    self.assertEqual(result.metadata["engine"], "glm_maas")

  @mock.patch("langextract.ocr.requests.post", return_value=_glm_ok_response())
  def test_sends_bearer_token(self, mock_post):
    engine = ocr.GlmMaasOcrEngine(api_key="my-secret")
    engine.run_ocr(_FAKE_IMAGE)

    headers = mock_post.call_args[1]["headers"]
    self.assertEqual(headers["Authorization"], "Bearer my-secret")

  @mock.patch("langextract.ocr.requests.post", return_value=_glm_ok_response())
  def test_sends_base64_image(self, mock_post):
    engine = ocr.GlmMaasOcrEngine(api_key="key")
    engine.run_ocr(_FAKE_IMAGE)

    payload = mock_post.call_args[1]["json"]
    decoded = base64.b64decode(payload["image"])
    self.assertEqual(decoded, _FAKE_IMAGE)

  @mock.patch("langextract.ocr.requests.post", return_value=_glm_ok_response())
  def test_custom_api_url(self, mock_post):
    engine = ocr.GlmMaasOcrEngine(
        api_key="key", api_url="http://custom:8080/ocr"
    )
    engine.run_ocr(_FAKE_IMAGE)

    called_url = mock_post.call_args[0][0]
    self.assertEqual(called_url, "http://custom:8080/ocr")

  def test_empty_image_raises(self):
    engine = ocr.GlmMaasOcrEngine(api_key="key")
    with self.assertRaisesRegex(ValueError, "non-empty bytes"):
      engine.run_ocr(b"")

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.ReadTimeout("timed out"),
  )
  def test_timeout_raises(self, _):
    engine = ocr.GlmMaasOcrEngine(api_key="key", timeout=5)
    with self.assertRaises(ocr.OcrTimeoutError):
      engine.run_ocr(_FAKE_IMAGE)

  @mock.patch(
      "langextract.ocr.requests.post",
      side_effect=requests.exceptions.ConnectionError("refused"),
  )
  def test_connection_error_raises(self, _):
    engine = ocr.GlmMaasOcrEngine(api_key="key")
    with self.assertRaises(ocr.OcrConnectionError):
      engine.run_ocr(_FAKE_IMAGE)

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_error_response(500),
  )
  def test_500_raises_ocr_error(self, _):
    engine = ocr.GlmMaasOcrEngine(api_key="key")
    with self.assertRaises(ocr.OcrError):
      engine.run_ocr(_FAKE_IMAGE)


class GlmMaasResponseParsingTest(absltest.TestCase):

  def test_missing_result_returns_empty(self):
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}

    result = ocr._parse_glm_response(resp)
    self.assertEqual(result.text, "")

  def test_non_json_raises(self):
    resp = mock.MagicMock(spec=requests.Response)
    resp.json.side_effect = ValueError("not json")

    with self.assertRaises(ocr.OcrResponseError):
      ocr._parse_glm_response(resp)

  def test_non_dict_body_raises(self):
    resp = mock.MagicMock(spec=requests.Response)
    resp.json.return_value = ["list"]

    with self.assertRaises(ocr.OcrResponseError):
      ocr._parse_glm_response(resp)


class DeepSeekOcrEngineTest(absltest.TestCase):

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_ok_response("deepseek text"),
  )
  def test_named_adapter_delegates_to_ollama(self, mock_post):
    engine = ocr.DeepSeekOcrEngine(base_url="http://ollama:11434")

    result = engine.run_ocr(_FAKE_IMAGE)

    self.assertEqual(result.text, "deepseek text")
    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["model"], "deepseek-ocr")


class GlmOcrEngineTest(absltest.TestCase):

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_glm_ok_response("glm maas text"),
  )
  def test_named_adapter_defaults_to_maas(self, _):
    engine = ocr.GlmOcrEngine(api_key="key")

    result = engine.run_ocr(_FAKE_IMAGE)

    self.assertEqual(result.text, "glm maas text")

  @mock.patch(
      "langextract.ocr.requests.post",
      return_value=_ok_response("glm ollama text"),
  )
  def test_named_adapter_can_use_ollama_transport(self, mock_post):
    engine = ocr.GlmOcrEngine(
        transport="ollama", base_url="http://ollama:11434"
    )

    result = engine.run_ocr(_FAKE_IMAGE)

    self.assertEqual(result.text, "glm ollama text")
    payload = mock_post.call_args[1]["json"]
    self.assertEqual(payload["model"], "glm-ocr")

  def test_invalid_transport_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported GLM OCR transport"):
      ocr.GlmOcrEngine(transport="http2")


class GlmMaasCustomizationTest(absltest.TestCase):

  @mock.patch("langextract.ocr.requests.post")
  def test_custom_request_builder_and_response_parser_are_used(self, mock_post):
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = 200
    mock_post.return_value = resp

    request_builder = mock.Mock(
        return_value=ocr.HttpOcrRequest(
            url="https://custom.example/ocr",
            headers={"Authorization": "Bearer custom"},
            json_body={"blob": "abc"},
        )
    )
    response_parser = mock.Mock(
        return_value=ocr.OcrResult(text="custom parser result")
    )

    engine = ocr.GlmMaasOcrEngine(
        api_key="key",
        request_builder=request_builder,
        response_parser=response_parser,
    )

    result = engine.run_ocr(_FAKE_IMAGE, prompt="custom")

    self.assertEqual(result.text, "custom parser result")
    request_builder.assert_called_once_with(_FAKE_IMAGE, "custom")
    response_parser.assert_called_once_with(resp)
    self.assertEqual(mock_post.call_args[0][0], "https://custom.example/ocr")
    self.assertEqual(mock_post.call_args[1]["json"], {"blob": "abc"})


class OcrFactoryTest(absltest.TestCase):

  def test_create_deepseek_engine(self):
    engine = ocr.create_ocr_engine("deepseek-ocr")
    self.assertIsInstance(engine, ocr.DeepSeekOcrEngine)

  def test_create_glm_engine(self):
    engine = ocr.create_ocr_engine(
        "glm-ocr",
        config=ocr.OcrEngineConfig(api_key="key"),
    )
    self.assertIsInstance(engine, ocr.GlmOcrEngine)

  def test_create_unknown_engine_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported OCR engine"):
      ocr.create_ocr_engine("tesseract")


class ResolveOcrEngineTest(absltest.TestCase):

  def test_none_returns_none(self):
    self.assertIsNone(ocr.resolve_ocr_engine())

  def test_concrete_engine_passthrough(self):
    engine = ocr.DeepSeekOcrEngine()
    self.assertIs(ocr.resolve_ocr_engine(engine), engine)

  def test_string_and_mapping_config(self):
    engine = ocr.resolve_ocr_engine(
        "deepseek-ocr",
        config={"timeout": 9, "base_url": "http://ollama:11434"},
    )
    self.assertIsInstance(engine, ocr.DeepSeekOcrEngine)

  def test_config_only_with_engine_name(self):
    engine = ocr.resolve_ocr_engine(
        config=ocr.OcrEngineConfig(engine_name="glm-ocr", api_key="key"),
    )
    self.assertIsInstance(engine, ocr.GlmOcrEngine)

  def test_engine_and_config_conflict_raises(self):
    with self.assertRaisesRegex(ValueError, "not both"):
      ocr.resolve_ocr_engine(ocr.DeepSeekOcrEngine(), config={"timeout": 1})


if __name__ == "__main__":
  absltest.main()
