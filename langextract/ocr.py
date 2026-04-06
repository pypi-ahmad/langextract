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

"""OCR engine abstraction for LangExtract.

Provides a pluggable ``OcrEngine`` interface with concrete adapters for
Ollama-hosted vision models and the GLM-OCR MaaS API.  OCR output feeds
into the normal ``extract()`` pipeline through ``ingestion.normalize_input()``.

This is intentionally separate from the langextract provider system.
OCR models are preprocessing tools that produce text, not extraction
providers that produce structured extractions.

Usage::

    from langextract.ocr import OllamaOcrEngine

    engine = OllamaOcrEngine(model_id="deepseek-ocr")
    result = engine.run_ocr(open("scan.png", "rb").read())
    print(result.text)
"""

from __future__ import annotations

import abc
import base64
from collections.abc import Callable, Mapping
import dataclasses
import os
import typing
from typing import Any
from urllib.parse import urljoin

import requests

from langextract.core import exceptions

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OcrError(exceptions.LangExtractError):
  """Base error for OCR operations."""


class OcrTimeoutError(OcrError):
  """Raised when an OCR request times out."""


class OcrConnectionError(OcrError):
  """Raised when the OCR server is unreachable."""


class OcrModelNotFoundError(OcrError):
  """Raised when the requested OCR model is not available."""


class OcrResponseError(OcrError):
  """Raised when the OCR response is malformed or empty."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class OcrResult:
  """Structured result from an OCR engine."""

  text: str
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class HttpOcrRequest:
  """HTTP request description used by configurable OCR adapters."""

  url: str
  headers: Mapping[str, str]
  json_body: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class OcrEngineConfig:
  """Serializable configuration used to build OCR engines."""

  engine_name: str | None = None
  transport: str | None = None
  prompt: str | None = None
  timeout: int | None = None
  base_url: str | None = None
  api_url: str | None = None
  api_key: str | None = None


# ---------------------------------------------------------------------------
# Abstract engine
# ---------------------------------------------------------------------------


class OcrEngine(abc.ABC):
  """Abstract base class for OCR engines.

  Subclasses must implement ``run_ocr`` to extract text from a single
  image.  The optional ``run_ocr_pdf_page`` hook can be overridden for
  engines with native page-aware behaviour.
  """

  @abc.abstractmethod
  def run_ocr(
      self,
      image_data: bytes,
      *,
      prompt: str | None = None,
  ) -> OcrResult:
    """Extract text from raw image bytes.

    Args:
      image_data: Raw image bytes (PNG, JPEG, etc.).
      prompt: Optional instruction sent alongside the image.

    Returns:
      An ``OcrResult`` containing extracted text and metadata.
    """

  def run_ocr_pdf_page(
      self,
      page_image: bytes,
      *,
      page_number: int,
      prompt: str | None = None,
  ) -> OcrResult:
    """OCR a single PDF page rendered as an image.

    The default implementation delegates to ``run_ocr``.  Override this
    in engines that have native page-aware behaviour.
    """
    return self.run_ocr(image_data=page_image, prompt=prompt)


# ---------------------------------------------------------------------------
# Ollama engine
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120
_DEFAULT_PROMPT = "Extract all text from this image."

SUPPORTED_OCR_MODELS: frozenset[str] = frozenset({
    "deepseek-ocr",
    "glm-ocr",
})
SUPPORTED_OCR_ENGINES: frozenset[str] = SUPPORTED_OCR_MODELS
_SUPPORTED_GLM_TRANSPORTS: frozenset[str] = frozenset({"maas", "ollama"})


class OllamaOcrEngine(OcrEngine):
  """OCR engine backed by a local Ollama vision model.

  Supports any Ollama-hosted model listed in ``SUPPORTED_OCR_MODELS``
  (currently ``deepseek-ocr`` and ``glm-ocr``).
  """

  def __init__(
      self,
      model_id: str = "deepseek-ocr",
      *,
      base_url: str | None = None,
      timeout: int = _DEFAULT_TIMEOUT,
      prompt: str | None = None,
  ):
    _validate_model_id(model_id)
    self._model_id = model_id
    self._base_url = _resolve_base_url(base_url)
    self._timeout = timeout
    self._default_prompt = prompt or _DEFAULT_PROMPT

  @property
  def model_id(self) -> str:
    return self._model_id

  def run_ocr(
      self,
      image_data: bytes,
      *,
      prompt: str | None = None,
  ) -> OcrResult:
    """Send an image to the Ollama vision model and return the result."""
    if not image_data:
      raise ValueError("image_data must be non-empty bytes.")

    effective_prompt = prompt or self._default_prompt
    api_url = urljoin(
        self._base_url
        if self._base_url.endswith("/")
        else self._base_url + "/",
        "api/generate",
    )

    encoded_image = base64.b64encode(image_data).decode("ascii")

    payload = {
        "model": self._model_id,
        "prompt": effective_prompt,
        "images": [encoded_image],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
      response = requests.post(
          api_url,
          headers=headers,
          json=payload,
          timeout=self._timeout,
      )
    except requests.exceptions.ReadTimeout as e:
      raise OcrTimeoutError(
          f"OCR request timed out after {self._timeout}s "
          f"(model={self._model_id}, url={api_url})"
      ) from e
    except requests.exceptions.ConnectionError as e:
      raise OcrConnectionError(
          f"Cannot connect to Ollama at {self._base_url}. "
          "Is the Ollama server running?"
      ) from e
    except requests.exceptions.RequestException as e:
      raise OcrError(f"OCR request failed: {e}") from e

    if response.status_code == 404:
      raise OcrModelNotFoundError(
          f"OCR model {self._model_id!r} not found in Ollama. "
          f"Try: ollama pull {self._model_id}"
      )
    if response.status_code != 200:
      raise OcrError(
          f"Ollama returned HTTP {response.status_code} for OCR request."
      )

    text = _parse_ollama_response(response)
    return OcrResult(
        text=text,
        metadata={"model": self._model_id, "engine": "ollama"},
    )


class DeepSeekOcrEngine(OcrEngine):
  """DeepSeek OCR adapter backed by Ollama-hosted ``deepseek-ocr``."""

  def __init__(
      self,
      *,
      base_url: str | None = None,
      timeout: int = _DEFAULT_TIMEOUT,
      prompt: str | None = None,
  ):
    self._engine = OllamaOcrEngine(
        model_id="deepseek-ocr",
        base_url=base_url,
        timeout=timeout,
        prompt=prompt,
    )

  def run_ocr(
      self,
      image_data: bytes,
      *,
      prompt: str | None = None,
  ) -> OcrResult:
    return self._engine.run_ocr(image_data, prompt=prompt)

  def run_ocr_pdf_page(
      self,
      page_image: bytes,
      *,
      page_number: int,
      prompt: str | None = None,
  ) -> OcrResult:
    return self._engine.run_ocr_pdf_page(
        page_image,
        page_number=page_number,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# GLM-OCR MaaS engine
# ---------------------------------------------------------------------------

_GLM_DEFAULT_API_URL = "https://api.z.ai/api/paas/v4/layout_parsing"
_GLM_DEFAULT_TIMEOUT = 120


class GlmMaasOcrEngine(OcrEngine):
  """OCR engine for the GLM-OCR hosted MaaS API.

  Authenticates via an API key provided explicitly or through the
  ``GLM_OCR_API_KEY`` environment variable.
  """

  def __init__(
      self,
      *,
      api_key: str | None = None,
      api_url: str | None = None,
      timeout: int = _GLM_DEFAULT_TIMEOUT,
      request_builder: (
          Callable[[bytes, str | None], HttpOcrRequest] | None
      ) = None,
      response_parser: Callable[[requests.Response], OcrResult] | None = None,
  ):
    self._api_key = api_key or os.getenv("GLM_OCR_API_KEY", "")
    if not self._api_key:
      raise ValueError(
          "GLM-OCR API key is required. Provide it explicitly or set the "
          "GLM_OCR_API_KEY environment variable."
      )
    self._api_url = api_url or os.getenv(
        "GLM_OCR_API_URL", _GLM_DEFAULT_API_URL
    )
    self._timeout = timeout
    self._request_builder = request_builder or self._build_request
    self._response_parser = response_parser or _parse_glm_response

  def run_ocr(
      self,
      image_data: bytes,
      *,
      prompt: str | None = None,
  ) -> OcrResult:
    """Send an image to the GLM-OCR MaaS API and return the result."""
    if not image_data:
      raise ValueError("image_data must be non-empty bytes.")

    request = self._request_builder(image_data, prompt)

    try:
      response = requests.post(
          request.url,
          headers=dict(request.headers),
          json=dict(request.json_body),
          timeout=self._timeout,
      )
    except requests.exceptions.ReadTimeout as e:
      raise OcrTimeoutError(
          f"GLM-OCR request timed out after {self._timeout}s"
      ) from e
    except requests.exceptions.ConnectionError as e:
      raise OcrConnectionError(
          f"Cannot connect to GLM-OCR API at {self._api_url}."
      ) from e
    except requests.exceptions.RequestException as e:
      raise OcrError(f"GLM-OCR request failed: {e}") from e

    if response.status_code != 200:
      raise OcrError(f"GLM-OCR API returned HTTP {response.status_code}.")

    return self._response_parser(response)

  def _build_request(
      self,
      image_data: bytes,
      prompt: str | None,
  ) -> HttpOcrRequest:
    return _build_glm_request(
        image_data,
        api_key=self._api_key,
        api_url=self._api_url,
        prompt=prompt,
    )


class GlmOcrEngine(OcrEngine):
  """GLM OCR adapter with selectable transport."""

  def __init__(
      self,
      *,
      transport: str = "maas",
      api_key: str | None = None,
      api_url: str | None = None,
      base_url: str | None = None,
      timeout: int = _GLM_DEFAULT_TIMEOUT,
      prompt: str | None = None,
      request_builder: (
          Callable[[bytes, str | None], HttpOcrRequest] | None
      ) = None,
      response_parser: Callable[[requests.Response], OcrResult] | None = None,
  ):
    if transport not in _SUPPORTED_GLM_TRANSPORTS:
      raise ValueError(
          f"Unsupported GLM OCR transport: {transport!r}. "
          f"Supported transports: {sorted(_SUPPORTED_GLM_TRANSPORTS)}"
      )

    if transport == "ollama":
      self._engine = OllamaOcrEngine(
          model_id="glm-ocr",
          base_url=base_url,
          timeout=timeout,
          prompt=prompt,
      )
    else:
      self._engine = GlmMaasOcrEngine(
          api_key=api_key,
          api_url=api_url,
          timeout=timeout,
          request_builder=request_builder,
          response_parser=response_parser,
      )

  def run_ocr(
      self,
      image_data: bytes,
      *,
      prompt: str | None = None,
  ) -> OcrResult:
    return self._engine.run_ocr(image_data, prompt=prompt)

  def run_ocr_pdf_page(
      self,
      page_image: bytes,
      *,
      page_number: int,
      prompt: str | None = None,
  ) -> OcrResult:
    return self._engine.run_ocr_pdf_page(
        page_image,
        page_number=page_number,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_base_url(base_url: str | None) -> str:
  """Return the effective Ollama base URL.

  Priority order:
    1. Explicit ``base_url`` argument.
    2. ``OLLAMA_BASE_URL`` environment variable.
    3. Default ``http://localhost:11434``.
  """
  if base_url is not None:
    return base_url
  return os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)


def _validate_model_id(model_id: str) -> None:
  """Raise ``ValueError`` if *model_id* is not a recognised OCR model."""
  if model_id not in SUPPORTED_OCR_MODELS:
    raise ValueError(
        f"Unsupported OCR model: {model_id!r}. "
        f"Supported models: {sorted(SUPPORTED_OCR_MODELS)}"
    )


def _coerce_ocr_config(
    config: OcrEngineConfig | Mapping[str, Any] | None,
) -> OcrEngineConfig | None:
  if config is None:
    return None
  if isinstance(config, OcrEngineConfig):
    return config
  return OcrEngineConfig(**dict(config))


def _is_ocr_engine_instance(candidate: Any) -> bool:
  return isinstance(candidate, OcrEngine) or callable(
      getattr(candidate, "run_ocr", None)
  )


def _build_glm_request(
    image_data: bytes,
    *,
    api_key: str,
    api_url: str,
    prompt: str | None,
) -> HttpOcrRequest:
  encoded_image = base64.b64encode(image_data).decode("ascii")
  payload: dict[str, Any] = {"image": encoded_image}
  if prompt is not None:
    payload["prompt"] = prompt
  return HttpOcrRequest(
      url=api_url,
      headers={
          "Content-Type": "application/json",
          "Authorization": f"Bearer {api_key}",
      },
      json_body=payload,
  )


def _parse_ollama_response(response: requests.Response) -> str:
  """Extract the text payload from an Ollama generate response."""
  try:
    body = response.json()
  except ValueError as e:
    raise OcrResponseError(
        "Ollama returned non-JSON response for OCR request."
    ) from e

  if not isinstance(body, dict):
    raise OcrResponseError(
        f"Expected JSON object from Ollama, got {type(body).__name__}."
    )

  text = body.get("response")
  if text is None:
    raise OcrResponseError(
        "Ollama response missing 'response' field. "
        f"Keys received: {sorted(body.keys())}"
    )
  if not isinstance(text, str):
    raise OcrResponseError(
        f"Expected string in 'response' field, got {type(text).__name__}."
    )

  return text


def _parse_glm_response(response: requests.Response) -> OcrResult:
  """Extract the text payload from a GLM-OCR MaaS response."""
  try:
    body = response.json()
  except ValueError as e:
    raise OcrResponseError("GLM-OCR returned non-JSON response.") from e

  if not isinstance(body, dict):
    raise OcrResponseError(
        f"Expected JSON object from GLM-OCR, got {type(body).__name__}."
    )

  result_obj = body.get("result")
  if isinstance(result_obj, dict):
    text = result_obj.get("md_results", "")
  else:
    text = ""

  if not isinstance(text, str):
    text = str(text) if text is not None else ""

  return OcrResult(
      text=text,
      metadata={"engine": "glm_maas", "raw_keys": sorted(body.keys())},
  )


def create_ocr_engine(
    engine_name: str,
    *,
    config: OcrEngineConfig | None = None,
    transport: str | None = None,
    prompt: str | None = None,
    timeout: int | None = None,
    base_url: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    request_builder: (
        Callable[[bytes, str | None], HttpOcrRequest] | None
    ) = None,
    response_parser: Callable[[requests.Response], OcrResult] | None = None,
) -> OcrEngine:
  """Build a concrete OCR engine from a simple engine identifier."""
  effective_config = config or OcrEngineConfig(engine_name=engine_name)
  if effective_config.engine_name not in (None, engine_name):
    effective_config = dataclasses.replace(
        effective_config,
        engine_name=engine_name,
    )

  effective_transport = transport or effective_config.transport
  effective_prompt = prompt or effective_config.prompt
  effective_timeout = timeout or effective_config.timeout
  effective_base_url = base_url or effective_config.base_url
  effective_api_url = api_url or effective_config.api_url
  effective_api_key = api_key or effective_config.api_key

  if engine_name == "deepseek-ocr":
    return DeepSeekOcrEngine(
        base_url=effective_base_url,
        timeout=effective_timeout or _DEFAULT_TIMEOUT,
        prompt=effective_prompt,
    )

  if engine_name == "glm-ocr":
    return GlmOcrEngine(
        transport=effective_transport or "maas",
        api_key=effective_api_key,
        api_url=effective_api_url,
        base_url=effective_base_url,
        timeout=effective_timeout or _GLM_DEFAULT_TIMEOUT,
        prompt=effective_prompt,
        request_builder=request_builder,
        response_parser=response_parser,
    )

  raise ValueError(
      f"Unsupported OCR engine: {engine_name!r}. "
      f"Supported engines: {sorted(SUPPORTED_OCR_ENGINES)}"
  )


def resolve_ocr_engine(
    ocr_engine: OcrEngine | OcrEngineConfig | str | None = None,
    *,
    config: OcrEngineConfig | Mapping[str, Any] | None = None,
) -> OcrEngine | None:
  """Resolve a user-facing OCR engine selection into a concrete engine."""
  if ocr_engine is None and config is None:
    return None

  if _is_ocr_engine_instance(ocr_engine):
    if config is not None:
      raise ValueError(
          "Pass either a concrete ocr_engine or ocr_config, not both."
      )
    return typing.cast(OcrEngine, ocr_engine)

  resolved_config = _coerce_ocr_config(config)

  if isinstance(ocr_engine, OcrEngineConfig):
    if resolved_config is not None:
      raise ValueError(
          "Pass either an OcrEngineConfig as ocr_engine or via ocr_config, "
          "not both."
      )
    resolved_config = ocr_engine
    ocr_engine = None

  engine_name = typing.cast(str | None, ocr_engine)
  if engine_name is None and resolved_config is not None:
    engine_name = resolved_config.engine_name

  if engine_name is None:
    raise ValueError("OCR engine selection requires an engine name.")

  return create_ocr_engine(engine_name, config=resolved_config)


# ---------------------------------------------------------------------------
# Backward-compatible convenience function
# ---------------------------------------------------------------------------


def ocr_image(
    image_data: bytes,
    *,
    model_id: str = "deepseek-ocr",
    base_url: str | None = None,
    prompt: str = _DEFAULT_PROMPT,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
  """Send an image to an Ollama vision model and return extracted text.

  This is a convenience wrapper around :class:`OllamaOcrEngine`.  For
  new code, prefer instantiating an engine directly.

  Args:
    image_data: Raw image bytes (PNG, JPEG, etc.).
    model_id: Ollama vision model to use for OCR.
    base_url: Ollama server URL.  Falls back to the ``OLLAMA_BASE_URL``
      environment variable, then ``http://localhost:11434``.
    prompt: Instruction sent alongside the image.
    timeout: HTTP request timeout in seconds.

  Returns:
    The text extracted from the image.

  Raises:
    ValueError: If *model_id* is not a supported OCR model or
      *image_data* is empty.
    OcrTimeoutError: If the request times out.
    OcrConnectionError: If the Ollama server is unreachable.
    OcrModelNotFoundError: If the model is not pulled in Ollama.
    OcrResponseError: If the response is malformed or contains no text.
  """
  engine = OllamaOcrEngine(
      model_id=model_id,
      base_url=base_url,
      timeout=timeout,
  )
  result = engine.run_ocr(image_data, prompt=prompt)
  return result.text


# Keep old name as private alias for backward compatibility with tests.
_parse_response = _parse_ollama_response
