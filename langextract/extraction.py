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

"""Main extraction API for LangExtract.

This module keeps the public API centered around one high-level entry point:
``extract()``. Inputs are routed through normalization, ingestion, optional
OCR, and provider-backed extraction without requiring callers to stitch those
layers together manually.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
import json
import pathlib
import typing
import warnings

try:
  import tomllib
except ImportError:  # pragma: no cover
  try:
    import tomli as tomllib
  except ImportError:  # pragma: no cover
    tomllib = None

import yaml

from langextract import annotation
from langextract import factory
from langextract import ingestion_backends as ingestion_backend_registry
from langextract import prompt_validation as pv
from langextract import prompting
from langextract import resolver
from langextract.core import base_model
from langextract.core import data
from langextract.core import format_handler as fh
from langextract.core import tokenizer as tokenizer_lib
from langextract.ingestion_backends import ParserBackendOptions

DEFAULT_MODEL_ID = factory.DEFAULT_MODEL_ID
DEFAULT_FETCH_URLS = True
DEFAULT_MAX_CHAR_BUFFER = 1000
DEFAULT_BATCH_LENGTH = 10
DEFAULT_MAX_WORKERS = 10
DEFAULT_EXTRACTION_PASSES = 1
DEFAULT_TEXT_COLUMN = "text"

ExtractResult: typing.TypeAlias = (
    data.AnnotatedDocument | list[data.AnnotatedDocument]
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "ExtractResult",
    "ExtractionOptions",
    "IngestionOptions",
    "OcrOptions",
    "ParserBackendOptions",
    "extract",
    "load_extraction_config",
    "parse_extraction_config",
]

_MISSING = object()


@dataclasses.dataclass(slots=True, frozen=True)
class OcrOptions:
  """Optional OCR configuration for ``extract()``.

  Attributes:
    engine: OCR engine instance, OCR engine config object, or a named engine
      such as ``"deepseek-ocr"`` or ``"glm-ocr"``.
    config: Optional engine-specific configuration forwarded to
      ``langextract.ocr.resolve_ocr_engine()``.
  """

  engine: typing.Any | None = None
  config: typing.Any = None


@dataclasses.dataclass(slots=True, frozen=True)
class IngestionOptions:
  """Optional input normalization controls for ``extract()``.

  Attributes:
    fetch_urls: Whether plain string URLs should be fetched before extraction.
    text_column: Preferred text column name when serializing CSV/XLSX/table-like
      inputs into extractable text.
    id_column: Optional document-id column name for tabular inputs.
    additional_context_column: Optional tabular column name that should be
      treated as per-row additional context.
    parser_backends: Optional parser backend selections keyed by input type.
  """

  fetch_urls: bool = DEFAULT_FETCH_URLS
  text_column: str = DEFAULT_TEXT_COLUMN
  id_column: str | None = None
  additional_context_column: str | None = None
  parser_backends: ParserBackendOptions = dataclasses.field(
      default_factory=ParserBackendOptions
  )

  def __post_init__(self) -> None:
    if not self.text_column:
      raise ValueError(
          "IngestionOptions.text_column must be a non-empty string."
      )
    if not isinstance(self.parser_backends, ParserBackendOptions):
      raise TypeError(
          "IngestionOptions.parser_backends must be a ParserBackendOptions "
          "instance."
      )


@dataclasses.dataclass(slots=True, frozen=True)
class ExtractionOptions:
  """Optional advanced configuration for the high-level ``extract()`` API.

  This keeps the common path ergonomic while allowing advanced callers to pack
  provider/model, ingestion, and OCR settings into one object.

  Example:
    ```python
    options = lx.ExtractionOptions.for_model(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        provider_kwargs={"api_key": "..."},
        ocr=lx.OcrOptions(engine="glm-ocr"),
    )
    result = lx.extract(
        "scan.pdf",
        prompt_description="Extract invoice fields",
        examples=examples,
        options=options,
    )
    ```

  Attributes:
    model: Pre-configured language model instance. If both ``model`` and
      ``model_config`` are provided, ``model`` takes precedence.
    model_config: Advanced provider/model configuration for factory-backed model
      creation.
    ingestion: Optional ingestion settings for URLs and tabular sources.
    ocr: Optional OCR settings for image and scanned-document inputs.
  """

  model: base_model.BaseLanguageModel | None = None
  model_config: factory.ModelConfig | None = None
  ingestion: IngestionOptions = dataclasses.field(
      default_factory=IngestionOptions
  )
  ocr: OcrOptions = dataclasses.field(default_factory=OcrOptions)

  @classmethod
  def for_model(
      cls,
      *,
      model_id: str | None = None,
      provider: factory.ProviderSelection | None = None,
      provider_kwargs: Mapping[str, typing.Any] | None = None,
      ingestion: IngestionOptions | None = None,
      ocr: OcrOptions | None = None,
  ) -> ExtractionOptions:
    """Build an ``ExtractionOptions`` object from high-level model settings."""
    return cls(
        model_config=factory.ModelConfig(
            model_id=model_id,
            provider=provider,
            provider_kwargs=dict(provider_kwargs or {}),
        ),
        ingestion=ingestion or IngestionOptions(),
        ocr=ocr or OcrOptions(),
    )


def _config_backend_field_aliases() -> dict[str, str]:
  aliases: dict[str, str] = {}
  for field in dataclasses.fields(ParserBackendOptions):
    if not field.name.endswith("_backend"):
      continue
    aliases[field.name] = field.name
    aliases[field.name.removesuffix("_backend")] = field.name
  return aliases


def _require_mapping(
    raw: Any,
    *,
    context: str,
) -> dict[str, Any]:
  if raw is None:
    return {}
  if not isinstance(raw, Mapping):
    raise TypeError(
        f"Expected an object for {context}, got {type(raw).__name__}."
    )
  return dict(raw)


def _set_config_value(
    section: dict[str, Any],
    key: str,
    value: Any,
    *,
    label: str,
) -> None:
  if value is _MISSING:
    return
  if key in section and section[key] != value:
    raise ValueError(f"Conflicting config values for {label}.")
  section[key] = value


def _require_optional_string(value: Any, *, label: str) -> str | None:
  if value is _MISSING or value is None:
    return typing.cast(str | None, value)
  if not isinstance(value, str):
    raise TypeError(f"Expected {label} to be a string.")
  return value


def _require_optional_bool(value: Any, *, label: str) -> bool | None:
  if value is _MISSING or value is None:
    return typing.cast(bool | None, value)
  if not isinstance(value, bool):
    raise TypeError(f"Expected {label} to be a boolean.")
  return value


def _normalize_backend_config(
    raw: Any,
) -> dict[str, Any]:
  aliases = _config_backend_field_aliases()
  values = _require_mapping(raw, context="backends")
  normalized: dict[str, Any] = {}
  unknown_fields = sorted(set(values) - set(aliases))
  if unknown_fields:
    raise TypeError("Unknown backend config keys: " + ", ".join(unknown_fields))

  for key, value in values.items():
    if value is None or value == "default":
      continue
    if not isinstance(value, str):
      raise TypeError(
          f"Expected a string backend name for {key!r}, got "
          f"{type(value).__name__}."
      )
    normalized[aliases[key]] = value
  return normalized


def _coerce_config_dataclass(
    cls: type[Any],
    raw: Any,
) -> Any:
  if raw is None:
    return cls()
  if isinstance(raw, cls):
    return raw

  values = _require_mapping(raw, context=cls.__name__)
  field_names = {field.name for field in dataclasses.fields(cls)}
  unknown_fields = sorted(set(values) - field_names)
  if unknown_fields:
    raise TypeError(
        f"Unknown {cls.__name__} fields: {', '.join(unknown_fields)}"
    )

  if cls is ExtractionOptions:
    if "model" in values and values["model"] is not None:
      raise TypeError(
          "Config files cannot deserialize a pre-built 'model' object. "
          "Use 'model'/'model_id' string settings instead."
      )
    if "model_config" in values:
      values["model_config"] = _coerce_config_dataclass(
          factory.ModelConfig,
          values["model_config"],
      )
    if "ingestion" in values:
      values["ingestion"] = _coerce_config_dataclass(
          IngestionOptions,
          values["ingestion"],
      )
    if "ocr" in values:
      values["ocr"] = _coerce_config_dataclass(
          OcrOptions,
          values["ocr"],
      )
  elif cls is IngestionOptions:
    if "parser_backends" in values:
      values["parser_backends"] = _coerce_config_dataclass(
          ParserBackendOptions,
          values["parser_backends"],
      )
  elif cls is factory.ModelConfig:
    if "model_id" in values:
      values["model_id"] = _require_optional_string(
          values["model_id"],
          label="model_id",
      )
    if "provider" in values:
      values["provider"] = _require_optional_string(
          values["provider"],
          label="provider",
      )
    if "provider_kwargs" in values:
      values["provider_kwargs"] = _require_mapping(
          values["provider_kwargs"],
          context="provider_kwargs",
      )

  return cls(**values)


def parse_extraction_config(
    raw: Mapping[str, Any] | ExtractionOptions | None,
) -> ExtractionOptions:
  """Parse a user config mapping into ``ExtractionOptions``.

  Supported top-level aliases keep config files concise while still mapping to
  the same typed runtime model used by ``extract()``:

  - ``provider`` / ``model`` / ``model_id`` -> ``model_config``
  - ``ocr_engine`` / ``ocr_config`` -> ``ocr``
  - ``backends`` -> ``ingestion.parser_backends``
  - ``fetch_urls`` / ``text_column`` / ``id_column`` /
    ``additional_context_column`` -> ``ingestion``

  The fully nested typed shape (``model_config``, ``ingestion``, ``ocr``) is
  also accepted.
  """
  if raw is None:
    return ExtractionOptions()
  if isinstance(raw, ExtractionOptions):
    return raw

  values = _require_mapping(raw, context="config")

  model_config_values = _require_mapping(
      values.pop("model_config", None),
      context="model_config",
  )
  ingestion_values = _require_mapping(
      values.pop("ingestion", None),
      context="ingestion",
  )
  ocr_values = _require_mapping(
      values.pop("ocr", None),
      context="ocr",
  )
  parser_backend_values = _require_mapping(
      ingestion_values.pop("parser_backends", None),
      context="ingestion.parser_backends",
  )

  allowed_aliases = {
      "provider",
      "provider_kwargs",
      "model",
      "model_id",
      "ocr_engine",
      "ocr_config",
      "backends",
      "fetch_urls",
      "text_column",
      "id_column",
      "additional_context_column",
  }
  unknown_top_level = sorted(set(values) - allowed_aliases)
  if unknown_top_level:
    raise TypeError(
        "Unknown extraction config keys: " + ", ".join(unknown_top_level)
    )

  _set_config_value(
      model_config_values,
      "provider",
      _require_optional_string(
          values.pop("provider", _MISSING),
          label="provider",
      ),
      label="provider",
  )
  _set_config_value(
      model_config_values,
      "provider_kwargs",
      values.pop("provider_kwargs", _MISSING),
      label="provider_kwargs",
  )

  model_value = values.pop("model", _MISSING)
  model_id_value = values.pop("model_id", _MISSING)
  if model_value is not _MISSING and model_id_value is not _MISSING:
    if model_value != model_id_value:
      raise ValueError("Conflicting config values for model/model_id.")
    model_id_value = model_value
  elif model_value is not _MISSING:
    model_id_value = model_value
  _set_config_value(
      model_config_values,
      "model_id",
      _require_optional_string(model_id_value, label="model/model_id"),
      label="model/model_id",
  )

  _set_config_value(
      ocr_values,
      "engine",
      _require_optional_string(
          values.pop("ocr_engine", _MISSING),
          label="ocr_engine",
      ),
      label="ocr_engine",
  )
  _set_config_value(
      ocr_values,
      "config",
      values.pop("ocr_config", _MISSING),
      label="ocr_config",
  )

  _set_config_value(
      ingestion_values,
      "fetch_urls",
      _require_optional_bool(
          values.pop("fetch_urls", _MISSING),
          label="fetch_urls",
      ),
      label="fetch_urls",
  )
  _set_config_value(
      ingestion_values,
      "text_column",
      _require_optional_string(
          values.pop("text_column", _MISSING),
          label="text_column",
      ),
      label="text_column",
  )
  _set_config_value(
      ingestion_values,
      "id_column",
      _require_optional_string(
          values.pop("id_column", _MISSING),
          label="id_column",
      ),
      label="id_column",
  )
  _set_config_value(
      ingestion_values,
      "additional_context_column",
      _require_optional_string(
          values.pop("additional_context_column", _MISSING),
          label="additional_context_column",
      ),
      label="additional_context_column",
  )

  backend_alias_values = _normalize_backend_config(values.pop("backends", None))
  for field_name, field_value in backend_alias_values.items():
    _set_config_value(
        parser_backend_values,
        field_name,
        field_value,
        label=f"backends.{field_name.removesuffix('_backend')}",
    )

  if parser_backend_values:
    ingestion_values["parser_backends"] = _coerce_config_dataclass(
        ParserBackendOptions,
        parser_backend_values,
    )

  config_values: dict[str, Any] = {}
  if model_config_values:
    config_values["model_config"] = _coerce_config_dataclass(
        factory.ModelConfig,
        model_config_values,
    )
  if ingestion_values:
    config_values["ingestion"] = _coerce_config_dataclass(
        IngestionOptions,
        ingestion_values,
    )
  if ocr_values:
    config_values["ocr"] = _coerce_config_dataclass(OcrOptions, ocr_values)

  return ExtractionOptions(**config_values)


def load_extraction_config(
    path: str | pathlib.Path,
) -> ExtractionOptions:
  """Load an extraction config file into ``ExtractionOptions``.

  Supported file types are ``.yaml``, ``.yml``, ``.toml``, and ``.json``.
  """
  config_path = pathlib.Path(path)
  raw_text = config_path.read_text(encoding="utf-8")
  suffix = config_path.suffix.lower()

  if suffix == ".json":
    raw = json.loads(raw_text)
  elif suffix in {".yaml", ".yml"}:
    raw = yaml.safe_load(raw_text)
  elif suffix == ".toml":
    if tomllib is None:
      raise ImportError(
          "TOML config files require tomllib (Python 3.11+) or tomli."
      )
    raw = tomllib.loads(raw_text)
  else:
    raise ValueError(
        "Unsupported config file type. Use .yaml, .yml, .toml, or .json."
    )

  if raw is None:
    raw = {}
  if not isinstance(raw, Mapping):
    raise TypeError(
        "Config file must contain an object/table at the top level."
    )
  return parse_extraction_config(raw)


def extract(
    text_or_documents: typing.Any,
    prompt_description: str | None = None,
    examples: typing.Sequence[typing.Any] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    api_key: str | None = None,
    language_model_type: typing.Type[typing.Any] | None = None,
    format_type: typing.Any = None,
    max_char_buffer: int = DEFAULT_MAX_CHAR_BUFFER,
    temperature: float | None = None,
    fence_output: bool | None = None,
    use_schema_constraints: bool = True,
    batch_length: int = DEFAULT_BATCH_LENGTH,
    max_workers: int = DEFAULT_MAX_WORKERS,
    additional_context: str | None = None,
    resolver_params: Mapping[str, typing.Any] | None = None,
    language_model_params: Mapping[str, typing.Any] | None = None,
    debug: bool = False,
    model_url: str | None = None,
    extraction_passes: int = DEFAULT_EXTRACTION_PASSES,
    context_window_chars: int | None = None,
    config: factory.ModelConfig | ExtractionOptions | None = None,
    model: base_model.BaseLanguageModel | None = None,
    *,
    provider: factory.ProviderSelection | None = None,
    fetch_urls: bool | None = None,
    parser_backends: ParserBackendOptions | None = None,
    readable_pdf_backend: str | None = None,
    scanned_pdf_backend: str | None = None,
    image_backend: str | None = None,
    docx_backend: str | None = None,
    html_backend: str | None = None,
    url_backend: str | None = None,
    prompt_validation_level: pv.PromptValidationLevel = pv.PromptValidationLevel.WARNING,
    prompt_validation_strict: bool = False,
    show_progress: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
    ocr_engine: typing.Any | None = None,
    ocr_config: typing.Any = None,
    options: ExtractionOptions | None = None,
) -> ExtractResult:
  """Extract structured information from text, files, tables, or documents.

  Supported input kinds all flow through the same high-level API:
  raw text, file paths, image inputs, PDFs, CSV/XLSX/table-like records, byte
  payloads, URLs, and ``Document`` inputs. Internally the input is normalized,
  ingested into a common representation, optionally OCR-processed, and then
  passed through provider-backed extraction.

  Return type behavior:
    - Text-like sources return ``AnnotatedDocument``.
    - ``Document`` inputs and iterables of ``Document`` return
      ``list[AnnotatedDocument]``.

  Examples:
    ```python
    result = lx.extract(
        "Ada Lovelace wrote the first algorithm.",
        prompt_description="Extract people",
        examples=examples,
    )

    options = lx.ExtractionOptions.for_model(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        provider_kwargs={"api_key": "..."},
        ocr=lx.OcrOptions(engine="deepseek-ocr"),
    )
    scanned = lx.extract(
        "invoice_scan.pdf",
        prompt_description="Extract invoice fields",
        examples=examples,
      config=options,
    )
    ```

  Args:
    text_or_documents: Source content to extract from. Supports raw text,
      URLs, local paths, bytes, mapping records, table-like inputs, image
      inputs with OCR, or ``Document`` inputs.
    prompt_description: Instructions for what to extract.
    examples: Example extractions that guide the model.
    model_id: Model ID for the built-in provider flow when ``config`` and
      ``model`` are not supplied.
    api_key: Provider API key for the built-in provider flow.
    language_model_type: Deprecated legacy model class selection hook.
    format_type: Output format for downstream parsing.
    max_char_buffer: Maximum characters per extraction chunk.
    temperature: Sampling temperature.
    fence_output: Optional explicit fence handling override.
    use_schema_constraints: Whether provider schema constraints should be used
      when supported.
    batch_length: Number of chunks processed per batch.
    max_workers: Maximum concurrency for provider inference.
    additional_context: Additional prompt context shared across chunks.
    resolver_params: Advanced parser/resolver settings.
    language_model_params: Advanced provider kwargs forwarded into factory model
      creation.
    debug: Enable LangExtract debug logging.
    model_url: Optional self-hosted model endpoint URL.
    extraction_passes: Number of sequential extraction passes.
    context_window_chars: Context overlap between adjacent chunks.
    config: Preferred configuration entry point for advanced usage. Pass an
      ``ExtractionOptions`` object to configure model, ingestion, and OCR in
      one place. For backward compatibility, ``factory.ModelConfig`` is also
      accepted for low-level model-only configuration.
    model: Pre-configured language model instance. Takes precedence over every
      other model-selection path.
    provider: Optional explicit built-in provider selection.
    fetch_urls: Override whether URL strings are fetched before extraction.
      When omitted, ``options.ingestion.fetch_urls`` or the package default is
      used.
    parser_backends: A ``ParserBackendOptions`` object that selects parsing
      backends for each input category. Replaces
      ``options.ingestion.parser_backends`` when both are provided. For a
      quick single-category override, prefer the convenience kwargs below.
    readable_pdf_backend: Shorthand for
      ``ParserBackendOptions(readable_pdf_backend=...)``.
    scanned_pdf_backend: Shorthand for
      ``ParserBackendOptions(scanned_pdf_backend=...)``.
    image_backend: Shorthand for
      ``ParserBackendOptions(image_backend=...)``.
    docx_backend: Shorthand for
      ``ParserBackendOptions(docx_backend=...)``.
    html_backend: Shorthand for
      ``ParserBackendOptions(html_backend=...)``.
    url_backend: Shorthand for
      ``ParserBackendOptions(url_backend=...)``.
    prompt_validation_level: Controls few-shot prompt validation behavior.
    prompt_validation_strict: Whether non-exact prompt matches raise in ERROR
      mode.
    show_progress: Whether to show extraction progress.
    tokenizer: Optional tokenizer override.
    ocr_engine: Optional OCR engine override.
    ocr_config: Optional OCR engine configuration override.
    options: Backward-compatible alias for passing ``ExtractionOptions``.
      Prefer ``config=ExtractionOptions(...)`` for new code. Direct ``model``
      and low-level ``config=factory.ModelConfig(...)`` still take precedence
      over values inside the high-level config object, and direct raw model
      kwargs override fields from ``config.model_config``.

  Raises:
    ValueError: If examples are missing.
    TypeError: If resolver_params contains an unknown key.
    pv.PromptAlignmentError: If prompt validation fails in ERROR mode.
  """
  if not examples:
    raise ValueError(
        "Examples are required for reliable extraction. Please provide at least"
        " one ExampleData object with sample extractions."
    )

  resolved_public_config = _resolve_extraction_options(
      config=config,
      options=options,
  )
  resolved_model_config_input = _resolve_model_config_input(config=config)
  resolved_resolver_params = _copy_mapping(resolver_params)
  resolved_language_model_params = _copy_mapping(language_model_params)
  resolved_model = (
      model
      if model is not None
      else resolved_public_config.model
      if resolved_public_config
      else None
  )
  resolved_config = _resolve_model_config(
      config=resolved_model_config_input,
      model=resolved_model,
      model_id=model_id,
      api_key=api_key,
      format_type=format_type,
      temperature=temperature,
      model_url=model_url,
      max_workers=max_workers,
      provider=provider,
      language_model_params=resolved_language_model_params,
      options=resolved_public_config,
  )
  resolved_ingestion = _resolve_ingestion_options(
      fetch_urls=fetch_urls,
      parser_backends=parser_backends,
      readable_pdf_backend=readable_pdf_backend,
      scanned_pdf_backend=scanned_pdf_backend,
      image_backend=image_backend,
      docx_backend=docx_backend,
      html_backend=html_backend,
      url_backend=url_backend,
      options=resolved_public_config,
  )
  resolved_ocr = _resolve_ocr_options(
      ocr_engine=ocr_engine,
      ocr_config=ocr_config,
      options=resolved_public_config,
  )

  if prompt_validation_level is not pv.PromptValidationLevel.OFF:
    report = pv.validate_prompt_alignment(
        examples=examples,
        aligner=resolver.WordAligner(),
        policy=pv.AlignmentPolicy(),
        tokenizer=tokenizer,
    )
    pv.handle_alignment_report(
        report,
        level=prompt_validation_level,
        strict_non_exact=prompt_validation_strict,
    )

  if debug:
    # pylint: disable=import-outside-toplevel
    from langextract.core import debug_utils

    debug_utils.configure_debug_logging()

  if format_type is None:
    format_type = data.FormatType.JSON

  if max_workers is not None and batch_length < max_workers:
    warnings.warn(
        f"batch_length ({batch_length}) < max_workers ({max_workers}). "
        f"Only {batch_length} workers will be used. "
        "Set batch_length >= max_workers for optimal parallelization.",
        UserWarning,
    )

  normalized_input = _normalize_extract_input(
      text_or_documents,
      ingestion_options=resolved_ingestion,
      ocr_options=resolved_ocr,
  )

  prompt_template = prompting.PromptTemplateStructured(
      description=prompt_description
  )
  prompt_template.examples.extend(examples)

  language_model = _create_language_model(
      model=resolved_model,
      config=resolved_config,
      model_id=model_id,
      api_key=api_key,
      language_model_type=language_model_type,
      format_type=format_type,
      temperature=temperature,
      fence_output=fence_output,
      use_schema_constraints=use_schema_constraints,
      max_workers=max_workers,
      model_url=model_url,
      provider=provider,
      language_model_params=resolved_language_model_params,
      examples=prompt_template.examples,
  )

  format_handler, remaining_params = fh.FormatHandler.from_resolver_params(
      resolver_params=resolved_resolver_params,
      base_format_type=format_type,
      base_use_fences=language_model.requires_fence_output,
      base_attribute_suffix=data.ATTRIBUTE_SUFFIX,
      base_use_wrapper=True,
      base_wrapper_key=data.EXTRACTIONS_KEY,
  )

  if language_model.schema is not None:
    language_model.schema.validate_format(format_handler)

  alignment_kwargs = {}
  for key in resolver.ALIGNMENT_PARAM_KEYS:
    val = remaining_params.pop(key, None)
    if val is not None:
      alignment_kwargs[key] = val
  alignment_kwargs.setdefault("suppress_parse_errors", True)

  effective_params = {"format_handler": format_handler, **remaining_params}

  try:
    res = resolver.Resolver(**effective_params)
  except TypeError as e:
    msg = str(e)
    if (
        "unexpected keyword argument" in msg
        or "got an unexpected keyword argument" in msg
    ):
      raise TypeError(
          f"Unknown key in resolver_params; check spelling: {e}"
      ) from e
    raise

  annotator = annotation.Annotator(
      language_model=language_model,
      prompt_template=prompt_template,
      format_handler=format_handler,
  )

  if normalized_input.is_text:
    return annotator.annotate_text(
        text=normalized_input.text or "",
        resolver=res,
        max_char_buffer=max_char_buffer,
        batch_length=batch_length,
        additional_context=additional_context,
        debug=debug,
        extraction_passes=extraction_passes,
        context_window_chars=context_window_chars,
        show_progress=show_progress,
        max_workers=max_workers,
        tokenizer=tokenizer,
        **alignment_kwargs,
    )

  result = annotator.annotate_documents(
      documents=typing.cast(
          Iterable[data.Document], normalized_input.documents
      ),
      resolver=res,
      max_char_buffer=max_char_buffer,
      batch_length=batch_length,
      debug=debug,
      extraction_passes=extraction_passes,
      context_window_chars=context_window_chars,
      show_progress=show_progress,
      max_workers=max_workers,
      tokenizer=tokenizer,
      **alignment_kwargs,
  )
  return list(result)


def _copy_mapping(
    value: Mapping[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
  if value is None:
    return None
  return dict(value)


def _resolve_extraction_options(
    *,
    config: factory.ModelConfig | ExtractionOptions | None,
    options: ExtractionOptions | None,
) -> ExtractionOptions | None:
  """Resolve the high-level extraction config object.

  ``config=ExtractionOptions(...)`` is the preferred public path. The older
  ``options=...`` alias remains supported for backward compatibility.
  """
  if isinstance(config, ExtractionOptions):
    if options is not None:
      raise ValueError(
          "Pass ExtractionOptions via either 'config' or 'options', not both."
      )
    return config
  return options


def _resolve_model_config_input(
    *,
    config: factory.ModelConfig | ExtractionOptions | None,
) -> factory.ModelConfig | None:
  """Extract the low-level model config from the public ``config`` argument."""
  if isinstance(config, ExtractionOptions):
    return None
  return config


def _resolve_ingestion_options(
    *,
    fetch_urls: bool | None,
    parser_backends: ParserBackendOptions | None = None,
    readable_pdf_backend: str | None = None,
    scanned_pdf_backend: str | None = None,
    image_backend: str | None = None,
    docx_backend: str | None = None,
    html_backend: str | None = None,
    url_backend: str | None = None,
    options: ExtractionOptions | None,
) -> IngestionOptions:
  base_options = options.ingestion if options else IngestionOptions()
  resolved_backends = _build_parser_backends(
      base=base_options.parser_backends,
      override=parser_backends,
      readable_pdf_backend=readable_pdf_backend,
      scanned_pdf_backend=scanned_pdf_backend,
      image_backend=image_backend,
      docx_backend=docx_backend,
      html_backend=html_backend,
      url_backend=url_backend,
  )
  return IngestionOptions(
      fetch_urls=base_options.fetch_urls if fetch_urls is None else fetch_urls,
      text_column=base_options.text_column,
      id_column=base_options.id_column,
      additional_context_column=base_options.additional_context_column,
      parser_backends=resolved_backends,
  )


def _build_parser_backends(
    *,
    base: ParserBackendOptions,
    override: ParserBackendOptions | None,
    readable_pdf_backend: str | None,
    scanned_pdf_backend: str | None,
    image_backend: str | None,
    docx_backend: str | None,
    html_backend: str | None,
    url_backend: str | None,
) -> ParserBackendOptions:
  """Merge parser backend selections from config objects and convenience kwargs.

  Priority (highest to lowest):
    1. Direct convenience kwargs (e.g. ``readable_pdf_backend="pdfplumber"``).
    2. Explicit ``parser_backends`` config object.
    3. Base config from ``options.ingestion.parser_backends``.
  """
  if override is not None:
    base = override

  kwarg_overrides: dict[str, str | None] = {}
  if readable_pdf_backend is not None:
    kwarg_overrides["readable_pdf_backend"] = readable_pdf_backend
  if scanned_pdf_backend is not None:
    kwarg_overrides["scanned_pdf_backend"] = scanned_pdf_backend
  if image_backend is not None:
    kwarg_overrides["image_backend"] = image_backend
  if docx_backend is not None:
    kwarg_overrides["docx_backend"] = docx_backend
  if html_backend is not None:
    kwarg_overrides["html_backend"] = html_backend
  if url_backend is not None:
    kwarg_overrides["url_backend"] = url_backend

  if not kwarg_overrides:
    return base

  fields = {
      field.name: getattr(base, field.name)
      for field in dataclasses.fields(base)
  }
  fields.update(kwarg_overrides)
  return ParserBackendOptions(**fields)


def _resolve_ocr_options(
    *,
    ocr_engine: typing.Any | None,
    ocr_config: typing.Any,
    options: ExtractionOptions | None,
) -> OcrOptions:
  base_options = options.ocr if options else OcrOptions()
  return OcrOptions(
      engine=base_options.engine if ocr_engine is None else ocr_engine,
      config=base_options.config if ocr_config is None else ocr_config,
  )


def _resolve_model_config(
    *,
    config: factory.ModelConfig | None,
    model: base_model.BaseLanguageModel | None,
    model_id: str,
    api_key: str | None,
    format_type: typing.Any,
    temperature: float | None,
    model_url: str | None,
    max_workers: int,
    provider: factory.ProviderSelection | None,
    language_model_params: dict[str, typing.Any] | None,
    options: ExtractionOptions | None,
) -> factory.ModelConfig | None:
  if config is not None or model is not None:
    return config

  if options is None or options.model_config is None:
    return None

  base_config = options.model_config
  provider_kwargs = dict(base_config.provider_kwargs)
  provider_kwargs.update(
      _direct_model_provider_kwargs(
          api_key=api_key,
          format_type=format_type,
          temperature=temperature,
          model_url=model_url,
          max_workers=max_workers,
          language_model_params=language_model_params,
      )
  )

  resolved_model_id = base_config.model_id
  if resolved_model_id is None and base_config.provider is None:
    resolved_model_id = model_id
  elif model_id != DEFAULT_MODEL_ID:
    resolved_model_id = model_id

  resolved_provider = base_config.provider if provider is None else provider
  return factory.ModelConfig(
      model_id=resolved_model_id,
      provider=resolved_provider,
      provider_kwargs=provider_kwargs,
  )


def _direct_model_provider_kwargs(
    *,
    api_key: str | None,
    format_type: typing.Any,
    temperature: float | None,
    model_url: str | None,
    max_workers: int,
    language_model_params: dict[str, typing.Any] | None,
    include_default_max_workers: bool = False,
) -> dict[str, typing.Any]:
  provider_kwargs: dict[str, typing.Any] = {}

  if api_key is not None:
    provider_kwargs["api_key"] = api_key
  if format_type is not None:
    provider_kwargs["format_type"] = format_type
  if temperature is not None:
    provider_kwargs["temperature"] = temperature
  if model_url is not None:
    provider_kwargs["model_url"] = model_url
    provider_kwargs["base_url"] = model_url
  if include_default_max_workers or max_workers != DEFAULT_MAX_WORKERS:
    provider_kwargs["max_workers"] = max_workers

  if "gemini_schema" in (language_model_params or {}):
    warnings.warn(
        "'gemini_schema' is deprecated. Schema constraints are now "
        "automatically handled. This parameter will be ignored.",
        FutureWarning,
        stacklevel=3,
    )
    language_model_params = dict(language_model_params or {})
    language_model_params.pop("gemini_schema", None)

  provider_kwargs.update(language_model_params or {})
  return provider_kwargs


def _normalize_extract_input(
    text_or_documents: typing.Any,
    *,
    ingestion_options: IngestionOptions,
    ocr_options: OcrOptions,
) -> typing.Any:
  from langextract import ingestion  # pylint: disable=import-outside-toplevel
  from langextract import ocr as ocr_lib  # pylint: disable=import-outside-toplevel

  resolved_ocr_engine = ocr_lib.resolve_ocr_engine(
      ocr_options.engine,
      config=ocr_options.config,
  )
  return ingestion.normalize_input(
      text_or_documents,
      text_column=ingestion_options.text_column,
      id_column=ingestion_options.id_column,
      additional_context_column=ingestion_options.additional_context_column,
      fetch_urls=ingestion_options.fetch_urls,
      ocr_engine=resolved_ocr_engine,
      parser_backends=ingestion_options.parser_backends,
  )


def _create_language_model(
    *,
    model: base_model.BaseLanguageModel | None,
    config: factory.ModelConfig | None,
    model_id: str,
    api_key: str | None,
    language_model_type: typing.Type[typing.Any] | None,
    format_type: typing.Any,
    temperature: float | None,
    fence_output: bool | None,
    use_schema_constraints: bool,
    max_workers: int,
    model_url: str | None,
    provider: factory.ProviderSelection | None,
    language_model_params: dict[str, typing.Any] | None,
    examples: typing.Sequence[typing.Any],
) -> base_model.BaseLanguageModel:
  if model is not None:
    if fence_output is not None:
      model.set_fence_output(fence_output)
    if use_schema_constraints:
      warnings.warn(
          "'use_schema_constraints' is ignored when 'model' is provided. "
          "The model should already be configured with schema constraints.",
          UserWarning,
          stacklevel=3,
      )
    return model

  if config is not None:
    if use_schema_constraints:
      warnings.warn(
          "With 'config', schema constraints are still applied via examples. "
          "Or pass explicit schema in config.provider_kwargs.",
          UserWarning,
          stacklevel=3,
      )
    return factory.create_model(
        config=config,
        examples=examples if use_schema_constraints else None,
        use_schema_constraints=use_schema_constraints,
        fence_output=fence_output,
    )

  if language_model_type is not None:
    warnings.warn(
        "'language_model_type' is deprecated and will be removed in v2.0.0. "
        "Use model, config, or model_id parameters instead.",
        FutureWarning,
        stacklevel=3,
    )

  provider_kwargs = _direct_model_provider_kwargs(
      api_key=api_key,
      format_type=format_type,
      temperature=temperature,
      model_url=model_url,
      max_workers=max_workers,
      language_model_params=language_model_params,
      include_default_max_workers=True,
  )
  filtered_kwargs = {k: v for k, v in provider_kwargs.items() if v is not None}

  resolved_model_id = model_id
  if provider is not None and model_id == DEFAULT_MODEL_ID:
    provider_family = factory.ModelConfig(provider=provider).provider_family
    if (
        provider_family is not None
        and provider_family != factory.ProviderFamily.GEMINI
    ):
      resolved_model_id = None

  created_config = factory.ModelConfig(
      model_id=resolved_model_id,
      provider=provider,
      provider_kwargs=filtered_kwargs,
  )
  return factory.create_model(
      config=created_config,
      examples=examples if use_schema_constraints else None,
      use_schema_constraints=use_schema_constraints,
      fence_output=fence_output,
  )
