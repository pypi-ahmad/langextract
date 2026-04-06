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

"""Provider backend definitions and selection helpers.

This module keeps provider-family metadata in one place so higher-level code
can reason about backend selection without depending on Gemini-only details.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import os
import re
from typing import Any, cast, Protocol, runtime_checkable, TypeAlias

from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import schema as schema_lib
from langextract.providers import patterns

__all__ = [
    "DEFAULT_GEMINI_MODEL_ID",
    "ProviderFamily",
    "ProviderSelection",
    "ProviderBackend",
    "BuiltinProviderBackend",
    "get_default_model_id",
    "get_provider_backend",
    "list_builtin_provider_backends",
    "match_provider_backend",
    "normalize_provider_selection",
    "resolve_provider_family",
]


class ProviderFamily(str, enum.Enum):
  """Supported built-in provider families."""

  GEMINI = "gemini"
  OLLAMA = "ollama"
  OPENAI = "openai"

  def __str__(self) -> str:
    return self.value


ProviderSelection: TypeAlias = ProviderFamily | str

DEFAULT_GEMINI_MODEL_ID = "gemini-3-flash-preview"


@runtime_checkable
class ProviderBackend(Protocol):
  """Structural interface for backend metadata entries."""

  family: ProviderFamily
  target: str
  patterns: tuple[str, ...]
  priority: int
  default_model_id: str | None
  optional_dependency: bool
  schema_target: str | None
  aliases: tuple[str, ...]
  api_key_env_vars: tuple[str, ...]
  base_url_env_var: str | None
  default_base_url: str | None
  model_id_validation_patterns: tuple[str, ...]
  requires_explicit_model_id: bool

  def load_model_class(self) -> type[base_model.BaseLanguageModel]:
    """Load the provider's language-model class."""

  def load_schema_class(self) -> type[schema_lib.BaseSchema] | None:
    """Load the provider's schema class if it has one."""

  def resolve_model_id(self, model_id: str | None) -> str | None:
    """Return the effective model id for this backend."""

  def validate_model_id(self, model_id: str | None) -> None:
    """Validate an explicit model id for this backend."""

  def apply_environment_defaults(
      self,
      kwargs: dict[str, Any],
  ) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Apply backend-specific env defaults to provider kwargs."""


def _load_symbol(target: str) -> object:
  module_path, class_name = target.rsplit(":", 1)
  module = importlib.import_module(module_path)
  return getattr(module, class_name)


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltinProviderBackend:
  """Metadata for a built-in provider family."""

  family: ProviderFamily
  target: str
  patterns: tuple[str, ...]
  priority: int
  default_model_id: str | None = None
  optional_dependency: bool = False
  schema_target: str | None = None
  aliases: tuple[str, ...] = ()
  api_key_env_vars: tuple[str, ...] = ()
  base_url_env_var: str | None = None
  default_base_url: str | None = None
  model_id_validation_patterns: tuple[str, ...] = ()
  requires_explicit_model_id: bool = False

  def load_model_class(self) -> type[base_model.BaseLanguageModel]:
    return cast(type[base_model.BaseLanguageModel], _load_symbol(self.target))

  def load_schema_class(self) -> type[schema_lib.BaseSchema] | None:
    if self.schema_target is None:
      return None
    return cast(type[schema_lib.BaseSchema], _load_symbol(self.schema_target))

  def resolve_model_id(self, model_id: str | None) -> str | None:
    resolved = model_id if model_id is not None else self.default_model_id
    self.validate_model_id(resolved)
    return resolved

  def validate_model_id(self, model_id: str | None) -> None:
    if model_id is None or not self.model_id_validation_patterns:
      return

    if any(
        re.search(pattern, model_id)
        for pattern in self.model_id_validation_patterns
    ):
      return

    expected_patterns = ", ".join(self.model_id_validation_patterns)
    raise exceptions.InferenceConfigError(
        f"Model id {model_id!r} is not valid for provider"
        f" {self.family.value!r}. Expected it to match one of:"
        f" {expected_patterns}"
    )

  def apply_environment_defaults(
      self,
      kwargs: dict[str, Any],
  ) -> tuple[dict[str, Any], tuple[str, ...]]:
    resolved = dict(kwargs)
    found_env_vars: list[str] = []

    if "api_key" not in resolved and not resolved.get("vertexai", False):
      for env_var in self.api_key_env_vars:
        key_val = os.getenv(env_var)
        if key_val:
          resolved["api_key"] = key_val
          found_env_vars.append(env_var)
          break
      if "api_key" in resolved:
        for env_var in self.api_key_env_vars:
          if env_var in found_env_vars:
            continue
          if os.getenv(env_var):
            found_env_vars.append(env_var)

    if (
        self.base_url_env_var is not None
        and "base_url" not in resolved
        and self.default_base_url is not None
    ):
      resolved["base_url"] = os.getenv(
          self.base_url_env_var,
          self.default_base_url,
      )

    return resolved, tuple(found_env_vars)


_BACKENDS: dict[ProviderFamily, BuiltinProviderBackend] = {
    ProviderFamily.GEMINI: BuiltinProviderBackend(
        family=ProviderFamily.GEMINI,
        target="langextract.providers.gemini:GeminiLanguageModel",
        patterns=patterns.GEMINI_PATTERNS,
        priority=patterns.GEMINI_PRIORITY,
        default_model_id=DEFAULT_GEMINI_MODEL_ID,
        schema_target="langextract.providers.schemas.gemini:GeminiSchema",
        aliases=("geminilanguagemodel",),
        api_key_env_vars=("GEMINI_API_KEY", "LANGEXTRACT_API_KEY"),
        model_id_validation_patterns=patterns.GEMINI_PATTERNS,
    ),
    ProviderFamily.OLLAMA: BuiltinProviderBackend(
        family=ProviderFamily.OLLAMA,
        target="langextract.providers.ollama:OllamaLanguageModel",
        patterns=patterns.OLLAMA_PATTERNS,
        priority=patterns.OLLAMA_PRIORITY,
        schema_target="langextract.core.schema:FormatModeSchema",
        aliases=("ollamalanguagemodel",),
        base_url_env_var="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434",
        requires_explicit_model_id=True,
    ),
    ProviderFamily.OPENAI: BuiltinProviderBackend(
        family=ProviderFamily.OPENAI,
        target="langextract.providers.openai:OpenAILanguageModel",
        patterns=patterns.OPENAI_PATTERNS,
        priority=patterns.OPENAI_PRIORITY,
        default_model_id="gpt-4o-mini",
        optional_dependency=True,
        aliases=("openailanguagemodel",),
        api_key_env_vars=("OPENAI_API_KEY", "LANGEXTRACT_API_KEY"),
    ),
}


def _build_alias_map() -> dict[str, ProviderFamily]:
  aliases: dict[str, ProviderFamily] = {}
  for family, backend in _BACKENDS.items():
    aliases[family.value] = family
    aliases[backend.target.rsplit(":", 1)[1].lower()] = family
    for alias in backend.aliases:
      aliases[alias.lower()] = family
  return aliases


_ALIAS_TO_FAMILY = _build_alias_map()


def list_builtin_provider_backends(
    include_optional: bool = True,
) -> tuple[BuiltinProviderBackend, ...]:
  """Return built-in provider metadata entries in registration order."""
  return tuple(
      backend
      for backend in _BACKENDS.values()
      if include_optional or not backend.optional_dependency
  )


def normalize_provider_selection(
    provider: ProviderSelection | None,
) -> ProviderFamily | None:
  """Normalize an explicit provider selection to a built-in family."""
  if provider is None:
    return None
  if isinstance(provider, ProviderFamily):
    return provider
  return _ALIAS_TO_FAMILY.get(provider.lower())


def get_provider_backend(
    provider: ProviderSelection | None,
) -> BuiltinProviderBackend | None:
  """Return backend metadata for an explicit built-in provider selection."""
  family = normalize_provider_selection(provider)
  if family is None:
    return None
  return _BACKENDS[family]


def get_default_model_id(
    provider: ProviderSelection | None,
) -> str | None:
  """Return the default model id for an explicit built-in provider."""
  backend = get_provider_backend(provider)
  if backend is None:
    return None
  return backend.default_model_id


def match_provider_backend(
    model_id: str | None = None,
    provider: ProviderSelection | None = None,
) -> BuiltinProviderBackend | None:
  """Resolve the most likely built-in backend for a provider/model pair."""
  explicit_backend = get_provider_backend(provider)
  if explicit_backend is not None:
    return explicit_backend

  if not model_id:
    return None

  sorted_backends = sorted(
      list_builtin_provider_backends(),
      key=lambda backend: backend.priority,
      reverse=True,
  )
  for backend in sorted_backends:
    if any(re.search(pattern, model_id) for pattern in backend.patterns):
      return backend
  return None


def resolve_provider_family(
    model_id: str | None = None,
    provider: ProviderSelection | None = None,
) -> ProviderFamily | None:
  """Return the built-in family for a provider/model pair if known."""
  backend = match_provider_backend(model_id=model_id, provider=provider)
  if backend is None:
    return None
  return backend.family
