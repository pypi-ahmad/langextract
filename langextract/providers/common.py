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

"""Shared helpers for provider integrations.

This module keeps small bits of provider-neutral response normalization and
error shaping out of the individual provider implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langextract.core import exceptions


def runtime_error(
    provider: str,
    message: str,
    *,
    original: BaseException | None = None,
) -> exceptions.InferenceRuntimeError:
  """Build a provider-tagged runtime error."""
  return exceptions.InferenceRuntimeError(
      message,
      original=original,
      provider=provider,
  )


def normalize_text_output(
    value: Any,
    *,
    provider: str,
    field_name: str,
) -> str:
  """Normalize provider response content into a plain text string."""
  if isinstance(value, str):
    return value

  if isinstance(value, Sequence) and not isinstance(
      value, (str, bytes, bytearray)
  ):
    parts: list[str] = []
    for item in value:
      text_part = _extract_text_part(item)
      if text_part is not None:
        parts.append(text_part)
    if parts:
      return "".join(parts)

  if value is None:
    raise runtime_error(
        provider,
        f"{provider} returned empty {field_name}.",
    )

  raise runtime_error(
      provider,
      (
          f"{provider} returned unsupported {field_name} type "
          f"{type(value).__name__}."
      ),
  )


def mapping_text_field(
    payload: Mapping[str, Any],
    *,
    provider: str,
    field_name: str,
) -> str:
  """Return a required text field from a mapping response."""
  if field_name not in payload:
    raise runtime_error(
        provider,
        f"{provider} response missing required field {field_name!r}.",
    )
  return normalize_text_output(
      payload[field_name],
      provider=provider,
      field_name=field_name,
  )


def _extract_text_part(item: Any) -> str | None:
  if isinstance(item, str):
    return item

  if isinstance(item, Mapping):
    direct = item.get("text")
    if isinstance(direct, str):
      return direct
    nested_value = _extract_nested_text(direct)
    if nested_value is not None:
      return nested_value
    content = item.get("content")
    if isinstance(content, str):
      return content

  for attr_name in ("text", "content"):
    attr_value = getattr(item, attr_name, None)
    if isinstance(attr_value, str):
      return attr_value
    nested_value = _extract_nested_text(attr_value)
    if nested_value is not None:
      return nested_value

  return None


def _extract_nested_text(value: Any) -> str | None:
  if isinstance(value, Mapping):
    nested = value.get("value")
    if isinstance(nested, str):
      return nested
  else:
    nested = getattr(value, "value", None)
    if isinstance(nested, str):
      return nested
  return None
