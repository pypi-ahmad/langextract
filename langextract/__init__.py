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

"""LangExtract: unified extraction API with optional advanced configuration.

The top-level package exposes one primary high-level entry point,
``langextract.extract()``, plus a small set of convenience data/config types so
simple extraction flows do not require importing multiple submodules.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict

from langextract import visualization
from langextract.core.data import AnnotatedDocument
from langextract.core.data import Document
from langextract.core.data import ExampleData
from langextract.core.data import Extraction
from langextract.core.data import FormatType
from langextract.extraction import extract
from langextract.extraction import ExtractionOptions
from langextract.extraction import ExtractResult
from langextract.extraction import IngestionOptions
from langextract.extraction import load_extraction_config
from langextract.extraction import OcrOptions
from langextract.extraction import parse_extraction_config
from langextract.extraction import ParserBackendOptions
from langextract.ingestion_backends import BackendCategoryInfo
from langextract.ingestion_backends import BackendOptionInfo
from langextract.ingestion_backends import list_available_backends

__all__ = [
    # Primary high-level API
    "extract",
    "visualize",
    "AnnotatedDocument",
    "Document",
    "ExampleData",
    "Extraction",
    "ExtractionOptions",
    "ExtractResult",
    "FormatType",
    "IngestionOptions",
    "BackendCategoryInfo",
    "BackendOptionInfo",
    "OcrOptions",
    "ParserBackendOptions",
    "list_available_backends",
    "load_extraction_config",
    "parse_extraction_config",
    # Submodules exposed lazily on attribute access for ergonomics:
    "annotation",
    "data",
    "providers",
    "schema",
    "inference",
    "factory",
    "resolver",
    "prompting",
    "io",
    "ingestion",
    "ingestion_backends",
    "visualization",
    "exceptions",
    "core",
    "plugins",
]

_CACHE: Dict[str, Any] = {}


def visualize(*args: Any, **kwargs: Any):
  """Top-level API: lx.visualize(...)."""
  return visualization.visualize(*args, **kwargs)


# PEP 562 lazy loading
_LAZY_MODULES = {
    "annotation": "langextract.annotation",
    "chunking": "langextract.chunking",
    "data": "langextract.data",
    "data_lib": "langextract.data_lib",
    "debug_utils": "langextract.core.debug_utils",
    "exceptions": "langextract.exceptions",
    "factory": "langextract.factory",
    "inference": "langextract.inference",
    "ingestion": "langextract.ingestion",
    "ingestion_backends": "langextract.ingestion_backends",
    "io": "langextract.io",
    "ocr": "langextract.ocr",
    "progress": "langextract.progress",
    "prompting": "langextract.prompting",
    "providers": "langextract.providers",
    "resolver": "langextract.resolver",
    "schema": "langextract.schema",
    "tokenizer": "langextract.tokenizer",
    "visualization": "langextract.visualization",
    "core": "langextract.core",
    "plugins": "langextract.plugins",
    "registry": "langextract.registry",  # Backward compat - will emit warning
}


def __getattr__(name: str) -> Any:
  if name in _CACHE:
    return _CACHE[name]
  modpath = _LAZY_MODULES.get(name)
  if modpath is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module = importlib.import_module(modpath)
  # ensure future 'import langextract.<name>' returns the same module
  sys.modules[f"{__name__}.{name}"] = module
  setattr(sys.modules[__name__], name, module)
  _CACHE[name] = module
  return module


def __dir__():
  return sorted(__all__)
