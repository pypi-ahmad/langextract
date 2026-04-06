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

"""Minimal CLI entry point for LangExtract.

Usage examples::

    langextract version
    langextract backends
    langextract backends --category readable_pdf
    langextract extract --prompt "Extract people" --examples examples.json input.txt
    langextract extract --prompt "Extract medications" --examples ex.json report.pdf --model-id gemini-2.5-flash
    langextract extract --prompt "Extract items" --examples ex.json scan.pdf --scanned-pdf-backend paddleocr
    langextract extract --prompt "Extract people" --examples ex.json input.txt --config config.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any


def _get_version() -> str:
  try:
    from importlib.metadata import version

    return version("langextract")
  except Exception:
    return "unknown"


# ---------------------------------------------------------------------------
# Subcommand: version
# ---------------------------------------------------------------------------


def _cmd_version(args: argparse.Namespace) -> None:
  del args  # unused
  print(f"langextract {_get_version()}")


# ---------------------------------------------------------------------------
# Subcommand: backends
# ---------------------------------------------------------------------------


def _cmd_backends(args: argparse.Namespace) -> None:
  from langextract import ingestion_backends

  try:
    reports = ingestion_backends.list_available_backends(args.category)
  except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1) from exc

  if args.json:
    print(
        json.dumps([dataclasses.asdict(report) for report in reports], indent=2)
    )
  else:
    for index, report in enumerate(reports):
      if index:
        print()
      auto_preference = report.auto_preference
      print(f"Category: {report.category}")
      print(f"  Default: {report.default or 'none'}")
      if auto_preference:
        print(f"  Auto preference: {' -> '.join(auto_preference)}")
      else:
        print("  Auto preference: not supported")
      print("  Backends:")
      for backend in report.backends:
        roles: list[str] = []
        if backend.default:
          roles.append("default")
        if backend.auto_preference_rank is not None:
          roles.append(f"auto#{backend.auto_preference_rank}")
        installed = (
            "yes"
            if backend.installed is True
            else "no"
            if backend.installed is False
            else "unknown"
        )
        usable = "yes" if backend.usable else "no"
        print(
            f"    {backend.name:20s}  {backend.implementation:11s}  "
            f"installed={installed:7s}  usable={usable:3s}  "
            f"roles={', '.join(roles) if roles else '-'}"
        )
        if backend.reason:
          print(f"      why: {backend.reason}")


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------


def _load_examples(path: str) -> list[Any]:
  """Load examples from a JSON file.

  Expected JSON structure — a list of objects, each containing ``text`` and
  ``extractions``::

      [
        {
          "text": "Ada Lovelace wrote the first algorithm.",
          "extractions": [
            {
              "extraction_class": "person",
              "extraction_text": "Ada Lovelace",
              "description": "Historical figure",
              "attributes": {"role": "mathematician"}
            }
          ]
        }
      ]

  Each extraction object requires ``extraction_class`` and
  ``extraction_text``.  Optional fields ``description``, ``attributes``,
  ``group_index`` and ``extraction_index`` are forwarded when present.
  """
  from langextract.core.data import ExampleData
  from langextract.core.data import Extraction

  raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  if not isinstance(raw, list):
    print("Examples JSON must be a list of objects.", file=sys.stderr)
    raise SystemExit(1)

  examples: list[ExampleData] = []
  for item in raw:
    extractions: list[Extraction] = []
    for e in item.get("extractions", []):
      kwargs: dict[str, Any] = {}
      if "description" in e:
        kwargs["description"] = e["description"]
      if "attributes" in e:
        kwargs["attributes"] = e["attributes"]
      if "group_index" in e:
        kwargs["group_index"] = e["group_index"]
      if "extraction_index" in e:
        kwargs["extraction_index"] = e["extraction_index"]
      extractions.append(
          Extraction(
              extraction_class=e["extraction_class"],
              extraction_text=e["extraction_text"],
              **kwargs,
          )
      )
    examples.append(ExampleData(text=item["text"], extractions=extractions))
  return examples


# CLI backend flags map directly to extract() convenience kwargs.
_BACKEND_CLI_FLAGS: tuple[str, ...] = (
    "readable_pdf_backend",
    "scanned_pdf_backend",
    "image_backend",
    "docx_backend",
    "html_backend",
    "url_backend",
)


def _validate_backend_flag(
    field_name: str,
    value: str,
) -> None:
  """Validate a backend name using ParserBackendOptions validation."""
  from langextract.ingestion_backends import ParserBackendOptions

  try:
    ParserBackendOptions(**{field_name: value})
  except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1) from exc


def _cmd_extract(args: argparse.Namespace) -> None:
  from langextract import data_lib
  import langextract as lx

  examples = _load_examples(args.examples)

  # Resolve the input source.
  input_path = pathlib.Path(args.input)
  if input_path.exists():
    source: Any = input_path
  else:
    # Treat as raw text or URL string.
    source = args.input

  # Build kwargs for lx.extract() from CLI flags.
  kwargs: dict[str, Any] = {
      "text_or_documents": source,
      "prompt_description": args.prompt,
      "examples": examples,
      "show_progress": not args.quiet,
  }

  if args.model_id:
    kwargs["model_id"] = args.model_id
  if args.api_key:
    kwargs["api_key"] = args.api_key
  if args.provider:
    kwargs["provider"] = args.provider
  if args.model_url:
    kwargs["model_url"] = args.model_url
  if args.ocr_engine:
    kwargs["ocr_engine"] = args.ocr_engine

  # Load optional config file as base.
  config = None
  if args.config:
    try:
      kwargs["config"] = lx.load_extraction_config(args.config)
    except (ImportError, TypeError, ValueError, json.JSONDecodeError) as exc:
      print(f"Invalid config file: {exc}", file=sys.stderr)
      raise SystemExit(1) from exc

  # Collect and validate category-specific backend flags.
  direct_backend_kwargs: dict[str, str] = {}
  for flag_dest in _BACKEND_CLI_FLAGS:
    value = getattr(args, flag_dest, None)
    if value is None:
      continue
    _validate_backend_flag(flag_dest, value)
    direct_backend_kwargs[flag_dest] = value

  kwargs.update(direct_backend_kwargs)

  result = lx.extract(**kwargs)

  # Serialize output.
  if isinstance(result, list):
    output = [data_lib.annotated_document_to_dict(doc) for doc in result]
  else:
    output = data_lib.annotated_document_to_dict(result)

  out_stream = sys.stdout
  if args.output:
    out_stream = open(args.output, "w", encoding="utf-8")  # noqa: SIM115
  try:
    json.dump(output, out_stream, indent=2, default=str, ensure_ascii=False)
    out_stream.write("\n")
  finally:
    if out_stream is not sys.stdout:
      out_stream.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="langextract",
      description=(
          "LangExtract — structured extraction from text, files, and documents."
      ),
  )
  sub = parser.add_subparsers(dest="command")

  # -- version ---------------------------------------------------------------
  sub.add_parser("version", help="Show the installed version.")

  # -- backends --------------------------------------------------------------
  bp = sub.add_parser(
      "backends",
      help="List backend availability and auto/default status.",
  )
  bp.add_argument(
      "--category",
      default=None,
      help="Filter to a single file category (e.g. readable_pdf, image).",
  )
  bp.add_argument(
      "--json",
      action="store_true",
      help="Output as JSON instead of a human-readable table.",
  )

  # -- extract ---------------------------------------------------------------
  ep = sub.add_parser("extract", help="Run extraction on an input source.")
  ep.add_argument(
      "input",
      help="Input text, file path, or URL.",
  )
  ep.add_argument(
      "--prompt",
      "-p",
      required=True,
      help="Prompt description for what to extract.",
  )
  ep.add_argument(
      "--examples",
      "-e",
      required=True,
      help="Path to a JSON file with example extractions.",
  )
  ep.add_argument(
      "--config",
      default=None,
      help="Path to a YAML, TOML, or JSON extraction config file.",
  )
  ep.add_argument(
      "--output",
      "-o",
      default=None,
      help="Write JSON output to this file instead of stdout.",
  )
  ep.add_argument(
      "--model-id",
      default=None,
      help="Model identifier (e.g. gemini-2.5-flash).",
  )
  ep.add_argument("--api-key", default=None, help="Provider API key.")
  ep.add_argument(
      "--provider",
      default=None,
      help="Provider name (e.g. gemini, openai, ollama).",
  )
  ep.add_argument(
      "--model-url", default=None, help="Self-hosted model endpoint URL."
  )
  ep.add_argument(
      "--ocr-engine",
      default=None,
      help="OCR engine override (e.g. deepseek-ocr, glm-ocr).",
  )

  # Category-specific backend selection flags.
  backend_group = ep.add_argument_group(
      "backend selection",
      'Choose parsing backends per file category. Use "auto" for '
      "automatic selection where supported. Run 'langextract backends' "
      "to see available options.",
  )
  backend_group.add_argument(
      "--readable-pdf-backend",
      default=None,
      help="Backend for readable PDFs (e.g. pymupdf, pdfplumber, pypdf, auto).",
  )
  backend_group.add_argument(
      "--scanned-pdf-backend",
      default=None,
      help="Backend for scanned PDFs (e.g. paddleocr, tesseract, auto).",
  )
  backend_group.add_argument(
      "--image-backend",
      default=None,
      help="Backend for image OCR (e.g. paddleocr, tesseract, auto).",
  )
  backend_group.add_argument(
      "--docx-backend",
      default=None,
      help="Backend for DOCX files (e.g. python-docx, docx2txt, auto).",
  )
  backend_group.add_argument(
      "--html-backend",
      default=None,
      help=(
          "Backend for HTML parsing (e.g. trafilatura, beautifulsoup, builtin,"
          " auto)."
      ),
  )
  backend_group.add_argument(
      "--url-backend",
      default=None,
      help="Backend for URL fetching (e.g. trafilatura, beautifulsoup, auto).",
  )
  ep.add_argument(
      "--quiet",
      "-q",
      action="store_true",
      help="Suppress progress output.",
  )

  return parser


def main(argv: list[str] | None = None) -> None:
  parser = _build_parser()
  args = parser.parse_args(argv)

  if args.command is None:
    parser.print_help()
    raise SystemExit(0)

  dispatch = {
      "version": _cmd_version,
      "backends": _cmd_backends,
      "extract": _cmd_extract,
  }
  dispatch[args.command](args)


if __name__ == "__main__":
  main()
