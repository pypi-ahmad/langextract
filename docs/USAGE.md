# LangExtract — Usage Guide & Features

LangExtract is a Python library for structured text extraction using large language models.
It takes unstructured text, documents, or URLs and returns grounded, character-aligned
extractions with custom attributes — powered by Gemini, OpenAI, or local Ollama models.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Supported Input Formats](#supported-input-formats)
- [Providers](#providers)
- [Extraction Options](#extraction-options)
- [Working with Results](#working-with-results)
- [Visualization](#visualization)
- [Saving & Loading Results](#saving--loading-results)
- [Long Documents & Multi-Pass Extraction](#long-documents--multi-pass-extraction)
- [OCR for Images & Scanned PDFs](#ocr-for-images--scanned-pdfs)
- [Parser Backends](#parser-backends)
- [Schema Constraints](#schema-constraints)
- [Configuration Files](#configuration-files)
- [Custom Provider Plugins](#custom-provider-plugins)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Core (Gemini provider included by default)
pip install langextract

# With OpenAI support
pip install langextract[openai]

# With PDF parsing
pip install langextract[pdf]

# With OCR for scanned documents
pip install langextract[ocr]

# With Office document support (.docx, .pptx, .xlsx)
pip install langextract[office]

# With HTML/URL parsing
pip install langextract[html]

# Everything
pip install langextract[full]
```

### Optional dependency groups

| Extra       | Packages included                                          |
|-------------|------------------------------------------------------------|
| `openai`    | `openai`                                                   |
| `pdf`       | `pymupdf`, `pdfplumber`, `pypdf`                           |
| `ocr`       | `paddleocr`, `pytesseract`, `Pillow`                       |
| `office`    | `openpyxl`, `python-docx`, `docx2txt`, `python-pptx`      |
| `html`      | `trafilatura`, `beautifulsoup4`                            |
| `url`       | Same as `html`                                             |
| `xlsx`      | `openpyxl`                                                 |
| `full`      | All of the above                                           |

---

## Quick Start

```python
import langextract as lx

# 1. Define what to extract via a prompt and few-shot examples
examples = [
    lx.ExampleData(
        text="Marie Curie discovered radium in 1898.",
        extractions=[
            lx.Extraction(
                extraction_class="scientist",
                extraction_text="Marie Curie",
                attributes={"discovery": "radium", "year": "1898"},
            )
        ],
    )
]

# 2. Run extraction
result = lx.extract(
    text_or_documents="Nikola Tesla invented the AC motor in 1887.",
    prompt_description="Extract scientists, their discoveries, and the year.",
    examples=examples,
    model_id="gemini-2.5-flash",
)

# 3. Inspect results
for ext in result.extractions:
    print(f"{ext.extraction_class}: {ext.extraction_text}")
    print(f"  attributes: {ext.attributes}")
    print(f"  position:   [{ext.char_interval.start_pos}:{ext.char_interval.end_pos}]")
```

---

## Core Concepts

### Few-shot examples

LangExtract is example-driven. You provide one or more `ExampleData` objects that
show the model what to extract. Each example contains input text and the expected
extractions, including class labels, extracted text spans, and optional attributes.

```python
example = lx.ExampleData(
    text="The patient was prescribed Metformin 500mg daily.",
    extractions=[
        lx.Extraction(
            extraction_class="medication",
            extraction_text="Metformin",
            attributes={"dose": "500mg", "frequency": "daily"},
        )
    ],
)
```

### Prompt description

The `prompt_description` tells the model what to look for:

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description="Extract all medications with dose and frequency.",
    examples=[example],
)
```

### Character-grounded extractions

Every extraction is aligned back to the source text with character offsets.
The `char_interval` field gives the exact `[start_pos, end_pos)` span, and the
`alignment_status` indicates how the LLM output was matched:

| Status           | Meaning                                 |
|------------------|-----------------------------------------|
| `MATCH_EXACT`    | Exact substring match in source text    |
| `MATCH_FUZZY`    | Approximate match (minor differences)   |
| `MATCH_GREATER`  | LLM returned more text than the span    |
| `MATCH_LESSER`   | LLM returned less text than the span    |

---

## Supported Input Formats

Pass any of these directly to `lx.extract(text_or_documents=...)`:

| Input                        | Python type                              | Extra needed  |
|------------------------------|------------------------------------------|---------------|
| Plain text                   | `str`                                    | —             |
| URL                          | `str` (starting with `http://`/`https://`) | `[html]`    |
| Text file                    | `pathlib.Path` (`.txt`, `.md`)           | —             |
| CSV file                     | `pathlib.Path` (`.csv`)                  | —             |
| PDF                          | `pathlib.Path` (`.pdf`)                  | `[pdf]`       |
| Word document                | `pathlib.Path` (`.docx`)                 | `[office]`    |
| PowerPoint                   | `pathlib.Path` (`.pptx`)                 | `[office]`    |
| Excel spreadsheet            | `pathlib.Path` (`.xlsx`)                 | `[xlsx]`      |
| HTML file                    | `pathlib.Path` (`.html`, `.htm`)         | `[html]`      |
| Image file                   | `pathlib.Path` (`.png`, `.jpg`, etc.)    | `[ocr]`       |
| Raw bytes                    | `bytes` (PDF or image auto-detected)     | `[pdf]`/`[ocr]` |
| pandas DataFrame             | `pandas.DataFrame`                       | —             |
| List of dicts                | `list[dict]`                             | —             |
| Pre-built Document           | `lx.Document`                            | —             |
| List of Documents            | `list[lx.Document]`                      | —             |

### Examples

```python
import pathlib
import pandas as pd

# Plain text
result = lx.extract(text_or_documents="Some text to extract from.", ...)

# URL (fetched automatically)
result = lx.extract(text_or_documents="https://example.com/article.html", ...)

# PDF file
result = lx.extract(text_or_documents=pathlib.Path("report.pdf"), ...)

# pandas DataFrame
df = pd.DataFrame({"text": ["Note 1", "Note 2"], "id": ["a", "b"]})
result = lx.extract(text_or_documents=df, text_column="text", id_column="id", ...)

# Pre-built document with additional context
doc = lx.Document(text="Patient note", additional_context="Age: 45, Male")
result = lx.extract(text_or_documents=doc, ...)
```

---

## Providers

LangExtract automatically selects a provider based on `model_id`. You can also
force a specific provider with the `provider=` parameter.

### Gemini (default)

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",        # or gemini-2.5-pro, etc.
)
```

Supports Vertex AI for enterprise use:

```python
result = lx.extract(
    ...,
    model_id="gemini-2.5-pro",
    language_model_params={
        "vertexai": True,
        "project": "my-gcp-project",
        "location": "us-central1",
    },
)
```

**Matched model IDs**: any ID starting with `gemini`.

### OpenAI

Requires `pip install langextract[openai]`.

```python
import os

result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
    fence_output=True,
    use_schema_constraints=False,
)
```

**Matched model IDs**: `gpt-3.5-*`, `gpt-4*` (gpt-4o, gpt-4.1, etc.), `gpt-5*`
(gpt-5.4, gpt-5.4-mini, etc.), `o1`, `o3`, `o4-mini`, `chatgpt-*`.

> **Note:** OpenAI models currently require `fence_output=True` and
> `use_schema_constraints=False`.

### Ollama (local models)

No API key required. Install [Ollama](https://ollama.com/) and pull a model first.

```bash
ollama pull gemma2:2b
ollama serve
```

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",  # auto-routed to Ollama
)
```

**Matched model IDs**: `llama*`, `gemma*`, `mistral*`, `mixtral*`, `phi*`,
`qwen*`, `deepseek*`, `command-r*`, `codellama*`, `starcoder*`, and HuggingFace-style
IDs like `meta-llama/Llama-3.2-1B-Instruct`.

Configure a remote Ollama instance:

```python
result = lx.extract(
    ...,
    model_id="gemma2:2b",
    model_url="http://remote-server:11434",
)
```

Or set the `OLLAMA_BASE_URL` environment variable.

### Explicit provider selection

When a model ID could match multiple providers, force one explicitly:

```python
result = lx.extract(
    ...,
    model_id="my-custom-model",
    provider="ollama",    # "gemini", "openai", or "ollama"
)
```

---

## Extraction Options

### Key parameters for `lx.extract()`

| Parameter                 | Type           | Default                | Description                                         |
|---------------------------|----------------|------------------------|-----------------------------------------------------|
| `text_or_documents`       | any            | *(required)*           | Input text, file, URL, DataFrame, etc.              |
| `prompt_description`      | `str`          | `None`                 | Natural language extraction instructions            |
| `examples`                | `list`         | `None`                 | Few-shot `ExampleData` objects                      |
| `model_id`                | `str`          | `"gemini-3-flash-preview"` | Model identifier                               |
| `api_key`                 | `str`          | `None`                 | API key (or use env vars)                           |
| `temperature`             | `float`        | Provider default       | Sampling temperature                                |
| `max_char_buffer`         | `int`          | `1000`                 | Characters of context around each chunk             |
| `batch_length`            | `int`          | `10`                   | Documents per batch                                 |
| `max_workers`             | `int`          | `10`                   | Parallel inference workers                          |
| `extraction_passes`       | `int`          | `1`                    | Passes over the text (higher = better recall)       |
| `fence_output`            | `bool`         | Provider default       | Wrap LLM output in fence markers                    |
| `use_schema_constraints`  | `bool`         | `True`                 | Use structured output schema                        |
| `additional_context`      | `str`          | `None`                 | Extra context appended to each prompt               |
| `debug`                   | `bool`         | `False`                | Enable debug output                                 |
| `show_progress`           | `bool`         | `True`                 | Show progress bars                                  |
| `fetch_urls`              | `bool`         | `True`                 | Auto-fetch URL inputs                               |
| `provider`                | `str`          | Auto-detected          | Force a specific provider                           |
| `ocr_engine`              | engine/str     | `None`                 | OCR engine for scanned documents                    |

### Using `ExtractionOptions`

For reusable configuration, bundle settings into an `ExtractionOptions` object:

```python
options = lx.ExtractionOptions.for_model(
    model_id="gemini-2.5-flash",
    provider="gemini",
    provider_kwargs={"api_key": "your-key"},
    ocr=lx.OcrOptions(engine="deepseek-ocr"),
)

result = lx.extract(
    text_or_documents="input.pdf",
    prompt_description=prompt,
    examples=examples,
    options=options,
)
```

---

## Working with Results

`lx.extract()` returns an `AnnotatedDocument` (single input) or a
`list[AnnotatedDocument]` (multiple inputs).

### Accessing extractions

```python
result = lx.extract(...)

for ext in result.extractions:
    print(f"Class:      {ext.extraction_class}")
    print(f"Text:       {ext.extraction_text}")
    print(f"Attributes: {ext.attributes}")
    print(f"Span:       [{ext.char_interval.start_pos}:{ext.char_interval.end_pos}]")
    print(f"Alignment:  {ext.alignment_status}")
    print()
```

### Filtering by class

```python
medications = [e for e in result.extractions if e.extraction_class == "medication"]
```

### Extracting source snippets

```python
for ext in result.extractions:
    start = ext.char_interval.start_pos
    end = ext.char_interval.end_pos
    print(f"Source: ...{result.text[start:end]}...")
```

---

## Visualization

Generate interactive HTML visualizations that highlight extractions in the source
text with color-coded animations.

### From an extraction result

```python
html = lx.visualize(result)

# Save to file
with open("output.html", "w") as f:
    f.write(html.data if hasattr(html, "data") else html)
```

### From a saved JSONL file

```python
# Renders all documents in the file
html = lx.visualize("results.jsonl")

# Render only the third document
html = lx.visualize("results.jsonl", document_index=2)
```

### Options

| Parameter         | Type    | Default | Description                              |
|-------------------|---------|---------|------------------------------------------|
| `data_source`     | any     | —       | `AnnotatedDocument`, file path, or string path to JSONL |
| `animation_speed` | `float` | `1.0`   | Seconds between extraction transitions   |
| `show_legend`     | `bool`  | `True`  | Display color legend for extraction classes |
| `gif_optimized`   | `bool`  | `True`  | Larger fonts and contrast for screen capture |
| `document_index`  | `int`   | `None`  | Render a specific document from JSONL (0-based); `None` renders all |

### Features

- Color-coded highlighting by extraction class
- Animated progression through extractions with play/pause controls
- Slider for jumping to any extraction
- Current extraction attributes panel
- Auto-scroll to the active extraction
- Multiple documents rendered with independent controls when loading JSONL

---

## Saving & Loading Results

### Save to JSONL

```python
lx.io.save_annotated_documents(
    [result],
    output_dir=".",
    output_name="extractions.jsonl",
)
```

### Load from JSONL

```python
documents = list(lx.io.load_annotated_documents_jsonl("extractions.jsonl"))

for doc in documents:
    print(f"Document {doc.document_id}: {len(doc.extractions)} extractions")
```

---

## Long Documents & Multi-Pass Extraction

For long texts (articles, books, reports), LangExtract automatically chunks
the input and runs extraction on each chunk, then merges and aligns the results.

### Tuning parameters

```python
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,        # More passes = better recall
    max_workers=20,             # Parallel API calls
    max_char_buffer=1000,       # Context window per chunk
)
```

| Parameter           | Effect                                                      |
|---------------------|-------------------------------------------------------------|
| `extraction_passes` | Number of full passes over the text. Higher values improve recall at the cost of more API calls. |
| `max_char_buffer`   | Character budget per chunk. Smaller = more chunks, finer granularity. |
| `max_workers`       | Concurrency for API calls. Increase for faster throughput.  |
| `batch_length`      | Number of documents processed per batch.                    |

---

## OCR for Images & Scanned PDFs

LangExtract can OCR images and scanned PDFs before extraction, using Ollama
vision models or other OCR engines.

### Using Ollama vision models

```python
from langextract import ocr

# OCR a single image
text = ocr.ocr_image(
    image_data=pathlib.Path("scan.png").read_bytes(),
    model_id="deepseek-ocr",
)
print(text)
```

### OCR during extraction

```python
result = lx.extract(
    text_or_documents=pathlib.Path("scanned_report.pdf"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    ocr_engine="deepseek-ocr",       # Ollama model name
)
```

### Supported OCR engines

| Engine            | Installation        | Usage                                  |
|-------------------|---------------------|----------------------------------------|
| Ollama deepseek   | `ollama pull deepseek-ocr` | `ocr_engine="deepseek-ocr"`     |
| Ollama GLM        | `ollama pull glm-ocr`     | `ocr_engine="glm-ocr"`          |
| PaddleOCR         | `pip install langextract[ocr]` | `ocr_engine="paddleocr"`   |
| Tesseract         | `pip install langextract[ocr]` | `ocr_engine="pytesseract"` |

---

## Parser Backends

LangExtract supports multiple parser backends per file format. You can select
one explicitly or let the library choose the default.

### Selecting backends

```python
result = lx.extract(
    text_or_documents=pathlib.Path("report.pdf"),
    prompt_description=prompt,
    examples=examples,
    readable_pdf_backend="pdfplumber",   # or "pypdf", "pymupdf"
    html_backend="beautifulsoup",        # or "trafilatura" (default)
    docx_backend="docx2txt",             # or "python-docx" (default)
)
```

### Using `ParserBackendOptions`

```python
from langextract.extraction import ParserBackendOptions

backends = ParserBackendOptions(
    readable_pdf_backend="pymupdf",
    scanned_pdf_backend="paddleocr",
    html_backend="trafilatura",
)

result = lx.extract(
    ...,
    parser_backends=backends,
)
```

### Available backends by format

| Format       | Default         | Alternatives              | Extra     |
|--------------|-----------------|---------------------------|-----------|
| Readable PDF | pdfplumber      | pypdf, pymupdf            | `[pdf]`   |
| Scanned PDF  | *(needs OCR)*   | paddleocr, pytesseract    | `[ocr]`   |
| DOCX         | python-docx     | docx2txt                  | `[office]`|
| HTML         | trafilatura     | beautifulsoup4            | `[html]`  |
| Excel        | openpyxl        | —                         | `[xlsx]`  |

---

## Schema Constraints

When `use_schema_constraints=True` (the default for Gemini), LangExtract
generates a JSON schema from your examples and sends it to the provider's
structured output mode. This improves output consistency.

```python
# Schema constraints enabled (default for Gemini)
result = lx.extract(
    ...,
    model_id="gemini-2.5-flash",
    use_schema_constraints=True,
)

# Disable schema constraints (required for OpenAI currently)
result = lx.extract(
    ...,
    model_id="gpt-4o",
    use_schema_constraints=False,
    fence_output=True,
)
```

---

## Configuration Files

Load extraction settings from YAML, JSON, or TOML files.

### YAML example

```yaml
# extract_config.yaml
model_id: gemini-2.5-flash
fetch_urls: true
text_column: content
ocr_engine: deepseek-ocr
```

### Loading a config

```python
options = lx.load_extraction_config("extract_config.yaml")

result = lx.extract(
    text_or_documents="input.txt",
    prompt_description=prompt,
    examples=examples,
    options=options,
)
```

---

## Custom Provider Plugins

Third-party providers can be packaged as pip-installable plugins using Python
entry points.

### Scaffold a new plugin

```bash
python scripts/create_provider_plugin.py
```

This generates a project with:
- A provider class inheriting from `lx.core.base_model.BaseLanguageModel`
- A schema class inheriting from `lx.core.schema.BaseSchema`
- A `pyproject.toml` with the `langextract.providers` entry point

### Plugin entry point

```toml
# In your plugin's pyproject.toml
[project.entry-points."langextract.providers"]
my_provider = "my_package:MyLanguageModel"
```

### Provider class skeleton

```python
import langextract as lx

@lx.providers.registry.register(r'^my-model-')
class MyLanguageModel(lx.core.base_model.BaseLanguageModel):

    @classmethod
    def get_schema_class(cls):
        return MySchema  # or None

    def infer(self, batch_prompts, **kwargs):
        # Call your API and yield ScoredOutput objects
        for prompt in batch_prompts:
            response = call_my_api(prompt)
            yield lx.core.types.ScoredOutput(output=response, score=0.0)
```

---

## Environment Variables

| Variable              | Used by         | Description                                      |
|-----------------------|-----------------|--------------------------------------------------|
| `LANGEXTRACT_API_KEY` | Gemini, OpenAI  | Universal API key (works for any provider)       |
| `GEMINI_API_KEY`      | Gemini          | Gemini-specific API key                          |
| `OPENAI_API_KEY`      | OpenAI          | OpenAI-specific API key                          |
| `OLLAMA_BASE_URL`     | Ollama          | Ollama server URL (default: `http://localhost:11434`) |

### Using a `.env` file

Create a `.env` file (see `.env.example`) and load it with `python-dotenv`:

```bash
LANGEXTRACT_API_KEY=your-key-here
# Or provider-specific:
# GEMINI_API_KEY=your-gemini-key
# OPENAI_API_KEY=your-openai-key
```

```python
import dotenv
dotenv.load_dotenv()  # Loads .env into os.environ

import langextract as lx
result = lx.extract(...)  # API key picked up automatically
```

---

## Troubleshooting

### Common issues

**No extractions returned**
- Check that your `prompt_description` clearly describes what to extract.
- Ensure your examples include the exact extraction classes and attribute keys
  you expect in the output.
- Try increasing `extraction_passes` for long documents.

**ModuleNotFoundError for a file format**
- Install the required extra: `pip install langextract[pdf]`, `langextract[ocr]`, etc.

**OpenAI returns malformed output**
- Use `fence_output=True` and `use_schema_constraints=False` with OpenAI models.

**Ollama connection refused**
- Ensure `ollama serve` is running.
- If using a remote instance, set `OLLAMA_BASE_URL` or pass `model_url=`.

**Extractions have `None` char_interval**
- The LLM output couldn't be aligned to the source text.
- Check `alignment_status` for details. `MATCH_FUZZY` extractions may have
  approximate positions.

**Rate limiting / timeout**
- Reduce `max_workers` to lower concurrency.
- For large jobs, consider the Gemini Batch API (see `docs/examples/batch_api_example.md`).
