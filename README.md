<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

<h1 align="center">LangExtract</h1>

<p align="center">
  <strong>LLM-powered structured extraction from unstructured text — with source grounding, interactive visualization, and multi-format ingestion.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/langextract/"><img src="https://img.shields.io/pypi/v/langextract.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version" /></a>
  <a href="https://github.com/google/langextract"><img src="https://img.shields.io/github/stars/google/langextract.svg?style=flat-square&logo=github&label=Stars" alt="GitHub stars" /></a>
  <a href="https://github.com/google/langextract/actions/workflows/ci.yaml"><img src="https://img.shields.io/github/actions/workflow/status/google/langextract/ci.yaml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="Tests" /></a>
  <a href="https://doi.org/10.5281/zenodo.17015089"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17015089-blue?style=flat-square&logo=doi&logoColor=white" alt="DOI" /></a>
  <a href="https://github.com/google/langextract/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square&logo=apache&logoColor=white" alt="License" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Gemini-supported-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/OpenAI-supported-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/Ollama-supported-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" />
</p>

## 📑 Table of Contents

- [🔍 Introduction](#-introduction)
- [✨ Why LangExtract?](#-why-langextract)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔑 API Key Setup for Cloud Models](#-api-key-setup-for-cloud-models)
- [🔌 Adding Custom Model Providers](#-adding-custom-model-providers)
- [🤖 Using OpenAI Models](#-using-openai-models)
- [🏠 Using Local LLMs with Ollama](#-using-local-llms-with-ollama)
- [📚 More Examples](#-more-examples)
- [🌐 Community Providers](#-community-providers)
- [🤝 Contributing](#-contributing)
- [🧪 Testing](#-testing)
- [⚖️ Disclaimer](#%EF%B8%8F-disclaimer)

## 🔍 Introduction

LangExtract is a Python library that uses LLMs to extract structured information from unstructured text documents based on user-defined instructions. It processes materials such as clinical notes or reports, identifying and organizing key details while ensuring the extracted data corresponds to the source text.

## ✨ Why LangExtract?

1.  🎯 **Precise Source Grounding** — Maps every extraction to its exact location in the source text, enabling visual highlighting for easy traceability and verification.
2.  🔒 **Reliable Structured Outputs** — Enforces a consistent output schema based on your few-shot examples, leveraging controlled generation in supported models like Gemini to guarantee robust, structured results.
3.  📄 **Optimized for Long Documents** — Overcomes the "needle-in-a-haystack" challenge of large document extraction by using an optimized strategy of text chunking, parallel processing, and multiple passes for higher recall.
4.  📊 **Interactive Visualization** — Instantly generates a self-contained, interactive HTML file to visualize and review thousands of extracted entities in their original context.
5.  🔀 **Flexible LLM Support** — Supports your preferred models, from cloud-based LLMs like the Google Gemini family to local open-source models via the built-in Ollama interface.
6.  🧩 **Adaptable to Any Domain** — Define extraction tasks for any domain using just a few examples. LangExtract adapts to your needs without requiring any model fine-tuning.
7.  🧠 **Leverages LLM World Knowledge** — Utilize precise prompt wording and few-shot examples to influence how the extraction task may utilize LLM knowledge. The accuracy of any inferred information and its adherence to the task specification are contingent upon the selected LLM, the complexity of the task, the clarity of the prompt instructions, and the nature of the prompt examples.

## 🚀 Quick Start

> **Note:** Using cloud-hosted models like Gemini requires an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section for instructions on how to get and configure your key.

Extract structured information with just a few lines of code.

### 1. Define Your Extraction Task

First, create a prompt that clearly describes what you want to extract. Then, provide a high-quality example to guide the model.

```python
import langextract as lx
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]
```

> **Note:** Examples drive model behavior. Each `extraction_text` should ideally be verbatim from the example's `text` (no paraphrasing), listed in order of appearance. LangExtract raises `Prompt alignment` warnings by default if examples don't follow this pattern—resolve these for best results.
>
> **Grounding:** LLMs may occasionally extract content from few-shot examples rather than the input text. LangExtract automatically detects this: extractions that cannot be located in the source text will have `char_interval = None`. Filter these out with `[e for e in result.extractions if e.char_interval]` to keep only grounded results.

### 2. Run the Extraction

Provide your input text and the prompt materials to the `lx.extract` function.

```python
# The input text to be processed
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**: `gemini-2.5-flash` is the recommended default, offering an excellent balance of speed, cost, and quality. For highly complex tasks requiring deeper reasoning, `gemini-2.5-pro` may provide superior results. For large-scale or production use, a Tier 2 Gemini quota is suggested to increase throughput and avoid rate limits. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> `model_id` is fully configurable. If you omit it, LangExtract currently defaults to `gemini-3-flash-preview`. For pinned production behavior, pass the exact Gemini model you want, such as `gemini-2.5-flash` or `gemini-2.5-pro`.
>
> **Model Lifecycle**: Note that Gemini models have a lifecycle with defined retirement dates. Users should consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed about the latest stable and legacy versions.

### 3. Visualize the Results

The extractions can be saved to a `.jsonl` file, a popular format for working with language model data. LangExtract can then generate an interactive HTML visualization from this file to review the entities in context.

```python
# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to the text evidence - extracting "longing" for Lady Juliet's emotional state and identifying "yearning" from "gazed longingly at the stars." The task could be modified to generate attributes that draw more heavily from the LLM's world knowledge (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`). The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### 📈 Scaling to Longer Documents

For larger texts, you can process entire documents directly from URLs with parallel processing and enhanced sensitivity:

```python
# Process Romeo & Juliet directly from Project Gutenberg
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=20,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)
```

This approach can extract hundreds of entities from full novels while maintaining high accuracy. The interactive visualization seamlessly handles large result sets, making it easy to explore hundreds of entities from the output JSONL file. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results and performance insights.

### 📂 Additional Input Types

`text_or_documents` also accepts `pathlib.Path`, `pandas.DataFrame`, `bytes`, and URL strings:

```python
import pathlib
import langextract as lx

# 1. Plain text input
result = lx.extract(
    text_or_documents="Acme Corp paid invoice INV-104 for $1,250 on 2026-03-01.",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# 2. Readable PDF with an explicit backend choice
result = lx.extract(
    text_or_documents=pathlib.Path("report.pdf"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    readable_pdf_backend="pdfplumber",
)

# 3. Scanned PDF with an OCR parser backend choice
result = lx.extract(
    text_or_documents=pathlib.Path("scan.pdf"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    scanned_pdf_backend="paddleocr",
)

# 4. Image extraction
result = lx.extract(
    text_or_documents=pathlib.Path("receipt.png"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    image_backend="paddleocr",
)

# 5. DOCX extraction
result = lx.extract(
    text_or_documents=pathlib.Path("contract.docx"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    docx_backend="python-docx",
)
```

```python
# 6. CSV works with the default pandas backend in the base install
csv_result = lx.extract(
    text_or_documents=pathlib.Path("notes.csv"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Excel needs openpyxl installed; use config for category-specific control
xlsx_result = lx.extract(
    text_or_documents=pathlib.Path("notes.xlsx"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    config=lx.ExtractionOptions(
        ingestion=lx.IngestionOptions(
            parser_backends=lx.ParserBackendOptions(excel_backend="openpyxl"),
        ),
    ),
)

# 7. URL / HTML extraction
html_result = lx.extract(
    text_or_documents=pathlib.Path("page.html"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    html_backend="trafilatura",
)

url_result = lx.extract(
    text_or_documents="https://example.com/article",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    url_backend="beautifulsoup",
)
```

Plain text and CSV work in the base install. Digital PDFs need `langextract[pdf]`; scanned PDFs need `langextract[pdf,ocr]`; image OCR needs `langextract[ocr]`; DOCX, PPTX, and XLSX need `langextract[office]` or `langextract[xlsx]`; parsed HTML and URL extraction needs `langextract[html]` or `langextract[url]`. Remote binary URLs for PDFs, spreadsheets, and images are not enabled yet.

If you want to inspect backend choices from Python instead of reading the registry or CLI output directly, use `lx.list_available_backends()`:

```python
import langextract as lx

for report in lx.list_available_backends():
    print(report.category, "default=", report.default)
    for backend in report.backends:
        print(
            " ",
            backend.name,
            "installed=", backend.installed,
            "usable=", backend.usable,
            "install=", backend.install_commands,
        )
```

Each backend entry includes its description, whether it is installed and usable in the current environment, whether it is the default, its auto-selection rank when applicable, and the relevant `langextract[...]` extra or direct `pip install ...` command when an optional dependency is needed.

### ⚙️ Advanced Extraction Configuration

Use `ExtractionOptions` when you want one typed configuration object for provider selection, model selection, OCR, and ingestion behavior:

```python
import pathlib
import langextract as lx

options = lx.ExtractionOptions.for_model(
    model_id="gpt-4o-mini",
    provider="openai",
    provider_kwargs={"api_key": "your-openai-key"},
    ingestion=lx.IngestionOptions(
        fetch_urls=False,
        text_column="body",
        parser_backends=lx.ParserBackendOptions(
            readable_pdf_backend="pymupdf",
            csv_backend="pandas",
            excel_backend="pandas",
        ),
    ),
    ocr=lx.OcrOptions(engine="deepseek-ocr", config={"timeout": 30}),
)

result = lx.extract(
    text_or_documents=pathlib.Path("report.pdf"),
    prompt_description=prompt,
    examples=examples,
    config=options,
    use_schema_constraints=False,
)
```

All sub-configurations are optional. If you omit them, LangExtract keeps the existing defaults. Direct per-call arguments such as `api_key=`, `fetch_urls=`, or `ocr_engine=` still override values inside `config`. The older `options=` alias remains supported for backward compatibility. Parser backend names are validated up front, and selecting a backend whose dependency is missing now raises a runtime error that names the selected backend, the missing package, and the relevant `langextract[...]` extra to install.

Use `config=lx.ExtractionOptions(...)` as the main backend-selection path. The direct keyword shortcuts on `extract()` are intentionally limited to the most common cases: `readable_pdf_backend=`, `scanned_pdf_backend=`, `image_backend=`, `docx_backend=`, `html_backend=`, and `url_backend=`. For every other category, use `lx.ParserBackendOptions(...)` inside `config.ingestion.parser_backends`.

When you do not specify a backend, LangExtract uses the category default backend if one exists. It does not silently search other libraries in that mode. Use `"auto"` when you want deterministic fallback across implemented backends. Auto mode skips backends that are registered but not yet wired, and also skips backends whose optional dependency is not installed.

Backend selection precedence is: direct `extract(..., readable_pdf_backend=...)` style keyword overrides, then `parser_backends=...`, then `config.ingestion.parser_backends`, then the category default.

Implemented parser categories and backend choices:

| Category | Implemented backends | Default / auto behavior |
| --- | --- | --- |
| `txt` | `builtin` | Default is `builtin`. `charset-normalizer` is registered but not wired yet. |
| `readable_pdf` | `pymupdf`, `pdfplumber`, `pypdf` | Default is `pymupdf`. `auto` is supported. |
| `scanned_pdf` | `paddleocr`, `tesseract` | Default is `paddleocr`. `auto` is supported. `ocrmypdf` is registered but not wired yet. |
| `image` | `paddleocr`, `tesseract` | Default is `paddleocr`. `auto` is supported. `easyocr` is registered but not wired yet. |
| `csv` | `pandas`, `builtin_csv` | Default is `pandas`. `auto` is supported. |
| `excel` | `pandas`, `openpyxl` | Default is `pandas`. `auto` is supported. |
| `docx` | `python-docx`, `docx2txt` | Default is `python-docx`. `auto` is supported. `mammoth` is registered but not wired yet. |
| `pptx` | `python-pptx` | Default is `python-pptx`. `auto` is supported, but today it resolves only to `python-pptx`. `libreoffice` and `tika` are registered but not wired yet. |
| `html` (local files) | `builtin`, `trafilatura`, `beautifulsoup` | Default is `trafilatura`. `auto` is supported. Use `builtin` when you want raw HTML text instead of parsed page content. |
| `url` (HTML pages) | `trafilatura`, `beautifulsoup` | Default is `trafilatura`. `auto` is supported. Use `fetch_urls=False` when you want the URL string treated as plain text instead of fetching the page. If you fetch HTML yourself and want raw markup, pass that HTML content directly and use `html/builtin`. |

Registered but not yet fully wired categories:

- `table_pdf`: `pdfplumber`, `camelot`, `tabula`
- `doc`: `libreoffice`, `antiword`, `tika`
- `ppt`: `libreoffice`, `tika`

Explicit backend selection example:

```python
import pathlib
import langextract as lx

result = lx.extract(
    text_or_documents=pathlib.Path("report.pdf"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    readable_pdf_backend="pdfplumber",
)
```

Auto mode example:

```python
import pathlib
import langextract as lx

result = lx.extract(
    text_or_documents=pathlib.Path("report.pdf"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    readable_pdf_backend="auto",
)
```

Use `backend_preference_order` when you want to customize the fallback order:

```python
import pathlib
import langextract as lx

config = lx.ExtractionOptions(
    ingestion=lx.IngestionOptions(
        parser_backends=lx.ParserBackendOptions(
            readable_pdf_backend="auto",
            backend_preference_order={
                "readable_pdf": ("pypdf", "pdfplumber", "pymupdf"),
            },
        ),
    ),
)

result = lx.extract(
    text_or_documents=pathlib.Path("report.pdf"),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    config=config,
)
```

Config file example (see also `examples/configs/` for ready-to-use templates
in YAML, TOML, and JSON):

```yaml
# extract.yaml
provider: gemini
model: gemini-2.5-flash
fetch_urls: true
ocr_engine: deepseek-ocr

backends:
    readable_pdf: auto
    scanned_pdf: paddleocr
    image: paddleocr
    docx: python-docx
    excel: openpyxl
    url: trafilatura
```

The same configuration can also be expressed as TOML or JSON:

```toml
# extract.toml
provider = "gemini"
model = "gemini-2.5-flash"
fetch_urls = true
ocr_engine = "deepseek-ocr"

[backends]
readable_pdf = "auto"
scanned_pdf = "paddleocr"
url = "trafilatura"
```

```json
{
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "fetch_urls": true,
    "backends": {
        "readable_pdf": "auto",
        "url": "trafilatura"
    }
}
```

**Supported config keys:**

| Key | Type | Description |
|-----|------|-------------|
| `provider` | string | Provider name: `gemini`, `openai`, or `ollama` |
| `model` | string | Model identifier (alias for `model_id`) |
| `fetch_urls` | boolean | Whether to fetch URL content (default: `true`) |
| `ocr_engine` | string | OCR engine: `deepseek-ocr` or `glm-ocr` |
| `text_column` | string | Column name for text in tabular input (default: `text`) |
| `id_column` | string | Column name for row IDs |
| `additional_context_column` | string | Column name for extra context |
| `backends` | mapping | Category-to-backend mapping (see backend table above) |

Inside `backends`, use category names as keys (`readable_pdf`, `scanned_pdf`,
`image`, `docx`, `html`, `url`, `csv`, `excel`, `pptx`, `txt`) and backend
names or `"auto"` as values. Use `"default"` or `null` to leave a category at
its default.

```python
import pathlib
import langextract as lx

config = lx.load_extraction_config("extract.yaml")

result = lx.extract(
        text_or_documents=pathlib.Path("document.pdf"),
        prompt_description=prompt,
        examples=examples,
        config=config,
)
```

### 💻 CLI Usage

The CLI uses the same extraction API and config-file model as Python. Store your few-shot examples in JSON:

```json
[
    {
        "text": "Ada Lovelace wrote the first algorithm.",
        "extractions": [
            {
                "extraction_class": "person",
                "extraction_text": "Ada Lovelace"
            }
        ]
    }
]
```

Then run commands such as:

```bash
langextract backends
langextract backends --category readable_pdf

langextract extract --prompt "Extract people" --examples examples.json input.txt

langextract extract --prompt "Extract invoice fields" --examples examples.json report.pdf --readable-pdf-backend pdfplumber --model-id gemini-2.5-flash

langextract extract --prompt "Extract receipt fields" --examples examples.json scan.pdf --scanned-pdf-backend paddleocr --config extract.yaml --output result.json

langextract extract --prompt "Extract article metadata" --examples examples.json https://example.com/article --url-backend beautifulsoup --model-id gemini-2.5-flash
```

#### 📖 CLI Reference

**Subcommands:**

| Command | Description |
|---------|-------------|
| `langextract version` | Print the installed LangExtract version |
| `langextract backends` | List available ingestion backends and their status |
| `langextract extract` | Run extraction on text, files, or URLs |

**`backends` flags:**

| Flag | Description |
|------|-------------|
| `--category NAME` | Filter output to a single category (e.g. `csv`, `readable_pdf`) |
| `--json` | Output structured JSON instead of a human-readable table |

**`extract` flags:**

| Flag | Description |
|------|-------------|
| `INPUT` (positional) | Text string, file path, or URL to extract from |
| `--prompt`, `-p` | Extraction task description (required) |
| `--examples`, `-e` | Path to a JSON file with few-shot examples (required) |
| `--config` | Path to a YAML, JSON, or TOML config file |
| `--output`, `-o` | Write JSON output to a file instead of stdout |
| `--model-id` | Model identifier (default: `gemini-3-flash-preview`) |
| `--provider` | Provider name: `gemini`, `openai`, or `ollama` |
| `--api-key` | Provider API key (can also be set via environment variable) |
| `--model-url` | Self-hosted model endpoint URL |
| `--ocr-engine` | OCR engine override: `deepseek-ocr` or `glm-ocr` |
| `--quiet`, `-q` | Suppress progress output |

**Backend selection flags on `extract`:**

| Flag | Category | Example backends |
|------|----------|-----------------|
| `--readable-pdf-backend` | `readable_pdf` | `pymupdf`, `pdfplumber`, `pypdf`, `auto` |
| `--scanned-pdf-backend` | `scanned_pdf` | `paddleocr`, `tesseract`, `auto` |
| `--image-backend` | `image` | `paddleocr`, `tesseract`, `auto` |
| `--docx-backend` | `docx` | `python-docx`, `docx2txt`, `auto` |
| `--html-backend` | `html` | `builtin`, `trafilatura`, `beautifulsoup`, `auto` |
| `--url-backend` | `url` | `trafilatura`, `beautifulsoup`, `auto` |

CLI backend flags become direct `extract()` keyword arguments, so they take
precedence over config-file backend settings loaded via `--config`.

### ☁️ Vertex AI Batch Processing

Save costs on large-scale tasks by enabling Vertex AI Batch API: `language_model_params={"vertexai": True, "batch": {"enabled": True}}`.

See an example of the Vertex AI Batch API usage in [this example](docs/examples/batch_api_example.md).

## 📦 Installation

### ⚡ From PyPI

```bash
pip install langextract
```

*Recommended for most users. For isolated environments, consider using a virtual environment:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### 📥 From Source

LangExtract uses modern Python packaging with `pyproject.toml` for dependency management:

*Installing with `-e` puts the package in development mode, allowing you to modify the code without reinstalling.*


```bash
git clone https://github.com/google/langextract.git
cd langextract

# For basic installation:
pip install -e .

# For development (includes linting tools):
pip install -e ".[dev]"

# For testing (includes pytest):
pip install -e ".[test]"
```

### 🐳 Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

### 🧰 Optional Extras

Install optional dependencies for additional input formats:

```bash
pip install langextract[pdf]     # PDF parser family (pymupdf, pdfplumber, pypdf)
pip install langextract[ocr]     # Wired OCR parser backends for images/scanned PDFs (paddleocr, pytesseract)
pip install langextract[office]  # Wired Office backends (openpyxl, python-docx, docx2txt, python-pptx)
pip install langextract[html]    # Local HTML parsing plus the same parser libraries used for URLs
pip install langextract[url]     # Alias focused on URL extraction; installs the same libraries as [html]
pip install langextract[xlsx]    # Legacy alias for Excel/openpyxl support
pip install langextract[openai]  # OpenAI model support
pip install "langextract[pdf,ocr]"  # Common combo for scanned PDF workflows
pip install langextract[full]    # All optional backend families + OpenAI
pip install langextract[all]     # Backward-compatible alias for full
```

| Extra | Categories unlocked | Backends provided |
|-------|--------------------|--------------------|
| `pdf` | `readable_pdf` | pymupdf, pdfplumber, pypdf |
| `ocr` | `scanned_pdf`, `image` | paddleocr, pytesseract |
| `office` | `docx`, `pptx`, `excel` | python-docx, docx2txt, python-pptx, openpyxl |
| `html` | `html`, `url` | trafilatura, beautifulsoup4 |
| `xlsx` | `excel` | openpyxl (legacy alias) |
| `openai` | — | OpenAI provider support |

Built-in backends such as `txt/builtin`, `html/builtin`, and `csv/builtin_csv` do not require optional extras. Scanned-PDF parser backends render pages through PyMuPDF first, so the default scanned-PDF path needs both `langextract[pdf]` and `langextract[ocr]`. If you choose the `tesseract` parser backend, install the system `tesseract` executable as well. Installing an extra only makes its Python dependency available; it does not turn a registered-only backend into a fully supported ingestion path.

## 🔑 API Key Setup for Cloud Models

When using LangExtract with cloud-hosted models (like Gemini or OpenAI), you'll need to
set up an API key. On-device models don't require an API key. LangExtract has
built-in Gemini, OpenAI, and Ollama provider support, and custom providers can
be added through the plugin system.

### 🔗 API Key Sources

Get API keys from:

*   [AI Studio](https://aistudio.google.com/app/apikey) for Gemini models
*   [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) for enterprise use
*   [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI models

### 🌐 Setting up API key in your environment

**Option 1: Environment Variable**

```bash
export LANGEXTRACT_API_KEY="your-api-key-here"
```

**Option 2: .env File (Recommended)**

Add your API key to a `.env` file (see [.env.example](.env.example)):

```bash
# Add API key to .env file
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
# Or use provider-specific keys:
# GEMINI_API_KEY=your-gemini-key
# OPENAI_API_KEY=your-openai-key
EOF

# Keep your API key secure
echo '.env' >> .gitignore
```

In your Python code:
```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash"
)
```

**Option 3: Direct API Key (Not Recommended for Production)**

You can also provide the API key directly in your code, though this is not recommended for production use:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only use this for testing/development
)
```

**Option 4: Vertex AI (Service Accounts)**

Use [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) for authentication with service accounts:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    language_model_params={
        "vertexai": True,
        "project": "your-project-id",
        "location": "global"  # or regional endpoint
    }
)
```

## 🔌 Adding Custom Model Providers

LangExtract supports custom LLM providers via a lightweight plugin system. You can add support for new models without changing core code.

- Add new model support independently of the core library
- Distribute your provider as a separate Python package
- Keep custom dependencies isolated
- Override or extend built-in providers via priority-based resolution

See the detailed guide in [Provider System Documentation](langextract/providers/README.md) to learn how to:

- Register a provider with `@registry.register(...)`
- Publish an entry point for discovery
- Optionally provide a schema with `get_schema_class()` for structured output
- Integrate with the factory via `create_model(...)`

## 🤖 Using OpenAI Models

LangExtract supports OpenAI models (requires optional dependency: `pip install langextract[openai]`):

```python
import os
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",  # Automatically selects OpenAI provider
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False` because LangExtract doesn't implement schema constraints for OpenAI yet. If you need to force built-in provider selection explicitly, pass `provider="openai"`.

## 🏠 Using Local LLMs with Ollama
LangExtract supports local inference using Ollama, allowing you to run models without API keys:

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",  # Automatically selects Ollama provider
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False
)
```

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

### 🔍 OCR Preprocessing with Ollama

LangExtract can use Ollama vision models (`deepseek-ocr`, `glm-ocr`) to OCR images and scanned PDF pages. This runs locally through your Ollama server:

```python
import pathlib
import langextract as lx
from langextract import ocr

# OCR a single image
text = ocr.ocr_image(
    image_data=pathlib.Path("scan.png").read_bytes(),
    model_id="deepseek-ocr",
)

# Or let extract() OCR a local image or scanned PDF first
result = lx.extract(
    text_or_documents=pathlib.Path("scan.png"),
    prompt_description=prompt,
    examples=examples,
    ocr_engine="deepseek-ocr",
)
```

OCR-backed ingestion in `lx.extract()` can use the default parser backends for local image files and scanned PDFs when their optional dependencies are installed. Pass `ocr_engine=` or `config=lx.ExtractionOptions(ocr=...)` when you want to override the parser defaults with an OCR engine choice. Remote binary URL ingestion is still not enabled.

For detailed installation, Docker setup, and examples, see [`examples/ollama/`](examples/ollama/).

## 📚 More Examples

Additional examples of LangExtract in action:

### *Romeo and Juliet* Full Text Extraction

LangExtract can process complete documents directly from URLs. This example demonstrates extraction from the full text of *Romeo and Juliet* from Project Gutenberg (147,843 characters), showing parallel processing, sequential extraction passes, and performance optimization for long document processing.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes of LangExtract's baseline capability only. It does not represent a finished or approved product, is not intended to diagnose or suggest treatment of any disease or condition, and should not be used for medical advice.

LangExtract excels at extracting structured medical information from clinical text. These examples demonstrate both basic entity recognition (medication names, dosages, routes) and relationship extraction (connecting medications to their attributes), showing LangExtract's effectiveness for healthcare applications.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live interactive demo on HuggingFace Spaces that shows how LangExtract can automatically structure radiology reports. Try it directly in your browser with no setup required.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## 🌐 Community Providers

Extend LangExtract with custom model providers! Check out our [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry to discover providers created by the community or add your own.

For detailed instructions on creating a provider plugin, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to get started
with development, testing, and pull requests. You must sign a
[Contributor License Agreement](https://cla.developers.google.com/about)
before submitting patches.



## 🧪 Testing

To run tests locally from the source:

```bash
# Clone the repository
git clone https://github.com/google/langextract.git
cd langextract

# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests
```

Or reproduce the full CI matrix locally with tox:

```bash
tox  # runs pylint + pytest on Python 3.10, 3.11, and 3.12
```

### 🧪 Ollama Integration Testing

If you have Ollama installed locally, you can run integration tests:

```bash
# Test Ollama integration (requires Ollama running with gemma2:2b model)
tox -e ollama-integration
```

This test will automatically detect if Ollama is available and run real inference tests.

## 🛠️ Development

### 🎨 Code Formatting

This project uses automated formatting tools to maintain consistent code style:

```bash
# Auto-format all code
./autoformat.sh

# Or run formatters separately
isort langextract trafficmind tests --profile google --line-length 80
pyink langextract trafficmind tests --config pyproject.toml
```

For the same checks CI runs, use:

```bash
tox -e format,imports
```

### 🧹 Pre-commit Hooks

For automatic formatting checks:
```bash
pre-commit install  # One-time setup
pre-commit run --all-files  # Manual run
```

### 🔎 Linting

Run linting before submitting PRs:

```bash
tox -e lint-src,lint-tests
```

TrafficMind remains a local-first subsystem. For supported runtime
profiles, startup validation, architecture boundaries, and integration
foundations, see
[docs/trafficmind/deployment.md](docs/trafficmind/deployment.md) and
[docs/trafficmind/architecture.md](docs/trafficmind/architecture.md).
Use [.env.example](.env.example) as the starting point for local config.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## ⚖️ Disclaimer

This is not an officially supported Google product. If you use
LangExtract in production or publications, please cite accordingly and
acknowledge usage. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).
For health-related applications, use of LangExtract is also subject to the
[Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

<p align="center">
  Made with ❤️ by the LangExtract team at Google
  <br />
  <sub>🌟 Star this repo if you find it useful!</sub>
</p>
