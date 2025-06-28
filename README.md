# Model Info Extractor

A tiny command-line helper to download a model card (README) from the Hugging Face Hub, list any URLs it contains, and optionally generate a short summary via an LLM.

## Prerequisites

* Python ≥ 3.8
* [uv](https://github.com/astral-sh/uv) (fast Python package manager / virtual-env tool)

## Usage

```bash
# Basic: download and show the README / model card
uv run python extract_readme.py <model_id>

# Optional: choose a different chat model for summarisation
uv run python extract_readme.py <model_id> <llm_model_id>
```

Example:

```bash
uv run python extract_readme.py bert-base-uncased
```

To enable the optional LLM summary step, set the `HF_TOKEN` environment variable to an access token that can call the chosen model:

```bash
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
``` 