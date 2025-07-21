#!/usr/bin/env python3
"""
gradio_app.py
--------------
Gradio application (with MCP support) exposing the functionality of
`extract_readme.py` as an interactive tool.  After launching, the app can be
used via a regular web UI *or* programmatically by any MCP-compatible LLM
client (Cursor, Claude Desktop, etc.).

Run locally:
    python gradio_app.py

This will start both the Gradio web server *and* the MCP endpoint.  The latter
is announced in the terminal when the app starts.
"""

from __future__ import annotations

import os
import re
import time
from types import TracebackType
from typing import Any, List, Sequence, Tuple, Type
from urllib.parse import urlparse

import gradio as gr
import requests
from huggingface_hub import HfApi, InferenceClient, ModelCard  # type: ignore

# -----------------------------------------------------------------------------
# Core logic (adapted from extract_readme.py)
# -----------------------------------------------------------------------------


def _extract_urls(text: str) -> List[str]:
    """Return a list of unique URLs found inside *text* preserving order."""
    url_pattern = re.compile(r"https?://[^\s\)\]\>'\"`]+")
    urls = url_pattern.findall(text)
    # Preserve insertion order while removing duplicates.
    seen: set[str] = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


def _summarise_external_urls(urls: Sequence[str]) -> List[Tuple[str, str]]:
    """Return a list of (url, summary) tuples using the r.jina.ai proxy."""
    if not urls:
        return []

    summaries: List[Tuple[str, str]] = []
    url_pattern = re.compile(r"https?://[^\s\)\]\>'\"`]+")

    for idx, original_url in enumerate(urls):
        proxy_url = f"https://r.jina.ai/{original_url}"
        try:
            resp = requests.get(proxy_url, timeout=15)
            resp.raise_for_status()
            cleaned_text = url_pattern.sub("", resp.text)
            summaries.append((original_url, cleaned_text))
        except Exception as err:  # pylint: disable=broad-except
            summaries.append((original_url, f"❌ Failed to fetch summary: {err}"))
        # Respect ~15 req/min rate-limit of r.jina.ai
        if idx < len(urls) - 1:
            time.sleep(4.1)
    return summaries


# -----------------------------------------------------------------------------
# Public MCP-exposed function
# -----------------------------------------------------------------------------


def extract_model_info(
    model_id: str,
    llm_model_id: str = "CohereLabs/c4ai-command-a-03-2025",
) -> str:
    """Fetch a Hugging Face model card, analyse it and optionally summarise it.

    Args:
        model_id: The *repository ID* of the model on Hugging Face (e.g.
            "bert-base-uncased").
        llm_model_id: ID of the LLM used for summarisation via the Inference
            Endpoint.  Defaults to Cohere Command R+.
        open_pr: If *True*, the generated summary will be posted as a **new
            discussion** in the specified model repo.  Requires a valid
            `HF_TOKEN` environment variable with write permissions.

    Returns:
        A single markdown-formatted string containing:
            1. The raw README.
            2. Extracted external URLs.
            3. Brief summaries of the external URLs (via r.jina.ai).
            4. A concise LLM-generated summary of the model card.
    """

    # ------------------------------------------------------------------
    # 1. Load model card
    # ------------------------------------------------------------------
    try:
        card = ModelCard.load(model_id)
    except Exception as err:  # pylint: disable=broad-except
        return f"❌ Failed to load model card for '{model_id}': {err}"

    combined_sections: List[str] = ["=== README markdown ===", card.text]

    # ------------------------------------------------------------------
    # 2. Extract URLs
    # ------------------------------------------------------------------
    unique_urls = _extract_urls(card.text)
    if unique_urls:
        combined_sections.append("\n=== URLs found ===")
        combined_sections.extend(unique_urls)

        EXCLUDED_KEYWORDS = ("colab.research.google.com", "github.com")
        filtered_urls = [
            u for u in unique_urls if not any(k in urlparse(u).netloc for k in EXCLUDED_KEYWORDS)
        ]

        if filtered_urls:
            combined_sections.append("\n=== Summaries via r.jina.ai ===")
            for url, summary in _summarise_external_urls(filtered_urls):
                combined_sections.append(f"\n--- {url} ---\n{summary}")
        else:
            combined_sections.append("\nNo external URLs (after filtering) detected in the model card.")
    else:
        combined_sections.append("\nNo URLs detected in the model card.")

    # ------------------------------------------------------------------
    # 3. Summarise with LLM (if token available)
    # ------------------------------------------------------------------
    hf_token = os.getenv("HF_TOKEN")
    summary_text: str | None = None
    if hf_token:
        client = InferenceClient(provider="auto", api_key=hf_token)
        prompt = (
            "You are given a lot of information about a machine learning model "
            "available on Hugging Face. Create a concise, technical and to the point "
            "summary highlighting the technical details, comparisons and instructions "
            "to run the model (if available). Think of the summary as a gist with all "
            "the information someone should need to know about the model without "
            "overwhelming them. Do not add any text formatting to your output text, "
            "keep it simple and plain text. If you have to then sparingly just use "
            "markdown for Heading and lists. Specifically do not use ** to bold text, "
            "just use # for headings and - for lists. No need to put any contact "
            "information in the summary. The summary is supposed to be insightful and "
            "information dense and should not be more than 200-300 words. Don't "
            "hallucinate and refer only to the content provided to you. Remember to "
            "be concise. Here is the information:\n\n" + "\n".join(combined_sections)
        )
        try:
            completion = client.chat.completions.create(
                model=llm_model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            summary_text = completion.choices[0].message.content
        except Exception as err:  # pylint: disable=broad-except
            return f"❌ Failed to generate summary: {err}"
    else:
        return "⚠️  HF_TOKEN environment variable not set. Please set it to enable summarisation."
    # Return only the summary text if available
    return summary_text or "❌ Summary generation failed for unknown reasons."


# -----------------------------------------------------------------------------
# Gradio UI & MCP launch
# -----------------------------------------------------------------------------

demo = gr.Interface(
    fn=extract_model_info,
    inputs=[
        gr.Textbox(value="bert-base-uncased", label="Model ID"),
        gr.Textbox(value="CohereLabs/c4ai-command-a-03-2025", label="LLM Model ID"),
    ],
    outputs=gr.Textbox(label="Result", lines=25),
    title="Model Card Inspector & Summariser",
    description=(
        "Fetch a model card from Hugging Face, extract useful links, optionally "
        "summarise it with an LLM and (optionally) open a discussion on the Hub. "
        "This tool is also available via MCP so LLM clients can call it directly."
    ),
)

if __name__ == "__main__":
    demo.launch(mcp_server=True) 