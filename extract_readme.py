#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["huggingface_hub", "requests"]
# ///
"""
extract_readme.py
-----------------
Simple CLI tool to fetch and inspect the README (model card) of a model hosted on
https://huggingface.co.

Usage::

    python extract_readme.py <model_id>

Example::

    python extract_readme.py bert-base-uncased

This will download the model card and print two sections:
1. Metadata (YAML front-matter) parsed as a Python dict.
2. The human-readable markdown body of the card.

Requires:
    pip install huggingface_hub
"""
import sys
from huggingface_hub import ModelCard
import re
from urllib.parse import urlparse
import os

from huggingface_hub import InferenceClient  # type: ignore


def main() -> None:  # pragma: no cover
    if len(sys.argv) < 2:
        print("Usage: python extract_readme.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]

    try:
        # Load the model card directly from the Hub.
        card = ModelCard.load(model_id)
    except Exception as err:  # pylint: disable=broad-except
        print(f"❌ Failed to load model card for '{model_id}': {err}")
        sys.exit(1)

    # Prepare container for combined output (README first).
    combined_sections: list[str] = ["=== README markdown ===", card.text]

    # Extract and display all URLs found in the markdown body.
    url_pattern = re.compile(r"https?://[^\s\)\]\>\'\"`]+")
    urls = url_pattern.findall(card.text)

    if urls:
        # Preserve order while removing duplicates.
        unique_urls = list(dict.fromkeys(urls))

        # Record URLs section for the final output.
        combined_sections.append("\n=== URLs found ===")
        combined_sections.extend(unique_urls)

        # Filter out arxiv, colab, and GitHub links.
        EXCLUDED_KEYWORDS = (
            "arxiv.org",
            "ar5iv.org",
            "colab.research.google.com",
            "github.com",
        )

        filtered_urls = [
            u for u in unique_urls if not any(k in urlparse(u).netloc for k in EXCLUDED_KEYWORDS)
        ]

        if filtered_urls:
            import time
            import requests

            combined_sections.append("\n=== Summaries via r.jina.ai ===")

            # NOTE: The free r.jina.ai endpoint allows ~15 requests/min (4s/request).

            for idx, original_url in enumerate(filtered_urls):
                proxy_url = f"https://r.jina.ai/{original_url}"
                try:
                    resp = requests.get(proxy_url, timeout=15)
                    resp.raise_for_status()
                    # Remove URLs from the extracted text to keep output concise.
                    cleaned_text = url_pattern.sub("", resp.text)
                    combined_sections.append(f"\n--- {original_url} ---\n{cleaned_text}")
                except Exception as err:  # pylint: disable=broad-except
                    sys.stderr.write(f"❌ Failed to fetch '{original_url}': {err}\n")

                # Respect rate limit: sleep ~4 seconds between calls (<=15/minute)
                if idx < len(filtered_urls) - 1:
                    time.sleep(4.1)
        else:
            combined_sections.append("\nNo external URLs (after filtering) detected in the model card.")
    else:
        combined_sections.append("\nNo URLs detected in the model card.")

    # Print the final aggregated output.
    combined_output = "\n".join(combined_sections)

    # Summarize the collected information using Cohere's LLM via Hugging Face Inference Client.
    try:
        hf_token = os.environ["HF_TOKEN"]
    except KeyError:
        sys.stderr.write("⚠️  HF_TOKEN environment variable not set. Skipping summarization.\n")
    else:
        client = InferenceClient(provider="cohere", api_key=hf_token)

        prompt = f"You are given a lot of information about a machine learning model available on Hugging Face. \
        Create a concise, technical and to the point summary highlighting the technical details, comparisons and instuctions to run the model. \
            Don't hallucinate and refer only to the content provided to you. Remember to be concise. Here is the information:\n\n{combined_output}"

        try:
            completion = client.chat.completions.create(
                model="CohereLabs/c4ai-command-a-03-2025",
                messages=[{"role": "user", "content": prompt}],
            )
            print("\n=== SUMMARY ===")
            print(completion.choices[0].message)
        except Exception as err:  # pylint: disable=broad-except
            sys.stderr.write(f"❌ Failed to generate summary: {err}\n")


if __name__ == "__main__":
    main() 