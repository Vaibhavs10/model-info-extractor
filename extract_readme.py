#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["huggingface_hub"]
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

    # Print the markdown content excluding the metadata header.
    print("\n=== README markdown ===")
    print(card.text)

    # Extract and display all URLs found in the markdown body.
    url_pattern = re.compile(r"https?://[^\s\)\]\>\'\"`]+")
    urls = url_pattern.findall(card.text)

    if urls:
        # Preserve order while removing duplicates.
        unique_urls = list(dict.fromkeys(urls))
        print("\n=== URLs found ===")
        for url in unique_urls:
            print(url)

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
            print("\n=== Fetching summaries via r.jina.ai ===")

            import time
            import requests

            # NOTE: The free r.jina.ai endpoint allows ~15 requests/min (4s/request).
            JINA_TOKEN = "jina_9d5517f4235c47eeb6441889ab773ffd_s2uEm0MrfTXdoBC6nSdMBna66ZT"
            headers = {"Authorization": f"Bearer {JINA_TOKEN}"}

            for idx, original_url in enumerate(filtered_urls):
                proxy_url = f"https://r.jina.ai/{original_url}"
                try:
                    resp = requests.get(proxy_url, headers=headers, timeout=15)
                    resp.raise_for_status()
                    print(f"\n--- {original_url} ---")
                    print(resp.text)
                except Exception as err:  # pylint: disable=broad-except
                    print(f"❌ Failed to fetch '{original_url}': {err}")

                # Respect rate limit: sleep ~4 seconds between calls (<=15/minute)
                if idx < len(filtered_urls) - 1:
                    time.sleep(4.1)
        else:
            print("\n=== URLs found ===")
            print("No URLs detected in the model card.")
    else:
        print("\n=== URLs found ===")
        print("No URLs detected in the model card.")


if __name__ == "__main__":
    main() 