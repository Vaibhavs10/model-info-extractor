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


if __name__ == "__main__":
    main() 