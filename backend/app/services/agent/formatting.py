"""Markdown helpers for text streamed to the chat UI, which renders it as HTML."""

from __future__ import annotations

import re

MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]<>#|])")


def escape_markdown(text: str) -> str:
    return MARKDOWN_SPECIALS.sub(r"\\\1", " ".join(text.split()))


def escape_markdown_chunk(text: str) -> str:
    """Like escape_markdown, for one piece of streamed text: keeps spaces at the edges."""
    return MARKDOWN_SPECIALS.sub(r"\\\1", re.sub(r"\s+", " ", text))


def display_author(name: str) -> str:
    # Gutenberg stores "Last, First"; show "First Last" when the split is unambiguous.
    parts = [part.strip() for part in name.split(",")]
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 and all(parts) else name
