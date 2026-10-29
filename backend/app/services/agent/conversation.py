"""Helpers over the chat history, which the client sends in full with every request."""

from __future__ import annotations

import re

from backend.app.schemas import ChatMessage

# Longer messages are rejected; it is also the per-message cap on what any LLM call sees.
MAX_MESSAGE_CHARS = 2000
# Recent user/assistant turns each LLM call sees: enough for follow-ups, small prompts.
HISTORY_WINDOW = 6
# The answer links every recommended book to its Gutenberg page.
BOOK_LINK = re.compile(r"https?://(?:www\.)?gutenberg\.org/ebooks/(\d+)")


def last_user_message(messages: list[ChatMessage]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")


def recent_conversation(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """The last HISTORY_WINDOW user/assistant turns, in the Responses API input format."""
    # Clients can send role="system"; never forward those, they could override our rules.
    turns = [m for m in messages if m.role in ("user", "assistant")]
    return [
        {"role": m.role, "content": m.content[:MAX_MESSAGE_CHARS]} for m in turns[-HISTORY_WINDOW:]
    ]


def recommended_book_ids(messages: list[ChatMessage]) -> frozenset[str]:
    """Ids of books linked in earlier assistant replies, i.e. already recommended."""
    return frozenset(
        f"gutenberg:{number}"
        for m in messages
        if m.role == "assistant"
        for number in BOOK_LINK.findall(m.content)
    )
