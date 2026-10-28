from __future__ import annotations

import re

from backend.app.schemas import ChatMessage

# Per-message cap on what any LLM call sees from the conversation.
MAX_MESSAGE_CHARS = 2000


def last_user_message(messages: list[ChatMessage]) -> str:
    return next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    )


def recent_conversation(messages: list[ChatMessage], window: int) -> list[dict[str, str]]:
    """Last `window` user/assistant turns, in the Responses API input format."""
    # Clients can send role="system"; never forward those, they could override our rules.
    turns = [message for message in messages if message.role in {"user", "assistant"}]
    return [
        {"role": message.role, "content": message.content[:MAX_MESSAGE_CHARS]}
        for message in turns[-window:]
    ]


# The answer links every recommended book to its Gutenberg page.
RECOMMENDED_BOOK_LINK = re.compile(r"https?://(?:www\.)?gutenberg\.org/ebooks/(\d+)")


def recommended_book_ids(messages: list[ChatMessage]) -> frozenset[str]:
    """Ids of books linked in earlier assistant replies, i.e. already recommended."""
    return frozenset(
        f"gutenberg:{number}"
        for message in messages
        if message.role == "assistant"
        for number in RECOMMENDED_BOOK_LINK.findall(message.content)
    )
