from __future__ import annotations

from typing import Literal, TypedDict

from backend.app.schemas import ChatMessage
from backend.app.services.agent.models import BookHit, SearchPlan

Intent = Literal["book_request", "needs_clarification", "off_topic", "malicious"]


class AgentState(TypedDict, total=False):
    messages: list[ChatMessage]
    # "rejected" is set by deterministic checks and failures, not by the classifier.
    intent: Intent | Literal["rejected"]
    # Final text for turns that end without an answer from the catalogue.
    reply: str
    plan: SearchPlan
    # True when filter extraction failed and the plan is just the raw message, unfiltered.
    # Also makes the rerank step reorder the results.
    plan_degraded: bool
    books: list[BookHit]
    # True when `books` holds every catalogue book matching the request's filters, so the
    # answer must not offer to find more of the same.
    books_exhaustive: bool
    # True when the search matched books but all were recommended earlier in the conversation.
    already_shown: bool
    # Human-readable names of filters the books do not meet, because no book met them all.
    relaxed_filters: list[str]
    # True when the strict search found nothing and the books come from a fallback search,
    # so they miss some of the user's constraints. Makes the rerank step reorder them.
    search_degraded: bool
