"""Rerank node: shortlists the search candidates for the answer.

Search results are ordered by vector similarity. When they are degraded (the plan or the
search fell back) and there are more than the answer can weigh, a cross-encoder reorders
them first. Otherwise the vector order is kept.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from pinecone import Pinecone

from backend.app.config import get_settings
from backend.app.services.agent.models import BookHit, SearchPlan
from backend.app.services.agent.state import AgentState

logger = logging.getLogger(__name__)

# The answer weighs at most this many books; above it, degraded results are reranked.
ANSWER_CANDIDATES = 10
MAX_RERANK_DOCUMENTS = 30
RERANK_DOCUMENT_CHARS = 1500


@lru_cache
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=get_settings().pinecone_api_key)


def get_rerank_model() -> str:
    return os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0")


def needs_rerank(state: AgentState) -> bool:
    degraded = state.get("plan_degraded") or state.get("search_degraded")
    return bool(degraded) and len(state.get("books") or []) > ANSWER_CANDIDATES


async def rerank_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    books = state.get("books") or []
    ordered = books
    if needs_rerank(state):
        try:
            ordered = await asyncio.to_thread(cross_encoder_order, plan.query, books[:MAX_RERANK_DOCUMENTS])
        except Exception:
            logger.exception("Rerank failed; keeping the search order")
    shortlisted = shortlist(ordered, plan)
    return {
        "books": shortlisted,
        # Cutting the list means the answer no longer sees every matching book.
        "books_exhaustive": bool(state.get("books_exhaustive")) and len(shortlisted) == len(books),
    }


def cross_encoder_order(query: str, books: list[BookHit]) -> list[BookHit]:
    """Books ordered by a cross-encoder's relevance to the query, most relevant first."""
    result = get_pinecone_client().inference.rerank(
        model=get_rerank_model(),
        query=query,
        documents=[{"text": f"{book.title}\n{book.summary[:RERANK_DOCUMENT_CHARS]}"} for book in books],
        rank_fields=["text"],
        top_n=len(books),
        return_documents=False,
    )
    logger.info("Reranked %d books with %s", len(books), get_rerank_model())
    return [books[ranked.index] for ranked in result.data]


def shortlist(books: list[BookHit], plan: SearchPlan) -> list[BookHit]:
    if plan.prefer_popular:
        # Relevance picks the shortlist; popularity orders it.
        books = sorted(books[: ANSWER_CANDIDATES * 2], key=lambda book: book.download_count, reverse=True)
    return books[:ANSWER_CANDIDATES]
