"""Rerank node: shortlists the candidates for the answer.

Vector order is kept unless constraints were dropped and the pool is broad: then the raw
order is least reliable, and a cross-encoder reorders it first.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from pinecone import Pinecone

from backend.app.config import get_settings
from backend.app.services.agent.models import AgentState, BookHit

logger = logging.getLogger(__name__)

# How many books the answer weighs.
ANSWER_CANDIDATES = 10
RERANK_DOCUMENT_CHARS = 1500


@lru_cache
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=get_settings().pinecone_api_key)


def get_rerank_model() -> str:
    return os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0")


async def rerank_node(state: AgentState) -> AgentState:
    plan, candidates = state["plan"], state["books"]
    books = candidates
    if state.get("relaxed_filters") and len(books) > ANSWER_CANDIDATES:
        try:
            books = await asyncio.to_thread(cross_encoder_order, plan.query, books)
        except Exception:
            logger.exception("Rerank failed; keeping the search order")
    if plan.prefer_popular:
        # Relevance picks the shortlist; popularity orders it.
        books = sorted(books[: ANSWER_CANDIDATES * 2], key=lambda b: b.download_count, reverse=True)
    shortlist = books[:ANSWER_CANDIDATES]
    return {
        "books": shortlist,
        # Cutting the list means the answer no longer sees every matching book.
        "exhaustive": state.get("exhaustive", False) and len(shortlist) == len(candidates),
    }


def cross_encoder_order(query: str, books: list[BookHit]) -> list[BookHit]:
    """Books ordered by a cross-encoder's relevance to the query, most relevant first."""
    result = get_pinecone_client().inference.rerank(
        model=get_rerank_model(),
        query=query,
        documents=[{"text": f"{b.title}\n{b.summary[:RERANK_DOCUMENT_CHARS]}"} for b in books],
        rank_fields=["text"],
        top_n=len(books),
        return_documents=False,
    )
    logger.info("Reranked %d books with %s", len(books), get_rerank_model())
    return [books[ranked.index] for ranked in result.data]
