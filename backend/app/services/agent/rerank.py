"""Rerank node: shortlists the candidates for the answer.

Whenever the search returned more than the answer weighs, a cross-encoder reorders the pool
first, because the cut decides what the user gets and vector order is the least reliable
order there is. A pool the answer can take whole, or a failed rerank call, keeps the search
order.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from pinecone import Pinecone, RetryConfig

from backend.app.config import get_settings
from backend.app.services.agent.conversation import last_user_message
from backend.app.services.agent.models import AgentState, BookHit, SearchPlan

logger = logging.getLogger(__name__)

# How many books the answer weighs. Also the rerank threshold.
ANSWER_CANDIDATES = 16
# pinecone-rerank-v0 rejects the whole request when one query+document pair exceeds 512
# tokens; 1000 characters of English stays under that with the query.
RERANK_DOCUMENT_CHARS = 1000
RERANK_QUERY_CHARS = 300
# One rerank call, no SDK retries: a 429 here is a per-minute or monthly quota, which does not
# clear within a request, and the SDK's default of 3 retries with backoff cost 6 s per failure.
RERANK_TIMEOUT_SECONDS = 5


@lru_cache
def get_pinecone_client() -> Pinecone:
    return Pinecone(
        api_key=get_settings().pinecone_api_key,
        timeout=RERANK_TIMEOUT_SECONDS,
        retry_config=RetryConfig(max_retries=0),
    )


def get_rerank_model() -> str:
    """The cross-encoder model; empty disables reranking (e.g. while there is no quota)."""
    return os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0")


async def rerank_node(state: AgentState) -> AgentState:
    plan, candidates = state["plan"], state["books"]
    books = candidates
    if len(books) > ANSWER_CANDIDATES and get_rerank_model():
        try:
            query = rerank_query(plan, last_user_message(state["messages"]))
            books = await asyncio.to_thread(cross_encoder_order, query, books)
        except Exception:  # rate limit, quota, outage: the vector order is still a ranking
            logger.exception("Rerank failed; keeping the search order")
    shortlist = books[:ANSWER_CANDIDATES]
    if plan.prefer_popular:
        # Relevance picks the shortlist; popularity only orders it, so a well-known book that is
        # off topic can never push a matching one out.
        shortlist = sorted(shortlist, key=lambda b: b.download_count, reverse=True)
    return {
        "books": shortlist,
        # Cutting the list means the answer no longer sees every matching book.
        "exhaustive": state.get("exhaustive", False) and len(shortlist) == len(candidates),
    }


def rerank_query(plan: SearchPlan, user_message: str) -> str:
    """Cross-encoders are trained on short natural queries, and measured on the evaluation
    set the user's own words rank better than the plan's rewritten query. The plan's query is
    used when the words alone do not describe the books wanted: a follow-up ("shorter ones")
    or a "books like X" request, whose query is the similarity profile."""
    if plan.follows_up or plan.similar_to_authors or plan.similar_to_titles:
        return plan.query
    return user_message[:RERANK_QUERY_CHARS] or plan.query


def cross_encoder_order(query: str, books: list[BookHit]) -> list[BookHit]:
    """Books ordered by a cross-encoder's relevance to the query, most relevant first."""
    result = get_pinecone_client().inference.rerank(
        model=get_rerank_model(),
        query=query,
        documents=[{"text": rerank_document(b)} for b in books],
        rank_fields=["text"],
        top_n=len(books),
        return_documents=False,
    )
    logger.info("Reranked %d books with %s", len(books), get_rerank_model())
    return [books[ranked.index] for ranked in result.data]


def rerank_document(book: BookHit) -> str:
    """What the cross-encoder reads: authors and subjects help it on "by X" and genre requests."""
    text = (
        f"{book.title}\n{', '.join(book.authors)}\n{'; '.join(book.subjects[:8])}\n{book.summary}"
    )
    return text[:RERANK_DOCUMENT_CHARS]
