"""Rerank step: when the cross-encoder runs, the shortlist and the exhaustive flag."""

import asyncio

from backend.app.services.agent import rerank
from backend.app.services.agent.models import SearchPlan
from tests.conftest import hits, record

PLAN = SearchPlan(query="q")
MANY = hits([record(i, f"Book {i}", download_count=float(i)) for i in range(30)])


def run_rerank(state):
    return asyncio.run(rerank.rerank_node({"plan": PLAN, **state}))


def test_reranks_only_relaxed_results_with_more_than_the_answer_takes(monkeypatch):
    sent = []

    def reversed_order(query, books):
        sent.append((query, len(books)))
        return list(reversed(books))

    monkeypatch.setattr(rerank, "cross_encoder_order", reversed_order)
    state = run_rerank({"books": MANY, "relaxed_filters": ["author"]})
    assert (
        sent == [("q", 30)] and len(state["books"]) == 10 and state["books"][0].title == "Book 29"
    )

    sent.clear()
    state = run_rerank({"books": MANY[:15]})
    assert not sent and [b.title for b in state["books"]] == [f"Book {i}" for i in range(10)]
    state = run_rerank({"books": MANY[:6], "relaxed_filters": ["author"]})
    assert not sent and len(state["books"]) == 6


def test_popularity_orders_the_relevance_shortlist(monkeypatch):
    monkeypatch.setattr(rerank, "cross_encoder_order", lambda query, books: list(reversed(books)))
    state = asyncio.run(
        rerank.rerank_node(
            {
                "plan": SearchPlan(query="q", prefer_popular=True),
                "books": MANY,
                "relaxed_filters": ["x"],
            }
        )
    )
    top_twenty = [b.download_count for b in reversed(MANY)][:20]
    assert [b.download_count for b in state["books"]] == sorted(top_twenty, reverse=True)[:10]


def test_rerank_failure_keeps_the_search_order(monkeypatch):
    def failing(query, books):
        raise RuntimeError("rerank down")

    monkeypatch.setattr(rerank, "cross_encoder_order", failing)
    state = run_rerank({"books": MANY, "relaxed_filters": ["author"]})
    assert [b.title for b in state["books"]] == [f"Book {i}" for i in range(10)]


def test_exhaustive_survives_only_when_nothing_is_cut():
    assert run_rerank({"books": MANY[:6], "exhaustive": True})["exhaustive"] is True
    state = run_rerank({"books": MANY[:15], "exhaustive": True})
    assert state["exhaustive"] is False and len(state["books"]) == 10
