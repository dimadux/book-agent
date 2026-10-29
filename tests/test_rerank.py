"""Rerank step: when the cross-encoder runs, the shortlist and the exhaustive flag."""

import asyncio

from backend.app.services.agent import rerank
from backend.app.services.agent.models import SearchPlan
from tests.conftest import hits, message, record

PLAN = SearchPlan(query="q")
MANY = hits([record(i, f"Book {i}", download_count=float(i)) for i in range(30)])


def run_rerank(state):
    return asyncio.run(
        rerank.rerank_node({"plan": PLAN, "messages": [message("user", "q")], **state})
    )


def test_reranks_whenever_there_are_more_books_than_the_answer_takes(monkeypatch):
    sent = []

    def reversed_order(query, books):
        sent.append((query, len(books)))
        return list(reversed(books))

    monkeypatch.setattr(rerank, "cross_encoder_order", reversed_order)
    state = run_rerank({"books": MANY, "relaxed_filters": ["author"]})
    assert (
        sent == [("q", 30)] and len(state["books"]) == 16 and state["books"][0].title == "Book 29"
    )

    sent.clear()
    state = run_rerank({"books": MANY[:15]})
    assert not sent and [b.title for b in state["books"]] == [f"Book {i}" for i in range(15)]
    state = run_rerank({"books": MANY[:6], "relaxed_filters": ["author"]})
    assert not sent and len(state["books"]) == 6
    state = run_rerank({"books": MANY[:17]})  # strict searches rerank too
    assert sent == [("q", 17)] and len(state["books"]) == 16


def test_reranker_query_is_the_users_words_unless_they_do_not_describe_the_books():
    assert (
        rerank.rerank_query(SearchPlan(query="rewritten"), "funny dog books") == "funny dog books"
    )
    assert (
        rerank.rerank_query(SearchPlan(query="rewritten", follows_up=True), "shorter")
        == "rewritten"
    )
    profile = SearchPlan(query="profile", similar_to_titles=["Emma"])
    assert rerank.rerank_query(profile, "books like Emma") == "profile"
    assert len(rerank.rerank_query(SearchPlan(query="q"), "x" * 2000)) == rerank.RERANK_QUERY_CHARS


def test_popularity_orders_the_relevance_shortlist(monkeypatch):
    monkeypatch.setattr(rerank, "cross_encoder_order", lambda query, books: list(reversed(books)))
    state = asyncio.run(
        rerank.rerank_node(
            {
                "plan": SearchPlan(query="q", prefer_popular=True),
                "messages": [message("user", "q")],
                "books": MANY,
                "relaxed_filters": ["x"],
            }
        )
    )
    # The 16 most relevant (Book 29 down to Book 14), ordered by downloads: the same books.
    assert [b.title for b in state["books"]] == [f"Book {i}" for i in range(29, 13, -1)]


def test_popularity_never_replaces_a_relevant_book(monkeypatch):
    monkeypatch.setattr(rerank, "cross_encoder_order", lambda query, books: books)
    state = asyncio.run(
        rerank.rerank_node(
            {
                "plan": SearchPlan(query="q", prefer_popular=True),
                "messages": [message("user", "q")],
                "books": MANY,  # relevance order Book 0..29, downloads 0..29
            }
        )
    )
    # Book 29 is the most downloaded but not among the 16 most relevant, so it is not shown.
    assert [b.title for b in state["books"]] == [f"Book {i}" for i in range(15, -1, -1)]


def test_empty_rerank_model_disables_reranking(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rerank, "cross_encoder_order", lambda query, books: calls.append(query) or books
    )
    monkeypatch.setenv("PINECONE_RERANK_MODEL", "")
    state = run_rerank({"books": MANY})
    assert not calls and [b.title for b in state["books"]] == [f"Book {i}" for i in range(16)]


def test_rerank_failure_keeps_the_search_order(monkeypatch):
    def failing(query, books):
        raise RuntimeError("rerank down")

    monkeypatch.setattr(rerank, "cross_encoder_order", failing)
    state = run_rerank({"books": MANY, "relaxed_filters": ["author"]})
    assert [b.title for b in state["books"]] == [f"Book {i}" for i in range(16)]


def test_exhaustive_survives_only_when_nothing_is_cut():
    assert run_rerank({"books": MANY[:6], "exhaustive": True})["exhaustive"] is True
    state = run_rerank({"books": MANY[:20], "exhaustive": True})
    assert state["exhaustive"] is False and len(state["books"]) == 16
