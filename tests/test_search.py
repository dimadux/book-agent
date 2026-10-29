"""Search step: constraints, relaxation, nothing-new, exhaustive flag and catalogue helpers."""

import asyncio
import time

from backend.app.services.agent import search
from backend.app.services.agent.models import SearchPlan
from backend.app.services.agent.search import (
    plan_constraints,
    same_author,
    same_title,
    unique_editions,
)
from tests.conftest import TRAVEL, hits, message, record


def run_search(agent, plan: SearchPlan, messages=(), records=TRAVEL):
    agent.books(records)
    return asyncio.run(search.search_node({"messages": list(messages), "plan": plan}))


def titles(state):
    return [book.title for book in state["books"]]


def test_constraints_least_important_first():
    plan = SearchPlan(
        query="q",
        authors=["Twain, Mark"],
        languages=["fr"],
        published_before=1800,
        published_after=1700,
    )
    names = [name for name, _ in plan_constraints(plan)]
    assert names == ["publication years", "language", "author"]
    years = dict(plan_constraints(plan))["publication years"]
    assert set(years) == {"$and"} and all("$or" in part for part in years["$and"])


def test_strict_hit_is_one_call(agent):
    state = run_search(agent, SearchPlan(query="travel", authors=["Twain, Mark"]))
    assert len(agent.catalogue.calls) == 1
    assert titles(state) == ["Roughing It"] and state["relaxed_filters"] == []


def test_wrong_author_relaxes_only_the_author(agent):
    state = run_search(
        agent, SearchPlan(query="travel", authors=["Twian, Mark"], published_before=1800)
    )
    assert titles(state) == ["Humphry Clinker"] and state["relaxed_filters"] == ["author"]
    # The strict search, the name lookup (no catalogue spelling is close enough), then the
    # smaller sets at once: {author}, {years}, {}. The concurrent calls land in any order; once
    # the author is not filtered on, the name moves into the text.
    assert len(agent.catalogue.calls) == 5 and agent.catalogue.calls[1][:2] == ("Twian, Mark", None)
    queries = {query for query, _, _ in agent.catalogue.calls}
    assert "travel by Twian, Mark" in queries


def test_author_spelt_differently_is_resolved_before_relaxing(agent):
    state = run_search(agent, SearchPlan(query="travel", authors=["Mark Twain"]))
    assert titles(state) == ["Roughing It"] and state["relaxed_filters"] == []
    assert state["plan"].authors == ["Twain, Mark"]
    # Strict search, name lookup, strict search again with the catalogue's spelling.
    assert [filters for _, filters, _ in agent.catalogue.calls] == [
        {"authors": {"$in": ["Mark Twain"]}},
        None,
        {"authors": {"$in": ["Twain, Mark"]}},
    ]


def test_drops_as_few_constraints_as_possible(agent):
    state = run_search(
        agent,
        SearchPlan(
            query="travel", authors=["Twain, Mark"], languages=["zz"], published_before=1950
        ),
    )
    assert titles(state) == ["Roughing It"] and state["relaxed_filters"] == ["language"]
    state = run_search(
        agent,
        SearchPlan(query="travel", authors=["Nobody, X"], languages=["fr"], published_after=2090),
    )
    assert titles(state) == ["Voyage"] and state["relaxed_filters"] == [
        "publication years",
        "author",
    ]


def test_on_a_tie_the_author_is_kept(agent):
    state = run_search(
        agent,
        SearchPlan(
            query="travel", authors=["Twain, Mark"], languages=["fr"], published_before=1950
        ),
    )
    assert titles(state) == ["Roughing It"] and state["relaxed_filters"] == ["language"]


def test_nothing_matches_anything(agent):
    state = run_search(
        agent,
        SearchPlan(query="travel", authors=["Nobody, X"], languages=["zz"], published_after=2090),
    )
    assert len(state["books"]) == 3 and state["relaxed_filters"] == [
        "publication years",
        "language",
        "author",
    ]
    # Strict, the author lookup, then the 7 smaller sets.
    assert len(agent.catalogue.calls) == 1 + 1 + 7 and None in agent.catalogue.filters


def test_single_constraint_and_no_constraints(agent):
    state = run_search(agent, SearchPlan(query="travel", languages=["zz"]))
    assert state["relaxed_filters"] == ["language"] and agent.catalogue.filters == [
        {"languages": {"$in": ["zz"]}},
        None,
    ]
    state = run_search(agent, SearchPlan(query="travel"))
    assert (
        len(agent.catalogue.calls) == 1
        and state["relaxed_filters"] == []
        and state["exhaustive"] is False
    )


def test_relaxed_searches_run_concurrently(agent):
    def slow_search(query, filters, top_k):
        time.sleep(0.2)
        return []

    started = time.perf_counter()
    run_search(
        agent,
        SearchPlan(query="q", authors=["X, Y"], languages=["zz"], published_after=2090),
        records=slow_search,
    )
    assert time.perf_counter() - started < 0.2 * 4  # two rounds, not eight sequential searches


def test_exhaustive_when_fewer_than_top_k_match(agent):
    assert (
        run_search(agent, SearchPlan(query="travel", authors=["Twain, Mark"]))["exhaustive"] is True
    )
    full = [record(i, f"B{i}", ("X, Y",)) for i in range(search.SEARCH_TOP_K)]
    assert (
        run_search(agent, SearchPlan(query="q", authors=["X, Y"]), records=full)["exhaustive"]
        is False
    )
    assert (
        run_search(
            agent, SearchPlan(query="travel", authors=["Twian, Mark"], published_before=1800)
        )["exhaustive"]
        is False
    )


def test_nothing_new_when_every_match_was_shown_or_excluded(agent):
    shown = [message("assistant", "1. **[Roughing It](https://www.gutenberg.org/ebooks/10)**")]
    state = run_search(agent, SearchPlan(query="travel", authors=["Twain, Mark"]), shown)
    assert state == {"reply": search.ALREADY_SHOWN_REPLY} and len(agent.catalogue.calls) == 1
    state = run_search(
        agent, SearchPlan(query="travel", authors=["Twain, Mark"], exclude_titles=["Roughing It"])
    )
    assert state == {"reply": search.ALREADY_SHOWN_REPLY} and len(agent.catalogue.calls) == 1


def test_shown_books_are_skipped_in_relaxed_searches_too(agent):
    shown = [message("assistant", "https://www.gutenberg.org/ebooks/11")]
    state = run_search(
        agent, SearchPlan(query="travel", authors=["Twian, Mark"], published_before=1950), shown
    )
    assert titles(state) == ["Roughing It", "Voyage"] and state["relaxed_filters"] == ["author"]


def test_pinecone_failure_gives_a_fixed_reply(agent):
    def boom(query, filters, top_k):
        raise RuntimeError("401")

    assert run_search(agent, SearchPlan(query="x"), records=boom) == {
        "reply": search.SEARCH_ERROR_REPLY
    }


def test_volumes_and_copies_collapse_to_one_record():
    shakespeare = ("Shakespeare, William",)
    books = unique_editions(
        hits(
            [
                record(1, "Works [Cambridge Edition] [Vol. 8 of 9]", shakespeare),
                record(2, "Complete Works", shakespeare),
                record(3, "Works [Cambridge Edition] [Vol. 5 of 9]", shakespeare),
                record(4, "Works [Cambridge Edition] [Vol. 8 of 9]", shakespeare),
                record(5, "Works [Cambridge Edition] [Vol. 2 of 9]", ("Other, A.",)),
            ]
        )
    )
    assert [b.id for b in books] == ["gutenberg:1", "gutenberg:2", "gutenberg:5"]


def test_title_and_author_matching():
    assert same_title("Frankenstein", "Frankenstein; Or, The Modern Prometheus")
    assert not same_title("Frankenstein", "Frankensteinia") and not same_title("", "X")
    assert same_author("Conan Doyle, Arthur", "Doyle, Arthur Conan")
    assert same_author("Shelley, Mary", "Shelley, Mary Wollstonecraft")
    assert same_author("Dostoevsky", "Dostoyevsky, Fyodor")
    assert same_author("H. G. Wells", "Wells, H. G. (Herbert George)")
    assert same_author("Bronte, Emily", "Brontë, Emily")
    assert not same_author("King, Stephen", "Kingsley, Charles") and not same_author("", "X")
    assert not same_author("Poe", "Pope, Alexander")  # short words must match exactly
    assert not same_author("Arthur", "Doyle, Arthur Conan")  # the surname must be given


def test_exclusions_by_title_and_author():
    plan = SearchPlan(query="q", exclude_titles=["Frankenstein"], exclude_authors=["Dickens"])
    frankenstein, bleak_house, emma = hits(
        [
            record(1, "Frankenstein; Or, The Modern Prometheus"),
            record(2, "Bleak House", ("Dickens, Charles",)),
            record(3, "Emma", ("Austen, Jane",)),
        ]
    )
    assert search.is_excluded(frankenstein, plan) and search.is_excluded(bleak_house, plan)
    assert not search.is_excluded(emma, plan)


def test_hit_parsing():
    (book,) = hits([record(7, "T", first_publish_year=1818.0, download_count=12.0)])
    assert (book.first_publish_year, book.download_count, book.summary) == (
        1818,
        12,
        "A scientist creates life. It goes badly.",
    )
    assert book.url == "https://www.gutenberg.org/ebooks/7"
    assert search.to_book_hit({"id": "x", "metadata": {}}) is None
