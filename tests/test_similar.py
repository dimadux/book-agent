"""Similar step: reference lookup, the streamed note, exclusions and its failure paths."""

import asyncio
import re
import time

from backend.app.services.agent import search, similar
from backend.app.services.agent.answer import Answer
from backend.app.services.agent.models import SearchPlan
from backend.app.services.agent.plan import Decision
from backend.app.services.agent.similar import SimilarityProfile
from tests.conftest import LIBRARY, book_request, message, pick, reply

PROFILE = SimilarityProfile(
    query="humorous river adventure, boyhood mischief",
    note="Mark Twain is known for *humorous* river adventures, so I'll look for that.",
)


def like(agent, plan: Decision, profile=PROFILE, chosen="gutenberg:72"):
    """A 'books like X' conversation over LIBRARY that recommends `chosen`."""
    agent.books(LIBRARY)
    return agent.llm(
        {Decision: plan, SimilarityProfile: profile, Answer: reply(picks=[pick(chosen)])}
    )


def run_similar(agent, plan: SearchPlan, text="x"):
    agent.node_writes()
    return asyncio.run(similar.similar_node({"plan": plan, "messages": [message("user", text)]}))


def test_like_an_author(agent):
    fake = like(
        agent, book_request(query="something like Twain", similar_to_authors=["Twain, Mark"])
    )
    out = agent.run("something like Mark Twain")
    assert out.startswith(
        "*Mark Twain is known for \\*humorous\\* river adventures, so I'll look for that.*\n\n"
    )
    assert any(query == PROFILE.query for query, _, _ in agent.catalogue.calls)
    prompt = fake.prompt(Answer)
    assert "Huckleberry" not in prompt and "Tom Sawyer" not in prompt and "Treasure Island" in out
    reference_prompt = fake.prompt(SimilarityProfile)
    assert (
        "Huckleberry" in reference_prompt
        and "Category: Adventure" in reference_prompt
        and '"not_in_catalogue": []' in reference_prompt
    )


def test_unknown_reference_asks_for_the_mood(agent):
    fake = like(agent, book_request(query="q", similar_to_authors=["King, Stephen"]))
    out = agent.run("something like Stephen King")
    assert out.startswith("I don't have information about Stephen King") and "mood or vibe" in out
    # The exact lookup, then the name lookup that finds no catalogue spelling either.
    assert agent.catalogue.filters == [{"authors": {"$in": ["King, Stephen"]}}, None]
    assert not fake.called(Answer)


def test_name_in_another_form_is_resolved(agent):
    agent.books(LIBRARY)
    state = run_similar(agent, SearchPlan(query="q", similar_to_authors=["Conan Doyle, Arthur"]))
    assert "reply" not in state
    assert any("Baskervilles" in title for title in state["plan"].exclude_titles)
    assert agent.catalogue.filters[-1] == {"authors": {"$in": ["Doyle, Arthur Conan"]}}


def test_partly_known_references_continue(agent):
    fake = like(agent, book_request(query="q", similar_to_authors=["Twain, Mark", "King, Stephen"]))
    out = agent.run("like Twain or Stephen King")
    assert (
        '"not_in_catalogue": ["Stephen King"]' in fake.prompt(SimilarityProfile)
        and "Treasure Island" in out
    )


def test_reference_title_found_by_text_and_excluded(agent):
    like(agent, book_request(query="q"))
    for title in ("Frankenstein", "Frankenstein; Or, The Modern Prometheus"):
        state = run_similar(agent, SearchPlan(query="q", similar_to_titles=[title]))
        assert state["plan"].exclude_titles == ["Frankenstein; Or, The Modern Prometheus"]
        assert state["plan"].exclude_authors == []  # like a title: its author is still fair game


def test_all_references_looked_up_at_once(agent):
    like(agent, book_request(query="q"))
    agent.catalogue.calls.clear()
    plan = SearchPlan(
        query="q",
        similar_to_authors=["Twain, Mark", "Doyle, Arthur Conan"],
        similar_to_titles=["Treasure Island"],
    )
    state = run_similar(agent, plan)
    assert sorted(map(str, agent.catalogue.filters)) == sorted(
        map(str, [{"authors": {"$in": ["Twain, Mark", "Doyle, Arthur Conan"]}}, None])
    )
    assert state["plan"].exclude_authors == ["Twain, Mark", "Doyle, Arthur Conan"]
    assert "Treasure Island" in state["plan"].exclude_titles


def test_lookups_run_concurrently(agent):
    def slow_search(query, filters, top_k):
        time.sleep(0.2)
        return LIBRARY

    agent.books(slow_search)
    agent.llm({SimilarityProfile: PROFILE})
    started = time.perf_counter()
    run_similar(
        agent,
        SearchPlan(
            query="q",
            similar_to_authors=["Twain, Mark"],
            similar_to_titles=["Treasure Island", "Frankenstein"],
        ),
    )
    assert time.perf_counter() - started < 0.5


def test_description_failure_keeps_the_plan_query(agent):
    for profile in (None, RuntimeError("down")):
        like(agent, book_request(query="q"), profile=profile)
        writes = agent.node_writes()
        state = asyncio.run(
            similar.similar_node(
                {
                    "plan": SearchPlan(query="witty tales", similar_to_authors=["Twain, Mark"]),
                    "messages": [],
                }
            )
        )
        assert (
            state["plan"].query == "witty tales"
            and state["plan"].exclude_authors == ["Twain, Mark"]
            and writes == []
        )


def test_lookup_failure_falls_through_to_the_search(agent):
    def boom(query, filters, top_k):
        raise RuntimeError("pinecone down")

    agent.books(boom)
    agent.llm({Decision: book_request(query="q", similar_to_authors=["Twain, Mark"])})
    assert agent.run("like Twain") == search.SEARCH_ERROR_REPLY


def test_unknown_name_is_escaped_in_the_reply(agent):
    like(agent, book_request(query="q", similar_to_authors=["<img src=x onerror=alert(1)>"]))
    out = agent.run("like <img>")
    assert not re.search(r"(?<!\\)[<>]", out) and r"\<img" in out


def test_no_references_skips_the_step(agent):
    fake = like(agent, book_request(query="pirates"))
    agent.run("pirate books")
    assert not fake.called(SimilarityProfile)
