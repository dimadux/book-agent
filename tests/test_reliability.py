"""Timeouts, the model fallback and the guard that keeps every turn's stream complete."""

import asyncio
import time

from backend.app.services import book_agent
from backend.app.services.agent import answer, llm, search
from backend.app.services.agent.answer import Answer
from backend.app.services.agent.plan import Decision
from backend.app.services.agent.similar import SimilarityProfile
from tests.conftest import GOTHIC, LIBRARY, book_request, message, pick, reply


def test_structured_calls_are_deterministic_and_try_the_fallback_model(agent, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "primary")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "backup")
    fake = agent.llm({Decision: book_request(query="q")})
    fake.failing_models = {"primary"}
    decision = asyncio.run(llm.structured("i", [message("user", "x")], Decision))
    assert decision.intent == "book_request"
    assert [kwargs["model"] for _, kwargs in fake.calls] == ["primary", "backup"]
    assert all(kwargs["temperature"] == 0 for _, kwargs in fake.calls)


def test_fallback_is_not_used_for_a_refusal_or_when_disabled(agent, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "primary")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "backup")
    fake = agent.llm({Decision: None})  # the model returned nothing: not a transport failure
    assert asyncio.run(llm.structured("i", [], Decision)) is None
    assert [kwargs["model"] for _, kwargs in fake.calls] == ["primary"]

    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "")
    fake = agent.llm({Decision: book_request(query="q")})
    fake.failing_models = {"primary"}
    assert asyncio.run(llm.structured("i", [], Decision)) is None
    assert [kwargs["model"] for _, kwargs in fake.calls] == ["primary"]
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "primary")
    assert llm.llm_models() == ["primary"]


def test_a_slow_turn_ends_with_the_timeout_reply(agent, monkeypatch):
    monkeypatch.setattr(book_agent, "TURN_TIMEOUT_SECONDS", 0.2)
    fake = agent.llm({Decision: book_request(query="q")})
    fake.delay = 5
    started = time.perf_counter()
    assert agent.run("x") == book_agent.TIMEOUT_REPLY
    assert time.perf_counter() - started < 1


def test_timeout_after_partial_output_keeps_what_was_shown(agent, monkeypatch):
    monkeypatch.setattr(book_agent, "TURN_TIMEOUT_SECONDS", 0.5)
    agent.books(LIBRARY)
    profile = SimilarityProfile(
        query="river adventures", note="Twain is known for river adventures."
    )
    fake = agent.llm(
        {
            Decision: book_request(query="q", similar_to_authors=["Twain, Mark"]),
            SimilarityProfile: profile,
            Answer: reply(picks=[pick("gutenberg:72")]),
        }
    )
    original_stream = fake.stream

    def slow_answer_stream(*, text_format, **kwargs):
        fake.delay = 5  # the note is out; the answer then stalls
        return original_stream(text_format=text_format, **kwargs)

    fake.stream = slow_answer_stream
    out = agent.run("like Twain")
    assert out.startswith("*Twain is known for river adventures.*") and out.endswith(
        f"\n\n{book_agent.TIMEOUT_REPLY}"
    )


def test_an_unexpected_error_ends_with_the_error_reply(agent, monkeypatch):
    def broken(position, book):
        raise ValueError("bug")

    monkeypatch.setattr(answer, "render_heading", broken)
    agent.books(GOTHIC)
    agent.llm({Decision: book_request(query="q"), Answer: reply(picks=[pick("gutenberg:4")])})
    out = agent.run("x")
    # The fallback list had started (its intro is out) when the bug hit; the turn ends cleanly.
    assert out.startswith(answer.FALLBACK_INTRO) and out.endswith(f"\n\n{book_agent.ERROR_REPLY}")


def test_a_stuck_catalogue_call_gives_the_catalogue_reply(agent, monkeypatch):
    monkeypatch.setattr(search, "SEARCH_TIMEOUT_SECONDS", 0.05)

    def stuck(query, filters, top_k):
        time.sleep(0.3)
        return []

    agent.books(stuck)
    agent.llm({Decision: book_request(query="q")})
    # An empty result would give the no-results reply; the catalogue reply proves the deadline.
    assert agent.run("x") == search.SEARCH_ERROR_REPLY
