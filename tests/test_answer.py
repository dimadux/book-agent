"""Answer step: the streaming renderer, grounding checks, fixed notices and failure paths."""

import asyncio
import logging

import jiter

from backend.app.services.agent import answer
from backend.app.services.agent.answer import Answer, AnswerRenderer, is_grounded, render_heading
from backend.app.services.agent.plan import Decision
from tests.conftest import GOTHIC, TRAVEL, book_request, hits, pick, record, reply

BOOKS = hits(
    [
        record(1, "Dracula", ("Stoker, Bram",)),
        record(2, "Carmilla <b>", ("Le Fanu, J.",)),
        record(3, "Frankenstein"),
    ]
)
DRACULA, CARMILLA, FRANKENSTEIN = BOOKS
GOOD_EVIDENCE = "a scientist creates life"


def render(result: Answer, books=BOOKS, *, intro=None, closing=True, chunk=3):
    """Feeds the answer JSON to the renderer in small pieces, like the real stream."""
    writes: list[str] = []
    renderer = AnswerRenderer(writes.append, books, intro=intro, closing=closing)
    text, buffer = result.model_dump_json(), ""
    for start in range(0, len(text), chunk):
        buffer += text[start : start + chunk]
        renderer.update(jiter.from_json(buffer.encode(), partial_mode="trailing-strings"))
    return writes, renderer


PICKS = reply(
    intro="Dark *gothic* picks:",
    picks=[
        pick("gutenberg:2", "A vampire tale."),
        pick("gutenberg:999", "made up"),
        pick("gutenberg:1", "The classic."),
    ],
    closing="Want more?",
)
EXPECTED = (
    "Dark \\*gothic\\* picks:\n\n"
    + render_heading(1, CARMILLA)
    + "A vampire tale.\n"
    + render_heading(2, DRACULA)
    + "The classic.\n\nWant more?"
)


def test_stream_renders_the_full_markdown(monkeypatch):
    monkeypatch.setattr(answer, "is_grounded", lambda evidence, book: True)
    writes, renderer = render(PICKS)
    out = "".join(writes)
    assert out == EXPECTED
    assert (
        writes[0] == "Dark \\*gothic\\* picks:\n\n" + render_heading(1, CARMILLA)
        and len(writes) > 15
    )
    assert "999" not in out and "<b>" not in out and "\\<b\\>" in out
    assert [b.id for b in renderer.picked.values()] == ["gutenberg:2", "gutenberg:1"]
    for chunk in (1, 2, 50, 10_000):
        assert "".join(render(PICKS, chunk=chunk)[0]) == EXPECTED


def test_fixed_intro_and_hidden_closing(monkeypatch):
    monkeypatch.setattr(answer, "is_grounded", lambda evidence, book: True)
    out = "".join(render(PICKS, intro="Notice.", closing=False)[0])
    assert out.startswith("Notice.\n\n1. **") and "gothic" not in out and "Want more" not in out
    out = "".join(render(reply(intro="", picks=[pick("gutenberg:1")]))[0])
    assert out.startswith(answer.FALLBACK_INTRO + "\n\n1. ")


def test_at_most_max_picks(monkeypatch):
    monkeypatch.setattr(answer, "is_grounded", lambda evidence, book: True)
    books = hits([record(i, f"T{i}") for i in range(8)])
    writes, renderer = render(reply(picks=[pick(b.id) for b in books]), books)
    assert (
        len(renderer.picked) == answer.MAX_PICKS
        and "".join(writes).count("**[") == answer.MAX_PICKS
    )


def test_nothing_fits_streams_the_intro_and_ignores_picks():
    result = reply(
        has_matches=False,
        intro="The catalogue has nothing on quantum computing.",
        picks=[pick("gutenberg:3", "sneaky", evidence=GOOD_EVIDENCE)],
        closing="Try classic science.",
    )
    writes, _ = render(result)
    assert (
        "".join(writes) == "The catalogue has nothing on quantum computing.\n\nTry classic science."
        and len(writes) > 5
    )


def test_intro_is_held_until_a_pick_passes():
    writes: list[str] = []
    renderer = AnswerRenderer(writes.append, BOOKS, intro=None, closing=True)
    renderer.update(
        jiter.from_json(
            b'{"has_matches": true, "intro": "Here are some', partial_mode="trailing-strings"
        )
    )
    assert writes == [] and not renderer.wrote_anything
    writes, renderer = render(
        reply(
            intro="Here are some great books:",
            picks=[pick("gutenberg:1", fit="weak", evidence=GOOD_EVIDENCE)],
        )
    )
    assert writes == [] and not renderer.wrote_anything


def test_weak_and_ungrounded_picks_are_dropped():
    result = reply(
        intro="Gothic picks:",
        picks=[
            pick("gutenberg:1", "weak one", fit="weak", evidence=GOOD_EVIDENCE),
            pick("gutenberg:2", "invented", fit="strong", evidence="dog characters on a farm"),
            pick("gutenberg:3", "real", fit="partial", evidence="Scientist   CREATES life!"),
        ],
        closing="More?",
    )
    writes, renderer = render(result)
    out = "".join(writes)
    assert "weak one" not in out and "invented" not in out
    assert [b.id for b in renderer.picked.values()] == ["gutenberg:3"] and out.startswith(
        "Gothic picks:\n\n1. **[Frankenstein]"
    )


def test_evidence_check():
    assert is_grounded("creates life", FRANKENSTEIN) and is_grounded("It goes badly", FRANKENSTEIN)
    assert not is_grounded("scientist", FRANKENSTEIN)  # needs two content words
    assert not is_grounded("horror", FRANKENSTEIN) and not is_grounded(None, FRANKENSTEIN)
    assert is_grounded(
        "rafts down the Mississippi ... runaway slave",
        hits([record(5, "Huck", summary="A boy rafts down the Mississippi with a runaway slave.")])[
            0
        ],
    )


def test_only_retrieved_books_are_shown_once_and_drops_are_logged(caplog):
    result = reply(
        intro="Picks:",
        picks=[
            pick("gutenberg:1", "real", evidence=GOOD_EVIDENCE),
            pick("gutenberg:12345", "INVENTED"),
            pick("gutenberg:1", "REPEATED"),
            pick("", "EMPTY"),
        ],
    )
    with caplog.at_level(logging.WARNING, logger=answer.logger.name):
        writes, renderer = render(result)
    out = "".join(writes)
    assert [b.id for b in renderer.picked.values()] == ["gutenberg:1"] and out.count("**[") == 1
    assert not any(word in out for word in ("INVENTED", "REPEATED", "EMPTY"))
    assert sum("not a candidate" in r.message for r in caplog.records) == 3


def run_answer(agent, state):
    writes = agent.node_writes()
    asyncio.run(answer.answer_node({"messages": [], **state}))
    return "".join(writes)


def test_no_candidates(agent):
    assert run_answer(agent, {"books": []}) == answer.NO_RESULTS_REPLY


def test_llm_failure_falls_back_to_the_search_results(agent):
    agent.llm({Answer: RuntimeError("llm down")})
    out = run_answer(agent, {"books": hits(GOTHIC)})
    assert (
        out.startswith(answer.FALLBACK_INTRO)
        and "A scientist creates life." in out
        and "It goes badly" not in out
    )
    agent.llm({Answer: RuntimeError("llm down")})
    out = run_answer(agent, {"books": hits(TRAVEL), "relaxed_filters": ["author"]})
    assert (
        out.startswith(answer.RELAXED_NOTICE.format(filters="author") + "\n\n1. ")
        and answer.FALLBACK_INTRO not in out
    )


def test_stream_cut_midway(agent):
    cut = reply(
        intro="An intro that is long enough to arrive first.",
        picks=[
            pick("gutenberg:4", "A long reason that is still streaming when the connection drops.")
        ],
    )
    fake = agent.llm({Answer: cut})
    fake.stream_fail_after = cut.model_dump_json().index("still streaming")
    out = run_answer(agent, {"books": hits(GOTHIC)})
    assert (
        out.startswith("An intro")
        and "Carmilla" in out
        and "still streaming" not in out
        and answer.FALLBACK_INTRO not in out
    )
    fake = agent.llm({Answer: cut})
    fake.stream_fail_after = 60  # inside the intro: nothing was shown yet
    assert run_answer(agent, {"books": hits(GOTHIC)}).startswith(answer.FALLBACK_INTRO)


def test_relaxed_notice_replaces_the_model_intro(agent):
    duplicate = "I couldn't find books by Twian in the catalogue, but here are some."
    fake = agent.llm({Answer: reply(intro=duplicate, picks=[pick("gutenberg:11", "Travel.")])})
    out = run_answer(agent, {"books": hits(TRAVEL), "relaxed_filters": ["author"]})
    assert duplicate not in out and out.count("couldn't find books matching") == 1
    assert out.startswith(answer.RELAXED_NOTICE.format(filters="author") + "\n\n1. ")
    assert "leave intro empty" in fake.prompt(Answer)
    agent.llm({Answer: reply(intro=duplicate, picks=[])})
    assert (
        run_answer(agent, {"books": hits(TRAVEL), "relaxed_filters": ["author"]})
        == answer.NO_RESULTS_REPLY
    )


def test_exhaustive_results(agent):
    books = hits(TRAVEL[:1])
    fake = agent.llm({Answer: reply(picks=[pick("gutenberg:10")], closing="Want more Twain?")})
    out = run_answer(agent, {"books": books, "exhaustive": True})
    assert out.rstrip().endswith(answer.EXHAUSTIVE_NOTICE) and "Want more" not in out
    assert answer.EXHAUSTIVE_INSTRUCTIONS in fake.prompt(Answer)
    agent.llm({Answer: reply(picks=[pick("gutenberg:10")], closing="Want more Twain?")})
    out = run_answer(agent, {"books": hits(TRAVEL[:2]), "exhaustive": True})
    assert (
        answer.EXHAUSTIVE_NOTICE not in out and "Want more" not in out
    )  # some left out: no claim either way
    fake = agent.llm({Answer: reply(picks=[pick("gutenberg:10")], closing="Want more?")})
    out = run_answer(agent, {"books": books})
    assert out.rstrip().endswith(
        "Want more?"
    ) and answer.EXHAUSTIVE_INSTRUCTIONS not in fake.prompt(Answer)


def test_all_picks_rejected_gives_the_no_results_reply(agent, monkeypatch):
    monkeypatch.setattr(answer, "is_grounded", lambda evidence, book: False)
    agent.books(GOTHIC)
    agent.llm(
        {
            Decision: book_request(query="q"),
            Answer: reply(intro="Here are some!", picks=[pick("gutenberg:4")], closing="c"),
        }
    )
    assert agent.run("x") == answer.NO_RESULTS_REPLY
