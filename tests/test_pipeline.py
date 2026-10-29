"""End to end through the graph, with the fakes: the happy path and the cross-step flows."""

from backend.app.services.agent import answer, rerank
from backend.app.services.agent.answer import Answer
from backend.app.services.agent.conversation import recommended_book_ids
from backend.app.services.agent.plan import Decision
from tests.conftest import GOTHIC, LIBRARY, TRAVEL, book_request, message, pick, record, reply


def test_happy_path(agent):
    agent.books(GOTHIC)
    fake = agent.llm(
        {
            Decision: book_request(
                query="gothic science moral dilemmas", exclude_titles=["Frankenstein"]
            ),
            Answer: reply(
                intro="Gothic picks:",
                picks=[
                    pick("gutenberg:999", "made up"),
                    pick("gutenberg:2", "A *vampire* tale."),
                    pick("gutenberg:4", "Classic."),
                ],
                closing="Want more?",
            ),
        }
    )
    out = agent.run("something like Frankenstein")
    assert "999" not in out and "made up" not in out  # hallucinated id dropped
    assert "Frankenstein" not in out  # reference title excluded
    assert "(https://www.gutenberg.org/ebooks/2)" in out and "John William Polidori" in out
    assert "<script>" not in out and "\\*vampire\\*" in out
    assert fake.prompt(Answer).count('"title": "The Vampyre') == 1  # duplicate copy dropped
    assert out.index("Vampyre") < out.index("Carmilla") and out.rstrip().endswith("Want more?")


def test_relaxed_notice_names_only_the_dropped_constraint(agent):
    agent.books(TRAVEL)
    agent.llm(
        {
            Decision: book_request(query="travel", authors=["Twian, Mark"], published_before=1800),
            Answer: reply(intro="Here:", picks=[pick("gutenberg:11", "Travel.")]),
        }
    )
    out = agent.run("Twain travel before 1800")
    assert "your author filter" in out and "publication years" not in out


def test_no_results_and_catalogue_error(agent):
    agent.llm({Decision: book_request(query="x")})
    assert agent.run("x") == answer.NO_RESULTS_REPLY

    def boom(query, filters, top_k):
        raise RuntimeError("401")

    agent.books(boom)
    assert "catalogue is unavailable" in agent.run("x")


def test_reranked_shortlist_reaches_the_answer(agent, monkeypatch):
    sent = []

    def reversed_order(query, books):
        sent.append((query, len(books)))
        return list(reversed(books))

    monkeypatch.setattr(rerank, "cross_encoder_order", reversed_order)
    agent.books(
        lambda query, filters, top_k: (
            [] if filters else [record(i, f"Book {i}") for i in range(40)][:top_k]
        )
    )
    fake = agent.llm(
        {
            Decision: book_request(query="dark revenge stories", authors=["Nobody, X"]),
            Answer: reply(picks=[pick("gutenberg:29")]),
        }
    )
    out = agent.run("dark revenge stories by Nobody")
    prompt = fake.prompt(Answer)
    assert sent == [("dark revenge stories by Nobody", 40)]  # the reranker gets the user's words
    assert (
        '"id": "gutenberg:29"' in prompt
        and '"id": "gutenberg:0"' not in prompt
        and prompt.count('"id": "gutenberg:') == 16
    )
    assert "Book 29" in out


def test_recommended_ids_come_from_assistant_links_only():
    history = [
        message("user", "see https://www.gutenberg.org/ebooks/999"),
        message(
            "assistant",
            "1. **[Roughing It](https://www.gutenberg.org/ebooks/10)** \n2. **[X](http://gutenberg.org/ebooks/72)**",
        ),
    ]
    assert recommended_book_ids(history) == {"gutenberg:10", "gutenberg:72"}


def test_more_after_everything_was_shown(agent):
    agent.books(TRAVEL)
    fake = agent.llm(
        {
            Decision: book_request(query="travel", follows_up=True, authors=["Twain, Mark"]),
            Answer: reply(picks=[pick("gutenberg:11")]),
        }
    )
    history = [
        message("user", "Twain travel"),
        message("assistant", "1. **[Roughing It](https://www.gutenberg.org/ebooks/10)**"),
        message("user", "more by him"),
    ]
    out = agent.run(history)
    assert (
        "no other books in the catalogue" in out
        and len(agent.catalogue.calls) == 1
        and not fake.called(Answer)
    )


def test_more_excludes_earlier_picks(agent):
    agent.books(LIBRARY)
    fake = agent.llm(
        {Decision: book_request(query="adventure"), Answer: reply(picks=[pick("gutenberg:72")])}
    )
    history = [
        message("user", "adventure"),
        message(
            "assistant",
            "1. **[Huck](https://www.gutenberg.org/ebooks/70)** 2. **[Tom](https://www.gutenberg.org/ebooks/71)**",
        ),
        message("user", "more"),
    ]
    out = agent.run(history)
    prompt = fake.prompt(Answer)
    assert (
        "gutenberg:70" not in prompt
        and "gutenberg:71" not in prompt
        and "gutenberg:72" in prompt
        and "Treasure Island" in out
    )


def test_exhaustive_author_request_end_to_end(agent):
    agent.books(TRAVEL)
    fake = agent.llm(
        {
            Decision: book_request(query="travel", authors=["Twain, Mark"]),
            Answer: reply(picks=[pick("gutenberg:10")], closing="Want more Twain?"),
        }
    )
    out = agent.run("books by Twain")
    assert out.rstrip().endswith(answer.EXHAUSTIVE_NOTICE) and "Want more" not in out
    assert answer.EXHAUSTIVE_INSTRUCTIONS in fake.prompt(Answer) and "{notes}" not in fake.prompt(
        Answer
    )


def test_inferred_names_never_steer_the_search(agent):
    agent.books(LIBRARY)
    fake = agent.llm(
        {
            Decision: book_request(
                query="detective fiction",
                similar_to_authors=["Doyle, Arthur Conan"],
                authors=["Christie, Agatha"],
            ),
            Answer: reply(picks=[pick("gutenberg:73")]),
        }
    )
    agent.run("I want to read a classic book about a detective")
    assert agent.catalogue.filters == [None] and "The Hound of the Baskervilles" in fake.prompt(
        Answer
    )


def test_new_request_does_not_carry_the_earlier_author(agent):
    agent.books(TRAVEL)
    agent.llm(
        {
            Decision: book_request(query="pirates", authors=["Twain, Mark"]),
            Answer: reply(picks=[pick("gutenberg:12")]),
        }
    )
    history = [
        message("user", "Twain travel"),
        message("assistant", "1. **[Roughing It](https://www.gutenberg.org/ebooks/10)**"),
        message("user", "Now something about pirates"),
    ]
    agent.run(history)
    assert agent.catalogue.filters[0] is None
