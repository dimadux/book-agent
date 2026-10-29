"""Shared fakes for the agent tests.

- `record()` builds a search_books() match as Pinecone returns it.
- `Catalogue` answers searches in memory with Pinecone's filter semantics.
- `FakeLLM` answers each structured call with the object registered for its schema, and
  streams the answer JSON in small deltas like responses.stream().
- `agent` wires both into the pipeline and runs the graph.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.schemas import ChatMessage
from backend.app.services import book_agent
from backend.app.services.agent import answer, llm, rerank, search, similar
from backend.app.services.agent.answer import Answer, Pick
from backend.app.services.agent.models import SearchPlan
from backend.app.services.agent.plan import Decision

# ---------------------------------------------------------------- catalogue

SUMMARY = "A scientist creates life. It goes badly."


def record(
    number: int,
    title: str,
    authors: tuple[str, ...] = ("Shelley, Mary Wollstonecraft",),
    *,
    summary: str = SUMMARY,
    subjects: tuple[str, ...] = ("Horror",),
    languages: tuple[str, ...] = ("en",),
    download_count: float = 100.0,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "id": f"gutenberg:{number}",
        "score": 0.9,
        "metadata": {
            "title": title,
            "authors": list(authors),
            "languages": list(languages),
            "subjects": list(subjects),
            "chunk_text": f"Title: {title}\nSummary: {summary}",
            "download_count": download_count,
            **metadata,
        },
    }


def matches_filter(metadata: dict[str, Any], clause: dict[str, Any] | None) -> bool:
    """Pinecone's metadata filter semantics, for the in-memory catalogue."""
    if clause is None:
        return True
    if "$and" in clause:
        return all(matches_filter(metadata, part) for part in clause["$and"])
    if "$or" in clause:
        return any(matches_filter(metadata, part) for part in clause["$or"])
    ((field, condition),) = clause.items()
    ((operator, expected),) = condition.items()
    value = metadata.get(field)
    if operator == "$in":
        return bool(set(value) & set(expected)) if isinstance(value, list) else value in expected
    if not isinstance(value, (int, float)):
        return False  # missing field: Pinecone does not match it either
    return value <= expected if operator == "$lte" else value >= expected


class Catalogue:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.calls: list[tuple[str, dict[str, Any] | None, int]] = []

    def search_books(self, query: str, filters: dict[str, Any] | None, top_k: int):
        self.calls.append((query, filters, top_k))
        return [r for r in self.records if matches_filter(r["metadata"], filters)][:top_k]

    @property
    def filters(self) -> list[dict[str, Any] | None]:
        return [filters for _, filters, _ in self.calls]


# Three authors, two languages, author life years: enough for every relaxation case.
TRAVEL = [
    record(
        10,
        "Roughing It",
        ("Twain, Mark",),
        summary="West travels.",
        max_author_death_year=1910.0,
        min_author_birth_year=1835.0,
    ),
    record(
        11,
        "Humphry Clinker",
        ("Smollett, T.",),
        summary="Comic travel letters.",
        max_author_death_year=1771.0,
        min_author_birth_year=1721.0,
    ),
    record(
        12,
        "Voyage",
        ("Verne, Jules",),
        summary="A voyage.",
        languages=("fr",),
        max_author_death_year=1905.0,
        min_author_birth_year=1828.0,
    ),
]

# Reference points for the "books like X" step.
LIBRARY = [
    record(
        70,
        "Adventures of Huckleberry Finn",
        ("Twain, Mark",),
        summary="A boy rafts down the Mississippi.",
        subjects=("Adventure stories", "Humorous stories"),
        bookshelves=["Category: Adventure"],
    ),
    record(
        71,
        "The Adventures of Tom Sawyer",
        ("Twain, Mark",),
        summary="Mischief on the river.",
        subjects=("Boys -- Fiction",),
    ),
    record(
        72,
        "Treasure Island",
        ("Stevenson, Robert Louis",),
        summary="Pirates and treasure.",
        subjects=("Adventure stories",),
    ),
    record(
        73,
        "The Hound of the Baskervilles",
        ("Doyle, Arthur Conan",),
        summary="A detective and a hound.",
        subjects=("Detective",),
    ),
    record(
        74,
        "Frankenstein; Or, The Modern Prometheus",
        summary="A scientist creates life.",
        subjects=("Gothic fiction",),
    ),
]

# Gothic hits with a duplicate copy and markup in a title.
GOTHIC = [
    record(1, "Frankenstein; Or, The Modern Prometheus"),
    record(2, "The Vampyre <script>", ("Polidori, John William",)),
    record(3, "The Vampyre <script>", ("Polidori, John William",)),
    record(4, "Carmilla", ("Le Fanu, Joseph Sheridan",)),
]


def hits(records: list[dict[str, Any]]):
    return [search.to_book_hit(r) for r in records]


# ---------------------------------------------------------------- LLM


class FakeStream:
    """Mimics responses.stream(): emits the answer JSON in small text deltas."""

    def __init__(self, result, chunk: int = 7, fail_after: int | None = None):
        self.result, self.chunk, self.fail_after = result, chunk, fail_after

    async def __aenter__(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        text = self.result.model_dump_json() if self.result is not None else ""
        for start in range(0, len(text), self.chunk):
            if self.fail_after is not None and start >= self.fail_after:
                raise RuntimeError("stream cut")
            yield SimpleNamespace(
                type="response.output_text.delta", delta=text[start : start + self.chunk]
            )


class FakeLLM:
    """Answers each structured call with the object registered for its schema.

    An Exception registered for a schema is raised; None means the model returned nothing.
    `calls` records (schema name, call kwargs); `prompt()` returns the instructions sent.
    """

    def __init__(self, responses: dict[type, Any]):
        self.by_schema = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stream_fail_after: int | None = None

    @property
    def responses(self):
        return self

    async def parse(self, *, text_format, **kwargs):
        self.calls.append((text_format.__name__, kwargs))
        result = self.by_schema.get(text_format)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(output_parsed=result)

    def stream(self, *, text_format, **kwargs):
        self.calls.append((text_format.__name__, kwargs))
        return FakeStream(self.by_schema.get(text_format), fail_after=self.stream_fail_after)

    def called(self, schema: type) -> bool:
        return any(name == schema.__name__ for name, _ in self.calls)

    def prompt(self, schema: type) -> str:
        return next(
            kwargs["instructions"] for name, kwargs in self.calls if name == schema.__name__
        )

    def last_input(self, schema: type) -> list[dict[str, str]]:
        return [kwargs["input"] for name, kwargs in self.calls if name == schema.__name__][-1]


def book_request(**plan) -> Decision:
    return Decision(intent="book_request", plan=SearchPlan(**plan))


def pick(
    book_id: str, reason: str = "r", fit: str = "strong", evidence: str = "scientist creates life"
) -> Pick:
    return Pick(book_id=book_id, fit=fit, evidence=evidence, reason=reason)


def reply(
    intro: str = "i", picks: list[Pick] = (), closing: str = "", has_matches: bool | None = None
) -> Answer:
    picks = list(picks)
    return Answer(
        has_matches=bool(picks) if has_matches is None else has_matches,
        intro=intro,
        picks=picks,
        closing=closing,
    )


def message(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


# ---------------------------------------------------------------- harness


class Agent:
    """Wires the fakes into the pipeline and runs the graph."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.patch = monkeypatch
        self.fake = FakeLLM({})
        self.catalogue = Catalogue([])
        self.llm({})
        self.books([])
        # Most tests are about other steps; the evidence check has its own tests.
        monkeypatch.setattr(answer, "is_grounded", lambda evidence, book: True)
        # The cross-encoder needs Pinecone; tests of the rerank step install their own.
        monkeypatch.setattr(rerank, "cross_encoder_order", lambda query, books: books)

    def llm(self, responses: dict[type, Any]) -> FakeLLM:
        self.fake = FakeLLM(responses)
        self.patch.setattr(llm, "get_llm_client", lambda: self.fake)
        self.patch.setattr(answer, "get_llm_client", lambda: self.fake)
        return self.fake

    def books(self, records_or_search) -> Catalogue:
        """An in-memory catalogue from records, or a search_books() replacement."""
        if callable(records_or_search):
            self.catalogue = Catalogue([])
            self.patch.setattr(search, "search_books", records_or_search)
        else:
            self.catalogue = Catalogue(list(records_or_search))
            self.patch.setattr(search, "search_books", self.catalogue.search_books)
        return self.catalogue

    def node_writes(self) -> list[str]:
        """Captures what nodes write when they are called directly, outside the graph."""
        writes: list[str] = []
        for module in (answer, similar, book_agent):
            self.patch.setattr(module, "get_stream_writer", lambda: writes.append)
        return writes

    def run(self, messages: str | list[ChatMessage]) -> str:
        if isinstance(messages, str):
            messages = [message("user", messages)]

        async def collect():
            return "".join([c async for c in book_agent.stream_book_agent_response(messages)])

        return asyncio.run(collect())


@pytest.fixture
def agent(monkeypatch) -> Agent:
    return Agent(monkeypatch)
