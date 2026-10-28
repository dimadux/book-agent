"""Search node: finds candidate books for the plan, falling back when nothing matches.

1. Strict: search with every constraint. Most requests end here.
2. Partial: if nothing met them all, search for books meeting any constraint and keep the
   ones meeting the most. Skipped with a single constraint, where it would repeat step 1.
3. Unfiltered: if no book meets any constraint, search on the query alone.

The constraints the returned books miss are reported, so the answer can tell the user.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any

from backend.app.services.agent.conversation import recommended_book_ids
from backend.app.services.agent.models import BookHit, SearchPlan
from backend.app.services.agent.pinecone_filters import (
    Filter,
    FilterGroup,
    all_of,
    any_of,
    filter_groups,
)
from backend.app.services.agent.state import AgentState
from backend.app.services.pinecone_utils import search_books

logger = logging.getLogger(__name__)

# A search result as returned by search_books(): {"id": ..., "score": ..., "metadata": {...}}.
RawMatch = dict[str, Any]

SEARCH_TOP_K = 30
# The partial step ranks results in Python, so it fetches as many as the search API returns.
PARTIAL_TOP_K = 100
# Enough for the rerank step to choose from; it cuts the list down for the answer.
MAX_CANDIDATES = 30

SEARCH_ERROR_REPLY = "Sorry, the book catalogue is unavailable right now. Please try again in a moment."


@dataclass
class SearchOutcome:
    books: list[BookHit]
    # Constraints the books do not meet; empty when they meet all of them.
    unmet: list[FilterGroup] = field(default_factory=list)
    # The books are every catalogue book matching the filters.
    exhaustive: bool = False
    # Books matched, but all of them were recommended earlier in the conversation.
    already_shown: bool = False


async def search_node(state: AgentState) -> AgentState:
    try:
        # Never recommend a book twice in a conversation, e.g. after "give me more".
        shown = recommended_book_ids(state["messages"])
        outcome = await search_with_fallback(state["plan"], shown)
    except Exception:
        logger.exception("Pinecone search failed")
        return {"reply": SEARCH_ERROR_REPLY}

    return {
        "books": outcome.books,
        "relaxed_filters": [group.label for group in outcome.unmet],
        # Unmet constraints mean the books come from a fallback step.
        "search_degraded": bool(outcome.unmet),
        "books_exhaustive": outcome.exhaustive,
        "already_shown": outcome.already_shown,
    }


async def search_with_fallback(plan: SearchPlan, shown: frozenset[str] = frozenset()) -> SearchOutcome:
    groups = filter_groups(plan)

    matches = await run_search(plan.query, all_of([group.clause for group in groups]), SEARCH_TOP_K)
    books = select_candidates(matches, plan, shown)
    if books:
        # Fewer matches than requested: the filters let nothing else through.
        return SearchOutcome(books, exhaustive=bool(groups) and len(matches) < SEARCH_TOP_K)
    if any(match.get("id") in shown for match in matches):
        # The constraints do match books, but the user has seen them (the filter step may
        # also have listed them in exclude_titles): relaxing would only show books that
        # miss the constraints.
        return SearchOutcome([], already_shown=True)
    if not groups:
        return SearchOutcome([])  # nothing to relax

    if len(groups) > 1:
        outcome = await find_partial_matches(plan, groups, shown)
        if outcome.books:
            return outcome

    matches = await run_search(query_with_authors(plan), None, SEARCH_TOP_K)
    books = select_candidates(matches, plan, shown)
    return SearchOutcome(books, unmet=groups if books else [])


async def find_partial_matches(
    plan: SearchPlan, groups: list[FilterGroup], shown: frozenset[str]
) -> SearchOutcome:
    matches = await run_search(
        query_with_authors(plan), any_of([group.clause for group in groups]), PARTIAL_TOP_K
    )
    met = [met_groups(match, groups) for match in matches]
    if not any(met):
        return SearchOutcome([])

    best = max(met, key=lambda names: preference(names, groups))
    kept = [match for match, names in zip(matches, met, strict=True) if names == best]
    unmet = [group for group in groups if group.name not in best]
    return SearchOutcome(select_candidates(kept, plan, shown), unmet)


def met_groups(match: RawMatch, groups: list[FilterGroup]) -> frozenset[str]:
    metadata = match.get("metadata") or {}
    return frozenset(group.name for group in groups if group.is_met_by(metadata))


def preference(met: frozenset[str], groups: list[FilterGroup]) -> tuple[int, tuple[bool, ...]]:
    """Sort key for sets of met constraints: more is better; on a tie, keeping the groups
    that come last in `groups` is better, as earlier groups are the first to give up."""
    return len(met), tuple(group.name in met for group in reversed(groups))


def query_with_authors(plan: SearchPlan) -> str:
    # Fallback steps may not filter on the author, so put the name in the query text.
    return f"{plan.query} by {' or '.join(plan.authors)}" if plan.authors else plan.query


async def run_search(query: str, filters: Filter | None, top_k: int) -> list[RawMatch]:
    matches = await asyncio.to_thread(search_books, query, filters, top_k)
    logger.info("Search filters=%s returned %d matches", filters, len(matches))
    return matches


def select_candidates(
    matches: list[RawMatch], plan: SearchPlan, shown: frozenset[str] = frozenset()
) -> list[BookHit]:
    """Turn matches into at most MAX_CANDIDATES books, in search order.

    Drops books recommended before (`shown`), excluded titles and authors, and duplicates.
    """
    books = [book for book in map(to_book_hit, matches) if book is not None]
    books = [book for book in books if book.id not in shown]
    books = [book for book in books if not is_excluded(book, plan.exclude_titles)]
    books = [book for book in books if not has_author(book, plan.exclude_authors)]
    return unique_editions(books)[:MAX_CANDIDATES]


def to_book_hit(match: RawMatch) -> BookHit | None:
    metadata = match.get("metadata") or {}
    title = metadata.get("title")
    if not isinstance(title, str) or not title:
        return None
    # chunk_text is "Title: ...\nSummary: ..."; keep only the summary.
    chunk_text = str(metadata.get("chunk_text") or "")
    summary = chunk_text.split("Summary:", 1)[1].strip() if "Summary:" in chunk_text else ""
    year = metadata.get("first_publish_year")
    return BookHit(
        id=str(match.get("id") or metadata.get("gutendex_id") or ""),
        title=title,
        authors=[str(name) for name in metadata.get("authors") or []],
        languages=[str(code) for code in metadata.get("languages") or []],
        subjects=[str(subject) for subject in metadata.get("subjects") or []],
        bookshelves=[str(shelf) for shelf in metadata.get("bookshelves") or []],
        summary=summary,
        first_publish_year=int(year) if isinstance(year, (int, float)) else None,
        download_count=int(metadata.get("download_count") or 0),
        score=float(match.get("score") or 0.0),
    )


# "[Vol. 8 of 9]", "Volume II", "Part 3 (of 4)", "v. 2": marks one volume of a multi-volume edition.
VOLUME_MARKER = re.compile(
    r"[\[(]?\b(?:vol(?:ume)?\.?|part|tome|v\.)\s*(?:\d+|[ivxlc]+)\b"
    r"(?:\s*(?:\(\s*)?(?:of|/)\s*(?:\d+|[ivxlc]+)\)?)?[\])]?",
    re.IGNORECASE,
)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def edition_title(title: str) -> str:
    """Normalized title with volume markers removed, shared by all volumes of an edition."""
    return normalize_title(VOLUME_MARKER.sub(" ", title))


def is_excluded(book: BookHit, exclude_titles: list[str]) -> bool:
    title = normalize_title(book.title)
    for excluded in map(normalize_title, exclude_titles):
        # Reference books are often matched by a longer full title ("Frankenstein; Or, ...").
        if excluded and (title == excluded or title.startswith(f"{excluded} ")):
            return True
    return False


def name_words(name: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", name.casefold()))


def same_author(wanted: str, catalogue_name: str) -> bool:
    """Whether a name refers to a catalogue author, ignoring word order and extra names.

    "Conan Doyle, Arthur" matches "Doyle, Arthur Conan"; "Shelley, Mary" matches
    "Shelley, Mary Wollstonecraft".
    """
    words = name_words(wanted)
    return bool(words) and words <= name_words(catalogue_name)


def has_author(book: BookHit, names: list[str]) -> bool:
    return any(same_author(name, author) for name in names for author in book.authors)


def unique_editions(books: list[BookHit]) -> list[BookHit]:
    """Keep the first, best-scoring book per edition.

    Gutenberg has several copies of many books, and multi-volume editions as separate
    records. Copies are dropped; other volumes are counted on the kept one.
    """
    kept: dict[tuple[str, str], BookHit] = {}
    for book in books:
        key = (edition_title(book.title), book.authors[0] if book.authors else "")
        first = kept.get(key)
        if first is None:
            kept[key] = book
        elif normalize_title(book.title) != normalize_title(first.title):
            kept[key] = replace(first, extra_volumes=first.extra_volumes + 1)
    return list(kept.values())
