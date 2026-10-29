"""Search node: Pinecone candidates for the plan, dropping constraints only when nothing matches.

Also home of the catalogue helpers (hit parsing, title and author matching, duplicate editions)
that the similar step shares.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from backend.app.services.agent.conversation import recommended_book_ids
from backend.app.services.agent.models import AgentState, BookHit, SearchPlan
from backend.app.services.pinecone_utils import search_books

logger = logging.getLogger(__name__)

SEARCH_TOP_K = 30
# The SDK retries on its own; this keeps one stuck call from holding the whole turn.
SEARCH_TIMEOUT_SECONDS = 10
SEARCH_ERROR_REPLY = (
    "Sorry, the book catalogue is unavailable right now. Please try again in a moment."
)
ALREADY_SHOWN_REPLY = (
    "There are no other books in the catalogue matching this. "
    "Try a different author, theme or mood."
)

Filter = dict[str, Any]
# A user constraint: its user-facing name and its Pinecone filter clause.
Constraint = tuple[str, Filter]


@dataclass
class SearchResult:
    books: list[BookHit] = field(default_factory=list)
    # Names of the constraints the books do not meet, because no book met them all.
    relaxed: list[str] = field(default_factory=list)
    # The books are every catalogue book matching the constraints.
    exhaustive: bool = False
    # Books matched, but the user has already seen or excluded them all.
    already_shown: bool = False


async def search_node(state: AgentState) -> AgentState:
    try:
        # Never recommend a book twice in a conversation, e.g. after "give me more".
        result = await search_with_relaxation(
            state["plan"], recommended_book_ids(state["messages"])
        )
    except Exception:
        logger.exception("Pinecone search failed")
        return {"reply": SEARCH_ERROR_REPLY}
    if result.already_shown:
        return {"reply": ALREADY_SHOWN_REPLY}
    return {
        "books": result.books,
        "relaxed_filters": result.relaxed,
        "exhaustive": result.exhaustive,
    }


async def search_with_relaxation(plan: SearchPlan, shown: frozenset[str]) -> SearchResult:
    constraints = plan_constraints(plan)
    books, matched = await search(plan, constraints, shown)
    if books:
        # Fewer matches than asked for: the constraints let nothing else through.
        return SearchResult(books, exhaustive=bool(constraints) and matched < SEARCH_TOP_K)
    if matched:
        # Relaxing would only show books that miss the constraints.
        return SearchResult(already_shown=True)

    # Nothing meets every constraint. Try every smaller set of them at once and take the first
    # that matches: the sets are ordered to drop as few constraints as possible, and the least
    # important first.
    subsets = [
        list(kept)
        for size in range(len(constraints) - 1, -1, -1)
        for kept in combinations(reversed(constraints), size)
    ]
    results = await asyncio.gather(*(search(plan, kept, shown) for kept in subsets))
    for kept, (books, _) in zip(subsets, results, strict=True):
        if books:
            kept_names = {name for name, _ in kept}
            return SearchResult(
                books, relaxed=[name for name, _ in constraints if name not in kept_names]
            )
    return SearchResult()


def plan_constraints(plan: SearchPlan) -> list[Constraint]:
    """The plan's constraints as Pinecone filters, least important first.

    Years are only approximated, so they are the first to go; "books by X" is usually the
    point of the request, so the author goes last.
    """
    constraints: list[Constraint] = []
    # first_publish_year is missing for many books, so author life years stand in: an author
    # who died before X wrote before X, one born after X wrote after X.
    years: list[Filter] = []
    if (year := plan.published_before) is not None:
        years.append(
            {
                "$or": [
                    {"first_publish_year": {"$lte": year}},
                    {"max_author_death_year": {"$lte": year}},
                ]
            }
        )
    if (year := plan.published_after) is not None:
        years.append(
            {
                "$or": [
                    {"first_publish_year": {"$gte": year}},
                    {"min_author_birth_year": {"$gte": year}},
                ]
            }
        )
    if years:
        constraints.append(("publication years", all_of(years)))
    if plan.languages:
        constraints.append(("language", {"languages": {"$in": plan.languages}}))
    if plan.authors:
        constraints.append(("author", {"authors": {"$in": plan.authors}}))
    return constraints


def all_of(clauses: list[Filter]) -> Filter | None:
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


async def search(
    plan: SearchPlan, kept: list[Constraint], shown: frozenset[str]
) -> tuple[list[BookHit], int]:
    """Candidates meeting the kept constraints, and how many records matched before clean-up."""
    query = plan.query
    if plan.authors and "author" not in [name for name, _ in kept]:
        # The author is no longer filtered on, so name them in the query text.
        query = f"{query} by {' or '.join(plan.authors)}"
    found = await find_books(query, all_of([clause for _, clause in kept]), SEARCH_TOP_K)
    books = [book for book in found if book.id not in shown and not is_excluded(book, plan)]
    return unique_editions(books), len(found)


async def find_books(query: str, filters: Filter | None, top_k: int) -> list[BookHit]:
    matches = await asyncio.wait_for(
        asyncio.to_thread(search_books, query, filters, top_k), SEARCH_TIMEOUT_SECONDS
    )
    logger.info("Search filters=%s returned %d matches", filters, len(matches))
    return [book for book in map(to_book_hit, matches) if book is not None]


def to_book_hit(match: dict[str, Any]) -> BookHit | None:
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
    )


def is_excluded(book: BookHit, plan: SearchPlan) -> bool:
    return any(same_title(title, book.title) for title in plan.exclude_titles) or any(
        same_author(name, author) for name in plan.exclude_authors for author in book.authors
    )


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def same_title(wanted: str, catalogue_title: str) -> bool:
    """Whether a title as people write it names a catalogue title, which is often longer
    ("Frankenstein" vs "Frankenstein; Or, The Modern Prometheus")."""
    wanted, title = normalize_title(wanted), normalize_title(catalogue_title)
    return bool(wanted) and (title == wanted or title.startswith(f"{wanted} "))


def same_author(wanted: str, catalogue_name: str) -> bool:
    """Whether a name refers to a catalogue author, ignoring word order and extra names:
    "Conan Doyle, Arthur" matches "Doyle, Arthur Conan", "Shelley, Mary" matches
    "Shelley, Mary Wollstonecraft"."""
    words = frozenset(re.findall(r"[a-z]+", wanted.casefold()))
    return bool(words) and words <= frozenset(re.findall(r"[a-z]+", catalogue_name.casefold()))


# "[Vol. 8 of 9]", "Volume II", "Part 3 (of 4)", "v. 2": marks one volume of a multi-volume edition.
VOLUME_MARKER = re.compile(
    r"[\[(]?\b(?:vol(?:ume)?\.?|part|tome|v\.)\s*(?:\d+|[ivxlc]+)\b"
    r"(?:\s*(?:\(\s*)?(?:of|/)\s*(?:\d+|[ivxlc]+)\)?)?[\])]?",
    re.IGNORECASE,
)


def unique_editions(books: list[BookHit]) -> list[BookHit]:
    """Keeps the first, best-scoring record per work: Gutenberg has several copies of many
    books, and the volumes of multi-volume editions as separate records."""
    kept: dict[tuple[str, str], BookHit] = {}
    for book in books:
        key = (
            normalize_title(VOLUME_MARKER.sub(" ", book.title)),
            book.authors[0] if book.authors else "",
        )
        kept.setdefault(key, book)
    return list(kept.values())
