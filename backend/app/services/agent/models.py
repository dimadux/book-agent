"""Data shared by the pipeline steps: the search plan, catalogue hits and the graph state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field

from backend.app.schemas import ChatMessage


class SearchPlan(BaseModel):
    """What the user wants, independent of the search backend."""

    # Decided first: whether the latest message builds on the previous request or reply.
    follows_up: bool = Field(
        default=False, description='E.g. "more", "more by him", "like the first one".'
    )
    query: str = Field(description="Self-contained semantic description of the books wanted.")
    authors: list[str] = Field(default_factory=list, description='Gutenberg "Last, First" names.')
    languages: list[str] = Field(default_factory=list, description="ISO 639-1 codes.")
    published_after: int | None = None
    published_before: int | None = None
    prefer_popular: bool = False
    exclude_titles: list[str] = Field(default_factory=list)
    exclude_authors: list[str] = Field(
        default_factory=list, description='Gutenberg "Last, First" names.'
    )
    # Reference points for "books like X": looked up in the catalogue by the similar step.
    similar_to_authors: list[str] = Field(
        default_factory=list, description='Gutenberg "Last, First" names.'
    )
    similar_to_titles: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class BookHit:
    """One catalogue record, as returned by the search."""

    id: str
    title: str
    authors: list[str]
    languages: list[str]
    subjects: list[str]
    bookshelves: list[str]
    summary: str
    first_publish_year: int | None
    download_count: int

    @property
    def url(self) -> str | None:
        prefix = "gutenberg:"
        if self.id.startswith(prefix):
            return f"https://www.gutenberg.org/ebooks/{self.id[len(prefix) :]}"
        return None


class AgentState(TypedDict, total=False):
    """Graph state. Nodes return partial updates; a key not declared here is silently dropped."""

    messages: list[ChatMessage]
    # Final text for turns that end without an answer from the catalogue: refusals,
    # clarifying questions, "not in the catalogue", errors.
    reply: str
    plan: SearchPlan
    books: list[BookHit]
    # User-facing names of the constraints the books do not meet, because no book met them all.
    relaxed_filters: list[str]
    # The books are every catalogue book matching the constraints: the answer must not offer more.
    exhaustive: bool
