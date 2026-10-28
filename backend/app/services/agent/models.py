from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    # Decided first: whether the latest message builds on the previous request or reply.
    follows_up: bool = Field(default=False, description='E.g. "more", "more by him", "like the first one".')
    query: str = Field(description="Self-contained semantic description of the books wanted.")
    authors: list[str] = Field(default_factory=list, description='Gutenberg "Last, First" names.')
    languages: list[str] = Field(default_factory=list, description="ISO 639-1 codes.")
    published_after: int | None = None
    published_before: int | None = None
    prefer_popular: bool = False
    exclude_titles: list[str] = Field(default_factory=list)
    exclude_authors: list[str] = Field(default_factory=list, description='Gutenberg "Last, First" names.')
    # Reference points for "books like X": looked up in the catalogue by the similar step.
    similar_to_authors: list[str] = Field(default_factory=list, description='Gutenberg "Last, First" names.')
    similar_to_titles: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class BookHit:
    id: str
    title: str
    authors: list[str]
    languages: list[str]
    subjects: list[str]
    summary: str
    first_publish_year: int | None
    download_count: int
    score: float
    # Other volumes of the same multi-volume edition merged into this one by the search.
    extra_volumes: int = 0
    bookshelves: list[str] = field(default_factory=list)

    @property
    def url(self) -> str | None:
        prefix = "gutenberg:"
        if self.id.startswith(prefix):
            return f"https://www.gutenberg.org/ebooks/{self.id[len(prefix):]}"
        return None
