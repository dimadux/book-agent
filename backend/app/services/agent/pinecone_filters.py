"""Translates a SearchPlan into Pinecone metadata filters for the books index.

Field names and value types follow the index schema documented in pinecone_utils.search_books().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.services.agent.models import SearchPlan

Filter = dict[str, Any]


@dataclass(frozen=True)
class FilterGroup:
    """One user constraint (years, author or language) as a Pinecone filter clause."""

    name: str
    label: str  # user-facing name, used in the "couldn't match your ... filter" notice
    clause: Filter

    def is_met_by(self, metadata: dict[str, Any]) -> bool:
        return matches_filter(metadata, self.clause)


def filter_groups(plan: SearchPlan) -> list[FilterGroup]:
    """The plan's constraints, in the order they are given up first when nothing matches.

    Years go first: they are a soft preference and only approximated. The author goes last:
    "books by X" is usually the point of the request.
    """
    groups: list[FilterGroup] = []
    years = _years_clause(plan)
    if years:
        groups.append(FilterGroup("years", "publication years", years))
    if plan.languages:
        groups.append(FilterGroup("languages", "language", {"languages": {"$in": plan.languages}}))
    if plan.authors:
        groups.append(FilterGroup("authors", "author", {"authors": {"$in": plan.authors}}))
    return groups


def _years_clause(plan: SearchPlan) -> Filter | None:
    # first_publish_year is missing for many books, so author life years stand in:
    # an author who died before X wrote before X, one born after X wrote after X.
    bounds = []
    if plan.published_before is not None:
        year = plan.published_before
        bounds.append(any_of([{"first_publish_year": {"$lte": year}}, {"max_author_death_year": {"$lte": year}}]))
    if plan.published_after is not None:
        year = plan.published_after
        bounds.append(any_of([{"first_publish_year": {"$gte": year}}, {"min_author_birth_year": {"$gte": year}}]))
    return all_of(bounds)


def all_of(clauses: list[Filter]) -> Filter | None:
    return _combine("$and", clauses)


def any_of(clauses: list[Filter]) -> Filter | None:
    return _combine("$or", clauses)


def _combine(operator: str, clauses: list[Filter]) -> Filter | None:
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {operator: clauses}


def matches_filter(metadata: dict[str, Any], clause: Filter) -> bool:
    """Evaluate a filter built by this module in Python, the way Pinecone evaluates it."""
    if "$and" in clause:
        return all(matches_filter(metadata, part) for part in clause["$and"])
    if "$or" in clause:
        return any(matches_filter(metadata, part) for part in clause["$or"])

    field, condition = next(iter(clause.items()))
    operator, expected = next(iter(condition.items()))
    value = metadata.get(field)
    if operator == "$in":
        # On a list field, $in matches when any element is one of the expected values.
        values = value if isinstance(value, list) else [value]
        return any(item in expected for item in values)
    if operator in ("$lte", "$gte"):
        if not isinstance(value, (int, float)):
            return False  # missing field: Pinecone does not match it either
        return value <= expected if operator == "$lte" else value >= expected
    raise ValueError(f"Unsupported filter operator: {operator}")
