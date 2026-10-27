from __future__ import annotations

from typing import Any

from pinecone import Pinecone

from backend.app.config import get_settings


def get_index():
    """Return a Pinecone Index handle using settings from .env."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


def search_books(
    query_text: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Search the books index using Pinecone integrated embedding.

    Args:
        query_text: The raw user query to embed and search.
        filters: Optional Pinecone metadata filter dict (see schema below).
        top_k: Number of results to return.

    Returns:
        List of match dicts: [{"id": ..., "score": ..., "metadata": {...}}, ...]

    Metadata schema (fields available in each result["metadata"]):
        title: str                    — book title
        authors: list[str]            — e.g. ["Keller, Helen"]
        languages: list[str]          — ISO codes, e.g. ["en"]
        subjects: list[str]           — e.g. ["Perception", "Senses and sensation"]
        bookshelves: list[str]        — e.g. ["Category: Biographies"]
        download_count: float             — Gutenberg download count (stored as float)
        chunk_text: str               — semantic text used for embedding
        source: str                   — always "gutendex"
        gutendex_id: str              — e.g. "gutenberg:27683"
        author_birth_years: list[str] — e.g. ["1880"]
        author_death_years: list[str] — e.g. ["1968"]
        min_author_birth_year: float  — earliest author birth year (stored as float)
        max_author_death_year: float  — latest author death year (stored as float)
        first_publish_year: float     — approximate first publication year (from Open Library, NOT present in all documents)
        openlibrary_work_key: str     — e.g. "/works/OL53602W" (NOT present in all documents)

    Filter examples:
        {"languages": {"$in": ["en"]}}
        {"authors": {"$in": ["Shakespeare, William"]}}
        {"first_publish_year": {"$gte": 1800, "$lte": 1900}}
        {"subjects": {"$in": ["Science fiction"]}}
        {"download_count": {"$gte": 1000}}

    Example usage:
        results = search_books(
            query_text="dark revenge obsession",
            filters={"languages": {"$in": ["en"]}},
            top_k=10,
        )
        for match in results:
            print(match["metadata"]["title"], match["score"])
    """
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your .env file."
        )

    index = get_index()

    # Integrated embedding: pass query text directly.
    # Uses the flat keyword API of pinecone SDK v9+.
    search_kwargs: dict[str, Any] = {
        "namespace": settings.pinecone_namespace,
        "query": {"inputs": {"text": query_text}, "top_k": top_k},
    }
    if filters:
        search_kwargs["query"]["filter"] = filters

    try:
        response = index.search(**search_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Pinecone search failed: {exc}. Check your PINECONE_API_KEY and index name."
        ) from exc

    hits = response.result.hits if response.result else []
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "metadata": hit.fields or {},
        }
        for hit in hits
    ]
