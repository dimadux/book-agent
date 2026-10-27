from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

from ingest_gutendex_to_pinecone import (
    BookRecord,
    batched,
    embed_texts,
    iter_gutendex_books,
)

OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"


def fetch_json(url: str, retries: int = 5, delay_seconds: float = 1.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "book-indexer/0.1 enrichment"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(delay_seconds * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def build_openlibrary_url(book: BookRecord) -> str:
    params = {
        "title": book.title,
        "fields": "key,title,author_name,first_publish_year",
        "limit": "5",
    }
    if book.authors:
        params["author"] = book.authors[0]
    return f"{OPEN_LIBRARY_SEARCH_URL}?{urllib.parse.urlencode(params)}"


def score_openlibrary_doc(book: BookRecord, doc: dict[str, Any]) -> int:
    score = 0
    doc_title = normalize_text(str(doc.get("title") or ""))
    book_title = normalize_text(book.title)
    if doc_title == book_title:
        score += 5
    elif book_title in doc_title or doc_title in book_title:
        score += 2

    doc_authors = {normalize_text(author) for author in doc.get("author_name", [])}
    book_authors = {normalize_text(author) for author in book.authors}
    if doc_authors & book_authors:
        score += 5

    if isinstance(doc.get("first_publish_year"), int):
        score += 1

    return score


def find_first_publish_year(book: BookRecord) -> dict[str, Any]:
    payload = fetch_json(build_openlibrary_url(book))
    docs = payload.get("docs", [])
    if not docs:
        return {}

    scored_docs = sorted(
        docs,
        key=lambda doc: score_openlibrary_doc(book, doc),
        reverse=True,
    )
    best = scored_docs[0]
    year = best.get("first_publish_year")
    if not isinstance(year, int):
        return {}
    return {
        "first_publish_year": year,
        "openlibrary_work_key": best.get("key"),
    }


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            cache[row["id"]] = row["metadata"]
    return cache


def append_cache(path: Path, book_id: str, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"id": book_id, "metadata": metadata}, ensure_ascii=False) + "\n")


def enrich_books(
    *,
    books: list[BookRecord],
    cache_path: Path,
    lookup_delay_seconds: float,
    workers: int,
) -> list[tuple[BookRecord, dict[str, Any]]]:
    cache = load_cache(cache_path)
    enriched_by_id: dict[str, dict[str, Any]] = dict(cache)
    missing_books = [book for book in books if book.id not in cache]
    matched = 0

    print(
        f"Open Library cache has {len(cache)}/{len(books)} books; "
        f"looking up {len(missing_books)} missing books with {workers} workers",
        flush=True,
    )

    if workers <= 1:
        for index, book in enumerate(missing_books, start=1):
            metadata = find_first_publish_year(book)
            append_cache(cache_path, book.id, metadata)
            enriched_by_id[book.id] = metadata
            if lookup_delay_seconds > 0:
                time.sleep(lookup_delay_seconds)

            processed = len(cache) + index
            matched = sum(
                1 for metadata_value in enriched_by_id.values() if metadata_value.get("first_publish_year")
            )
            if processed % 100 == 0 or processed == len(books):
                print(
                    f"Open Library enriched {processed}/{len(books)} books; "
                    f"{matched} with first_publish_year",
                    flush=True,
                )
    else:
        completed = len(cache)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(find_first_publish_year, book): book for book in missing_books}
            for future in as_completed(futures):
                book = futures[future]
                try:
                    metadata = future.result()
                except Exception as error:
                    print(f"Open Library lookup failed for {book.id}: {error}", flush=True)
                    metadata = {}
                append_cache(cache_path, book.id, metadata)
                enriched_by_id[book.id] = metadata

                completed += 1
                if lookup_delay_seconds > 0:
                    time.sleep(lookup_delay_seconds)

                if completed % 100 == 0 or completed == len(books):
                    matched = sum(
                        1
                        for metadata_value in enriched_by_id.values()
                        if metadata_value.get("first_publish_year")
                    )
                    print(
                        f"Open Library enriched {completed}/{len(books)} books; "
                        f"{matched} with first_publish_year",
                        flush=True,
                    )

    return [(book, enriched_by_id.get(book.id, {})) for book in books]


def upsert_enriched_dense(
    *,
    index: Any,
    api_key: str,
    namespace: str,
    model: str,
    enriched_books: list[tuple[BookRecord, dict[str, Any]]],
    batch_size: int,
    batch_delay_seconds: float,
) -> None:
    total = 0
    for batch in batched([book for book, _ in enriched_books], batch_size):
        batch_lookup = {book.id: metadata for book, metadata in enriched_books[total : total + len(batch)]}
        vectors = embed_texts(api_key, model, [book.semantic_text for book in batch])
        records = []
        for book, vector in zip(batch, vectors, strict=True):
            metadata = book.metadata
            metadata.update(batch_lookup.get(book.id, {}))
            records.append({"id": book.id, "values": vector, "metadata": metadata})

        index.upsert(vectors=records, namespace=namespace)
        total += len(batch)
        print(f"Upserted {total}/{len(enriched_books)} enriched dense records", flush=True)
        if batch_delay_seconds > 0 and total < len(enriched_books):
            time.sleep(batch_delay_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich Gutendex books with Open Library years.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--languages", default="en")
    parser.add_argument("--sort", default="popular", choices=["popular", "ascending", "descending"])
    parser.add_argument("--max-pages", type=int, default=250)
    parser.add_argument("--page-delay", type=float, default=1.0)
    parser.add_argument("--lookup-delay", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-delay", type=float, default=3.0)
    parser.add_argument("--cache-path", default=".cache/openlibrary_years.jsonl")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.getenv("PINECONE_INDEX_NAME", "books")
    namespace = os.getenv("PINECONE_NAMESPACE", "books-v1")
    embed_model = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")

    print("Fetching Gutendex books...", flush=True)
    books = list(
        iter_gutendex_books(
            limit=args.limit,
            languages=args.languages,
            topic=None,
            search=None,
            sort=args.sort,
            page_delay_seconds=args.page_delay,
            max_pages=args.max_pages,
        )
    )
    print(f"Fetched {len(books)} Gutendex books", flush=True)

    enriched_books = enrich_books(
        books=books,
        cache_path=Path(args.cache_path),
        lookup_delay_seconds=args.lookup_delay,
        workers=args.workers,
    )

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(index_name)
    upsert_enriched_dense(
        index=pinecone_index,
        api_key=api_key,
        namespace=namespace,
        model=embed_model,
        enriched_books=enriched_books,
        batch_size=args.batch_size,
        batch_delay_seconds=args.batch_delay,
    )
    print("Done", flush=True)


if __name__ == "__main__":
    main()
