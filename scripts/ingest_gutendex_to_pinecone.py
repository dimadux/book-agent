from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

GUTENDEX_BASE_URL = "https://gutendex.com/books"
PINECONE_EMBED_URL = "https://api.pinecone.io/embed"
PINECONE_API_VERSION = "2025-10"


@dataclass(frozen=True)
class BookRecord:
    id: str
    title: str
    authors: list[str]
    author_birth_years: list[int]
    author_death_years: list[int]
    summaries: list[str]
    subjects: list[str]
    bookshelves: list[str]
    languages: list[str]
    download_count: int

    @property
    def semantic_text(self) -> str:
        summary_text = "\n".join(self.summaries).strip()
        if summary_text:
            return f"Title: {self.title}\nSummary: {summary_text}"
        return f"Title: {self.title}"

    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "gutendex",
            "gutendex_id": self.id,
            "title": self.title,
            "download_count": self.download_count,
            "chunk_text": self.semantic_text,
        }
        if self.authors:
            metadata["authors"] = self.authors
        if self.author_birth_years:
            metadata["author_birth_years"] = [str(year) for year in self.author_birth_years]
            metadata["min_author_birth_year"] = min(self.author_birth_years)
        if self.author_death_years:
            metadata["author_death_years"] = [str(year) for year in self.author_death_years]
            metadata["max_author_death_year"] = max(self.author_death_years)
        if self.subjects:
            metadata["subjects"] = self.subjects[:100]
        if self.bookshelves:
            metadata["bookshelves"] = self.bookshelves[:50]
        if self.languages:
            metadata["languages"] = self.languages
        return metadata


def fetch_json(url: str, retries: int = 3, delay_seconds: float = 1.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "book-indexer/0.1"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                import json

                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(delay_seconds * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def build_gutendex_url(
    *,
    page: int,
    languages: str | None,
    topic: str | None,
    search: str | None,
    sort: str,
) -> str:
    params = {"page": str(page), "sort": sort}
    if languages:
        params["languages"] = languages
    if topic:
        params["topic"] = topic
    if search:
        params["search"] = search
    return f"{GUTENDEX_BASE_URL}?{urllib.parse.urlencode(params)}"


def parse_book(raw: dict[str, Any]) -> BookRecord | None:
    title = raw.get("title")
    summaries = [value for value in raw.get("summaries", []) if isinstance(value, str) and value]
    if not title or not summaries:
        return None

    authors_raw = raw.get("authors", [])
    authors = []
    birth_years = []
    death_years = []
    for author in authors_raw:
        name = author.get("name")
        if name:
            authors.append(name)
        birth_year = author.get("birth_year")
        death_year = author.get("death_year")
        if isinstance(birth_year, int):
            birth_years.append(birth_year)
        if isinstance(death_year, int):
            death_years.append(death_year)

    return BookRecord(
        id=f"gutenberg:{raw['id']}",
        title=title,
        authors=authors,
        author_birth_years=birth_years,
        author_death_years=death_years,
        summaries=summaries,
        subjects=[value for value in raw.get("subjects", []) if isinstance(value, str)],
        bookshelves=[value for value in raw.get("bookshelves", []) if isinstance(value, str)],
        languages=[value for value in raw.get("languages", []) if isinstance(value, str)],
        download_count=int(raw.get("download_count") or 0),
    )


def iter_gutendex_books(
    *,
    limit: int,
    languages: str | None,
    topic: str | None,
    search: str | None,
    sort: str,
    page_delay_seconds: float,
    max_pages: int | None,
) -> Iterable[BookRecord]:
    page = 1
    yielded = 0
    while yielded < limit:
        if max_pages is not None and page > max_pages:
            print(f"Stopped after {max_pages} Gutendex pages with {yielded} usable books", flush=True)
            return
        url = build_gutendex_url(
            page=page,
            languages=languages,
            topic=topic,
            search=search,
            sort=sort,
        )
        payload = fetch_json(url)
        results = payload.get("results", [])
        if not results:
            return

        before_page = yielded
        for raw_book in results:
            book = parse_book(raw_book)
            if book is None:
                continue
            yield book
            yielded += 1
            if yielded >= limit:
                return

        print(
            f"Fetched page {page}: +{yielded - before_page} usable books, total {yielded}/{limit}",
            flush=True,
        )
        if not payload.get("next"):
            return
        page += 1
        if page_delay_seconds > 0:
            time.sleep(page_delay_seconds)


def batched(items: list[BookRecord], batch_size: int) -> Iterable[list[BookRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(
    api_key: str,
    model: str,
    texts: list[str],
    retries: int = 8,
    delay_seconds: float = 5.0,
) -> list[list[float]]:
    import json

    body = json.dumps(
        {
            "model": model,
            "parameters": {"input_type": "passage", "truncate": "END"},
            "inputs": [{"text": text} for text in texts],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        PINECONE_EMBED_URL,
        data=body,
        headers={
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": PINECONE_API_VERSION,
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [item["values"] for item in payload["data"]]
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            retry_after = error.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else delay_seconds * attempt
            print(
                f"Pinecone embed HTTP {error.code}; retry {attempt}/{retries} "
                f"after {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        except Exception as error:
            last_error = error
            if attempt >= retries:
                break
            wait_seconds = delay_seconds * attempt
            print(f"Pinecone embed error; retry {attempt}/{retries} after {wait_seconds:.1f}s", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError("Failed to embed texts") from last_error


def upsert_dense(
    *,
    index: Any,
    api_key: str,
    namespace: str,
    model: str,
    books: list[BookRecord],
    batch_size: int,
    batch_delay_seconds: float,
) -> None:
    total = 0
    for batch in batched(books, batch_size):
        vectors = embed_texts(api_key, model, [book.semantic_text for book in batch])
        index.upsert(
            vectors=[
                {"id": book.id, "values": vector, "metadata": book.metadata}
                for book, vector in zip(batch, vectors, strict=True)
            ],
            namespace=namespace,
        )
        total += len(batch)
        print(f"Upserted {total}/{len(books)} dense records", flush=True)
        if batch_delay_seconds > 0 and total < len(books):
            time.sleep(batch_delay_seconds)


def upsert_integrated(
    *,
    index: Any,
    namespace: str,
    text_field: str,
    books: list[BookRecord],
    batch_size: int,
    batch_delay_seconds: float,
) -> None:
    total = 0
    for batch in batched(books, batch_size):
        records = []
        for book in batch:
            record = {"_id": book.id, text_field: book.semantic_text}
            record.update(book.metadata)
            records.append(record)
        index.upsert_records(namespace=namespace, records=records)
        total += len(batch)
        print(f"Upserted {total}/{len(books)} integrated records", flush=True)
        if batch_delay_seconds > 0 and total < len(books):
            time.sleep(batch_delay_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Gutendex books into Pinecone.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--languages", default="en")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--search", default=None)
    parser.add_argument("--sort", default="popular", choices=["popular", "ascending", "descending"])
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--page-delay", type=float, default=1.0)
    parser.add_argument("--batch-delay", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.getenv("PINECONE_INDEX_NAME", "books")
    namespace = os.getenv("PINECONE_NAMESPACE", "books-v1")
    upsert_mode = os.getenv("PINECONE_UPSERT_MODE", "dense").lower()
    embed_model = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
    text_field = os.getenv("PINECONE_TEXT_FIELD", "chunk_text")

    print("Fetching books from Gutendex...", flush=True)
    books = list(
        iter_gutendex_books(
            limit=args.limit,
            languages=args.languages,
            topic=args.topic,
            search=args.search,
            sort=args.sort,
            page_delay_seconds=args.page_delay,
            max_pages=args.max_pages,
        )
    )
    print(f"Fetched {len(books)} books with summaries", flush=True)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    if upsert_mode == "dense":
        upsert_dense(
            index=index,
            api_key=api_key,
            namespace=namespace,
            model=embed_model,
            books=books,
            batch_size=args.batch_size,
            batch_delay_seconds=args.batch_delay,
        )
    elif upsert_mode == "integrated":
        upsert_integrated(
            index=index,
            namespace=namespace,
            text_field=text_field,
            books=books,
            batch_size=args.batch_size,
            batch_delay_seconds=args.batch_delay,
        )
    else:
        raise ValueError("PINECONE_UPSERT_MODE must be 'dense' or 'integrated'")

    print("Done", flush=True)


if __name__ == "__main__":
    main()
