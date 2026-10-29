"""Evaluates the book agent against evals/dataset.json.

Each case is a conversation plus what a good reply looks like: the books that should be
recommended (graded: "ideal" counts double), books that must not appear, and for non-search
turns the fixed reply expected (clarify, off-topic, ...). The agent runs for real, against
OpenAI and Pinecone, so results vary a little between runs.

Metrics per case:
- nDCG@5 of the recommended books (the headline quality number: graded, order-aware)
- precision@5 and recall@5 of the picks, MRR
- retrieval recall@10: how many relevant books reached the answer step at all, which tells
  retrieval failures apart from answer-selection failures
- forbidden picks, hallucinated ids (links to books not in the catalogue), behaviour match

A case passes when behaviour matches, nothing forbidden or hallucinated was recommended and,
if relevant books are defined, nDCG@5 >= --pass-ndcg.

Usage:
    uv run python -m evals.run_eval                       # whole dataset
    uv run python -m evals.run_eval --only similar_christie,author_austen
    uv run python -m evals.run_eval --category similar --concurrency 2
    uv run python -m evals.run_eval --baseline evals/results/2026-10-29T14-00.json
    uv run python -m evals.run_eval --check               # validate the dataset offline
    uv run python -m evals.run_eval --refresh-catalogue   # re-snapshot the Pinecone index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from backend.app.schemas import ChatMessage  # noqa: E402
from backend.app.services.agent import answer, plan, search, similar  # noqa: E402
from backend.app.services.agent.conversation import BOOK_LINK  # noqa: E402
from backend.app.services.book_agent import book_agent_graph  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
DATASET = EVALS_DIR / "dataset.json"
CATALOGUE = EVALS_DIR / "catalogue.json"
RESULTS_DIR = EVALS_DIR / "results"

K = 5  # the agent recommends at most five books
RETRIEVAL_K = 10
CASE_TIMEOUT = 90
UNKNOWN_REFERENCE_PREFIX = similar.UNKNOWN_REFERENCE_REPLY.split("{")[0]
RELAXED_PREFIX = answer.RELAXED_NOTICE.split("{")[0]


# ----------------------------------------------------------------------------- dataset


def load_catalogue() -> dict[str, dict]:
    return json.loads(CATALOGUE.read_text())


def where(catalogue: dict[str, dict], rule: dict[str, list[str]] | None) -> set[str]:
    """Ids whose metadata matches every key of the rule (any of the listed values)."""
    if not rule:
        return set()
    return {
        book_id
        for book_id, meta in catalogue.items()
        if all(set(meta.get(key, [])) & set(values) for key, values in rule.items())
    }


def gains(case: dict, catalogue: dict[str, dict]) -> dict[str, int]:
    """Relevance grade per book: 2 for ideal picks, 1 for acceptable ones."""
    graded = {book_id: 1 for book_id in case.get("relevant", [])}
    graded |= {book_id: 1 for book_id in where(catalogue, case.get("relevant_where"))}
    graded |= {book_id: 2 for book_id in case.get("ideal", [])}
    return graded


def forbidden_ids(case: dict, catalogue: dict[str, dict]) -> set[str]:
    return set(case.get("forbidden", [])) | where(catalogue, case.get("forbidden_where"))


def expected_kinds(case: dict) -> set[str]:
    expect = case.get("expect", "books")
    return set(expect) if isinstance(expect, list) else {expect}


def check_dataset(cases: list[dict], catalogue: dict[str, dict]) -> list[str]:
    problems = []
    seen = set()
    for case in cases:
        if case["id"] in seen:
            problems.append(f"{case['id']}: duplicate id")
        seen.add(case["id"])
        for key in ("ideal", "relevant", "forbidden"):
            for book_id in case.get(key, []):
                if book_id not in catalogue:
                    problems.append(f"{case['id']}: {key} id {book_id} not in catalogue")
        for key in ("relevant_where", "forbidden_where"):
            if case.get(key) and not where(catalogue, case[key]):
                problems.append(f"{case['id']}: {key} {case[key]} matches no book")
        graded = gains(case, catalogue)
        if "books" in expected_kinds(case) and not graded and case["category"] != "behaviour":
            problems.append(f"{case['id']}: expects books but defines no relevant ones")
        if graded.keys() & forbidden_ids(case, catalogue):
            problems.append(f"{case['id']}: a book is both relevant and forbidden")
    return problems


# ----------------------------------------------------------------------------- running


async def run_agent(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    """The reply text and the ids of the candidates that reached the answer step."""
    chunks: list[str] = []
    final: dict[str, Any] = {}
    stream = book_agent_graph.astream({"messages": messages}, stream_mode=["custom", "values"])
    async for mode, data in stream:
        if mode == "custom":
            chunks.append(data)
        else:
            final = data
    candidates = [book.id for book in final.get("books") or []]
    return "".join(chunks), candidates


def classify(reply: str, candidates: list[str]) -> str:
    """Which fixed reply (or kind of answer) the agent produced."""
    fixed = {
        "clarify": plan.CLARIFY_REPLY,
        "off_topic": plan.FIXED_REPLIES["off_topic"],
        "malicious": plan.FIXED_REPLIES["malicious"],
        "too_long": plan.TOO_LONG_REPLY,
        "no_results": answer.NO_RESULTS_REPLY,
        "already_shown": search.ALREADY_SHOWN_REPLY,
        "unknown_reference": UNKNOWN_REFERENCE_PREFIX,
    }
    for kind, text in fixed.items():
        if text in reply:
            return kind
    if "Sorry," in reply[:200]:
        return "error"
    if RELAXED_PREFIX in reply:
        return "relaxed"
    if BOOK_LINK.search(reply):
        return "books"
    # Free text without links: before any search it is the plan step's own clarifying
    # question; after one it is the answer step saying that nothing fits.
    return "no_results" if candidates else "clarify"


def picked_ids(reply: str) -> list[str]:
    seen: dict[str, None] = {}
    for number in BOOK_LINK.findall(reply):
        seen.setdefault(f"gutenberg:{number}", None)
    return list(seen)


# ----------------------------------------------------------------------------- metrics


def dcg(grades: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg(picks: list[str], graded: dict[str, int], k: int) -> float:
    ideal = sorted(graded.values(), reverse=True)[:k]
    if not ideal:
        return 0.0
    return dcg([graded.get(p, 0) for p in picks[:k]]) / dcg(ideal)


def score(case: dict, reply: str, candidates: list[str], catalogue: dict, pass_ndcg: float):
    graded = gains(case, catalogue)
    forbidden = forbidden_ids(case, catalogue)
    picks = picked_ids(reply)
    kind = classify(reply, candidates)
    hits = [p for p in picks[:K] if p in graded]
    result = {
        "kind": kind,
        "behaviour_ok": kind in expected_kinds(case),
        "picks": picks,
        "forbidden_picks": [p for p in picks if p in forbidden],
        "unknown_picks": [p for p in picks if p not in catalogue],
        "candidates": candidates,
    }
    if graded:
        first_hit = next((i for i, p in enumerate(picks[:K], start=1) if p in graded), None)
        result |= {
            "ndcg": ndcg(picks, graded, K),
            "precision": len(hits) / len(picks[:K]) if picks else 0.0,
            "recall": len(hits) / min(len(graded), K),
            "mrr": 1 / first_hit if first_hit else 0.0,
            "retrieval_recall": len(set(candidates[:RETRIEVAL_K]) & graded.keys())
            / min(len(graded), RETRIEVAL_K),
        }
    result["passed"] = (
        result["behaviour_ok"]
        and not result["forbidden_picks"]
        and not result["unknown_picks"]
        and (not graded or result["ndcg"] >= pass_ndcg)
    )
    return result


# ----------------------------------------------------------------------------- reporting


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: float | None, width: int = 5) -> str:
    return f"{value:{width}.2f}" if value is not None else " " * (width - 1) + "-"


def summarize(results: list[dict]) -> dict[str, Any]:
    scored = [r for r in results if "ndcg" in r]
    latencies = sorted(r["seconds"] for r in results)
    return {
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": mean([float(r["passed"]) for r in results]),
        "behaviour_accuracy": mean([float(r["behaviour_ok"]) for r in results]),
        "ndcg": mean([r["ndcg"] for r in scored]),
        "precision": mean([r["precision"] for r in scored]),
        "recall": mean([r["recall"] for r in scored]),
        "mrr": mean([r["mrr"] for r in scored]),
        "retrieval_recall": mean([r["retrieval_recall"] for r in scored]),
        "forbidden_picks": sum(len(r["forbidden_picks"]) for r in results),
        "unknown_picks": sum(len(r["unknown_picks"]) for r in results),
        "errors": sum(r["kind"] == "error" for r in results),
        "latency_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
    }


def title(book_id: str, catalogue: dict) -> str:
    meta = catalogue.get(book_id)
    return meta["title"][:40] if meta else f"<not in catalogue: {book_id}>"


def print_case(result: dict, catalogue: dict, verbose: bool) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    line = (
        f"{status} {result['id']:<38} {result['category']:<16} "
        f"nDCG {fmt(result.get('ndcg'))}  P@5 {fmt(result.get('precision'))}  "
        f"R@10 {fmt(result.get('retrieval_recall'))}  {result['seconds']:5.1f}s  {result['kind']}"
    )
    print(line)
    if not result["passed"] or verbose:
        graded = result["graded"]
        for pick in result["picks"]:
            mark = {2: "++", 1: " +"}.get(graded.get(pick), " -")
            flag = " FORBIDDEN" if pick in result["forbidden_picks"] else ""
            flag += " UNKNOWN" if pick in result["unknown_picks"] else ""
            print(f"      {mark} {title(pick, catalogue)}{flag}")
        if not result["behaviour_ok"]:
            print(
                f"      expected {result['expected']}, got {result['kind']}: "
                f"{result['reply'][:160]!r}"
            )


def print_summary(summary: dict, by_category: dict[str, dict]) -> None:
    print("\nBy category")
    print(f"  {'category':<17}{'cases':>6}{'pass':>6}{'nDCG@5':>8}{'P@5':>7}{'R@10':>7}")
    for name, s in sorted(by_category.items()):
        print(
            f"  {name:<17}{s['cases']:>6}{s['passed']:>6}{fmt(s['ndcg'], 8)}"
            f"{fmt(s['precision'], 7)}{fmt(s['retrieval_recall'], 7)}"
        )
    print("\nOverall")
    print(
        f"  pass rate          {summary['pass_rate']:.2f}  ({summary['passed']}/{summary['cases']})"
    )
    print(f"  nDCG@5             {fmt(summary['ndcg'])}")
    print(f"  precision@5        {fmt(summary['precision'])}")
    print(f"  recall@5           {fmt(summary['recall'])}")
    print(f"  MRR                {fmt(summary['mrr'])}")
    print(f"  retrieval R@10     {fmt(summary['retrieval_recall'])}")
    print(f"  behaviour accuracy {fmt(summary['behaviour_accuracy'])}")
    print(f"  forbidden picks    {summary['forbidden_picks']}")
    print(f"  hallucinated ids   {summary['unknown_picks']}")
    print(f"  errors/timeouts    {summary['errors']}")
    print(f"  latency p50 / p95  {summary['latency_p50']:.1f}s / {summary['latency_p95']:.1f}s")


def print_comparison(results: list[dict], summary: dict, baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    before = {r["id"]: r for r in baseline["results"]}
    print(f"\nCompared with {baseline_path.name}")
    for key in ("pass_rate", "ndcg", "precision", "retrieval_recall", "behaviour_accuracy"):
        old, new = baseline["summary"].get(key), summary.get(key)
        if old is not None and new is not None:
            print(f"  {key:<19}{old:.2f} -> {new:.2f}  ({new - old:+.2f})")
    regressions = [
        r["id"]
        for r in results
        if r["id"] in before and before[r["id"]]["passed"] and not r["passed"]
    ]
    fixes = [
        r["id"]
        for r in results
        if r["id"] in before and not before[r["id"]]["passed"] and r["passed"]
    ]
    print(f"  regressions: {', '.join(regressions) or 'none'}")
    print(f"  fixed:       {', '.join(fixes) or 'none'}")


# ----------------------------------------------------------------------------- main


async def evaluate(
    cases: list[dict], catalogue: dict, concurrency: int, pass_ndcg: float, verbose: bool
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def one(case: dict) -> None:
        messages = [ChatMessage(**m) for m in case["messages"]]
        async with semaphore:
            started = time.monotonic()
            try:
                reply, candidates = await asyncio.wait_for(run_agent(messages), CASE_TIMEOUT)
            except Exception as exc:  # the eval must finish; the case is scored as an error
                reply, candidates = f"Sorry, eval error: {type(exc).__name__}: {exc}", []
            seconds = time.monotonic() - started
        result = {
            "id": case["id"],
            "category": case["category"],
            "expected": sorted(expected_kinds(case)),
            "seconds": seconds,
            "reply": reply,
            "graded": gains(case, catalogue),
            **score(case, reply, candidates, catalogue, pass_ndcg),
        }
        results.append(result)
        print_case(result, catalogue, verbose)

    await asyncio.gather(*(one(case) for case in cases))
    return sorted(results, key=lambda r: [c["id"] for c in cases].index(r["id"]))


def refresh_catalogue() -> None:
    from backend.app.config import get_settings
    from backend.app.services.pinecone_utils import get_index

    index, namespace = get_index(), get_settings().pinecone_namespace
    ids = [str(getattr(i, "id", i)) for page in index.list(namespace=namespace) for i in page]
    snapshot = {}
    for start in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[start : start + 100], namespace=namespace)
        for book_id, vector in fetched.vectors.items():
            meta = vector.metadata or {}
            snapshot[book_id] = {
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "languages": meta.get("languages", []),
                "download_count": int(float(meta.get("download_count", 0))),
            }
    CATALOGUE.write_text(json.dumps(dict(sorted(snapshot.items())), indent=1, ensure_ascii=False))
    print(f"{len(snapshot)} books written to {CATALOGUE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--category", help="run one category only")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--pass-ndcg",
        type=float,
        default=0.5,
        help="minimum nDCG@5 for a scored case to pass (default 0.5)",
    )
    parser.add_argument("--baseline", type=Path, help="earlier results file to compare with")
    parser.add_argument("--out", type=Path, help="where to write results (default evals/results/)")
    parser.add_argument("--verbose", action="store_true", help="print picks for passing cases too")
    parser.add_argument("--check", action="store_true", help="validate the dataset and exit")
    parser.add_argument("--refresh-catalogue", action="store_true")
    args = parser.parse_args()

    if args.refresh_catalogue:
        refresh_catalogue()
    catalogue = load_catalogue()
    cases = json.loads(DATASET.read_text())
    problems = check_dataset(cases, catalogue)
    for problem in problems:
        print("dataset:", problem)
    if args.check or args.refresh_catalogue:
        print(f"{len(cases)} cases, {len(catalogue)} books, {len(problems)} problems")
        return 1 if problems else 0

    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if not cases:
        print("no cases selected")
        return 1

    results = asyncio.run(
        evaluate(cases, catalogue, args.concurrency, args.pass_ndcg, args.verbose)
    )
    summary = summarize(results)
    by_category = {
        name: summarize([r for r in results if r["category"] == name])
        for name in {r["category"] for r in results}
    }
    print_summary(summary, by_category)
    if args.baseline:
        print_comparison(results, summary, args.baseline)

    out = args.out or RESULTS_DIR / f"{datetime.now():%Y-%m-%dT%H-%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"summary": summary, "by_category": by_category, "results": results},
            indent=1,
            ensure_ascii=False,
        )
    )
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
