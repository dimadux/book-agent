"""Quick sanity check that the project is configured correctly."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

MIN_PYTHON = (3, 11)


def main() -> None:
    errors: list[str] = []

    # 1. Python version
    if sys.version_info < MIN_PYTHON:
        errors.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )
        # Can't continue with wrong Python
        _report(errors)
        return

    # 2. Key imports
    try:
        import fastapi  # noqa: F401
        import langgraph  # noqa: F401
        import langchain_core  # noqa: F401
    except ImportError as exc:
        errors.append(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt")

    # 3. .env and Pinecone key
    try:
        from backend.app.config import get_settings

        settings = get_settings()
        if not settings.pinecone_api_key or settings.pinecone_api_key.startswith("your-"):
            errors.append(
                "PINECONE_API_KEY is not configured. Edit .env and set a real key."
            )
    except Exception as exc:
        errors.append(f"Failed to load settings: {exc}")

    # 4. Pinecone connectivity
    if not errors:
        try:
            from backend.app.services.pinecone_utils import search_books

            results = search_books(query_text="test", top_k=1)
            if not results:
                errors.append(
                    "Pinecone returned 0 results. The index may be empty or "
                    "the namespace is wrong. Check PINECONE_INDEX_NAME and PINECONE_NAMESPACE."
                )
            else:
                print(f"  Pinecone OK — sample result: {results[0]['metadata'].get('title')}")
        except Exception as exc:
            errors.append(f"Pinecone search failed: {exc}")

    _report(errors)


def _report(errors: list[str]) -> None:
    if errors:
        print("\n❌ Setup issues found:\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print()
        sys.exit(1)
    else:
        print("\n✅ All checks passed. You're ready to code!\n")
        print("  Start the server:  make run")
        print("  Then open:         http://127.0.0.1:8000")
        print("  Edit:              backend/app/services/book_agent.py\n")


if __name__ == "__main__":
    main()
