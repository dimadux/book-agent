# Book Recommendation Agent Starter

Starter project for a streamed book recommendation agent over a prepared Pinecone `books` index.

The repository contains:

- `backend/` — FastAPI app with a streamed `/api/chat` endpoint.
- `frontend/` — static chat UI served by FastAPI (already done, do not modify).
- `scripts/` — utility scripts (ingestion, verification).

## What's already done (do NOT touch)

- Frontend chat UI with streaming support and markdown rendering.
- FastAPI server with SSE streaming protocol.
- Pinecone index filled with ~1000+ books (metadata, embeddings).
- `search_books()` helper that handles embedding + filtered search.
- Dev server with hot-reload.

## Your task

Implement the book recommendation agent in **one file**:

```text
backend/app/services/book_agent.py
```

## Quick start

```bash
make setup        # creates venv, installs deps, copies .env
# Edit .env — set PINECONE_API_KEY (provided to you) and your LLM key
make verify       # checks Python, deps, Pinecone connectivity
make run          # starts server at http://127.0.0.1:8000
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env
python scripts/verify_setup.py
python scripts/run_dev.py
```

**Requirements:** Python 3.11+

## Candidate task

Implement a book recommendation agent that:

- accepts natural-language user requests;
- extracts structured filters (author, language, genre/topic, year) where possible;
- searches the Pinecone `books` index via the provided `search_books()`;
- streams a concise answer with recommended books and reasons;
- never recommends books that were not retrieved from the index.

Example queries:

```text
Recommend a dark classic book about obsession and revenge
Find books by Shakespeare
Suggest popular adventure books for teenagers
Find English books about ghosts or supernatural horror
What are some philosophical novels written before 1900?
I want something like Frankenstein — gothic, science, moral dilemmas
Recommend a short book about survival in the wilderness
Find books about political revolution in French
What did Mark Twain write about travel?
Suggest a book for someone who loved "The Count of Monte Cristo"
```

### LLM provider

Choose any LLM and install it yourself:

```bash
pip install openai           # OpenAI GPT-4o, GPT-4.1, etc.
pip install anthropic        # Claude
pip install ollama           # local models via Ollama
```

Add the API key to `.env`. The starter code does not assume any particular model.

### Reranking

Optional — use a dedicated reranker model, score-based filtering, or skip it entirely.

### Pinecone helper

`backend/app/services/pinecone_utils.py` provides a ready-to-use `search_books()` function. **Read its docstring** — it documents all available metadata fields and filter examples.

### Streaming protocol

Your function must be an `AsyncIterator[str]` that yields text chunks. The server wraps them into SSE automatically. See the existing stub in `book_agent.py`.

## Submission

1. Create a **new public repository** on GitHub.
2. Push the contents of this archive as the **initial commit** (no changes).
3. Implement your solution and push to `main`.
4. Share the repository link with us.
