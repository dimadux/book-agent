# Backend

FastAPI app for the book recommendation agent.

Main files:

- `app/main.py` — API routes and streaming response protocol.
- `app/schemas.py` — Pydantic request/response models.
- `app/services/book_agent.py` — candidate implementation entrypoint.

Run from repository root:

```bash
uvicorn backend.app.main:app --reload
```

Streaming protocol:

```text
POST /api/chat
Content-Type: application/json
Accept: text/event-stream
```

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Recommend a dark classic book about revenge"
    }
  ]
}
```

Each streamed event is JSON inside an SSE `data:` frame:

```json
{"type":"content","content":"..."}
{"type":"done","content":""}
```
