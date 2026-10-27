# Frontend

Static chat UI. No build step and no npm dependencies.

The UI is served by FastAPI from:

```text
http://127.0.0.1:8000
```

It calls:

```text
POST /api/chat
```

and renders streamed SSE chunks from the backend.
