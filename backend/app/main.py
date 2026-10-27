from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.schemas import ChatRequest, StreamChunk
from backend.app.services.book_agent import stream_book_agent_response

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Book Recommendation Agent Starter")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        async for content in stream_book_agent_response(request.messages):
            chunk = StreamChunk(type="content", content=content)
            yield f"data: {chunk.model_dump_json()}\n\n"

        done = json.dumps({"type": "done", "content": ""})
        yield f"data: {done}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
