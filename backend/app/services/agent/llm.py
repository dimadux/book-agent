"""The LLM client and the call shape every step uses: structured output over the conversation."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.app.schemas import ChatMessage
from backend.app.services.agent.conversation import recent_conversation

logger = logging.getLogger(__name__)

# Per attempt; the SDK retries once more on connection errors, 429 and 5xx. For the streamed
# answer it is the longest allowed silence between two chunks.
LLM_TIMEOUT_SECONDS = 20
# Classification and extraction should give the same plan for the same request.
TEMPERATURE = 0
DEFAULT_FALLBACK_MODEL = "gpt-4.1"

T = TypeVar("T", bound=BaseModel)


@lru_cache
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=LLM_TIMEOUT_SECONDS, max_retries=1)


def get_llm_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def llm_models() -> list[str]:
    """The model to use, then the one to fall back to when it fails (OPENAI_FALLBACK_MODEL,
    empty to disable)."""
    primary = get_llm_model()
    fallback = os.getenv("OPENAI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
    return [primary] if fallback in ("", primary) else [primary, fallback]


async def structured(instructions: str, messages: list[ChatMessage], schema: type[T]) -> T | None:
    """One structured-output call, tried once more on the fallback model when the primary
    fails. None when every model fails or the model refuses (all logged)."""
    for model in llm_models():
        try:
            response = await get_llm_client().responses.parse(
                model=model,
                instructions=instructions,
                input=recent_conversation(messages),
                text_format=schema,
                temperature=TEMPERATURE,
            )
        except Exception:
            logger.exception("%s call failed on %s", schema.__name__, model)
            continue
        if response.output_parsed is None:
            logger.warning("%s call returned no output", schema.__name__)
        return response.output_parsed
    return None
