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

T = TypeVar("T", bound=BaseModel)


@lru_cache
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI()


def get_llm_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


async def structured(instructions: str, messages: list[ChatMessage], schema: type[T]) -> T | None:
    """One structured-output call. None when the call fails or the model refuses (both logged)."""
    try:
        response = await get_llm_client().responses.parse(
            model=get_llm_model(),
            instructions=instructions,
            input=recent_conversation(messages),
            text_format=schema,
        )
    except Exception:
        logger.exception("%s call failed", schema.__name__)
        return None
    if response.output_parsed is None:
        logger.warning("%s call returned no output", schema.__name__)
    return response.output_parsed
