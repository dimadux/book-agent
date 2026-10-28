from __future__ import annotations

import os
from functools import lru_cache

from openai import AsyncOpenAI


@lru_cache
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI()


def get_llm_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
