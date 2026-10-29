"""The book recommendation agent: a LangGraph pipeline that streams markdown to the chat UI.

    plan ─► similar ─► search ─► rerank ─► answer ─► END
      │        │          │
      └────────┴──────────┴─► reply ─► END

`reply` writes a fixed text for turns that end early: refusals, clarifying questions,
references not in the catalogue, "nothing new to show", errors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from backend.app.schemas import ChatMessage
from backend.app.services.agent.answer import answer_node
from backend.app.services.agent.models import AgentState
from backend.app.services.agent.plan import plan_node
from backend.app.services.agent.rerank import rerank_node
from backend.app.services.agent.search import search_node
from backend.app.services.agent.similar import similar_node


def configure_logging() -> None:
    """Show the agent's logs (intent, search plan, filters, rerank) next to uvicorn's.

    Uvicorn configures only its own loggers, so without this our INFO logs are dropped.
    Set LOG_LEVEL (e.g. DEBUG, WARNING) to change the level.
    """
    agent_logger = logging.getLogger("backend.app.services")
    if agent_logger.handlers:
        return  # already configured, e.g. when the module is reloaded
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    agent_logger.addHandler(handler)
    agent_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    agent_logger.propagate = False  # avoid printing twice if the root logger is configured


configure_logging()

logger = logging.getLogger(__name__)

# Wall-clock budget for one chat turn, all steps included.
TURN_TIMEOUT_SECONDS = 45
TIMEOUT_REPLY = "Sorry, that took too long. Please try again."
ERROR_REPLY = "Sorry, something went wrong on my side. Please try again in a moment."


async def reply_node(state: AgentState) -> AgentState:
    get_stream_writer()(state["reply"])
    return {}


def route_after_plan(state: AgentState) -> str:
    if state.get("reply"):
        return "reply"
    plan = state["plan"]
    # "Books like X": look X up first and search for what it is like.
    return "similar" if plan.similar_to_authors or plan.similar_to_titles else "search"


def unless_replied(next_node: str):
    """Router for nodes that end the turn by setting `reply`, else continue to `next_node`."""
    return lambda state: "reply" if state.get("reply") else next_node


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("similar", similar_node)
    graph.add_node("search", search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reply", reply_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, ["similar", "search", "reply"])
    graph.add_conditional_edges("similar", unless_replied("search"), ["search", "reply"])
    graph.add_conditional_edges("search", unless_replied("rerank"), ["rerank", "reply"])
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("reply", END)
    return graph.compile()


book_agent_graph = build_graph()


async def stream_book_agent_response(messages: list[ChatMessage]) -> AsyncIterator[str]:
    """Run the LangGraph pipeline and stream the text chunks its nodes write.

    A turn that overruns its budget or hits an unexpected error ends with a fixed message
    instead of a broken stream, so the client always gets a complete reply.
    """
    chunks = book_agent_graph.astream({"messages": messages}, stream_mode="custom")
    deadline = time.monotonic() + TURN_TIMEOUT_SECONDS
    streamed = False
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(anext(chunks), deadline - time.monotonic())
            except StopAsyncIteration:
                return
            streamed = True
            yield chunk
    except TimeoutError:
        logger.warning("Turn timed out after %ss", TURN_TIMEOUT_SECONDS)
        yield f"\n\n{TIMEOUT_REPLY}" if streamed else TIMEOUT_REPLY
    except Exception:
        logger.exception("Agent failed")
        yield f"\n\n{ERROR_REPLY}" if streamed else ERROR_REPLY
    finally:
        await chunks.aclose()
