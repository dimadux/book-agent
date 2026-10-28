from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from backend.app.schemas import ChatMessage
from backend.app.services.agent.answer import answer_node
from backend.app.services.agent.filters import extract_filters_node
from backend.app.services.agent.intent import intent_node
from backend.app.services.agent.rerank import rerank_node
from backend.app.services.agent.search import search_node
from backend.app.services.agent.similar import similar_node
from backend.app.services.agent.state import AgentState


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


async def join_node(state: AgentState) -> AgentState:
    # Waits for intent and extract_filters, which run in parallel to save one LLM round trip.
    return {}


def route_after_join(state: AgentState) -> str:
    if state.get("intent") != "book_request":
        return "reply"
    plan = state.get("plan")
    # "Books like X": look X up first and search for what it is like.
    if plan and (plan.similar_to_authors or plan.similar_to_titles):
        return "similar"
    return "search"


def route_after_similar(state: AgentState) -> str:
    # similar_node only sets a reply when none of the references is in the catalogue.
    return "reply" if state.get("reply") else "search"


def route_after_search(state: AgentState) -> str:
    # search_node only sets a reply when the catalogue could not be queried.
    return "reply" if state.get("reply") else "rerank"


async def reply_node(state: AgentState) -> AgentState:
    get_stream_writer()(state["reply"])
    return {}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("extract_filters", extract_filters_node)
    graph.add_node("join", join_node)
    graph.add_node("similar", similar_node)
    graph.add_node("search", search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reply", reply_node)

    graph.add_edge(START, "intent")
    graph.add_edge(START, "extract_filters")
    # A list of sources makes join wait until both branches have finished.
    graph.add_edge(["intent", "extract_filters"], "join")
    graph.add_conditional_edges("join", route_after_join, ["similar", "search", "reply"])
    graph.add_conditional_edges("similar", route_after_similar, ["search", "reply"])
    graph.add_conditional_edges("search", route_after_search, ["rerank", "reply"])
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("reply", END)
    return graph.compile()


book_agent_graph = build_graph()


async def stream_book_agent_response(messages: list[ChatMessage]) -> AsyncIterator[str]:
    """Run the LangGraph pipeline and stream the text chunks its nodes write."""
    async for chunk in book_agent_graph.astream({"messages": messages}, stream_mode="custom"):
        yield chunk
