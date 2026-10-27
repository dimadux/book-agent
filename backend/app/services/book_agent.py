from collections.abc import AsyncIterator

from backend.app.schemas import ChatMessage


async def stream_book_agent_response(messages: list[ChatMessage]) -> AsyncIterator[str]:
    """
    Candidate task entrypoint.

    Replace this stub with a LangGraph graph that:
    1. classifies user intent / extracts structured filters;
    2. searches the Pinecone `books` index via pinecone_utils.search_books();
    3. optionally reranks results;
    4. streams a final answer with cited book recommendations.

    LangGraph is required — implement the pipeline as a StateGraph.
    """
    last_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    )

    chunks = [
        "This is a stubbed streaming response.\n\n",
        "Implement the book recommendation agent in ",
        "`backend/app/services/book_agent.py`.\n\n",
        f"Received query: {last_user_message}",
    ]
    for chunk in chunks:
        yield chunk
