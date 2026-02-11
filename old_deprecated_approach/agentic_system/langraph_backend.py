"""
LangGraph-powered chat backend for the dashboard.

This module exposes a minimal graph that can later be upgraded with
LLM-backed tools. For now it applies a simple heuristic: if the user's
latest message length exceeds five characters we respond with "long",
otherwise with "short".
"""

from __future__ import annotations

from typing import List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from config import load_llm

llm = load_llm()


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatState(TypedDict):
    messages: List[ChatMessage]


def _length_router(state: ChatState) -> ChatState:
    """Simple rule-based response generator."""
    history = state["messages"]
    if not history:
        return {"messages": history}

    # Find the most recent user message.
    latest_user = next(
        (msg for msg in reversed(history) if msg["role"] == "user"), None
    )
    if latest_user is None:
        return {"messages": history}

    content = latest_user["content"].strip()
    reply_text = "long" if len(content) > 5 else "short"

    updated_history = history + [{"role": "assistant", "content": reply_text}]
    return {"messages": updated_history}


graph_builder = StateGraph(ChatState)
graph_builder.add_node("length_router", _length_router)
graph_builder.add_edge(START, "length_router")
graph_builder.add_edge("length_router", END)

chat_graph = graph_builder.compile()


def chat_step(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Run a single chat step through the graph.

    Args:
        messages: Conversation history with the latest user message appended.

    Returns:
        Updated history including the assistant's response.
    """

    result = chat_graph.invoke({"messages": messages})
    return result["messages"]
