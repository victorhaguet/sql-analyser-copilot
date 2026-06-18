"""Public package exports."""

from __future__ import annotations

from typing import Any

__all__ = ["answer_question", "build_sql_agent_graph", "stream_question"]


def __getattr__(name: str) -> Any:
    """Load graph-related exports lazily so non-graph modules import cleanly."""
    if name == "build_sql_agent_graph":
        from graph import build_sql_agent_graph

        return build_sql_agent_graph
    if name == "answer_question":
        from main import answer_question

        return answer_question
    if name == "stream_question":
        from main import stream_question

        return stream_question
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
