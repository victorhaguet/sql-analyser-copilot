"""Public package exports."""

from __future__ import annotations

from typing import Any

__all__ = ["answer_question", "build_sql_agent_graph"]


def __getattr__(name: str) -> Any:
    """Load graph-related exports lazily so non-graph modules import cleanly."""
    if name == "build_sql_agent_graph":
        from sql_copilot.graph import build_sql_agent_graph

        return build_sql_agent_graph
    if name == "answer_question":
        from sql_copilot.main import answer_question

        return answer_question
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
