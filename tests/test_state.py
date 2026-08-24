"""Tests for state definitions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import get_type_hints

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from state import SQLAgentState


class StateTestCase(unittest.TestCase):
    """Test state type definitions."""

    def test_sql_agent_state_is_typeddict(self) -> None:
        """SQLAgentState should be a TypedDict."""

        self.assertTrue(hasattr(SQLAgentState, "__total__"))

        hints = get_type_hints(SQLAgentState)
        self.assertIn("question", hints)
        self.assertIn("analysis", hints)
        self.assertIn("metadata", hints)

    def test_sql_agent_state_optional_fields(self) -> None:
        """All SQLAgentState fields should be optional."""

        self.assertFalse(SQLAgentState.__total__)

    def test_sql_agent_state_includes_agent_loop_keys(self) -> None:
        """SQLAgentState should carry the messages transcript and its budget/status keys."""
        hints = get_type_hints(SQLAgentState)

        for key in (
            "messages",
            "agent_status",
            "agent_iterations",
            "max_agent_iterations",
            "probe_count",
            "max_probes",
            "clarification_rounds",
            "max_clarifications",
            "clarification_answers",
            "agent_rationale",
            "agent_error",
        ):
            self.assertIn(key, hints)

    def test_sql_agent_state_preserves_fallback_repair_keys(self) -> None:
        """The keys the approval-dialog diff UI reads must survive this change (D6)."""
        hints = get_type_hints(SQLAgentState)

        for key in (
            "retry_count",
            "max_retries",
            "last_execution_error",
            "previous_sql",
            "regeneration_explanation",
        ):
            self.assertIn(key, hints)

    def test_messages_field_appends_across_graph_steps_instead_of_overwriting(self) -> None:
        """The add_messages reducer must accumulate messages, not replace them.

        This is the behaviour the whole agent loop depends on (D1): without it,
        each node's update would overwrite the transcript instead of extending it.
        """

        def node_a(state: SQLAgentState) -> dict:
            return {"messages": [HumanMessage(content="hi")]}

        def node_b(state: SQLAgentState) -> dict:
            return {"messages": [AIMessage(content="hello")]}

        builder = StateGraph(SQLAgentState)
        builder.add_node("a", node_a)
        builder.add_node("b", node_b)
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        graph = builder.compile()

        result = graph.invoke({})

        self.assertEqual(len(result["messages"]), 2)
        self.assertIsInstance(result["messages"][0], HumanMessage)
        self.assertIsInstance(result["messages"][1], AIMessage)


if __name__ == "__main__":
    unittest.main()
