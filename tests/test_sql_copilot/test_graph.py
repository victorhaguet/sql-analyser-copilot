"""Tests for LangGraph wiring and orchestration."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from sql_copilot.tools.database import SQLiteDatabase, register_database


class FakeResponse:
    """Simple model response object with a content field."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """Minimal test double for invoke-based text models."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> FakeResponse:
        del prompt
        return FakeResponse(self.response)


class FakeCompiledGraph:
    """Small in-memory executor that mimics the compiled graph API."""

    def __init__(self, builder: "FakeStateGraph") -> None:
        self.builder = builder

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        current = self.builder.start_target
        state = dict(state)

        while current and current != "__end__":
            node: object = self.builder.nodes[current]
            state.update(node(state))  # type: ignore[operator]
            if current in self.builder.conditional_edges:
                router, mapping = self.builder.conditional_edges[current]
                current = mapping[router(state)]  # type: ignore[operator]
            else:
                current = self.builder.edges.get(current)
        return state


class FakeStateGraph:
    """Small StateGraph stand-in used to test wiring without langgraph installed."""

    def __init__(self, _state_type: object) -> None:
        self.nodes: dict[str, object] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[object, dict[str, str]]] = {}
        self.start_target: str | None = None

    def add_node(self, name: str, node: object) -> None:
        self.nodes[name] = node

    def add_edge(self, source: str, target: str) -> None:
        if source == "__start__":
            self.start_target = target
        else:
            self.edges[source] = target

    def add_conditional_edges(
        self,
        source: str,
        router: object,
        mapping: dict[str, str],
    ) -> None:
        self.conditional_edges[source] = (router, mapping)

    def compile(self) -> FakeCompiledGraph:
        return FakeCompiledGraph(self)


class SQLGraphTestCase(unittest.TestCase):
    """Test graph-specific behavior."""

    def setUp(self) -> None:
        self._original_modules = {
            name: sys.modules.get(name)
            for name in ("langgraph", "langgraph.graph")
        }
        langgraph_module = types.ModuleType("langgraph")
        graph_module = types.ModuleType("langgraph.graph")
        setattr(graph_module, "START", "__start__")
        setattr(graph_module, "END", "__end__")
        setattr(graph_module, "StateGraph", FakeStateGraph)
        sys.modules["langgraph"] = langgraph_module
        sys.modules["langgraph.graph"] = graph_module

    def tearDown(self) -> None:
        for name, module in self._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_build_graph_stops_when_validation_fails(self) -> None:
        """Test that the graph correctly halts execution when SQL validation fails."""
        from sql_copilot.graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("DELETE FROM Artist"),
            databases=[register_database(SQLiteDatabase())],
        )
        result = graph.invoke({"question": "Delete artists"})
        self.assertIn("sql_validation_error", result)
        self.assertNotIn("query_result", result)

    def test_build_graph_stops_when_no_database_matches(self) -> None:
        """Questions outside the catalog should abort before SQL generation."""
        from sql_copilot.graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "No configured database matches."}'
            ),
            databases=[
                register_database(SQLiteDatabase(), name="music", description="Music data"),
                register_database(SQLiteDatabase(), name="sales", description="Sales data"),
            ],
        )
        result = graph.invoke({"question": "What is the weather in Berlin?"})
        self.assertIn("execution_error", result)
        self.assertNotIn("generated_sql", result)

    def test_build_graph_stops_when_database_selection_is_ambiguous(self) -> None:
        """Questions that fit multiple databases should abort before SQL generation."""
        from sql_copilot.graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": ["music", "sales"], "reason": "The question could be answered from either catalog."}'
            ),
            databases=[
                register_database(SQLiteDatabase(), name="music", description="Music data"),
                register_database(SQLiteDatabase(), name="sales", description="Sales data"),
            ],
        )
        result = graph.invoke({"question": "Show me recent invoice activity."})
        self.assertTrue(result["metadata"]["database_selection_ambiguous"])
        self.assertNotIn("generated_sql", result)


if __name__ == "__main__":
    unittest.main()
