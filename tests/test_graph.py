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

from tools.database import SQLiteDatabase, register_database
from graph import _normalize_stream_event
from state import SQLAgentState
from graph import _summarize_step_outcome

SQLAgentStateUpdate = SQLAgentState


class TestNormalizeStreamEvent(unittest.TestCase):
    """Test the _normalize_stream_event function."""

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

    def test_normalize_stream_event_with_updates_mode(self) -> None:
        """Test normalization in updates stream mode."""
        

        event = {"node1": {"key": "value"}}
        current_state = SQLAgentState(question="test")
        node_name, update, _ = _normalize_stream_event(
            event, current_state, "updates"
        )
        self.assertEqual(node_name, "node1")
        self.assertEqual(update, {"key": "value"})

    def test_normalize_stream_event_with_values_mode(self) -> None:
        """Test normalization in values stream mode."""

        event = {"node1": {"key": "value"}}
        current_state = SQLAgentState(question="test")
        node_name, _, _ = _normalize_stream_event(
            event, current_state, "values"
        )
        self.assertEqual(node_name, "state")

    def test_normalize_stream_event_raises_type_error_for_invalid_updates(self) -> None:
        """Test that invalid updates raise TypeError."""
        event = "invalid"
        current_state = SQLAgentState(question="test")

        with self.assertRaises(TypeError):
            _normalize_stream_event(event, current_state, "updates")

    def test_normalize_stream_event_raises_type_error_for_invalid_values(self) -> None:
        """Test that invalid values raise TypeError."""
        event = ["invalid"]
        current_state = SQLAgentState(question="test")

        with self.assertRaises(TypeError):
            _normalize_stream_event(event, current_state, "values")

    def test_normalize_stream_event_raises_type_error_for_non_string_node_name(self) -> None:
        """Test that non-string node names raise TypeError."""
        event = {123: {"key": "value"}}
        current_state = SQLAgentState(question="test")

        with self.assertRaises(TypeError):
            _normalize_stream_event(event, current_state, "updates")

    def test_normalize_stream_event_raises_type_error_for_non_dict_update(self) -> None:
        """Test that non-dict updates raise TypeError."""
        event = {"node1": "not-a-dict"}
        current_state = SQLAgentState(question="test")

        with self.assertRaises(TypeError):
            _normalize_stream_event(event, current_state, "updates")


class TestSummarizeStepOutcome(unittest.TestCase):
    """Test the _summarize_step_outcome function."""

    def test_summarize_validation_failed(self) -> None:
        """Test summarization when SQL validation fails."""
        result = _summarize_step_outcome({"sql_validation_error": "syntax error"})
        self.assertEqual(result, "validation_failed")

    def test_summarize_execution_failed(self) -> None:
        """Test summarization when execution fails."""
        result = _summarize_step_outcome({"execution_error": "error"})
        self.assertEqual(result, "execution_failed")

    def test_summarize_database_selection_ambiguous(self) -> None:
        """Test summarization when database selection is ambiguous."""
        result = _summarize_step_outcome({
            "execution_error": "error",
            "metadata": {"database_selection_ambiguous": True},
        })
        self.assertEqual(result, "database_selection_ambiguous")

    def test_summarize_database_selection_failed(self) -> None:
        """Test summarization when database selection fails."""
        result = _summarize_step_outcome({
            "execution_error": "error",
            "metadata": {"database_selection_failed": True},
        })
        self.assertEqual(result, "database_selection_failed")

    def test_summarize_query_executed(self) -> None:
        """Test summarization when query is executed."""
        result = _summarize_step_outcome({"query_result": {}}) # type: ignore[typeddict-item]
        self.assertEqual(result, "query_executed")

    def test_summarize_analysis_ready(self) -> None:
        """Test summarization when analysis is ready."""
        result = _summarize_step_outcome({"analysis": "result"})
        self.assertEqual(result, "analysis_ready")

    def test_summarize_sql_generated(self) -> None:
        """Test summarization when SQL is generated."""
        result = _summarize_step_outcome({"generated_sql": "SELECT * FROM t"})
        self.assertEqual(result, "sql_generated")

    def test_summarize_database_selected(self) -> None:
        """Test summarization when database is selected."""
        result = _summarize_step_outcome({"selected_database": "db1"})# type: ignore[typeddict-item]
        self.assertEqual(result, "database_selected")

    def test_summarize_updated(self) -> None:
        """Test summarization for default case."""
        result = _summarize_step_outcome({"metadata": {}})
        self.assertEqual(result, "updated")


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

    def stream(
        self,
        state: dict[str, object],
        *,
        stream_mode: str = "updates",
    ):
        current = self.builder.start_target
        state = dict(state)

        while current and current != "__end__":
            node: object = self.builder.nodes[current]
            update = node(state)  # type: ignore[operator]
            state.update(update)
            if stream_mode == "updates":
                yield {current: update}
            elif stream_mode == "values":
                yield dict(state)
            else:
                raise ValueError(f"Unsupported stream mode: {stream_mode}")
            if current in self.builder.conditional_edges:
                router, mapping = self.builder.conditional_edges[current]
                current = mapping[router(state)]  # type: ignore[operator]
            else:
                current = self.builder.edges.get(current)


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
        from graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("DELETE FROM Artist"),
            databases=[register_database(SQLiteDatabase())],
        )
        result = graph.invoke({"question": "Delete artists"})
        self.assertIn("sql_validation_error", result)
        self.assertNotIn("query_result", result)

    def test_build_graph_stops_when_no_database_matches(self) -> None:
        """Questions outside the catalog should abort before SQL generation."""
        from graph import build_sql_agent_graph

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
        from graph import build_sql_agent_graph

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

    def test_stream_sql_agent_execution_yields_normalized_steps(self) -> None:
        """Streaming should expose node names, updates, and merged state."""
        from graph import build_sql_agent_graph, stream_sql_agent_execution

        graph = build_sql_agent_graph(
            FakeModel("SELECT Name FROM Artist WHERE ArtistId = 1"),
            databases=[register_database(SQLiteDatabase())],
        )

        steps = list(stream_sql_agent_execution(graph, {"question": "Who is artist 1?"}))

        self.assertEqual(
            [step["node"] for step in steps],
            ["database_selector", "sql_generator", "sql_validator", "sql_executor", "result_analyst"],
        )
        self.assertEqual(steps[0]["outcome"], "database_selected")
        self.assertEqual(steps[1]["outcome"], "sql_generated")
        self.assertEqual(steps[-1]["outcome"], "analysis_ready")
        self.assertEqual(steps[-1]["state"]["analysis"][:8], "Returned")


if __name__ == "__main__":
    unittest.main()
