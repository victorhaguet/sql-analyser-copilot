"""Tests for LangGraph wiring and orchestration."""

from __future__ import annotations

import sys
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from uuid import uuid4

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from langgraph.types import Command, Interrupt
from graph import (
    INTERRUPT_EVENT,
    NODE_RESULT_ANALYST,
    NODE_SQL_FALLBACK_REGENERATOR,
    ROUTE_ABORT,
    _normalize_stream_event,
    _route_after_executor,
    _route_from_fallback_regenerator,
)
from state import SQLAgentState
from graph import _summarize_step_outcome
from tools.database import SQLiteDatabase

SQLAgentStateUpdate = SQLAgentState


def build_thread_config() -> dict[str, dict[str, str]]:
    """Create a LangGraph config with a unique thread id for tests."""
    return {"configurable": {"thread_id": uuid4().hex}}


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

    def test_normalize_stream_event_empty_dict_raises_error(self) -> None:
        """Test that empty dict raises TypeError in updates mode."""
        event = {}
        current_state = SQLAgentState(question="test")

        with self.assertRaises(TypeError):
            _normalize_stream_event(event, current_state, "updates")

    def test_normalize_stream_event_handles_interrupt_event(self) -> None:
        """Interrupt events should be normalized into approval state."""
        event = {INTERRUPT_EVENT: [Interrupt({"draft": "DELETE FROM Artist"})]}
        node_name, update, state = _normalize_stream_event(event, {}, "updates")
        self.assertEqual(node_name, INTERRUPT_EVENT)
        self.assertEqual(update["interrupt"], {"draft": "DELETE FROM Artist"})
        self.assertTrue(update["execution_requested"])
        self.assertEqual(state["interrupt"], {"draft": "DELETE FROM Artist"})

    def test_normalize_stream_event_rejects_empty_interrupt_list(self) -> None:
        """Interrupt events without payloads should fail fast."""
        with self.assertRaises(TypeError):
            _normalize_stream_event({INTERRUPT_EVENT: []}, {}, "updates")

    def test_normalize_stream_event_rejects_non_interrupt_payload(self) -> None:
        """Interrupt events should require a LangGraph Interrupt object."""
        with self.assertRaises(TypeError):
            _normalize_stream_event({INTERRUPT_EVENT: ["invalid"]}, {}, "updates")


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

    def test_summarize_database_checked(self) -> None:
        """Test summarization when database is selected."""
        result = _summarize_step_outcome({"selected_database": "db1"})# type: ignore[typeddict-item]
        self.assertEqual(result, "database_checked")

    def test_summarize_updated(self) -> None:
        """Test summarization for default case."""
        result = _summarize_step_outcome({"metadata": {}})
        self.assertEqual(result, "updated")

    def test_summarize_authorization_failed(self) -> None:
        """Authorization errors should use the dedicated outcome label."""
        result = _summarize_step_outcome({"authorization_error": "denied"})
        self.assertEqual(result, "authorization_failed")

    def test_summarize_execution_pending_approval(self) -> None:
        """Interrupt state should be summarized as pending approval."""
        result = _summarize_step_outcome({"interrupt": {"draft": "DELETE"}})
        self.assertEqual(result, "execution_pending_approval")


class RetryRoutingTestCase(unittest.TestCase):
    """Test retry boundaries and fallback failure routing."""

    def test_execution_failure_routes_to_regenerator_below_retry_limit(self) -> None:
        route = _route_after_executor(
            {"execution_error": "failed", "retry_count": 2, "max_retries": 3}
        )

        self.assertEqual(route, NODE_SQL_FALLBACK_REGENERATOR)

    def test_execution_failure_aborts_at_retry_limit(self) -> None:
        route = _route_after_executor(
            {"execution_error": "failed", "retry_count": 3, "max_retries": 3}
        )

        self.assertEqual(route, ROUTE_ABORT)

    def test_successful_execution_routes_to_result_analysis(self) -> None:
        self.assertEqual(_route_after_executor({"execution_error": None}), NODE_RESULT_ANALYST)

    def test_regeneration_failure_aborts_without_reexecuting_failed_sql(self) -> None:
        route = _route_from_fallback_regenerator(
            {"regeneration_error": "model returned no SQL", "intent": "modification"}
        )

        self.assertEqual(route, ROUTE_ABORT)


class FakeResponseTestCase(unittest.TestCase):
    """Test FakeResponse class."""

    def test_fake_response_has_content(self) -> None:
        """FakeResponse should store and expose content."""
        response = FakeResponse("test content")
        self.assertEqual(response.content, "test content")


class FakeModelTestCase(unittest.TestCase):
    """Test FakeModel class."""

    def test_fake_model_invoke_returns_response(self) -> None:
        """Invoke should return FakeResponse with configured content."""
        model = FakeModel("test response")
        response = model.invoke("any prompt")
        self.assertEqual(response.content, "test response")

    def test_fake_model_stores_response(self) -> None:
        """FakeModel should store the response."""
        model = FakeModel("stored response")
        self.assertEqual(model.response, "stored response")


class FakeCompiledGraphTestCase(unittest.TestCase):
    """Test FakeCompiledGraph class."""

    def test_fake_compiled_graph_invoke(self) -> None:
        """Test invoke method with simple linear graph."""
        builder = FakeStateGraph(dict)
        builder.add_node("node1", lambda s: {"key": "value1"})
        builder.add_edge("__start__", "node1")

        compiled = builder.compile()
        result = compiled.invoke({})
        self.assertEqual(result, {"key": "value1"})

    def test_fake_compiled_graph_stream_updates(self) -> None:
        """Test stream method with updates mode."""
        builder = FakeStateGraph(dict)
        builder.add_node("node1", lambda s: {"key": "value1"})
        builder.add_edge("__start__", "node1")

        compiled = builder.compile()
        steps = list(compiled.stream({}, stream_mode="updates"))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], {"node1": {"key": "value1"}})

    def test_fake_compiled_graph_stream_values(self) -> None:
        """Test stream method with values mode."""
        builder = FakeStateGraph(dict)
        builder.add_node("node1", lambda s: {"key": "value1"})
        builder.add_edge("__start__", "node1")

        compiled = builder.compile()
        steps = list(compiled.stream({}, stream_mode="values"))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], {"key": "value1"})

    def test_fake_compiled_graph_stream_raises_invalid_mode(self) -> None:
        """Test that invalid stream mode raises ValueError."""
        builder = FakeStateGraph(dict)
        builder.add_node("node1", lambda s: {"key": "value1"})
        builder.add_edge("__start__", "node1")

        compiled = builder.compile()
        with self.assertRaises(ValueError) as cm:
            list(compiled.stream({}, stream_mode="invalid"))
        self.assertIn("invalid", str(cm.exception).lower())


class FakeStateGraphTestCase(unittest.TestCase):
    """Test FakeStateGraph class."""

    def test_add_node(self) -> None:
        """Test adding a node."""
        graph = FakeStateGraph(dict)
        node = lambda s: s
        graph.add_node("test_node", node)
        self.assertEqual(graph.nodes["test_node"], node)

    def test_add_edge_sets_start_target(self) -> None:
        """Test adding edge from start sets start target."""
        graph = FakeStateGraph(dict)
        graph.add_edge("__start__", "target_node")
        self.assertEqual(graph.start_target, "target_node")

    def test_add_edge_sets_normal_edge(self) -> None:
        """Test adding normal edge."""
        graph = FakeStateGraph(dict)
        graph.add_edge("source", "target")
        self.assertEqual(graph.edges["source"], "target")

    def test_add_conditional_edges(self) -> None:
        """Test adding conditional edges."""
        graph = FakeStateGraph(dict)
        router = lambda s: "route1"
        mapping = {"route1": "node1", "route2": "node2"}
        graph.add_conditional_edges("source", router, mapping)
        self.assertEqual(graph.conditional_edges["source"], (router, mapping))


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


class RepairingModel:
    """Generate invalid initial SQL and repair it from fallback context."""

    def __init__(self) -> None:
        self.fallback_prompt = ""

    def invoke(self, prompt: str) -> str:
        if "authoritative repair target" in prompt:
            self.fallback_prompt = prompt
            if "'Suikon Blaz AD'" in prompt:
                return "INSERT INTO Artist (Name) VALUES ('Suikon Blaz AD');"
            return "INSERT INTO Artist (Name) VALUES ('Mandyspie');"
        return "INSERT INTO Artists (Name) VALUES ('Mandyspie');"


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
        config: object | None = None,
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

    def compile(self, checkpointer=None) -> FakeCompiledGraph:
        return FakeCompiledGraph(self)


class TestNormalizeStreamEventTestCase(unittest.TestCase):
    """Test _normalize_stream_event function edge cases."""

    def test_normalize_stream_event_empty_dict_raises_error(self) -> None:
        """Test that empty dict raises TypeError."""
        event = {}
        current_state = SQLAgentState(question="test")

        with self.assertRaises(TypeError):
            _normalize_stream_event(event, current_state, "updates")


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
            selector_model=FakeModel('{"match": true, "database": "chinook", "reason": "Matches question"}'),
            intent_model=FakeModel('{"intent": "query"}'),
            selected_database=SQLiteDatabase(),
        )
        result = graph.invoke({"question": "Delete artists"}, config=build_thread_config())
        self.assertIn("sql_validation_error", result)
        self.assertNotIn("query_result", result)

    def test_build_graph_executes_valid_query(self) -> None:
        """Test that the graph successfully executes valid queries."""

        from graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("SELECT * FROM Artist"),
            selector_model=FakeModel('{"match": true, "database": "chinook", "reason": "Matches question"}'),
            intent_model=FakeModel('{"intent": "query"}'),
            selected_database=SQLiteDatabase(),
        )
        result = graph.invoke(
            {"question": "Select from Artist", "selected_database": "chinook"},
            config=build_thread_config(),
        )
        self.assertIn("query_result", result)
        self.assertIsNone(result.get("execution_error"))
        self.assertIsNotNone(result["query_result"])

    def test_failed_modification_is_regenerated_and_requires_new_approval(self) -> None:
        """A repaired modification must pause again before the corrected SQL executes."""
        from graph import build_sql_agent_graph

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "retry.sqlite"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    "CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT)"
                )

            model = RepairingModel()
            graph = build_sql_agent_graph(
                model,
                selector_model=FakeModel(
                    '{"match": true, "database": "music", "reason": "Matches question"}'
                ),
                intent_model=FakeModel('{"intent": "modification"}'),
                selected_database=SQLiteDatabase(database_path),
            )
            config = build_thread_config()
            initial_result = graph.invoke(
                {
                    "question": "Create the artist Mandyspie",
                    "user_role": "admin",
                    "retry_count": 0,
                    "max_retries": 3,
                },
                config=config,
            )
            user_edited_sql = (
                "INSERT INTO Artists (Name) VALUES ('Suikon Blaz AD');"
            )
            retry_result = graph.invoke(
                Command(
                    resume="approve",
                    update={"generated_sql": user_edited_sql},
                ),
                config=config,
            )

            self.assertIn("__interrupt__", initial_result)
            self.assertIn("__interrupt__", retry_result)
            retry_payload = retry_result["__interrupt__"][0].value
            self.assertEqual(
                retry_payload["previous_sql"],
                user_edited_sql,
            )
            self.assertIn("no such table: Artists", retry_payload["previous_error"])
            self.assertIn(
                "rewrote the statement",
                retry_payload["regeneration_explanation"],
            )
            retry_state = graph.get_state(config).values
            self.assertEqual(retry_state["retry_count"], 1)
            self.assertEqual(
                retry_state["generated_sql"],
                "INSERT INTO Artist (Name) VALUES ('Suikon Blaz AD');",
            )
            self.assertIn("no such table: Artists", model.fallback_prompt)

            graph.invoke(Command(resume="approve"), config=config)
            final_state = graph.get_state(config).values

            self.assertIsNone(final_state["execution_error"])
            with sqlite3.connect(database_path) as connection:
                artist_count = connection.execute(
                    "SELECT COUNT(*) FROM Artist WHERE Name = 'Suikon Blaz AD'"
                ).fetchone()[0]
            self.assertEqual(artist_count, 1)

    def test_build_graph_stops_when_no_database_matches(self) -> None:
        """Questions outside the catalog should abort before SQL generation."""
        from graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "reason": "No configured database matches."}'
            ),
            selected_database=SQLiteDatabase(),
        )
        result = graph.invoke(
            {"question": "What is the weather in Berlin?"},
            config=build_thread_config(),
        )
        self.assertIn("execution_error", result)
        self.assertNotIn("generated_sql", result)

    def test_build_graph_stops_when_database_selection_is_ambiguous(self) -> None:
        """Questions that fit multiple databases should abort before SQL generation."""
        from graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "reason": "Question does not relate to database."}'
            ),
            selected_database=SQLiteDatabase(),
        )
        result = graph.invoke(
            {"question": "Show me recent invoice activity."},
            config=build_thread_config(),
        )
        self.assertTrue(result["metadata"]["database_selection_failed"])
        self.assertNotIn("generated_sql", result)

    def test_stream_sql_agent_execution_yields_normalized_steps(self) -> None:
        """Streaming should expose node names, updates, and merged state."""
        from graph import build_sql_agent_graph, stream_sql_agent_execution

        graph = build_sql_agent_graph(
            FakeModel('SELECT Name FROM Artist WHERE ArtistId = 1'),
            selector_model=FakeModel('{"match": true, "database": "chinook", "reason": "Matches question"}'),
            intent_model=FakeModel('{"intent": "query"}'),
            selected_database=SQLiteDatabase(),
        )

        steps = list(stream_sql_agent_execution(graph, {"question": "Who is artist 1?"}))

        self.assertEqual(
            [step["node"] for step in steps],
            ["database_checker", "intent_classifier", "sql_generator", "sql_validator", "sql_executor", "result_analyst"],
        )
        self.assertEqual(steps[0]["outcome"], "database_checked")
        self.assertEqual(steps[1]["outcome"], "Intention classified")
        self.assertEqual(steps[2]["outcome"], "sql_generated")
        self.assertEqual(steps[-1]["outcome"], "analysis_ready")
        self.assertEqual(steps[-1]["state"]["analysis"][:8], "Returned")

    def test_build_graph_stops_when_intent_is_modification(self) -> None:
        """Questions that intent to modify should abort before SQL generation."""
        from graph import build_sql_agent_graph

        graph = build_sql_agent_graph(
            FakeModel('{"intent": "modification"}'),
            selector_model=FakeModel('{"match": true, "database": "chinook", "reason": "Matches question"}'),
            selected_database=SQLiteDatabase(),
        )
        result = graph.invoke(
            {"question": "Delete all artists"},
            config=build_thread_config(),
        )
        self.assertIn("authorization_error", result)
        self.assertEqual(result["intent"], "modification")
        self.assertNotIn("generated_sql", result)

    def test_stream_sql_agent_execution_with_intent_classifier(self) -> None:
        """Streaming should expose intent_classifier after database_selector."""
        from graph import build_sql_agent_graph, stream_sql_agent_execution

        graph = build_sql_agent_graph(
            FakeModel('SELECT Name FROM Artist WHERE ArtistId = 1'),
            selector_model=FakeModel('{"match": true, "database": "chinook", "reason": "Matches question"}'),
            intent_model=FakeModel('{"intent": "query"}'),
            selected_database=SQLiteDatabase(),
        )

        steps = list(stream_sql_agent_execution(graph, {"question": "Who is artist 1?"}))

        self.assertEqual(
            [step["node"] for step in steps],
            [
                "database_checker",
                "intent_classifier",
                "sql_generator",
                "sql_validator",
                "sql_executor",
                "result_analyst",
            ],
        )
        self.assertEqual(steps[0]["outcome"], "database_checked")
        self.assertEqual(steps[1]["outcome"], "Intention classified")

    def test_stream_sql_agent_execution_with_values_mode(self) -> None:
        """Test streaming with values mode instead of updates mode."""

        from graph import build_sql_agent_graph, stream_sql_agent_execution

        graph = build_sql_agent_graph(
            FakeModel("SELECT Name FROM Artist WHERE ArtistId = 1"),
            selected_database=SQLiteDatabase(),
        )

        steps = list(
            stream_sql_agent_execution(
                graph, {"question": "Who is artist 1?"}, stream_mode="values"
            )
        )

        self.assertGreater(len(steps), 0)
        for step in steps:
            self.assertIn("node", step)
            self.assertIn("update", step)
            self.assertIn("outcome", step)
            self.assertIn("state", step)


if __name__ == "__main__":
    unittest.main()
