"""Tests for core business logic."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from tools.database import SQLiteDatabase, register_database
from core import answer_question, stream_question


class FakeResponse:
    """Simple model response object with a content field."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """Minimal test double for invoke-based text models."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.response)


class FakeCommand:
    """Fake Command class for testing."""

    def __init__(self, update: dict[str, object]) -> None:
        self.update = update


class FakeCompiledGraph:
    """Small in-memory executor that mimics the compiled graph API."""

    def __init__(self, builder: "FakeStateGraph") -> None:
        self.builder = builder

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        current = self.builder.start_target
        state = dict(state)

        while current and current != "__end__":
            node = self.builder.nodes[current]
            state.update(node(state))
            if current in self.builder.conditional_edges:
                router, mapping = self.builder.conditional_edges[current]
                current = mapping[router(state)]
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
            node = self.builder.nodes[current]
            update = node(state)
            state.update(update)
            if stream_mode == "updates":
                yield {current: update}
            elif stream_mode == "values":
                yield dict(state)
            else:
                raise ValueError(f"Unsupported stream mode: {stream_mode}")
            if current in self.builder.conditional_edges:
                router, mapping = self.builder.conditional_edges[current]
                current = mapping[router(state)]
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


class CoreTestCase(unittest.TestCase):
    """Test core business logic functions."""

    def setUp(self) -> None:
        """Patch sys.modules to provide fake langgraph modules for testing without langgraph installed."""
        self._original_modules = {
            name: sys.modules.get(name)
            for name in ("langgraph", "langgraph.graph", "langgraph.graph.state")
        }
        langgraph_module = types.ModuleType("langgraph")
        graph_module = types.ModuleType("langgraph.graph")
        graph_state_module = types.ModuleType("langgraph.graph.state")
        graph_module.START = "__start__"
        graph_module.END = "__end__"
        graph_module.StateGraph = FakeStateGraph
        graph_state_module.Command = FakeCommand
        sys.modules["langgraph"] = langgraph_module
        sys.modules["langgraph.graph"] = graph_module
        sys.modules["langgraph.graph.state"] = graph_state_module

    def tearDown(self) -> None:
        """Restore original sys.modules after tests."""
        for name, module in self._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_answer_question_runs_full_pipeline(self) -> None:
        """Test that the answer_question function correctly orchestrates the full pipeline from question to analysis."""
        

        generator_model = FakeModel(
            "SELECT Name FROM Artist WHERE ArtistId <= 2 ORDER BY ArtistId"
        )
        analyst_model = FakeModel("The first two artists are AC/DC and Accept.")
        result = answer_question(
            question="What are the first two artists?",
            sql_generator_model=generator_model,
            analyst_model=analyst_model,
            databases=[register_database(SQLiteDatabase())],
        )
        self.assertEqual(
            result["validated_sql"],
            "SELECT Name FROM Artist WHERE ArtistId <= 2 ORDER BY ArtistId",
        )
        self.assertEqual(result["query_result"].row_count, 2)
        self.assertEqual(result["analysis"], "The first two artists are AC/DC and Accept.")

    def test_answer_question_returns_error_when_no_database_matches(self) -> None:
        """The entrypoint should abort before SQL generation for irrelevant questions."""

        result = answer_question(
            question="What will the weather be tomorrow?",
            sql_generator_model=FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "No configured database matches."}'
            ),
            databases=[
                register_database(SQLiteDatabase(), name="music", description="Music data"),
                register_database(SQLiteDatabase(), name="sales", description="Sales data"),
            ],
        )
        self.assertIn("No configured database matches.", result["analysis"])
        self.assertNotIn("generated_sql", result)

    def test_answer_question_rejects_irrelevant_single_database_query(self) -> None:
        """Single-database mode should still reject clearly irrelevant questions."""

        result = answer_question(
            question="What will the weather be tomorrow?",
            sql_generator_model=FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "This question is unrelated to the configured database."}'
            ),
            databases=[register_database(SQLiteDatabase())],
        )
        self.assertIn("unrelated to the configured database", result["analysis"])
        self.assertNotIn("generated_sql", result)

    def test_answer_question_returns_ambiguity_error_when_multiple_databases_match(self) -> None:
        """The entrypoint should stop before SQL generation for ambiguous routing."""

        result = answer_question(
            question="Show me recent invoice activity.",
            sql_generator_model=FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": ["music", "sales"], "reason": "The question could be answered from either catalog."}'
            ),
            databases=[
                register_database(SQLiteDatabase(), name="music", description="Music data"),
                register_database(SQLiteDatabase(), name="sales", description="Sales data"),
            ],
        )
        self.assertTrue(result["metadata"]["database_selection_ambiguous"])
        self.assertEqual(result["metadata"]["candidate_databases"], ["music", "sales"])
        self.assertNotIn("generated_sql", result)

    def test_answer_question_can_include_execution_trace(self) -> None:
        """The entrypoint can attach a normalized execution trace for debugging."""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = answer_question(
                question="What are the first two artists?",
                sql_generator_model=FakeModel(
                    "SELECT Name FROM Artist WHERE ArtistId <= 2 ORDER BY ArtistId"
                ),
                analyst_model=FakeModel("The first two artists are AC/DC and Accept."),
                databases=[register_database(SQLiteDatabase())],
                include_trace=True,
                trace_log_dir=temp_dir,
            )

            trace = result["metadata"]["execution_trace"]
            self.assertEqual(trace[0]["node"], "database_selector")
            self.assertEqual(trace[-1]["node"], "result_analyst")
            self.assertEqual(trace[-1]["state"]["analysis"], result["analysis"])
            self.assertTrue(Path(result["metadata"]["trace_log_path"]).exists())

    def test_answer_question_creates_trace_log_file(self) -> None:
        """Each run with include_trace=True should create a persisted trace log file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = answer_question(
                question="What are the first two artists?",
                sql_generator_model=FakeModel(
                    "SELECT Name FROM Artist WHERE ArtistId <= 2 ORDER BY ArtistId"
                ),
                analyst_model=FakeModel("The first two artists are AC/DC and Accept."),
                databases=[register_database(SQLiteDatabase())],
                include_trace=True,
                trace_log_dir=temp_dir,
            )

            log_path = Path(result["metadata"]["trace_log_path"])
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.parent, Path(temp_dir))
            log_content = log_path.read_text(encoding="utf-8")

        self.assertIn("Human Message", log_content)
        self.assertIn("What are the first two artists?", log_content)
        self.assertIn("Name: sql_executor", log_content)
        self.assertIn("The first two artists are AC/DC and Accept.", log_content)

    def test_stream_question_returns_database_selection_failure_step(self) -> None:
        """Streaming should expose early aborts at the selector step."""

        steps = list(
            stream_question(
                question="What will the weather be tomorrow?",
                sql_generator_model=FakeModel("SELECT Name FROM Artist"),
                selector_model=FakeModel(
                    '{"match": false, "database": "", "candidate_databases": [], "reason": "No configured database matches."}'
                ),
                databases=[
                    register_database(SQLiteDatabase(), name="music", description="Music data"),
                    register_database(SQLiteDatabase(), name="sales", description="Sales data"),
                ],
            )
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["node"], "database_selector")
        self.assertEqual(steps[0]["outcome"], "database_selection_failed")


if __name__ == "__main__":
    unittest.main()
