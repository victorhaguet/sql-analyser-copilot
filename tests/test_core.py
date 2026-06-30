"""Tests for core business logic."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from langgraph.types import Interrupt
from tools.database import DatabaseError
from core import (
    PendingApprovalSession,
    answer_question,
    stream_question,
    start_question,
    resume_question,
    _select_requested_databases,
    _serialize_interrupt,
    _extract_interrupt_payload,
    _snapshot_to_state,
    _snapshot_interrupt_payload,
    _serialize_user,
    _serialize_query_result,
    _serialize_state,
)
from tests.test_db.helpers import fixture_registered_database


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


class FakeResumeCommand:
    """Fake resume command used to avoid real LangGraph resumes in unit tests."""

    def __init__(self, *, resume: str) -> None:
        self.resume = resume


class FakeApprovalGraph:
    """Minimal graph double for approval-session tests."""

    def __init__(
        self,
        *,
        start_result: object,
        start_snapshot: SimpleNamespace,
        resume_result: object | None = None,
        resume_snapshot: SimpleNamespace | None = None,
    ) -> None:
        self.start_result = start_result
        self.start_snapshot = start_snapshot
        self.resume_result = resume_result if resume_result is not None else {}
        self.resume_snapshot = resume_snapshot or start_snapshot
        self.phase = "start"

    def invoke(self, state: object, config: dict[str, object] | None = None) -> object:
        del config
        if hasattr(state, "resume"):
            self.phase = "resume"
            return self.resume_result
        self.phase = "start"
        return self.start_result

    def get_state(self, config: dict[str, object]) -> SimpleNamespace:
        del config
        if self.phase == "resume":
            return self.resume_snapshot
        return self.start_snapshot


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
        intent_model = FakeModel('{"intent": "query"}')
        result = answer_question(
            question="What are the first two artists?",
            sql_generator_model=generator_model,
            analyst_model=analyst_model,
            intent_model=intent_model,
            databases=[fixture_registered_database()],
        )
        self.assertEqual(
            result["validated_sql"],
            "SELECT Name FROM Artist WHERE ArtistId <= 2 ORDER BY ArtistId",
        )
        self.assertEqual(result["query_result"].row_count, 2)
        self.assertEqual(result["analysis"], "The first two artists are AC/DC and Accept.")

    def test_answer_question_returns_top_five_artists_with_trace(self) -> None:
        """Query flow should return five rows for the top artists report."""
        sql_query = (
            "SELECT Artist.Name, COUNT(Album.AlbumId) AS AlbumCount "
            "FROM Artist "
            "JOIN Album ON Artist.ArtistId = Album.ArtistId "
            "GROUP BY Artist.ArtistId, Artist.Name "
            "ORDER BY AlbumCount DESC "
            "LIMIT 5"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = answer_question(
                question="Which 5 artists have the most albums ?",
                sql_generator_model=FakeModel(sql_query),
                analyst_model=FakeModel("Top artists returned."),
                intent_model=FakeModel('{"intent": "query"}'),
                databases=[fixture_registered_database()],
                include_trace=True,
                trace_log_dir=temp_dir,
            )

        self.assertEqual(result["query_result"].row_count, 5)
        trace = result["metadata"]["execution_trace"]
        trace_nodes = [step["node"] for step in trace]
        self.assertIn("sql_validator", trace_nodes)
        self.assertIn("sql_executor", trace_nodes)

    def test_answer_question_returns_error_when_no_database_matches(self) -> None:
        """The entrypoint should abort before SQL generation for irrelevant questions."""

        result = answer_question(
            question="What will the weather be tomorrow?",
            sql_generator_model=FakeModel("SELECT Name FROM Artist"),
            selector_model=FakeModel(
                '{"match": false, "database": "", "candidate_databases": [], "reason": "No configured database matches."}'
            ),
            intent_model=FakeModel('{"intent": "query"}'),
            databases=[
                fixture_registered_database(name="music", description="Music data"),
                fixture_registered_database(name="sales", description="Sales data"),
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
            intent_model=FakeModel('{"intent": "query"}'),
            databases=[fixture_registered_database()],
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
            intent_model=FakeModel('{"intent": "query"}'),
            databases=[
                fixture_registered_database(name="music", description="Music data"),
                fixture_registered_database(name="sales", description="Sales data"),
            ],
        )
        self.assertTrue(result["metadata"]["database_selection_ambiguous"])
        self.assertEqual(result["metadata"]["candidate_databases"], ["music", "sales"])
        self.assertNotIn("generated_sql", result)

    def test_answer_question_rejects_modification_intent(self) -> None:
        """Readonly users should be blocked before SQL generation on modifications."""
        result = answer_question(
            question="Delete all artists",
            sql_generator_model=FakeModel("DELETE FROM Artist"),
            selector_model=FakeModel(
                '{"match": true, "database": "music", "candidate_databases": [], "reason": "Match found"}'
            ),
            intent_model=FakeModel('{"intent": "modification"}'),
            databases=[fixture_registered_database(name="music", description="Music data")],
        )
        self.assertIn("authorization_error", result)
        self.assertEqual(result["intent"], "modification")
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
                intent_model=FakeModel('{"intent": "query"}'),
                databases=[fixture_registered_database()],
                include_trace=True,
                trace_log_dir=temp_dir,
            )

            trace = result["metadata"]["execution_trace"]
            self.assertEqual(trace[0]["node"], "database_selector")
            self.assertEqual(trace[-1]["node"], "result_analyst")
            self.assertEqual(trace[-1]["update"]["analysis"], result["analysis"])
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
                intent_model=FakeModel('{"intent": "query"}'),
                databases=[fixture_registered_database()],
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

    def test_answer_question_no_trace_returns_without_trace_data(self) -> None:
        """Without include_trace, result should not have trace metadata."""

        result = answer_question(
            question="What are the first two artists?",
            sql_generator_model=FakeModel(
                "SELECT Name FROM Artist WHERE ArtistId <= 2 ORDER BY ArtistId"
            ),
            analyst_model=FakeModel("The first two artists are AC/DC and Accept."),
            databases=[fixture_registered_database()],
            include_trace=False,
        )
        self.assertNotIn("execution_trace", result.get("metadata", {}))
        self.assertNotIn("trace_log_path", result.get("metadata", {}))

    def test_stream_question_returns_intent_classifier_failure_step(self) -> None:
        """Streaming should expose readonly authorization failures for modifications."""

        steps = list(
            stream_question(
                question="Delete all artists",
                sql_generator_model=FakeModel("DELETE FROM Artist"),
                intent_model=FakeModel('{"intent": "modification"}'),
                databases=[fixture_registered_database(name="music", description="Music data")],
            )
        )

        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["node"], "database_selector")
        self.assertEqual(steps[1]["node"], "intent_classifier")
        self.assertEqual(steps[2]["node"], "role_authorizer")
        self.assertEqual(steps[2]["outcome"], "authorization_failed")

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
                    fixture_registered_database(name="music", description="Music data"),
                    fixture_registered_database(name="sales", description="Sales data"),
                ],
            )
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["node"], "database_selector")
        self.assertEqual(steps[0]["outcome"], "database_selection_failed")

    def test_stream_question_exposes_execution_steps(self) -> None:
        """Streaming should expose each execution step with node name, update, and outcome."""

        generator_model = FakeModel(
            "SELECT Name FROM Artist WHERE ArtistId = 1"
        )
        analyst_model = FakeModel("The first artist is AC/DC.")

        steps = list(
            stream_question(
                question="Who is the first artist?",
                sql_generator_model=generator_model,
                analyst_model=analyst_model,
                intent_model=FakeModel('{"intent": "query"}'),
                databases=[fixture_registered_database()],
            )
        )

        self.assertGreater(len(steps), 0)
        for step in steps:
            self.assertIn("node", step)
            self.assertIn("update", step)
            self.assertIn("outcome", step)
            self.assertIn("state", step)

    def test_select_requested_databases_returns_all_when_no_filter(self) -> None:
        """When no names requested, return all configured databases."""

        db1 = fixture_registered_database(name="db1")
        db2 = fixture_registered_database(name="db2")
        result = _select_requested_databases(None, [db1, db2])
        self.assertEqual(result, [db1, db2])

    def test_select_requested_databases_returns_filtered_list(self) -> None:
        """When names provided, return only matching databases."""

        db1 = fixture_registered_database(name="db1")
        db2 = fixture_registered_database(name="db2")
        db3 = fixture_registered_database(name="db3")
        result = _select_requested_databases(["db2", "db1"], [db1, db2, db3])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "db2")
        self.assertEqual(result[1].name, "db1")

    def test_select_requested_databases_raises_on_empty_list(self) -> None:
        """Empty requested names should raise DatabaseError."""

        db1 = fixture_registered_database(name="db1")
        with self.assertRaises(DatabaseError) as cm:
            _select_requested_databases([], [db1])
        self.assertIn("At least one database must be selected", str(cm.exception))

    def test_select_requested_databases_raises_on_unknown_name(self) -> None:
        """Unknown database names should raise DatabaseError."""

        db1 = fixture_registered_database(name="db1")
        with self.assertRaises(DatabaseError) as cm:
            _select_requested_databases(["db1", "unknown"], [db1])
        self.assertIn("Unknown databases requested", str(cm.exception))
        self.assertIn("unknown", str(cm.exception))

    def test_select_requested_databases_raises_when_all_unknown(self) -> None:
        """When all requested names are unknown, raise error about unknown databases."""

        db1 = fixture_registered_database(name="db1")
        with self.assertRaises(DatabaseError) as cm:
            _select_requested_databases(["unknown1", "unknown2"], [db1])
        self.assertIn("Unknown databases requested", str(cm.exception))

    def test_serialize_user_returns_correct_dict(self) -> None:
        """Test user serialization with all fields."""

        user = {
            "sub": "user123",
            "username": "testuser",
            "name": "Test User",
            "role": "admin",
            "is_active": True,
            "created_at": "2024-01-01T00:00:00",
        }
        result = _serialize_user(user)
        self.assertEqual(result, user)

    def test_serialize_query_result_returns_none_for_none_input(self) -> None:
        """Test that None input returns None."""

        result = _serialize_query_result(None)
        self.assertIsNone(result)

    def test_serialize_query_result_returns_dict_for_valid_result(self) -> None:
        """Test serialization of a valid QueryResult."""

        from tools.database import QueryResult

        result = _serialize_query_result(
            QueryResult(
                columns=["id", "name"],
                rows=[[1, "Alice"], [2, "Bob"]],
                row_count=2,
                truncated=False,
            )
        )
        self.assertEqual(result["columns"], ["id", "name"])
        self.assertEqual(result["rows"], [[1, "Alice"], [2, "Bob"]])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["truncated"], False)

    def test_serialize_state_returns_complete_response(self) -> None:
        """Test serialization of complete state."""

        from tools.database import QueryResult

        state: dict = {
            "question": "Test question",
            "schema_overview": "overview",
            "selected_database": "db1",
            "metadata": {"selected_database": "db1"},
            "generated_sql": "SELECT * FROM t",
            "validated_sql": "SELECT * FROM t",
            "sql_validation_error": None,
            "query_result": QueryResult(
                columns=["id"], rows=[[1]], row_count=1, truncated=False
            ),
            "execution_error": None,
            "analysis": "Result analysis",
        }
        result = _serialize_state(state, "Test question")
        self.assertEqual(result["question"], "Test question")
        self.assertEqual(result["schema_overview"], "overview")
        self.assertEqual(result["selected_database"], "db1")
        self.assertEqual(result["generated_sql"], "SELECT * FROM t")
        self.assertEqual(result["validated_sql"], "SELECT * FROM t")
        self.assertIsNone(result["sql_validation_error"])
        self.assertEqual(result["query_result"]["row_count"], 1)
        self.assertIsNone(result["execution_error"])
        self.assertEqual(result["analysis"], "Result analysis")
        self.assertIn("metadata", result)

    def test_serialize_interrupt_wraps_scalar_value(self) -> None:
        """Interrupt values should be JSON-serializable even when scalar."""
        result = _serialize_interrupt(Interrupt("approve"))
        self.assertEqual(result, {"value": "approve"})

    def test_extract_interrupt_payload_returns_none_for_non_dict_result(self) -> None:
        """Non-dict invoke results should not expose an interrupt payload."""
        self.assertIsNone(_extract_interrupt_payload("invalid"))

    def test_extract_interrupt_payload_returns_none_for_non_interrupt_list(self) -> None:
        """Interrupt payload extraction should ignore unexpected list contents."""
        result = _extract_interrupt_payload({"__interrupt__": ["invalid"]})
        self.assertIsNone(result)

    def test_snapshot_to_state_returns_fallback_for_non_dict_values(self) -> None:
        """Snapshot conversion should fall back when values are unavailable."""
        result = _snapshot_to_state(SimpleNamespace(values=None), {"question": "fallback"})
        self.assertEqual(result, {"question": "fallback"})

    def test_snapshot_interrupt_payload_returns_none_for_non_interrupt(self) -> None:
        """Snapshot interrupt extraction should ignore invalid interrupt entries."""
        result = _snapshot_interrupt_payload(SimpleNamespace(interrupts=["invalid"]))
        self.assertIsNone(result)

    def test_start_question_returns_pending_approval_state_for_admin_modification(self) -> None:
        """Admin modifications should pause for approval and register the session."""
        pending_sessions: dict[str, PendingApprovalSession] = {}
        interrupt = Interrupt({"draft": "DELETE FROM Artist WHERE ArtistId = 1"})
        fake_graph = FakeApprovalGraph(
            start_result={"__interrupt__": [interrupt]},
            start_snapshot=SimpleNamespace(
                values={"question": "Delete artist 1", "intent": "modification"},
                interrupts=[interrupt],
            ),
        )

        with patch("core.build_sql_agent_graph", return_value=fake_graph):
            result = start_question(
                question="Delete artist 1",
                sql_generator_model=FakeModel("DELETE FROM Artist WHERE ArtistId = 1"),
                user_role="admin",
                user_sub="user-1",
                pending_approval_sessions=pending_sessions,
            )

        self.assertTrue(result["execution_requested"])
        self.assertFalse(result["execution_confirmed"])
        self.assertIn("thread_id", result)
        self.assertIn("interrupt", result)
        self.assertIn(result["thread_id"], pending_sessions)

    def test_resume_question_reject_clears_pending_session(self) -> None:
        """Rejecting a modification should stop execution and remove the session."""
        pending_sessions: dict[str, PendingApprovalSession] = {
            "thread-1": PendingApprovalSession(
                graph=FakeApprovalGraph(
                    start_result={},
                    start_snapshot=SimpleNamespace(values={}, interrupts=[]),
                    resume_result={},
                    resume_snapshot=SimpleNamespace(
                        values={
                            "question": "Delete artist 1",
                            "intent": "modification",
                            "execution_confirmed": False,
                        },
                        interrupts=[],
                    ),
                ),
                question="Delete artist 1",
                user_sub="user-1",
            )
        }

        with patch("core.Command", FakeResumeCommand):
            result = resume_question(
                "thread-1",
                "reject",
                pending_approval_sessions=pending_sessions,
                user_sub="user-1",
            )

        self.assertFalse(result["execution_confirmed"])
        self.assertNotIn("thread_id", result)
        self.assertNotIn("interrupt", result)
        self.assertEqual(pending_sessions, {})

    def test_resume_question_approve_with_trace_executes_modification(self) -> None:
        """Approving a modification should execute it and attach a trace."""
        fake_graph = FakeApprovalGraph(
            start_result={},
            start_snapshot=SimpleNamespace(values={}, interrupts=[]),
            resume_result={},
            resume_snapshot=SimpleNamespace(
                values={
                    "question": "Delete artist 1",
                    "intent": "modification",
                    "execution_confirmed": True,
                    "query_result": {"rows_affected": 1},
                    "analysis": "Executed.",
                },
                interrupts=[],
            ),
        )
        pending_sessions: dict[str, PendingApprovalSession] = {
            "thread-1": PendingApprovalSession(
                graph=fake_graph,
                question="Delete artist 1",
                user_sub="user-1",
            )
        }
        trace = [
            {
                "node": "sql_executor",
                "update": {"query_result": {"rows_affected": 1}},
                "state": {
                    "question": "Delete artist 1",
                    "intent": "modification",
                    "execution_confirmed": True,
                    "query_result": {"rows_affected": 1},
                },
                "outcome": "query_executed",
            },
            {
                "node": "result_analyst",
                "update": {"analysis": "Executed."},
                "state": {
                    "question": "Delete artist 1",
                    "intent": "modification",
                    "execution_confirmed": True,
                    "query_result": {"rows_affected": 1},
                    "analysis": "Executed.",
                },
                "outcome": "analysis_ready",
            },
        ]

        with (
            patch("core.Command", FakeResumeCommand),
            patch("core.stream_sql_agent_execution", return_value=iter(trace)),
            patch("core.write_trace_log", return_value=Path("/tmp/trace.log")),
        ):
            result = resume_question(
                "thread-1",
                "approve",
                pending_approval_sessions=pending_sessions,
                include_trace=True,
                trace_log_dir="/tmp",
                user_sub="user-1",
            )

        self.assertTrue(result["execution_confirmed"])
        self.assertIn("query_result", result)
        self.assertIn("execution_trace", result["metadata"])
        self.assertNotIn("thread_id", result)
        self.assertEqual(pending_sessions, {})

    def test_resume_question_raises_for_missing_thread(self) -> None:
        """Unknown approval sessions should raise a clear error."""
        with self.assertRaises(ValueError) as context:
            resume_question(
                "missing-thread",
                "approve",
                pending_approval_sessions={},
            )
        self.assertIn("No pending approval found for this thread", str(context.exception))

    def test_resume_question_raises_for_wrong_user(self) -> None:
        """Approval sessions should be scoped to the initiating user."""
        pending_sessions: dict[str, PendingApprovalSession] = {
            "thread-1": PendingApprovalSession(
                graph=FakeApprovalGraph(
                    start_result={},
                    start_snapshot=SimpleNamespace(values={}, interrupts=[]),
                ),
                question="Delete artist 1",
                user_sub="user-1",
            )
        }

        with self.assertRaises(ValueError) as context:
            resume_question(
                "thread-1",
                "approve",
                pending_approval_sessions=pending_sessions,
                user_sub="user-2",
            )
        self.assertIn("No pending approval found for this user", str(context.exception))


class FakeResponseTestCase(unittest.TestCase):
    """Test FakeResponse class."""

    def test_fake_response_has_content(self) -> None:
        """FakeResponse should store and expose content."""
        response = FakeResponse("test content")
        self.assertEqual(response.content, "test content")


class FakeModelTestCase(unittest.TestCase):
    """Test FakeModel class."""

    def test_fake_model_invoke_stores_prompt(self) -> None:
        """Invoke should store the prompt in prompts list."""
        model = FakeModel("response")
        model.invoke("test prompt")
        self.assertEqual(model.prompts, ["test prompt"])

    def test_fake_model_invoke_returns_response(self) -> None:
        """Invoke should return FakeResponse with configured content."""
        model = FakeModel("test response")
        response = model.invoke("any prompt")
        self.assertEqual(response.content, "test response")


class FakeCommandTestCase(unittest.TestCase):
    """Test FakeCommand class."""

    def test_fake_command_stores_update(self) -> None:
        """FakeCommand should store the update dict."""
        update = {"key": "value"}
        command = FakeCommand(update)
        self.assertEqual(command.update, update)


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


if __name__ == "__main__":
    unittest.main()
