"""End-to-end scenario tests for the SQL generation agent loop (Step 11).

Each test drives the real compiled graph (real LangGraph StateGraph, checkpointer,
interrupt/resume) against the committed Chinook fixture, with a `ScriptedChatModel`
standing in for the agent's LLM. Unlike tests/test_nodes/test_sql_agent.py (which
exercises the loop nodes directly) these lock in the *graph-level* behaviours and
invariants the feature exists for: fast reads, the artist clarification flow,
probe-driven self-correction, cancellation, budget enforcement, the read-only
guarantee, mixed tool batches, and in-loop write repair (D6) including its budget.
"""

from __future__ import annotations

import sqlite3
import unittest
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.types import Command

from graph import build_sql_agent_graph
from tests.test_db.helpers import fixture_database, mutable_fixture_database


def build_thread_config() -> dict[str, dict[str, str]]:
    """Create a LangGraph config with a unique thread id for tests."""
    return {"configurable": {"thread_id": uuid4().hex}}


class FakeResponse:
    """Simple model response object with a content field."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """Minimal test double for invoke-based text models (selector/intent)."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> FakeResponse:
        del prompt
        return FakeResponse(self.response)


class ScriptedChatModel:
    """Fake tool-calling chat model: bind_tools returns self, invoke pops a scripted reply."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.bound_tools: list[Any] | None = None
        self.invocations: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedChatModel":
        """Record the bound tools and return self, like a real bind_tools call."""
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Record the messages sent and pop the next scripted response."""
        self.invocations.append(list(messages))
        if not self._responses:
            raise AssertionError("ScriptedChatModel ran out of scripted responses")
        return self._responses.pop(0)


def _build_graph(database, agent_model, *, intent: str = "query"):
    """Build a graph wired for these scenarios: fixed database/intent selection."""
    return build_sql_agent_graph(
        FakeModel("unused"),
        agent_model=agent_model,
        selector_model=FakeModel('{"match": true, "database": "chinook", "reason": "Matches question"}'),
        intent_model=FakeModel(f'{{"intent": "{intent}"}}'),
        selected_database=database,
    )


class SQLAgentScenarioTestCase(unittest.TestCase):
    """The 9 end-to-end scenarios from AGENTIC_SQL_GENERATION_PLAN.md Step 11."""

    def test_simple_query_fast_path_stays_at_or_below_two_iterations(self) -> None:
        """Scenario 1: a fast read needs at most one probe before finalizing."""
        sql = (
            "SELECT ar.Name, COUNT(al.AlbumId) AS AlbumCount FROM Artist ar "
            "JOIN Album al ON al.ArtistId = ar.ArtistId "
            "GROUP BY ar.ArtistId ORDER BY AlbumCount DESC LIMIT 5"
        )
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "run_readonly_probe", "args": {"sql": sql, "limit": 5}, "id": "call-1"}
                    ],
                ),
                AIMessage(content=sql),
            ]
        )
        graph = _build_graph(fixture_database(), model, intent="query")

        result = graph.invoke(
            {"question": "Top 5 artists by album count"}, config=build_thread_config()
        )

        self.assertIsNone(result.get("execution_error"))
        self.assertIn("query_result", result)
        self.assertLessEqual(result["agent_iterations"], 2)
        self.assertEqual(result["probe_count"], 1)

    def test_artist_case_reaches_approval_after_inspection_and_clarification(self) -> None:
        """Scenario 2 (acceptance criterion): inspect, ask, answer, propose, approve-dialog."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "inspect_schema",
                            "args": {"tables": ["Artist", "Album"]},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {
                                "questions": [
                                    {
                                        "key": "add_albums",
                                        "question": "Should I also create any albums for this new artist?",
                                        "why": (
                                            "Album rows reference ArtistId; deciding whether to "
                                            "create matching albums before finalizing."
                                        ),
                                    }
                                ]
                            },
                            "id": "call-2",
                        }
                    ],
                ),
                AIMessage(content="INSERT INTO Artist (Name) VALUES ('Mandyspie')"),
            ]
        )
        with mutable_fixture_database() as database:
            graph = _build_graph(database, model, intent="modification")
            config = build_thread_config()

            awaiting_clarification = graph.invoke(
                {"question": "Add a new artist called Mandyspie", "user_role": "admin"},
                config=config,
            )
            self.assertIn("__interrupt__", awaiting_clarification)
            clarification_payload = awaiting_clarification["__interrupt__"][0].value
            self.assertEqual(clarification_payload["kind"], "clarification")

            awaiting_approval = graph.invoke(
                Command(
                    resume={
                        "decision": "answer",
                        "answers": [{"key": "add_albums", "answer": "No, just the artist."}],
                    }
                ),
                config=config,
            )
            self.assertIn("__interrupt__", awaiting_approval)
            approval_payload = awaiting_approval["__interrupt__"][0].value
            self.assertEqual(approval_payload["kind"], "modification_approval")
            self.assertEqual(
                approval_payload["draft"], "INSERT INTO Artist (Name) VALUES ('Mandyspie')"
            )

    def test_probe_driven_correction_recovers_from_wrong_column_name(self) -> None:
        """Scenario 3: a failed probe is corrected via inspect_schema, not left broken."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_readonly_probe",
                            "args": {"sql": "SELECT ArtistName FROM Artist LIMIT 5"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "call-2"}
                    ],
                ),
                AIMessage(content="SELECT Name FROM Artist LIMIT 5"),
            ]
        )
        graph = _build_graph(fixture_database(), model, intent="query")

        result = graph.invoke(
            {"question": "List artist names"}, config=build_thread_config()
        )

        self.assertIsNone(result.get("execution_error"))
        self.assertIn("query_result", result)
        self.assertEqual(result["probe_count"], 1)
        failure_reported = any(
            "no such column" in str(getattr(message, "content", "")).lower()
            for message in result["messages"]
        )
        self.assertTrue(failure_reported, "the probe's column error must reach the transcript")

    def test_cancelled_clarification_aborts_cleanly_with_nothing_executed(self) -> None:
        """Scenario 4: a cancelled clarification must abort with no SQL ever generated."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {
                                "questions": [
                                    {
                                        "key": "metric",
                                        "question": "Which metric should 'most active' use?",
                                        "why": "The question does not specify a ranking metric.",
                                    }
                                ]
                            },
                            "id": "call-1",
                        }
                    ],
                )
            ]
        )
        graph = _build_graph(fixture_database(), model, intent="query")
        config = build_thread_config()

        paused = graph.invoke(
            {"question": "Who is our most active customer?"}, config=config
        )
        self.assertIn("__interrupt__", paused)

        result = graph.invoke(
            Command(resume={"decision": "cancel"}), config=config
        )

        self.assertEqual(result.get("agent_status"), "cancelled")
        self.assertIsNotNone(result.get("execution_error"))
        self.assertNotIn("generated_sql", result)
        self.assertNotIn("query_result", result)

    def test_budget_exhaustion_aborts_with_no_sql_executed(self) -> None:
        """Scenario 5: looping past the iteration cap ends in a clean explanation, not a hang."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "call-1"}]
                ),
                AIMessage(
                    content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "call-2"}]
                ),
                AIMessage(
                    content=(
                        "I inspected the schema but could not settle on a final query within "
                        "my iteration budget."
                    )
                ),
            ]
        )
        graph = _build_graph(fixture_database(), model, intent="query")

        result = graph.invoke(
            {"question": "Some deliberately ambiguous request", "max_agent_iterations": 2},
            config=build_thread_config(),
        )

        self.assertEqual(result.get("agent_status"), "budget_exhausted")
        self.assertIsNotNone(result.get("agent_error"))
        self.assertNotIn("generated_sql", result)
        self.assertNotIn("query_result", result)

    def test_read_only_invariant_probe_delete_never_mutates_database(self) -> None:
        """Scenario 6: a DELETE probe must be rejected, and the row count must never move."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "run_readonly_probe", "args": {"sql": "DELETE FROM Artist"}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="SELECT COUNT(*) FROM Artist"),
            ]
        )
        with mutable_fixture_database() as database:
            with sqlite3.connect(database.database_path) as connection:
                artist_count_before = connection.execute("SELECT COUNT(*) FROM Artist").fetchone()[0]

            graph = _build_graph(database, model, intent="query")
            result = graph.invoke(
                {"question": "How many artists do we have?"}, config=build_thread_config()
            )

            self.assertIsNone(result.get("execution_error"))
            self.assertIn("query_result", result)
            rejection_reported = any(
                "rejected" in str(getattr(message, "content", "")).lower()
                for message in result["messages"]
            )
            self.assertTrue(rejection_reported, "the DELETE probe must be reported as rejected")

            with sqlite3.connect(database.database_path) as connection:
                artist_count_after = connection.execute("SELECT COUNT(*) FROM Artist").fetchone()[0]
            self.assertEqual(artist_count_after, artist_count_before)

    def test_mixed_batch_ask_user_and_inspect_schema_skips_interrupt(self) -> None:
        """Scenario 7 (D3): a mixed batch never interrupts, and ask_user is rejected in place."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {
                                "questions": [
                                    {
                                        "key": "row_limit",
                                        "question": "How many artist names do you want?",
                                        "why": "No limit was specified.",
                                    }
                                ]
                            },
                            "id": "call-1",
                        },
                        {"name": "inspect_schema", "args": {}, "id": "call-2"},
                    ],
                ),
                AIMessage(content="SELECT Name FROM Artist LIMIT 5"),
            ]
        )
        graph = _build_graph(fixture_database(), model, intent="query")

        result = graph.invoke(
            {"question": "Show me a few artist names"}, config=build_thread_config()
        )

        self.assertNotIn("__interrupt__", result)
        self.assertIsNone(result.get("execution_error"))
        self.assertIn("query_result", result)
        rejection_reported = any(
            "ask_user must be called alone" in str(getattr(message, "content", ""))
            for message in result["messages"]
        )
        self.assertTrue(rejection_reported)

    def test_write_repair_in_loop_recovers_from_not_null_violation(self) -> None:
        """Scenario 8 (D6): a failed write is repaired in-loop and re-approved, with no
        sql_fallback_regenerator involved (that node no longer exists)."""
        first_attempt = "INSERT INTO Album (AlbumId, ArtistId) VALUES (9999, 1)"
        second_attempt = (
            "INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (9999, 'Test Album', 1)\n\n"
            "Rationale: added the required Title column that was missing from the first attempt."
        )
        model = ScriptedChatModel([AIMessage(content=first_attempt), AIMessage(content=second_attempt)])

        with mutable_fixture_database() as database:
            graph = _build_graph(database, model, intent="modification")
            config = build_thread_config()

            first_approval = graph.invoke(
                {"question": "Add a new album called 'Test Album' for artist 1", "user_role": "admin"},
                config=config,
            )
            self.assertIn("__interrupt__", first_approval)
            self.assertEqual(first_approval["__interrupt__"][0].value["draft"], first_attempt)

            second_approval = graph.invoke(Command(resume="approve"), config=config)
            self.assertIn("__interrupt__", second_approval)
            retry_payload = second_approval["__interrupt__"][0].value
            self.assertEqual(retry_payload["previous_sql"], first_attempt)
            self.assertIn("NOT NULL constraint failed", retry_payload["previous_error"])
            self.assertIn("Title", retry_payload["regeneration_explanation"])
            self.assertIn(
                "INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (9999, 'Test Album', 1)",
                retry_payload["draft"],
            )
            self.assertIn("NOT NULL constraint failed", str(model.invocations[1]))

            executed = graph.invoke(Command(resume="approve"), config=config)
            self.assertIsNone(executed.get("execution_error"))

            with sqlite3.connect(database.database_path) as connection:
                title = connection.execute(
                    "SELECT Title FROM Album WHERE AlbumId = 9999"
                ).fetchone()
            self.assertEqual(title, ("Test Album",))

    def test_repair_budget_exhausted_aborts_with_execution_error_surfaced(self) -> None:
        """Scenario 9: repeated write failures stop at max_retries, with no further LLM calls."""
        first_attempt = "INSERT INTO Album (AlbumId, ArtistId) VALUES (9998, 1)"
        second_attempt = "INSERT INTO Album (AlbumId, ArtistId) VALUES (9997, 1)"
        model = ScriptedChatModel([AIMessage(content=first_attempt), AIMessage(content=second_attempt)])

        with mutable_fixture_database() as database:
            graph = _build_graph(database, model, intent="modification")
            config = build_thread_config()

            first_approval = graph.invoke(
                {
                    "question": "Add a new album called 'Broken' for artist 1",
                    "user_role": "admin",
                    "max_retries": 2,
                },
                config=config,
            )
            self.assertIn("__interrupt__", first_approval)

            second_approval = graph.invoke(Command(resume="approve"), config=config)
            self.assertIn("__interrupt__", second_approval)

            final_result = graph.invoke(Command(resume="approve"), config=config)

            self.assertNotIn("__interrupt__", final_result)
            self.assertIsNotNone(final_result.get("execution_error"))
            self.assertEqual(final_result.get("retry_count"), 2)
            self.assertEqual(len(model.invocations), 2)

            with sqlite3.connect(database.database_path) as connection:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM Album WHERE AlbumId IN (9998, 9997)"
                ).fetchone()[0]
            self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
