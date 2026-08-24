"""Tests for the SQL generation agent loop nodes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from nodes.sql_agent import (
    SQLAgentBudgetExhaustedNode,
    SQLAgentClarifyNode,
    SQLAgentFinalizeNode,
    SQLAgentLLMNode,
    SQLAgentToolsNode,
    _load_required_prompt,
)
from state import SQLAgentState
from tools.agent_tools import build_agent_tools
from tools.sql_safety import SQLSafetyValidator
from tests.test_db.helpers import built_database

# A small relational schema (independent of the Chinook fixture), matching the
# convention already used for tools/database tests.
_SCHEMA = (
    """
    CREATE TABLE Artist (
        ArtistId INTEGER PRIMARY KEY,
        Name NVARCHAR(120) NOT NULL
    )
    """,
    """
    CREATE TABLE Album (
        AlbumId INTEGER PRIMARY KEY,
        Title NVARCHAR(160) NOT NULL,
        ArtistId INTEGER NOT NULL REFERENCES Artist(ArtistId)
    )
    """,
    "INSERT INTO Artist (ArtistId, Name) VALUES (1, 'AC/DC'), (2, 'Accept')",
    "INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (1, 'Back In Black', 1)",
)


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


def _merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Apply a node update to state, replicating the `messages` reducer's append semantics."""
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            merged["messages"] = list(state.get("messages") or []) + list(value)
        else:
            merged[key] = value
    return merged


def _run_agent_loop(
    llm_node: SQLAgentLLMNode,
    tools_node: SQLAgentToolsNode,
    state: dict[str, Any],
    max_iterations: int = 5,
) -> dict[str, Any]:
    """Drive llm_node <-> tools_node until the model stops requesting tool calls.

    Mirrors what the graph router (Step 6) will do, without needing the graph
    itself: call the LLM, and if its last message has tool calls, run the tools
    node and loop; otherwise stop.
    """
    for _ in range(max_iterations):
        state = _merge(state, llm_node(state))
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return state
        state = _merge(state, tools_node(state))
    raise AssertionError("Agent loop did not converge within max_iterations")


class SQLAgentLLMNodeTestCase(unittest.TestCase):
    """Test the llm_call node: seeding, continuation, and iteration tracking."""

    def test_seeds_system_and_human_message_on_first_entry(self) -> None:
        """First entry (no messages yet) should seed System + Human + the response."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel([AIMessage(content="SELECT 1")])
            node = SQLAgentLLMNode(model, tools, database)

            result = node({"question": "How many artists?", "intent": "query"})

        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(type(result["messages"][0]).__name__, "SystemMessage")
        self.assertEqual(type(result["messages"][1]).__name__, "HumanMessage")
        self.assertEqual(result["messages"][1].content, "How many artists?")
        self.assertEqual(type(result["messages"][2]).__name__, "AIMessage")
        self.assertEqual(result["agent_iterations"], 1)

    def test_system_prompt_embeds_full_schema_and_intent(self) -> None:
        """The seeded system prompt should carry the complete schema and the intent."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel([AIMessage(content="SELECT 1")])
            node = SQLAgentLLMNode(model, tools, database)

            result = node({"question": "q", "intent": "modification"})

        system_prompt = result["messages"][0].content
        self.assertIn("Table: Artist", system_prompt)
        self.assertIn("Album.ArtistId -> Artist.ArtistId", system_prompt)
        self.assertIn("modification", system_prompt)

    def test_continues_existing_transcript_without_reseeding(self) -> None:
        """A non-empty transcript should only append the new response, not reseed."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel([AIMessage(content="SELECT 1")])
            node = SQLAgentLLMNode(model, tools, database)

            existing = [AIMessage(content="previous turn")]
            result = node({"messages": existing, "agent_iterations": 2})

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].content, "SELECT 1")
        self.assertEqual(result["agent_iterations"], 3)
        # The model should have seen the existing transcript, not a reseeded one.
        self.assertEqual(model.invocations[0], existing)

    def test_trims_oldest_tool_exchanges_once_transcript_exceeds_the_cap(self) -> None:
        """A very long transcript should be trimmed before the model call, keeping
        the system message and the original question, and never orphaning a
        ToolMessage — but persisted state must keep the full history regardless."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel([AIMessage(content="SELECT 1")])
            node = SQLAgentLLMNode(model, tools, database)

            system = SystemMessage(content="sys")
            human = HumanMessage(content="question")
            long_history: list[Any] = [system, human]
            for i in range(20):
                long_history.append(
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "inspect_schema", "args": {"i": i}, "id": f"c{i}"}],
                    )
                )
                long_history.append(
                    ToolMessage(content=f"result {i}", tool_call_id=f"c{i}", name="inspect_schema")
                )
            # 42 messages total, well above the 30-message context cap.

            result = node({"messages": long_history, "agent_iterations": 20})

        sent_to_model = model.invocations[0]
        self.assertLess(len(sent_to_model), len(long_history))
        self.assertIs(sent_to_model[0], system)
        self.assertIs(sent_to_model[1], human)
        self.assertIsInstance(sent_to_model[2], AIMessage)  # never starts on an orphaned ToolMessage
        # Trimming only affects what's sent to the model: the returned update
        # is still just the new response, never a rewrite of persisted history.
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].content, "SELECT 1")

    def test_binds_all_three_tools(self) -> None:
        """bind_tools should be called with inspect_schema, ask_user, run_readonly_probe."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel([AIMessage(content="SELECT 1")])
            SQLAgentLLMNode(model, tools, database)

        self.assertEqual(
            {t.name for t in model.bound_tools},
            {"inspect_schema", "ask_user", "run_readonly_probe"},
        )


class LoadRequiredPromptTestCase(unittest.TestCase):
    """Test that the sql_agent.j2 prompt has no silent fallback (unlike other nodes)."""

    def test_raises_when_prompt_file_is_missing(self) -> None:
        """A missing prompt file should fail loudly, not fall back to a built-in default."""
        missing_path = Path(tempfile.mkdtemp()) / "does-not-exist.j2"

        with self.assertRaises(FileNotFoundError):
            _load_required_prompt(missing_path)

    def test_raises_when_prompt_file_is_empty(self) -> None:
        """An empty prompt file should fail loudly, not fall back to a built-in default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as handle:
            handle.write("   \n")
            empty_path = Path(handle.name)

        try:
            with self.assertRaises(ValueError):
                _load_required_prompt(empty_path)
        finally:
            empty_path.unlink()

    def test_loads_the_real_sql_agent_prompt(self) -> None:
        """The committed sql_agent.j2 file should load successfully."""
        from nodes.sql_agent import PROMPT_PATH

        text = _load_required_prompt(PROMPT_PATH)

        self.assertIn("ask_user", text)


class SQLAgentToolsNodeTestCase(unittest.TestCase):
    """Test the tool_node: dispatch, mixed-batch rejection, probe budget, tool failures."""

    def test_dispatches_inspect_schema_call(self) -> None:
        """A single inspect_schema call should return one matching ToolMessage."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            message = AIMessage(
                content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "c1"}]
            )
            result = node({"messages": [message]})

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].tool_call_id, "c1")
        self.assertIn("Artist", result["messages"][0].content)

    def test_dispatches_run_readonly_probe_and_increments_probe_count(self) -> None:
        """A probe call should execute and bump probe_count by one."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_readonly_probe", "args": {"sql": "SELECT Name FROM Artist"}, "id": "c1"}
                ],
            )
            result = node({"messages": [message], "probe_count": 0})

        self.assertIn("AC/DC", result["messages"][0].content)
        self.assertEqual(result["probe_count"], 1)

    def test_rejects_ask_user_in_mixed_batch_but_still_executes_others(self) -> None:
        """A mixed batch (D3) should reject ask_user but still run the other tool."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_user",
                        "args": {"questions": [{"key": "genre", "question": "?", "why": "need it"}]},
                        "id": "c1",
                    },
                    {"name": "inspect_schema", "args": {}, "id": "c2"},
                ],
            )
            result = node({"messages": [message]})

        self.assertEqual(len(result["messages"]), 2)
        rejection, schema_result = result["messages"]
        self.assertEqual(rejection.tool_call_id, "c1")
        self.assertIn("alone", rejection.content)
        self.assertEqual(schema_result.tool_call_id, "c2")
        self.assertIn("Artist", schema_result.content)
        # No interrupt fires from a mixed batch: no probe budget was touched either.
        self.assertEqual(result["probe_count"], 0)

    def test_probe_budget_exhausted_returns_budget_message_without_executing(self) -> None:
        """Once max_probes is reached, further probes should not execute."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_readonly_probe", "args": {"sql": "SELECT * FROM Artist"}, "id": "c1"}
                ],
            )
            result = node({"messages": [message], "probe_count": 2, "max_probes": 2})

        self.assertIn("budget exhausted", result["messages"][0].content.lower())
        self.assertNotIn("AC/DC", result["messages"][0].content)
        self.assertEqual(result["probe_count"], 2)

    def test_identical_repeated_call_is_answered_without_reexecution(self) -> None:
        """A call identical (name + args) to an earlier one should reuse the prior result."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            prior_call = AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_readonly_probe", "args": {"sql": "SELECT Name FROM Artist"}, "id": "c0"}
                ],
            )
            prior_result = ToolMessage(content="AC/DC", tool_call_id="c0", name="run_readonly_probe")
            repeat_call = AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_readonly_probe", "args": {"sql": "SELECT Name FROM Artist"}, "id": "c1"}
                ],
            )

            result = node(
                {"messages": [prior_call, prior_result, repeat_call], "probe_count": 1}
            )

        self.assertIn("Identical call already made", result["messages"][0].content)
        self.assertIn("AC/DC", result["messages"][0].content)
        # No new execution happened: the probe budget must not have moved.
        self.assertEqual(result["probe_count"], 1)

    def test_identical_calls_in_the_same_batch_are_deduplicated(self) -> None:
        """Two identical calls within one batch should only execute the first."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_readonly_probe", "args": {"sql": "SELECT Name FROM Artist"}, "id": "c1"},
                    {"name": "run_readonly_probe", "args": {"sql": "SELECT Name FROM Artist"}, "id": "c2"},
                ],
            )
            result = node({"messages": [message], "probe_count": 0})

        self.assertEqual(len(result["messages"]), 2)
        first, second = result["messages"]
        self.assertIn("AC/DC", first.content)
        self.assertIn("Identical call already made", second.content)
        self.assertEqual(result["probe_count"], 1)

    def test_different_args_are_not_deduplicated(self) -> None:
        """A call whose arguments differ from an earlier one must execute normally."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            prior_call = AIMessage(
                content="",
                tool_calls=[{"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c0"}],
            )
            prior_result = ToolMessage(content="Artist schema", tool_call_id="c0", name="inspect_schema")
            new_call = AIMessage(
                content="",
                tool_calls=[{"name": "inspect_schema", "args": {"tables": ["Album"]}, "id": "c1"}],
            )

            result = node({"messages": [prior_call, prior_result, new_call]})

        self.assertNotIn("Identical call already made", result["messages"][0].content)
        self.assertIn("Album", result["messages"][0].content)

    def test_tool_exception_becomes_tool_message_not_crash(self) -> None:
        """A tool that raises must produce a ToolMessage, never propagate the exception."""

        @tool
        def broken_tool(x: str) -> str:
            """A tool that always raises, to exercise the error boundary."""
            raise RuntimeError("boom")

        node = SQLAgentToolsNode([broken_tool])
        message = AIMessage(
            content="", tool_calls=[{"name": "broken_tool", "args": {"x": "y"}, "id": "c1"}]
        )

        result = node({"messages": [message]})

        self.assertIn("boom", result["messages"][0].content)

    def test_unknown_tool_name_returns_error_message(self) -> None:
        """A tool call for a name the node does not know should not crash."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            node = SQLAgentToolsNode(tools)

            message = AIMessage(
                content="", tool_calls=[{"name": "does_not_exist", "args": {}, "id": "c1"}]
            )
            result = node({"messages": [message]})

        self.assertIn("Unknown tool", result["messages"][0].content)

    def test_raises_when_last_message_is_not_an_ai_message(self) -> None:
        """A graph wiring bug (non-AIMessage last) should fail loudly, not crash obscurely."""
        node = SQLAgentToolsNode([])

        with self.assertRaises(TypeError):
            node({"messages": [HumanMessage(content="not an AI message")]})


class SQLAgentClarifyNodeTestCase(unittest.TestCase):
    """Test the human-input node: interrupt payload, cancel, resume, and full replay."""

    def _ask_user_message(self) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ask_user",
                    "args": {
                        "questions": [
                            {"key": "genre", "question": "Which genre?", "why": "needed for the insert"}
                        ]
                    },
                    "id": "c1",
                }
            ],
        )

    def test_calls_interrupt_with_kind_and_questions_from_the_pending_tool_call(self) -> None:
        """The interrupt payload should carry kind=clarification and the requested questions."""
        with patch(
            "nodes.sql_agent.interrupt",
            return_value={"decision": "answer", "answers": [{"key": "genre", "answer": "Rock"}]},
        ) as mock_interrupt:
            SQLAgentClarifyNode()({"messages": [self._ask_user_message()]})

        payload = mock_interrupt.call_args[0][0]
        self.assertEqual(payload["kind"], "clarification")
        self.assertEqual(payload["questions"][0]["key"], "genre")

    def test_refuses_past_the_clarification_budget_without_interrupting(self) -> None:
        """Past max_clarifications, the node must refuse and never call interrupt()."""
        with patch("nodes.sql_agent.interrupt") as mock_interrupt:
            result = SQLAgentClarifyNode(max_clarifications=2)(
                {"messages": [self._ask_user_message()], "clarification_rounds": 2}
            )

        mock_interrupt.assert_not_called()
        self.assertIn("budget exhausted", result["messages"][0].content.lower())
        self.assertIn("assumptions", result["messages"][0].content.lower())
        self.assertEqual(result["messages"][0].tool_call_id, "c1")
        self.assertNotIn("clarification_rounds", result)
        self.assertNotIn("agent_status", result)

    def test_state_max_clarifications_overrides_constructor_default(self) -> None:
        """A max_clarifications set on state should take priority, like max_probes does."""
        with patch("nodes.sql_agent.interrupt") as mock_interrupt:
            result = SQLAgentClarifyNode(max_clarifications=10)(
                {
                    "messages": [self._ask_user_message()],
                    "clarification_rounds": 1,
                    "max_clarifications": 1,
                }
            )

        mock_interrupt.assert_not_called()
        self.assertIn("budget exhausted", result["messages"][0].content.lower())

    def test_under_budget_still_interrupts_normally(self) -> None:
        """Below the cap, the node must behave exactly as before (D3 unaffected)."""
        with patch(
            "nodes.sql_agent.interrupt",
            return_value={"decision": "answer", "answers": [{"key": "genre", "answer": "Rock"}]},
        ) as mock_interrupt:
            result = SQLAgentClarifyNode(max_clarifications=3)(
                {"messages": [self._ask_user_message()], "clarification_rounds": 1}
            )

        mock_interrupt.assert_called_once()
        self.assertEqual(result["clarification_rounds"], 2)

    def test_cancel_sets_terminal_status(self) -> None:
        """A cancel resume value should mark the run cancelled, not proceed."""
        with patch("nodes.sql_agent.interrupt", return_value={"decision": "cancel"}):
            result = SQLAgentClarifyNode()({"messages": [self._ask_user_message()]})

        self.assertEqual(result["agent_status"], "cancelled")
        self.assertTrue(result["execution_error"])

    def test_resume_appends_tool_message_and_accumulates_answers(self) -> None:
        """An answer resume value should append a ToolMessage and record clarification_answers."""
        with patch(
            "nodes.sql_agent.interrupt",
            return_value={"decision": "answer", "answers": [{"key": "genre", "answer": "Rock"}]},
        ):
            result = SQLAgentClarifyNode()(
                {"messages": [self._ask_user_message()], "clarification_rounds": 0}
            )

        self.assertEqual(len(result["messages"]), 1)
        tool_message = result["messages"][0]
        self.assertEqual(tool_message.tool_call_id, "c1")
        self.assertIn("Rock", tool_message.content)
        self.assertEqual(
            result["clarification_answers"],
            [{"key": "genre", "question": "Which genre?", "answer": "Rock"}],
        )
        self.assertEqual(result["clarification_rounds"], 1)

    def test_raises_when_no_ask_user_call_present(self) -> None:
        """A malformed call (graph wiring bug) should fail loudly, not silently."""
        message = AIMessage(
            content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "c1"}]
        )
        with self.assertRaises(ValueError):
            SQLAgentClarifyNode()({"messages": [message]})

    def test_raises_when_last_message_is_not_an_ai_message(self) -> None:
        """A graph wiring bug (non-AIMessage last) should fail loudly, not crash obscurely."""
        with self.assertRaises(TypeError):
            SQLAgentClarifyNode()({"messages": [HumanMessage(content="not an AI message")]})

    def test_ask_user_alone_reaches_the_interrupt_end_to_end(self) -> None:
        """Full round trip through a real compiled graph: pause, then resume with answers."""
        builder = StateGraph(SQLAgentState)
        builder.add_node("clarify", SQLAgentClarifyNode())
        builder.add_edge(START, "clarify")
        builder.add_edge("clarify", END)
        graph = builder.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-thread"}}

        paused = graph.invoke({"messages": [self._ask_user_message()]}, config=config)

        self.assertIn("__interrupt__", paused)
        interrupt_payload = paused["__interrupt__"][0].value
        self.assertEqual(interrupt_payload["kind"], "clarification")

        resumed = graph.invoke(
            Command(resume={"decision": "answer", "answers": [{"key": "genre", "answer": "Rock"}]}),
            config=config,
        )

        self.assertNotIn("__interrupt__", resumed)
        self.assertEqual(len(resumed["messages"]), 2)
        self.assertEqual(resumed["messages"][1].content, "genre: Rock")
        self.assertEqual(resumed["clarification_rounds"], 1)


class SQLAgentFinalizeNodeTestCase(unittest.TestCase):
    """Test extraction of the final SQL statement (and repair-pass state resets)."""

    def test_unescapes_literal_backslash_n_between_sql_clauses(self) -> None:
        """A model that types literal backslash-n instead of real newlines should still parse."""
        state = {
            "messages": [
                AIMessage(
                    content=(
                        "SELECT a.Name AS ArtistName\\nFROM Artist a\\n"
                        "JOIN Album al ON al.ArtistId = a.ArtistId\\n"
                        "ORDER BY a.Name\\nLIMIT 1;"
                    )
                )
            ]
        }

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(
            result["generated_sql"],
            "SELECT a.Name AS ArtistName\n"
            "FROM Artist a\n"
            "JOIN Album al ON al.ArtistId = a.ArtistId\n"
            "ORDER BY a.Name\n"
            "LIMIT 1;",
        )
        self.assertNotIn("\\n", result["generated_sql"])

    def test_unescapes_literal_backslash_n_before_rationale_marker(self) -> None:
        """The rationale split must still work when the separator itself is a literal backslash-n."""
        state = {
            "messages": [
                AIMessage(content="SELECT 1\\n\\nRationale: trivial query")
            ]
        }

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["generated_sql"], "SELECT 1")
        self.assertEqual(result["agent_rationale"], "trivial query")

    def test_real_newlines_are_left_untouched(self) -> None:
        """A well-formed response with real newlines should be unaffected."""
        state = {"messages": [AIMessage(content="SELECT 1\nFROM Artist\nLIMIT 1")]}

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["generated_sql"], "SELECT 1\nFROM Artist\nLIMIT 1")

    def test_extracts_sql_and_sets_final_status(self) -> None:
        """A plain SQL final answer should populate generated_sql and agent_status."""
        state = {"messages": [AIMessage(content="SELECT * FROM Artist")]}

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["generated_sql"], "SELECT * FROM Artist")
        self.assertEqual(result["agent_status"], "final")
        self.assertEqual(result["agent_rationale"], "")

    def test_extracts_rationale_when_present(self) -> None:
        """A trailing Rationale: line should be split off into agent_rationale."""
        state = {
            "messages": [
                AIMessage(content="SELECT * FROM Artist\n\nRationale: lists every artist row")
            ]
        }

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["generated_sql"], "SELECT * FROM Artist")
        self.assertEqual(result["agent_rationale"], "lists every artist row")

    def test_strips_code_fences(self) -> None:
        """Markdown fences around the SQL should be removed, like every other node."""
        state = {"messages": [AIMessage(content="```sql\nSELECT 1\n```")]}

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["generated_sql"], "SELECT 1")

    def test_empty_content_fails_closed(self) -> None:
        """An empty final answer should fail rather than produce an empty statement."""
        state = {"messages": [AIMessage(content="")]}

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["agent_status"], "failed")
        self.assertTrue(result["agent_error"])

    def test_non_sql_content_fails_closed(self) -> None:
        """Prose instead of SQL should fail rather than being treated as a statement."""
        state = {"messages": [AIMessage(content="I'm not sure how to answer that.")]}

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["agent_status"], "failed")
        self.assertTrue(result["agent_error"])

    def test_repair_pass_sets_previous_sql_and_resets_validation_state(self) -> None:
        """On a repair pass (retry_count > 0), mirror the retired fallback regenerator (D6)."""
        state = {
            "messages": [AIMessage(content="SELECT * FROM Artist\n\nRationale: fixed the typo")],
            "generated_sql": "SELECT * FROM Artst",
            "retry_count": 1,
            "validated_sql": "SELECT * FROM Artst",
            "sql_validation_error": "some prior error",
            "execution_error": "no such table: Artst",
            "execution_confirmed": True,
        }

        result = SQLAgentFinalizeNode()(state)

        self.assertEqual(result["generated_sql"], "SELECT * FROM Artist")
        self.assertEqual(result["previous_sql"], "SELECT * FROM Artst")
        self.assertEqual(result["regeneration_explanation"], "fixed the typo")
        self.assertEqual(result["validated_sql"], "")
        self.assertIsNone(result["sql_validation_error"])
        self.assertIsNone(result["execution_error"])
        self.assertFalse(result["execution_confirmed"])


class SQLAgentBudgetExhaustedNodeTestCase(unittest.TestCase):
    """Test the wrap-up node reached when the model still wants tools past the iteration cap."""

    def test_asks_unbound_model_and_reports_terminal_status(self) -> None:
        """The node should invoke the model unbound and surface a terminal explanation."""
        model = ScriptedChatModel(
            [
                AIMessage(
                    content=(
                        "I found the Artist and Album tables and confirmed the artist exists, "
                        "but could not settle on the right aggregation before running out of "
                        "iterations. I reached my iteration limit and could not complete the request."
                    )
                )
            ]
        )
        state = {
            "messages": [
                HumanMessage(content="question"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_readonly_probe", "args": {"sql": "SELECT 1"}, "id": "c1"}],
                ),
            ]
        }

        result = SQLAgentBudgetExhaustedNode(model)(state)

        self.assertEqual(result["agent_status"], "budget_exhausted")
        self.assertTrue(result["agent_error"])
        self.assertIn("iteration limit", result["analysis"])
        # bind_tools must never be called: the model must be physically unable to emit a tool call.
        self.assertIsNone(model.bound_tools)

    def test_appends_wrap_up_exchange_to_messages(self) -> None:
        """The wrap-up request/response pair should be appended, not replace, the transcript."""
        model = ScriptedChatModel([AIMessage(content="Here is what I found so far...")])
        original_messages = [HumanMessage(content="question")]
        state = {"messages": list(original_messages)}

        result = SQLAgentBudgetExhaustedNode(model)(state)

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0].content, model.invocations[0][-1].content)
        self.assertEqual(result["messages"][1].content, "Here is what I found so far...")

    def test_invokes_model_with_full_transcript_plus_wrap_up_request(self) -> None:
        """The model should see everything gathered so far, not just the wrap-up instruction."""
        model = ScriptedChatModel([AIMessage(content="Summary.")])
        transcript = [
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "c1"}]),
        ]
        state = {"messages": list(transcript)}

        SQLAgentBudgetExhaustedNode(model)(state)

        sent_messages = model.invocations[0]
        self.assertEqual(len(sent_messages), len(transcript) + 1)
        self.assertEqual(sent_messages[: len(transcript)], transcript)


class SQLAgentScenarioTestCase(unittest.TestCase):
    """Scenario tests driving llm_call <-> tool_node together, like the graph will."""

    def test_immediate_final_answer_with_zero_tool_calls(self) -> None:
        """The model can finalize on the very first turn, with no tool calls at all."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel([AIMessage(content="SELECT COUNT(*) FROM Artist")])
            llm_node = SQLAgentLLMNode(model, tools, database)
            tools_node = SQLAgentToolsNode(tools)

            final_state = _run_agent_loop(
                llm_node, tools_node, {"question": "How many artists?", "intent": "query"}
            )

        self.assertEqual(final_state["agent_iterations"], 1)
        self.assertEqual(final_state["messages"][-1].content, "SELECT COUNT(*) FROM Artist")
        self.assertFalse(final_state["messages"][-1].tool_calls)

    def test_two_inspect_schema_rounds_then_final(self) -> None:
        """The model can call inspect_schema twice before settling on a final answer."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel(
                [
                    AIMessage(
                        content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "c1"}]
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c2"}
                        ],
                    ),
                    AIMessage(content="SELECT Name FROM Artist"),
                ]
            )
            llm_node = SQLAgentLLMNode(model, tools, database)
            tools_node = SQLAgentToolsNode(tools)

            final_state = _run_agent_loop(
                llm_node, tools_node, {"question": "List artist names", "intent": "query"}
            )

        self.assertEqual(final_state["agent_iterations"], 3)
        self.assertEqual(final_state["messages"][-1].content, "SELECT Name FROM Artist")
        tool_messages = [m for m in final_state["messages"] if type(m).__name__ == "ToolMessage"]
        self.assertEqual(len(tool_messages), 2)

    def test_probe_error_fed_back_then_corrected(self) -> None:
        """A failed probe should come back as feedback, and the model can recover from it."""
        with built_database(_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            model = ScriptedChatModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "run_readonly_probe",
                                "args": {"sql": "SELECT * FROM DoesNotExistTable"},
                                "id": "c1",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "run_readonly_probe",
                                "args": {"sql": "SELECT * FROM Artist"},
                                "id": "c2",
                            }
                        ],
                    ),
                    AIMessage(content="SELECT * FROM Artist"),
                ]
            )
            llm_node = SQLAgentLLMNode(model, tools, database)
            tools_node = SQLAgentToolsNode(tools)

            final_state = _run_agent_loop(
                llm_node, tools_node, {"question": "Show all artists", "intent": "query"}
            )

        tool_messages = [m for m in final_state["messages"] if type(m).__name__ == "ToolMessage"]
        self.assertTrue(tool_messages[0].content.startswith("Rejected:"))
        self.assertFalse(tool_messages[1].content.startswith("Rejected:"))
        self.assertEqual(final_state["probe_count"], 2)
        self.assertEqual(final_state["messages"][-1].content, "SELECT * FROM Artist")


if __name__ == "__main__":
    unittest.main()
