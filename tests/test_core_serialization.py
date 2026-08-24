"""Tests for message serialization and tool-log derivation in core.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core import _derive_agent_tool_log, _serialize_messages, _serialize_state


class SerializeMessagesTestCase(unittest.TestCase):
    """Test _serialize_messages converts LangChain messages into plain dicts."""

    def test_empty_list_returns_empty_list(self) -> None:
        """No messages should serialize to an empty list."""
        self.assertEqual(_serialize_messages([]), [])

    def test_system_and_human_messages_serialize_without_tool_fields(self) -> None:
        """Plain text messages should carry role/content and no tool metadata."""
        result = _serialize_messages(
            [SystemMessage(content="You are a SQL agent."), HumanMessage(content="Top 5 artists?")]
        )
        self.assertEqual(
            result,
            [
                {
                    "role": "system",
                    "content": "You are a SQL agent.",
                    "tool_calls": None,
                    "tool_call_id": None,
                },
                {
                    "role": "human",
                    "content": "Top 5 artists?",
                    "tool_calls": None,
                    "tool_call_id": None,
                },
            ],
        )

    def test_ai_message_with_tool_calls_serializes_tool_calls(self) -> None:
        """An AIMessage requesting tool calls should expose name/args/id per call."""
        message = AIMessage(
            content="",
            tool_calls=[{"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "call-1"}],
        )
        result = _serialize_messages([message])
        self.assertEqual(result[0]["role"], "ai")
        self.assertEqual(
            result[0]["tool_calls"],
            [{"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "call-1"}],
        )
        self.assertIsNone(result[0]["tool_call_id"])

    def test_ai_message_without_tool_calls_has_none_tool_calls(self) -> None:
        """A final-answer AIMessage (no tool calls) should carry tool_calls=None."""
        result = _serialize_messages([AIMessage(content="SELECT 1")])
        self.assertIsNone(result[0]["tool_calls"])

    def test_tool_message_serializes_tool_call_id(self) -> None:
        """A ToolMessage should carry its tool_call_id and no tool_calls."""
        message = ToolMessage(content="8 columns, 2 incoming FKs", tool_call_id="call-1", name="inspect_schema")
        result = _serialize_messages([message])
        self.assertEqual(result[0]["role"], "tool")
        self.assertEqual(result[0]["content"], "8 columns, 2 incoming FKs")
        self.assertIsNone(result[0]["tool_calls"])
        self.assertEqual(result[0]["tool_call_id"], "call-1")

    def test_output_is_json_serializable(self) -> None:
        """Serialized messages must round-trip through json.dumps without error."""
        import json

        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"name": "run_readonly_probe", "args": {"sql": "SELECT 1"}, "id": "c1"}]),
            ToolMessage(content="1 row", tool_call_id="c1", name="run_readonly_probe"),
            AIMessage(content="SELECT 1"),
        ]
        json.dumps(_serialize_messages(messages))


class DeriveAgentToolLogTestCase(unittest.TestCase):
    """Test _derive_agent_tool_log builds the UI-facing log from the transcript."""

    def test_empty_transcript_returns_empty_log(self) -> None:
        """No messages should produce an empty tool log."""
        self.assertEqual(_derive_agent_tool_log([]), [])

    def test_final_answer_with_no_tool_calls_produces_a_generate_sql_entry(self) -> None:
        """A final AIMessage with no tool calls should surface as a generate_sql step."""
        messages = [SystemMessage(content="sys"), HumanMessage(content="q"), AIMessage(content="SELECT 1")]
        self.assertEqual(
            _derive_agent_tool_log(messages),
            [{"iteration": 1, "tool": "generate_sql", "arguments": {}, "result": "SELECT 1"}],
        )

    def test_empty_final_answer_produces_no_entry(self) -> None:
        """An AIMessage with neither tool calls nor content should not add a log entry."""
        messages = [AIMessage(content="")]
        self.assertEqual(_derive_agent_tool_log(messages), [])

    def test_single_tool_call_paired_with_its_result(self) -> None:
        """A tool call followed by its ToolMessage should pair into one entry."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c1"}],
            ),
            ToolMessage(content="8 columns, 2 incoming FKs", tool_call_id="c1", name="inspect_schema"),
        ]
        log = _derive_agent_tool_log(messages)
        self.assertEqual(
            log,
            [
                {
                    "iteration": 1,
                    "tool": "inspect_schema",
                    "arguments": {"tables": ["Artist"]},
                    "result": "8 columns, 2 incoming FKs",
                }
            ],
        )

    def test_call_awaiting_result_has_none_result(self) -> None:
        """A tool call not yet matched by a ToolMessage should keep result=None."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "ask_user", "args": {"questions": []}, "id": "c1"}],
            )
        ]
        log = _derive_agent_tool_log(messages)
        self.assertEqual(log[0]["result"], None)

    def test_iteration_increments_per_ai_message(self) -> None:
        """Each AIMessage turn should bump the iteration number for its calls."""
        messages = [
            AIMessage(content="", tool_calls=[{"name": "inspect_schema", "args": {}, "id": "c1"}]),
            ToolMessage(content="schema", tool_call_id="c1", name="inspect_schema"),
            AIMessage(content="", tool_calls=[{"name": "run_readonly_probe", "args": {"sql": "SELECT 1"}, "id": "c2"}]),
            ToolMessage(content="1 row", tool_call_id="c2", name="run_readonly_probe"),
            AIMessage(content="SELECT 1"),
        ]
        log = _derive_agent_tool_log(messages)
        self.assertEqual([entry["iteration"] for entry in log], [1, 2, 3])

    def test_multiple_tool_calls_in_one_batch_share_iteration(self) -> None:
        """Two tool calls in the same AIMessage turn should share one iteration."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c1"},
                    {"name": "inspect_schema", "args": {"tables": ["Album"]}, "id": "c2"},
                ],
            ),
            ToolMessage(content="Artist schema", tool_call_id="c1", name="inspect_schema"),
            ToolMessage(content="Album schema", tool_call_id="c2", name="inspect_schema"),
        ]
        log = _derive_agent_tool_log(messages)
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]["iteration"], 1)
        self.assertEqual(log[1]["iteration"], 1)
        self.assertEqual(log[0]["result"], "Artist schema")
        self.assertEqual(log[1]["result"], "Album schema")


class SerializeStateAgentFieldsTestCase(unittest.TestCase):
    """Test _serialize_state carries the agent transcript and derived fields."""

    def test_serialize_state_includes_serialized_messages_and_tool_log(self) -> None:
        """The public state dict should carry plain-dict messages and a derived tool log."""
        state = {
            "question": "add a new artist called X",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "inspect_schema", "args": {"tables": ["Artist"]}, "id": "c1"}],
                ),
                ToolMessage(content="1 incoming FK from Album", tool_call_id="c1", name="inspect_schema"),
            ],
            "agent_status": "needs_clarification",
            "agent_iterations": 1,
            "max_agent_iterations": 8,
            "probe_count": 0,
            "max_probes": 6,
            "clarification_rounds": 0,
            "max_clarifications": 3,
            "clarification_answers": [],
            "agent_rationale": None,
            "agent_error": None,
        }
        result = _serialize_state(state, "add a new artist called X")

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "ai")
        self.assertEqual(result["agent_status"], "needs_clarification")
        self.assertEqual(result["agent_iterations"], 1)
        self.assertEqual(result["max_agent_iterations"], 8)
        self.assertEqual(
            result["agent_tool_log"],
            [
                {
                    "iteration": 1,
                    "tool": "inspect_schema",
                    "arguments": {"tables": ["Artist"]},
                    "result": "1 incoming FK from Album",
                }
            ],
        )

    def test_serialize_state_defaults_agent_fields_when_absent(self) -> None:
        """A state with no agent keys should still produce well-typed defaults."""
        result = _serialize_state({"question": "hello"}, "hello")
        self.assertEqual(result["messages"], [])
        self.assertEqual(result["agent_tool_log"], [])
        self.assertIsNone(result["agent_status"])
        self.assertEqual(result["agent_iterations"], 0)
        self.assertEqual(result["clarification_answers"], [])

    def test_serialize_state_output_is_json_serializable(self) -> None:
        """The full response dict, including the transcript, must be JSON-serializable."""
        import json

        state = {
            "question": "q",
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="q"),
                AIMessage(content="", tool_calls=[{"name": "run_readonly_probe", "args": {"sql": "SELECT 1"}, "id": "c1"}]),
                ToolMessage(content="1 row", tool_call_id="c1", name="run_readonly_probe"),
                AIMessage(content="SELECT 1"),
            ],
        }
        json.dumps(_serialize_state(state, "q"))


if __name__ == "__main__":
    unittest.main()
