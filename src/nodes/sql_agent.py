"""SQL generation agent loop: llm_call / tool_node / clarify / finalize.

Four LangGraph nodes replacing the single-shot `SQLGeneratorNode`. The model
decides on its own how many times to inspect the schema, ask the user for
business details, and probe the database read-only before committing to a
final SQL statement. `messages` (see `state.SQLAgentState`) is the transcript;
these nodes only ever append to it, never overwrite it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from state import SQLAgentState
from tools.database import SQLiteDatabase, format_full_schema
from utils.llm import extract_text_from_response, strip_code_fences
from utils.nodes import ToolCallingChatModel, render_prompt

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "sql_agent.j2"

# Placeholders; Step 10 of the agentic SQL generation plan wires these to env vars.
DEFAULT_MAX_PROBES = 6
DEFAULT_MAX_CLARIFICATIONS = 3

_RATIONALE_MARKER = re.compile(r"(?im)^[ \t]*rationale[ \t]*:[ \t]*")
_LEADING_SQL_PATTERN = re.compile(r"(?i)^\s*(select|insert|update|delete|with)\b")


def _unescape_literal_whitespace(text: str) -> str:
    """Repair a model output artifact: literal backslash-n/backslash-t typed as
    text instead of real whitespace.

    Tool-call arguments go through structured JSON decoding, where `\\n`
    always means a real newline. The model's *final plain-text* answer has no
    such decoding step, and this model occasionally types the literal
    two-character sequence `\\n` where it means a line break (seen consistently
    in agent traces: a probe's tool-call SQL is fine, but the same query
    retyped as the final answer has `\\n` for real). SQLite has no
    backslash-escape syntax at all — a bare `\\` is always a syntax error, in
    or out of a string literal — so this replacement is safe unconditionally,
    with no risk of corrupting an intentional character in valid SQL.

    Args:
        text: The raw final-answer text, before SQL/rationale splitting.

    Returns:
        str: The same text with literal `\\r\\n` / `\\n` / `\\t` sequences
        replaced by real whitespace.
    """
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _last_ai_message(state: SQLAgentState) -> AIMessage:
    """Return the last message, narrowed to the AIMessage every caller expects.

    `messages` is typed as `AnyMessage` (a union of every LangChain message
    type), but the graph router (Step 6) only sends control to
    `sql_agent_tools` / `sql_agent_clarify` when the last message is an
    AIMessage with tool calls. This narrows that for callers and fails fast if
    the invariant is ever violated.

    Args:
        state: Current graph state.

    Returns:
        AIMessage: The last message in the transcript.

    Raises:
        TypeError: If the last message is not an AIMessage (a graph wiring bug).
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(
            f"Expected the last message to be an AIMessage, got {type(last_message).__name__}."
        )
    return last_message


class SQLAgentLLMNode:
    """The loop's `llm_call`: seed the transcript on first entry, else continue it."""

    def __init__(
        self,
        model: ToolCallingChatModel,
        tools: Sequence[BaseTool],
        database: SQLiteDatabase,
        prompt_template: str | None = None,
        max_schema_chars: int | None = None,
    ) -> None:
        """Initialize the SQLAgentLLMNode.

        Args:
            model: Tool-calling capable chat model (see utils.nodes.ToolCallingChatModel).
            tools: Tools the model may call (inspect_schema, ask_user, run_readonly_probe).
            database: Database used to seed the complete schema on first entry.
            prompt_template: Optional override of the rendered sql_agent.j2 system prompt.
            max_schema_chars: Optional cap forwarded to format_full_schema (Step 1).
        """
        self.model = model
        self.bound_model = model.bind_tools(list(tools))
        self.database = database
        self.prompt_template = prompt_template or _load_required_prompt(PROMPT_PATH)
        self.max_schema_chars = max_schema_chars

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """Seed the transcript on first entry, else continue it, then call the model.

        Re-entered from three places (after sql_agent_tools, after
        sql_agent_clarify, and after a failed sql_executor), and the body is
        identical for all of them: `messages` already carries the context.

        Args:
            state: Current graph state.

        Returns:
            dict[str, Any]: `messages` update (new messages only — the reducer
            appends them) plus the incremented `agent_iterations`.
        """
        existing_messages = list(state.get("messages") or [])

        if not existing_messages:
            schema_overview, schema_truncated = format_full_schema(
                self.database, max_chars=self.max_schema_chars
            )
            system_prompt = render_prompt(
                self.prompt_template,
                intent=state.get("intent") or "query",
                schema_overview=schema_overview,
                schema_truncated=schema_truncated,
            )
            seed_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.get("question", "")),
            ]
            response = self.bound_model.invoke(seed_messages)
            return {
                "messages": [*seed_messages, response],
                "agent_iterations": state.get("agent_iterations", 0) + 1,
            }

        response = self.bound_model.invoke(existing_messages)
        return {
            "messages": [response],
            "agent_iterations": state.get("agent_iterations", 0) + 1,
        }


class SQLAgentToolsNode:
    """The loop's `tool_node`: dispatch inspect_schema / run_readonly_probe calls.

    An `ask_user` call reaches this node only as part of a mixed batch (a lone
    `ask_user` call is routed to `sql_agent_clarify` instead) — see D3 in the
    agentic SQL generation plan. Here it is rejected with instructions to retry
    alone, never executed.
    """

    def __init__(self, tools: Sequence[BaseTool], max_probes: int = DEFAULT_MAX_PROBES) -> None:
        """Initialize the SQLAgentToolsNode.

        Args:
            tools: The tools to dispatch calls to, keyed by their `.name`.
            max_probes: Default probe budget, used when the state has none set.
        """
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.default_max_probes = max_probes

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """Execute every tool call requested by the last AIMessage.

        Args:
            state: Current graph state. The last message must be an `AIMessage`
                with at least one tool call.

        Returns:
            dict[str, Any]: One `ToolMessage` per requested tool call (in
            order, matching `tool_call_id`), plus the updated `probe_count`.
        """
        last_message = _last_ai_message(state)
        probe_count = state.get("probe_count", 0)
        max_probes = state.get("max_probes", self.default_max_probes)

        tool_messages: list[ToolMessage] = []
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            call_id = tool_call["id"]

            if name == "ask_user":
                tool_messages.append(
                    ToolMessage(
                        content=(
                            "ask_user must be called alone, with no other tool calls in "
                            "the same turn. Retry with only ask_user."
                        ),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue

            if name == "run_readonly_probe" and probe_count >= max_probes:
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Probe budget exhausted ({max_probes} probes used). "
                            "Stop probing and finalize your answer with what you already know."
                        ),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue

            tool = self.tools_by_name.get(name)
            if tool is None:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool: {name}.", tool_call_id=call_id, name=name)
                )
                continue

            if name == "run_readonly_probe":
                probe_count += 1

            try:
                result = tool.invoke(tool_call.get("args", {}))
            except Exception as exc:  # a tool bug must become feedback, never a crash
                result = f"Tool {name} failed: {exc}"

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id, name=name))

        return {"messages": tool_messages, "probe_count": probe_count}


class SQLAgentClarifyNode:
    """The human-input node (D3): reached only when `ask_user` is the sole tool call.

    The body is side-effect-free apart from the `interrupt()` call, so replay
    on resume is free: LangGraph re-runs this node from the top, and the same
    tool call is re-extracted identically before `interrupt()` returns the
    resume value instead of raising.
    """

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """Pause for user input, then fold the answers back into the transcript.

        Args:
            state: Current graph state. The last message must be an `AIMessage`
                whose sole tool call is `ask_user`.

        Returns:
            dict[str, Any]: On cancel, a terminal `agent_status`/`execution_error`.
            On answer, a `ToolMessage` appended to `messages`, the accumulated
            `clarification_answers`, and the incremented `clarification_rounds`.
        """
        last_message = _last_ai_message(state)
        tool_call = _find_tool_call(last_message, "ask_user")
        questions = tool_call.get("args", {}).get("questions") or []

        resume_value = interrupt({"kind": "clarification", "questions": questions})

        decision = resume_value.get("decision") if isinstance(resume_value, dict) else None
        if decision == "cancel":
            return {
                "agent_status": "cancelled",
                "execution_error": "The user cancelled the clarification request.",
            }

        raw_answers = resume_value.get("answers", []) if isinstance(resume_value, dict) else []
        answer_entries = _merge_questions_and_answers(questions, raw_answers)

        return {
            "messages": [
                ToolMessage(
                    content=_format_clarification_answers(answer_entries),
                    tool_call_id=tool_call["id"],
                    name="ask_user",
                )
            ],
            "clarification_answers": answer_entries,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }


class SQLAgentFinalizeNode:
    """Extract the agent's final SQL statement (and optional rationale) from the transcript."""

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """Turn the last AIMessage into `generated_sql`, or a terminal failure.

        Args:
            state: Current graph state. The last message must be an `AIMessage`
                with no tool calls.

        Returns:
            dict[str, Any]: On success, `generated_sql`, `agent_status="final"`,
            and `agent_rationale`; on a repair pass (`retry_count > 0`), also
            `previous_sql` / `regeneration_explanation` and a reset of
            `validated_sql` / `sql_validation_error` / `execution_error` /
            `execution_confirmed`, mirroring the retired fallback regenerator
            (D6) so the approval-dialog diff UI keeps working unchanged. On
            failure, `agent_status="failed"` and `agent_error`.
        """
        messages = state.get("messages") or []
        raw_text = extract_text_from_response(messages[-1]) if messages else ""
        raw_text = _unescape_literal_whitespace(raw_text)

        sql_part, rationale = _split_sql_and_rationale(raw_text)
        generated_sql = strip_code_fences(sql_part, language_prefix="sql")

        if not generated_sql or not _LEADING_SQL_PATTERN.match(generated_sql):
            return {
                "agent_status": "failed",
                "agent_error": (
                    "The agent finished without returning a SQL statement: "
                    f"{raw_text!r}"
                ),
            }

        update: dict[str, Any] = {
            "generated_sql": generated_sql,
            "agent_status": "final",
            "agent_rationale": rationale,
        }

        if state.get("retry_count", 0) > 0:
            update["previous_sql"] = state.get("generated_sql", "")
            update["regeneration_explanation"] = rationale
            update["validated_sql"] = ""
            update["sql_validation_error"] = None
            update["execution_error"] = None
            update["execution_confirmed"] = False

        return update


class SQLAgentBudgetExhaustedNode:
    """Explain what the agent learned when it runs out of iterations mid-task.

    Reached only when the model still wants to call more tools past the
    iteration cap — a clean finalize (no tool calls left) is never blocked by
    the budget, see `_route_after_agent_llm` in graph.py. Rather than the run
    silently dying with nothing to show, this asks the model to turn the
    transcript gathered so far into a plain-language explanation.
    """

    _WRAP_UP_INSTRUCTION = (
        "You have used all of your allowed iterations without producing a final "
        "SQL statement. Do not call any tools - none are available anymore. "
        "Write a short, plain-language explanation for the user covering: "
        "(1) what you learned from the schema and the tool calls you already "
        "made, (2) what specifically made this request hard to finish, and "
        "(3) state clearly that you reached your iteration limit and could not "
        "complete the request."
    )

    def __init__(self, model: ToolCallingChatModel) -> None:
        """Initialize the SQLAgentBudgetExhaustedNode.

        Args:
            model: The same model driving the loop, invoked **unbound** (no
                `bind_tools`) here so it is physically unable to emit another
                tool call regardless of what the prompt says.
        """
        self.model = model

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """Ask the model to summarize its progress and explain the limit.

        Args:
            state: Current graph state; `messages` holds the full transcript
                of schema, probes, and reasoning gathered so far.

        Returns:
            dict[str, Any]: The wrap-up exchange appended to `messages`,
            `agent_status="budget_exhausted"`, a terminal `agent_error`, and
            `analysis` — read directly as the user-facing answer, mirroring
            how `intent_error`/`authorization_error` bypass `result_analyst`
            since there is no `query_result` to analyze here.
        """
        transcript = list(state.get("messages") or [])
        wrap_up_request = HumanMessage(content=self._WRAP_UP_INSTRUCTION)
        response = self.model.invoke([*transcript, wrap_up_request])
        explanation = extract_text_from_response(response)

        return {
            "messages": [wrap_up_request, response],
            "agent_status": "budget_exhausted",
            "agent_error": (
                "The agent reached its iteration limit before producing a final SQL statement."
            ),
            "analysis": explanation,
        }


def _load_required_prompt(path: Path) -> str:
    """Load the sql_agent.j2 system prompt, with no hardcoded fallback.

    Unlike the other nodes' `load_prompt(path, DEFAULT_PROMPT)` pattern, this
    prompt encodes rules that are load-bearing for the agent's safety
    (probe-before-finalize, ask_user-alone, INSERT/UPDATE/DELETE
    pre-conditions) — silently falling back to a weaker built-in prompt would
    hide a missing or empty file instead of failing loudly.

    Args:
        path: Path to the sql_agent.j2 prompt file.

    Returns:
        str: The prompt template text.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt file exists but is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required prompt file is missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Required prompt file is empty: {path}")
    return text


def _find_tool_call(message: AIMessage, name: str) -> ToolCall:
    """Find the tool call with the given name on an AIMessage.

    Args:
        message: The AIMessage to search.
        name: The tool name to look for.

    Returns:
        ToolCall: The matching tool call.

    Raises:
        ValueError: If no tool call with that name is present (a graph wiring bug).
    """
    for tool_call in message.tool_calls:
        if tool_call["name"] == name:
            return tool_call
    raise ValueError(f"Expected a {name!r} tool call on the last message, found none.")


def _merge_questions_and_answers(
    questions: list[dict[str, Any]],
    raw_answers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair each asked question with its answer, keyed by `key`.

    Args:
        questions: The `{key, question, why, suggested_default?}` dicts the
            model requested via `ask_user`.
        raw_answers: The `{key, answer}` dicts resumed from the interrupt.

    Returns:
        list[dict[str, Any]]: `{key, question, answer}` entries, in the order
        the questions were asked.
    """
    answers_by_key = {
        item["key"]: item.get("answer", item.get("value", ""))
        for item in raw_answers
        if isinstance(item, dict) and "key" in item
    }
    return [
        {
            "key": question.get("key", ""),
            "question": question.get("question", ""),
            "answer": answers_by_key.get(question.get("key", ""), ""),
        }
        for question in questions
        if isinstance(question, dict)
    ]


def _format_clarification_answers(entries: list[dict[str, Any]]) -> str:
    """Render clarification answers as the ask_user tool's ToolMessage content.

    Args:
        entries: `{key, question, answer}` dicts.

    Returns:
        str: Plain text summary for the model's next turn.
    """
    if not entries:
        return "The user did not provide any answers."
    return "\n".join(f"{entry['key']}: {entry['answer']}" for entry in entries)


def _split_sql_and_rationale(content: str) -> tuple[str, str]:
    """Split a final answer into its SQL statement and optional rationale.

    Args:
        content: The raw AIMessage text content.

    Returns:
        tuple[str, str]: `(sql_part, rationale)`; `rationale` is `""` when the
        model did not include a `Rationale:` line.
    """
    match = _RATIONALE_MARKER.search(content)
    if not match:
        return content.strip(), ""
    return content[: match.start()].strip(), content[match.end() :].strip()
