"""Regenerate SQL after a database execution failure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nodes.sql_generator import LLM
from state import SQLAgentState
from utils.llm import extract_text_from_response, strip_code_fences
from utils.nodes import load_prompt, render_prompt

PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "sql_fallback_regenerator.j2"
)
DEFAULT_PROMPT = """You repair a failed SQLite statement.

Original request (background context only):
{{ question }}

Database schema:
{{ schema_overview }}

Failed SQL approved by the user (authoritative repair target):
{{ previous_sql }}

Execution error:
{{ execution_error }}

Treat the failed SQL as the user's latest intent, because it may contain edits
that supersede the original request. Fix only the specific execution error.
Preserve the operation and user-approved data semantics, including literal,
inserted, assignment, and filter values and the rows targeted. Change table or
column identifiers and SQLite syntax only when necessary to fix the reported
error. Never replace data or behavior with content from the original request.
If the two conflict, the failed SQL wins. Use only tables and columns from the
schema. Return exactly one corrected SQLite statement without explanations,
comments, or Markdown fences.
"""


@dataclass(frozen=True, slots=True)
class SQLRegenerationContext:
    """Inputs required to repair a failed SQL statement."""

    question: str
    schema_overview: str
    previous_sql: str
    execution_error: str
    retry_count: int
    max_retries: int


def _build_regeneration_context(state: SQLAgentState) -> SQLRegenerationContext:
    """Build prompt context from the failed execution state.

    Args:
        state (SQLAgentState): Current state of the graph

    Returns:
        SQLRegenerationContext: Context with all the information needed 
        to regenerate the SQL query
    """

    return SQLRegenerationContext(
        question=state.get("question", ""),
        schema_overview=state.get("schema_overview", ""),
        previous_sql=state.get("generated_sql", ""),
        execution_error=(
            state.get("last_execution_error") or state.get("execution_error") or ""
        ),
        retry_count=state.get("retry_count",0)+1,
        max_retries=state.get("max_retries",3)
    )


def _render_regeneration_prompt(context: SQLRegenerationContext) -> str:
    """Render the fallback prompt without discarding its placeholders.

    Args:
        context (SQLRegenerationContext): Context with all the data needed to
        regenerate the SQL query

    Returns:
        str: Prompt ready to use.
    """

    template = load_prompt(PROMPT_PATH, DEFAULT_PROMPT)
    return render_prompt(
        template,
        question=context.question,
        schema_overview=context.schema_overview,
        previous_sql=context.previous_sql,
        execution_error=context.execution_error,
    )


def _explain_regeneration(
    context: SQLRegenerationContext,
    regenerated_sql: str,
) -> str:
    """Explain why a retry exists without inventing database behavior.

    Args:
        context (SQLRegenerationContext): Context of the previous SQL execution failure
        regenerated_sql (str): New query generated

    Returns:
        str: Small explanation of why the previous SQL query didn't work
    """

    if regenerated_sql.strip() == context.previous_sql.strip():
        return (
            f"The previous execution failed with: {context.execution_error}. "
            "The fallback kept the SQL unchanged, so verify it carefully before retrying."
        )
    return (
        f"The previous SQL failed with: {context.execution_error}. "
        "The fallback rewrote the statement using the supplied database schema. "
        "Review the highlighted changes before approving it."
    )


class SQLFallbackRegeneratorNode:
    """Generate a corrected SQL statement from execution failure context."""

    def __init__(self, model: LLM) -> None:
        """Initialize the node with its language model."""

        self.model = model

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """Return corrected SQL or a terminal regeneration error.

        Args:
            state (SQLAgentState): Current state of the graph

        Raises:
            RuntimeError: Raise an error if the LLM couldn't regenerate a new SQL query

        Returns:
            dict[str, Any]: Dictionnayry with the new SQL query
        """

        context = _build_regeneration_context(state)
        try:
            response = self.model.invoke(_render_regeneration_prompt(context))
            regenerated_sql = strip_code_fences(
                extract_text_from_response(response),
                language_prefix="sql",
            )
            if not regenerated_sql:
                raise RuntimeError("The model returned no SQL statement.")
        except (RuntimeError, AttributeError, TypeError) as exc:
            message = (
                f"SQL regeneration failed on retry {context.retry_count}/"
                f"{context.max_retries}: {exc}"
            )
            return {
                "retry_count": context.retry_count,
                "last_execution_error": context.execution_error,
                "regeneration_error": message,
                "execution_error": message,
                "analysis": message,
            }

        return {
            "generated_sql": regenerated_sql,
            "previous_sql": context.previous_sql,
            "regeneration_explanation": _explain_regeneration(
                context,
                regenerated_sql,
            ),
            "validated_sql": "",
            "sql_validation_error": None,
            "retry_count": context.retry_count,
            "last_execution_error": context.execution_error,
            "regeneration_error": None,
            "execution_error": None,
            "analysis": "",
            "execution_confirmed": False,
        }
