"""Result analysis node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.nodes import LLM, load_prompt, render_prompt
from utils.llm import extract_text_from_response
from state import SQLAgentState
from tools.database import QueryResult

PROMPT_PATH_QUERY = Path(__file__).resolve().parents[1] / "prompts" / "result_analyst_query.j2"
PROMPT_PATH_MODIFICATION = Path(__file__).resolve().parents[1] / "prompts" / "result_analyst_modification.j2"
PROMPT_DEFAULT = """
You explain SQL query results to a business user.\n
Question:\n{{ question }}\n\n
SQL:\n{{ sql }}\n\n
Result:\n{{ result_payload }}\n
"""


def _serialize_result(result: QueryResult) -> str:
    """
    Serialize a QueryResult object into a JSON string for prompt input.
    Args:
        result: The QueryResult object to serialize.

    Returns:
        A JSON string representation of the QueryResult.
    """
    payload = {
        "columns": result.columns,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "rows": result.rows[:10],
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)


class ResultAnalystNode:
    """Turn query output into a user-facing answer."""

    def __init__(
        self,
        model: LLM | None = None,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the ResultAnalystNode."""
        self.model = model
        self.prompt_template_query = prompt_template or load_prompt(
            PROMPT_PATH_QUERY,
            PROMPT_DEFAULT,

        )
        self.prompt_template_modification = prompt_template or load_prompt(
            PROMPT_PATH_MODIFICATION,
            PROMPT_DEFAULT,
        )

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Analyze the query result and update the state with the analysis.

        Args:
            state: The current state of the SQL agent.

        Returns:
            A dictionary containing the analysis of the query result.
        """
        result: QueryResult = state["query_result"]
        if self.model is None:
            summary = (
                f"Returned {result.row_count} row(s) with columns {', '.join(result.columns)}."
            )
            if result.truncated:
                summary += " Results were truncated by the execution limit."
            if result.rows:
                summary += f" First row: {result.rows[0]}."
            return {"analysis": summary}
        
        if state["intent"] == "query":
            prompt = render_prompt(
                self.prompt_template_query,
                question=state["question"],
                sql=state["validated_sql"],
                result_payload=_serialize_result(result),
            )
            return {"analysis": extract_text_from_response(self.model.invoke(prompt))}

        prompt = render_prompt(
            self.prompt_template_modification,
            question=state["question"],
            sql=state["generated_sql"],
            result_payload=_serialize_result(result),
        )
        return {"analysis": extract_text_from_response(self.model.invoke(prompt))}
        
