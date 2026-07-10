"""Database checker node."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from utils.nodes import LLM, load_prompt, render_prompt
from utils.llm import extract_text_from_response, strip_code_fences
from state import SQLAgentState
from tools.database import SQLiteDatabase, format_database_schema

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "database_checker.j2"
MISSING_MODEL_ERROR: str = "Please initialize the llm checker before runnin the application (check the README.md file)."


def _normalize_model_decision(raw_text: str) -> dict[str, Any]:
    """
    Parse selector output into a normalized dict.

    Args:
        raw_text: The raw text output from the selector model.

    Returns:
        A dictionary with keys `match`, `database`, `candidate_databases`, and `reason`.
    """
    cleaned = strip_code_fences(raw_text.strip(), language_prefix="json")

    try:
        decision = json.loads(cleaned) # Load the selector model's output
    except json.JSONDecodeError:
        return {"match": False, "database": "", "reason": "The router returned an invalid selection payload."}
    if not isinstance(decision, dict):
        return {"match": False, "database": "", "reason": "The router returned an invalid selection payload."}
    return {
        "match": bool(decision.get("match")),
        "database": str(decision.get("database") or "").strip(),
        "reason": str(decision.get("reason") or "").strip(),
    }

class DatabaseCheckerNode:
    """Check if query is related to the user-selected database."""

    def __init__(
        self,
        database: SQLiteDatabase,
        model: LLM | None = None,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the checker node."""
        if not database:
            raise ValueError("DatabaseCheckerNode requires a database instance.")
        self.database = database
        self.model = model
        self.prompt_template = prompt_template or load_prompt(
            PROMPT_PATH,
            (
                "You verify if a user question relates to the selected database.\n"
                "Database schema:\n{{ schema_overview }}\n\n"
                "Question: {{ question }}\n\n"
                "Return only 'true' if the question could be answered using this database, "
                "or 'false' if it is clearly unrelated."
            ),
        )

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Check if the question relates to the selected database.
        
        Args:
            state: A dictionary containing the current state in the graph,
            expected to have a "question" key.

        Returns:
            A dictionary containing the updated state with either the schema or an error message.
        """
        question: str = state["question"]
        
        if self.model is None:
            return self._error_result(MISSING_MODEL_ERROR)

        schema_overview = format_database_schema(self.database)
        prompt = render_prompt(
            self.prompt_template,
            question=question,
            schema_overview=schema_overview,
        )
        decision = _normalize_model_decision(extract_text_from_response(self.model.invoke(prompt)))
        
        if not decision["match"]:
            reason = decision["reason"] or (
                "The question does not clearly correspond to the selected database."
            )
            return self._error_result(
                f"{reason} Reformulate the question and be more specific about the data you want to retrieve/modify."
            )

        return self._selection_result(reason=decision["reason"])
    

    def _selection_result(self, *, reason: str) -> dict[str, Any]:
        """
        Build the success payload for a selected database.
        
        Args:
            reason: The reason for selection to include in the metadata.

        Returns:
            A dictionary containing the selected database and related metadata.
        """
        schema_overview = format_database_schema(self.database)
        return {
            "selected_database": "default",
            "schema_overview": schema_overview,
            "metadata": {
                "database_selection_reason": reason,
            },
        }

    @staticmethod
    def _error_result(message: str) -> dict[str, Any]:
        """
        Build the failure payload for unmatched questions.
        
        Args:
            message: The error message to include in the payload.

        Returns:
            A dictionary containing the error message and related metadata.
        """
        return {
            "execution_error": message,
            "analysis": message,
            "metadata": {
                "database_selection_failed": True,
            },
        }
