"""Database selection node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sql_copilot.nodes.sql_generator import (
    SQLGeneratorModel,
    _load_prompt,
    _model_output_to_text,
    _render_prompt,
)
from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import RegisteredDatabase, format_database_schema

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "database_selector.j2"


def _build_catalog_payload(databases: list[RegisteredDatabase]) -> str:
    """
    Serialize the database catalog for the selector prompt.
    
    Args:
        databases: The list of registered databases to serialize.

    Returns:
        A JSON string representing the database catalog for the selector prompt.
    """
    payload = []
    for entry in databases:
        payload.append(
            {
                "name": entry.name,
                "description": entry.description,
                "schema": format_database_schema(entry.database),
            }
        )
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _normalize_model_decision(raw_text: str) -> dict[str, Any]:
    """
    Parse selector output into a normalized dict.

    Args:
        raw_text: The raw text output from the selector model.

    Returns:
        A dictionary with keys `match`, `database`, and `reason`.
    """
    # The model is expected to return a JSON blob, but we need to be resilient to formatting issues like code blocks or extra text.
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    cleaned = cleaned.removeprefix("json").strip()

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


class DatabaseSelectorNode:
    """Select the most relevant database for a user question."""

    def __init__(
        self,
        databases: list[RegisteredDatabase],
        model: SQLGeneratorModel | None = None,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the selector node."""
        if not databases:
            raise ValueError("DatabaseSelectorNode requires at least one registered database.")
        self.databases = databases
        self.model = model
        self.prompt_template = prompt_template or _load_prompt(
            PROMPT_PATH,
            (
                "You route a user question to the most relevant SQLite database.\n"
                "Return JSON with keys `match` (boolean), `database` (string), and `reason` (string).\n"
                "Only set `match` to true when one database clearly fits the question.\n"
                "If none fit, return `match: false` and explain why.\n\n"
                "Question:\n{{ question }}\n\n"
                "Available databases:\n{{ catalog_payload }}\n"
            ),
        )

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Select a database or stop the graph early with a user-facing error.
        
        Args:
            state: A dictionary containing the current state in the graph,
            expected to have a "question" key.

        Returns:
            A dictionary containing the updated state with either the selected database or an error message.
        """
        # Single database configuration
        question = state["question"]
        if len(self.databases) == 1 and self.model is None:
            return self._selection_result(self.databases[0], reason="Single database configured.")

        # No model provided for multi-database selection
        if self.model is None:
            return self._no_match_result(
                "The question could not be matched to a configured database. "
                "Reformulate it with more domain-specific details."
            )

        # Model-driven selection for multi-database configuration
        prompt = _render_prompt(
            self.prompt_template,
            question=question,
            catalog_payload=_build_catalog_payload(self.databases),
        )
        decision = _normalize_model_decision(_model_output_to_text(self.model.invoke(prompt)))
        if not decision["match"]: # If no match, return an error with the model's reason or a default message
            reason = decision["reason"] or (
                "The question does not clearly correspond to any available database."
            )
            return self._no_match_result(
                f"{reason} Reformulate the question and be more specific about the data you want."
            )

        # Return the corresponding database
        selected_name = decision["database"]
        if len(self.databases) == 1 and decision["match"]:
            return self._selection_result(self.databases[0], reason=decision["reason"])
        for entry in self.databases:
            if entry.name == selected_name:
                return self._selection_result(entry, reason=decision["reason"])
        return self._no_match_result(
            "The question could not be matched to a configured database. "
            "Reformulate the question and be more specific about the data you want."
        )

    def _selection_result(self, entry: RegisteredDatabase, *, reason: str) -> dict[str, Any]:
        """
        Build the success payload for a selected database.
        
        Args:
            entry: The database entry that was selected.
            reason: The reason for selection to include in the metadata.

        Returns:
            A dictionary containing the selected database and related metadata.
        """
        schema_overview = format_database_schema(entry.database)
        return {
            "selected_database": entry.database,
            "schema_overview": schema_overview,
            "metadata": {
                "selected_database": entry.name,
                "selected_database_description": entry.description,
                "database_selection_reason": reason,
                "available_databases": [database.name for database in self.databases],
            },
        }

    @staticmethod
    def _no_match_result(message: str) -> dict[str, Any]:
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
