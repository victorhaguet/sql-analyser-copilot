"""SQL generation node."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from utils.nodes import load_prompt, render_prompt
from state import SQLAgentState
from tools.database import (
    SQLiteDatabase,
    format_database_schema,
    get_default_database,
)
from utils.llm import extract_text_from_response, strip_code_fences

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "sql_generator.j2"


class SQLGeneratorModel(Protocol):
    """Minimal protocol shared by SQL generation and analysis nodes."""

    def invoke(self, prompt: str) -> Any:
        """
        Return model output for the given prompt.
        
        Args:
            prompt: The input prompt string to generate a response for. 

        Returns:
            The raw output from the model
        """


TextModel = SQLGeneratorModel


class SQLGeneratorNode:
    """Generate SQL from a natural-language question."""

    def __init__(
        self,
        model: SQLGeneratorModel,
        database: SQLiteDatabase | None = None,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the SQLGeneratorNode."""
        self.model = model
        self.database = database or get_default_database()
        self.prompt_template = prompt_template or load_prompt(
            PROMPT_PATH,
            (
                "You translate questions into SQLite SELECT queries.\n"
                "Use only the schema below.\n"
                "Return only SQL.\n\n"
                "Schema:\n{{ schema_overview }}\n\n"
                "Question:\n{{ question }}\n"
            ),
        )

    def __call__(self, state: SQLAgentState) -> dict[str, Any]:
        """
        Generate SQL from the question in the state and return it in a new state dictionary.

        Args:
            state: A dictionary containing the current state in the graph, 
            expected to have a "question" key.

        Returns:
            A dictionary containing the updated state with the generated SQL.
        """
        question = state["question"]

        selected_database = state.get("selected_database") or self.database # Use the selected database from state or fall back to the default
        schema_overview = state.get("schema_overview") or format_database_schema(selected_database)
        prompt = render_prompt(
            self.prompt_template,
            question=question,
            schema_overview=schema_overview,
        )
        generated_sql = strip_code_fences(extract_text_from_response(self.model.invoke(prompt)), language_prefix="sql")
        return {
            "schema_overview": schema_overview,
            "generated_sql": generated_sql,
        }
