"""SQL generation node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.nodes import LLM, load_prompt, render_prompt
from utils.llm import extract_text_from_response, strip_code_fences
from state import SQLAgentState
from tools.database import SQLiteDatabase, format_database_schema

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "sql_generator.j2"


class SQLGeneratorNode:
    """Generate SQL from a natural-language question."""

    def __init__(
        self,
        model: LLM,
        database: SQLiteDatabase,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the SQLGeneratorNode."""
        self.model = model
        self.database = database
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

        schema_overview = state.get("schema_overview") or format_database_schema(self.database)
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
