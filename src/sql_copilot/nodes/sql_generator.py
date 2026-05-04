"""SQL generation node."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined

from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import SQLiteDatabase, get_default_database

# Get the prompt template for SQL generation.
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "sql_generator.j2"
PROMPT_ENVIRONMENT = Environment(
    autoescape=False, # No autoescaping for plain text prompts
    trim_blocks=True, # Remove newlines around Jinja blocks for cleaner formatting
    lstrip_blocks=True, # Remove leading spaces around Jinja blocks
    undefined=StrictUndefined, # Raise error for undefined variables
)


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


def _load_prompt(path: Path, fallback: str) -> str:
    """
    Load a prompt template from a jinja file, 
    falling back to a default string if the file does not exist.

    Args:
        path: The path to the prompt template file.
        fallback: The fallback prompt template string to use if the file does not exist.

    Returns:
        The loaded prompt template string.
    """
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    return text or fallback


def _render_prompt(template: str, **context: Any) -> str:
    """
    Render a Jinja prompt template with the given context.

    Args:
        template: The Jinja template string to render.
        **context: The context variables to render the template with.

    Returns:
        The rendered prompt string.
    """
    return PROMPT_ENVIRONMENT.from_string(template).render(**context).strip()


def _model_output_to_text(response: Any) -> str:
    """
    Convert the raw output from the model into a clean text string.

    Args:
        response: The raw output from the model
    
    Returns:
        A cleaned text string extracted from the model output.
    """
    # If it is already a string, just return it stripped. 
    if isinstance(response, str):
        return response.strip()
    # If it has a 'content' attribute, try to extract it.
    content = getattr(response, "content", None)
    # If content is a string, return it stripped.
    if isinstance(content, str):
        return content.strip()
    # If content is a list of dicts, extract text parts.
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(part for part in text_parts if part).strip()
    return str(response).strip()


def _strip_sql_fence(text: str) -> str:
    """
    Remove SQL code fences from the given text.

    Args:
        text: The text potentially containing SQL code fences.

    Returns:
        The text with SQL code fences removed.
    """
    # Strip the query
    stripped = text.strip()
    # If the text starts with a code fence, remove it and any trailing code fence.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    # If the text starts with "sql\n", remove that prefix.
    return stripped.removeprefix("sql").strip() if stripped.lower().startswith("sql\n") else stripped


def format_database_schema(database: SQLiteDatabase) -> str:
    """
    Render the SQLite schema into a compact prompt-friendly string.
    
    Args:
        database: The SQLiteDatabase instance to format the schema of.
    
    Returns:
        A string representation of the database schema suitable for inclusion in prompts.
    """
    lines: list[str] = []

    # For each table, list its columns and types in a compact format.
    for table_name, columns in database.get_database_schema().items():
        column_bits = []
        for column in columns:
            descriptor = f'{column["name"]} {column["type"]}'
            if column["primary_key"]:
                descriptor += " PRIMARY KEY"
            column_bits.append(descriptor)
        lines.append(f'{table_name}({", ".join(column_bits)})')
    return "\n".join(lines)


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
        self.prompt_template = prompt_template or _load_prompt(
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

        # Get the schema overview from the state or format it from the database if not present.
        schema_overview = state.get("schema_overview") or format_database_schema(self.database)
        prompt = _render_prompt(
            self.prompt_template,
            question=question,
            schema_overview=schema_overview,
        )
        generated_sql = _strip_sql_fence(_model_output_to_text(self.model.invoke(prompt)))
        return {
            "schema_overview": schema_overview,
            "generated_sql": generated_sql,
        }
