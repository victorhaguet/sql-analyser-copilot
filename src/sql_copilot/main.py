"""Convenience entrypoints for invoking the SQL copilot graph."""

from __future__ import annotations

from typing import Any

from src.sql_copilot.graph import build_sql_agent_graph
from src.sql_copilot.nodes.sql_generator import TextModel
from src.sql_copilot.tools.database import SQLiteDatabase
from src.sql_copilot.tools.sql_safety import SQLSafetyValidator


def answer_question(
    question: str,
    sql_generator_model: TextModel,
    analyst_model: TextModel | None = None,
    database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> dict[str, Any]:
    """
    Run the SQL copilot graph.

    Args:
        question: The natural language question to answer.
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        database: The SQLite database instance to use for query execution (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).
    """

    graph = build_sql_agent_graph(
        sql_generator_model=sql_generator_model,
        analyst_model=analyst_model,
        database=database,
        validator=validator,
        execution_limit=execution_limit,
    )
    return graph.invoke({"question": question})
