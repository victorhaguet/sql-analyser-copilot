"""LangGraph orchestration for the SQL copilot agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from sql_copilot.nodes.result_analyst import ResultAnalystNode
from sql_copilot.nodes.sql_executor import SQLExecutorNode
from sql_copilot.nodes.sql_generator import SQLGeneratorModel, SQLGeneratorNode
from sql_copilot.nodes.sql_validator import SQLValidatorNode
from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import SQLiteDatabase
from sql_copilot.tools.sql_safety import SQLSafetyValidator


def _route_after_validation(state: SQLAgentState) -> str:
    """Route to the next node after SQL validation based on the presence of validation errors."""
    return "abort" if state.get("sql_validation_error") else "sql_executor"


def _route_after_execution(state: SQLAgentState) -> str:
    """Route to the next node after SQL execution based on the presence of execution errors."""
    return "abort" if state.get("execution_error") else "result_analyst"


def build_sql_agent_graph(
    sql_generator_model: SQLGeneratorModel,
    analyst_model: SQLGeneratorModel | None = None,
    database: SQLiteDatabase | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> Any:
    """
    Build the Question -> Generator -> Validator -> Executor -> Analyst graph.
    
    Args:
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        database: The SQLite database instance to use for query execution (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        A compiled StateGraph instance representing the SQL agent workflow.
    """

    # Nodes
    graph = StateGraph(SQLAgentState)
    graph.add_node("sql_generator", SQLGeneratorNode(sql_generator_model, database))
    graph.add_node("sql_validator", SQLValidatorNode(validator))
    graph.add_node("sql_executor", SQLExecutorNode(database, limit=execution_limit))
    graph.add_node("result_analyst", ResultAnalystNode(analyst_model))

    # Edges
    graph.add_edge(START, "sql_generator") # Generate SQL from the question
    graph.add_edge("sql_generator", "sql_validator") # Validate the generated SQL

    graph.add_conditional_edges(
        "sql_validator", # If SQL is valid proceed to execution, otherwise abort
        _route_after_validation,
        {
            "sql_executor": "sql_executor",  # execution
            "abort": END, # abortion
        },
    )
    graph.add_conditional_edges(
        "sql_executor", # If execution is successful proceed to analysis, otherwise abort
        _route_after_execution,
        {
            "result_analyst": "result_analyst",  # analysis
            "abort": END, # abortion
        },
    )
    graph.add_edge("result_analyst", END) # End after analysis
    return graph.compile()
