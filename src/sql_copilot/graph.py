"""LangGraph orchestration for the SQL copilot agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from sql_copilot.nodes.database_selector import DatabaseSelectorNode
from sql_copilot.nodes.result_analyst import ResultAnalystNode
from sql_copilot.nodes.sql_executor import SQLExecutorNode
from sql_copilot.nodes.sql_generator import SQLGeneratorModel, SQLGeneratorNode
from sql_copilot.nodes.sql_validator import SQLValidatorNode
from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import RegisteredDatabase, get_default_database_catalog
from sql_copilot.tools.sql_safety import SQLSafetyValidator


def _route_after_database_selection(state: SQLAgentState) -> str:
    """
    Route to SQL generation only when a database was selected.
    
    Args:
        state: The current state of the SQL agent after database selection.

    Returns:
        The next node to transition to based on the state.
    """
    return "abort" if state.get("execution_error") else "sql_generator"


def _route_after_validation(state: SQLAgentState) -> str:
    """
    Route to the next node after SQL validation based on the presence of validation errors.
    
    Args:
        state: The current state of the SQL agent after SQL validation.

    Returns:
        The next node to transition to based on the state.
    """
    return "abort" if state.get("sql_validation_error") else "sql_executor"


def _route_after_execution(state: SQLAgentState) -> str:
    """
    Route to the next node after SQL execution based on the presence of execution errors.
    
    Args:
        state: The current state of the SQL agent after SQL execution.

    Returns:
        The next node to transition to based on the state.
    """
    return "abort" if state.get("execution_error") else "result_analyst"


def build_sql_agent_graph(
    sql_generator_model: SQLGeneratorModel,
    analyst_model: SQLGeneratorModel | None = None,
    selector_model: SQLGeneratorModel | None = None,
    databases: list[RegisteredDatabase] | None = None,
    validator: SQLSafetyValidator | None = None,
    execution_limit: int = 200,
) -> Any:
    """
    Build the Question -> Generator -> Validator -> Executor -> Analyst graph.
    
    Args:
        sql_generator_model: The language model to use for SQL generation.
        analyst_model: The language model to use for result analysis (optional).
        selector_model: The language model to use for database selection (optional).
        databases: The registered databases available for routing (optional).
        validator: The SQLSafetyValidator instance to use for query validation (optional).
        execution_limit: The maximum number of rows to return from query execution (default: 200).

    Returns:
        A compiled StateGraph instance representing the SQL agent workflow.
    """

    # Nodes
    database_catalog = databases or get_default_database_catalog()
    default_database = database_catalog[0].database
    graph = StateGraph(SQLAgentState)
    graph.add_node("database_selector", DatabaseSelectorNode(database_catalog, model=selector_model))
    graph.add_node("sql_generator", SQLGeneratorNode(sql_generator_model, default_database))
    graph.add_node("sql_validator", SQLValidatorNode(validator))
    graph.add_node("sql_executor", SQLExecutorNode(default_database, limit=execution_limit))
    graph.add_node("result_analyst", ResultAnalystNode(analyst_model))

    # Edges
    graph.add_edge(START, "database_selector") # Select the relevant database for the question
    graph.add_conditional_edges(
        "database_selector",
        _route_after_database_selection,
        {
            "sql_generator": "sql_generator",
            "abort": END, # abortion if no database was selected
        },
    )
    graph.add_edge("sql_generator", "sql_validator") # Validate the generated SQL

    graph.add_conditional_edges(
        "sql_validator", # If SQL is valid proceed to execution, otherwise abort
        _route_after_validation,
        {
            "sql_executor": "sql_executor",  # execution
            "abort": END, # abortion if SQL validation fails
        },
    )
    graph.add_conditional_edges(
        "sql_executor", # If execution is successful proceed to analysis, otherwise abort
        _route_after_execution,
        {
            "result_analyst": "result_analyst",  # analysis
            "abort": END, # abortion if SQL execution fails
        },
    )
    graph.add_edge("result_analyst", END) # End after analysis
    return graph.compile()
