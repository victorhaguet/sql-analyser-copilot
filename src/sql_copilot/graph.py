"""LangGraph orchestration for the SQL copilot agent."""

from __future__ import annotations

from typing import Any, Iterator, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from sql_copilot.nodes.database_selector import DatabaseSelectorNode
from sql_copilot.nodes.result_analyst import ResultAnalystNode
from sql_copilot.nodes.sql_executor import SQLExecutorNode
from sql_copilot.nodes.sql_generator import SQLGeneratorModel, SQLGeneratorNode
from sql_copilot.nodes.sql_validator import SQLValidatorNode
from sql_copilot.state import SQLAgentState
from sql_copilot.tools.database import RegisteredDatabase, get_default_database_catalog
from sql_copilot.tools.sql_safety import SQLSafetyValidator

TraceStreamMode = Literal["updates", "values"]
SQLAgentStateUpdate = SQLAgentState


class CompiledSQLAgentGraph(Protocol):
    """Protocol for the compiled graph methods used by the trace helpers."""

    def stream(
        self,
        state: SQLAgentState,
        *,
        stream_mode: TraceStreamMode = "updates",
    ) -> Iterator[object]:
        """Yield raw LangGraph stream events."""


class SQLAgentTraceStep(TypedDict):
    """Normalized view of a single graph step."""

    node: str
    update: SQLAgentStateUpdate
    state: SQLAgentState
    outcome: str


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


def _clone_state(state: SQLAgentState) -> SQLAgentState:
    """Return a shallow copy of the state."""
    return cast(SQLAgentState, dict(state))


def _merge_state(
    state: SQLAgentState,
    update: SQLAgentStateUpdate,
) -> SQLAgentState:
    """Apply a partial state update and return the merged state."""
    merged_state = dict(state)
    merged_state.update(dict(update))
    return cast(SQLAgentState, merged_state)


def _summarize_step_outcome(update: SQLAgentStateUpdate) -> str:
    """Classify the result of a graph step for debugging output."""
    metadata = update.get("metadata") or {}
    if update.get("sql_validation_error"):
        return "validation_failed"
    if update.get("execution_error"):
        if metadata.get("database_selection_ambiguous"):
            return "database_selection_ambiguous"
        if metadata.get("database_selection_failed"):
            return "database_selection_failed"
        return "execution_failed"
    if "query_result" in update:
        return "query_executed"
    if "analysis" in update:
        return "analysis_ready"
    if "generated_sql" in update:
        return "sql_generated"
    if "selected_database" in update:
        return "database_selected"
    return "updated"


def _normalize_stream_event(
    event: object,
    current_state: SQLAgentState,
    stream_mode: TraceStreamMode,
) -> tuple[str, SQLAgentStateUpdate, SQLAgentState]:
    """Convert a LangGraph stream event into a normalized step payload."""
    # One update at a time with incremental state
    if stream_mode == "updates":
        if not isinstance(event, dict) or len(event) != 1:
            raise TypeError(f"Unexpected graph update event: {event!r}")
        node_name, update = next(iter(event.items()))
        if not isinstance(node_name, str) or not isinstance(update, dict):
            raise TypeError(f"Unexpected graph update event: {event!r}")
        typed_update = cast(SQLAgentStateUpdate, update)
        return node_name, typed_update, _merge_state(current_state, typed_update)

    # Full state of the graph after each step
    if not isinstance(event, dict):
        raise TypeError(f"Unexpected graph values event: {event!r}")
    state = cast(SQLAgentState, event)
    return "state", _clone_state(state), _clone_state(state)


def stream_sql_agent_execution(
    graph: CompiledSQLAgentGraph,
    initial_state: SQLAgentState,
    *,
    stream_mode: TraceStreamMode = "updates",
) -> Iterator[SQLAgentTraceStep]:
    """
    Yield normalized execution steps from a compiled SQL agent graph.
    
    Args:
    graph: The compiled SQL agent graph to execute.
    initial_state: The initial state to start the graph execution from.
    stream_mode: The mode of streaming updates from the graph ("updates" for incremental updates,
                    "values" for full state after each step).
    
    Yields:
    Normalized SQLAgentTraceStep dictionaries containing the node name, state update, full state,
    and a summarized outcome of each step in the graph execution.
    """
    state = _clone_state(initial_state)
    for event in graph.stream(initial_state, stream_mode=stream_mode):
        node_name, update, state = _normalize_stream_event(event, state, stream_mode)
        yield {
            "node": node_name,
            "update": update,
            "state": _clone_state(state),
            "outcome": _summarize_step_outcome(update),
        }


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
