"""State definitions for the SQL copilot graph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from tools.database import QueryResult


class SQLAgentState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes."""

    question: str # User question in natural language
    selected_database: str # Database name selected for the question
    schema_overview: str # A textual overview of the database schema
    intent: str | None # User intention (query or modification)
    intent_error: str | None # Error message if intent classification fails
    generated_sql: str # SQL produced by the agent loop
    validated_sql: str # The SQL query after validation (could be the same as generated_sql or modified for safety)
    sql_validation_error: str | None # Error message if SQL validation fails
    query_result: QueryResult # The result of executing the SQL query
    execution_error: str | None # Error message if query execution fails
    analysis: str # Analysis of the query result
    metadata: dict[str, Any] # Additional metadata related to the query and its execution
    user_role: str # Role of the authenticated user
    is_authorized: bool # Whether the user has authorization to proceed
    authorization_error: str | None # Error message if authorization fails
    needs_confirmation: bool # Whether user confirmation is required before execution
    execution_requested: bool # Whether execution is pending user confirmation
    execution_confirmed: bool # Whether user has confirmed execution
    thread_id: str # Thread identifier used to resume an interrupted graph
    interrupt: dict[str, Any] # Serialized LangGraph interrupt payload
    edited_sql: str | None # User-modified SQL (if different from generated_sql)
    retry_count: int # Current retry attempt number
    max_retries: int # Maximum retry attempts allowed
    last_execution_error: str | None # Last execution error sent back to the agent
    previous_sql: str # SQL statement that failed before agent repair
    regeneration_explanation: str # User-facing reason for the repaired statement

    # SQL generation agent loop (messages is the transcript; everything else here
    # is derived from or scoped alongside it — no parallel transcript is kept).
    messages: Annotated[list[AnyMessage], add_messages] # Agent loop transcript (system/human/AI/tool messages)
    agent_status: str | None # "final" | "needs_clarification" | "budget_exhausted" | "cancelled" | "failed" | "repairing"
    agent_iterations: int # Number of sql_agent_llm turns taken so far
    max_agent_iterations: int # Loop budget, intent-scoped
    probe_count: int # Number of run_readonly_probe calls made so far
    max_probes: int # Probe budget
    clarification_rounds: int # Number of ask_user rounds taken so far
    max_clarifications: int # Clarification budget
    clarification_answers: list[dict[str, Any]] # Accumulated {key, question, answer}, appended never overwritten
    agent_rationale: str # Short explanation of the final SQL, shown in the approval dialog
    agent_error: str | None # Terminal agent failure message
