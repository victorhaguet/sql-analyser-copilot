"""Trace logging and formatting utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from tools.database import QueryResult
from graph import SQLAgentTraceStep


TRACE_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
TRACE_HEADER_WIDTH = 80


def format_trace_header(title: str) -> str:
    """Return a centered trace section header.

    Args:
        title: The title to display in the header.

    Returns:
        A formatted header string with the title centered and padded by '=' characters.
    """
    return f" {title} ".center(TRACE_HEADER_WIDTH, "=")


def format_trace_payload(payload: Any) -> str:
    """Serialize a trace payload into readable text.

    Args:
        payload: The payload to serialize.

    Returns:
        A string representation of the payload.
    """
    if payload is None:
        return "(none)"
    if isinstance(payload, str):
        return payload
    if isinstance(payload, QueryResult):
        summary = [
            f"Columns: {', '.join(payload.columns) if payload.columns else '(none)'}",
            f"Row Count: {payload.row_count}",
            f"Truncated: {payload.truncated}",
            "Rows:",
            json.dumps(payload.rows, indent=2, ensure_ascii=True),
        ]
        return "\n".join(summary)
    return json.dumps(payload, indent=2, ensure_ascii=True, default=str)


def format_trace_step(step: SQLAgentTraceStep) -> str:
    """Render one normalized step into the persisted trace format.

    Args:
        step: The SQLAgentTraceStep dictionary containing node name, state update, and outcome label.

    Returns:
        A formatted string representing the step for trace logs.
    """
    node = step["node"]
    update = step["update"]
    metadata = update.get("metadata") or {}

    if node == "database_selector":
        selected_database = metadata.get("selected_database")
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        if selected_database:
            lines.append(f"Selected Database: {selected_database}")
        if metadata.get("candidate_databases"):
            lines.append(
                "Candidate Databases: "
                + ", ".join(cast(list[str], metadata["candidate_databases"]))
            )
        if metadata.get("database_selection_reason"):
            lines.append(f"Reason: {metadata['database_selection_reason']}")
        if update.get("execution_error"):
            lines.append(f"Error: {update['execution_error']}")
        body = "\n".join(lines)
        return f"{format_trace_header('Ai Message')}\n{body}"

    if node == "sql_generator":
        body = "\n".join(
            [
                f"Node: {node}",
                f"Outcome: {step['outcome']}",
                update.get("generated_sql", ""),
            ]
        ).strip()
        return f"{format_trace_header('Ai Message')}\n{body}"

    if node == "sql_validator":
        body = update.get("validated_sql") or update.get("sql_validation_error") or "(no output)"
        return (
            f"{format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{body}"
        )

    if node == "sql_executor":
        parts = [f"Outcome: {step['outcome']}"]
        validated_sql = update.get("validated_sql")
        if validated_sql:
            parts.extend(["SQL:", str(validated_sql)])
        if update.get("query_result") is not None:
            parts.append(format_trace_payload(update["query_result"]))
        elif update.get("execution_error"):
            parts.append(f"Error: {update['execution_error']}")
        return (
            f"{format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{chr(10).join(parts)}"
        )

    body = update.get("analysis") or format_trace_payload(update)
    return f"{format_trace_header('Ai Message')}\n{body}"


def build_trace_log_content(question: str, trace: list[SQLAgentTraceStep]) -> str:
    """Render the full persisted trace file content for one run.

    Args:
        question: The original natural language question that was asked.
        trace: The list of SQLAgentTraceStep dictionaries representing the execution trace.

    Returns:
        A formatted string representing the entire trace log content for this run.
    """
    sections = [format_trace_header("Human Message"), question]
    sections.extend(format_trace_step(step) for step in trace)
    return "\n".join(sections) + "\n"


def write_trace_log(
    question: str,
    trace: list[SQLAgentTraceStep],
    trace_log_dir: str | Path | None,
) -> Path:
    """Persist one run trace under the configured log directory.

    Args:
        question: The original natural language question that was asked.
        trace: The list of SQLAgentTraceStep dictionaries representing the execution trace.
        trace_log_dir: The directory where the trace log should be saved.

    Returns:
        The path to the saved trace log file.
    """
    log_dir = Path(trace_log_dir) if trace_log_dir is not None else TRACE_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"trace_{timestamp}_{uuid4().hex[:8]}.log"
    log_path.write_text(build_trace_log_content(question, trace), encoding="utf-8")
    return log_path
