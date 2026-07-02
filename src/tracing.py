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
    state = step["state"]
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
        return f"{format_trace_header('Graph step')}\n{body}"
    
    if node == "intent_classifier":
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        if update.get("intent"):
            lines.append(f"User intent: {update.get("intent")}")
        if update.get("intent_error"):
            lines.append(f"Error: {update['intent_error']}")
        elif update.get("needs_confirmation"):
            lines.append("Needs user's confirmation.")
        else:
            lines.append("No need for user's confirmation.")
        body = "\n".join(lines)
        return f"{format_trace_header('Graph step')}\n{body}"
    
    if node == "role_authorizer":
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        if update.get("user_role"):
            lines.append(f"User Role: {update.get("user_role")}")
        if update.get("is_authorized"):
            lines.append(f"Is the user authorized to run the request ? : {update['is_authorized']}")
        if update.get("authorization_error"):
            lines.append(f"Error: {update['authorization_error']}")
        body = "\n".join(lines)
        return f"{format_trace_header('Graph step')}\n{body}"

    if node == "sql_generator":
        body = "\n".join(
            [
                f"Node: {node}",
                f"Outcome: {step['outcome']}",
                update.get("generated_sql", ""),
            ]
        ).strip()
        return f"{format_trace_header('Graph step')}\n{body}"
    
    if node == "__interrupt__":
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        interrupt_data = update.get("interrupt", {})
        if interrupt_data:
            request = interrupt_data.get("request", "")
            options = interrupt_data.get("options", [])
    
            if request:
                lines.append(f"Request: {request}")
            if options:
                lines.append(f"Options: {options[0]}/{options[1]}")
        
        body = "\n".join(lines)
        return f"{format_trace_header('Graph step')}\n{body}"
    
    if node == "sql_modification_validator":
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        if update.get("execution_confirmed") is not None:
            if update.get("execution_confirmed"):
                body = "Request was approved.\nSQL execution will proceed."
            else:
                body = "Request was rejected.\nSQL execution was cancelled."
        else:
            # First interrupt event - use existing format
            body = update.get("validated_sql") or update.get("sql_validation_error") or "(no output)"
        return (
            f"{format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{body}"
        )
    
    if node == "sql_validator":
        body = update.get("validated_sql") or update.get("sql_validation_error") or "(no output)"
        return (
            f"{format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{body}"
        )
    
    if node == "sql_modification_validator":
        lines = [f"Node: {node}", f"Outcome: {step['outcome']}"]
        if update.get("execution_confirmed") is not None:
            if update.get("execution_confirmed"):
                lines.append("SQL query was approved")
            else:
                lines.append("SQL query was rejected")
        body = "\n".join(lines)
        return (
            f"{format_trace_header('Tool Message')}\n"
            f"Name: {node}\n\n"
            f"{body}"
        )

    if node == "sql_executor":
        parts = [f"Outcome: {step['outcome']}"]
        if state.get("intent")=="query":
            if state.get("validated_sql"):
                parts.extend(["SQL:", str(state.get("validated_sql"))])
            if update.get("query_result") is not None:
                parts.append(format_trace_payload(update["query_result"]))
            elif update.get("execution_error"):
                parts.append(f"Error: {update['execution_error']}")
            return (
                f"{format_trace_header('Tool Message')}\n"
                f"Name: {node}\n\n"
                f"{chr(10).join(parts)}"
            )
        elif state.get("intent")=="modification":
            if state.get("generated_sql"):
                parts.extend(["SQL:", str(state.get("generated_sql"))])
            if update.get("execution_error"):
                parts.append(f"Error: {update['execution_error']}")
            else:
                parts.append("The SQL request was properly executed.")
            return (
                f"{format_trace_header('Tool Message')}\n"
                f"Name: {node}\n\n"
                f"{chr(10).join(parts)}"
            )


    body = update.get("analysis") or format_trace_payload(update)
    return f"{format_trace_header('Graph step')}\n{body}"


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


def extract_trace_steps_from_content(content: str) -> list[str]:
    """Extract trace steps from existing log content.

    Args:
        content: The existing log content.

    Returns:
        List of trace step lines (excluding Human Message header and question).
    """
    lines = content.split("\n")
    result = []
    in_trace_section = False
    skip_next = False
    
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
            
        if line.startswith("=") and "Human Message" in line:
            # Skip the Human Message header
            if i + 1 < len(lines):
                # Also skip the question line
                skip_next = True
            in_trace_section = True
            continue
        
        if in_trace_section:
            if line.startswith("=") and "Human Message" in line:
                # New Human Message - stop here (this is the start of appended content)
                break
            result.append(line)
    
    return result


def write_trace_log(
    question: str,
    trace: list[SQLAgentTraceStep],
    trace_log_dir: str | Path | None,
    existing_log_path: Path | str | dict | None = None,
) -> Path:
    """Persist one run trace under the configured log directory.

    Args:
        question: The original natural language question that was asked.
        trace: The list of SQLAgentTraceStep dictionaries representing the execution trace.
        trace_log_dir: The directory where the trace log should be saved.
        existing_log_path: Optional existing log path to append to instead of creating a new file.

    Returns:
        The path to the saved trace log file.
    """
    log_dir = Path(trace_log_dir) if trace_log_dir is not None else TRACE_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    normalized_existing_log_path: Path | None = None
    if existing_log_path is not None:
        if isinstance(existing_log_path, dict):
            if "path" in existing_log_path:
                normalized_existing_log_path = Path(existing_log_path["path"])
            elif "trace_log_path" in existing_log_path:
                normalized_existing_log_path = Path(existing_log_path["trace_log_path"])
            else:
                normalized_existing_log_path = Path(str(existing_log_path))
        elif isinstance(existing_log_path, str):
            normalized_existing_log_path = Path(existing_log_path)
        else:
            normalized_existing_log_path = existing_log_path

    if normalized_existing_log_path is not None and normalized_existing_log_path.exists():
        old_content = normalized_existing_log_path.read_text(encoding="utf-8")
        old_trace_steps = extract_trace_steps_from_content(old_content)
        
        # Build new content but only keep the trace steps (skip human message)
        new_full_content = build_trace_log_content(question, trace)
        new_lines = new_full_content.split("\n")
        
        # Find where trace steps start in the new content
        trace_start_idx = 0
        for i, line in enumerate(new_lines):
            if line.startswith("=") and "Human Message" in line:
                trace_start_idx = i + 2  # Skip header and question
                break
        
        # Skip the human message from new content, keep only trace steps
        new_trace_steps = new_lines[trace_start_idx:]
        
        # Reconstruct content: old header + question + old steps + new steps
        old_lines = old_content.split("\n")
        # Find where to split old content (after question)
        question_end_idx = 0
        for i, line in enumerate(old_lines):
            if line.startswith("=") and "Human Message" in line:
                if i + 1 < len(old_lines):
                    question_end_idx = i + 1  # Include the question line
                break
        
        old_header_and_question = old_lines[:question_end_idx + 1]
        
        # Combine all parts
        merged_lines = old_header_and_question + old_trace_steps + new_trace_steps
        merged_content = "\n".join(merged_lines)
        
        normalized_existing_log_path.write_text(merged_content, encoding="utf-8")
        return normalized_existing_log_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"trace_{timestamp}_{uuid4().hex[:8]}.log"
    log_path.write_text(build_trace_log_content(question, trace), encoding="utf-8")
    return log_path
