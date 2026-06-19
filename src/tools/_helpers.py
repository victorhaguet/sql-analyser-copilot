"""Internal helper functions for SQL safety validation."""

import re
from typing import Any, Mapping, Sequence

NAMED_PARAMETER_PATTERN = re.compile(r"(?<!:)([:@$])([A-Za-z_][A-Za-z0-9_]*)")
POSITIONAL_PARAMETER_PATTERN = re.compile(r"\?")


def mask_literals_and_comments(sql: str) -> str:
    """
    Replace strings, quoted identifiers, and comments with spaces while preserving
    statement structure and keyword positions.

    Args:
        sql: The SQL query to mask.

    Returns:
        str: The masked SQL query.
    """
    chars = list(sql)
    masked: list[str] = []
    i = 0
    length = len(chars)

    while i < length:
        current = chars[i]
        next_char = chars[i + 1] if i + 1 < length else ""

        if current == "-" and next_char == "-":
            masked.extend("  ")
            i += 2
            while i < length and chars[i] != "\n":
                masked.append(" ")
                i += 1
            continue

        if current == "/" and next_char == "*":
            masked.extend("  ")
            i += 2
            while i < length:
                if chars[i] == "*" and i + 1 < length and chars[i + 1] == "/":
                    masked.extend("  ")
                    i += 2
                    break
                masked.append("\n" if chars[i] == "\n" else " ")
                i += 1
            continue

        if current == "'":
            masked.append(" ")
            i += 1
            while i < length:
                char = chars[i]
                masked.append("\n" if char == "\n" else " ")
                if char == "'" and not (i + 1 < length and chars[i + 1] == "'"):
                    i += 1
                    break
                if char == "'" and i + 1 < length and chars[i + 1] == "'":
                    masked.append(" ")
                    i += 2
                    continue
                i += 1
            continue

        if current in {'"', "`"}:
            quote = current
            masked.append(" ")
            i += 1
            while i < length:
                char = chars[i]
                masked.append("\n" if char == "\n" else " ")
                if char == quote:
                    i += 1
                    break
                i += 1
            continue

        if current == "[":
            masked.append(" ")
            i += 1
            while i < length:
                char = chars[i]
                masked.append("\n" if char == "\n" else " ")
                if char == "]":
                    i += 1
                    break
                i += 1
            continue

        masked.append(current)
        i += 1

    return "".join(masked)


def build_dummy_parameters(masked_query: str) -> Sequence[Any] | Mapping[str, Any]:
    """
    Build placeholder values so SQLite can parse parameterized statements with EXPLAIN.

    Args:
        masked_query: The SQL query with literals and comments masked.

    Returns:
        Sequence[Any] | Mapping[str, Any]: A sequence of None for positional parameters or a
        mapping of parameter names to None for named parameters.
    """
    named_matches = NAMED_PARAMETER_PATTERN.findall(masked_query)
    positional_count = len(POSITIONAL_PARAMETER_PATTERN.findall(masked_query))

    if named_matches and positional_count:
        return {"mixed_parameters_not_supported": None}

    if named_matches:
        return {name: None for _, name in named_matches}

    if positional_count:
        return tuple(None for _ in range(positional_count))

    return []
