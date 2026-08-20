"""Tests that Step 10's guardrail constants are actually env-tunable.

These constants are read once, at module import time (`int(os.getenv(...))`),
so the only reliable way to test "setting the env var changes the value" is
importing the module fresh — done here in a subprocess rather than via
`importlib.reload()`, to avoid mutating shared module state for every other
test file that imports `graph`, `core`, `nodes.sql_agent`, or `tools.agent_tools`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

SRC_DIR: Path | None = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        SRC_DIR = parent / "src"
        break

assert SRC_DIR is not None, "Could not locate the src/ directory."


def _read_constant_with_env(module_name: str, constant_name: str, env_overrides: dict[str, str]) -> str:
    """Import `module_name` fresh in a subprocess with `env_overrides` set,
    and return the printed value of `constant_name`.

    Args:
        module_name: Dotted module path under src/, e.g. "graph".
        constant_name: The module-level attribute to print.
        env_overrides: Extra/overriding environment variables for the subprocess.

    Returns:
        str: The stdout value (stripped).
    """
    script = (
        f"import {module_name} as m\n"
        f"print(getattr(m, {constant_name!r}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(SRC_DIR),
        env={**os.environ, **env_overrides},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class EnvTunableGuardrailsTestCase(unittest.TestCase):
    """Test that each Step 10 guardrail constant reads from its documented env var."""

    def test_max_iterations_query_reads_env(self) -> None:
        self.assertEqual(
            _read_constant_with_env(
                "graph", "DEFAULT_MAX_AGENT_ITERATIONS_QUERY", {"SQL_AGENT_MAX_ITERATIONS_QUERY": "9"}
            ),
            "9",
        )

    def test_max_iterations_query_default_when_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "SQL_AGENT_MAX_ITERATIONS_QUERY"}
        script = "import graph as m\nprint(getattr(m, 'DEFAULT_MAX_AGENT_ITERATIONS_QUERY'))\n"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(SRC_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "6")

    def test_max_iterations_modification_reads_env(self) -> None:
        self.assertEqual(
            _read_constant_with_env(
                "graph",
                "DEFAULT_MAX_AGENT_ITERATIONS_MODIFICATION",
                {"SQL_AGENT_MAX_ITERATIONS_MODIFICATION": "15"},
            ),
            "15",
        )

    def test_max_probes_reads_env(self) -> None:
        self.assertEqual(
            _read_constant_with_env("nodes.sql_agent", "DEFAULT_MAX_PROBES", {"SQL_AGENT_MAX_PROBES": "9"}),
            "9",
        )

    def test_max_clarifications_reads_env(self) -> None:
        self.assertEqual(
            _read_constant_with_env(
                "nodes.sql_agent", "DEFAULT_MAX_CLARIFICATIONS", {"SQL_AGENT_MAX_CLARIFICATIONS": "5"}
            ),
            "5",
        )

    def test_probe_row_limit_reads_env(self) -> None:
        self.assertEqual(
            _read_constant_with_env(
                "tools.agent_tools", "DEFAULT_PROBE_ROW_LIMIT", {"SQL_AGENT_PROBE_ROW_LIMIT": "35"}
            ),
            "35",
        )

    def test_recursion_limit_reads_env(self) -> None:
        self.assertEqual(
            _read_constant_with_env("core", "DEFAULT_AGENT_RECURSION_LIMIT", {"SQL_AGENT_RECURSION_LIMIT": "77"}),
            "77",
        )

    def test_max_schema_chars_defaults_to_none_when_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "SQL_AGENT_MAX_SCHEMA_CHARS"}
        script = "from nodes.sql_agent import _default_max_schema_chars\nprint(_default_max_schema_chars())\n"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(SRC_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "None")

    def test_max_schema_chars_reads_env(self) -> None:
        script = "from nodes.sql_agent import _default_max_schema_chars\nprint(_default_max_schema_chars())\n"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(SRC_DIR),
            env={**os.environ, "SQL_AGENT_MAX_SCHEMA_CHARS": "12000"},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "12000")


if __name__ == "__main__":
    unittest.main()
