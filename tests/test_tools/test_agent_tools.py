"""Test the agent tools used by the SQL generation agent loop."""
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from tools.agent_tools import AgentToolLimits, build_agent_tools
from tools.database import SQLiteDatabase
from tools.sql_safety import SQLSafetyValidator
from tests.test_db.helpers import built_database

# A small relational schema (independent of the Chinook fixture) used to exercise
# inspect_schema and run_readonly_probe deterministically.
_RELATIONAL_SCHEMA = (
    """
    CREATE TABLE Artist (
        ArtistId INTEGER PRIMARY KEY,
        Name NVARCHAR(120) NOT NULL
    )
    """,
    """
    CREATE TABLE Album (
        AlbumId INTEGER PRIMARY KEY,
        Title NVARCHAR(160) NOT NULL,
        ArtistId INTEGER NOT NULL REFERENCES Artist(ArtistId)
    )
    """,
    "INSERT INTO Artist (ArtistId, Name) VALUES (1, 'AC/DC'), (2, 'Accept')",
    "INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (1, 'Back In Black', 1)",
)


class _PermissiveValidator:
    """Stub validator that never rejects anything.

    Used to prove that `run_readonly_probe`'s read-only guard is real (the
    `mode=ro` connection) rather than decorative (relying on the validator alone).
    """

    def assert_safe_select(
        self,
        query: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> str:
        """Return the query unchanged, without validating anything."""
        return query


def _get_tool(tools, name: str):
    """Look up a tool by name from the list returned by build_agent_tools."""
    for candidate in tools:
        if candidate.name == name:
            return candidate
    raise AssertionError(f"No tool named {name!r} in {[t.name for t in tools]}")


class BuildAgentToolsTestCase(unittest.TestCase):
    """Test that build_agent_tools returns the three expected tools."""

    def test_returns_the_three_tools_by_name(self) -> None:
        """Test that build_agent_tools returns exactly inspect_schema, ask_user, run_readonly_probe."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))

        self.assertEqual(
            {t.name for t in tools},
            {"inspect_schema", "ask_user", "run_readonly_probe"},
        )

    def test_ask_user_description_instructs_calling_it_alone(self) -> None:
        """Test that ask_user's docstring tells the model to call it alone (D3)."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))

        ask_user = _get_tool(tools, "ask_user")
        self.assertIn("ALONE", ask_user.description)
        self.assertIn("why", ask_user.description)


class InspectSchemaToolTestCase(unittest.TestCase):
    """Test the inspect_schema tool."""

    def test_overview_lists_every_table_with_row_and_column_counts(self) -> None:
        """Test that calling with no arguments returns a table overview."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "inspect_schema").invoke({})

        self.assertIn("Artist (2 rows, 2 columns)", result)
        self.assertIn("Album (1 rows, 3 columns)", result)

    def test_detail_for_requested_table_includes_columns_and_foreign_keys(self) -> None:
        """Test that requesting a specific table returns full detail, both FK directions."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "inspect_schema").invoke({"tables": ["Artist"]})

        self.assertIn("Table: Artist", result)
        self.assertIn("Foreign keys (incoming):", result)
        self.assertIn("Album.ArtistId -> Artist.ArtistId", result)

    def test_sample_rows_included_only_when_requested(self) -> None:
        """Test that sample rows only appear when include_sample_rows=True."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            inspect_schema = _get_tool(tools, "inspect_schema")

            without_samples = inspect_schema.invoke({"tables": ["Artist"]})
            with_samples = inspect_schema.invoke(
                {"tables": ["Artist"], "include_sample_rows": True}
            )

        self.assertNotIn("Sample rows:", without_samples)
        self.assertIn("Sample rows:", with_samples)
        self.assertIn("AC/DC", with_samples)

    def test_unknown_table_returns_error_text_listing_valid_names(self) -> None:
        """Test that an unknown table name never raises, and lists valid tables."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "inspect_schema").invoke({"tables": ["DoesNotExist"]})

        self.assertIn("Unknown table(s): DoesNotExist", result)
        self.assertIn("Artist", result)
        self.assertIn("Album", result)


class RunReadonlyProbeToolTestCase(unittest.TestCase):
    """Test the run_readonly_probe tool."""

    def test_returns_rows_for_a_valid_select(self) -> None:
        """Test that a valid SELECT returns columns and rows as compact text."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "run_readonly_probe").invoke(
                {"sql": "SELECT Name FROM Artist ORDER BY ArtistId"}
            )

        self.assertIn("2 row(s)", result)
        self.assertIn("AC/DC", result)
        self.assertIn("Accept", result)

    def test_rejects_delete_statement(self) -> None:
        """Test that a DELETE is rejected by the validator, not executed."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "run_readonly_probe").invoke({"sql": "DELETE FROM Artist"})

            remaining = database.preview_table("Artist", limit=10)

        self.assertTrue(result.startswith("Rejected:"))
        self.assertEqual(len(remaining), 2)

    def test_rejects_stacked_statement_injection(self) -> None:
        """Test that a SELECT stacked with a DROP is rejected as multiple statements."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "run_readonly_probe").invoke(
                {"sql": "SELECT * FROM Artist; DROP TABLE Artist"}
            )

            self.assertTrue(database.table_exists("Artist"))

        self.assertTrue(result.startswith("Rejected:"))

    def test_survives_a_syntax_or_semantic_error(self) -> None:
        """Test that an invalid statement returns an error string instead of raising."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, SQLSafetyValidator(database.database_path))
            result = _get_tool(tools, "run_readonly_probe").invoke(
                {"sql": "SELECT * FROM DoesNotExistTable"}
            )

        self.assertTrue(result.startswith("Rejected:"))

    def test_row_cap_is_enforced_and_flags_truncation(self) -> None:
        """Test that results are capped at probe_row_limit_max even if a larger limit is requested."""
        many_rows_schema = (
            'CREATE TABLE Numbers (n INTEGER PRIMARY KEY)',
            "INSERT INTO Numbers (n) VALUES "
            + ", ".join(f"({i})" for i in range(20)),
        )
        with built_database(many_rows_schema) as database:
            tools = build_agent_tools(
                database,
                SQLSafetyValidator(database.database_path),
                limits=AgentToolLimits(probe_row_limit_max=5),
            )
            result = _get_tool(tools, "run_readonly_probe").invoke(
                {"sql": "SELECT n FROM Numbers ORDER BY n", "limit": 1000}
            )

        self.assertIn("5 row(s) (truncated)", result)

    def test_write_fails_even_with_validator_stubbed_out(self) -> None:
        """Test that the mode=ro connection blocks writes even when the validator is bypassed.

        This is the test that proves the read-only guard is real, not decorative:
        the validator is replaced with a stub that approves everything.
        """
        with built_database(_RELATIONAL_SCHEMA) as database:
            tools = build_agent_tools(database, _PermissiveValidator())
            result = _get_tool(tools, "run_readonly_probe").invoke(
                {"sql": "DELETE FROM Artist"}
            )

            remaining = database.preview_table("Artist", limit=10)

        self.assertTrue(result.startswith("Query failed:"))
        self.assertEqual(len(remaining), 2)

    def test_timeout_aborts_an_expensive_query(self) -> None:
        """Test that a probe exceeding the configured timeout is aborted, not left hanging."""
        cross_join_schema = (
            'CREATE TABLE Numbers (n INTEGER PRIMARY KEY)',
            "INSERT INTO Numbers (n) VALUES "
            + ", ".join(f"({i})" for i in range(2000)),
        )
        with built_database(cross_join_schema) as database:
            tools = build_agent_tools(
                database,
                SQLSafetyValidator(database.database_path),
                limits=AgentToolLimits(probe_timeout_seconds=0.0),
            )
            result = _get_tool(tools, "run_readonly_probe").invoke(
                {"sql": "SELECT COUNT(*) FROM Numbers a, Numbers b"}
            )

        self.assertTrue(result.startswith("Query failed:"))


if __name__ == "__main__":
    unittest.main()
