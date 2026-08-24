"""Test the database module."""
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copy2
from unittest.mock import patch

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from tools.database import (
    DatabaseError,
    DatabaseNotFoundError,
    QueryResult,
    SQLiteDatabase,
    TableNotFoundError,
    format_database_schema,
    format_full_schema,
    format_table_detail,
    get_default_database,
    list_referencing_tables,
)
from tests.test_db.helpers import FIXTURE_DB_PATH, built_database, fixture_database

# A small, self-contained schema (independent of the Chinook fixture) used to
# exercise foreign key / index introspection deterministically.
_RELATIONAL_SCHEMA = (
    """
    CREATE TABLE Artist (
        ArtistId INTEGER PRIMARY KEY,
        Name NVARCHAR(120) NOT NULL,
        Country TEXT DEFAULT 'Unknown'
    )
    """,
    """
    CREATE UNIQUE INDEX IX_Artist_Name ON Artist(Name)
    """,
    """
    CREATE TABLE Album (
        AlbumId INTEGER PRIMARY KEY,
        Title NVARCHAR(160) NOT NULL,
        ArtistId INTEGER NOT NULL REFERENCES Artist(ArtistId) ON DELETE CASCADE
    )
    """,
    "INSERT INTO Artist (ArtistId, Name, Country) VALUES (1, 'AC/DC', 'Australia')",
    "INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (1, 'Back In Black', 1)",
)


class SQLiteDatabaseTestCase(unittest.TestCase):
    """Test case for the SQLiteDatabase class and related functions."""
    database: SQLiteDatabase

    @classmethod
    def setUpClass(cls) -> None:
        cls.database = fixture_database()

    def test_fixture_database_path_exists(self) -> None:
        """Test that the committed fixture database exists and is used by the tests."""
        self.assertTrue(FIXTURE_DB_PATH.exists())
        self.assertEqual(self.database.database_path, FIXTURE_DB_PATH.resolve())

    def test_get_default_database_returns_database_instance(self) -> None:
        """Test that get_default_database discovers a database from a data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            data_dir = project_root / "data"
            data_dir.mkdir()
            fixture_copy = data_dir / "fixture.sqlite"
            copy2(FIXTURE_DB_PATH, fixture_copy)

            with patch("tools.database._project_root", project_root):
                database = get_default_database()

        self.assertIsInstance(database, SQLiteDatabase)
        self.assertEqual(database.database_path, fixture_copy.resolve())

    def test_missing_database_path_raises_error(self) -> None:
        """Test that initializing SQLiteDatabase with a missing path raises DatabaseNotFoundError."""
        missing_path = Path("data/does-not-exist.sqlite")
        with self.assertRaises(DatabaseNotFoundError):
            SQLiteDatabase(missing_path)

    def test_list_tables_returns_known_tables(self) -> None:
        """Test that list_tables returns known tables in the database."""
        tables = self.database.list_tables()
        self.assertIn("Artist", tables)
        self.assertIn("Album", tables)
        self.assertEqual(tables, sorted(tables))

    def test_table_exists_returns_expected_values(self) -> None:
        """Test that table_exists returns True for existing tables and False for non-existing tables."""
        self.assertTrue(self.database.table_exists("Artist"))
        self.assertFalse(self.database.table_exists("DoesNotExist"))

    def test_get_table_schema_returns_expected_columns(self) -> None:
        """Test that get_table_schema returns the expected columns for a given table."""
        schema = self.database.get_table_schema("Album")
        self.assertGreater(len(schema), 0)
        self.assertEqual(schema[0]["name"], "AlbumId")
        self.assertEqual(schema[0]["type"], "INTEGER")
        self.assertTrue(schema[0]["primary_key"])

    def test_get_database_schema_includes_known_tables(self) -> None:
        """Test that get_database_schema includes known tables in the database."""
        schema = self.database.get_database_schema()
        self.assertIn("Artist", schema)
        self.assertIn("Album", schema)
        self.assertGreater(len(schema["Artist"]), 0)

    def test_preview_table_returns_limited_rows(self) -> None:
        """Test that preview_table returns a limited number of rows."""
        rows = self.database.preview_table("Artist", limit=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ArtistId"], 1)
        self.assertEqual(rows[0]["Name"], "AC/DC")

    def test_execute_command_returns_query_result(self) -> None:
        """Test that execute_command returns a QueryResult instance."""
        result = self.database.execute_command(
            "SELECT Name FROM Artist WHERE ArtistId <= ? ORDER BY ArtistId",
            parameters=(2,),
        )
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.columns, ["Name"])
        self.assertEqual(result.rows, [{"Name": "AC/DC"}, {"Name": "Accept"}])
        self.assertEqual(result.row_count, 2)
        self.assertFalse(result.truncated)

    def test_execute_command_supports_named_parameters(self) -> None:
        """Test that execute_command supports named parameters."""
        result = self.database.execute_command(
            "SELECT Name FROM Artist WHERE ArtistId = :artist_id",
            parameters={"artist_id": 1},
        )
        self.assertEqual(result.rows, [{"Name": "AC/DC"}])

    def test_execute_command_marks_truncated_results(self) -> None:
        """Test that execute_command marks results as truncated when the limit is reached."""
        result = self.database.execute_command(
            "SELECT ArtistId FROM Artist ORDER BY ArtistId",
            limit=2,
        )
        self.assertEqual(result.row_count, 2)
        self.assertTrue(result.truncated)

    def test_check_limit_rejects_non_positive_values(self) -> None:
        """Test that check_limit rejects non-positive values."""
        with self.assertRaises(DatabaseError):
            self.database.preview_table("Artist", limit=0)

        with self.assertRaises(DatabaseError):
            self.database.execute_command("SELECT Name FROM Artist", limit=-1)

    def test_describe_returns_path_and_tables(self) -> None:
        """Test that describe returns the database path and tables."""
        description = self.database.describe()
        self.assertEqual(description["database_path"], str(FIXTURE_DB_PATH.resolve()))
        self.assertIn("Artist", description["tables"])


class RelationalIntrospectionTestCase(unittest.TestCase):
    """Test FK/index introspection and prompt-friendly rendering, against a small
    schema built from scratch rather than the committed Chinook fixture."""

    def test_get_foreign_keys_returns_outgoing_reference(self) -> None:
        """Test that get_foreign_keys reports Album's outgoing reference to Artist."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            foreign_keys = database.get_foreign_keys("Album")

        self.assertEqual(len(foreign_keys), 1)
        self.assertEqual(
            foreign_keys[0],
            {
                "from_column": "ArtistId",
                "to_table": "Artist",
                "to_column": "ArtistId",
                "on_delete": "CASCADE",
            },
        )

    def test_get_foreign_keys_returns_empty_for_table_without_fk(self) -> None:
        """Test that get_foreign_keys returns an empty list when a table has no FKs."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            self.assertEqual(database.get_foreign_keys("Artist"), [])

    def test_get_indexes_reports_unique_index(self) -> None:
        """Test that get_indexes reports the unique index declared on Artist.Name."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            indexes = database.get_indexes("Artist")

        self.assertEqual(len(indexes), 1)
        self.assertEqual(indexes[0]["name"], "IX_Artist_Name")
        self.assertTrue(indexes[0]["unique"])
        self.assertEqual(indexes[0]["columns"], ["Name"])

    def test_list_referencing_tables_returns_incoming_reference(self) -> None:
        """Test that list_referencing_tables finds Album referencing Artist."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            referencing = list_referencing_tables(database, "Artist")

        self.assertEqual(
            referencing,
            [{"table": "Album", "from_column": "ArtistId", "to_column": "ArtistId"}],
        )

    def test_list_referencing_tables_empty_when_nothing_references_it(self) -> None:
        """Test that list_referencing_tables returns an empty list for a leaf table."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            self.assertEqual(list_referencing_tables(database, "Album"), [])

    def test_format_table_detail_marks_not_null_default_and_primary_key(self) -> None:
        """Test that format_table_detail annotates NOT NULL, DEFAULT, and PRIMARY KEY."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            detail = format_table_detail(database, "Artist")

        self.assertIn("ArtistId INTEGER PRIMARY KEY", detail)
        self.assertIn("Name NVARCHAR(120) NOT NULL", detail)
        self.assertIn("Country TEXT DEFAULT 'Unknown'", detail)

    def test_format_table_detail_includes_outgoing_foreign_keys(self) -> None:
        """Test that format_table_detail lists Album's outgoing FK to Artist."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            detail = format_table_detail(database, "Album")

        self.assertIn("Foreign keys (outgoing):", detail)
        self.assertIn("Album.ArtistId -> Artist.ArtistId", detail)

    def test_format_table_detail_includes_incoming_foreign_keys(self) -> None:
        """Test that format_table_detail lists Artist's incoming FK from Album."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            detail = format_table_detail(database, "Artist")

        self.assertIn("Foreign keys (incoming):", detail)
        self.assertIn("Album.ArtistId -> Artist.ArtistId", detail)

    def test_format_table_detail_includes_unique_indexes(self) -> None:
        """Test that format_table_detail lists unique indexes."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            detail = format_table_detail(database, "Artist")

        self.assertIn("Unique indexes:", detail)
        self.assertIn("IX_Artist_Name (Name)", detail)

    def test_format_table_detail_omits_sample_rows_by_default(self) -> None:
        """Test that format_table_detail does not include sample rows unless asked."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            detail = format_table_detail(database, "Artist")

        self.assertNotIn("Sample rows:", detail)

    def test_format_table_detail_includes_sample_rows_when_requested(self) -> None:
        """Test that format_table_detail includes sample rows when requested."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            detail = format_table_detail(database, "Artist", include_sample_rows=True)

        self.assertIn("Sample rows:", detail)
        self.assertIn("AC/DC", detail)

    def test_format_table_detail_raises_for_unknown_table(self) -> None:
        """Test that format_table_detail raises TableNotFoundError for an unknown table."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            with self.assertRaises(TableNotFoundError):
                format_table_detail(database, "DoesNotExist")

    def test_format_full_schema_includes_every_table_and_both_fk_directions(self) -> None:
        """Test that format_full_schema renders all tables with both FK directions."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            text, truncated = format_full_schema(database)

        self.assertFalse(truncated)
        self.assertIn("Table: Artist", text)
        self.assertIn("Table: Album", text)
        self.assertIn("Album.ArtistId -> Artist.ArtistId", text)
        # No sample rows in the full schema seed.
        self.assertNotIn("Sample rows:", text)

    def test_format_full_schema_falls_back_when_max_chars_exceeded(self) -> None:
        """Test that format_full_schema falls back to the compact schema when too large."""
        with built_database(_RELATIONAL_SCHEMA) as database:
            text, truncated = format_full_schema(database, max_chars=1)
            compact_text = format_database_schema(database)

        self.assertTrue(truncated)
        self.assertEqual(text, compact_text)


if __name__ == "__main__":
    unittest.main()
