"""Test the database module."""
import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from tools.database import (
    DEFAULT_DB_PATH,
    DatabaseError,
    DatabaseNotFoundError,
    QueryResult,
    SQLiteDatabase,
    get_default_database,
)


class SQLiteDatabaseTestCase(unittest.TestCase):
    """Test case for the SQLiteDatabase class and related functions."""
    database: SQLiteDatabase

    @classmethod
    def setUpClass(cls) -> None:
        cls.database = SQLiteDatabase()

    def test_default_database_path_exists(self) -> None:
        """Test that the default database path exists and is correctly set in the SQLiteDatabase instance."""
        self.assertTrue(DEFAULT_DB_PATH.exists())
        self.assertEqual(self.database.database_path, DEFAULT_DB_PATH.resolve())

    def test_get_default_database_returns_database_instance(self) -> None:
        """Test that get_default_database returns a SQLiteDatabase instance."""
        database = get_default_database()
        self.assertIsInstance(database, SQLiteDatabase)
        self.assertEqual(database.database_path, DEFAULT_DB_PATH.resolve())

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
        self.assertEqual(description["database_path"], str(DEFAULT_DB_PATH.resolve()))
        self.assertIn("Artist", description["tables"])


if __name__ == "__main__":
    unittest.main()
