"""Test the SQL safety module."""
import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent))
        break

from src.sql_copilot.tools.sql_safety import (
    EmptyQueryError,
    ForbiddenKeywordError,
    InvalidQueryError,
    MultipleStatementsError,
    NonSelectQueryError,
    SQLSafetyValidator,
    ValidatedSQL,
    ensure_safe_select_query,
    validate_select_query,
)


class SQLSafetyValidatorTestCase(unittest.TestCase):
    """Test case for SQL safety validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = SQLSafetyValidator()

    def test_validate_select_accepts_simple_query(self) -> None:
        """Test that a simple SELECT query is accepted and normalized correctly."""
        result = self.validator.validate_select(
            "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2"
        )
        self.assertIsInstance(result, ValidatedSQL)
        self.assertEqual(
            result.normalized_query,
            "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2",
        )

    def test_validate_select_accepts_cte_query(self) -> None:
        """Test that a query with a Common Table Expression (CTE) is accepted and normalized correctly."""
        result = self.validator.validate_select(
            """
            WITH top_artists AS (
                SELECT ArtistId, Name
                FROM Artist
                WHERE ArtistId <= 2
            )
            SELECT Name FROM top_artists
            """
        )
        self.assertTrue(result.normalized_query.startswith("WITH"))

    def test_validate_select_strips_trailing_semicolon(self) -> None:
        """Test that a trailing semicolon is stripped from the query."""
        normalized = ensure_safe_select_query("SELECT Name FROM Artist;")
        self.assertEqual(normalized, "SELECT Name FROM Artist")

    def test_validate_select_supports_positional_parameters(self) -> None:
        """Test that positional parameters are supported in SELECT queries."""
        result = self.validator.validate_select(
            "SELECT Name FROM Artist WHERE ArtistId = ?",
            parameters=(1,),
        )
        self.assertEqual(
            result.normalized_query,
            "SELECT Name FROM Artist WHERE ArtistId = ?",
        )

    def test_validate_select_supports_named_parameters(self) -> None:
        """Test that named parameters are supported in SELECT queries."""
        result = validate_select_query(
            "SELECT Name FROM Artist WHERE ArtistId = :artist_id",
            parameters={"artist_id": 1},
        )
        self.assertEqual(
            result.normalized_query,
            "SELECT Name FROM Artist WHERE ArtistId = :artist_id",
        )

    def test_is_safe_select_returns_true_for_valid_query(self) -> None:
        """Test that is_safe_select returns True for a valid SELECT query."""
        self.assertTrue(self.validator.is_safe_select("SELECT * FROM Artist"))

    def test_is_safe_select_returns_false_for_invalid_query(self) -> None:
        """Test that is_safe_select returns False for an invalid query."""
        self.assertFalse(self.validator.is_safe_select("DELETE FROM Artist"))

    def test_empty_query_is_rejected(self) -> None:
        """Test that an empty query is rejected."""
        with self.assertRaises(EmptyQueryError):
            self.validator.validate_select("   ")

    def test_multiple_statements_are_rejected(self) -> None:
        """Test that multiple statements in a query are rejected."""
        with self.assertRaises(MultipleStatementsError):
            self.validator.validate_select("SELECT 1; SELECT 2")

    def test_non_select_query_is_rejected(self) -> None:
        """Test that a non-SELECT query is rejected."""
        with self.assertRaises(NonSelectQueryError):
            self.validator.validate_select("DELETE FROM Artist")

    def test_forbidden_keyword_inside_select_is_rejected(self) -> None:
        """Test that a query with a forbidden keyword inside a SELECT is rejected."""
        with self.assertRaises(ForbiddenKeywordError):
            self.validator.validate_select(
                "WITH changed_rows AS (DELETE FROM Artist RETURNING ArtistId) SELECT * FROM changed_rows"
            )

    def test_comments_and_strings_do_not_trigger_keyword_rejection(self) -> None:
        """Test that comments and string literals do not trigger forbidden keyword rejection."""
        normalized = self.validator.assert_safe_select(
            """
            -- delete is mentioned in a comment
            SELECT 'drop table' AS note, Name
            FROM Artist
            WHERE Name = 'AC/DC'
            """
        )
        self.assertIn("SELECT", normalized)

    def test_invalid_sql_is_rejected(self) -> None:
        """Test that invalid SQL syntax is rejected."""
        with self.assertRaises(InvalidQueryError):
            self.validator.validate_select("SELECT FROM Artist")

    def test_unknown_table_is_rejected(self) -> None:
        """Test that a query referencing an unknown table is rejected."""
        with self.assertRaises(InvalidQueryError):
            self.validator.validate_select("SELECT * FROM MissingTable")


if __name__ == "__main__":
    unittest.main()
