"""Tests for core business logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from core import _select_requested_databases
from tools.database import DatabaseError
from tests.test_db.helpers import fixture_registered_database


class CoreTestCase(unittest.TestCase):
    """Test core business logic functions."""

    def test_select_requested_databases_returns_none_when_no_configured(self) -> None:
        """Should return None when no configured databases are provided."""
        result = _select_requested_databases(None, None)
        self.assertIsNone(result)

    def test_select_requested_databases_returns_none_when_no_request(self) -> None:
        """Should return configured databases when no request is made."""
        configured = [fixture_registered_database(name="db1")]
        result = _select_requested_databases(None, configured)
        self.assertEqual(result, configured)

    def test_select_requested_databases_selects_requested(self) -> None:
        """Should filter to only requested databases."""
        configured = [
            fixture_registered_database(name="db1"),
            fixture_registered_database(name="db2"),
        ]
        result = _select_requested_databases(["db2"], configured)
        self.assertIsNotNone(result)
        assert result is not None
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "db2")

    def test_select_requested_databases_raises_for_unknown(self) -> None:
        """Should raise DatabaseError for unknown database names."""
        configured = [fixture_registered_database(name="db1")]
        
        with self.assertRaises(DatabaseError) as context:
            _select_requested_databases(["unknown"], configured)
        
        self.assertIn("unknown", str(context.exception).lower())

    def test_select_requested_databases_raises_for_empty(self) -> None:
        """Should raise DatabaseError when empty list is requested."""
        configured = [fixture_registered_database(name="db1")]
        
        with self.assertRaises(DatabaseError) as context:
            _select_requested_databases([], configured)
        
        self.assertIn("at least one", str(context.exception).lower())

    def test_select_requested_databases_raises_when_no_databases_match(self) -> None:
        """Should raise DatabaseError when no requested databases match."""
        configured = [fixture_registered_database(name="db1")]
        
        with self.assertRaises(DatabaseError) as context:
            _select_requested_databases(["db2", "db3"], configured)
        
        self.assertIn("unknown", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()
