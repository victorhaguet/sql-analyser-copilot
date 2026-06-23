"""Tests for the path utilities."""

from __future__ import annotations

import unittest
from pathlib import Path

from utils.paths import DEFAULT_DB_PATH


class PathsTestCase(unittest.TestCase):
    """Test path utility behavior."""

    def test_default_db_path_is_valid(self) -> None:
        """The default database path should resolve to a valid Path object."""
        self.assertIsInstance(DEFAULT_DB_PATH, Path)
        expected_path = Path(__file__).resolve().parents[2] / "data" / "Chinook_Sqlite.sqlite"
        self.assertEqual(DEFAULT_DB_PATH, expected_path)


if __name__ == "__main__":
    unittest.main()
