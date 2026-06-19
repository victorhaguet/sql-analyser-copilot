"""Path utilities for the project."""

from pathlib import Path

# Default database path - will be overridden by auto-discovery
# This is used only when no SQLite files are found in the data/ directory
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "Chinook_Sqlite.sqlite"
