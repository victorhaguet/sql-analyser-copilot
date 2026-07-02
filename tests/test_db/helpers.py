"""Helpers for database-backed tests."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2
from typing import Iterator

from tools.database import RegisteredDatabase, SQLiteDatabase, register_database

FIXTURE_DB_PATH = Path(__file__).resolve().parent / "Chinook_Sqlite.sqlite"


def fixture_database() -> SQLiteDatabase:
    """Return a database handle bound to the committed test fixture."""
    return SQLiteDatabase(FIXTURE_DB_PATH)


def fixture_registered_database(
    *,
    name: str = "chinook",
    description: str = "Chinook fixture data",
) -> RegisteredDatabase:
    """Return a registered database backed by the committed test fixture."""
    return register_database(
        fixture_database(),
        name=name,
        description=description,
    )


@contextmanager
def mutable_fixture_database() -> Iterator[SQLiteDatabase]:
    """Yield an isolated writable copy of the fixture database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / FIXTURE_DB_PATH.name
        copy2(FIXTURE_DB_PATH, database_path)
        yield SQLiteDatabase(database_path)
