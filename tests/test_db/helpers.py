"""Helpers for database-backed tests."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2
from typing import Iterator, Sequence

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


@contextmanager
def built_database(ddl_statements: Sequence[str]) -> Iterator[SQLiteDatabase]:
    """Yield a SQLiteDatabase built from scratch out of the given DDL statements.

    Lets tests define an exact, self-contained schema (tables, foreign keys,
    unique indexes) instead of depending on the shape of a committed fixture file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "built.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            for statement in ddl_statements:
                connection.execute(statement)
            connection.commit()
        finally:
            connection.close()
        yield SQLiteDatabase(database_path)
