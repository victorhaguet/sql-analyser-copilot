"""User database management for authentication."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, overload
from uuid import uuid4

_USERS_DIR = Path(__file__).resolve().parents[3] / ".users"
_USERS_DB = _USERS_DIR / "users.db"

_ROLE_ADMIN = "admin"
_ROLE_READONLY = "readonly"


def _ensure_users_dir() -> None:
    """Ensure the users directory exists."""
    _USERS_DIR.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    """Create a database connection.

    Returns:
        sqlite3.Connection: Database connection with row factory set.
    """
    _ensure_users_dir()
    conn = sqlite3.connect(_USERS_DB)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db_connection() -> Iterator[sqlite3.Connection]:
    """Context manager for database connections.

    Yields:
        sqlite3.Connection: Database connection that will be closed automatically.
    """
    conn = _get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_user_db() -> None:
    """Initialize the SQL table (if needed)"""
    conn = _get_connection()

    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                sub TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                name TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'readonly',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


def create_user(
    username: str,
    password_hash: str,
    name: str | None = None,
    role: str = _ROLE_READONLY,
) -> dict[str, Any]:
    """Create a new user.

    Args:
        username: Username of the user.
        password_hash: Encoded password.
        name: Name in the app of the user. Defaults to None.
        role: Role of the user. Defaults to _ROLE_READONLY.

    Raises:
        ValueError: If the user already exists.

    Returns:
        Dictionary of the created user information.
    """
    init_user_db()
    with _db_connection() as conn:
        existing = conn.execute(
            "SELECT sub FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        sub = str(uuid4())
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO users (sub, username, name, password_hash, role, is_active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (sub, username, name, password_hash, role, now),
        )
        conn.commit()

        user = _row_to_user(conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone())
        if user is None:
            raise RuntimeError("User could not be created.")
        return user


def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Get user by username.

    Args:
        username: Username to look up.

    Returns:
        User information dictionary or None if not found.
    """
    init_user_db()
    with _db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return _row_to_user(row) if row else None


def get_user_by_sub(sub: str) -> dict[str, Any] | None:
    """Get user by ID.

    Args:
        sub: User ID.

    Returns:
        User information dictionary or None if not found.
    """
    init_user_db()
    with _db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        return _row_to_user(row) if row else None


def list_users() -> list[dict[str, Any]]:
    """List all users.

    Returns:
        List of user information dictionaries (without password hashes).
    """
    init_user_db()
    with _db_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [
            _row_to_user(row, include_hash=False)
            for row in rows
            if row is not None
        ]


def update_user(
    sub: str,
    name: str | None = None,
    role: str | None = None,
    is_active: bool | None = None,
) -> dict[str, Any] | None:
    """Update user information.

    Args:
        sub: User ID.
        name: New name. Defaults to None.
        role: New role. Defaults to None.
        is_active: New active status. Defaults to None.

    Returns:
        Updated user information or None if user not found.
    """
    init_user_db()
    with _db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        if not user:
            return None

        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if role is not None:
            updates.append("role = ?")
            params.append(role)
        if is_active is not None:
            updates.append("is_active = ?")
            params.append("1" if is_active else "0")

        if updates:
            params.append(sub)
            conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE sub = ?", params)
            conn.commit()

        row = conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        return _row_to_user(row, include_hash=False)


def deactivate_user(sub: str) -> bool:
    """Deactivate a user account.

    Args:
        sub: User ID.

    Returns:
        True if the account was deactivated, False if user not found.
    """
    return update_user(sub, is_active=False) is not None


def delete_user(sub: str) -> bool:
    """Delete a user.

    Args:
        sub: User ID.

    Returns:
        True if the user was deleted, False if not found.
    """
    init_user_db()
    with _db_connection() as conn:
        cursor = conn.execute("DELETE FROM users WHERE sub = ?", (sub,))
        conn.commit()
        return cursor.rowcount > 0


def user_count() -> int:
    """Count the number of users.

    Returns:
        Number of users in the database.
    """
    init_user_db()
    with _db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return count


@overload
def _row_to_user(row: sqlite3.Row, include_hash: bool = True) -> dict[str, Any]: ...

@overload
def _row_to_user(row: None, include_hash: bool = True) -> None: ...

def _row_to_user(row: sqlite3.Row | None, include_hash: bool = True) -> dict[str, Any] | None:
    """Transform a user SQL row into a python dictionnary

    Args:
        row (sqlite3.Row | None): user row
        include_hash (bool, optional): Boolean to know if the hash_code should also be accessible. Defaults to True.

    Returns:
        dict[str, Any] | None: _description_
    """
    if row is None:
        return None
    
    user = {
        "sub": row["sub"],
        "username": row["username"],
        "name": row["name"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }

    if include_hash:
        user["password_hash"] = row["password_hash"]
    return user