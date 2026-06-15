"""User database management for authentication."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_USERS_DIR = Path(__file__).resolve().parents[3] / ".users"
_USERS_DB = _USERS_DIR / "users.db"

_ROLE_ADMIN = "admin"
_ROLE_READONLY = "readonly"


def _ensure_users_dir() -> None:
    """Make sure the user database exist
    """
    _USERS_DIR.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    """Connect to the database if it exist

    Returns:
        sqlite3.Connection: _description_
    """
    # Check the user database exist
    _ensure_users_dir()

    # Connect to the user database
    conn = sqlite3.connect(_USERS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db() -> None:
    """Initialize the SQL table (if needed)"""
    # Connect to the user database
    conn = _get_connection()
    try:
        # Initialize the user table if it doesn't already exist
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
    """Create a new user

    Args:
        username (str): username of the user
        password_hash (str): encoded password
        name (str | None, optional): name in the app of the user. Defaults to None.
        role (str, optional): Role of the user. Defaults to _ROLE_READONLY.

    Raises:
        ValueError: raise an error if the user already exist

    Returns:
        dict[str, Any]: Return a dictionnary of the created user information.
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()


    try:
        # Check if the user already exist in the table
        existing = conn.execute(
            "SELECT sub FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        # Insert the new user
        sub = str(uuid4()) # Create an ID
        now = datetime.utcnow().isoformat() # Get the current time
        conn.execute(
            "INSERT INTO users (sub, username, name, password_hash, role, is_active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (sub, username, name, password_hash, role, now),
        )
        conn.commit()

        user = _row_to_user(conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone())

        # Make sure user isn't None
        if user is None:
            raise RuntimeError("User could not be created.")
        
        return user
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict[str, Any] | None:
    """_summary_

    Args:
        username (str): username

    Returns:
        dict[str, Any] | None: user information
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()

    # Get the user information from its username
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return _row_to_user(row) if row else None
    finally:
        conn.close()


def get_user_by_sub(sub: str) -> dict[str, Any] | None:
    """Get the user info using its ID

    Args:
        sub (str): ID

    Returns:
        dict[str, Any] | None: user information
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()

    # Get the user information from its ID
    try:
        row = conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        return _row_to_user(row) if row else None
    finally:
        conn.close()


def list_users() -> list[dict[str, Any]]:
    """List all the users of the table

    Returns:
        list[dict[str, Any]]: _description_
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()
    l_users: list[dict[str, Any]] = [] # Empty list of users

    # List all the users of the table
    try:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()

        for row in rows:
            user = _row_to_user(row, include_hash=False)

            # Make sure user isn't None
            if user is None:
                raise RuntimeError("User could not be created.")
            
            l_users.append(user)
        
        return l_users
    finally:
        conn.close()


def update_user(
    sub: str,
    name: str | None = None,
    role: str | None = None,
    is_active: bool | None = None,
) -> dict[str, Any] | None:
    """_summary_

    Args:
        sub (str): _description_
        name (str | None, optional): _description_. Defaults to None.
        role (str | None, optional): _description_. Defaults to None.
        is_active (bool | None, optional): _description_. Defaults to None.

    Returns:
        dict[str, Any] | None: _description_
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()

    try:
        # Find a user from its ID
        user = conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        if not user:
            return None

        # Update its parameters
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

        # Return the information of the user
        row = conn.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        return _row_to_user(row, include_hash=False)
    finally:
        conn.close()


def deactivate_user(sub: str) -> bool:
    """Deactivate the account of a user

    Args:
        sub (str): User ID

    Returns:
        bool: Success of the deactivation (or not)
    """

    updated = update_user(sub, is_active=False)
    return updated is not None


def delete_user(sub: str) -> bool:
    """Delete a user

    Args:
        sub (str): user ID

    Returns:
        bool: Success of the deletion
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()

    # Delete the user of the database
    try:
        conn.execute("DELETE FROM users WHERE sub = ?", (sub,))
        conn.commit()
        return True
    finally:
        conn.close()


def user_count() -> int:
    """Count the number of users in the table

    Returns:
        int: Number of users
    """
    # Initiaize the SQL table (if it doesn't exist) and connect to it
    init_user_db()
    conn = _get_connection()

    # Count the number of users
    try:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return count
    finally:
        conn.close()


def _row_to_user(row: sqlite3.Row | None, include_hash: bool = True) -> dict[str, Any] | None:
    """Transform a user SQL row into a python dictionnary

    Args:
        row (sqlite3.Row | None): user row
        include_hash (bool, optional): Boolean to know if the hash_code should also be accessible. Defaults to True.

    Returns:
        dict[str, Any] | None: _description_
    """
    # If the row doesn't exist, ignore this function
    if row is None:
        return None
    
    # Format the user information
    user = {
        "sub": row["sub"],
        "username": row["username"],
        "name": row["name"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }

    # Add the hashed code if asked
    if include_hash:
        user["password_hash"] = row["password_hash"]
    return user