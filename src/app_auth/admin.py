"""First-run admin bootstrap.

If this is the first time the app has been run, create the admin profile
and credentials using the variables in env.
"""

import os

from .database import create_user, user_count
from .password import hash_password


def _get_admin_credentials() -> tuple[str | None, str | None]:
    """Get admin credentials from environment variables.

    Returns:
        Tuple of (username, password) or (None, None) if not configured.
    """
    username = os.getenv("INITIAL_ADMIN_USERNAME")
    password = os.getenv("INITIAL_ADMIN_PASSWORD")
    return username, password


def ensure_admin_exists() -> None:
    """Create admin account if credentials are configured and database is empty."""
    admin_username, admin_password = _get_admin_credentials()
    if not admin_username or not admin_password:
        return

    if user_count() > 0:
        return

    create_user(
        username=admin_username,
        password_hash=hash_password(admin_password),
        name="Administrator",
        role="admin",
    )


def admin_credentials_configured() -> bool:
    """Check if admin credentials are configured in environment variables.
    
    Returns:
        True if the credentials are configured. False if not
    """
    username, password = _get_admin_credentials()
    return bool(username and password)


def is_admin_account_created() -> bool:
    """
    Check if any admin account exists in the database.
    
    Returns:
        True if an admin account exists. False if not
    """
    return user_count() > 0