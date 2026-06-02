"""First-run admin bootstrap.
If this is the first time the app has been run, 
create the admin profile and credentials using the variables in env.
"""

import os

from sql_copilot.auth.database import create_user, init_user_db, user_count
from sql_copilot.auth.password import hash_password


def ensure_admin_exists() -> None:
    """
    Initialize the user database, if it doesn't exist yet.
    Create the admin account if credentials are configured and
    the database is empty.
    """
    # Create an empty user database (if it doesn't already exist)
    init_user_db()

    # Get credentials of the admin
    admin_username = os.getenv("INITIAL_ADMIN_USERNAME")
    admin_password = os.getenv("INITIAL_ADMIN_PASSWORD")

    # If the credentials are missing, stop the function
    if not admin_username or not admin_password:
        return

    # If the database already has user, it isn't needed to create a new admin
    if user_count() > 0:
        return

    # If the database is empty, create the admin account
    create_user(
        username=admin_username,
        password_hash=hash_password(admin_password),
        name="Administrator",
        role="admin",
    )


def admin_credentials_configured() -> bool:
    username = os.getenv("INITIAL_ADMIN_USERNAME")
    password = os.getenv("INITIAL_ADMIN_PASSWORD")
    return bool(username and password)


def is_admin_account_created() -> bool:
    init_user_db()
    return user_count() > 0