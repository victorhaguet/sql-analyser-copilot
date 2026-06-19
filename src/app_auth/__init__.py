"""Authentication module — not accessible to the LLM agent."""

from .admin import admin_credentials_configured, ensure_admin_exists, is_admin_account_created
from .database import (
    create_user,
    deactivate_user,
    get_user_by_sub,
    get_user_by_username,
    delete_user,
    init_user_db,
    list_users,
    update_user,
    user_count,
)
from .password import hash_password, verify_password

__all__ = [
    "admin_credentials_configured",
    "create_user",
    "deactivate_user",
    "ensure_admin_exists",
    "get_user_by_sub",
    "get_user_by_username",
    "delete_user",
    "hash_password",
    "init_user_db",
    "is_admin_account_created",
    "list_users",
    "update_user",
    "user_count",
    "verify_password",
]