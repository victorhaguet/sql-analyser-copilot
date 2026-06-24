# src/app_auth/ AGENTS_CONTEXT.md

## Purpose
Authentication module for user management

## Key Files

- **database.py**: User database operations (CRUD, initialization)
- **password.py**: Password hashing (Argon2) and verification
- **admin.py**: Admin account bootstrap (ensure_admin_exists, is_admin_account_created)

## Important Exports

- create_user(), update_user(), delete_user(), list_users() - User management
- get_user_by_username(), get_user_by_sub() - User lookup
- hash_password(), verify_password() - Password operations
- ensure_admin_exists(), is_admin_account_created() - Admin bootstrap

## Key Features

- SQLite-based user database
- Argon2 password hashing
- Role-based access control (admin, editor, readonly)
- Admin bootstrap for first-time setup
- User deactivation support

## When to Read Files

- **database.py**: When adding user database features
- **password.py**: When updating password hashing or verification
- **admin.py**: When modifying admin account creation logic

## Related Contexts

- **src/AGENTS_CONTEXT.md** - Authentication endpoints in main.py
