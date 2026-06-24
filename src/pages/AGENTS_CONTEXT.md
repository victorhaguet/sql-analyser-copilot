# src/pages/ AGENTS_CONTEXT.md

## Purpose
Streamlit page components for the web UI.

## Key Files

- **login.py**: Login page component with authentication form
- **auth.py**: Shared authentication helpers (is_authenticated, is_admin, require_auth, get_api_client)
- **sql_copilot.py**: Main SQL copilot page with question input, database selection, results
- **user_management.py**: User administration page (admin-only)
- **config.py**: Configuration constants (DEFAULT_API_BASE_URL, DEFAULT_QUESTION)

## Important Exports

- Page rendering functions for Streamlit app
- HTTP client with authentication headers
- Session state management helpers

## Key Features

- Login form with username/password
- Role-based page access (admin, editor, readonly)
- Database catalog selection with toggles
- Question input with execution limit slider
- Result display with tabs (Answer, SQL, Preview)
- API client with user authentication headers

## When to Read Files

- **login.py**: When modifying login UI
- **auth.py**: When updating auth helpers
- **sql_copilot.py**: When changing main SQL copilot interface
- **user_management.py**: When updating admin user management

## Related Contexts

- **src/AGENTS_CONTEXT.md** - Streamlit UI helpers in streamlit_ui.py
