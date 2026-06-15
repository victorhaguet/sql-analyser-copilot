"""Login page — first page shown when user is not authenticated."""

from __future__ import annotations

import sys
import httpx
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_copilot.auth import admin_credentials_configured, is_admin_account_created # pylint: disable=no-name-in-module

load_dotenv()

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def show_login_page(api_base_url: str | None = None) -> None:
    """Show the login page

    Args:
        api_base_url (str | None, optional): API address of the backend. Defaults to None.
    """

    # Get the API base url
    base = api_base_url or st.session_state.get("api_base_url") or DEFAULT_API_BASE_URL
    st.session_state["api_base_url"] = base

    # Introduce the page
    st.title("Sign in")
    st.markdown("Use your account credentials to access the SQL copilot.")

    # Show en error message if no admin credentials are accessible in the SQL database
    if not admin_credentials_configured() and not is_admin_account_created():
        st.error(
            "No admin account configured. "
            "Set `INITIAL_ADMIN_USERNAME` and `INITIAL_ADMIN_PASSWORD` in your `.env` file "
            "to create the first admin account. See `.env.example` for reference."
        )

    # Create the login form
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
        )
        col1, col2 = st.columns([1, 3]) # pylint: disable=unused-variable
        with col1:
            login_button = st.form_submit_button("Sign in", use_container_width=True)

    # Create the login button
    if login_button:
        if not username or not password:
            st.error("Please enter both username and password.")
            return

        try:
            # Send login request to the API
            response = httpx.post(
                f"{base.rstrip('/')}/auth/login",
                json={"username": username, "password": password},
                timeout=10.0,
            )
        except Exception as exc: # pylint: disable=broad-exception-caught
            st.error(f"Could not reach the server: {exc}")
            return

        # If it doesn't work
        if response.is_error:
            st.error("Invalid username or password.")
            return

        # Set the user information as session state
        data = response.json()
        st.session_state["user"] = {
            "sub": data["sub"],
            "username": data["username"],
            "name": data.get("name"),
            "role": data["role"],
            "is_active": data["is_active"],
        }
        st.session_state["user_authenticated"] = True
        st.rerun()

    st.caption("Contact your administrator if you don't have an account or forgot your password.")


def is_authenticated() -> bool:
    """Check if the user is authenticated

    Returns:
        bool: User is connected or not
    """
    return bool(st.session_state.get("user_authenticated"))


def logout() -> None:
    """Log out of the app
    """
    # Clear the state and the user_authentication
    st.session_state.pop("user", None)
    st.session_state["user_authenticated"] = False

    # Rerun the login page
    st.rerun()
