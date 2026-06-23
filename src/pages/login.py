"""Login page — first page shown when user is not authenticated."""

from __future__ import annotations

import httpx

import streamlit as st
from dotenv import load_dotenv

from app_auth.admin import admin_credentials_configured, is_admin_account_created
from pages.config import DEFAULT_API_BASE_URL

load_dotenv()


def show_login_page(api_base_url: str | None = None) -> None:
    """Show the login page

    Args:
        api_base_url (str | None, optional): API address of the backend. Defaults to None.
    """

    base = api_base_url or st.session_state.get("api_base_url") or DEFAULT_API_BASE_URL
    st.session_state["api_base_url"] = base

    st.title("Sign in")
    st.markdown("Use your account credentials to access the SQL copilot.")

    if not admin_credentials_configured() and not is_admin_account_created():
        st.error(
            "No admin account configured. "
            "Set `INITIAL_ADMIN_USERNAME` and `INITIAL_ADMIN_PASSWORD` in your `.env` file "
            "to create the first admin account. See `.env.example` for reference."
        )

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
        )
        col1, _ = st.columns([1, 3])
        with col1:
            login_button = st.form_submit_button("Sign in", use_container_width=True)

    if login_button:
        if not username or not password:
            st.error("Please enter both username and password.")
            return

        try:
            response = httpx.post(
                f"{base.rstrip('/')}/auth/login",
                json={"username": username, "password": password},
                timeout=10.0,
            )
        except Exception as exc:
            st.error(f"Could not reach the server: {exc}")
            return

        if response.is_error:
            st.error("Invalid username or password.")
            return

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


def is_authenticated_page() -> bool:
    """Check if a user is logged in.

    Returns:
        True if a user is logged in, otherwise False.
    """
    return bool(st.session_state.get("user_authenticated"))
