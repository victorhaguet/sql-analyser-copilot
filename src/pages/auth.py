"""Shared authentication helpers for the pages module."""

from __future__ import annotations

import httpx
import streamlit as st

from pages.config import DEFAULT_API_BASE_URL


def is_authenticated_page() -> bool:
    """Check if a user is logged in.

    Returns:
        True if a user is logged in, otherwise False.
    """
    return bool(st.session_state.get("user_authenticated"))


def is_admin() -> bool:
    """Check if the current user is an admin.

    Returns:
        True if the user has admin role, otherwise False.
    """
    user = st.session_state.get("user", {})
    return user.get("role") == "admin"


def require_auth() -> None:
    """Stop page execution if the user is not authenticated."""
    if not is_authenticated_page():
        st.error("Please log in first.")
        st.stop()


def require_admin() -> None:
    """Stop page execution if the current user is not an admin."""
    if not is_admin():
        st.error("Your account is not authorized to access this page.")
        st.stop()


def get_api_client() -> httpx.Client:
    """Get the FastAPI backend client with user authentication headers.

    Returns:
        An httpx.Client configured with user headers.
    """
    user = st.session_state.get("user", {})
    base = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    headers = {
        "X-User-Sub": user.get("sub", ""),
        "X-User-Role": user.get("role", ""),
    }
    return httpx.Client(base_url=base.rstrip("/"), headers=headers, timeout=30.0)


def render_logout_button() -> None:
    """Render the sign-out button with user information."""
    user = st.session_state.get("user", {})
    
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"Signed in as {user.get('username', '')}")
        st.space("stretch")

    if st.button("Sign out"):
        logout()


def logout() -> None:
    """Log out the current user and redirect to the login page."""
    st.session_state.pop("user", None)
    st.session_state["user_authenticated"] = False
    st.rerun()
