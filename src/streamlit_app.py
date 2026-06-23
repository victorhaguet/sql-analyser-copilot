"""Streamlit entrypoint for the SQL copilot UI."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit.navigation.page import StreamlitPage

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pages.sql_copilot import render_sql_copilot_page
from pages.user_management import render_user_management_page
from streamlit_ui import load_stylesheet


def _build_navigation_pages() -> list[StreamlitPage]:
    """Build the side bar based on the user role 

    Returns:
        list[StreamlitPage]: List of accessible pages the user can see
    """
    pages = [
        st.Page(
            render_sql_copilot_page,
            title="SQL Copilot",
            icon=":material/database:",
        )
    ]

    user = st.session_state.get("user", {})
    if user.get("role") == "admin":
        pages.append(
            st.Page(
                render_user_management_page,
                title="User Management",
                icon=":material/manage_accounts:",
            )
        )

    return pages


def render_app() -> None:
    """Configure the Streamlit shell and render the selected page."""
    st.set_page_config(
        page_title="SQL Copilot",
        page_icon=":material/database:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(load_stylesheet(), unsafe_allow_html=True)

    navigation = st.navigation(_build_navigation_pages())
    navigation.run()


if __name__ == "__main__":
    render_app()
