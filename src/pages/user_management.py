"""User management page — only accessible by admin users."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from pages.auth import get_api_client, is_authenticated_page, require_admin, require_auth
from pages.config import ROLE_OPTIONS, STATUS_OPTIONS
from pages.auth import render_logout_button

load_dotenv()


def _fetch_users() -> list[dict]:
    """Fetch users info (admin only)

    Returns:
        list[dict]: Admin information
    """
    require_admin()
    client = get_api_client()
    response = client.get("/auth/users")
    if response.is_error:
        st.error(f"Failed to load users: {response.text}")
        return []
    return response.json()


def _create_user(username: str, password: str, name: str | None, role: str) -> bool:
    """Create a new user

    Args:
        username (str): Username
        password (str): password
        name (str | None): name/description
        role (str): role
    Returns:
        bool: Check if the user is created or not
    """
    require_admin()
    client = get_api_client()

    payload = {"username": username, "password": password, "name": name, "role": role}
    response = client.post("/auth/users", json=payload)
    if response.is_error:
        st.error(f"Failed to create user: {response.text}")
        return False
    return True


def _update_user_role(sub: str, role: str) -> bool:
    """Update user role

    Args:
        sub (str): ID of the user
        role (str): new role

    Returns:
        bool: Check if the modification worked
    """
    require_admin()
    client = get_api_client()

    response = client.put(f"/auth/users/{sub}", json={"role": role})
    if response.is_error:
        st.error(f"Failed to update role: {response.text}")
        return False
    return True


def _update_user_active(sub: str, is_active: bool) -> bool:
    """Activate or deactivate a user account

    Args:
        sub (str): Id of the user
        is_active (bool): new active status of the user

    Returns:
        bool: Check if the modification worked
    """
    require_admin()
    client = get_api_client()

    response = client.put(f"/auth/users/{sub}", json={"is_active": is_active})
    if response.is_error:
        st.error(f"Failed to update status: {response.text}")
        return False
    return True


def _delete_user(sub: str) -> bool:
    """Delete a user account

    Args:
        sub (str): ID of the user

    Returns:
        bool: Check if the deletion worked
    """
    require_admin()
    client = get_api_client()
    response = client.delete(f"/auth/users/{sub}")
    if response.is_error:
        st.error(f"Failed to delete user: {response.text}")
        return False
    return True


def render_user_management() -> None:
    """Render the full user management page."""
    require_auth()
    require_admin()

    st.title("User Management")
    st.markdown(
        f"Logged in as **{st.session_state.get('user', {}).get('username', '')}** (admin)"
    )

    tab_create, tab_list = st.tabs(["Create User", "Manage Users"])

    with tab_create:
        st.markdown("### Create new user")
        with st.form("create_user_form", clear_on_submit=True):
            new_username = st.text_input("Username *", placeholder="Unique username")
            new_name = st.text_input(
                "Display name", placeholder="Optional display name"
            )
            new_password = st.text_input(
                "Password *", type="password", placeholder="Choose a password"
            )
            new_role = st.selectbox("Role", options=ROLE_OPTIONS)

            submitted = st.form_submit_button(
                "Create user", use_container_width=True
            )

            if submitted:
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                elif new_role not in set(ROLE_OPTIONS):
                    st.error("Invalid role selected.")
                else:
                    if _create_user(
                        new_username, new_password, new_name or None, new_role
                    ):
                        st.success(
                            f"User '{new_username}' created successfully."
                        )
                        st.rerun()

    with tab_list:
        st.markdown("### Existing users")
        users = _fetch_users()
        if not users:
            st.info("No users found.")
            return

        current_sub = st.session_state.get("user", {}).get("sub", "")

        for user in users:
            is_self = user["sub"] == current_sub
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1.3, 1.3, 0.7])
                col1.markdown(
                    f"**{user['username']}**" + (" (you)" if is_self else "")
                )
                col2.write(user.get("name") or "—")
                current_status = "Active" if user["is_active"] else "Inactive"

                with col3:
                    if is_self:
                        st.write(user["role"])
                    else:
                        new_role = st.selectbox(
                            "Role",
                            options=ROLE_OPTIONS,
                            index=ROLE_OPTIONS.index(user["role"]),
                            key=f"role_{user['sub']}",
                            label_visibility="collapsed",
                        )
                        if new_role != user["role"]:
                            if _update_user_role(user["sub"], new_role):
                                st.success("Role updated.")
                                st.rerun()

                with col4:
                    if is_self:
                        st.write(current_status)
                    else:
                        status_label = st.selectbox(
                            "Status",
                            options=list(STATUS_OPTIONS.keys()),
                            index=list(STATUS_OPTIONS.keys()).index(current_status),
                            key=f"status_{user['sub']}",
                            label_visibility="collapsed",
                        )
                        new_active = STATUS_OPTIONS[status_label]
                        if new_active != user["is_active"]:
                            if _update_user_active(user["sub"], new_active):
                                st.success("Status updated.")
                                st.rerun()

                with col5:
                    if is_self:
                        st.caption("(you)")
                    else:
                        with st.container(key=f"danger-delete-{user['sub']}"):
                            if st.button(
                                "",
                                icon=":material/delete:",
                                key=f"delete_{user['sub']}",
                                help="Permanently delete this account",
                                use_container_width=True,
                            ):
                                if _delete_user(user["sub"]):
                                    st.success("User deleted.")
                                    st.rerun()


def render_user_management_page() -> None:
    """Render the full user management page with page-level actions."""
    if is_authenticated_page():
        render_logout_button()

    render_user_management()
