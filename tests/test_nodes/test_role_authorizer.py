"""Tests for the role authorizer node."""

from __future__ import annotations

import unittest

from nodes.role_authorizer import AUTHORIZATION_ERROR, RoleAuthorizerNode


class RoleAuthorizerNodeTestCase(unittest.TestCase):
    """Test role-based authorization decisions."""

    def test_denies_readonly_user(self) -> None:
        """Readonly users should not be allowed to modify data."""
        result = RoleAuthorizerNode()({"user_role": "readonly"})
        self.assertFalse(result["is_authorized"])
        self.assertEqual(result["authorization_error"], AUTHORIZATION_ERROR)

    def test_allows_editor_user(self) -> None:
        """Editors should be allowed to continue."""
        result = RoleAuthorizerNode()({"user_role": "editor"})
        self.assertTrue(result["is_authorized"])
        self.assertIsNone(result["authorization_error"])

    def test_allows_admin_user(self) -> None:
        """Admins should be allowed to continue."""
        result = RoleAuthorizerNode()({"user_role": "admin"})
        self.assertTrue(result["is_authorized"])
        self.assertIsNone(result["authorization_error"])
