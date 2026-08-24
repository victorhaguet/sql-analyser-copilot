"""Tests for the SQL modification validator node."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from nodes.sql_modification_validator import SQLModificationValidatorNode


class SQLModificationValidatorNodeTestCase(unittest.TestCase):
    """Test modification approval interrupts."""

    def test_approves_execution_when_user_confirms(self) -> None:
        """Approved interrupts should mark execution as confirmed."""
        with patch(
            "nodes.sql_modification_validator.interrupt",
            return_value="approve",
        ):
            result = SQLModificationValidatorNode()(
                {
                    "question": "Delete artist 1",
                    "generated_sql": "DELETE FROM Artist WHERE ArtistId = 1",
                }
            )

        self.assertTrue(result["execution_confirmed"])
        self.assertFalse(result["execution_requested"])

    def test_rejects_execution_when_user_declines(self) -> None:
        """Rejected interrupts should leave execution unconfirmed."""
        with patch(
            "nodes.sql_modification_validator.interrupt",
            return_value="reject",
        ):
            result = SQLModificationValidatorNode()(
                {
                    "question": "Delete artist 1",
                    "generated_sql": "DELETE FROM Artist WHERE ArtistId = 1",
                }
            )

        self.assertFalse(result["execution_confirmed"])
        self.assertFalse(result["execution_requested"])

    def test_interrupt_payload_carries_modification_approval_kind(self) -> None:
        """The interrupt payload must be tagged so the UI can dispatch on `kind`."""
        with patch(
            "nodes.sql_modification_validator.interrupt",
            return_value="approve",
        ) as mock_interrupt:
            SQLModificationValidatorNode()(
                {
                    "question": "Delete artist 1",
                    "generated_sql": "DELETE FROM Artist WHERE ArtistId = 1",
                }
            )

        payload = mock_interrupt.call_args[0][0]
        self.assertEqual(payload["kind"], "modification_approval")
