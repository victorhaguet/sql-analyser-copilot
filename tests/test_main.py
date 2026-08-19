"""Tests for main entrypoint endpoints."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from unittest.mock import patch

from main import app, confirm_query, healthcheck
from models import ClarificationAnswer, QueryResumeRequest

class MainTestCase(unittest.TestCase):
    """Test main FastAPI endpoints."""

    def test_healthcheck(self) -> None:
        """Healthcheck endpoint should return status ok."""

        result = healthcheck()
        self.assertEqual(result, {"status": "ok"})


class ConfirmQueryTestCase(unittest.TestCase):
    """Test the /query/confirm endpoint's clarification resume path."""

    def test_confirm_query_forwards_clarification_answers(self) -> None:
        """An 'answer' decision should forward the answers to resume_question as plain dicts."""
        payload = QueryResumeRequest(
            thread_id="thread-1",
            decision="answer",
            answers=[ClarificationAnswer(key="add_albums", answer="no")],
        )
        resumed_state = {
            "question": "add a new artist called X",
            "thread_id": "thread-1",
            "interrupt": {"kind": "modification_approval", "draft": "INSERT INTO Artist (Name) VALUES ('X')"},
        }

        with (
            patch("main.get_user_by_sub", return_value={"sub": "user-1"}),
            patch("main.resume_question", return_value=resumed_state) as mock_resume,
        ):
            result = confirm_query(payload, x_user_sub="user-1")

        mock_resume.assert_called_once()
        _, kwargs = mock_resume.call_args
        self.assertEqual(kwargs["answers"], [{"key": "add_albums", "answer": "no"}])
        self.assertEqual(result["interrupt"]["kind"], "modification_approval")

    def test_confirm_query_session_survives_clarification_resume(self) -> None:
        """The pending session must not be cleared by a clarification resume,
        since the same thread may pause again for modification approval."""
        app.state.pending_approval_sessions["thread-1"] = object()
        payload = QueryResumeRequest(thread_id="thread-1", decision="answer", answers=[])
        resumed_state = {
            "question": "add a new artist called X",
            "thread_id": "thread-1",
            "interrupt": {"kind": "modification_approval"},
        }

        try:
            with (
                patch("main.get_user_by_sub", return_value={"sub": "user-1"}),
                patch("main.resume_question", return_value=resumed_state),
            ):
                confirm_query(payload, x_user_sub="user-1")

            self.assertIn("thread-1", app.state.pending_approval_sessions)
        finally:
            app.state.pending_approval_sessions.pop("thread-1", None)

    def test_confirm_query_passes_none_when_no_answers_supplied(self) -> None:
        """An approve/reject decision without answers should pass answers=None through."""
        payload = QueryResumeRequest(thread_id="thread-1", decision="approve")
        resumed_state = {"question": "q", "execution_confirmed": True}

        with (
            patch("main.get_user_by_sub", return_value={"sub": "user-1"}),
            patch("main.resume_question", return_value=resumed_state) as mock_resume,
        ):
            confirm_query(payload, x_user_sub="user-1")

        _, kwargs = mock_resume.call_args
        self.assertIsNone(kwargs["answers"])


if __name__ == "__main__":
    unittest.main()
