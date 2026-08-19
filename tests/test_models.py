"""Tests for Pydantic models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from models import (
    ClarificationAnswer,
    QueryRequest,
    QueryResumeRequest,
    LoginRequest,
    UserCreate,
    UserUpdate,
    UserResponse,
    QueryResultPayload,
    QueryResponse,
)

class ModelsTestCase(unittest.TestCase):
    """Test Pydantic model validation and serialization."""

    def test_query_request_validation(self) -> None:
        """QueryRequest should validate required fields."""
        
        request = QueryRequest(question="Test question")
        self.assertEqual(request.question, "Test question")
        self.assertEqual(request.execution_limit, 200)
        self.assertIsNone(request.selected_database)

    def test_query_request_validation_with_custom_limit(self) -> None:
        """QueryRequest should accept custom execution_limit."""
        
        request = QueryRequest(question="Test question", execution_limit=500)
        self.assertEqual(request.execution_limit, 500)

    def test_query_request_validation_min_length(self) -> None:
        """QueryRequest should require non-empty question."""
        
        with self.assertRaises(ValueError):
            QueryRequest(question="")

    def test_query_request_validation_positive_limit(self) -> None:
        """QueryRequest should require positive execution_limit."""
        
        with self.assertRaises(ValueError):
            QueryRequest(question="Test", execution_limit=0)

    def test_login_request_validation(self) -> None:
        """LoginRequest should validate required fields."""
        
        request = LoginRequest(username="testuser", password="testpass")
        self.assertEqual(request.username, "testuser")
        self.assertEqual(request.password, "testpass")

    def test_user_create_validation(self) -> None:
        """UserCreate should validate required fields."""
        
        request = UserCreate(username="testuser", password="testpass")
        self.assertEqual(request.username, "testuser")
        self.assertEqual(request.password, "testpass")
        self.assertIsNone(request.name)
        self.assertEqual(request.role, "readonly")

    def test_user_create_with_role(self) -> None:
        """UserCreate should accept custom role."""
        
        request = UserCreate(username="testuser", password="testpass", role="admin")
        self.assertEqual(request.role, "admin")

    def test_user_create_with_name(self) -> None:
        """UserCreate should accept optional name."""
        
        request = UserCreate(username="testuser", password="testpass", name="Test User")
        self.assertEqual(request.name, "Test User")

    def test_user_update_validation(self) -> None:
        """UserUpdate should accept partial updates."""
        
        request = UserUpdate()
        self.assertIsNone(request.name)
        self.assertIsNone(request.role)
        self.assertIsNone(request.is_active)

    def test_user_update_with_values(self) -> None:
        """UserUpdate should accept partial field updates."""
        
        request = UserUpdate(name="New Name", is_active=False)
        self.assertEqual(request.name, "New Name")
        self.assertFalse(request.is_active)
        self.assertIsNone(request.role)

    def test_user_response_validation(self) -> None:
        """UserResponse should validate required fields."""
        
        response = UserResponse(
            sub="test-sub",
            username="testuser",
            role="readonly",
            is_active=True,
            created_at="2024-01-01",
        )
        self.assertEqual(response.sub, "test-sub")
        self.assertEqual(response.username, "testuser")
        self.assertEqual(response.role, "readonly")
        self.assertTrue(response.is_active)
        self.assertEqual(response.created_at, "2024-01-01")
        self.assertIsNone(response.name)

    def test_query_result_payload_validation(self) -> None:
        """QueryResultPayload should validate required fields."""
        
        payload = QueryResultPayload(
            columns=["id", "name"],
            rows=[{"id": 1, "name": "Test"}],
            row_count=1,
            truncated=False,
        )
        self.assertEqual(payload.columns, ["id", "name"])
        self.assertEqual(payload.rows, [{"id": 1, "name": "Test"}])
        self.assertEqual(payload.row_count, 1)
        self.assertFalse(payload.truncated)

    def test_query_response_validation(self) -> None:
        """QueryResponse should validate all fields."""
        
        response = QueryResponse(
            question="Test question",
            schema_overview="Test schema",
            selected_database="db1",
            generated_sql="SELECT * FROM test",
            validated_sql="SELECT * FROM test",
            sql_validation_error=None,
            query_result=None,
            execution_error=None,
            analysis="Test analysis",
            metadata={"key": "value"},
        )
        self.assertEqual(response.question, "Test question")
        self.assertEqual(response.schema_overview, "Test schema")
        self.assertEqual(response.analysis, "Test analysis")
        self.assertEqual(response.metadata, {"key": "value"})

    def test_query_response_defaults_agent_fields(self) -> None:
        """QueryResponse should default the agent-loop fields when not supplied."""

        response = QueryResponse(question="Test question")
        self.assertEqual(response.messages, [])
        self.assertIsNone(response.agent_status)
        self.assertEqual(response.agent_iterations, 0)
        self.assertEqual(response.clarification_answers, [])
        self.assertEqual(response.agent_tool_log, [])

    def test_clarification_answer_validation(self) -> None:
        """ClarificationAnswer should require both key and answer."""

        answer = ClarificationAnswer(key="add_albums", answer="no")
        self.assertEqual(answer.key, "add_albums")
        self.assertEqual(answer.answer, "no")

    def test_query_resume_request_accepts_approve_decision(self) -> None:
        """QueryResumeRequest should still accept the original approve/reject decisions."""

        request = QueryResumeRequest(thread_id="thread-1", decision="approve")
        self.assertEqual(request.decision, "approve")
        self.assertIsNone(request.answers)

    def test_query_resume_request_accepts_answer_decision_with_answers(self) -> None:
        """QueryResumeRequest should accept a clarification 'answer' decision with answers."""

        request = QueryResumeRequest(
            thread_id="thread-1",
            decision="answer",
            answers=[ClarificationAnswer(key="add_albums", answer="yes")],
        )
        self.assertEqual(request.decision, "answer")
        self.assertEqual(request.answers[0].key, "add_albums")

    def test_query_resume_request_rejects_invalid_decision(self) -> None:
        """QueryResumeRequest should reject decisions outside the known set."""

        with self.assertRaises(ValueError):
            QueryResumeRequest(thread_id="thread-1", decision="maybe")


if __name__ == "__main__":
    unittest.main()
