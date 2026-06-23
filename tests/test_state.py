"""Tests for state definitions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import get_type_hints

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from state import SQLAgentState


class StateTestCase(unittest.TestCase):
    """Test state type definitions."""

    def test_sql_agent_state_is_typeddict(self) -> None:
        """SQLAgentState should be a TypedDict."""

        self.assertTrue(hasattr(SQLAgentState, "__total__"))
        
        hints = get_type_hints(SQLAgentState)
        self.assertIn("question", hints)
        self.assertIn("analysis", hints)
        self.assertIn("metadata", hints)

    def test_sql_agent_state_optional_fields(self) -> None:
        """All SQLAgentState fields should be optional."""
        
        self.assertFalse(SQLAgentState.__total__)


if __name__ == "__main__":
    unittest.main()
