"""Tests for main entrypoint endpoints."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from main import healthcheck

class MainTestCase(unittest.TestCase):
    """Test main FastAPI endpoints."""

    def test_healthcheck(self) -> None:
        """Healthcheck endpoint should return status ok."""
        
        result = healthcheck()
        self.assertEqual(result, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
