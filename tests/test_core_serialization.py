"""Tests for core business logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break


class CoreTestCase(unittest.TestCase):
    """Test core business logic functions."""


if __name__ == "__main__":
    unittest.main()
