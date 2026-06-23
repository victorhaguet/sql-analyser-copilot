"""Tests for the node utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.nodes import load_prompt, render_prompt


class NodesTestCase(unittest.TestCase):
    """Test node utility behavior."""

    def test_render_prompt_with_simple_template(self) -> None:
        """Should render a basic Jinja template."""
        template = "Hello {{ name }}!"
        result = render_prompt(template, name="World")

        self.assertEqual(result, "Hello World!")

    def test_render_prompt_with_loop(self) -> None:
        """Should handle Jinja loops."""
        template = "{% for item in items %}{{ item }} {% endfor %}"
        result = render_prompt(template, items=["a", "b", "c"])

        self.assertEqual(result, "a b c")

    def test_render_prompt_strips_whitespace(self) -> None:
        """Should strip leading and trailing whitespace from rendered output."""
        template = """
            {{ value }}
        """
        result = render_prompt(template, value="test")

        self.assertEqual(result, "test")

    def test_render_prompt_with_strict_undefined_raises_error(self) -> None:
        """Should raise error for undefined variables."""
        template = "{{ missing_var }}"

        with self.assertRaises(Exception):
            render_prompt(template)

    def test_load_prompt_returns_file_content(self) -> None:
        """Should load and return content from an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write("Template content")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prompt(path, "fallback")

            self.assertEqual(result, "Template content")
        finally:
            path.unlink()

    def test_load_prompt_returns_fallback_when_file_missing(self) -> None:
        """Should return fallback when file does not exist."""
        path = Path("/nonexistent/file.jinja")
        fallback = "default text"

        result = load_prompt(path, fallback)

        self.assertEqual(result, fallback)

    def test_load_prompt_strips_content(self) -> None:
        """Should strip content from loaded file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write("  content with spaces  ")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prompt(path, "fallback")

            self.assertEqual(result, "content with spaces")
        finally:
            path.unlink()

    def test_load_prompt_empty_file_returns_fallback(self) -> None:
        """Should return fallback when file is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            result = load_prompt(path, "fallback")

            self.assertEqual(result, "fallback")
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
