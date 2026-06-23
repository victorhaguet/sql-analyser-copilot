"""Tests for configuration loading and app creation."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

import config


class ConfigTestCase(unittest.TestCase):
    """Test configuration functions."""

    def setUp(self) -> None:
        """Save original environment variables and set up test mocks."""
        self._original_env = {
            "OPENAI_API_KEY": None,
            "OPENAI_BASE_URL": None,
            "SQL_COPILOT_MODEL": None,
            "SQL_COPILOT_ANALYST_MODEL": None,
            "ENABLE_TRACE_LOGGING": None,
            "TRACE_LOG_DIR": None,
        }
        for key in self._original_env:
            self._original_env[key] = os.getenv(key)
        
        self._original_modules = {
            name: sys.modules.get(name)
            for name in ("fastapi",)
        }

    def tearDown(self) -> None:
        """Restore original environment variables and modules."""
        for key in self._original_env:
            if key in os.environ:
                del os.environ[key]
        for key, value in self._original_env.items():
            if value is not None:
                os.environ[key] = value
        
        for name, module in self._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_load_models_from_env_returns_none_when_no_api_key(self) -> None:
        """Should return (None, None) when OPENAI_API_KEY is not set."""        
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("SQL_COPILOT_MODEL", None)
        
        result = config.load_models_from_env()
        self.assertEqual(result, (None, None))

    def test_load_models_from_env_returns_none_when_no_model_name(self) -> None:
        """Should return (None, None) when SQL_COPILOT_MODEL is not set."""     
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ.pop("SQL_COPILOT_MODEL", None)
        
        result = config.load_models_from_env()
        self.assertEqual(result, (None, None))

    def test_load_models_from_env_creates_models_from_env(self) -> None:
        """Should create generator and analyst models from environment variables."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["SQL_COPILOT_MODEL"] = "gpt-4"
        os.environ["SQL_COPILOT_ANALYST_MODEL"] = "gpt-4-analyst"
        
        result = config.load_models_from_env()
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        assert result[0] is not None
        assert result[1] is not None
        self.assertEqual(result[0].model, "gpt-4")
        self.assertEqual(result[1].model, "gpt-4-analyst")

    def test_load_models_from_env_uses_different_analyst_model(self) -> None:
        """Should use SQL_COPILOT_ANALYST_MODEL for analyst when set."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["SQL_COPILOT_MODEL"] = "gpt-4"
        os.environ["SQL_COPILOT_ANALYST_MODEL"] = "gpt-4-analyst"
        
        result = config.load_models_from_env()
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        assert result[0] is not None
        assert result[1] is not None
        self.assertEqual(result[0].model, "gpt-4")
        self.assertEqual(result[1].model, "gpt-4-analyst")

    def test_load_models_from_env_includes_base_url(self) -> None:
        """Should pass OPENAI_BASE_URL to model constructors when set."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["SQL_COPILOT_MODEL"] = "gpt-4"
        os.environ["OPENAI_BASE_URL"] = "https://custom.api.endpoint"
        os.environ.pop("SQL_COPILOT_ANALYST_MODEL", None)
        
        result = config.load_models_from_env()
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        assert result[0] is not None
        assert result[1] is not None
        self.assertEqual(result[0].model, "gpt-4")
        self.assertEqual(result[1].model, "gpt-4")

    def test_load_trace_config_defaults(self) -> None:
        """Should return default values when environment variables are not set."""
        os.environ.pop("ENABLE_TRACE_LOGGING", None)
        os.environ.pop("TRACE_LOG_DIR", None)
        
        enable_trace, trace_log_dir = config.load_trace_config()
        
        self.assertFalse(enable_trace)
        self.assertEqual(trace_log_dir, "logs")

    def test_load_trace_config_enabled(self) -> None:
        """Should enable trace when ENABLE_TRACE_LOGGING is true."""
        os.environ["ENABLE_TRACE_LOGGING"] = "true"
        os.environ["TRACE_LOG_DIR"] = "/tmp/traces"
        
        enable_trace, trace_log_dir = config.load_trace_config()
        
        self.assertTrue(enable_trace)
        self.assertEqual(trace_log_dir, "/tmp/traces")

    def test_load_trace_config_enabled_variations(self) -> None:
        """Should enable trace for various truthy values."""
        for value in ("1", "yes", "true"):
            os.environ["ENABLE_TRACE_LOGGING"] = value
            enable_trace, _ = config.load_trace_config()
            self.assertTrue(enable_trace, f"value {value} should be truthy")

    def test_load_trace_config_enabled_false(self) -> None:
        """Should disable trace for various falsy values."""
        for value in ("false", "0", "no", ""):
            os.environ["ENABLE_TRACE_LOGGING"] = value
            enable_trace, _ = config.load_trace_config()
            self.assertFalse(enable_trace, f"value {value} should be falsy")


if __name__ == "__main__":
    unittest.main()
