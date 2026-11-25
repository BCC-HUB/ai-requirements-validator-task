from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Изолировать тесты от настроек LLM в окружении разработчика."""
    for name in (
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "REQVALIDATOR_CACHE_DIR",
        "LLM_TEMPERATURE",
        "LLM_MIN_INTERVAL",
        "LLM_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
