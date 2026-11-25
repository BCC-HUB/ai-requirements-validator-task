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
def _clean_env(monkeypatch, tmp_path):
    """Изолировать тесты от настроек LLM разработчика: переменных окружения и файла ./.env."""
    monkeypatch.chdir(tmp_path)
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
        "CLAUDE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "LLM_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
