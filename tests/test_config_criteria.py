from pathlib import Path

import pytest

from reqvalidator.config import ConfigError, load_settings
from reqvalidator.criteria import ALL, normalize_criterion, normalize_severity
from reqvalidator.version import agent_version


def test_gemini_defaults_from_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc123\nLLM_MIN_INTERVAL=2\n", encoding="utf-8")
    s = load_settings(env_file=env)
    assert s.provider == "gemini" and s.api_key == "abc123"
    assert s.model == "gemini-flash-latest"
    assert s.min_interval == 2 and s.concurrency == 1
    assert "abc123" not in repr(s)  # ключ не попадает в журналы


def test_placeholder_key_is_not_accepted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(ConfigError, match="LLM"):
        load_settings(env_file=None)


def test_explicit_provider_and_model_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    s = load_settings()
    assert (s.provider, s.model, s.base_url) == ("ollama", "llama3", "http://localhost:11434")
    s2 = load_settings(provider="claude-cli", model="sonnet")
    assert (s2.provider, s2.model, s2.effort) == ("claude-cli", "sonnet", "high")


def test_missing_key_for_selected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("REQVALIDATOR_CACHE_DIR", "")
    assert load_settings().cache_dir is None


def test_bad_numbers_are_reported(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("LLM_CONCURRENCY", "много")
    with pytest.raises(ConfigError, match="LLM_CONCURRENCY"):
        load_settings()


def test_missing_env_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        load_settings(env_file=tmp_path / "nope.env")


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("И3", "И3"),
        ("и 7", "И7"),
        ("I5", "И5"),
        ("N2", "Н2"),
        ("Н8 Прослеживаемость", "Н8"),
        ("однозначность", "И3"),
        ("verifiable", "И7"),
        ("singular", "И5"),
        ("traceability", "Н8"),
        ("неизвестно", None),
        ("", None),
    ],
)
def test_normalize_criterion(raw, code):
    assert normalize_criterion(raw) == code


@pytest.mark.parametrize(
    ("raw", "sev"),
    [
        ("critical", "critical"),
        ("Критический", "critical"),
        ("major", "major"),
        ("существенный", "major"),
        ("minor", "minor"),
        ("незначительный", "minor"),
        (None, "minor"),
    ],
)
def test_normalize_severity(raw, sev):
    assert normalize_severity(raw) == sev


def test_criteria_catalog_is_complete():
    codes = [c.code for c in ALL]
    assert codes == [f"И{i}" for i in range(1, 10)] + [f"Н{i}" for i in range(1, 9)]
    assert all(c.source and c.question for c in ALL)


def test_agent_version_reports_commit():
    v = agent_version(ignore_paths=[Path("reports/report.md")])
    assert v.version
    assert "коммит" in v.label


def test_user_key_names_and_new_providers(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "c")
    s = load_settings()
    assert (s.provider, s.api_key, s.model) == ("anthropic", "c", "claude-opus-5-5")
    monkeypatch.delenv("CLAUDE_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    s = load_settings()
    assert (s.provider, s.base_url, s.effort) == ("openrouter", "https://openrouter.ai/api/v1", "high")
    s = load_settings(provider="deepseek")
    assert (s.model, s.base_url, s.api_key) == ("deepseek-v4-pro", "https://api.deepseek.com", "d")


def test_provider_priority_claude_then_openrouter_then_gemini(monkeypatch):
    for name in ("GEMINI_API_KEY", "CLAUDE_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(name, "k")
    s = load_settings()
    assert (s.provider, s.model, s.effort) == ("anthropic", "claude-opus-5-5", "high")
    monkeypatch.delenv("CLAUDE_API_KEY")
    assert load_settings().provider == "openrouter"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    s = load_settings()
    assert s.provider == "gemini"
    assert s.fallback_models == ("gemini-3.5-flash", "gemini-flash-lite-latest")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", " a , b ,")
    assert load_settings().fallback_models == ("a", "b")
