"""Настройки агента: провайдер LLM, модель, лимиты. Источники — .env, окружение, флаги CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROVIDERS = ("gemini", "anthropic", "openai", "ollama", "claude-cli")

# Значения-заглушки из .env.example, которые нельзя принимать за настоящий ключ.
_PLACEHOLDER_MARKERS = ("your_", "_here", "changeme", "<", "xxx")


class ConfigError(RuntimeError):
    """Ошибка конфигурации, понятная пользователю."""


@dataclass(frozen=True)
class ProviderDefaults:
    model: str | None
    max_output_tokens: int
    timeout: float
    concurrency: int
    min_interval: float
    effort: str | None = None
    base_url: str | None = None


DEFAULTS: dict[str, ProviderDefaults] = {
    # Бесплатный тариф Google AI Studio: несколько запросов в минуту, поэтому
    # запросы идут последовательно и с паузой. Алиас *-latest указывает на
    # актуальную модель Flash и не устаревает при выходе новых версий.
    "gemini": ProviderDefaults(
        model="gemini-flash-latest",
        max_output_tokens=32768,
        timeout=300,
        concurrency=1,
        min_interval=4.0,
        base_url="https://generativelanguage.googleapis.com/v1beta",
    ),
    "anthropic": ProviderDefaults(
        model="claude-opus-5-5", max_output_tokens=32000, timeout=600, concurrency=4, min_interval=0.0,
        effort="high",
    ),
    "openai": ProviderDefaults(
        model="gpt-4o-mini", max_output_tokens=16000, timeout=300, concurrency=4, min_interval=0.0,
        base_url="https://api.openai.com/v1",
    ),
    "ollama": ProviderDefaults(
        model="qwen2.5-coder:7b", max_output_tokens=8192, timeout=1200, concurrency=1, min_interval=0.0,
        base_url="http://localhost:11434",
    ),
    # Claude Code CLI в неинтерактивном режиме (claude -p): использует уже
    # выполненный вход в Claude Code, отдельный API-ключ не нужен.
    "claude-cli": ProviderDefaults(
        model=None, max_output_tokens=32000, timeout=1200, concurrency=4, min_interval=0.0, effort="high",
    ),
}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str | None
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    temperature: float | None = None
    max_output_tokens: int = 16000
    timeout: float = 300.0
    max_retries: int = 5
    min_interval: float = 0.0
    concurrency: int = 1
    batch_size: int = 6
    review: bool = True
    cache_dir: Path | None = Path(".reqvalidator_cache")
    effort: str | None = None
    cli_path: str = "claude"
    num_ctx: int = 32768

    @property
    def model_label(self) -> str:
        return self.model or "модель по умолчанию Claude Code"

    def with_overrides(self, **kwargs) -> Settings:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _real_key(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return None
    return value


def _float(name: str, default: float | None) -> float | None:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть числом, получено: {raw!r}") from exc


def _int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from exc


def load_env_file(env_file: str | Path | None = None) -> Path | None:
    """Загрузить .env (не перетирая уже заданные переменные окружения)."""
    path = Path(env_file) if env_file else Path.cwd() / ".env"
    if path.is_file():
        load_dotenv(path, override=False)
        return path
    if env_file:
        raise ConfigError(f"Файл настроек не найден: {path}")
    return None


def detect_provider() -> str:
    explicit = _env("LLM_PROVIDER")
    if explicit:
        explicit = explicit.lower()
        if explicit not in PROVIDERS:
            raise ConfigError(f"Неизвестный LLM_PROVIDER={explicit!r}. Допустимо: {', '.join(PROVIDERS)}")
        return explicit
    if _real_key(_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")):
        return "gemini"
    if _real_key(_env("ANTHROPIC_API_KEY")):
        return "anthropic"
    if _real_key(_env("OPENAI_API_KEY")):
        return "openai"
    if _env("OLLAMA_MODEL"):
        return "ollama"
    raise ConfigError(
        "Не настроен доступ к LLM. Скопируйте .env.example в .env и укажите GEMINI_API_KEY "
        "(бесплатный ключ: https://aistudio.google.com/apikey) или задайте LLM_PROVIDER и ключ "
        "другого провайдера. Подробности — в README, раздел «Ключ LLM»."
    )


def load_settings(
    env_file: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Settings:
    """Собрать настройки из .env и окружения; ``provider``/``model`` — переопределения из CLI."""
    load_env_file(env_file)
    provider = (provider or detect_provider()).lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"Неизвестный провайдер {provider!r}. Допустимо: {', '.join(PROVIDERS)}")
    d = DEFAULTS[provider]

    api_key: str | None = None
    base_url = d.base_url
    effort = d.effort
    if provider == "gemini":
        api_key = _real_key(_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
        model = model or _env("GEMINI_MODEL", d.model)
        base_url = _env("GEMINI_BASE_URL", base_url)
        if not api_key:
            raise ConfigError(
                "Не задан GEMINI_API_KEY. Получите бесплатный ключ на https://aistudio.google.com/apikey "
                "и запишите его в файл .env (см. .env.example)."
            )
    elif provider == "anthropic":
        api_key = _real_key(_env("ANTHROPIC_API_KEY"))
        model = model or _env("ANTHROPIC_MODEL", d.model)
        effort = _env("ANTHROPIC_EFFORT", effort)
        if not api_key:
            raise ConfigError("Не задан ANTHROPIC_API_KEY (ключ: https://platform.claude.com/settings/keys).")
    elif provider == "openai":
        api_key = _real_key(_env("OPENAI_API_KEY"))
        model = model or _env("OPENAI_MODEL", d.model)
        base_url = _env("OPENAI_BASE_URL", base_url)
        if not api_key and base_url == DEFAULTS["openai"].base_url:
            raise ConfigError("Не задан OPENAI_API_KEY.")
    elif provider == "ollama":
        model = model or _env("OLLAMA_MODEL", d.model)
        base_url = _env("OLLAMA_BASE_URL", base_url)
    elif provider == "claude-cli":
        model = model or _env("CLAUDE_CLI_MODEL", d.model)
        effort = _env("CLAUDE_CLI_EFFORT", effort)

    cache_raw = os.getenv("REQVALIDATOR_CACHE_DIR")
    if cache_raw is None:
        cache_dir: Path | None = Path(".reqvalidator_cache")
    else:
        cache_dir = Path(cache_raw) if cache_raw.strip() else None

    return Settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else None,
        temperature=_float("LLM_TEMPERATURE", None),
        max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", d.max_output_tokens),
        timeout=_float("LLM_TIMEOUT", d.timeout) or d.timeout,
        max_retries=_int("LLM_MAX_RETRIES", 5),
        min_interval=_float("LLM_MIN_INTERVAL", d.min_interval) or 0.0,
        concurrency=max(1, _int("LLM_CONCURRENCY", d.concurrency)),
        batch_size=max(1, _int("REQVALIDATOR_BATCH_SIZE", 6)),
        review=(_env("REQVALIDATOR_REVIEW", "1") or "1") not in ("0", "false", "no", "off"),
        cache_dir=cache_dir,
        effort=effort,
        cli_path=_env("CLAUDE_CLI_PATH", "claude") or "claude",
        num_ctx=_int("OLLAMA_NUM_CTX", 32768),
    )
