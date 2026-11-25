"""Провайдеры LLM и фабрика клиента по настройкам."""

from __future__ import annotations

from reqvalidator.config import Settings
from reqvalidator.llm.base import (
    LLMClient,
    LLMError,
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMTruncatedError,
    extract_json,
)
from reqvalidator.llm.session import CallRecord, LLMSession, ResponseCache

__all__ = [
    "CallRecord",
    "LLMClient",
    "LLMError",
    "LLMFatalError",
    "LLMResponse",
    "LLMRetryableError",
    "LLMSession",
    "LLMTruncatedError",
    "ResponseCache",
    "create_client",
    "extract_json",
]


def create_client(settings: Settings) -> LLMClient:
    if settings.provider == "gemini":
        from reqvalidator.llm.gemini import GeminiClient

        return GeminiClient(settings)
    if settings.provider == "openai":
        from reqvalidator.llm.openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient(settings)
    if settings.provider == "ollama":
        from reqvalidator.llm.openai_compat import OllamaClient

        return OllamaClient(settings)
    if settings.provider == "anthropic":
        from reqvalidator.llm.anthropic_api import AnthropicClient

        return AnthropicClient(settings)
    if settings.provider == "claude-cli":
        from reqvalidator.llm.claude_cli import ClaudeCLIClient

        return ClaudeCLIClient(settings)
    raise LLMFatalError(f"Неизвестный провайдер: {settings.provider}")
