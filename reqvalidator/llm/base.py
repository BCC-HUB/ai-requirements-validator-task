"""Общий интерфейс клиентов LLM, типы ответов, ошибки и разбор JSON из ответа модели."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from reqvalidator.config import Settings


@dataclass
class LLMResponse:
    text: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Базовая ошибка обращения к модели."""


class LLMFatalError(LLMError):
    """Повтор не поможет: неверный ключ, нет модели, запрос отклонён."""


class LLMRetryableError(LLMError):
    """Временная ошибка: лимит запросов, перегрузка, сеть. ``retry_after`` — пауза в секундах."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTruncatedError(LLMError):
    """Ответ обрезан по лимиту выходных токенов — нужно уменьшить объём запроса."""

    def __init__(self, message: str, partial_text: str = ""):
        super().__init__(message)
        self.partial_text = partial_text


class LLMClient(ABC):
    """Минимальный контракт провайдера: один системный промпт + одно сообщение пользователя."""

    provider: str = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.model

    @abstractmethod
    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        """Выполнить запрос. ``stage`` — имя этапа конвейера (для журналов и тестов)."""

    def list_models(self) -> list[str]:
        raise LLMFatalError(f"Провайдер {self.provider} не поддерживает получение списка моделей")

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model or 'default'}"


# ---------------------------------------------------------------------------- JSON
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class JSONExtractionError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Достать JSON из ответа модели: голый JSON, блок ```json``` или первый сбалансированный объект."""
    if text is None:
        raise JSONExtractionError("пустой ответ")
    candidates: list[str] = [text.strip()]
    candidates += [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    balanced = _first_balanced(text)
    if balanced:
        candidates.append(balanced)
    last_error: Exception | None = None
    for cand in candidates:
        if not cand:
            continue
        for variant in (cand, _TRAILING_COMMA_RE.sub(r"\1", cand)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as exc:
                last_error = exc
    raise JSONExtractionError(f"ответ не является корректным JSON: {last_error}")


def _first_balanced(text: str) -> str | None:
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    stack: list[str] = []
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return text[start : i + 1]
    return None


def parse_retry_after(value: str | None) -> float | None:
    """Разобрать Retry-After (секунды) или задержку Google вида «37s» / «1.5s»."""
    if not value:
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$", str(value))
    return float(m.group(1)) if m else None
