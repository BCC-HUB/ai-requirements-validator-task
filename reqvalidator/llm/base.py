"""Общий интерфейс клиентов LLM, типы ответов, ошибки и разбор JSON из ответа модели."""

from __future__ import annotations

import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
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


log = logging.getLogger("reqvalidator")


class LLMError(RuntimeError):
    """Базовая ошибка обращения к модели.

    ``model_unavailable`` — модель перегружена, снята с обслуживания или исчерпала квоту:
    имеет смысл переключиться на резервную модель (см. ``LLMClient.call_with_fallback``).
    """

    model_unavailable: bool = False


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
        # Кольцо моделей: основная и резервные. Временно перегруженная модель (503) остаётся в кольце,
        # навсегда недоступная (404, исчерпана дневная квота) исключается.
        self._models = [settings.model, *(m for m in settings.fallback_models if m and m != settings.model)]
        self._retired: set[str | None] = set()
        self._switch_lock = threading.Lock()
        self.switches: list[str] = []

    def _after(self, model: str | None) -> list[str | None]:
        if model not in self._models:
            return list(self._models)
        i = self._models.index(model)
        return self._models[i + 1 :] + self._models[:i]

    def call_with_fallback(self, call: Callable[[str | None], LLMResponse]) -> LLMResponse:
        """Вызвать ``call(model)``; если модель недоступна — перейти на следующую модель кольца.

        Если в рамках одного вызова недоступны все модели, ошибка пробрасывается: временную
        перегрузку дальше обрабатывает сессия (повтор с нарастающей паузой).
        """
        tried: set[str | None] = set()
        while True:
            model = self.model
            tried.add(model)
            try:
                return call(model)
            except LLMError as exc:
                if not exc.model_unavailable:
                    raise
                with self._switch_lock:
                    if isinstance(exc, LLMFatalError):
                        self._retired.add(model)
                    if self.model == model:
                        nxt = next(
                            (m for m in self._after(model) if m not in self._retired and m not in tried), None
                        )
                        if nxt is None:
                            raise
                        self.model = nxt
                        note = f"Модель {model} недоступна ({exc}); прогон продолжен на модели {nxt}"
                        log.warning(note)
                        self.switches.append(note)
                    elif self.model in tried:  # другой поток уже переключил на опробованную модель
                        raise

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


def error_payload(response) -> dict:
    """Тело ошибки API как словарь (Google иногда отдаёт список из одного объекта)."""
    try:
        data = response.json()
    except ValueError:
        return {}
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        return {}
    err = data.get("error")
    return err if isinstance(err, dict) else ({"message": str(err)} if err else {})


def unavailable(exc: LLMError) -> LLMError:
    exc.model_unavailable = True
    return exc


def parse_retry_after(value: str | None) -> float | None:
    """Разобрать Retry-After (секунды) или задержку Google вида «37s» / «1.5s»."""
    if not value:
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$", str(value))
    return float(m.group(1)) if m else None
