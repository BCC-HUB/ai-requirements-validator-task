"""Сессия обращений к модели: кэш, ограничение частоты, повторы, проверка JSON-схемы, журнал вызовов."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from reqvalidator.llm.base import (
    JSONExtractionError,
    LLMClient,
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMTruncatedError,
    extract_json,
)

log = logging.getLogger("reqvalidator")
# Ошибки, которые пробрасываются наружу без попытки «починить» JSON.
_PASSTHROUGH = (LLMFatalError, LLMTruncatedError)
T = TypeVar("T", bound=BaseModel)

_REPAIR_TEMPLATE = """Твой предыдущий ответ не удалось принять: {error}

Предыдущий ответ (фрагмент):
<<<
{previous}
>>>

Верни ИСПРАВЛЕННЫЙ ответ целиком — только один JSON-объект по заданной схеме, без пояснений и без Markdown."""


@dataclass
class CallRecord:
    stage: str
    label: str
    started_at: str
    duration_s: float
    attempts: int
    cached: bool
    ok: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    error: str | None = None


class ResponseCache:
    """Кэш ответов модели на диске: повторный прогон с теми же входами не тратит лимиты."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(*parts: object) -> str:
        blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return LLMResponse(
            text=data.get("text", ""),
            model=data.get("model"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            finish_reason=data.get("finish_reason"),
        )

    def put(self, key: str, resp: LLMResponse) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "text": resp.text,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "finish_reason": resp.finish_reason,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # кэш — оптимизация, его сбой не должен ронять прогон
            log.debug("Не удалось записать кэш: %s", exc)


class LLMSession:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_retries: int = 5,
        min_interval: float = 0.0,
        cache: ResponseCache | None = None,
        repair_attempts: int = 2,
        sleep=time.sleep,
    ):
        self.client = client
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.cache = cache
        self.repair_attempts = repair_attempts
        self.records: list[CallRecord] = []
        self.models_seen: list[str] = []
        self._sleep = sleep
        self._lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_request = 0.0

    # ------------------------------------------------------------------ public
    def ask_json(self, stage: str, system: str, prompt: str, schema: type[T], label: str = "") -> T:
        """Запросить у модели JSON и провалидировать его схемой ``schema``.

        При невалидном ответе модель получает текст ошибки и переделывает ответ
        (до ``repair_attempts`` раз). Валидный ответ кэшируется.
        """
        settings = self.client.settings
        cache_key = ResponseCache.key(
            "v1",
            self.client.provider,
            self.client.model,
            settings.temperature,
            settings.effort,
            system,
            prompt,
        )
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                try:
                    result = schema.model_validate(extract_json(cached.text))
                except (JSONExtractionError, ValidationError):
                    result = None
                if result is not None:
                    self._record(
                        CallRecord(
                            stage,
                            label,
                            started_at,
                            0.0,
                            0,
                            True,
                            True,
                            cached.input_tokens,
                            cached.output_tokens,
                            cached.model,
                        )
                    )
                    return result

        attempts = 0
        current_prompt = prompt
        last_error: str | None = None
        resp: LLMResponse | None = None
        for repair in range(self.repair_attempts + 1):
            try:
                resp, n = self._complete_with_retries(system, current_prompt, stage)
            except _PASSTHROUGH as exc:
                self._record(
                    CallRecord(
                        stage,
                        label,
                        started_at,
                        time.monotonic() - started,
                        attempts + 1,
                        False,
                        False,
                        error=str(exc),
                    )
                )
                raise
            attempts += n
            try:
                data = extract_json(resp.text)
                result = schema.model_validate(data)
            except (JSONExtractionError, ValidationError) as exc:
                last_error = _short_error(exc)
                log.warning(
                    "Ответ модели на этапе «%s» не прошёл проверку (%s), попытка исправления %d/%d",
                    stage,
                    last_error,
                    repair + 1,
                    self.repair_attempts,
                )
                current_prompt = (
                    prompt
                    + "\n\n"
                    + _REPAIR_TEMPLATE.format(error=last_error, previous=(resp.text or "")[:6000])
                )
                continue
            if self.cache is not None:
                self.cache.put(cache_key, resp)
            self._record(
                CallRecord(
                    stage,
                    label,
                    started_at,
                    time.monotonic() - started,
                    attempts,
                    False,
                    True,
                    resp.input_tokens,
                    resp.output_tokens,
                    resp.model,
                )
            )
            return result

        self._record(
            CallRecord(
                stage,
                label,
                started_at,
                time.monotonic() - started,
                attempts,
                False,
                False,
                resp.input_tokens if resp else None,
                resp.output_tokens if resp else None,
                resp.model if resp else None,
                error=last_error,
            )
        )
        raise LLMFatalError(f"Модель вернула некорректный ответ на этапе «{stage}»: {last_error}")

    # ------------------------------------------------------------------ stats
    @property
    def total_calls(self) -> int:
        return sum(1 for r in self.records if not r.cached)

    @property
    def cached_calls(self) -> int:
        return sum(1 for r in self.records if r.cached)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens or 0 for r in self.records if not r.cached)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens or 0 for r in self.records if not r.cached)

    # ------------------------------------------------------------------ internals
    def _record(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)
            if rec.model and rec.model not in self.models_seen:
                self.models_seen.append(rec.model)

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._rate_lock:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()

    def _complete_with_retries(self, system: str, prompt: str, stage: str) -> tuple[LLMResponse, int]:
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            try:
                return self.client.complete(system, prompt, json_mode=True, stage=stage), attempt
            except LLMRetryableError as exc:
                if attempt > self.max_retries:
                    raise LLMFatalError(f"Модель недоступна после {attempt} попыток: {exc}") from exc
                delay = (
                    exc.retry_after if exc.retry_after is not None else min(60.0, 2.0 * 2 ** (attempt - 1))
                )
                delay = min(delay, 120.0) + random.uniform(0, 1.0)
                log.warning(
                    "Временная ошибка модели (%s). Повтор %d/%d через %.0f с",
                    exc,
                    attempt,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:6]:
            loc = ".".join(str(x) for x in err.get("loc", ()))
            parts.append(f"{loc}: {err.get('msg')}")
        more = "" if len(exc.errors()) <= 6 else f" (и ещё {len(exc.errors()) - 6})"
        return "несоответствие схеме — " + "; ".join(parts) + more
    return str(exc)[:500]
