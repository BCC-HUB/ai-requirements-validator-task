"""Google Gemini API (Google AI Studio) через REST: generateContent с JSON-режимом."""

from __future__ import annotations

import requests

from reqvalidator.llm.base import (
    LLMClient,
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMTruncatedError,
    parse_retry_after,
)


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, settings, http: requests.Session | None = None):
        super().__init__(settings)
        self.base_url = settings.base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.http = http or requests.Session()

    def _headers(self) -> dict[str, str]:
        # Ключ передаётся заголовком, а не в URL, чтобы не попадать в журналы прокси.
        return {"x-goog-api-key": self.settings.api_key or "", "Content-Type": "application/json"}

    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        model = (self.model or "").removeprefix("models/")
        url = f"{self.base_url}/models/{model}:generateContent"
        generation: dict = {"maxOutputTokens": self.settings.max_output_tokens}
        if json_mode:
            generation["responseMimeType"] = "application/json"
        if self.settings.temperature is not None:
            generation["temperature"] = self.settings.temperature
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }
        try:
            r = self.http.post(url, headers=self._headers(), json=body, timeout=self.settings.timeout)
        except requests.Timeout as exc:
            raise LLMRetryableError(f"таймаут запроса к Gemini ({self.settings.timeout:.0f} с)") from exc
        except requests.RequestException as exc:
            raise LLMRetryableError(f"сетевая ошибка Gemini: {exc}") from exc
        if r.status_code != 200:
            raise self._error(r)
        data = r.json()
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMFatalError(f"Gemini отклонил запрос: {feedback.get('blockReason')}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMRetryableError("Gemini вернул пустой ответ")
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = cand.get("finishReason")
        usage = data.get("usageMetadata") or {}
        resp = LLMResponse(
            text=text,
            model=data.get("modelVersion") or model,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=(usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
            or None,
            finish_reason=finish,
        )
        if finish == "MAX_TOKENS":
            raise LLMTruncatedError("ответ Gemini обрезан по лимиту maxOutputTokens", partial_text=text)
        if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
            raise LLMFatalError(f"Gemini прервал ответ: finishReason={finish}")
        if not text.strip():
            raise LLMRetryableError(f"Gemini вернул ответ без текста (finishReason={finish})")
        return resp

    def _error(self, r: requests.Response) -> Exception:
        try:
            err = r.json().get("error", {})
        except ValueError:
            err = {}
        message = err.get("message") or r.text[:300]
        status = err.get("status", "")
        details = err.get("details") or []
        if r.status_code == 429:
            retry = None
            per_day = False
            for d in details:
                if d.get("@type", "").endswith("RetryInfo"):
                    retry = parse_retry_after(d.get("retryDelay"))
                for v in d.get("violations", []) or []:
                    if "perday" in str(v.get("quotaId", "")).lower():
                        per_day = True
            if per_day:
                return LLMFatalError(
                    "исчерпана дневная квота бесплатного тарифа Gemini для этой модели. "
                    "Повторите завтра или укажите другую модель в GEMINI_MODEL."
                )
            return LLMRetryableError(f"лимит запросов Gemini (429): {message}", retry_after=retry)
        if r.status_code >= 500 or status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
            return LLMRetryableError(f"Gemini временно недоступен ({r.status_code}): {message}")
        if r.status_code == 404:
            return LLMFatalError(
                f"модель {self.model!r} не найдена ({message}). "
                "Список доступных моделей: python -m reqvalidator --list-models"
            )
        if r.status_code in (401, 403) or "API key" in message:
            return LLMFatalError(
                f"Gemini отклонил ключ ({r.status_code}): {message}. Проверьте GEMINI_API_KEY."
            )
        return LLMFatalError(f"ошибка Gemini {r.status_code}: {message}")

    def list_models(self) -> list[str]:
        names: list[str] = []
        token = None
        while True:
            params = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            r = self.http.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                params=params,
                timeout=self.settings.timeout,
            )
            if r.status_code != 200:
                raise self._error(r)
            data = r.json()
            for m in data.get("models", []):
                if "generateContent" in (m.get("supportedGenerationMethods") or []):
                    names.append(m.get("name", "").removeprefix("models/"))
            token = data.get("nextPageToken")
            if not token:
                return sorted(names)
