"""OpenAI Chat Completions и совместимые серверы (vLLM, LM Studio, OpenRouter и т. п.), а также Ollama."""

from __future__ import annotations

import requests

from reqvalidator.llm.base import (
    LLMClient,
    LLMError,
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMTruncatedError,
    error_payload,
    parse_retry_after,
    unavailable,
)


class OpenAICompatibleClient(LLMClient):
    """Chat Completions API: OpenAI, OpenRouter, DeepSeek и любые совместимые серверы."""

    provider = "openai"

    def __init__(self, settings, http: requests.Session | None = None):
        super().__init__(settings)
        self.provider = settings.provider
        self.base_url = settings.base_url or "https://api.openai.com/v1"
        self.http = http or requests.Session()
        # Параметры, которые конкретный сервер отверг; повторный запрос идёт без них.
        self._drop: set[str] = set()
        self._token_param = "max_tokens"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _body(self, model: str | None, system: str, prompt: str, json_mode: bool) -> dict:
        body: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            self._token_param: self.settings.max_output_tokens,
        }
        if json_mode and "response_format" not in self._drop:
            body["response_format"] = {"type": "json_object"}
        if self.settings.temperature is not None and "temperature" not in self._drop:
            body["temperature"] = self.settings.temperature
        # OpenRouter принимает единый параметр глубины рассуждений для моделей с reasoning.
        if self.provider == "openrouter" and self.settings.effort and "reasoning" not in self._drop:
            body["reasoning"] = {"effort": self.settings.effort}
        return body

    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        return self.call_with_fallback(lambda model: self._complete_model(model, system, prompt, json_mode))

    def _complete_model(self, model: str | None, system: str, prompt: str, json_mode: bool) -> LLMResponse:
        for _ in range(4):  # несколько попыток адаптации к параметрам сервера
            try:
                r = self.http.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._body(model, system, prompt, json_mode),
                    timeout=self.settings.timeout,
                )
            except requests.Timeout as exc:
                raise LLMRetryableError(f"таймаут запроса ({self.settings.timeout:.0f} с)") from exc
            except requests.RequestException as exc:
                raise LLMRetryableError(f"сетевая ошибка: {exc}") from exc
            if r.status_code == 400 and self._adapt(r.text):
                continue
            break
        if r.status_code != 200:
            raise self._error(r, model)
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        finish = choice.get("finish_reason")
        if finish == "length":
            raise LLMTruncatedError("ответ обрезан по лимиту токенов", partial_text=text)
        if not text.strip():
            raise LLMRetryableError(f"пустой ответ модели (finish_reason={finish})")
        return LLMResponse(
            text=text,
            model=data.get("model") or model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=finish,
        )

    def _adapt(self, error_text: str) -> bool:
        low = error_text.lower()
        if "max_completion_tokens" in low and self._token_param == "max_tokens":
            self._token_param = "max_completion_tokens"
            return True
        if "response_format" in low and "response_format" not in self._drop:
            self._drop.add("response_format")
            return True
        if "temperature" in low and "temperature" not in self._drop and self.settings.temperature is not None:
            self._drop.add("temperature")
            return True
        if "reasoning" in low and "reasoning" not in self._drop and self.provider == "openrouter":
            self._drop.add("reasoning")
            return True
        return False

    def _error(self, r: requests.Response, model: str | None = None) -> LLMError:
        message = error_payload(r).get("message") or r.text[:300]
        if r.status_code == 503:
            return unavailable(LLMRetryableError(f"модель {model} перегружена (HTTP 503): {message}"))
        if r.status_code == 429 or r.status_code >= 500:
            return LLMRetryableError(
                f"HTTP {r.status_code}: {message}",
                retry_after=parse_retry_after(r.headers.get("retry-after")),
            )
        if r.status_code == 402:
            return LLMFatalError(f"на счёте провайдера недостаточно средств (HTTP 402): {message}")
        if r.status_code in (401, 403):
            return LLMFatalError(f"ключ отклонён (HTTP {r.status_code}): {message}")
        if r.status_code == 404:
            return unavailable(LLMFatalError(f"модель {model!r} или адрес не найдены (HTTP 404): {message}"))
        return LLMFatalError(f"HTTP {r.status_code}: {message}")

    def list_models(self) -> list[str]:
        r = self.http.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.settings.timeout)
        if r.status_code != 200:
            raise self._error(r)
        return sorted(m.get("id", "") for m in r.json().get("data", []))


class OllamaClient(LLMClient):
    """Нативный API Ollama: позволяет задать размер контекста (num_ctx) — у совместимого API его нет."""

    provider = "ollama"

    def __init__(self, settings, http: requests.Session | None = None):
        super().__init__(settings)
        self.base_url = (settings.base_url or "http://localhost:11434").removesuffix("/v1")
        self.http = http or requests.Session()

    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        options: dict = {"num_ctx": self.settings.num_ctx, "num_predict": self.settings.max_output_tokens}
        if self.settings.temperature is not None:
            options["temperature"] = self.settings.temperature
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        try:
            r = self.http.post(f"{self.base_url}/api/chat", json=body, timeout=self.settings.timeout)
        except requests.ConnectionError as exc:
            raise LLMFatalError(
                f"Ollama недоступна по адресу {self.base_url}. Запустите `ollama serve` или проверьте OLLAMA_BASE_URL."
            ) from exc
        except requests.Timeout as exc:
            raise LLMRetryableError(f"таймаут Ollama ({self.settings.timeout:.0f} с)") from exc
        if r.status_code == 404:
            raise LLMFatalError(
                f"модель {self.model!r} не найдена в Ollama. Выполните: ollama pull {self.model}"
            )
        if r.status_code >= 500:
            raise LLMRetryableError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code != 200:
            raise LLMFatalError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = (data.get("message") or {}).get("content") or ""
        if data.get("done_reason") == "length":
            raise LLMTruncatedError("ответ Ollama обрезан по num_predict", partial_text=text)
        return LLMResponse(
            text=text,
            model=data.get("model") or self.model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
        )

    def list_models(self) -> list[str]:
        try:
            r = self.http.get(f"{self.base_url}/api/tags", timeout=30)
        except requests.ConnectionError as exc:
            raise LLMFatalError(f"Ollama недоступна по адресу {self.base_url}") from exc
        return sorted(m.get("name", "") for m in r.json().get("models", []))
