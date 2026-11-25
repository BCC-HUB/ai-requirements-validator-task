"""Anthropic Claude API через официальный SDK — основной провайдер агента (Claude Opus 5.5)."""

from __future__ import annotations

from reqvalidator.llm.base import (
    LLMAbortError,
    LLMClient,
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMTruncatedError,
)

# Серверный резерв при отказе модели (stop_reason="refusal"): API само повторит
# запрос на подходящей модели, сохраняя один вызов.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, settings):
        super().__init__(settings)
        try:
            import anthropic
        except ImportError as exc:
            raise LLMAbortError("Не установлен пакет anthropic: pip install -r requirements.txt") from exc
        self._sdk = anthropic
        # SDK делает 2 быстрых повтора (с учётом Retry-After); длинные паузы — на стороне сессии агента.
        options: dict = {"api_key": settings.api_key, "timeout": settings.timeout, "max_retries": 2}
        options["base_url"] = settings.base_url or "https://api.anthropic.com"
        self._client = anthropic.Anthropic(**options)
        self._use_fallbacks = True

    def _kwargs(self, system: str, prompt: str) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.settings.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._use_fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        if self.settings.effort:
            kwargs["output_config"] = {"effort": self.settings.effort}
        return kwargs

    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        sdk = self._sdk
        try:
            try:
                message = self._stream(system, prompt)
            except sdk.BadRequestError as exc:
                # Если API или выбранная модель не поддерживает серверный резерв — повторяем без него.
                text = str(exc.message).lower()
                if self._use_fallbacks and ("fallback" in text or "beta" in text):
                    self._use_fallbacks = False
                    message = self._stream(system, prompt)
                else:
                    raise
        except sdk.NotFoundError as exc:
            raise LLMAbortError(f"модель {self.model!r} не найдена: {exc.message}") from exc
        except sdk.AuthenticationError as exc:
            raise LLMAbortError(
                "Anthropic отклонил ключ. Проверьте ANTHROPIC_API_KEY (или CLAUDE_API_KEY)."
            ) from exc
        except sdk.PermissionDeniedError as exc:
            raise LLMAbortError(f"нет доступа к модели: {exc.message}") from exc
        except sdk.BadRequestError as exc:
            if "credit balance" in str(exc.message).lower():
                raise LLMAbortError(f"на счёте Anthropic недостаточно средств: {exc.message}") from exc
            raise LLMFatalError(f"запрос отклонён Anthropic API: {exc.message}") from exc
        except sdk.RateLimitError as exc:
            raise LLMRetryableError(f"лимит запросов Anthropic: {exc.message}") from exc
        except sdk.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMRetryableError(f"Anthropic API недоступен ({exc.status_code})") from exc
            raise LLMFatalError(f"ошибка Anthropic API {exc.status_code}: {exc.message}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMRetryableError(f"сетевая ошибка Anthropic API: {exc}") from exc

        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        if message.stop_reason == "refusal":
            raise LLMFatalError("модель отказалась отвечать (stop_reason=refusal)")
        if message.stop_reason == "max_tokens":
            raise LLMTruncatedError("ответ обрезан по max_tokens", partial_text=text)
        usage = message.usage
        return LLMResponse(
            text=text,
            model=message.model,
            input_tokens=(usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
            output_tokens=usage.output_tokens,
            finish_reason=message.stop_reason,
        )

    def _stream(self, system: str, prompt: str):
        # Потоковый режим: длинные ответы не упираются в таймаут HTTP.
        with self._client.beta.messages.stream(**self._kwargs(system, prompt)) as stream:
            return stream.get_final_message()

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list())
