"""Claude Code CLI в неинтерактивном режиме (``claude -p``) как провайдер LLM.

Подходит, если на машине установлен Claude Code и выполнен вход: отдельный
API-ключ не нужен. Инструменты, MCP-серверы, плагины и CLAUDE.md отключаются —
модель получает только системный промпт и текст запроса.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile

from reqvalidator.llm.base import (
    LLMClient,
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMTruncatedError,
)

_RETRYABLE_MARKERS = ("rate limit", "rate_limit", "overloaded", "529", "503", "timeout", "temporarily")


class ClaudeCLIClient(LLMClient):
    provider = "claude-cli"

    def __init__(self, settings):
        super().__init__(settings)
        self.cli = shutil.which(settings.cli_path) or settings.cli_path
        self._workdir = tempfile.mkdtemp(prefix="reqvalidator-cli-")

    def _command(self, system: str) -> list[str]:
        cmd = [
            self.cli,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--safe-mode",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--system-prompt",
            system,
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.settings.effort:
            cmd += ["--effort", self.settings.effort]
        return cmd

    def complete(self, system: str, prompt: str, *, json_mode: bool = True, stage: str = "") -> LLMResponse:
        try:
            proc = subprocess.run(
                self._command(system),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout,
                cwd=self._workdir,
                check=False,
            )
        except FileNotFoundError as exc:
            raise LLMFatalError(
                f"Claude Code CLI не найден ({self.cli}). Установите Claude Code или задайте CLAUDE_CLI_PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMRetryableError(f"таймаут Claude Code CLI ({self.settings.timeout:.0f} с)") from exc

        stdout = proc.stdout.strip()
        try:
            data = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            data = {}
        if proc.returncode != 0 or data.get("is_error") or not data:
            detail = (data.get("result") or proc.stderr or stdout or "").strip()[:500]
            low = detail.lower()
            if any(m in low for m in _RETRYABLE_MARKERS) or data.get("api_error_status") in (
                429,
                500,
                503,
                529,
            ):
                raise LLMRetryableError(f"Claude Code CLI: {detail}")
            if "login" in low or "auth" in low or "api key" in low:
                raise LLMFatalError(
                    f"Claude Code CLI не авторизован: {detail}. Выполните `claude` и войдите."
                )
            raise LLMFatalError(f"Claude Code CLI завершился с ошибкой (код {proc.returncode}): {detail}")

        text = data.get("result") or ""
        usage = data.get("usage") or {}
        models = list((data.get("modelUsage") or {}).keys())
        if data.get("stop_reason") == "max_tokens":
            raise LLMTruncatedError("ответ обрезан по лимиту токенов", partial_text=text)
        input_tokens = sum(
            int(usage.get(k) or 0)
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        return LLMResponse(
            text=text,
            model=models[0] if models else self.model,
            input_tokens=input_tokens or None,
            output_tokens=usage.get("output_tokens"),
            finish_reason=data.get("stop_reason"),
            meta={"cost_usd": data.get("total_cost_usd")},
        )
