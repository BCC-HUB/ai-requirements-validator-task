import json
import subprocess

import pytest
from pydantic import BaseModel

from reqvalidator.llm import (
    LLMFatalError,
    LLMResponse,
    LLMRetryableError,
    LLMSession,
    LLMTruncatedError,
    ResponseCache,
    extract_json,
)
from reqvalidator.llm.base import LLMClient, parse_retry_after
from reqvalidator.llm.claude_cli import ClaudeCLIClient
from reqvalidator.llm.gemini import GeminiClient
from reqvalidator.llm.openai_compat import OllamaClient, OpenAICompatibleClient
from tests.fakes import fake_settings


class Out(BaseModel):
    value: int


class Scripted(LLMClient):
    provider = "scripted"

    def __init__(self, replies):
        super().__init__(fake_settings())
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, prompt, *, json_mode=True, stage=""):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, model="m", input_tokens=1, output_tokens=1)


# ---------------------------------------------------------------------------- JSON
@pytest.mark.parametrize(
    "text",
    [
        '{"value": 1}',
        'Вот ответ:\n```json\n{"value": 1}\n```',
        'prefix {"value": 1, } suffix',
        '\n\n  {"value": 1}  ',
    ],
)
def test_extract_json_variants(text):
    assert extract_json(text) == {"value": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("совсем не JSON")


def test_parse_retry_after():
    assert parse_retry_after("37s") == 37.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None


# ---------------------------------------------------------------------------- сессия
def test_session_repairs_invalid_json_and_records_calls():
    client = Scripted(["не JSON", '{"value": "abc"}', '{"value": 7}'])
    sess = LLMSession(client, sleep=lambda s: None)
    assert sess.ask_json("st", "sys", "prompt", Out).value == 7
    assert "не удалось принять" in client.prompts[1]
    assert sess.records[-1].ok and sess.records[-1].attempts == 3


def test_session_gives_up_after_repair_attempts():
    client = Scripted(["x", "y", "z"])
    sess = LLMSession(client, repair_attempts=2, sleep=lambda s: None)
    with pytest.raises(LLMFatalError, match="некорректный ответ"):
        sess.ask_json("st", "sys", "prompt", Out)
    assert not sess.records[-1].ok


def test_session_retries_transient_errors_with_backoff():
    sleeps = []
    client = Scripted([LLMRetryableError("429", retry_after=3), LLMRetryableError("503"), '{"value": 1}'])
    sess = LLMSession(client, max_retries=3, sleep=sleeps.append)
    assert sess.ask_json("st", "s", "p", Out).value == 1
    assert len(sleeps) == 2 and 3 <= sleeps[0] <= 4


def test_session_raises_fatal_when_retries_exhausted():
    client = Scripted([LLMRetryableError("429")] * 3)
    sess = LLMSession(client, max_retries=2, sleep=lambda s: None)
    with pytest.raises(LLMFatalError, match="3 попыток"):
        sess.ask_json("st", "s", "p", Out)


def test_session_passes_truncation_through():
    sess = LLMSession(Scripted([LLMTruncatedError("cut")]), sleep=lambda s: None)
    with pytest.raises(LLMTruncatedError):
        sess.ask_json("st", "s", "p", Out)


def test_cache_returns_stored_answer(tmp_path):
    cache = ResponseCache(tmp_path)
    first = LLMSession(Scripted(['{"value": 5}']), cache=cache)
    assert first.ask_json("st", "s", "p", Out).value == 5
    second_client = Scripted([])  # к модели обращаться нельзя
    second = LLMSession(second_client, cache=cache)
    assert second.ask_json("st", "s", "p", Out).value == 5
    assert second.cached_calls == 1 and second.total_calls == 0


def test_session_throttles_requests():
    sleeps = []
    sess = LLMSession(Scripted(['{"value": 1}', '{"value": 2}']), min_interval=10, sleep=sleeps.append)
    sess.ask_json("a", "s", "p1", Out)
    sess.ask_json("b", "s", "p2", Out)
    assert sleeps and sleeps[0] > 9


# ---------------------------------------------------------------------------- HTTP-заглушка
class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def _next(self, method, url, **kw):
        self.requests.append((method, url, kw))
        status, payload, headers = self.responses.pop(0)
        return FakeResp(status, payload, headers)

    def post(self, url, **kw):
        return self._next("POST", url, **kw)

    def get(self, url, **kw):
        return self._next("GET", url, **kw)


class FakeResp:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _gemini(responses, **kw):
    http = FakeHTTP([(s, p, h) for s, p, h in responses])
    return GeminiClient(fake_settings(base_url="https://g.test/v1beta", **kw), http=http), http


def test_gemini_request_and_response():
    ok = {
        "candidates": [
            {
                "content": {"parts": [{"text": "мысли", "thought": True}, {"text": '{"value": 1}'}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3},
        "modelVersion": "gemini-x-001",
    }
    client, http = _gemini([(200, ok, None)], temperature=0.2)
    resp = client.complete("SYS", "USER")
    assert resp.text == '{"value": 1}' and resp.model == "gemini-x-001"
    assert resp.input_tokens == 10 and resp.output_tokens == 8
    _method, url, kw = http.requests[0]
    assert url == "https://g.test/v1beta/models/fake-model:generateContent"
    assert kw["headers"]["x-goog-api-key"] == "test"
    body = kw["json"]
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["temperature"] == 0.2


def test_gemini_rate_limit_uses_retry_delay():
    err = {
        "error": {
            "code": 429,
            "message": "quota",
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "21s"}],
        }
    }
    client, _ = _gemini([(429, err, None)])
    with pytest.raises(LLMRetryableError) as exc:
        client.complete("s", "u")
    assert exc.value.retry_after == 21


def test_gemini_daily_quota_is_fatal():
    err = {
        "error": {
            "code": 429,
            "message": "quota",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
                }
            ],
        }
    }
    client, _ = _gemini([(429, err, None)])
    with pytest.raises(LLMFatalError, match="дневная квота"):
        client.complete("s", "u")


@pytest.mark.parametrize(
    ("status", "payload", "exc", "match"),
    [
        (404, {"error": {"message": "models/x is not found"}}, LLMFatalError, "list-models"),
        (400, {"error": {"message": "API key not valid"}}, LLMFatalError, "GEMINI_API_KEY"),
        (503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}, LLMRetryableError, "недоступен"),
    ],
)
def test_gemini_error_mapping(status, payload, exc, match):
    client, _ = _gemini([(status, payload, None)])
    with pytest.raises(exc, match=match):
        client.complete("s", "u")


def test_gemini_truncation():
    cut = {"candidates": [{"content": {"parts": [{"text": '{"val'}]}, "finishReason": "MAX_TOKENS"}]}
    client, _ = _gemini([(200, cut, None)])
    with pytest.raises(LLMTruncatedError):
        client.complete("s", "u")


def test_gemini_list_models_filters_generate_content():
    page = {
        "models": [
            {"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
        ]
    }
    client, _ = _gemini([(200, page, None)])
    assert client.list_models() == ["gemini-a"]


def test_openai_adapts_token_parameter_and_parses_response():
    http = FakeHTTP(
        [
            (
                400,
                {"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}},
                None,
            ),
            (
                200,
                {
                    "choices": [{"message": {"content": '{"value": 2}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                    "model": "gpt-x",
                },
                None,
            ),
        ]
    )
    client = OpenAICompatibleClient(fake_settings(provider="openai", base_url="https://o.test/v1"), http=http)
    resp = client.complete("s", "u")
    assert resp.text == '{"value": 2}' and resp.input_tokens == 3
    assert "max_completion_tokens" in http.requests[1][2]["json"]
    assert http.requests[1][2]["headers"]["Authorization"] == "Bearer test"
    assert http.requests[1][2]["json"]["response_format"] == {"type": "json_object"}


def test_openai_length_finish_is_truncation_and_429_is_retryable():
    http = FakeHTTP(
        [
            (200, {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]}, None),
            (429, {"error": {"message": "slow down"}}, {"retry-after": "5"}),
        ]
    )
    client = OpenAICompatibleClient(fake_settings(provider="openai"), http=http)
    with pytest.raises(LLMTruncatedError):
        client.complete("s", "u")
    with pytest.raises(LLMRetryableError) as exc:
        client.complete("s", "u")
    assert exc.value.retry_after == 5


def test_ollama_uses_native_api_with_context_size():
    http = FakeHTTP(
        [
            (
                200,
                {
                    "message": {"content": '{"value": 3}'},
                    "done_reason": "stop",
                    "prompt_eval_count": 9,
                    "eval_count": 2,
                    "model": "qwen",
                },
                None,
            )
        ]
    )
    client = OllamaClient(
        fake_settings(provider="ollama", base_url="http://localhost:11434/v1", num_ctx=16384), http=http
    )
    resp = client.complete("s", "u")
    _, url, kw = http.requests[0]
    assert url == "http://localhost:11434/api/chat"
    assert kw["json"]["options"]["num_ctx"] == 16384 and kw["json"]["format"] == "json"
    assert resp.output_tokens == 2


def test_claude_cli_success_and_errors(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        payload = {
            "result": '{"value": 4}',
            "is_error": False,
            "usage": {"input_tokens": 2, "output_tokens": 3},
            "modelUsage": {"claude-test": {}},
            "stop_reason": "end_turn",
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCLIClient(fake_settings(provider="claude-cli", model="opus", effort="high"))
    resp = client.complete("SYS", "PROMPT")
    cmd, kw = calls[0]
    assert resp.text == '{"value": 4}' and resp.model == "claude-test"
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "opus" and cmd[cmd.index("--effort") + 1] == "high"
    assert kw["input"] == "PROMPT"

    def rate_limited(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 1, json.dumps({"is_error": True, "result": "Rate limit reached"}), ""
        )

    monkeypatch.setattr(subprocess, "run", rate_limited)
    with pytest.raises(LLMRetryableError):
        client.complete("s", "p")

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(LLMFatalError, match="не найден"):
        client.complete("s", "p")
