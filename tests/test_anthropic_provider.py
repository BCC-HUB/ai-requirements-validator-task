"""Провайдер Anthropic через настоящий SDK против локальной заглушки Messages API (SSE)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reqvalidator.llm import LLMAbortError, LLMTruncatedError

anthropic = pytest.importorskip("anthropic")
from reqvalidator.llm.anthropic_api import AnthropicClient  # noqa: E402
from tests.fakes import fake_settings  # noqa: E402


def _sse(text: str, stop_reason: str = "end_turn") -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


class _Server:
    """Отвечает по очереди заданными ответами и запоминает запросы."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, payload = server.responses.pop(0)
                self.send_response(status)
                if status == 200:
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    data = json.dumps(payload).encode()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server(request):
    srv = _Server(request.param)
    yield srv
    srv.close()


def _client(srv):
    return AnthropicClient(
        fake_settings(
            provider="anthropic",
            model="claude-opus-5-5",
            base_url=srv.url,
            effort="high",
            max_output_tokens=32000,
        )
    )


@pytest.mark.parametrize("server", [[(200, _sse('{"value": 1}'))]], indirect=True)
def test_request_shape_and_streamed_response(server):
    resp = _client(server).complete("SYS", "PROMPT")
    assert resp.text == '{"value": 1}' and resp.model == "claude-opus-5-5"
    assert resp.input_tokens == 11 and resp.output_tokens == 7
    req = server.requests[0]
    assert req["path"].startswith("/v1/messages")
    body = req["body"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 32000 and body["stream"] is True
    assert body["system"] == "SYS" and body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body  # текущие модели Claude отклоняют temperature
    assert "server-side-fallback" in req["headers"].get("anthropic-beta", "")
    # Явно переданный ключ имеет приоритет над ANTHROPIC_AUTH_TOKEN/ANTHROPIC_API_KEY из окружения.
    assert req["headers"].get("Authorization") == "Bearer test"


@pytest.mark.parametrize(
    "server",
    [
        [
            (
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "fallbacks: not supported"},
                },
            ),
            (200, _sse("{}")),
        ]
    ],
    indirect=True,
)
def test_retries_without_fallback_when_rejected(server):
    client = _client(server)
    assert client.complete("s", "p").text == "{}"
    assert "fallbacks" not in server.requests[1]["body"]


@pytest.mark.parametrize(
    "server",
    [[(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})]],
    indirect=True,
)
def test_bad_key_aborts_run(server):
    with pytest.raises(LLMAbortError, match="ключ"):
        _client(server).complete("s", "p")


@pytest.mark.parametrize("server", [[(200, _sse('{"val', stop_reason="max_tokens"))]], indirect=True)
def test_truncated_answer(server):
    with pytest.raises(LLMTruncatedError):
        _client(server).complete("s", "p")
