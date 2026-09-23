"""Pins: a Stop reaches a REMOTE AbstractCore server (wire form = the severed request).

Mission H left `RemoteAbstractCoreLLMClient` blind to the effect's cancel
event: its strict body allowlist dropped it, and the POST sat in a blocking
read until the server finished the whole generation. Wire form chosen
2026-09-23: the client SEVERS its request (socket shutdown) the moment the
event is set; the AbstractCore server's client-disconnect watcher cancels the
generation and its provider severs the upstream model request in turn
(measured live: runtime -> core :18808 -> LM Studio, client returned in ~0 ms,
LM Studio worker CPU 49.5 % -> 0.0 %). No new endpoint, no request id.
"""

from __future__ import annotations

import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from abstractruntime.integrations.abstractcore.llm_client import (
    HttpxRequestSender,
    RemoteAbstractCoreLLMClient,
    RemoteGenerationCancelled,
)

HOLD_S = 6.0


def _holding_server():
    seen: Dict[str, Any] = {"disconnects": [], "posts": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            seen["posts"] += 1
            deadline = time.monotonic() + HOLD_S
            while time.monotonic() < deadline:
                r, _, _ = select.select([self.connection], [], [], 0.02)
                if r:
                    try:
                        data = self.connection.recv(1, socket.MSG_PEEK)
                    except OSError:
                        data = b""
                    if not data:
                        seen["disconnects"].append(time.monotonic())
                        return
            body = b'{"choices": [{"message": {"content": "late"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_sender_severs_a_blocked_post_and_the_server_sees_the_disconnect():
    server, seen = _holding_server()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    ev = threading.Event()
    out: Dict[str, Any] = {}

    def call():
        try:
            HttpxRequestSender().post(url, headers={}, json={"x": 1}, timeout=30.0, cancel_event=ev)
            out["result"] = "completed"
        except BaseException as e:  # noqa: BLE001
            out["error"] = e
        out["returned"] = time.monotonic()

    th = threading.Thread(target=call, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while seen["posts"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    t_cancel = time.monotonic()
    ev.set()
    th.join(HOLD_S + 2)
    try:
        assert out.get("returned", 1e9) - t_cancel < 1.0, out
        assert isinstance(out.get("error"), RemoteGenerationCancelled), out
        deadline = time.monotonic() + 1.0
        while not seen["disconnects"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen["disconnects"], "the server never saw the client go away"
    finally:
        server.shutdown()
        server.server_close()


class _RecordingSender:
    supports_cancel_event = True

    def __init__(self) -> None:
        self.kwargs: List[Dict[str, Any]] = []

    def post(self, url, *, headers, json, timeout, cancel_event=None):
        self.kwargs.append({"url": url, "cancel_event": cancel_event, "json": json})
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    def get(self, url, *, headers, timeout):
        return {}


class _LegacySender(_RecordingSender):
    supports_cancel_event = False

    def post(self, url, *, headers, json, timeout):  # no cancel_event kwarg at all
        self.kwargs.append({"url": url, "json": json})
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


def test_remote_client_hands_the_effect_event_to_the_sender_never_to_the_body():
    sender = _RecordingSender()
    client = RemoteAbstractCoreLLMClient(server_base_url="http://localhost:9", model="lmstudio/m", request_sender=sender)
    ev = threading.Event()
    client.generate(prompt="hi", params={"cancel_event": ev, "max_tokens": 4})
    chat = [k for k in sender.kwargs if k["url"].endswith("/chat/completions")]
    assert chat and chat[-1]["cancel_event"] is ev
    assert "cancel_event" not in chat[-1]["json"] and "cancel_event" not in (chat[-1]["json"].get("params") or {})


def test_remote_client_never_passes_the_event_to_a_sender_that_cannot_take_it():
    sender = _LegacySender()
    client = RemoteAbstractCoreLLMClient(server_base_url="http://localhost:9", model="lmstudio/m", request_sender=sender)
    client.generate(prompt="hi", params={"cancel_event": threading.Event(), "max_tokens": 4})
    assert [k for k in sender.kwargs if k["url"].endswith("/chat/completions")]
