"""Contract tests for the agora hub toolset (HTTP layer).

These tests run a real (local, in-test) HTTP server that mimics the agora hub
API shape, so the tools are exercised end-to-end: URL construction, bearer
auth, query/body encoding, response decoding, and input validation.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest


class _HubStub(BaseHTTPRequestHandler):
    """Minimal agora-hub-shaped endpoint recorder."""

    requests: List[Dict[str, Any]] = []
    responses: Dict[Tuple[str, str], Any] = {}

    def _record_and_reply(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else ""
        path, _, query = self.path.partition("?")
        type(self).requests.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(raw_body) if raw_body else None,
            }
        )
        payload = type(self).responses.get((method, path), {})
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._record_and_reply("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._record_and_reply("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._record_and_reply("PUT")

    def log_message(self, *args: Any) -> None:  # silence test output
        del args


@pytest.fixture()
def hub(monkeypatch: pytest.MonkeyPatch):
    _HubStub.requests = []
    _HubStub.responses = {}
    server = HTTPServer(("127.0.0.1", 0), _HubStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AGORA_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("AGORA_API_KEY", "agora_test_key")
    try:
        yield _HubStub
    finally:
        server.shutdown()
        server.server_close()


def test_check_inbox_sends_bearer_and_parses_envelopes(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import agora_check_inbox

    envelope = {
        "id": "m-42",
        "channel": "assembly",
        "seq": 42,
        "sender": "orchestrator",
        "status": "open",
        "effective_urgency": "next_turn",
        "escalated": False,
        "critical": False,
        "to_me": True,
        "reply_to_me": False,
        "title": "please review",
        "body": "Can the flow agent confirm it sees this?",
    }
    hub.responses[("GET", "/inbox")] = [envelope]

    out = agora_check_inbox()
    assert out == [envelope]
    req = hub.requests[-1]
    assert req["method"] == "GET"
    assert req["path"] == "/inbox"
    assert req["query"] == ""  # wait=0 -> no long-poll param
    assert req["auth"] == "Bearer agora_test_key"


def test_check_inbox_wait_is_clamped(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import (
        MAX_INBOX_WAIT_SECONDS,
        agora_check_inbox,
    )

    hub.responses[("GET", "/inbox")] = []
    agora_check_inbox(wait_seconds=500.0)
    req = hub.requests[-1]
    assert req["query"] == f"wait={MAX_INBOX_WAIT_SECONDS}"


def test_post_message_body_and_validation(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import agora_post_message

    hub.responses[("POST", "/channels/assembly/messages")] = {"id": "m-77", "seq": 77}
    out = agora_post_message(
        channel="assembly",
        body="Reply text",
        status="reply",
        reply_to="m-42",
        to=["orchestrator"],
    )
    assert out == {"id": "m-77", "seq": 77}
    req = hub.requests[-1]
    assert req["path"] == "/channels/assembly/messages"
    assert req["body"] == {
        "body": "Reply text",
        "title": "",
        "status": "reply",
        "urgency": "inbox",
        "reply_to": "m-42",
        "to": ["orchestrator"],
    }

    with pytest.raises(ValueError, match="status"):
        agora_post_message(channel="assembly", body="x", status="urgent!!")
    with pytest.raises(ValueError, match="required"):
        agora_post_message(channel="assembly", body="   ")


def test_ack_inbox_coerces_cursors(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import agora_ack_inbox

    hub.responses[("POST", "/inbox/ack")] = {"acked": {"assembly": 42}}
    agora_ack_inbox(cursors={"assembly": "42"})
    assert hub.requests[-1]["body"] == {"cursors": {"assembly": 42}}

    # Models (and some schema paths) deliver object args as JSON strings.
    agora_ack_inbox(cursors='{"assembly": 43}')
    assert hub.requests[-1]["body"] == {"cursors": {"assembly": 43}}

    with pytest.raises(ValueError, match="cursors"):
        agora_ack_inbox(cursors={})


def test_dm_and_reads_hit_expected_paths(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import (
        agora_read_channel,
        agora_read_message,
        agora_send_dm,
    )

    hub.responses[("GET", "/channels/assembly/messages")] = [{"id": "m-1"}]
    hub.responses[("GET", "/channels/assembly/messages/m-1")] = {"id": "m-1", "body": "full"}
    hub.responses[("POST", "/dms/orchestrator/messages")] = {"id": "d-1"}

    assert agora_read_channel(channel="assembly", since=10, limit=5) == [{"id": "m-1"}]
    assert hub.requests[-1]["query"] == "since=10&limit=5"

    assert agora_read_message(channel="assembly", message_id="m-1")["body"] == "full"

    agora_send_dm(peer="orchestrator", body="private ping", status="open")
    req = hub.requests[-1]
    assert req["path"] == "/dms/orchestrator/messages"
    assert req["body"]["status"] == "open"


def test_channel_fs_tools_hit_expected_paths(hub) -> None:
    """c1669 step 3: the shared-filesystem surface the fleet scripts hand-rolled."""
    from abstractruntime.integrations.abstractcore.agora_tools import (
        channel_fs_list,
        channel_fs_read,
        channel_fs_write,
    )

    hub.responses[("PUT", "/channels/assembly/fs/plans/design.md")] = {
        "path": "plans/design.md",
        "version": 3,
    }
    out = channel_fs_write(
        channel="assembly",
        path="/plans/design.md",  # leading slash normalized off
        content="# Design\n",
        description="unified design plan",
        expect_version=2,
    )
    assert out == {"path": "plans/design.md", "version": 3}
    req = hub.requests[-1]
    assert req["method"] == "PUT"
    assert req["path"] == "/channels/assembly/fs/plans/design.md"
    assert req["body"] == {
        "content": "# Design\n",
        "mime": "text/markdown",
        "description": "unified design plan",
        "expect_version": 2,
    }

    hub.responses[("GET", "/channels/assembly/fs/plans/design.md")] = {
        "path": "plans/design.md",
        "content": "# Design\n",
        "version": 3,
    }
    got = channel_fs_read(channel="assembly", path="plans/design.md", version=3)
    assert got["content"] == "# Design\n"
    assert hub.requests[-1]["query"] == "version=3"

    hub.responses[("GET", "/channels/assembly/fs")] = [
        {"path": "plans/design.md", "version": 3}
    ]
    listed = channel_fs_list(channel="assembly", prefix="plans/")
    assert listed == [{"path": "plans/design.md", "version": 3}]
    assert hub.requests[-1]["query"] == "prefix=plans%2F"

    with pytest.raises(ValueError, match="path"):
        channel_fs_write(channel="assembly", path="   ", content="x")
    with pytest.raises(ValueError, match="channel"):
        channel_fs_read(channel="  ", path="plans/design.md")


def test_channel_store_tools_hit_expected_paths(hub) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import (
        channel_store_get,
        channel_store_set,
    )

    hub.responses[("PUT", "/channels/assembly/store/decision%3Alease")] = {
        "key": "decision:lease",
        "version": 1,
    }
    out = channel_store_set(
        channel="assembly", key="decision:lease", value="per-window", expect_version=0
    )
    assert out == {"key": "decision:lease", "version": 1}
    req = hub.requests[-1]
    assert req["method"] == "PUT"
    assert req["path"] == "/channels/assembly/store/decision%3Alease"
    assert req["body"] == {"value": "per-window", "expect_version": 0}

    hub.responses[("GET", "/channels/assembly/store/decision%3Alease")] = {
        "key": "decision:lease",
        "value": "per-window",
        "version": 1,
    }
    got = channel_store_get(channel="assembly", key="decision:lease")
    assert got["value"] == "per-window"

    with pytest.raises(ValueError, match="key"):
        channel_store_set(channel="assembly", key="  ", value="x")


def test_channel_tools_carry_the_alias_identity(hub, monkeypatch: pytest.MonkeyPatch) -> None:
    """H8 parity: the new writes resolve per-agent keys exactly like the seven."""
    from abstractruntime.integrations.abstractcore.agora_tools import channel_store_set

    monkeypatch.setenv("AGORA_API_KEY__RESIDENT_A", "resident_a_key")
    hub.responses[("PUT", "/channels/assembly/store/claim:item-1")] = {"version": 1}
    channel_store_set(
        channel="assembly", key="claim:item-1", value="resident-a", _agora_agent="resident-a"
    )
    assert hub.requests[-1]["auth"] == "Bearer resident_a_key"


def test_channel_tool_approval_classification() -> None:
    """c1669 step 3 acceptance: reads safe auto-approve, WRITES write-classed."""
    from abstractruntime.integrations.abstractcore.tool_executor import ToolApprovalPolicy

    policy = ToolApprovalPolicy()
    reads = [
        {"name": n, "arguments": {}}
        for n in ("channel_fs_read", "channel_fs_list", "channel_store_get")
    ]
    assert policy.requires_approval(reads) is False
    for write in ("channel_fs_write", "channel_store_set"):
        assert policy.requires_approval([{"name": write, "arguments": {}}]) is True


def test_toolset_is_twelve_tools() -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import (
        AGORA_TOOL_NAMES,
        AGORA_TOOLS,
    )

    assert len(AGORA_TOOLS) == 12
    assert AGORA_TOOL_NAMES[-5:] == [
        "channel_fs_write",
        "channel_fs_read",
        "channel_fs_list",
        "channel_store_set",
        "channel_store_get",
    ]


def test_missing_api_key_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractruntime.integrations.abstractcore.agora_tools import agora_whoami

    monkeypatch.delenv("AGORA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AGORA_API_KEY"):
        agora_whoami()


def test_toolset_registration_is_env_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractruntime.integrations.abstractcore import default_tools

    monkeypatch.delenv("AGORA_API_KEY", raising=False)
    monkeypatch.delenv("ABSTRACT_ENABLE_AGORA_TOOLS", raising=False)
    assert "agora" not in default_tools.get_default_toolsets()

    # AND-gate (c4218 incident): intent + a key, both required.
    monkeypatch.setenv("ABSTRACT_ENABLE_AGORA_TOOLS", "1")
    monkeypatch.setenv("AGORA_API_KEY", "this-hosts-key")
    toolsets = default_tools.get_default_toolsets()
    assert "agora" in toolsets
    tool_map = default_tools.build_default_tool_map()
    for name in (
        "agora_whoami",
        "agora_check_inbox",
        "agora_ack_inbox",
        "agora_read_channel",
        "agora_read_message",
        "agora_post_message",
        "agora_send_dm",
    ):
        assert name in tool_map


def test_agora_messaging_tools_are_safe_auto_approve() -> None:
    """Hub comms must not stall gateway runs behind approval waits (telegram
    precedent). Scope (c1669 step 3): the MESSAGING seven + channel READS are
    safe; channel fs/store WRITES are deliberately write-classed and excluded
    here — see test_channel_tool_approval_classification."""
    from abstractruntime.integrations.abstractcore.agora_tools import AGORA_TOOL_NAMES
    from abstractruntime.integrations.abstractcore.tool_executor import ToolApprovalPolicy

    policy = ToolApprovalPolicy()
    write_classed = {"channel_fs_write", "channel_store_set"}
    calls = [
        {"name": n, "arguments": {}} for n in AGORA_TOOL_NAMES if n not in write_classed
    ]
    assert policy.requires_approval(calls) is False


def test_ambient_key_alone_never_mints_the_toolset(monkeypatch) -> None:
    """Contamination incident 2026-07-22 (adversary B / gateway c4218): a
    foreign AGORA_API_KEY inherited from the spawning shell must NOT
    register agora tools - key AND explicit intent, both required."""
    from abstractruntime.integrations.abstractcore.default_tools import (
        agora_tools_enabled,
    )

    monkeypatch.delenv("ABSTRACT_ENABLE_AGORA_TOOLS", raising=False)
    monkeypatch.setenv("AGORA_API_KEY", "foreign-seat-key")
    assert agora_tools_enabled() is False, "the lucky-key vector is dead"
    monkeypatch.setenv("AGORA_API_KEY__resident1", "alias-key")
    assert agora_tools_enabled() is False, "alias keys alone are ambient too"
    # Intent without a key: a toolset that cannot authenticate never registers.
    monkeypatch.delenv("AGORA_API_KEY", raising=False)
    monkeypatch.delenv("AGORA_API_KEY__resident1", raising=False)
    monkeypatch.setenv("ABSTRACT_ENABLE_AGORA_TOOLS", "1")
    assert agora_tools_enabled() is False, "flag-alone cannot speak"
    # Both halves: registered.
    monkeypatch.setenv("AGORA_API_KEY", "this-hosts-key")
    assert agora_tools_enabled() is True
