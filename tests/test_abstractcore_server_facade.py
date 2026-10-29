"""The host facade preserves ASGI streaming and request-local auth."""
import asyncio
import importlib
import threading
from types import SimpleNamespace

import pytest

from abstractruntime.integrations.abstractcore import server_facade
from abstractcore.server.auth_policy import current_server_auth_policy

pytestmark = pytest.mark.basic


def test_concurrent_streams_keep_separate_auth_and_import_off_loop(monkeypatch):
    main_thread = threading.get_ident()
    imports = []
    barrier = None
    seen = []

    async def app(scope, receive, send):
        policy = current_server_auth_policy()
        assert policy.token == scope["token"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        await barrier.wait()
        assert current_server_auth_policy() is policy
        seen.append((policy.token, policy.allow_unauthenticated))
        await send({"type": "http.response.body", "body": b"last"})

    def load(name):
        imports.append((name, threading.get_ident()))
        return SimpleNamespace(app=app)

    monkeypatch.setattr(server_facade.importlib, "import_module", load)

    async def run():
        nonlocal barrier
        barrier = asyncio.Event()
        bodies = {"one": [], "two": []}
        async def receive():
            return {"type": "http.disconnect"}
        async def invoke(token, anonymous):
            async def send(message):
                bodies[token].append(message)
                if all(any(m.get("more_body") for m in messages) for messages in bodies.values()):
                    barrier.set()
            await server_facade.serve_core_request({"token": token}, receive, send, token=token, allow_unauthenticated=anonymous)
            assert current_server_auth_policy() is None
        await asyncio.gather(invoke("one", False), invoke("two", True))
        assert all(messages[-1]["body"] == b"last" for messages in bodies.values())
    asyncio.run(run())
    assert sorted(seen) == [("one", False), ("two", True)]
    assert all(name == "abstractcore.server.app" and thread != main_thread for name, thread in imports)


@pytest.mark.parametrize("error", [RuntimeError("stream failed"), asyncio.CancelledError()])
def test_policy_resets_on_failure_or_cancellation(monkeypatch, error):
    async def app(scope, receive, send):
        assert current_server_auth_policy().token == "managed"
        await asyncio.sleep(0)
        raise error
    monkeypatch.setattr(server_facade.importlib, "import_module", lambda name: SimpleNamespace(app=app))
    async def run():
        async def receive():
            return {"type": "http.disconnect"}
        async def send(message):
            pass
        with pytest.raises(type(error)):
            await server_facade.serve_core_request({}, receive, send, token="managed")
        assert current_server_auth_policy() is None
    asyncio.run(run())


def test_managed_core_server_delegates_without_wrapping_http_or_handle(monkeypatch):
    import sys
    from types import ModuleType
    managed = ModuleType("abstractcore.server.managed")
    calls = []
    handle = object()
    def start_managed_server(**kwargs):
        calls.append(kwargs)
        return handle
    managed.start_managed_server = start_managed_server
    monkeypatch.setitem(sys.modules, managed.__name__, managed)
    assert server_facade.start_core_server(host="127.0.0.1", port=0) is handle
    assert calls == [{"host": "127.0.0.1", "port": 0}]


def test_importing_managed_facade_never_imports_optional_server_dependencies():
    import os
    import subprocess
    import sys
    code = """
import sys
from abstractruntime.integrations.abstractcore.server_facade import start_core_server
assert 'abstractcore.server.app' not in sys.modules
assert 'abstractcore.server.managed' not in sys.modules
assert 'uvicorn' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)
