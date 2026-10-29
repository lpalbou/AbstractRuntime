"""Host-facing ASGI delegation to AbstractCore's serving application.

Hosts own route admission, network policy, and token storage. This facade keeps
Core imports behind Runtime's integration boundary and scopes authentication to
one complete ASGI response, including streamed bodies.
"""
from __future__ import annotations

import asyncio
import importlib
from typing import Any, Awaitable, Callable, MutableMapping


async def serve_core_request(
    scope: MutableMapping[str, Any],
    receive: Callable[[], Awaitable[dict[str, Any]]],
    send: Callable[[dict[str, Any]], Awaitable[None]],
    *,
    token: str,
    allow_unauthenticated: bool = False,
) -> None:
    """Serve a host-admitted request using an independent inbound auth policy.

    Requires the AbstractCore server dependencies. Importing this module does
    not load Core or its application. The host must provide a nonempty token,
    even when allowing anonymous requests; supplied credentials still validate.
    """
    module = await asyncio.to_thread(importlib.import_module, "abstractcore.server.app")
    from abstractcore.server.auth_policy import ServerAuthPolicy, use_server_auth_policy

    with use_server_auth_policy(ServerAuthPolicy(token, allow_unauthenticated)):
        await module.app(scope, receive, send)


def start_core_server(*, host: str | None = None, port: int | None = None) -> Any:
    """Start Core's independently authenticated server in this execution process.

    The returned Core-owned handle supplies ``url``, ``is_running``, ``stop()``
    and ``status()``. Importing this facade never imports the optional server.
    """
    from abstractcore.server.managed import start_managed_server

    return start_managed_server(host=host, port=port)
