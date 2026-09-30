"""Runtime-owned facade over AbstractCore's mail library (`abstractcore.comms.email`).

Hosts (the gateway) import email names from HERE, never from `abstractcore` directly, so the
host -> Runtime -> AbstractCore boundary holds (framework backlog 0992; mirrors
`config_facade.py`). Every name is AbstractCore's own object, re-exported unchanged: accounts
and settings (`EmailAccount`, `ImapSettings`, `SmtpSettings`, `OAuthSettings`), the per-user
store and vault (`EmailAccountStore`, `SecretVault`, `EmailSecret`), the guarded send
(`EmailContext`, `guarded_send`, `OutgoingMessage`), the recipient policy (`RecipientPolicy`,
`evaluate`, `parse_recipients`), OAuth2 (`OAuthTokenClient`, `LoopbackAuthorization`,
`builtin_client`, `provider_preset`, `resolve_oauth_client`), reading (`SearchCriteria`,
`MailCursor`), TLS (`tls_context`), the typed errors (`EmailError` and its subclasses), and
the `legacy` module (one-time import of pre-2.20 settings).

Importing this module imports AbstractCore's mail package (the `integrations/abstractcore`
opt-in). `__all__` is AbstractCore's `abstractcore.comms.email.__all__` plus `legacy`, taken
from the installed AbstractCore at import time, so the facade can never lag behind or claim a
name core does not have. `GATEWAY_NAMES` lists the names hosts rely on today; a test checks
each is present (a core release that drops one fails that test, not a host at runtime).
"""

from __future__ import annotations

import abstractcore.comms.email as _core_email
from abstractcore.comms.email import *  # noqa: F401,F403 - re-exported unchanged
from abstractcore.comms.email import legacy  # noqa: F401 - re-exported module

# The names AbstractGateway imports (framework backlog 0992, WP3 request).
GATEWAY_NAMES = (
    "EmailAccount", "EmailAccountStore", "EmailContext", "EmailDisabled", "EmailError", "EmailInvalidMessage",
    "EmailInvalidSettings", "EmailNotConfigured", "EmailOAuthFailed", "EmailOAuthPending", "EmailRateLimited",
    "EmailSecret", "ImapSettings", "LoopbackAuthorization", "MailCursor", "OAuthSettings", "OAuthTokenClient",
    "OutgoingMessage", "SearchCriteria", "SecretVault", "SmtpSettings", "builtin_client", "evaluate", "guarded_send",
    "legacy", "parse_recipients", "provider_preset", "resolve_oauth_client", "tls_context",
)

__all__ = [*_core_email.__all__, "legacy", "GATEWAY_NAMES"]
