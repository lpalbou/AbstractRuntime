"""Hermetic email fixtures for the runtime (framework backlog 0992 WP2).

AbstractCore's test mail servers (`abstractcore.testing.mailserver`): a throwaway CA, a fake
IMAP server and an aiosmtpd SMTP server on 127.0.0.1 ephemeral ports. Every address is under
example.test; no real mailbox, no keychain (contexts are built in memory), no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List

import pytest

pytest.importorskip("aiosmtpd")

from abstractcore.comms.email import (  # noqa: E402
    EmailAccount,
    EmailContext,
    EmailSecret,
    ImapSettings,
    RecipientPolicy,
    SendLimits,
    SendRateLimiter,
    SmtpSettings,
)
from abstractcore.testing.mailserver import FakeImapServer, FakeSmtpServer, TestCA, build_message  # noqa: E402

ALICE = "alice@example.test"
BOB = "bob@example.test"
STRANGER = "attacker@evil.test"
# grep-able sentinels: must never appear in a run store, ledger, inbox or result.
ALICE_PASSWORD = "sentinel-Alice-Pa55-91f2c7"
BOB_PASSWORD = "sentinel-Bob-Pa55-4d8e10"


@pytest.fixture(scope="session")
def ca(tmp_path_factory) -> TestCA:
    return TestCA.create(tmp_path_factory.mktemp("email-ca"))


@pytest.fixture
def imap(ca: TestCA) -> Iterator[FakeImapServer]:
    server = FakeImapServer(ca, users={ALICE: ALICE_PASSWORD, BOB: BOB_PASSWORD}, security="ssl")
    yield server
    server.close()


@pytest.fixture
def smtp(ca: TestCA) -> Iterator[FakeSmtpServer]:
    server = FakeSmtpServer(ca, users={ALICE: ALICE_PASSWORD, BOB: BOB_PASSWORD}, security="starttls")
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _reset_core_resolver() -> Iterator[None]:
    from abstractruntime.email.binding import uninstall_core_resolver

    uninstall_core_resolver()
    yield
    uninstall_core_resolver()


def make_context(
    tmp_path: Path,
    ca: TestCA,
    *,
    imap: FakeImapServer = None,
    smtp: FakeSmtpServer = None,
    address: str = ALICE,
    password: str = ALICE_PASSWORD,
    policy_entries: List[str] = None,
    policy_mode: str = "allowlist",
) -> EmailContext:
    account = EmailAccount.build(
        address=address,
        imap=ImapSettings.build("localhost", port=imap.port, security="ssl", ca_file=str(ca.ca_pem)) if imap else None,
        smtp=SmtpSettings.build("localhost", port=smtp.port, security="starttls", ca_file=str(ca.ca_pem)) if smtp else None,
    )
    limits = SendLimits()
    return EmailContext(
        account=account,
        secret=EmailSecret(password),
        policy=RecipientPolicy.build(policy_mode, policy_entries if policy_entries is not None else [address]),
        limits=limits,
        limiter=SendRateLimiter(tmp_path / f"sends-{address}.json", limits),
        registered_address=address,
    )


def add_mail(server: FakeImapServer, *, from_: str, subject: str, text: str = "hello", to: str = ALICE, **kw) -> int:
    return server.add_message("INBOX", build_message(from_=from_, to=to, subject=subject, text=text, **kw))


def tree_text(*roots: Path) -> str:
    """Every byte under the given directories, decoded leniently (for sentinel greps)."""
    chunks: List[str] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                chunks.append(p.read_bytes().decode("utf-8", errors="replace"))
    return "\n".join(chunks)


__all__ = [
    "ALICE",
    "ALICE_PASSWORD",
    "BOB",
    "BOB_PASSWORD",
    "STRANGER",
    "add_mail",
    "ca",
    "imap",
    "make_context",
    "smtp",
    "tree_text",
    "_reset_core_resolver",
]
