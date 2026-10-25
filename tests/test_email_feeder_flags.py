"""Trigger messages keep their flags: the watcher stores each new message with its `flags`
(\\Seen, \\Flagged, IMAP keywords) and Gmail `labels`, so the occurrence's Email record carries
`seen` / `flagged` / `labels` like a sample's (they were null before: the flags were dropped)."""

from __future__ import annotations

from abstractcore.testing.mailserver import build_message

from email_wp2_support import ALICE, ca, imap, _reset_core_resolver  # noqa: F401 - fixtures
from test_email_inbox_and_trigger import _feeder

from abstractruntime.triggers.email_received import email_record


def test_stored_new_mail_keeps_its_flags_and_the_email_record_reads_them(tmp_path, ca, imap):
    inbox, feeder, ctx = _feeder(tmp_path, ca, imap)
    assert feeder.poll(ctx).baseline
    raw = build_message(from_="a@example.test", to=ALICE, subject="flagged one", text="x")
    uid = imap.add_message("INBOX", raw, flags={"\\Flagged", "Invoices"})
    plain = imap.add_message("INBOX", build_message(from_="b@example.test", to=ALICE, subject="plain", text="y"))
    assert len(feeder.poll(ctx, force=True).appended) == 2
    by_uid = {e["payload"]["uid"]: e["payload"] for e in inbox.read()}
    assert sorted(by_uid[uid]["flags"]) == ["Invoices", "\\Flagged"] and by_uid[uid]["labels"] == []
    rec = email_record(by_uid[uid], mailbox=ALICE)
    assert (rec["seen"], rec["flagged"], rec["labels"]) == (False, True, ["Invoices"])
    rec2 = email_record(by_uid[plain], mailbox=ALICE)
    assert (rec2["seen"], rec2["flagged"], rec2["labels"]) == (False, False, [])
    assert "\\Seen" not in imap.flags_of("INBOX", uid)  # still read with PEEK


def test_gmail_labels_are_kept_too(tmp_path, ca):
    from abstractcore.testing.mailserver import FakeImapServer

    from email_wp2_support import ALICE_PASSWORD

    gmail = FakeImapServer(ca, users={ALICE: ALICE_PASSWORD}, security="ssl", gmail=True)
    try:
        inbox, feeder, ctx = _feeder(tmp_path, ca, gmail)
        feeder.poll(ctx)
        uid = gmail.add_message("INBOX", build_message(from_="a@example.test", to=ALICE, subject="g", text="x"), labels={"Clients/ACME"})
        feeder.poll(ctx, force=True)
        [event] = inbox.read()
        assert event["payload"]["uid"] == uid and event["payload"]["labels"] == ["Clients/ACME"]
    finally:
        gmail.close()
