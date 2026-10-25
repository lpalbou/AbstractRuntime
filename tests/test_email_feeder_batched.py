"""The mail watcher reads a burst of new mail on ONE connection (measured by W1: one connection
per message cost ~10 s each remotely). One poll of 50 new messages: one sign-in, at most four
FETCHes (the new-mail listing, the headers/structure of all 50, one body section), the same
payloads as before, nothing marked read, the cursor at the last message appended."""

from __future__ import annotations

from email_wp2_support import ca, imap, _reset_core_resolver  # noqa: F401 - fixtures
from email_wp2_support import add_mail
from test_email_inbox_and_trigger import REF, _feeder


def test_fifty_new_messages_one_login_and_at_most_four_fetches(tmp_path, ca, imap):
    inbox, feeder, ctx = _feeder(tmp_path, ca, imap)
    assert feeder.poll(ctx).baseline
    uids = [add_mail(imap, from_=f"s{i}@example.test", subject=f"m{i}", text=f"hello {i}") for i in range(50)]
    logins, fetches = len(imap.logins), len(imap.fetches)
    report = feeder.poll(ctx, force=True)
    assert len(report.appended) == 50
    assert len(imap.logins) - logins == 1
    assert len(imap.fetches) - fetches <= 4, imap.fetches[fetches:]
    events = [e["payload"] for e in inbox.read()]
    assert [e["subject"] for e in events] == [f"m{i}" for i in range(50)]
    assert events[7]["body_text"].strip() == "hello 7" and events[7]["account_ref"] == REF and events[7]["uid"] == uids[7]
    for uid in uids:
        assert "\\Seen" not in imap.flags_of("INBOX", uid)
    assert feeder.status("INBOX")["cursor"]["last_uid"] == uids[-1]
    assert feeder.poll(ctx, force=True).appended == []  # nothing twice
