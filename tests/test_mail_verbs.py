"""Mail verbs for flows (`mail.*` fixed verbs through TOOL_INVOKE) and the Email record."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.core.models import RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_invoke_handler
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import FileArtifactStore, is_artifact_ref
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.triggers.email_received import email_id, email_record, parse_email_id, thread_id_of

ALICE_ADDR, ALICE_PW = "alice@example.test", "alice-pw"


# --- the Email record ------------------------------------------------------------------------------


def test_email_record_shape():
    rec = email_record({"uid": "7", "uidvalidity": 1000, "folder": "INBOX", "message_id": "<m7@x>", "from": "Bob <b@x>",
                        "to": "a@x", "cc": "", "subject": "Hi", "date": "d", "body_text": "t", "body_html": "<p>t</p>",
                        "references": ["<root@x>", "<m6@x>"], "flags": ["\\Seen", "\\Flagged", "Work"],
                        "attachments": [{"index": 0, "filename": "a.pdf", "content_type": "application/pdf", "size": 3}]},
                       mailbox="alice@example.test")
    assert rec == {"id": "1000:7:INBOX", "message_id": "<m7@x>", "mailbox": "alice@example.test", "folder": "INBOX",
                   "thread_id": "<root@x>", "from": "Bob <b@x>", "to": "a@x", "cc": "", "subject": "Hi", "date": "d",
                   "body_text": "t", "body_html": "<p>t</p>",
                   "attachments": [{"index": 0, "name": "a.pdf", "size": 3, "type": "application/pdf", "artifact": None}],
                   "seen": True, "flagged": True, "labels": ["Work"]}
    assert parse_email_id(rec["id"]) == ("INBOX", 1000, 7) and parse_email_id("x") is None
    assert parse_email_id(email_id("[Gmail]/All: Mail", 3, 4)) == ("[Gmail]/All: Mail", 3, 4)
    no_flags = email_record({"uid": 1, "uidvalidity": 1, "folder": "INBOX", "message_id": "<a@x>", "in_reply_to": "<p@x>"}, mailbox=None)
    assert (no_flags["seen"], no_flags["flagged"], no_flags["labels"], no_flags["thread_id"]) == (None, None, None, "<p@x>")
    assert thread_id_of({"message_id": "<own@x>"}) == "<own@x>"


# --- a runtime with a mailbox (AbstractCore's hermetic IMAP server, real client) ----------------------


@pytest.fixture(scope="module")
def ca(tmp_path_factory):
    from abstractcore.testing.mailserver import TestCA

    return TestCA.create(tmp_path_factory.mktemp("ca"))


@pytest.fixture
def imap(ca):
    from abstractcore.testing.mailserver import FakeImapServer

    server = FakeImapServer(ca, users={ALICE_ADDR: ALICE_PW}, security="ssl", folders=("INBOX", "Sent", "Archive", "Trash"))
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _reset_core_resolver(tmp_path, monkeypatch):
    from abstractruntime.email.binding import uninstall_core_resolver

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    uninstall_core_resolver()
    yield
    uninstall_core_resolver()


def _context(tmp_path, ca, imap, *, smtp=None):
    from abstractcore.comms.email import EmailAccount, EmailContext, EmailSecret, ImapSettings, RecipientPolicy, SmtpSettings
    from abstractcore.comms.email.limits import SendLimits, SendRateLimiter

    account = EmailAccount.build(
        address=ALICE_ADDR,
        imap=ImapSettings.build("localhost", port=imap.port, security="ssl", ca_file=str(ca.ca_pem)),
        smtp=SmtpSettings.build("localhost", port=smtp.port, security="starttls", ca_file=str(ca.ca_pem)) if smtp else None,
    )
    limits = SendLimits()
    return EmailContext(account=account, secret=EmailSecret(ALICE_PW), policy=RecipientPolicy.build("allowlist", [ALICE_ADDR, "bob@x.test"]),
                        limits=limits, limiter=SendRateLimiter(tmp_path / "sends.json", limits), registered_address=ALICE_ADDR)


def _node(run, ctx):
    call = run.vars["call"]
    return StepPlan(node_id="go", effect=Effect(type=EffectType.TOOL_INVOKE, payload={"name": call["name"], "arguments": call["arguments"]},
                                                result_key="out"), next_node="done")


def _done(run, ctx):
    return StepPlan(node_id="done", complete_output={"out": run.vars.get("out")})


FLOW = WorkflowSpec(workflow_id="mailflow", entry_node="go", nodes={"go": _node, "done": _done})


def _runtime(tmp_path, ctx_by_ref: Dict[str, Any]):
    registry = WorkflowRegistry()
    registry.register(FLOW)
    rt = Runtime(run_store=JsonFileRunStore(tmp_path / "runs"), ledger_store=JsonlLedgerStore(tmp_path / "ledger"),
                 workflow_registry=registry, artifact_store=FileArtifactStore(tmp_path / "artifacts"),
                 effect_handlers={EffectType.TOOL_INVOKE: make_tool_invoke_handler()})
    rt.set_email_binding({"account_ref": "t:alice:default", "address": ALICE_ADDR})
    seen = []

    def resolver(binding, use=None):
        seen.append((binding.account_ref, use))
        return ctx_by_ref.get(binding.account_ref)

    rt.set_email_context_resolver(resolver)
    return rt, seen


def _call(rt, name, arguments, *, vars_extra: Optional[Dict[str, Any]] = None):
    run_id = rt.start(workflow=FLOW, vars={"call": {"name": name, "arguments": arguments}, **(vars_extra or {})})
    state = rt.tick(workflow=FLOW, run_id=run_id)
    assert state.status == RunStatus.COMPLETED, state.error
    (item,) = state.output["out"]["results"]
    return item


def _fill(imap, n):
    from abstractcore.testing.mailserver import build_message

    for i in range(1, n + 1):
        att = [("note.txt", "text/plain", b"attached " + str(i).encode())] if i == 2 else []
        imap.add_message("INBOX", build_message(from_="bob@x.test" if i % 2 else "carol@x.test", to=ALICE_ADDR,
                                                subject=f"mail {i}", text=f"body {i}", attachments=att))


def test_search_get_and_attachment_on_the_runs_mailbox(tmp_path, ca, imap):
    _fill(imap, 3)
    rt, seen = _runtime(tmp_path, {"t:alice:default": _context(tmp_path, ca, imap)})
    found = _call(rt, "mail.search", {"query": {"from": "bob@x.test"}, "limit": 10})
    assert found["success"], found
    emails = found["output"]["emails"]
    assert [e["subject"] for e in emails] == ["mail 3", "mail 1"] and emails[0]["body_text"] is None
    assert emails[0]["mailbox"] == ALICE_ADDR and emails[0]["seen"] is False and emails[0]["folder"] == "INBOX"
    assert seen[-1] == ("t:alice:default", "action")
    got = _call(rt, "mail.get", {"ids": [emails[0]["id"]]})
    assert got["output"]["emails"][0]["body_text"] == "body 3"
    second = _call(rt, "mail.search", {"query": {"subject": "mail 2"}})["output"]["emails"][0]
    att = _call(rt, "mail.attachment", {"id": second["id"], "index": 0})["output"]
    assert is_artifact_ref(att["artifact"]) and att["name"] == "note.txt" and att["size"] == len(b"attached 2")
    assert rt.artifact_store.load(att["artifact"]["$artifact"]).content == b"attached 2"
    assert not any("\\Seen" in m.flags for m in imap.folders["INBOX"].messages)  # reading never marks read


def test_stale_ids_and_bad_arguments_are_typed(tmp_path, ca, imap):
    _fill(imap, 1)
    rt, _ = _runtime(tmp_path, {"t:alice:default": _context(tmp_path, ca, imap)})
    out = _call(rt, "mail.get", {"ids": ["1:1:INBOX"]})  # the fake folder's UIDVALIDITY is 1000
    assert not out["success"] and out["error_detail"]["code"] == "email_message_not_found"
    out = _call(rt, "mail.get", {"ids": ["not-an-id"]})
    assert out["error_detail"]["code"] == "invalid_arguments"
    out = _call(rt, "mail.search", {"query": {"body_regex": ".*"}})
    assert out["error_detail"]["code"] == "invalid_arguments"


def test_writes_are_refused_on_a_read_only_mailbox(tmp_path, ca, imap):
    _fill(imap, 1)
    rt, _ = _runtime(tmp_path, {"t:alice:default": _context(tmp_path, ca, imap)})  # IMAP only: read-only
    for name, args in (("mail.mark", {"ids": ["1000:1:INBOX"], "seen": True}), ("mail.move", {"ids": ["1000:1:INBOX"], "folder": "Archive"}),
                       ("mail.delete", {"ids": ["1000:1:INBOX"]}), ("mail.send", {"to": "bob@x.test", "subject": "s", "text": "t"}),
                       ("mail.reply", {"id": "1000:1:INBOX", "text": "t"})):
        out = _call(rt, name, args)
        assert out["error_detail"]["code"] == "email_mailbox_read_only", name


def test_the_mailbox_argument_and_the_triggers_mailbox(tmp_path, ca, imap):
    _fill(imap, 1)
    ctx = _context(tmp_path, ca, imap)
    rt, seen = _runtime(tmp_path, {"t:alice:default": ctx, "t:alice:second": ctx})
    _call(rt, "mail.search", {"mailbox": "second"})
    assert seen[-1][0] == "t:alice:second"
    _call(rt, "mail.search", {"mailbox": "self"})
    assert seen[-1][0] == "t:alice:default"
    # No mailbox argument: the run's own binding (an email-triggered occurrence carries its trigger's).
    _call(rt, "mail.search", {}, vars_extra={"_runtime": {"email_account": {"account_ref": "t:alice:second", "address": ""}}})
    assert seen[-1][0] == "t:alice:second"
    out = _call(rt, "mail.search", {"mailbox": "nobody"})
    assert out["error_detail"]["code"] == "email_not_configured"


def _writable_context(tmp_path, ca, imap):
    """IMAP + SMTP settings: a writable mailbox (`EmailAccount.can_write`). Mailbox changes use IMAP only."""
    from abstractcore.comms.email import EmailAccount, EmailContext, EmailSecret, ImapSettings, RecipientPolicy, SmtpSettings
    from abstractcore.comms.email.limits import SendLimits, SendRateLimiter

    account = EmailAccount.build(address=ALICE_ADDR, imap=ImapSettings.build("localhost", port=imap.port, security="ssl", ca_file=str(ca.ca_pem)),
                                 smtp=SmtpSettings.build("localhost", port=1, security="starttls", ca_file=str(ca.ca_pem)))
    limits = SendLimits()
    return EmailContext(account=account, secret=EmailSecret(ALICE_PW), policy=RecipientPolicy.build("allowlist", [ALICE_ADDR]),
                        limits=limits, limiter=SendRateLimiter(tmp_path / "sends.json", limits), registered_address=ALICE_ADDR)


def test_mark_label_move_and_delete_change_the_mailbox(tmp_path, ca, imap):
    _fill(imap, 4)
    rt, _ = _runtime(tmp_path, {"t:alice:default": _writable_context(tmp_path, ca, imap)})
    inbox = imap.folders["INBOX"].messages
    emails = _call(rt, "mail.search", {})["output"]["emails"]
    by = {e["subject"]: e["id"] for e in emails}
    out = _call(rt, "mail.mark", {"ids": [by["mail 1"]], "seen": True, "flagged": True})
    assert out["success"] and out["output"] == {"ids": [by["mail 1"]], "changed": 1}, out
    flags = next(m.flags for m in inbox if m.uid == 1)
    assert "\\Seen" in flags and "\\Flagged" in flags
    assert _call(rt, "mail.label", {"ids": [by["mail 2"]], "add": ["Work"]})["success"]
    got = _call(rt, "mail.get", {"ids": [by["mail 1"], by["mail 2"]]})["output"]["emails"]
    assert {e["subject"]: (e["seen"], e["flagged"], e["labels"]) for e in got} == {
        "mail 1": (True, True, []), "mail 2": (False, False, ["Work"])}
    moved = _call(rt, "mail.move", {"ids": [by["mail 3"]], "folder": "Archive"})
    assert moved["success"] and moved["output"]["moved"] == 1
    assert [m.uid for m in imap.folders["Archive"].messages] and 3 not in [m.uid for m in imap.folders["INBOX"].messages]
    deleted = _call(rt, "mail.delete", {"ids": [by["mail 4"]]})
    assert deleted["success"] and 4 not in [m.uid for m in imap.folders["INBOX"].messages]
    assert len(imap.folders["Trash"].messages) == 1  # to Trash, not gone


def test_send_goes_through_guarded_send(tmp_path, ca, imap):
    pytest.importorskip("aiosmtpd")
    from abstractcore.testing.mailserver import FakeSmtpServer

    smtp = FakeSmtpServer(ca, users={ALICE_ADDR: ALICE_PW}, security="starttls")
    try:
        rt, _ = _runtime(tmp_path, {"t:alice:default": _context(tmp_path, ca, imap, smtp=smtp)})
        meta = rt.artifact_store.store(b"report body", content_type="text/plain", tags={"filename": "report.txt"})
        from abstractruntime.storage.artifacts import artifact_ref

        out = _call(rt, "mail.send", {"to": ["bob@x.test"], "subject": "Report", "text": "see attached",
                                      "attachments": [artifact_ref(meta.artifact_id)]})
        assert out["success"], out
        assert out["output"]["accepted"] == ["bob@x.test"]
        refused = _call(rt, "mail.send", {"to": ["stranger@elsewhere.test"], "subject": "x", "text": "y"})
        assert not refused["success"] and refused["error_detail"]["code"]  # the account's recipient policy
    finally:
        smtp.close()


def test_mail_nodes_bake_the_verb():
    from abstractruntime.visualflow_compiler.compiler import _create_tool_invoke_base_handler

    handler = _create_tool_invoke_base_handler(node_id="n", node_type="mail_move", next_node=None, input_key="in", output_key="out")

    class _Run:
        vars = {"in": {"ids": ["1000:1:INBOX"], "folder": "Archive", "name": "mail.delete", "permanent": True}}

    plan = handler(_Run(), None)
    assert plan.effect.type == EffectType.TOOL_INVOKE and plan.effect.payload["name"] == "mail.move"
    assert plan.effect.payload["arguments"] == {"ids": ["1000:1:INBOX"], "folder": "Archive"}


def test_email_record_merges_gmail_labels_with_keywords():
    rec = email_record({"uid": 1, "uidvalidity": 1, "folder": "INBOX", "flags": ["\\Seen", "Work", "Urgent"],
                        "labels": ["Work", "Clients/ACME"]}, mailbox="m")
    assert rec["labels"] == ["Work", "Urgent", "Clients/ACME"]  # keywords first, no duplicates
    only_gmail = email_record({"uid": 1, "uidvalidity": 1, "folder": "INBOX", "labels": ["Clients/ACME"]}, mailbox="m")
    assert only_gmail["labels"] == ["Clients/ACME"] and only_gmail["seen"] is None


def test_ids_take_one_record_or_one_id_and_mailbox_takes_an_address(tmp_path, ca, imap):
    """Flow's ForEach hands one Email record to Mark as read; an unconnected Mailbox input picks up the
    trigger's `mailbox`, which is an address."""
    from types import SimpleNamespace

    _fill(imap, 2)
    full = _writable_context(tmp_path, ca, imap)
    second = SimpleNamespace(account=SimpleNamespace(address="box2@example.test", can_write=True, can_send=True), client=full.client)
    rt, seen = _runtime(tmp_path, {"t:alice:default": full, "t:alice:second": second})
    record = _call(rt, "mail.search", {"query": {"subject": "mail 1"}})["output"]["emails"][0]
    out = _call(rt, "mail.mark", {"ids": record, "seen": True})  # one record, not a list
    assert out["success"] and out["output"]["ids"] == [record["id"]], out
    assert _call(rt, "mail.mark", {"ids": record["id"], "flagged": True})["success"]  # one id string
    assert _call(rt, "mail.get", {"ids": [record]})["output"]["emails"][0]["flagged"] is True  # a list of records
    other = _call(rt, "mail.search", {"query": {"subject": "mail 2"}})["output"]["emails"][0]
    assert _call(rt, "mail.reply", {"id": other, "text": "x"})["error_detail"]["code"] != "invalid_arguments"  # `id` takes a record
    # The primary's address.
    _call(rt, "mail.search", {"mailbox": "Alice@Example.test"})
    assert seen[-1][0] == "t:alice:default"
    # The run's own mailbox (a trigger's further mailbox, bound without an address): matched on the resolved account.
    own = {"_runtime": {"email_account": {"account_ref": "t:alice:second", "address": ""}}}
    assert _call(rt, "mail.search", {"mailbox": "box2@example.test"}, vars_extra=own)["success"]
    out = _call(rt, "mail.search", {"mailbox": "stranger@example.test"}, vars_extra=own)
    assert out["error_detail"]["code"] == "invalid_mailbox" and "stranger@example.test" in out["error_detail"]["cause"]
