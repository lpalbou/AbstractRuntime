"""Mail verbs for flows: deterministic email handling (`mail.*` fixed verbs).

A flow node of a `mail_*` type compiles to ONE `EffectType.TOOL_INVOKE` whose verb is baked from the
node TYPE (the camera-node rule: never from a pin, never mintable by a model), and the TOOL_INVOKE
handler runs it here, without the approval gate (the person authored the node):

    mail.search     {query?, folder?, limit?}                   -> {emails: Email[], has_more}   (no bodies)
    mail.get        {ids}                                       -> {emails: Email[]}   (with bodies)
    mail.mark       {ids, seen?, flagged?}                      -> {ids, changed}
    mail.label      {ids, add?, remove?}                        -> {ids, changed}
    mail.move       {ids, folder}                               -> {ids, moved}
    mail.delete     {ids, permanent?=false}                     -> {ids, deleted}
    mail.send       {to, cc?, bcc?, subject, text, html?, attachments?: [artifact refs]} -> {message_id, accepted}
    mail.reply      {id, text, html?, reply_all?=false}         -> {message_id, accepted}
    mail.forward    {id, to, note?, include_attachments?=true}  -> {message_id, accepted}
    mail.attachment {id, index}                                 -> {artifact, name, size, type}

Every verb takes `mailbox` (optional): a mailbox id of the user (`"self"` / `"default"` = the
primary), or the ADDRESS of the run's own mailbox or of the primary (an Email record's or a
trigger's `mailbox` value). The default is the run's own mailbox (`_runtime.email_account`, which an email-triggered
occurrence binds to its trigger's mailbox), else the runtime's primary. The mailbox is resolved
through the host's resolver (`Runtime.set_email_context_resolver`, `use="action"`); credentials
never reach vars or results. Messages are named by the `Email` record's `id`
(`"<uidvalidity>:<uid>:<folder>"`, `triggers.email_received.email_id`); `ids` / `id` take one id, one
Email record (named by its `id`), or a list of either. A stale id (the server rebuilt the folder) is
refused, never guessed.

Built on AbstractCore >= 2.28.0: `EmailClient.search` (typed query), `get_many`, `mark`, `label`,
`move`, `delete`, `download_attachment`, `session(write=...)`, and `EmailContext.send` / `reply` /
`forward` (through `guarded_send`: the account's recipient policy and rate limits apply). Mailbox
changes (`mark`, `label`, `move`, `delete`) and sends need a mailbox set to Read and write
(`EmailAccount.can_write`, AbstractCore's explicit access choice), else `email_mailbox_read_only`;
sends also need its outgoing server (`can_send`), else `email_mailbox_cannot_send`. Both are refused
before connecting.

Results use the TOOL_CALLS envelope (`{mode: "executed", results: [{call_id, name, success, output,
error}]}`); a failure is `success: false` with `error = "<code>: <cause>"` and `error_detail =
{code, cause, fix}`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

MAIL_VERBS = (
    "mail.search", "mail.get", "mail.mark", "mail.label", "mail.move", "mail.delete",
    "mail.send", "mail.reply", "mail.forward", "mail.attachment",
)
CHANGE_VERBS = frozenset({"mail.mark", "mail.label", "mail.move", "mail.delete"})  # change the mailbox: can_write
SEND_VERBS = frozenset({"mail.send", "mail.reply", "mail.forward"})  # send mail: can_write and can_send
WRITE_VERBS = CHANGE_VERBS | SEND_VERBS
PRIMARY_MAILBOXES = ("", "self", "default", "primary")
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 500
# The typed query of `mail.search` -> AbstractCore's `EmailClient.search` keywords.
_QUERY_KEYS = {"from": "from_", "to": "to", "subject": "subject", "text": "text", "since": "since", "until": "until",
               "unseen": "unseen", "flagged": "flagged", "labels": "labels"}


class MailVerbError(Exception):
    """A typed refusal: `code`, `cause` (a sentence), `fix` (a sentence)."""

    def __init__(self, code: str, cause: str, fix: str = "") -> None:
        super().__init__(cause)
        self.code, self.cause, self.fix = code, cause, fix

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "cause": self.cause, "fix": self.fix}


def is_mail_verb(name: Any) -> bool:
    return isinstance(name, str) and name in MAIL_VERBS


# --- mailbox resolution ------------------------------------------------------------------------


def resolve_mailbox(run: Any, mailbox: Any) -> Tuple[Any, Any, str]:
    """`(EmailBinding, EmailContext, mailbox_label)` for a verb of `run` (see the module doc)."""
    from .email.binding import EmailBinding, account_id_of_ref, account_ref_with_id, binding_of

    primary = getattr(run, "_runtime_email_binding", None)
    own = binding_of(getattr(run, "vars", None))
    name = str(mailbox or "").strip()
    resolver = getattr(run, "_runtime_email_resolver", None)
    if not name:
        binding = own or primary
    elif name.lower() in PRIMARY_MAILBOXES:
        binding = primary
    elif "@" in name:
        binding = _binding_for_address(name, [own, primary], resolver)
    else:
        base = primary or own
        if base is None:
            raise MailVerbError("email_not_configured", "No mailbox is connected for this user.",
                                "Connect a mailbox in Settings -> Email.")
        if account_id_of_ref(base.account_ref) == name:
            binding = base
        else:
            if any(c in name for c in ": \t\r\n"):
                raise MailVerbError("invalid_mailbox", f"{name!r} is not a mailbox id.", "Use the mailbox id from the mail accounts.")
            binding = EmailBinding(account_ref=account_ref_with_id(base.account_ref, name), address="")
    if binding is None:
        raise MailVerbError("email_not_configured", "No mailbox is connected for this user.", "Connect a mailbox in Settings -> Email.")
    if resolver is None:
        raise MailVerbError("email_not_configured", "This runtime has no mailbox resolver.",
                            "The host must call Runtime.set_email_context_resolver.")
    ctx = _call_resolver(resolver, binding)
    if ctx is None:
        raise MailVerbError("email_not_configured", "The mailbox is not connected or not enabled.",
                            "Connect and enable the mailbox in Settings -> Email.")
    label = str(getattr(binding, "address", "") or "") or (account_id_of_ref(binding.account_ref) or name or "self")
    return binding, ctx, label


def _call_resolver(resolver: Any, binding: Any) -> Any:
    try:
        return resolver(binding, use="action")
    except TypeError:
        return resolver(binding)


def _binding_for_address(address: str, candidates: List[Any], resolver: Any) -> Any:
    """The mailbox named by its ADDRESS (what an Email record's or a trigger's `mailbox` holds): the
    run's own mailbox or the primary, compared with the binding's address, else with the address of
    the account the host resolves for it (case-insensitive). Another mailbox is named by its id."""
    wanted = address.strip().lower()
    for binding in candidates:
        if binding is None:
            continue
        if str(getattr(binding, "address", "") or "").strip().lower() == wanted:
            return binding
    for binding in candidates:
        if binding is None or resolver is None:
            continue
        try:
            ctx = _call_resolver(resolver, binding)
        except Exception:  # noqa: BLE001 - an unusable candidate is simply not this address
            continue
        if str(getattr(getattr(ctx, "account", None), "address", "") or "").strip().lower() == wanted:
            return binding
    raise MailVerbError("invalid_mailbox", f"{address} is not the address of this flow's mailbox or of your primary mailbox.",
                        "Leave Mailbox empty to use the trigger's mailbox, or name another mailbox by its id (Settings -> Email).")


# --- helpers --------------------------------------------------------------------------------------


def _ids(arguments: Mapping[str, Any], key: str = "ids") -> List[Tuple[str, int, int]]:
    from .triggers.email_received import parse_email_id

    raw = arguments.get(key)
    if isinstance(raw, (str, Mapping)):
        raw = [raw]  # one id, or one Email record (named by its `id`): a ForEach item
    if not isinstance(raw, list) or not raw:
        raise MailVerbError("invalid_arguments", f"{key} must be a non-empty list of email ids.", "Pass the `id` of Email records.")
    out = []
    for value in raw:
        if isinstance(value, Mapping):
            value = value.get("id")
        parsed = parse_email_id(value)
        if parsed is None:
            raise MailVerbError("invalid_arguments", f"{value!r} is not an email id.", "Pass the `id` of Email records.")
        out.append(parsed)
    return out


def _by_folder(ids: List[Tuple[str, int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    groups: Dict[str, List[Tuple[int, int]]] = {}
    for folder, uidvalidity, uid in ids:
        groups.setdefault(folder, []).append((uidvalidity, uid))
    return groups


def _check_uidvalidity(client: Any, folder: str, wanted: List[Tuple[int, int]]) -> List[int]:
    state = client.folder_state(folder)
    current = int(state.get("uidvalidity") or 0)
    stale = [uv for uv, _ in wanted if int(uv) != current]
    if stale:
        raise MailVerbError("email_message_not_found", f"The folder {folder!r} was rebuilt by the server; these ids are stale.",
                            "Search the folder again and use the new ids.")
    return [uid for _, uid in wanted]


def _record(detail_or_summary: Any, mailbox: str) -> Dict[str, Any]:
    from .triggers.email_received import email_record

    data = detail_or_summary.to_dict() if hasattr(detail_or_summary, "to_dict") else dict(detail_or_summary)
    return email_record(data, mailbox=mailbox)


def _require_writable(ctx: Any, name: str) -> None:
    """Changes need a mailbox the person set to Read and write (`can_write`); sends also need its
    outgoing server (`can_send`). Two refusals, before connecting."""
    account = getattr(ctx, "account", None)
    address = str(getattr(account, "address", "") or "") or "this mailbox"
    if not getattr(account, "can_write", False):
        raise MailVerbError("email_mailbox_read_only", "This mailbox is read-only: automations only read it.",
                            f"In My email, turn on Read and write for {address} to change it or send from it.")
    if name in SEND_VERBS and not getattr(account, "can_send", False):
        raise MailVerbError("email_mailbox_cannot_send", "This mailbox can't send: it has no outgoing server.",
                            f"In My email, add the outgoing server (SMTP) for {address}.")


def _sent(result: Any) -> Dict[str, Any]:
    return {"message_id": result.message_id, "accepted": list(result.accepted), "refused": dict(result.refused)}


def _attachments_from_refs(run: Any, refs: Any) -> tuple:
    from .integrations.abstractcore.email_facade import Attachment
    from .storage.artifacts import get_artifact_id, is_artifact_ref

    if not refs:
        return ()
    store = getattr(run, "_runtime_artifact_store", None)
    out = []
    for ref in refs if isinstance(refs, list) else [refs]:
        if not is_artifact_ref(ref) or store is None:
            raise MailVerbError("invalid_arguments", "attachments must be artifact refs.", "Pass artifacts produced by the flow.")
        art = store.load(get_artifact_id(ref))
        if art is None:
            raise MailVerbError("invalid_arguments", f"artifact {get_artifact_id(ref)} is missing.", "Pass an existing artifact.")
        meta = getattr(art, "metadata", None)
        name = (getattr(meta, "tags", None) or {}).get("filename") or getattr(meta, "filename", None) or f"{get_artifact_id(ref)}"
        out.append(Attachment(filename=str(name), content_type=str(getattr(meta, "content_type", "") or "application/octet-stream"),
                              data=art.content))
    return tuple(out)


def _addresses(value: Any, field: str, *, required: bool = False) -> tuple:
    if value is None or value == "":
        if required:
            raise MailVerbError("invalid_arguments", f"{field} is required.", f"Give at least one address in {field}.")
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or any(not isinstance(a, str) or not a.strip() for a in items):
        raise MailVerbError("invalid_arguments", f"{field} must be an address or a list of addresses.", "")
    return tuple(a.strip() for a in items)


# --- the verbs ------------------------------------------------------------------------------------


def _search(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    query = args.get("query") or {}
    if not isinstance(query, Mapping):
        raise MailVerbError("invalid_arguments", "query must be an object.", f"Use the fields {sorted(_QUERY_KEYS)}.")
    unknown = sorted(k for k in query if k not in _QUERY_KEYS)
    if unknown:
        raise MailVerbError("invalid_arguments", f"unknown query field(s) {unknown}.", f"Use the fields {sorted(_QUERY_KEYS)}.")
    limit = args.get("limit", DEFAULT_SEARCH_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise MailVerbError("invalid_arguments", f"limit must be 1..{MAX_SEARCH_LIMIT}.", "")
    keywords = {_QUERY_KEYS[k]: v for k, v in query.items() if v is not None}
    page = client.search(folder=args.get("folder") or None, limit=limit, **keywords)
    return {"emails": [_record(m, mailbox) for m in page.get("messages") or []], "has_more": bool(page.get("has_more"))}


def _get(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    emails: List[Dict[str, Any]] = []
    missing: List[str] = []
    for folder, wanted in _by_folder(_ids(args)).items():
        uids = _check_uidvalidity(client, folder, wanted)
        got = client.get_many(uids, folder=folder)
        for uid in uids:
            detail = got.get(uid)
            if detail is None or isinstance(detail, Exception):
                missing.append(f"{wanted[uids.index(uid)][0]}:{uid}:{folder}")
            else:
                emails.append(_record(detail, mailbox))
    return {"emails": emails, "missing": missing}


def _imap_write(verb: str) -> Callable[..., Dict[str, Any]]:
    def run_verb(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
        ids = _ids(args)
        done: List[str] = []
        for folder, wanted in _by_folder(ids).items():
            uids = _check_uidvalidity(client, folder, wanted)
            if verb == "mark":
                if args.get("seen") is None and args.get("flagged") is None:
                    raise MailVerbError("invalid_arguments", "give seen and/or flagged.", "")
                client.mark(uids, seen=args.get("seen"), flagged=args.get("flagged"), folder=folder)
            elif verb == "label":
                add, remove = list(args.get("add") or []), list(args.get("remove") or [])
                if not add and not remove:
                    raise MailVerbError("invalid_arguments", "give labels to add and/or remove.", "")
                client.label(uids, add=add, remove=remove, folder=folder)
            elif verb == "move":
                target = str(args.get("folder") or "").strip()
                if not target:
                    raise MailVerbError("invalid_arguments", "folder (the destination) is required.", "")
                client.move(uids, target, folder=folder)
            elif verb == "delete":
                client.delete(uids, permanent=bool(args.get("permanent", False)), folder=folder)
            done.extend(f"{uv}:{uid}:{folder}" for uv, uid in wanted)
        key = {"mark": "changed", "label": "changed", "move": "moved", "delete": "deleted"}[verb]
        return {"ids": done, key: len(done)}

    return run_verb


def _send_verb(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    from .integrations.abstractcore.email_facade import OutgoingMessage

    subject = args.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        raise MailVerbError("invalid_arguments", "subject is required.", "")
    message = OutgoingMessage(
        to=_addresses(args.get("to"), "to", required=True), cc=_addresses(args.get("cc"), "cc"), bcc=_addresses(args.get("bcc"), "bcc"),
        subject=subject.strip(), text=str(args.get("text") or ""), html=str(args.get("html") or ""),
        attachments=_attachments_from_refs(run, args.get("attachments")),
    )
    return _sent(ctx.send(message))


def _one_id(args: Mapping[str, Any]) -> Tuple[str, int, int]:
    ids = _ids({"ids": args.get("id")})
    if len(ids) != 1:
        raise MailVerbError("invalid_arguments", "id must name one email.", "")
    return ids[0]


def _reply(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    folder, uidvalidity, uid = _one_id(args)
    _check_uidvalidity(client, folder, [(uidvalidity, uid)])
    return _sent(ctx.reply(uid, str(args.get("text") or ""), html=str(args.get("html") or ""),
                           reply_all=bool(args.get("reply_all", False)), folder=folder))


def _forward(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    folder, uidvalidity, uid = _one_id(args)
    _check_uidvalidity(client, folder, [(uidvalidity, uid)])
    return _sent(ctx.forward(uid, list(_addresses(args.get("to"), "to", required=True)), note=str(args.get("note") or ""),
                             include_attachments=bool(args.get("include_attachments", True)), folder=folder))


def _attachment(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    from .storage.artifacts import artifact_ref

    folder, uidvalidity, uid = _one_id(args)
    index = args.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise MailVerbError("invalid_arguments", "index must be the attachment's index (0, 1, ...).", "")
    _check_uidvalidity(client, folder, [(uidvalidity, uid)])
    store = getattr(run, "_runtime_artifact_store", None)
    if store is None:
        raise MailVerbError("artifact_store_missing", "This runtime has no artifact store.", "Create the Runtime with artifact_store=...")
    with tempfile.TemporaryDirectory(prefix="mail-attachment-") as tmp:
        info = client.download_attachment(uid, index, tmp, folder=folder)
        path = Path(str(info.get("path") or ""))
        if not path.is_file():
            files = [p for p in Path(tmp).iterdir() if p.is_file()]
            path = files[0] if files else path
        data = path.read_bytes()
    name = str(info.get("original_filename") or info.get("filename") or path.name)
    ctype = str(info.get("content_type") or "application/octet-stream")
    meta = store.store(data, content_type=ctype, run_id=getattr(run, "run_id", None),
                       tags={"kind": "email_attachment", "filename": name})
    return {"artifact": artifact_ref(meta.artifact_id), "name": name, "size": len(data), "type": ctype}


_VERBS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "mail.search": _search,
    "mail.get": _get,
    "mail.mark": _imap_write("mark"),
    "mail.label": _imap_write("label"),
    "mail.move": _imap_write("move"),
    "mail.delete": _imap_write("delete"),
    "mail.send": _send_verb,
    "mail.reply": _reply,
    "mail.forward": _forward,
    "mail.attachment": _attachment,
}


def run_mail_verb(run: Any, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute verb `name` for `run`; returns the verb's output or raises MailVerbError."""
    if name not in _VERBS:
        raise MailVerbError("unknown_verb", f"{name!r} is not a mail verb.", f"Use one of {list(MAIL_VERBS)}.")
    args = dict(arguments or {})
    _binding, ctx, mailbox = resolve_mailbox(run, args.pop("mailbox", None))
    if name in WRITE_VERBS:
        _require_writable(ctx, name)
    try:
        client = ctx.client()
        if name == "mail.send":
            return _VERBS[name](run, args, client, ctx, mailbox)
        # One connection for the whole verb; a writable one for mailbox changes.
        with client.session(write=name in CHANGE_VERBS):
            return _VERBS[name](run, args, client, ctx, mailbox)
    except MailVerbError:
        raise
    except Exception as exc:  # AbstractCore's typed EmailError carries code / cause / fix
        code = getattr(exc, "code", None)
        if code:
            raise MailVerbError(str(code), str(getattr(exc, "cause", "") or exc), str(getattr(exc, "fix", "") or "")) from exc
        raise


def mail_verb_outcome(run: Any, name: str, arguments: Mapping[str, Any], *, call_id: str = "") -> Dict[str, Any]:
    """The TOOL_CALLS-shaped result of one mail verb (never raises for a typed refusal)."""
    try:
        output = run_mail_verb(run, name, arguments)
        item = {"call_id": call_id, "name": name, "success": True, "output": output, "error": None}
    except MailVerbError as exc:
        item = {"call_id": call_id, "name": name, "success": False, "output": None,
                "error": f"{exc.code}: {exc.cause}", "error_detail": exc.to_dict()}
    return {"mode": "executed", "results": [item]}


__all__ = [
    "CHANGE_VERBS",
    "MAIL_VERBS",
    "SEND_VERBS",
    "MailVerbError",
    "WRITE_VERBS",
    "is_mail_verb",
    "mail_verb_outcome",
    "resolve_mailbox",
    "run_mail_verb",
]
