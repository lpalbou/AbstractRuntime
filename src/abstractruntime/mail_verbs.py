"""Mail verbs for flows: deterministic email handling (`mail.*` fixed verbs).

A flow node of a `mail_*` type compiles to ONE `EffectType.TOOL_INVOKE` whose verb is baked from the
node TYPE (the camera-node rule: never from a pin, never mintable by a model), and the TOOL_INVOKE
handler runs it here, without the approval gate (the person authored the node):

    mail.search     {query?, folder?, limit?}                   -> {emails: Email[]}   (no bodies)
    mail.get        {ids}                                       -> {emails: Email[]}   (with bodies)
    mail.mark       {ids, seen?, flagged?}                      -> {ids, changed}
    mail.label      {ids, add?, remove?}                        -> {ids, changed}
    mail.move       {ids, folder}                               -> {ids, moved}
    mail.delete     {ids, permanent?=false}                     -> {ids, deleted}
    mail.send       {to, cc?, bcc?, subject, text, html?, attachments?: [artifact refs]} -> {message_id, accepted}
    mail.reply      {id, text, html?, reply_all?=false}         -> {message_id, accepted}
    mail.forward    {id, to, note?}                             -> {message_id, accepted}
    mail.attachment {id, index}                                 -> {artifact, name, size, type}

Every verb takes `mailbox` (optional): a mailbox id of the user (`"self"` / `"default"` = the
primary). The default is the run's own mailbox (`_runtime.email_account`, which an email-triggered
occurrence binds to its trigger's mailbox), else the runtime's primary. The mailbox is resolved
through the host's resolver (`Runtime.set_email_context_resolver`, `use="action"`); credentials
never reach vars or results. Messages are named by the `Email` record's `id`
(`"<uidvalidity>:<uid>:<folder>"`, `triggers.email_received.email_id`); a stale id (the server
rebuilt the folder) is refused, never guessed.

Writes (`mark`, `label`, `move`, `delete`, `send`, `reply`, `forward`) are refused on a read-only
mailbox (one without sending settings: `email_read_only`). Sends go through AbstractCore's
`guarded_send` (the account's recipient policy and rate limits apply). `mark`, `label`, `move`,
`delete` and `forward` call AbstractCore's `EmailClient` verbs of the same names; a client without
them answers `mail_verb_unavailable` (typed, never a silent no-op).

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
WRITE_VERBS = frozenset({"mail.mark", "mail.label", "mail.move", "mail.delete", "mail.send", "mail.reply", "mail.forward"})
PRIMARY_MAILBOXES = ("", "self", "default", "primary")
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 500
_QUERY_KEYS = ("from_address", "from_domain", "to_address", "subject_contains", "since", "before", "unseen", "has_attachment")


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
    if not name:
        binding = own or primary
    elif name.lower() in PRIMARY_MAILBOXES:
        binding = primary
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
    resolver = getattr(run, "_runtime_email_resolver", None)
    if resolver is None:
        raise MailVerbError("email_not_configured", "This runtime has no mailbox resolver.",
                            "The host must call Runtime.set_email_context_resolver.")
    try:
        ctx = resolver(binding, use="action")
    except TypeError:
        ctx = resolver(binding)
    if ctx is None:
        raise MailVerbError("email_not_configured", "The mailbox is not connected or not enabled.",
                            "Connect and enable the mailbox in Settings -> Email.")
    label = str(getattr(binding, "address", "") or "") or (account_id_of_ref(binding.account_ref) or name or "self")
    return binding, ctx, label


# --- helpers --------------------------------------------------------------------------------------


def _ids(arguments: Mapping[str, Any], key: str = "ids") -> List[Tuple[str, int, int]]:
    from .triggers.email_received import parse_email_id

    raw = arguments.get(key)
    if isinstance(raw, str):
        raw = [raw]
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


def _require_send(ctx: Any) -> None:
    if not getattr(getattr(ctx, "account", None), "can_send", False):
        raise MailVerbError("email_read_only", "This mailbox is read-only (it has no sending settings).",
                            "Connect the mailbox with its SMTP settings to change or send mail.")


def _client_verb(client: Any, name: str) -> Callable[..., Any]:
    fn = getattr(client, name, None)
    if not callable(fn):
        raise MailVerbError("mail_verb_unavailable", f"This AbstractCore mail client has no {name!r} verb.",
                            "Upgrade AbstractCore to the release that adds EmailClient mailbox-changing verbs.")
    return fn


def _send(ctx: Any, message: Any) -> Dict[str, Any]:
    from .integrations.abstractcore.email_facade import guarded_send

    result = guarded_send(ctx, message)
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
    from .integrations.abstractcore.email_facade import SearchCriteria

    query = args.get("query") or {}
    if not isinstance(query, Mapping):
        raise MailVerbError("invalid_arguments", "query must be an object.", f"Use the fields {list(_QUERY_KEYS)}.")
    unknown = sorted(k for k in query if k not in _QUERY_KEYS)
    if unknown:
        raise MailVerbError("invalid_arguments", f"unknown query field(s) {unknown}.", f"Use the fields {list(_QUERY_KEYS)}.")
    limit = args.get("limit", DEFAULT_SEARCH_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise MailVerbError("invalid_arguments", f"limit must be 1..{MAX_SEARCH_LIMIT}.", "")
    criteria = SearchCriteria.build(**dict(query))
    page = client.search(criteria, folder=args.get("folder") or None, limit=limit)
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
                _client_verb(client, "mark")(uids, folder=folder, seen=args.get("seen"), flagged=args.get("flagged"))
            elif verb == "label":
                add, remove = list(args.get("add") or []), list(args.get("remove") or [])
                if not add and not remove:
                    raise MailVerbError("invalid_arguments", "give labels to add and/or remove.", "")
                _client_verb(client, "label")(uids, folder=folder, add=add, remove=remove)
            elif verb == "move":
                target = str(args.get("folder") or "").strip()
                if not target:
                    raise MailVerbError("invalid_arguments", "folder (the destination) is required.", "")
                _client_verb(client, "move")(uids, folder=folder, to_folder=target)
            elif verb == "delete":
                _client_verb(client, "delete")(uids, folder=folder, permanent=bool(args.get("permanent", False)))
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
    return _send(ctx, message)


def _one_id(args: Mapping[str, Any]) -> Tuple[str, int, int]:
    ids = _ids({"ids": args.get("id")})
    if len(ids) != 1:
        raise MailVerbError("invalid_arguments", "id must name one email.", "")
    return ids[0]


def _reply(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    folder, uidvalidity, uid = _one_id(args)
    _check_uidvalidity(client, folder, [(uidvalidity, uid)])
    message, _original = client.build_reply(uid, text=str(args.get("text") or ""), html=str(args.get("html") or ""),
                                            reply_all=bool(args.get("reply_all", False)), folder=folder)
    return _send(ctx, message)


def _forward(run: Any, args: Mapping[str, Any], client: Any, ctx: Any, mailbox: str) -> Dict[str, Any]:
    folder, uidvalidity, uid = _one_id(args)
    _check_uidvalidity(client, folder, [(uidvalidity, uid)])
    message = _client_verb(client, "forward")(uid, folder=folder, to=list(_addresses(args.get("to"), "to", required=True)),
                                              note=str(args.get("note") or ""))
    return _send(ctx, message)


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
        _require_send(ctx)
    try:
        client = ctx.client()
        if name in ("mail.send",):
            return _VERBS[name](run, args, client, ctx, mailbox)
        with client.session():
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
    "MAIL_VERBS",
    "MailVerbError",
    "WRITE_VERBS",
    "is_mail_verb",
    "mail_verb_outcome",
    "resolve_mailbox",
    "run_mail_verb",
]
