"""Trigger connections: per-user instances of a trigger source (framework backlog 1040).

A trigger SOURCE is a type ("a feed item", "a webhook call", "an email"); a CONNECTION is one
user's instance of it ("this feed URL", "this webhook endpoint and its secret"), registered once
and reused by any number of automations. A binding references its connection by id in its config
(`trigger.config.connection_id`); the adapter stays pure and only checks the id's shape, the host
checks the id names a connection of the same principal and the same source
(`validate_binding_connection`, called by `create_automation` and the `revise` command).

Storage (one store per runtime plane, `TriggerConnectionStore(plane_dir)`):

    <plane_dir>/triggers/connections/<connection_id>.json          0600, atomic replace
    <plane_dir>/triggers/connections/secrets/<connection_id>/secret.enc
                                                                    AES-256-GCM (AbstractCore's
                                                                    SecretVault, the same sealing
                                                                    as the mailbox credentials)

The JSON file never holds a secret: the connection's secret fields (`connection_schema` marks them
`"x-secret": true`; a source may also generate them, e.g. the webhook signing secret) are sealed
with the host's sealing key (`sealing_key_file`; the gateway passes its data folder's
`secrets/sealing.key`, `abstractgateway.security.sealing.sealing_key_path`). `secret_ref` on the
record names the sealed store; `TriggerConnectionStore.secrets()` opens it for the owner.

Principals: every connection belongs to ONE principal, the string `"<tenant>:<user>"`
(`connection_principal`), the same pair an automation request carries. Reads given a principal
answer only that principal's connections; a foreign id reads exactly like an unknown one (no
existence leak). Reads without a principal are the admin view.

Events: each connection has ONE stream in the runtime's durable inbox,
`trigger:<connection_id>` (`connection_stream`). Feeders (feed poller, webhook ingress) append to
it; the automations bound to the connection read only that stream; `test_fire` appends a
synthetic `{data: {test: true}}` event to it, exactly like a real one; recent events are
`EventInbox.read(stream=connection_stream(id), after_seq=...)`.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

try:  # POSIX: serialize writers across processes too
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

from .protocol import TriggerConfigError

CONNECTION_STREAM_PREFIX = "trigger:"
CONNECTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
CONNECTION_STATUSES = ("active", "disabled", "error")
CONNECTION_LABEL_MAX = 120
# Where a source's connection lives: the trigger connection store (feed, webhook) or the mail
# accounts (email.received: its `account` field names a mailbox the mail accounts API owns).
PROVIDER_TRIGGER_CONNECTIONS = "trigger_connections"
PROVIDER_EMAIL_ACCOUNTS = "email_accounts"
CONNECTION_PROVIDERS = (PROVIDER_TRIGGER_CONNECTIONS, PROVIDER_EMAIL_ACCOUNTS)
TEST_EVENT_PREFIX = "test:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    """0600 temp file + fsync + replace (the event inbox's write, not imported: `email` imports
    the trigger package)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))


def connection_stream(connection_id: str) -> str:
    """The connection's inbox stream: `trigger:<connection_id>`."""
    return f"{CONNECTION_STREAM_PREFIX}{check_connection_id(connection_id)}"


def connection_principal(tenant: Any, user: Any) -> str:
    """The owner string of a connection: `"<tenant>:<user>"` (an automation request's pair)."""
    t, u = str(tenant or "local").strip() or "local", str(user or "local").strip() or "local"
    return f"{t}:{u}"


def check_connection_id(value: Any, *, field: str = "connection_id") -> str:
    if not isinstance(value, str) or not CONNECTION_ID_RE.match(value):
        raise TriggerConfigError(
            f"{field} must be a connection id (letters, digits, '_' or '-', at most 64), got {value!r}", field=field
        )
    return value


def new_connection_id() -> str:
    return "tc_" + uuid.uuid4().hex


# --- descriptor helpers ------------------------------------------------------------------------


def source_needs_connection(descriptor: Mapping[str, Any]) -> bool:
    """`descriptor.needs_connection` (a third-party source without the key: False)."""
    return bool((descriptor or {}).get("needs_connection"))


def source_connection_provider(descriptor: Mapping[str, Any]) -> Optional[str]:
    """Who owns the source's connections (`trigger_connections`, `email_accounts`) or None."""
    if not source_needs_connection(descriptor):
        return None
    return str((descriptor or {}).get("connection_provider") or PROVIDER_TRIGGER_CONNECTIONS)


def uses_connection_store(descriptor: Mapping[str, Any]) -> bool:
    return source_connection_provider(descriptor) == PROVIDER_TRIGGER_CONNECTIONS


def secret_fields(schema: Mapping[str, Any]) -> List[str]:
    """The connection-schema properties marked `"x-secret": true` (sealed, never in the JSON)."""
    props = (schema or {}).get("properties") or {}
    return sorted(k for k, v in props.items() if isinstance(v, Mapping) and v.get("x-secret"))


# --- errors ------------------------------------------------------------------------------------


class TriggerConnectionError(ValueError):
    """A connection problem with a wire reason code and the offending field.

    `reason_code`: `connection_not_found` (unknown OR another principal's: indistinguishable),
    `connection_required`, `connection_source_mismatch`, `connection_disabled`,
    `invalid_connection`, `unknown_trigger_source`.
    """

    def __init__(self, message: str, *, reason_code: str, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.field = field

    def to_error(self) -> Dict[str, Any]:
        err: Dict[str, Any] = {"reason_code": self.reason_code, "message": str(self)}
        if self.field:
            err["field"] = self.field
        return err


# --- the record --------------------------------------------------------------------------------


@dataclass
class TriggerConnection:
    """One user's instance of a trigger source (never carries a secret value)."""

    connection_id: str
    principal: str
    source_id: str
    source_version: int
    config: Dict[str, Any]
    label: str = ""
    status: str = "active"
    secret_ref: Optional[str] = None  # the sealed store's name (`secrets/<connection_id>`), None: no secret
    secret_fields: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    last_event_at: Optional[str] = None
    last_error: Optional[Dict[str, Any]] = None
    # The person's switches (independent of `status`): `paused` stops every feeder (nothing is
    # fetched or accepted); `always_listen` keeps a connection fed while no automation uses it.
    paused: bool = False
    always_listen: bool = False

    @property
    def stream(self) -> str:
        return connection_stream(self.connection_id)

    @property
    def source_ref(self) -> str:
        return f"{self.source_id}@{self.source_version}"

    def to_dict(self, *, usage: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The record (no secret values). `usage` (`connection_usage(runtime, conn)`) adds the
        derived `listening` and `used_by` fields."""
        out = asdict(self)
        out["stream"] = self.stream
        if usage is not None:
            out["listening"] = str(usage["listening"])
            out["used_by"] = int(usage["used_by"])
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TriggerConnection":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: copy.deepcopy(v) for k, v in raw.items() if k in known}
        data["source_version"] = int(data.get("source_version") or 0)
        return cls(**data)


def _flag(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TriggerConnectionError(f"{field} must be true or false", reason_code="invalid_connection", field=field)
    return value


LISTENING_STATES = ("paused", "listening", "idle")


def connection_usage(runtime: Any, conn: TriggerConnection) -> Dict[str, Any]:
    """The derived listening state of a connection (for APIs and feeders):

    - `used_by`: how many ACTIVE automations read its stream (`stream_consumers`);
    - `listening`: `"paused"` (the connection is paused or its status is `disabled`: nothing is
      fetched or accepted), `"listening"` (polled / accepting now: used by an active automation, or
      `always_listen`), `"idle"` (nothing uses it: nothing is fetched or stored).
    """
    from ..email.automations import stream_consumers

    used_by = len(stream_consumers(runtime, conn.stream)) if runtime is not None else 0
    if conn.paused or conn.status == "disabled":
        state = "paused"
    elif used_by > 0 or conn.always_listen:
        state = "listening"
    else:
        state = "idle"
    return {"listening": state, "used_by": used_by}


def _default_adapter_lookup(source_id: str, source_version: int) -> Any:
    from .registry import UnknownTriggerSource, get_trigger_adapter

    try:
        return get_trigger_adapter(source_id, source_version)
    except UnknownTriggerSource as exc:
        raise TriggerConnectionError(str(exc), reason_code="unknown_trigger_source", field="source_id") from exc


class TriggerConnectionStore:
    """Trigger connections of one runtime plane (see the module doc for the layout)."""

    def __init__(
        self,
        plane_dir: Union[str, Path],
        *,
        sealing_key_file: Optional[Union[str, Path]] = None,
        adapter_lookup: Optional[Callable[[str, int], Any]] = None,
    ) -> None:
        self.plane_dir = Path(plane_dir)
        self.base_dir = self.plane_dir / "triggers" / "connections"
        self.secrets_dir = self.base_dir / "secrets"
        # The host's sealing key (the gateway: `<data_dir>/secrets/sealing.key`); default: the plane's own.
        self.sealing_key_file = Path(sealing_key_file) if sealing_key_file is not None else self.plane_dir / "secrets" / "sealing.key"
        self._adapter_lookup = adapter_lookup or _default_adapter_lookup
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"TriggerConnectionStore({str(self.plane_dir)!r})"

    # -- locking / files ---------------------------------------------------------------------

    class _Locked:
        def __init__(self, store: "TriggerConnectionStore") -> None:
            self.store = store
            self.fh = None

        def __enter__(self):
            self.store._lock.acquire()
            self.store.base_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.store.base_dir, 0o700)
            except OSError:  # pragma: no cover
                pass
            if fcntl is not None:
                self.fh = open(self.store.base_dir / ".lock", "a+")
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            try:
                if self.fh is not None:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                    self.fh.close()
            finally:
                self.store._lock.release()

    def _path(self, connection_id: str) -> Path:
        return self.base_dir / f"{check_connection_id(connection_id)}.json"

    def _read(self, connection_id: str) -> Optional[TriggerConnection]:
        try:
            raw = json.loads(self._path(connection_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        return TriggerConnection.from_dict(raw)

    def _write(self, conn: TriggerConnection) -> None:
        data = asdict(conn)
        _atomic_write_json(self._path(conn.connection_id), data)

    def _vault(self, connection_id: str):
        from ..integrations.abstractcore.email_facade import SecretVault

        return SecretVault(self.secrets_dir / check_connection_id(connection_id), key_file=self.sealing_key_file)

    # -- validation --------------------------------------------------------------------------

    def _adapter(self, source_id: Any, source_version: Any) -> Any:
        if not isinstance(source_id, str) or not source_id.strip():
            raise TriggerConnectionError("source_id must be a non-empty string", reason_code="invalid_connection", field="source_id")
        if isinstance(source_version, bool) or not isinstance(source_version, int):
            raise TriggerConnectionError("source_version must be an integer", reason_code="invalid_connection", field="source_version")
        adapter = self._adapter_lookup(source_id, source_version)
        descriptor = getattr(adapter, "descriptor", None) or {}
        if not uses_connection_store(descriptor):
            raise TriggerConnectionError(
                f"trigger source {source_id}@{source_version} takes no trigger connection"
                + (" (its connection is a mailbox: use the mail accounts)" if source_connection_provider(descriptor) == PROVIDER_EMAIL_ACCOUNTS else ""),
                reason_code="invalid_connection",
                field="source_id",
            )
        return adapter

    def _validate(self, adapter: Any, config: Any, *, now: str) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TriggerConnectionError("config must be an object", reason_code="invalid_connection", field="config")
        try:
            out = adapter.validate_connection(dict(config), now=now)
        except TriggerConfigError as exc:
            raise TriggerConnectionError(str(exc), reason_code="invalid_connection", field=exc.field) from exc
        json.dumps(out)
        return out

    @staticmethod
    def _label(value: Any, default: str) -> str:
        if value is None:
            return default
        if not isinstance(value, str) or len(value.strip()) > CONNECTION_LABEL_MAX or any(c in value for c in "\r\n"):
            raise TriggerConnectionError(
                f"label must be one line of at most {CONNECTION_LABEL_MAX} characters", reason_code="invalid_connection", field="label"
            )
        return value.strip() or default

    def _split_secrets(self, adapter: Any, config: Dict[str, Any]) -> tuple:
        fields = secret_fields((adapter.descriptor or {}).get("connection_schema") or {})
        plain = {k: v for k, v in config.items() if k not in fields}
        secret = {k: config[k] for k in fields if k in config}
        return plain, secret, fields

    # -- API -------------------------------------------------------------------------------

    def create(
        self,
        *,
        principal: str,
        source_id: str,
        source_version: int,
        config: Mapping[str, Any],
        label: Optional[str] = None,
        connection_id: Optional[str] = None,
        now: Optional[str] = None,
        paused: bool = False,
        always_listen: bool = False,
    ) -> TriggerConnection:
        """Validate and store a new connection; secret fields (and secrets the source generates,
        `adapter.connection_secrets`) are sealed. Returns the record (no secret values: read
        them once with `secrets()` to show the owner, e.g. the webhook signing secret)."""
        if not isinstance(principal, str) or not principal.strip():
            raise TriggerConnectionError("principal must be a non-empty string", reason_code="invalid_connection", field="principal")
        at = now or _now_iso()
        adapter = self._adapter(source_id, source_version)
        normalized = self._validate(adapter, config, now=at)
        plain, secret, fields = self._split_secrets(adapter, normalized)
        make_secrets = getattr(adapter, "connection_secrets", None)
        if callable(make_secrets):
            for k, v in (make_secrets(normalized) or {}).items():
                secret.setdefault(k, v)
        cid = check_connection_id(connection_id) if connection_id is not None else new_connection_id()
        conn = TriggerConnection(
            connection_id=cid,
            principal=principal.strip(),
            source_id=source_id,
            source_version=int(source_version),
            config=plain,
            label=self._label(label, (adapter.descriptor or {}).get("label") or source_id),
            status="active",
            secret_ref=f"secrets/{cid}" if secret else None,
            secret_fields=sorted(secret),
            created_at=at,
            updated_at=at,
            paused=_flag(paused, "paused"),
            always_listen=_flag(always_listen, "always_listen"),
        )
        with self._Locked(self):
            if self._path(cid).exists():
                raise TriggerConnectionError(f"connection {cid!r} already exists", reason_code="invalid_connection", field="connection_id")
            if secret:
                self._vault(cid).store(secret)
            self._write(conn)
        return conn

    def get(self, connection_id: str, *, principal: Optional[str] = None) -> Optional[TriggerConnection]:
        """The connection, or None when unknown or (given `principal`) owned by someone else."""
        try:
            check_connection_id(connection_id)
        except TriggerConfigError:
            return None
        with self._Locked(self):
            conn = self._read(connection_id)
        if conn is None or (principal is not None and conn.principal != principal):
            return None
        return conn

    def require(self, connection_id: str, *, principal: Optional[str] = None) -> TriggerConnection:
        conn = self.get(connection_id, principal=principal)
        if conn is None:
            raise TriggerConnectionError(
                f"no trigger connection {connection_id!r} for this user", reason_code="connection_not_found", field="connection_id"
            )
        return conn

    def list(self, *, principal: Optional[str] = None, source_id: Optional[str] = None) -> List[TriggerConnection]:
        """The principal's connections (None: every principal's, the admin view), oldest first."""
        out: List[TriggerConnection] = []
        with self._Locked(self):
            for path in sorted(self.base_dir.glob("*.json")):
                try:
                    conn = TriggerConnection.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (ValueError, TypeError):
                    continue
                if principal is not None and conn.principal != principal:
                    continue
                if source_id is not None and conn.source_id != source_id:
                    continue
                out.append(conn)
        return sorted(out, key=lambda c: (c.created_at, c.connection_id))

    def update(
        self,
        connection_id: str,
        *,
        principal: Optional[str],
        config: Optional[Mapping[str, Any]] = None,
        label: Optional[str] = None,
        status: Optional[str] = None,
        now: Optional[str] = None,
        paused: Optional[bool] = None,
        always_listen: Optional[bool] = None,
    ) -> TriggerConnection:
        """Replace the config (validated whole; secret fields left out keep their sealed value),
        the label, the status (`active` / `disabled` / `error`), `paused` and/or `always_listen`."""
        at = now or _now_iso()
        with self._Locked(self):
            conn = self._read(connection_id) if CONNECTION_ID_RE.match(str(connection_id)) else None
            if conn is None or (principal is not None and conn.principal != principal):
                raise TriggerConnectionError(
                    f"no trigger connection {connection_id!r} for this user", reason_code="connection_not_found", field="connection_id"
                )
            adapter = self._adapter(conn.source_id, conn.source_version)
            if config is not None:
                existing_secret = self._vault(conn.connection_id).load() if conn.secret_ref else {}
                merged = {**{k: v for k, v in (existing_secret or {}).items() if k in secret_fields(adapter.descriptor.get("connection_schema") or {})}, **dict(config)}
                normalized = self._validate(adapter, merged, now=at)
                plain, secret, _fields = self._split_secrets(adapter, normalized)
                # Generated secrets (not schema fields) are kept as they are.
                for k, v in (existing_secret or {}).items():
                    secret.setdefault(k, v)
                if secret:
                    self._vault(conn.connection_id).store(secret)
                conn.config = plain
                conn.secret_ref = f"secrets/{conn.connection_id}" if secret else None
                conn.secret_fields = sorted(secret)
            if label is not None:
                conn.label = self._label(label, conn.label)
            if status is not None:
                if status not in CONNECTION_STATUSES:
                    raise TriggerConnectionError(
                        f"status must be one of {list(CONNECTION_STATUSES)}", reason_code="invalid_connection", field="status"
                    )
                conn.status = status
            if paused is not None:
                conn.paused = _flag(paused, "paused")
            if always_listen is not None:
                conn.always_listen = _flag(always_listen, "always_listen")
            conn.updated_at = at
            self._write(conn)
        return conn

    def delete(self, connection_id: str, *, principal: Optional[str]) -> bool:
        """Remove the connection and its sealed secrets; False when unknown / foreign. The
        inbox events stay (retention removes them) so the history of past occurrences reads."""
        with self._Locked(self):
            conn = self._read(connection_id) if CONNECTION_ID_RE.match(str(connection_id)) else None
            if conn is None or (principal is not None and conn.principal != principal):
                return False
            if conn.secret_ref:
                self._vault(conn.connection_id).delete()
                try:
                    (self.secrets_dir / conn.connection_id).rmdir()
                except OSError:
                    pass
            self._path(conn.connection_id).unlink()
        return True

    def secrets(self, connection_id: str, *, principal: Optional[str]) -> Dict[str, Any]:
        """The connection's sealed secret values (owner reveal, signature checks); {} when none."""
        conn = self.require(connection_id, principal=principal)
        if not conn.secret_ref:
            return {}
        return dict(self._vault(conn.connection_id).load() or {})

    def record_event(self, connection_id: str, *, at: Optional[str] = None, error: Optional[Dict[str, Any]] = None) -> None:
        """Feeder bookkeeping: `last_event_at` (an event appended) or `last_error` (a failed poll)."""
        with self._Locked(self):
            conn = self._read(connection_id)
            if conn is None:
                return
            if error is not None:
                conn.last_error = dict(error)
            else:
                conn.last_event_at = at or _now_iso()
                conn.last_error = None
            self._write(conn)


    def clear_error(self, connection_id: str) -> None:
        """Feeder bookkeeping: a poll succeeded, so the last failure no longer stands."""
        with self._Locked(self):
            conn = self._read(connection_id)
            if conn is None or conn.last_error is None:
                return
            conn.last_error = None
            self._write(conn)


# --- binding validation (host side; adapters stay pure) ---------------------------------------


def binding_connection_id(binding: Mapping[str, Any]) -> Optional[str]:
    config = (binding or {}).get("config") or {}
    cid = config.get("connection_id") if isinstance(config, Mapping) else None
    return str(cid) if cid is not None else None


def validate_binding_connection(
    principal: str,
    binding: Mapping[str, Any],
    store: Optional[TriggerConnectionStore],
    *,
    adapter: Any = None,
) -> Optional[TriggerConnection]:
    """Check a binding's `config.connection_id` at automation create / revise time.

    Returns the connection (None for a source that takes no trigger connection). Raises
    `TriggerConnectionError`: `connection_required` (missing id, or no store on this host),
    `connection_not_found` (unknown or another principal's), `connection_source_mismatch`,
    `connection_disabled`. `binding` is a TriggerBinding or a trigger request
    (`{source_id, source_version, config}`).
    """
    source_id, version = binding.get("source_id"), binding.get("source_version")
    if adapter is None:
        adapter = _default_adapter_lookup(str(source_id), int(version))
    descriptor = getattr(adapter, "descriptor", None) or {}
    if not uses_connection_store(descriptor):
        return None
    field = "trigger.config.connection_id"
    cid = binding_connection_id(binding)
    if not cid:
        raise TriggerConnectionError(
            f"trigger {source_id}@{version} needs a connection: set config.connection_id", reason_code="connection_required", field=field
        )
    if store is None:
        raise TriggerConnectionError(
            f"trigger {source_id}@{version} reads a trigger connection, and this host has no connection store "
            "(the host must call Runtime.set_trigger_connection_store)",
            reason_code="connection_required",
            field=field,
        )
    conn = store.get(cid, principal=principal)
    if conn is None:
        raise TriggerConnectionError(f"no trigger connection {cid!r} for this user", reason_code="connection_not_found", field=field)
    if conn.source_id != source_id or int(conn.source_version) != int(version):
        raise TriggerConnectionError(
            f"connection {cid!r} is a {conn.source_ref} connection, not {source_id}@{version}",
            reason_code="connection_source_mismatch",
            field=field,
        )
    if conn.status == "disabled":
        raise TriggerConnectionError(f"connection {cid!r} is disabled", reason_code="connection_disabled", field=field)
    return conn


# --- events on the connection's stream -------------------------------------------------------


def connection_event_type(source_id: str, source_version: int) -> str:
    """CloudEvents `type` of a source's events: `abstract.<source_id>.v<version>`."""
    from .cloudevents import event_type

    return event_type(source_id, source_version)


def connection_source_uri(connection_id: str) -> str:
    """CloudEvents `source` of a connection's events."""
    return f"abstract://triggers/connections/{connection_id}"


def connection_event(
    conn: TriggerConnection,
    *,
    event_id: str,
    data: Mapping[str, Any],
    time: Optional[str] = None,
    subject: Optional[str] = None,
    dedupekey: Optional[str] = None,
    trust: str = "untrusted",
) -> Dict[str, Any]:
    """The inbox payload of one connection event (CloudEvents 1.0 attribute names)."""
    at = time or _now_iso()
    out: Dict[str, Any] = {
        "kind": "trigger.connection_event",
        "specversion": "1.0",
        "id": str(event_id),
        "type": connection_event_type(conn.source_id, conn.source_version),
        "source": connection_source_uri(conn.connection_id),
        "time": at,
        "datacontenttype": "application/json",
        "connectionid": conn.connection_id,
        "abstracttrust": trust,
        "data": json.loads(json.dumps(dict(data))),
    }
    if subject is not None:
        out["subject"] = str(subject)
    if dedupekey is not None:
        out["dedupekey"] = str(dedupekey)
    return out


def append_connection_event(
    inbox: Any,
    conn: TriggerConnection,
    *,
    event_id: str,
    data: Mapping[str, Any],
    time: Optional[str] = None,
    subject: Optional[str] = None,
    dedupekey: Optional[str] = None,
    trust: str = "untrusted",
    store: Optional[TriggerConnectionStore] = None,
) -> Any:
    """Append one event to the connection's stream (deduplicated on `dedupekey` when given).

    Returns the inbox `AppendResult`, or None when the stream already holds `dedupekey`.
    `store` (optional) records `last_event_at` on the connection."""
    stream = conn.stream
    if dedupekey and inbox.has_dedupe_key(stream, dedupekey):
        return None
    payload = connection_event(conn, event_id=event_id, data=data, time=time, subject=subject, dedupekey=dedupekey, trust=trust)
    result = inbox.append(stream=stream, event_id=f"{stream}:{event_id}", payload=payload, dedupe_key=dedupekey)
    if store is not None and not result.duplicate:
        store.record_event(conn.connection_id, at=payload["time"])
    return result


def test_fire(
    store: TriggerConnectionStore,
    inbox: Any,
    connection_id: str,
    *,
    principal: Optional[str] = None,
    now: Optional[str] = None,
) -> Any:
    """Append a synthetic `{data: {test: true}}` event to the connection's stream (the console's
    "test" button). Bound automations admit it like a real event; the host then wakes them
    (`wake_connection_automations`). Returns the inbox `AppendResult`."""
    conn = store.require(connection_id, principal=principal)
    at = now or _now_iso()
    event_id = f"{TEST_EVENT_PREFIX}{uuid.uuid4().hex}"
    return append_connection_event(
        inbox, conn, event_id=event_id, data={"test": True}, time=at, subject="test", trust="trusted", store=store
    )


test_fire.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


# --- consumers --------------------------------------------------------------------------------


def connection_trigger_consumers(runtime: Any, *, connection_id: Optional[str] = None) -> List[str]:
    """Ids of active automations bound to a trigger connection (to `connection_id` when given)."""
    from ..automations.ledger import definition_of
    from ..automations.models import automation_status

    out: List[str] = []
    for row in runtime.run_store.list_run_index(limit=1_000_000, role="controller"):
        run = runtime.run_store.load(str(row.get("run_id")))
        if run is None:
            continue
        try:
            trigger = definition_of(run)["trigger"]
        except (LookupError, KeyError, TypeError):
            continue
        cid = binding_connection_id(trigger)
        if not cid or (connection_id is not None and cid != connection_id):
            continue
        if automation_status(run) == "active":
            out.append(run.run_id)
    return sorted(out)


def wake_connection_automations(runtime: Any, connection_id: str) -> List[str]:
    """After a feeder appended to `trigger:<connection_id>`, resume the idle controllers bound to
    it so they read the stream now (a missed wake only delays delivery). Returns the woken ids."""
    from ..automations.bundle import controller_workflow_spec
    from ..automations.models import wake_wait_key
    from ..core.models import RunStatus
    from ..core.runtime import StaleResumeError

    woken: List[str] = []
    for automation_id in connection_trigger_consumers(runtime, connection_id=connection_id):
        run = runtime.run_store.load(automation_id)
        if run is None or run.status != RunStatus.WAITING or run.waiting is None:
            continue
        if run.waiting.wait_key != wake_wait_key(automation_id):
            continue
        try:
            runtime.resume(
                workflow=controller_workflow_spec(),
                run_id=automation_id,
                wait_key=run.waiting.wait_key,
                payload={"wake": connection_stream(connection_id)},
                max_steps=0,
            )
        except StaleResumeError:
            continue
        woken.append(automation_id)
    return woken


__all__ = [
    "CONNECTION_PROVIDERS",
    "CONNECTION_STATUSES",
    "CONNECTION_STREAM_PREFIX",
    "PROVIDER_EMAIL_ACCOUNTS",
    "PROVIDER_TRIGGER_CONNECTIONS",
    "TriggerConnection",
    "TriggerConnectionError",
    "TriggerConnectionStore",
    "append_connection_event",
    "binding_connection_id",
    "LISTENING_STATES",
    "check_connection_id",
    "connection_event",
    "connection_usage",
    "connection_event_type",
    "connection_principal",
    "connection_source_uri",
    "connection_stream",
    "connection_trigger_consumers",
    "new_connection_id",
    "secret_fields",
    "source_connection_provider",
    "source_needs_connection",
    "test_fire",
    "uses_connection_store",
    "validate_binding_connection",
    "wake_connection_automations",
]
