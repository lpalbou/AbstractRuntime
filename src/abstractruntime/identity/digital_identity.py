"""An entity's digital identity: its look and its voice, one record in its home (ADR 0052).

`<home>/identity.json` holds the avatar design and colour the apps draw for the entity and the
voice they speak it with. The runtime owns the format and the writer; the gateway checks what only
it can check (the design is in the avatar catalog it loaded, the voice is offered and downloaded, the
home is not held) and then calls `write_identity`. Apps read through the gateway.

What this module guarantees:

- One shape, `schema_version` 1 (ADR 0052 section 1). A file this module cannot read in full (a
  newer schema, a corrupt or partial file, an unknown field, a value out of bounds) raises
  `IdentityUnreadable` on read AND on write; it is never treated as "no identity".
- `rev` starts at 1 and grows by one per write, never reused, not even after a write that failed
  half-way (the next rev is above both the file's and the biography's newest rev), and not when
  two writers race: every write holds `<home>/.identity.lock` from its read to its file replace.
- Every write appends one `identity_changed` row to `<home>/state_history.jsonl` BEFORE the file
  moves (the `phase_changed` file and row shape), so the home itself remembers every look and
  voice it ever had; `identity_at_rev` answers "which voice was this at rev N" from those rows.
- The recorded default (section 2) is a pure function of the entity id and the gateway's default
  voice: `default_identity`. `record_default_identity` writes it as rev 1 once, and never
  overwrites a record.
- The reference (section 3) is `{"of": <subject>, "rev": <int>}`, string form `entity:castor#3`;
  on a run it rides `vars["_runtime"]["identity_ref"]`, beside the signed entity stamp, never
  inside it.
- The fingerprint (section 4) is derived from the manifest and the diary's chain head, never stored.
- The summon cue (section 7) is one sentence about the newest change, for the summon prelude.

No flags and no environment variables: the home and the caller's arguments are the only inputs.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from .fingerprint import ActorFingerprint

__all__ = [
    "AVATAR_SOURCES",
    "AvatarChoice",
    "BIOGRAPHY_FILENAME",
    "DEFAULT_STYLE_POOL",
    "DigitalIdentity",
    "IDENTITY_CHANGED_MARKER",
    "IDENTITY_FILENAME",
    "IDENTITY_LOCK_FILENAME",
    "IDENTITY_MAX_BYTES",
    "IDENTITY_REF_KEY",
    "IDENTITY_SCHEMA_VERSION",
    "IdentityFingerprint",
    "IdentityInvalid",
    "IdentityRef",
    "IdentityUnreadable",
    "RECORDED_DEFAULT_CAUSE",
    "VoiceChoice",
    "default_identity",
    "identity_at_rev",
    "identity_change_cue",
    "identity_fingerprint",
    "identity_ref",
    "parse_identity_ref",
    "read_identity",
    "read_run_identity_ref",
    "record_default_identity",
    "stamp_run_identity_ref",
    "write_identity",
]

IDENTITY_FILENAME = "identity.json"
# The writers' lock: one write at a time per home, across threads and processes. The file is
# never removed (removing it would let two writers lock two different files).
IDENTITY_LOCK_FILENAME = ".identity.lock"
IDENTITY_SCHEMA_VERSION = 1
IDENTITY_MAX_BYTES = 4096
BIOGRAPHY_FILENAME = "state_history.jsonl"
IDENTITY_CHANGED_MARKER = "identity_changed"
# Where a run names the identity it started with: vars["_runtime"]["identity_ref"]. Beside the
# signed entity stamp (vars["_runtime"]["entity"]), never inside it: presentation is not authority.
IDENTITY_REF_KEY = "identity_ref"

# The eight engraved orb designs of the recorded default, in this order (ADR 0052 section 2).
# The order is part of the rule: changing it changes every default ever computed.
DEFAULT_STYLE_POOL: Tuple[str, ...] = (
    "watch-halo",
    "watch-vortex",
    "watch-nebula",
    "watch-rings",
    "watch-galaxy",
    "watch-lotus",
    "watch-crystal",
    "watch-aurora",
)
# The biography row's `cause` when the recorded default is written.
RECORDED_DEFAULT_CAUSE = "recorded_default"
DEFAULT_SATURATION = 0.62
DEFAULT_LIGHTNESS = 0.52

AVATAR_SOURCES = ("default", "chosen")
_TOP_FIELDS = ("schema_version", "rev", "avatar", "voice", "set_at", "set_by")
_AVATAR_FIELDS = ("style", "color", "source")
_VOICE_REQUIRED = ("provider", "model", "voice", "source")
_VOICE_OPTIONAL = ("speed", "quality_preset")
_VOICE_TEXT_MAX = 128
_PRESET_MAX = 32
_STYLE_MAX = 64
_SET_BY_MAX = 128
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
_HEX_LOWER = "0123456789abcdef"


class IdentityUnreadable(ValueError):
    """`identity.json` exists but cannot be read in full: a newer schema, a corrupt or partial
    file, an unknown field or a value out of bounds. Never read as "no identity"."""


class IdentityInvalid(ValueError):
    """A value given to the writer is out of the ADR 0052 bounds. `field` names it
    (`avatar.color`, `voice.speed`, ...)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field


# --------------------------------------------------------------------------- the record


@dataclass(frozen=True)
class AvatarChoice:
    """The design (an abstractavatar style id), its colour (`#rrggbb`) and whether it was chosen
    or is the recorded default. Whether the design exists in the avatar catalog and can be drawn
    on the Watch is checked by the gateway, which loads the catalog."""

    style: str
    color: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"style": self.style, "color": self.color, "source": self.source}


@dataclass(frozen=True)
class VoiceChoice:
    """The full voice triple (a voice id means something only to its provider and model), with
    an optional speed and quality preset, and whether it was chosen or is the recorded default."""

    provider: str
    model: str
    voice: str
    source: str
    speed: Optional[float] = None
    quality_preset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"provider": self.provider, "model": self.model, "voice": self.voice}
        if self.speed is not None:
            out["speed"] = self.speed
        if self.quality_preset is not None:
            out["quality_preset"] = self.quality_preset
        out["source"] = self.source
        return out


@dataclass(frozen=True)
class DigitalIdentity:
    """One entity's look and voice at one revision. `voice` is None only on a record the
    default pass wrote while the gateway had no default voice (the entity then has no voice and
    says so; it never borrows one)."""

    rev: int
    avatar: AvatarChoice
    voice: Optional[VoiceChoice]
    set_at: str
    set_by: str
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rev": self.rev,
            "avatar": self.avatar.to_dict(),
            "voice": self.voice.to_dict() if self.voice is not None else None,
            "set_at": self.set_at,
            "set_by": self.set_by,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "DigitalIdentity":
        """Strict parse of a stored record: every field known, present and in bounds, or
        `IdentityInvalid` naming the first one that is not."""
        if not isinstance(data, Mapping):
            raise IdentityInvalid("identity", "must be a JSON object")
        _refuse_unknown(data, _TOP_FIELDS, "")
        version = data.get("schema_version")
        if not _is_int(version):
            raise IdentityInvalid("schema_version", f"must be the integer {IDENTITY_SCHEMA_VERSION}")
        if version > IDENTITY_SCHEMA_VERSION:
            raise IdentityInvalid(
                "schema_version",
                f"{version} is newer than this runtime knows ({IDENTITY_SCHEMA_VERSION}); upgrade AbstractRuntime",
            )
        if version != IDENTITY_SCHEMA_VERSION:
            raise IdentityInvalid("schema_version", f"must be {IDENTITY_SCHEMA_VERSION}, not {version!r}")
        for name in _TOP_FIELDS:
            if name not in data:
                raise IdentityInvalid(name, "is missing")
        record = cls(
            rev=data["rev"],
            avatar=_avatar_from(data["avatar"]),
            voice=_voice_from(data["voice"]),
            set_at=data["set_at"],
            set_by=data["set_by"],
        )
        _check_record(record)
        return record


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _refuse_unknown(data: Mapping[str, Any], known: Tuple[str, ...], prefix: str) -> None:
    for key in data:
        if key not in known:
            raise IdentityInvalid(f"{prefix}{key}", "is not a field of identity schema 1")


def _avatar_from(data: Any) -> AvatarChoice:
    if not isinstance(data, Mapping):
        raise IdentityInvalid("avatar", "must be an object with style, color and source")
    _refuse_unknown(data, _AVATAR_FIELDS, "avatar.")
    for name in _AVATAR_FIELDS:
        if name not in data:
            raise IdentityInvalid(f"avatar.{name}", "is missing")
    avatar = AvatarChoice(style=data["style"], color=data["color"], source=data["source"])
    _check_avatar(avatar)
    return avatar


def _voice_from(data: Any) -> Optional[VoiceChoice]:
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise IdentityInvalid("voice", "must be an object (provider, model, voice, source) or null")
    _refuse_unknown(data, _VOICE_REQUIRED + _VOICE_OPTIONAL, "voice.")
    for name in _VOICE_REQUIRED:
        if name not in data:
            raise IdentityInvalid(f"voice.{name}", "is missing")
    voice = VoiceChoice(
        provider=data["provider"],
        model=data["model"],
        voice=data["voice"],
        source=data["source"],
        speed=data.get("speed"),
        quality_preset=data.get("quality_preset"),
    )
    _check_voice(voice)
    return voice


def _check_style(style: Any) -> None:
    # abstractavatar ids: a lowercase letter, then up to 63 lowercase letters, digits or dashes.
    ok = (
        isinstance(style, str)
        and 1 <= len(style) <= _STYLE_MAX
        and style[0] in _LOWER
        and all(ch in _LOWER or ch in _DIGITS or ch == "-" for ch in style)
    )
    if not ok:
        raise IdentityInvalid(
            "avatar.style",
            f"must be an avatar design id (a lowercase letter, then up to {_STYLE_MAX - 1} lowercase "
            f"letters, digits or dashes), not {style!r}",
        )


def _check_color(color: Any) -> None:
    ok = isinstance(color, str) and len(color) == 7 and color[0] == "#" and all(ch in _HEX_LOWER for ch in color[1:])
    if not ok:
        raise IdentityInvalid("avatar.color", f"must be #rrggbb in lowercase hex, not {color!r}")


def _check_source(source: Any, field: str) -> None:
    if source not in AVATAR_SOURCES:
        raise IdentityInvalid(field, f"must be one of {', '.join(AVATAR_SOURCES)}, not {source!r}")


def _check_avatar(avatar: AvatarChoice) -> None:
    _check_style(avatar.style)
    _check_color(avatar.color)
    _check_source(avatar.source, "avatar.source")


def _check_voice_text(value: Any, field: str, limit: int) -> None:
    ok = isinstance(value, str) and 0 < len(value) <= limit and value.isprintable() and value.strip() == value
    if not ok:
        raise IdentityInvalid(
            field, f"must be non-empty printable text of at most {limit} characters, without surrounding spaces, not {value!r}"
        )


def _check_voice(voice: VoiceChoice) -> None:
    _check_voice_text(voice.provider, "voice.provider", _VOICE_TEXT_MAX)
    _check_voice_text(voice.model, "voice.model", _VOICE_TEXT_MAX)
    _check_voice_text(voice.voice, "voice.voice", _VOICE_TEXT_MAX)
    if voice.speed is not None:
        speed = voice.speed
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not (0 < float(speed) <= 4):
            raise IdentityInvalid("voice.speed", f"must be a number above 0 and at most 4, not {speed!r}")
    if voice.quality_preset is not None:
        _check_voice_text(voice.quality_preset, "voice.quality_preset", _PRESET_MAX)
    _check_source(voice.source, "voice.source")


def _parse_aware_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise IdentityInvalid(field, f"must be an ISO-8601 time in UTC, not {value!r}")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise IdentityInvalid(field, f"must be an ISO-8601 time in UTC, not {value!r}") from None
    if moment.tzinfo is None or moment.utcoffset() != timezone.utc.utcoffset(None):
        raise IdentityInvalid(field, f"must carry the UTC offset (+00:00), not {value!r}")
    return moment


def _check_set_by(set_by: Any) -> None:
    if set_by in ("operator", "default"):
        return
    ok = False
    if isinstance(set_by, str) and len(set_by) <= _SET_BY_MAX and set_by.count(":") == 1:
        namespace, name = set_by.split(":")
        ok = bool(namespace) and bool(name) and set_by.isprintable() and not any(ch.isspace() for ch in set_by)
    if not ok:
        raise IdentityInvalid(
            "set_by", f"must be namespace:name (for example user:laurent), operator, or default, not {set_by!r}"
        )


def _check_record(record: DigitalIdentity) -> None:
    if not _is_int(record.rev) or record.rev < 1:
        raise IdentityInvalid("rev", f"must be an integer of at least 1, not {record.rev!r}")
    _check_avatar(record.avatar)
    if record.voice is not None:
        _check_voice(record.voice)
    _parse_aware_utc(record.set_at, "set_at")
    _check_set_by(record.set_by)
    if record.voice is None and not (record.set_by == "default" and record.avatar.source == "default"):
        raise IdentityInvalid(
            "voice",
            "may be null only on the recorded default (set_by default, avatar.source default) of a gateway "
            "with no default voice; a chosen identity names its voice",
        )


# --------------------------------------------------------------------------- reading


def _identity_path(home_dir: Path) -> Path:
    return Path(home_dir) / IDENTITY_FILENAME


def read_identity(home_dir: Path) -> Optional[DigitalIdentity]:
    """The home's identity, or None when it has no `identity.json` (a legacy home: its
    `voice.yaml` and `appearance.json` are the gateway's to read). Any file that exists but
    cannot be read in full raises `IdentityUnreadable` naming the reason."""
    path = _identity_path(home_dir)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise IdentityUnreadable(f"{path} cannot be read: {e}") from e
    if len(raw) > IDENTITY_MAX_BYTES:
        raise IdentityUnreadable(f"{path} is {len(raw)} bytes; an identity is at most {IDENTITY_MAX_BYTES}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise IdentityUnreadable(f"{path} is not valid JSON: {e}") from e
    try:
        return DigitalIdentity.from_dict(data)
    except IdentityInvalid as e:
        raise IdentityUnreadable(f"{path}: {e}") from e


def _biography_rows(home_dir: Path) -> List[Dict[str, Any]]:
    """The `identity_changed` rows of the home's biography, oldest first. A line that is not
    JSON is skipped (one bad line never hides the rest; `life_sleep_stats` reads the same way)."""
    path = Path(home_dir) / BIOGRAPHY_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("marker") == IDENTITY_CHANGED_MARKER and _is_int(row.get("rev")):
            rows.append(row)
    return rows


def identity_at_rev(home_dir: Path, rev: int) -> DigitalIdentity:
    """The identity as it was at `rev`: the current file when it is that rev, otherwise the
    record the biography row of that rev wrote — provided a later write moved on from it (a row
    whose file never landed was never in effect). Raises `LookupError` for any other rev."""
    if not _is_int(rev) or rev < 1:
        raise LookupError(f"identity rev must be an integer of at least 1, not {rev!r}")
    current = read_identity(home_dir)
    if current is not None and current.rev == rev:
        return current
    rows = _biography_rows(home_dir)
    moved_on = {
        row["from"].get("rev")
        for row in rows
        if isinstance(row.get("from"), dict)
    }
    for row in reversed(rows):
        if row["rev"] == rev and rev in moved_on:
            try:
                return DigitalIdentity.from_dict(row.get("to"))
            except IdentityInvalid as e:
                raise IdentityUnreadable(f"{BIOGRAPHY_FILENAME} row for identity rev {rev}: {e}") from e
    raise LookupError(f"{Path(home_dir).name} has no identity rev {rev}")


# --------------------------------------------------------------------------- writing


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _identity_write_lock(home_dir: Path) -> Iterator[None]:
    """Hold `<home>/.identity.lock` exclusively, waiting for any other holder. The lock belongs
    to the open file, so two threads of one process exclude each other as two processes do,
    and a writer that dies releases it."""
    with (Path(home_dir) / IDENTITY_LOCK_FILENAME).open("a+b") as fh:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            # Blocks, retrying for about 10 seconds, then raises OSError: loud, never unlocked.
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _append_biography_row(home_dir: Path, row: Dict[str, Any]) -> None:
    """Append one row and flush it to disk. Unlike `phase_changed` this is NOT best-effort: the
    identity file never moves unless its biography row is on disk first."""
    with (Path(home_dir) / BIOGRAPHY_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_identity(
    home_dir: Path,
    *,
    avatar: AvatarChoice,
    voice: Optional[VoiceChoice],
    set_by: str,
    cause: str,
    provenance: str,
    check: Optional[Callable[[DigitalIdentity], None]] = None,
) -> DigitalIdentity:
    """Record a new identity for the home and return it.

    The biography row has the `phase_changed` field names, so one reader handles both:
    `{marker: identity_changed, rev, from, to, written_by, cause, at, provenance}` — `from` and
    `to` are whole records (`from` is None for the first one), `written_by` is `set_by`, `cause`
    says why (the caller's word, for example `chosen` or `recorded_default`) and `provenance`
    which door or pass wrote it.

    In order: the current file is read (an unreadable one refuses: overwriting it would lose
    what a newer writer meant); the new record gets the next rev and `set_at` from this clock;
    every bound is checked (`IdentityInvalid` naming the field); `check(record)` runs, the
    gateway's hook for what only it knows (catalog, offered and downloaded voices) — it raises
    to refuse; the `identity_changed` row is appended to the biography; then the file is
    replaced atomically. A refusal at any step before the row writes nothing.

    The whole sequence holds the home's identity lock (`.identity.lock`), so writers in other
    threads or processes wait their turn: two writers never read the same rev, and a rev is
    never reused. `check` runs under the lock; keep it short."""
    home_dir = Path(home_dir)
    if not home_dir.is_dir():
        raise FileNotFoundError(f"no entity home at {home_dir}")
    with _identity_write_lock(home_dir):
        return _write_identity_locked(
            home_dir, avatar=avatar, voice=voice, set_by=set_by, cause=cause, provenance=provenance, check=check
        )


def _write_identity_locked(
    home_dir: Path,
    *,
    avatar: AvatarChoice,
    voice: Optional[VoiceChoice],
    set_by: str,
    cause: str,
    provenance: str,
    check: Optional[Callable[[DigitalIdentity], None]],
) -> DigitalIdentity:
    """`write_identity`'s steps; the caller holds `_identity_write_lock(home_dir)`."""
    if not isinstance(avatar, AvatarChoice):
        raise IdentityInvalid("avatar", "must be an AvatarChoice")
    if voice is not None and not isinstance(voice, VoiceChoice):
        raise IdentityInvalid("voice", "must be a VoiceChoice or None")
    cause_text = str(cause or "").strip()
    if not cause_text:
        raise IdentityInvalid("cause", "is required: say why the identity changed")
    provenance_text = str(provenance or "").strip()
    if not provenance_text:
        raise IdentityInvalid("provenance", "is required: say which door or pass wrote this identity")

    previous = read_identity(home_dir)
    newest_row_rev = max((row["rev"] for row in _biography_rows(home_dir)), default=0)
    rev = max(previous.rev if previous is not None else 0, newest_row_rev) + 1
    set_at = _utc_now().isoformat()
    record = DigitalIdentity(rev=rev, avatar=avatar, voice=voice, set_at=set_at, set_by=set_by)
    _check_record(record)
    text = record.to_json()
    if len(text.encode("utf-8")) > IDENTITY_MAX_BYTES:
        raise IdentityInvalid("identity", f"would be larger than {IDENTITY_MAX_BYTES} bytes")
    if check is not None:
        check(record)

    _append_biography_row(
        home_dir,
        {
            "marker": IDENTITY_CHANGED_MARKER,
            "rev": rev,
            "from": previous.to_dict() if previous is not None else None,
            "to": record.to_dict(),
            "written_by": set_by,
            "cause": cause_text,
            "at": set_at,
            "provenance": provenance_text,
        },
    )
    from ..utils.atomic_files import atomic_write_text

    atomic_write_text(_identity_path(home_dir), text)
    return record


# --------------------------------------------------------------------------- the recorded default


def _hsl_hex(hue_degrees: int, saturation: float, lightness: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue_degrees / 360.0, lightness, saturation)
    # Each channel = floor(c * 255 + 0.5): round half up, the same in every language.
    return "#" + "".join(f"{int(channel * 255 + 0.5):02x}" for channel in (red, green, blue))


def default_identity(
    entity_id: str, gateway_default_voice: Optional[Mapping[str, Any]]
) -> Tuple[AvatarChoice, Optional[VoiceChoice]]:
    """The recorded default look and voice for an entity (ADR 0052 section 2). Pure: no clock,
    no randomness, the same answer for the same inputs everywhere.

    - seed = SHA-256 of the entity id exactly as its manifest records it (`entity:castor` or
      the older `entity:castor@<home_id>`);
    - design = `DEFAULT_STYLE_POOL[seed[0] % 8]`;
    - colour = HSL(hue = (seed[1] * 256 + seed[2]) % 360, saturation 62 %, lightness 52 %) as
      `#rrggbb`, each channel rounded half up;
    - voice = the gateway's default voice `{provider, model, voice, speed?, quality_preset?}` as a
      value, or None when the gateway has none.
    Both parts carry `source: default`."""
    eid = str(entity_id or "").strip()
    if not eid:
        raise IdentityInvalid("entity_id", "is required to compute a default identity")
    seed = hashlib.sha256(eid.encode("utf-8")).digest()
    avatar = AvatarChoice(
        style=DEFAULT_STYLE_POOL[seed[0] % len(DEFAULT_STYLE_POOL)],
        color=_hsl_hex(((seed[1] << 8) | seed[2]) % 360, DEFAULT_SATURATION, DEFAULT_LIGHTNESS),
        source="default",
    )
    voice: Optional[VoiceChoice] = None
    if gateway_default_voice is not None:
        if not isinstance(gateway_default_voice, Mapping):
            raise IdentityInvalid("voice", "the gateway default voice must be an object or None")
        _refuse_unknown(gateway_default_voice, ("provider", "model", "voice") + _VOICE_OPTIONAL, "voice.")
        voice = VoiceChoice(
            provider=gateway_default_voice.get("provider"),
            model=gateway_default_voice.get("model"),
            voice=gateway_default_voice.get("voice"),
            source="default",
            speed=gateway_default_voice.get("speed"),
            quality_preset=gateway_default_voice.get("quality_preset"),
        )
        _check_voice(voice)
    return avatar, voice


def record_default_identity(
    home_dir: Path,
    *,
    entity_id: str,
    gateway_default_voice: Optional[Mapping[str, Any]],
    provenance: str,
    check: Optional[Callable[[DigitalIdentity], None]] = None,
) -> Tuple[DigitalIdentity, bool]:
    """Write the recorded default as the home's first identity, once. Returns (record, written):
    a home that already has a record keeps it untouched (`written` False) — a later change of the
    gateway's default voice never reaches a recorded identity. `set_by` is `default` and the
    biography row's `cause` is `recorded_default`. The look-then-write holds the home's identity
    lock, so two passes racing on one home write it once."""
    avatar, voice = default_identity(entity_id, gateway_default_voice)
    home_dir = Path(home_dir)
    if not home_dir.is_dir():
        raise FileNotFoundError(f"no entity home at {home_dir}")
    with _identity_write_lock(home_dir):
        existing = read_identity(home_dir)
        if existing is not None:
            return existing, False
        record = _write_identity_locked(
            home_dir,
            avatar=avatar,
            voice=voice,
            set_by="default",
            cause=RECORDED_DEFAULT_CAUSE,
            provenance=provenance,
            check=check,
        )
    return record, True


# --------------------------------------------------------------------------- the reference


@dataclass(frozen=True)
class IdentityRef:
    """Names one identity at one revision: `of` is the subject (`entity:castor`, or
    `session:<id>` for a regular agent), `rev` the revision. `str(ref)` is `entity:castor#3`."""

    of: str
    rev: int

    def __post_init__(self) -> None:
        _check_subject(self.of)
        if not _is_int(self.rev) or self.rev < 1:
            raise ValueError(f"an identity ref's rev must be an integer of at least 1, not {self.rev!r}")

    def __str__(self) -> str:
        return f"{self.of}#{self.rev}"

    def to_dict(self) -> Dict[str, Any]:
        return {"of": self.of, "rev": self.rev}


def _check_subject(subject: Any) -> None:
    ok = (
        isinstance(subject, str)
        and subject.count(":") >= 1
        and "#" not in subject
        and subject.isprintable()
        and not any(ch.isspace() for ch in subject)
    )
    if ok:
        kind, _, name = subject.partition(":")
        ok = bool(kind) and bool(name)
    if not ok:
        raise ValueError(f"an identity ref names its subject as kind:name (entity:castor, session:<id>), not {subject!r}")


def identity_ref(subject: str, rev: int) -> IdentityRef:
    """The reference to `subject`'s identity at `rev`."""
    return IdentityRef(of=subject, rev=rev)


def parse_identity_ref(text: str) -> IdentityRef:
    """Parse `entity:castor#3` (the header and log form). Refuses anything else, `castor`,
    `entity:castor`, `entity:castor#0` and `entity:castor#03` included."""
    if not isinstance(text, str) or "#" not in text:
        raise ValueError(f"an identity ref reads <subject>#<rev> (entity:castor#3), not {text!r}")
    subject, _, rev_text = text.rpartition("#")
    if not rev_text or not all(ch in _DIGITS for ch in rev_text) or rev_text[0] == "0":
        raise ValueError(f"an identity ref's rev is a whole number from 1, not {rev_text!r} in {text!r}")
    return IdentityRef(of=subject, rev=int(rev_text))


def stamp_run_identity_ref(run_vars: Dict[str, Any], ref: IdentityRef) -> None:
    """Put `ref` on a run's vars at `_runtime.identity_ref` as `{"of", "rev"}`, beside the signed
    entity stamp (`_runtime.entity`), never inside it."""
    if not isinstance(ref, IdentityRef):
        raise TypeError("stamp_run_identity_ref takes an IdentityRef")
    runtime_ns = run_vars.setdefault("_runtime", {})
    if not isinstance(runtime_ns, dict):
        raise TypeError("run vars _runtime must be an object")
    runtime_ns[IDENTITY_REF_KEY] = ref.to_dict()


def read_run_identity_ref(run_vars: Mapping[str, Any]) -> Optional[IdentityRef]:
    """The identity ref a run carries, or None when it carries none (a ref is never invented).
    A malformed one raises ValueError."""
    runtime_ns = run_vars.get("_runtime") if isinstance(run_vars, Mapping) else None
    if not isinstance(runtime_ns, Mapping) or IDENTITY_REF_KEY not in runtime_ns:
        return None
    value = runtime_ns[IDENTITY_REF_KEY]
    if not isinstance(value, Mapping) or set(value) != {"of", "rev"}:
        raise ValueError(f"_runtime.identity_ref must be {{of, rev}}, not {value!r}")
    return IdentityRef(of=value["of"], rev=value["rev"])


# --------------------------------------------------------------------------- the fingerprint


@dataclass(frozen=True)
class IdentityFingerprint:
    """Who the individual is (ADR 0052 section 4): `actor_id` derived from the manifest
    (entity id, home id, spark hash), with the continuity receipts it is served with. Derived on
    every read, never stored."""

    actor_id: str
    display_name: str
    entity_id: str
    home_id: str
    spark_hash: str
    book_head: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "kind": "entity",
            "display_name": self.display_name,
            "entity_id": self.entity_id,
            "home_id": self.home_id,
            "spark_hash": self.spark_hash,
            "book_head": self.book_head,
        }


def identity_fingerprint(manifest: Mapping[str, Any], *, book_head: Optional[str]) -> IdentityFingerprint:
    """The entity's fingerprint: `ActorFingerprint.from_metadata(kind="entity",
    display_name=<slug>, metadata={entity_id, home_id, spark_hash})`, served with `book_head`, the
    diary's chain head (`DiaryStore.head()`, None for an empty diary). A copied home with a new
    `home_id` is a different individual and gets a different `actor_id`."""
    fields: Dict[str, str] = {}
    for name in ("slug", "entity_id", "home_id", "spark_hash"):
        value = manifest.get(name) if isinstance(manifest, Mapping) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"the manifest has no {name}; a fingerprint needs slug, entity_id, home_id and spark_hash")
        fields[name] = value.strip()
    actor = ActorFingerprint.from_metadata(
        kind="entity",
        display_name=fields["slug"],
        metadata={"entity_id": fields["entity_id"], "home_id": fields["home_id"], "spark_hash": fields["spark_hash"]},
    )
    return IdentityFingerprint(
        actor_id=actor.actor_id,
        display_name=fields["slug"],
        entity_id=fields["entity_id"],
        home_id=fields["home_id"],
        spark_hash=fields["spark_hash"],
        book_head=book_head,
    )


# --------------------------------------------------------------------------- the summon cue

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _when(moment: datetime, now: datetime) -> str:
    day = moment.astimezone(timezone.utc).date()
    today = now.astimezone(timezone.utc).date()
    if day == today:
        return "today"
    text = f"on {day.day} {_MONTHS[day.month - 1]}"
    return text if day.year == today.year else f"{text} {day.year}"


def _by(set_by: str) -> str:
    if set_by == "operator":
        return " (by the operator)"
    if ":" in set_by:
        return f" (by {set_by.split(':', 1)[1]})"
    return ""


def identity_change_cue(
    home_dir: Path, *, since: Optional[str], now: Optional[datetime] = None
) -> Optional[str]:
    """One sentence for the summon prelude about the change that made the current identity, or
    None. The change is the biography row of the current rev; it is told when it is newer than
    `since` (the previous summon, aware ISO-8601), or whenever `since` is None (no previous summon
    is known to the caller). A home without `identity.json`, or whose current rev has no row,
    has nothing to tell. An unreadable identity raises `IdentityUnreadable`."""
    current = read_identity(home_dir)
    if current is None:
        return None
    row = next((r for r in reversed(_biography_rows(home_dir)) if r["rev"] == current.rev), None)
    if row is None:
        return None
    at = _parse_aware_utc(row.get("at"), "at")
    if since is not None and at <= _parse_aware_utc(since, "since"):
        return None
    when = _when(at, now or _utc_now())
    previous = row.get("from")
    if not isinstance(previous, dict):
        if current.voice is None:
            return f"You were given your look {when}; you have no voice yet."
        return f"You were given your look and voice {when}."
    by = _by(current.set_by)
    old_voice = previous.get("voice")
    new_voice = current.voice.to_dict() if current.voice is not None else None
    look = current.avatar
    look_changed = (previous.get("avatar") or {}).get("style") != look.style or (
        previous.get("avatar") or {}
    ).get("color") != look.color
    voice_changed = old_voice != new_voice
    voice_name = current.voice.voice if current.voice is not None else "no voice"
    if look_changed and voice_changed:
        return f"Your look and voice changed {when}{by}: now {look.style} in {look.color}, voice {voice_name}."
    if voice_changed:
        return f"Your voice changed {when}{by}: now {voice_name}."
    if look_changed:
        return f"Your look changed {when}{by}: now {look.style} in {look.color}."
    return f"Your identity was recorded again {when}{by}; your look and voice are unchanged."
