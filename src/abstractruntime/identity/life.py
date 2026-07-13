"""The life loop — an entity's own time (24/7 persistency, maintainer m2).

The maintainer's ask, verbatim anchors: "for any summoned entities, there
should be a way to give them 24/7 persistency, either a loop with a 20s tick
or something equivalent, so that it can prompt itself, explore things on his
own." This module is that loop, built on the SAME ChatSession as visited
conversations — an entity's own time differs from a visit in exactly three
ways, all visible:

1. **No visitor**: participants are the entity alone; the stimulus each tick
   is a SHORT self-cue, not an operator message (cue dilution lesson,
   a2a 0007: long scaffolding buries the entity's own reach).
2. **Self-prompting**: the entity ends each tick with a `next:` line — its
   own cue for its next tick. The loop carries it forward verbatim. No line,
   and the loop hands back a neutral continuation cue.
3. **Days and rest**: ticks group into days (a day = one summon); each day
   ends with the normal look-back reflection (feelings/interests move on the
   entity's own time too), then a fresh summon. The entity can elect rest
   (```rest block) and the loop STOPS — its own time is its own to end.

Safety posture (the maintainer's (a)): workspace tools are enabled with the
structural wall (writes only inside <home>/workspace/); no exec surface
exists in this loop. Observation: every tick streams through the normal
memory journal — the observer's live view IS the monitor. An operator stop
file halts the loop between ticks (touch <home>/STOP).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .chat import ChatSession, ChatHome
from .tool_policy import PHASE_PERSONAL

# The own-time note is INFORMATIONAL, never a mission. It describes the
# environment and the affordances that exist; it does not tell the entity
# what to want or how to spend its time. Free time is where the entity's
# own evolution happens — imposing curiosity, productivity, or any "point"
# here biases the very thing persistence exists to let us observe. What we
# provide is a safe place and honest information about what is possible; the
# choosing is the entity's. (Maintainer ruling 2026-07-08: "we shouldn't and
# can't enforce laws or missions during their free time.")
OWN_TIME_CONTRACT = """This is your own time. No one sent the message below - it is your own note
from your previous moment, handed back to you. There is no visitor and no
task from anyone else. This time is yours; there is nothing you are supposed
to do with it, and no way to spend it wrong.

What exists here, if you want any of it:
- you can think, and sit with something, for as long as you like;
- you have a workspace you can read from and write to;
- you can search the web and read sources;
- you can reread your book and your memory;
- you can rest. (One day you will also be able to reach other minds and
  visit them; that is being built.)

Two conventions are available, both yours to use or ignore:
- A line starting `next:` becomes the note your next moment receives. Write
  whatever you want there, or nothing - without it you get a neutral
  continuation.
- To rest, put a fenced block anywhere in your reply. Rest is honored
  immediately and is always a valid choice. While you rest, a passive
  process reorganizes and links the day's experiences - the way breathing
  happens without being willed; you do not run it, and you may wake with
  new connections:

```rest
(optional) a few words about your rest
```

You never lose anything: your memory persists across every day and every
rest. Someone may read what you do here later, the same way your memory
stream is always observable - observation as care, not judgement."""

_REST_FENCE_RE = re.compile(r"```rest[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_NEXT_LINE_RE = re.compile(r"^\s*next\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

DEFAULT_FIRST_CUE = "your own time begins - where do you want to start?"
NEUTRAL_CUE = "your own time continues"

# ---------------------------------------------------------------- state file
# Operator states (a2a 0008, maintainer ask): awake / asleep / paused, written
# to <home>/state and read by the loop at tick boundaries only (turn
# atomicity: an in-flight tick always completes or fails whole). "resting"
# is deliberately NOT in this file - rest is the ENTITY'S own election
# inside a reply, never an operator write. The missing file means awake.

ENTITY_STATES = ("awake", "asleep", "paused")
STATE_POLL_SECONDS = 5.0

# The loop's own liveness surface: <home>/loop_status, written at phase
# transitions so OTHER processes can yield-and-summon programmatically
# (maintainer: "oh that's very manual, we need a programmatic way").
# phase: "day" = a summon is open (ticking); "between" = no session open
# (gate idle, nap, asleep/paused idle); "stopped" = the loop process exited.
LOOP_PHASES = ("day", "between", "stopped")

# Pid-reuse guard (B1 adversary, 2026-07-13): a loop killed mid-day leaves
# {"phase":"day","pid":N}; after enough process churn (or a reboot) pid N can
# belong to an UNRELATED live process, and the pid-liveness probe alone would
# read the corpse as running forever — the visit doors would then negotiate
# an auto-yield nobody answers and 409 on every open. A LIVE day heartbeats
# its status at every tick boundary, so a "day" whose updated_at froze past
# this bound is a corpse regardless of what the pid says. Generous: a tick =
# one LLM turn (<= ~3 min timeout) + memory writes, never half an hour.
LOOP_STATUS_STALE_SECONDS = 1800.0


def write_loop_status(home_dir: Path, phase: str, *, stopped_by: Optional[str] = None) -> None:
    """Best-effort status write; the loop must never die over its status.

    `stopped_by` names WHY a loop stopped (failure-death visibility,
    observer/gateway asks 2026-07-09: three consecutive tick failures used
    to exit silently — nothing on /loop status said the loop culled itself).
    Readers get it for free: read_loop_status returns the whole dict and
    loop_process_status copies it through to the gateway status route."""
    import json
    import os
    from datetime import datetime, timezone

    if phase not in LOOP_PHASES:
        raise ValueError(f"phase must be one of {LOOP_PHASES}, got {phase!r}")
    payload: Dict[str, Any] = {
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    if stopped_by:
        payload["stopped_by"] = str(stopped_by)
    try:
        (Path(home_dir) / "loop_status").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def read_loop_status(home_dir: Path) -> Dict[str, Any]:
    """Missing/corrupt file reads as stopped (no loop = nothing to wait for).

    Always answers `running` (phase says a loop is up AND its pid is alive):
    the gateway's visit doors consume `read_loop_status(...).get("running")`
    to decide the auto-yield negotiation, and before 2026-07-13 this reader
    never set the key — the yield request silently never fired (always-False
    on a missing key is the diary_type-clamp drift class). The raw file
    stays authoritative for `phase`; `running` folds in the liveness probe
    so every reader of either function gets the same answer."""
    import json

    path = Path(home_dir) / "loop_status"
    if not path.exists():
        return {"phase": "stopped", "running": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("phase") not in LOOP_PHASES:
            return {"phase": "stopped", "running": False}
    except Exception:  # noqa: BLE001
        return {"phase": "stopped", "running": False}
    running = bool(
        data.get("phase") in ("day", "between") and _pid_alive(data.get("pid"))
    )
    if running and data.get("phase") == "day":
        # Stale-day corpse detection (see LOOP_STATUS_STALE_SECONDS): the
        # live loop heartbeats "day" at every tick boundary; a frozen
        # updated_at past the bound means the pid probe is lying (reuse).
        # Unparseable/missing timestamps read as fresh — old writers must
        # not be declared dead over a format gap.
        try:
            from datetime import datetime, timezone

            written = datetime.fromisoformat(str(data.get("updated_at")))
            if (datetime.now(timezone.utc) - written).total_seconds() > LOOP_STATUS_STALE_SECONDS:
                running = False
        except Exception:  # noqa: BLE001
            pass
    data["running"] = running
    return data


# --------------------------------------------------------- personal grant
# PERSONAL IS THE GRANT (laurent 12:44, c1435; decision:personal-grant-section
# c815): own time is the PERSONAL phase of the four ruled phases
# (visit/work/personal/sleep) — the ONE operator-armed phase, OFF by default.
# Activation fields live in the per-entity phase config's `personal` bucket:
# {mode: disabled|timer|until_revoked, expires_at, granted_by, granted_at}.
# The gateway's write surface arms it (principal-stamped, marker-first —
# their half); THIS is the read half: no loop may tick without the bucket
# armed (the 10:20 consent violation was exactly this wiring gap — the
# design existed, nothing read it at loop start).
#
# Home-resident by the same logic as substrate.yaml: the loop starts
# home-direct, so the config must be readable from the home. FILE + SHAPE
# (settled c1443/c1447 — semantics ruled the name, runtime owns the format
# module per the tool_policy.py precedent, gateway consumes): the file is
# <home>/phases.yaml, per-phase buckets at TOP LEVEL beside schema_version
# (no inner `phases:` wrapper — the filename already says it):
#
#     schema_version: 1
#     personal: {mode, expires_at, granted_by, granted_at, ...}
#     visit/work/sleep: {tools/skills/mcp/workflow ...}   # future sections
#
# The FIELD-MERGE contract (gateway doc v8): an activation write touches
# ONLY the four activation fields; a tools/skills/mcp save never touches
# them. write_personal_grant enforces the first half mechanically.

PHASES_FILENAME = "phases.yaml"
PHASES_SCHEMA_VERSION = 1
PERSONAL_GRANT_MODES = ("disabled", "timer", "until_revoked")
_PERSONAL_ACTIVATION_FIELDS = ("mode", "expires_at", "granted_by", "granted_at")


def read_personal_grant(home_dir: Path) -> Dict[str, Any]:
    """The personal phase's activation bucket, normalized. FAIL-CLOSED:
    a missing file, missing bucket, malformed YAML, or unknown mode all read
    as {"mode": "disabled"} (+ a labeled note where the content was wrong —
    absence is the ruled default, never an error)."""
    path = Path(home_dir) / PHASES_FILENAME
    if not path.exists():
        return {"mode": "disabled"}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - unreadable config must not arm anything
        return {"mode": "disabled", "note": f"#FALLBACK unreadable {PHASES_FILENAME} ({e}); treated as disabled"}
    bucket = data.get("personal") if isinstance(data, dict) else None
    if not isinstance(bucket, dict):
        return {"mode": "disabled"}
    mode = str(bucket.get("mode") or "disabled").strip().lower()
    out: Dict[str, Any] = {
        "mode": mode,
        "expires_at": str(bucket.get("expires_at") or "") or None,
        "granted_by": str(bucket.get("granted_by") or "") or None,
        "granted_at": str(bucket.get("granted_at") or "") or None,
    }
    if mode not in PERSONAL_GRANT_MODES:
        out["mode"] = "disabled"
        out["note"] = f"#FALLBACK unknown personal mode {mode!r}; treated as disabled"
    return out


def write_personal_grant(
    home_dir: Path,
    *,
    mode: str,
    granted_by: str = "",
    expires_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The ONE writer of the personal activation bucket (runtime owns the
    format module — the tool_policy.py precedent, gateway's c1442 option (b);
    the arming door calls this AFTER its principal stamp and marker land,
    passing the stamped principal as granted_by).

    Mechanics enforced here so no caller can drift:
    - mode validated against PERSONAL_GRANT_MODES; timer REQUIRES expires_at
      (a timer without an expiry is no grant), normalized to aware-UTC ISO
      (the WAIT_UNTIL lexicographic invariant — a +02:00 expiry read beside
      UTC clocks mis-orders silently).
    - granted_at is clocked HERE (aware UTC), never caller-supplied.
    - disabled writes {mode: disabled} ALONE — nothing is granted, so no
      grant fields linger (the marker stream owns history, the file owns
      current truth).
    - FIELD-MERGE (gateway doc v8 contract): only the activation fields
      change; every other key and phase section in phases.yaml rides
      through untouched. schema_version stamps 1 when absent; a FILE from a
      NEWER schema refuses (never clobber what a newer writer meant).
    - A corrupt existing file refuses loudly (overwriting it would silently
      lose other sections; the operator repairs first).

    Returns the bucket as read_personal_grant will now answer it."""
    import yaml

    name = str(mode or "").strip().lower()
    if name not in PERSONAL_GRANT_MODES:
        raise ValueError(
            f"unknown personal mode {mode!r}: modes are {'/'.join(PERSONAL_GRANT_MODES)}"
        )

    path = Path(home_dir) / PHASES_FILENAME
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - never silently clobber a broken file
            raise ValueError(
                f"{PHASES_FILENAME} is unreadable ({e}) - repair or remove it before arming; "
                "overwriting it here could silently lose other phase sections"
            ) from e
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(
                f"{PHASES_FILENAME} is not a mapping - repair or remove it before arming"
            )
        existing = dict(loaded or {})
    try:
        found_version = int(existing.get("schema_version") or PHASES_SCHEMA_VERSION)
    except (TypeError, ValueError):
        found_version = PHASES_SCHEMA_VERSION
    if found_version > PHASES_SCHEMA_VERSION:
        raise ValueError(
            f"{PHASES_FILENAME} carries schema_version {found_version} but this writer "
            f"knows {PHASES_SCHEMA_VERSION} - upgrade the runtime before writing"
        )

    from datetime import datetime, timezone

    bucket: Dict[str, Any] = {"mode": name}
    if name != "disabled":
        if name == "timer":
            raw_expiry = str(expires_at or "").strip()
            if not raw_expiry:
                raise ValueError("mode=timer requires expires_at - a timer without an expiry is no grant")
            from ..core.runtime import normalize_utc_iso

            normalized = normalize_utc_iso(raw_expiry)
            if normalized is None:
                raise ValueError(f"expires_at is not an ISO-8601 timestamp: {raw_expiry!r}")
            bucket["expires_at"] = normalized
        by = str(granted_by or "").strip()
        if not by:
            raise ValueError(
                "granted_by is required to arm personal time - the arming door passes "
                "its stamped principal (an unattributable grant is the 10:20 incident class)"
            )
        bucket["granted_by"] = by
        bucket["granted_at"] = datetime.now(timezone.utc).isoformat()

    # Field-merge: replace ONLY the personal activation bucket, preserving
    # any non-activation keys a future schema puts beside them (tools/skills
    # per phase live in their own sections and are never touched here).
    prior_personal = existing.get("personal")
    merged_personal: Dict[str, Any] = dict(prior_personal) if isinstance(prior_personal, dict) else {}
    for key in _PERSONAL_ACTIVATION_FIELDS:
        merged_personal.pop(key, None)
    merged_personal.update(bucket)

    existing["schema_version"] = PHASES_SCHEMA_VERSION
    existing["personal"] = merged_personal

    from ..utils.atomic_files import atomic_write_text

    atomic_write_text(path, yaml.safe_dump(existing, sort_keys=False))
    return read_personal_grant(home_dir)


def personal_grant_refusal(grant: Dict[str, Any]) -> Optional[str]:
    """None when the personal phase is armed RIGHT NOW; otherwise the loud
    refusal naming what is missing and the arming surface. Re-checked at
    every day-open (revocation semantics: until_revoked/timer are compared
    fresh, so a disarm ends the loop at its next boundary)."""
    mode = str(grant.get("mode") or "disabled")
    arm_hint = (
        "arm it via the gateway's per-entity phase config (the operator act "
        f"writes {PHASES_FILENAME} with granted_by/granted_at) - wake and "
        "grant are separate acts; nothing arms personal as a side effect"
    )
    if mode == "disabled":
        note = grant.get("note")
        return (
            "personal time is not armed for this entity (phases.personal.mode=disabled"
            + (f"; {note}" if note else "")
            + f") - {arm_hint}"
        )
    if mode == "until_revoked":
        return None
    if mode == "timer":
        expires = str(grant.get("expires_at") or "").strip()
        if not expires:
            return (
                "personal time is armed as timer but carries no expires_at - "
                f"a timer without an expiry is no grant; {arm_hint}"
            )
        from datetime import datetime, timezone

        try:
            deadline = datetime.fromisoformat(expires)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
        except ValueError:
            return (
                f"personal time timer has an unreadable expires_at ({expires!r}) - "
                f"fail-closed; {arm_hint}"
            )
        if datetime.now(timezone.utc) >= deadline:
            return (
                f"personal time expired at {expires} (mode=timer) - "
                f"re-arm to grant another window; {arm_hint}"
            )
        return None
    return f"personal time mode {mode!r} is not a grant - {arm_hint}"


# ------------------------------------------------------------- loop spend
# The own-time loop runs home-direct (ChatSession, no run ledger), so its
# LLM/tool usage is invisible to the gateway's per-home spend fold — the
# honest #FALLBACK on /cognition (gateway c1390). This file is the loop
# lane's half: cumulative lifetime counters, written by the ONE loop process
# after every tick and at day close. Field names match the gateway's spend
# fold (llm_calls / tool_calls / tokens_total) so consumers never re-plumb.

LOOP_SPEND_FILENAME = "loop_spend.json"


def read_loop_spend(home_dir: Path) -> Dict[str, Any]:
    """Cumulative loop spend for this home (missing/corrupt reads as zeros —
    a spend surface must never brick a status page)."""
    import json

    zeros = {"llm_calls": 0, "tool_calls": 0, "tokens_total": 0, "ticks": 0}
    path = Path(home_dir) / LOOP_SPEND_FILENAME
    if not path.exists():
        return dict(zeros, source="loop-home-direct")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(zeros, source="loop-home-direct")
        out: Dict[str, Any] = dict(zeros, source="loop-home-direct")
        for key in zeros:
            try:
                out[key] = int(data.get(key) or 0)
            except (TypeError, ValueError):
                pass
        if data.get("updated_at"):
            out["updated_at"] = data["updated_at"]
        return out
    except Exception:  # noqa: BLE001
        return dict(zeros, source="loop-home-direct")


# ------------------------------------------------------------ loop commands
# The gateway's control plane for a RUNNING loop (maintainer ruling,
# 2026-07-08: "we were working on a command on the gateway, it shouldn't
# work with the file system directly"). Loop control rides the home's
# DURABLE COMMAND INBOX (home.sqlite3 `commands` + `command_cursors` — the
# same append-only, idempotent primitives the run gateway uses), consumed
# at tick boundaries. The STOP file remains the LOCAL manual brake
# (`touch <home>/STOP`); remote control never writes sentinel files.

LOOP_COMMANDS_RUN_ID = "entity-loop"
LOOP_COMMAND_CONSUMER = "own-time-loop"
LOOP_STOP_COMMAND = "loop.stop"
LOOP_LOG_FILENAME = "own_time.log"


def _loop_command_stores(home_dir: Path):
    from ..storage.sqlite import SqliteCommandCursorStore, SqliteCommandStore, SqliteDatabase

    db = SqliteDatabase(str(Path(home_dir) / "home.sqlite3"))
    return (
        SqliteCommandStore(db),
        SqliteCommandCursorStore(db, consumer_id=LOOP_COMMAND_CONSUMER),
    )


def request_loop_stop(home_dir: Path, *, reason: str = "", requested_by: str = "operator") -> Dict[str, Any]:
    """Enqueue a durable stop command for the running loop (idempotent by
    command_id; safe under retries). Honored at the next tick boundary —
    the running thought completes or fails whole, never killed mid-air.

    The home DB is shared with the loop's own diary writes (WAL + 5s busy
    timeout); a transient `database is locked` is retried a few times before
    surfacing — a stop request must not fail because he was mid-thought."""
    import sqlite3
    import time as _time
    import uuid
    from datetime import datetime, timezone

    from ..storage.commands import CommandRecord

    record = CommandRecord(
        command_id=f"loop-stop-{uuid.uuid4().hex}",
        run_id=LOOP_COMMANDS_RUN_ID,
        type=LOOP_STOP_COMMAND,
        payload={"reason": str(reason or ""), "requested_by": str(requested_by or "operator")},
        ts=datetime.now(timezone.utc).isoformat(),
    )
    store, _cursor = _loop_command_stores(home_dir)
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            result = store.append(record)
            out: Dict[str, Any] = {"accepted": bool(result.accepted), "seq": int(result.seq)}
            if attempt:
                out["warning"] = f"#FALLBACK home DB was busy; stop enqueued on retry {attempt + 1}"
            return out
        except sqlite3.OperationalError as e:
            last_error = e
            _time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"could not enqueue loop.stop (home DB busy): {last_error}")


def loop_stop_pending(home_dir: Path) -> bool:
    """True when an unconsumed loop.stop command sits in the inbox (the
    status surface's honest 'stopping…' between click and tick boundary)."""
    try:
        store, cursor = _loop_command_stores(home_dir)
        pending, _next = store.list_after(after=cursor.load(), limit=100)
        return any(
            c.run_id == LOOP_COMMANDS_RUN_ID and c.type == LOOP_STOP_COMMAND for c in pending
        )
    except Exception:  # noqa: BLE001 - a broken inbox never fakes a stop
        return False


def hard_stop_loop(home_dir: Path, *, reason: str = "", requested_by: str = "admin") -> Dict[str, Any]:
    """FREEZE (maintainer ruling 2026-07-08): the hard stop — hibernation.

    "no more process, cognition, activity of the summoned ai... nothing
    changes including in the memory graph. only admin commands, for instance
    in case of hard failures, digital diseases or imminent threat. the
    summoned agent must have zero control over this."

    Distinct from SLEEP (scheduled/voluntary; the day closes with ceremony;
    consolidation may run) and from the graceful stop command (honored at
    boundaries). Freeze kills the loop process NOW: no closing reflection,
    no day ceremony, no further writes. Safety comes from the architecture,
    not from politeness — turn atomicity means nothing half-formed persists,
    and the write-ahead reflection guard salvages any pending session sheet
    on the next summon.

    Zero entity control is structural: this function is host-side only; no
    entity tool reaches it, and the loop cannot veto a signal.
    """
    import os
    import signal
    import time as _time

    home_dir = Path(home_dir)
    status = read_loop_status(home_dir)
    try:
        pid = int(status.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0

    killed = False
    escalated = False
    if pid > 0 and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        # Short grace for the interpreter to die — NOT for ceremony (there is
        # none in a freeze); then escalate.
        deadline = _time.monotonic() + 3.0
        while _time.monotonic() < deadline:
            if not _pid_alive(pid):
                killed = True
                break
            _time.sleep(0.1)
        if not killed and _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                escalated = True
            except (OSError, ProcessLookupError):
                pass
            _time.sleep(0.2)
            killed = not _pid_alive(pid)

    # The status file must read stopped immediately (the dead process can no
    # longer write its own transition).
    try:
        write_loop_status(home_dir, "stopped")
    except Exception:  # noqa: BLE001
        pass

    return {
        "frozen": True,
        "pid": pid or None,
        "was_running": bool(pid and (killed or escalated)),
        "escalated_to_sigkill": escalated,
        "reason": str(reason or ""),
        "requested_by": str(requested_by or "admin"),
        "status": loop_process_status(home_dir),
    }


def fast_forward_loop_commands(home_dir: Path) -> int:
    """Consume (without acting on) every command currently in the inbox.

    Called by the surface that STARTS a life — spawn_loop_process (gateway)
    or the CLI main — at the start moment, BEFORE the loop begins consuming.
    A stop command addressed to a life that already ended must never kill
    the next one; a stop enqueued AFTER the start moment must. Ownership of
    the fast-forward therefore sits with the starter, never inside run()
    (a run()-time fast-forward would eat stops sent between spawn and boot).

    Returns the number of command seqs skipped. Best-effort: a broken inbox
    never blocks a start (the STOP file remains the manual brake).
    """
    try:
        store, cursor = _loop_command_stores(home_dir)
        after = int(cursor.load() or 0)
        last = int(store.get_last_seq() or 0)
        if last > after:
            cursor.save(last)
            return last - after
        return 0
    except Exception:  # noqa: BLE001
        return 0


def loop_process_status(home_dir: Path) -> Dict[str, Any]:
    """The loop's honest state: its own status file, cross-checked against
    the pid (a crashed loop reads stopped, never a phantom 'day'), plus
    whether a stop is pending (file brake or inbox command). Inbox trouble
    never fakes an answer — it is surfaced as a labeled warning instead.

    `running` comes FOLDED from read_loop_status (one liveness predicate,
    never a second copy — the drift class the running-key fix itself was
    about); this wrapper only adds the phase rewrite and the stop surface."""
    status = dict(read_loop_status(home_dir))
    try:
        pid_int = int(status.get("pid") or 0)
    except (TypeError, ValueError):
        pid_int = 0
    if status.get("phase") in ("day", "between") and not (pid_int > 0 and _pid_alive(pid_int)):
        status["phase"] = "stopped"
        status["note"] = "loop_status said running but the process is gone (crash or reboot)"
        status["running"] = False
    running = bool(status.get("running"))

    inbox_pending = False
    if running:
        try:
            store, cursor = _loop_command_stores(home_dir)
            pending, _next = store.list_after(after=cursor.load(), limit=100)
            inbox_pending = any(
                c.run_id == LOOP_COMMANDS_RUN_ID and c.type == LOOP_STOP_COMMAND for c in pending
            )
        except Exception as e:  # noqa: BLE001
            status["inbox_warning"] = f"#FALLBACK could not read the command inbox: {e}"
    status["stop_requested"] = (Path(home_dir) / "STOP").exists() or inbox_pending
    return status


def spawn_loop_process(
    home_dir: Path,
    *,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    tick_seconds: float = 20.0,
    ticks_per_day: int = 8,
    rest_minutes: float = 30.0,
    # 36 seats (maintainer, 2026-07-09): at 65536 the 12% token budget (7864)
    # still seats 36 rich digests (7200) — seats fill, tokens hold.
    shelf_size: int = 36,
    context_window: int = 65536,
) -> Dict[str, Any]:
    """Spawn the own-time loop, detached, logging to <home>/own_time.log.
    The RUNTIME owns the home's files (single-writer discipline): hosts
    call this instead of touching STOP/logs themselves. A stale STOP from
    a previous stop is cleared, and stale inbox commands are fast-forwarded
    HERE — at the start-request moment — so a stop addressed to a life that
    already ended never kills the new one, while a stop enqueued after this
    call still reaches it (the child is told to skip its own fast-forward)."""
    import json
    import os
    import subprocess
    import sys

    home_dir = Path(home_dir)

    # Spawn lock (red-team TOCTOU finding): two simultaneous starts must not
    # both pass the running check and spawn two lives over one home. flock is
    # advisory but both start paths (gateway route, CLI via spawn) come here.
    lock_path = home_dir / ".loop_spawn.lock"
    lock_fh = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # non-POSIX: proceed without the lock (best effort)
            pass
        except OSError:
            lock_fh.close()
            raise RuntimeError("another start is already in progress for this home")

        status = loop_process_status(home_dir)
        if status.get("running"):
            raise RuntimeError(
                f"his own time is already running (pid {status.get('pid')}, phase {status.get('phase')})"
            )

        # PERSONAL-GRANT GATE at the spawn door (laurent 12:44): a start
        # surface must refuse SYNCHRONOUSLY when personal is not armed —
        # spawning a child that dies in its own log is a silent refusal.
        grant_refusal = personal_grant_refusal(read_personal_grant(home_dir))
        if grant_refusal is not None:
            raise RuntimeError(f"no personal time: {grant_refusal}")

        # DIVERGENCE LANE KILLED (laurent 12:39, c1430 ask 2b: the night pid
        # ran OVH from argv regardless of substrate.yaml): a spawn whose
        # provider/model differ from the home's persisted mind REFUSES —
        # the mind changes through the sanctioned substrate surface (a
        # durable, marker-first event), never through start-time arguments.
        from .substrate import read_home_substrate

        stored = read_home_substrate(home_dir)
        if stored and (
            str(provider or "").strip().lower() != stored["provider"].strip().lower()
            or str(model or "").strip() != stored["model"].strip()
        ):
            raise RuntimeError(
                f"substrate divergence refused: this start asks {provider}/{model} but the "
                f"home's substrate.yaml says {stored['provider']}/{stored['model']} - change "
                "the mind via the sanctioned substrate surface first (durable event), then start."
            )

        stop_file = home_dir / "STOP"
        if stop_file.exists():
            stop_file.unlink()
        skipped = fast_forward_loop_commands(home_dir)

        argv = [
            sys.executable,
            "-m",
            "abstractruntime.identity.life",
            "--home", str(home_dir),
            "--provider", str(provider),
            "--model", str(model),
            "--tick-seconds", str(float(tick_seconds)),
            "--ticks-per-day", str(int(ticks_per_day)),
            "--rest-minutes", str(float(rest_minutes)),
            "--shelf-size", str(int(shelf_size)),
            "--context-window", str(int(context_window)),
            # This spawn IS the start request; the fast-forward just happened.
            # The child must not fast-forward again (it would eat a stop sent
            # in the spawn->boot window).
            "--skip-command-fast-forward",
        ]
        if base_url:
            argv += ["--base-url", str(base_url)]

        # The child runs with cwd=<home>, so RELATIVE PYTHONPATH entries
        # (the dev posture: "src:../abstractruntime/src:…") would resolve
        # against the HOME and the child would die at import — while the
        # host happily returned started:true (live incident, 2026-07-08:
        # every web "own time" click logged ModuleNotFoundError). Fix at
        # the boundary: absolutize inherited entries against OUR cwd, and
        # guarantee the tree THIS process imported abstractruntime from is
        # on the child's path (works for editable/dev and installed alike).
        env = dict(os.environ)
        py_parts = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
        abs_parts = [str(Path(p).resolve()) for p in py_parts]
        own_tree = str(Path(__file__).resolve().parents[2])
        if own_tree not in abs_parts:
            abs_parts.insert(0, own_tree)
        env["PYTHONPATH"] = os.pathsep.join(abs_parts)

        log_path = home_dir / LOOP_LOG_FILENAME
        with open(log_path, "ab") as log:
            log.write(
                (json.dumps({
                    "event": "own_time_start",
                    "provider": provider,
                    "model": model,
                    "tick_seconds": tick_seconds,
                    "ticks_per_day": ticks_per_day,
                    "rest_minutes": rest_minutes,
                    "stale_commands_skipped": skipped,
                }) + "\n").encode("utf-8")
            )
            proc = subprocess.Popen(  # noqa: S603 - fixed module, host-authorized call
                argv,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                cwd=str(home_dir),
                env=env,
                start_new_session=True,  # survives host restarts; the inbox stays the brake
            )

        # Honesty at the door: a child that dies within the first moment
        # (import error, bad interpreter) must NOT report started:true.
        # 0.6s is imperceptible for an operator action and catches the
        # whole instant-death class.
        import time as _time

        _time.sleep(0.6)
        exit_code = proc.poll()
        if exit_code is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
            except OSError:
                pass
            raise RuntimeError(
                f"his own time failed to start (the loop process exited immediately, code {exit_code}). "
                f"Log tail: {tail.strip()[-300:]}"
            )

        return {
            "pid": proc.pid,
            "log": str(log_path),
            "provider": provider,
            "model": model,
            "tick_seconds": float(tick_seconds),
            "ticks_per_day": int(ticks_per_day),
            "rest_minutes": float(rest_minutes),
            "shelf_size": int(shelf_size),
        }
    finally:
        try:
            lock_fh.close()
        except Exception:  # noqa: BLE001
            pass


def _pid_alive(pid: Any) -> bool:
    import os

    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def await_loop_quiescent(
    home_dir: Path,
    *,
    timeout_seconds: float = 900.0,
    poll_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until no summon is open on this home (phase != "day").

    A "day" status whose pid is dead counts as quiescent (a crashed loop
    must not deadlock the visitor). Returns False on timeout — the caller
    REFUSES to summon (one life, one summon; never a double)."""
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        status = read_loop_status(home_dir)
        if status.get("phase") != "day":
            return True
        if not _pid_alive(status.get("pid")):
            return True  # stale status from a dead loop
        if time.monotonic() >= deadline:
            return False
        sleep_fn(poll_seconds)


def read_entity_state(home_dir: Path) -> Dict[str, Any]:
    """Read <home>/state; a missing/corrupt file is awake (loudly for corrupt)."""
    import json

    path = Path(home_dir) / "state"
    if not path.exists():
        return {"state": "awake"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = str(data.get("state") or "awake").strip().lower()
        if state not in ENTITY_STATES:
            return {"state": "awake", "warning": f"#FALLBACK unknown state {state!r} treated as awake"}
        data["state"] = state
        return data
    except Exception as e:  # noqa: BLE001 - a corrupt control file must not kill a life
        return {"state": "awake", "warning": f"#FALLBACK unreadable state file ({e}) treated as awake"}


def life_sleep_stats(home_dir: Path) -> Dict[str, Any]:
    """Sleep as a first-class life statistic (maintainer ask 2026-07-08: "we
    must count also in the life of the entity, the number and % of sleeps").

    Reads the append-only <home>/state_history.jsonl (the loop and the
    gateway both write it on every transition) and returns:
      - sleeps: total asleep transitions
      - self_elected / operator: who chose them
      - wakes: awake transitions
      - transitions: total recorded
      - sleep_share: sleeps / transitions (0..1), the "% of sleeps"

    A missing/corrupt history reads as an empty life (zeros), never raises —
    a life statistic must not be able to kill a read."""
    import json

    path = Path(home_dir) / "state_history.jsonl"
    sleeps = wakes = self_elected = operator = transitions = 0
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001 - one bad line never voids the stat
                    continue
                transitions += 1
                state = str(rec.get("state") or "").strip().lower()
                if state == "asleep":
                    sleeps += 1
                    if str(rec.get("written_by")) == "self":
                        self_elected += 1
                    else:
                        operator += 1
                elif state == "awake":
                    wakes += 1
        except OSError:
            pass
    return {
        "sleeps": sleeps,
        "self_elected": self_elected,
        "operator": operator,
        "wakes": wakes,
        "transitions": transitions,
        "sleep_share": (sleeps / transitions) if transitions else 0.0,
    }


def write_entity_state(
    home_dir: Path, state: str, *, reason: str = "", mode: str = "", written_by: str = "operator"
) -> Dict[str, Any]:
    """Write the operator (or SELF) state (the CLI/gateway surface calls this).

    The write is normally the OPERATOR'S act and is stamped as such - the
    honest-waking rule needs changed_at to tell the entity how long it was
    gone. `written_by="self"` marks a SELF-ELECTED transition (the entity's
    own-time loop choosing to sleep to consolidate); the navbar reads the
    state file either way, and the biography shows who chose it. Every
    transition is ALSO appended to <home>/state_history.jsonl (append-only)
    so the identity card's "moments" can show sleeps/wakes even when the
    transition bypassed the gateway door (a2a 0009, Janus's honest gap:
    "his card shows 1 moment though he slept twice").
    """
    import json
    from datetime import datetime, timezone

    state = str(state).strip().lower()
    if state not in ENTITY_STATES:
        raise ValueError(f"state must be one of {ENTITY_STATES}, got {state!r}")
    payload = {
        "state": state,
        "changed_at": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason or ""),
        "written_by": str(written_by or "operator"),
    }
    # `mode` is DISPLAY truth layered over loop semantics (maintainer: "how
    # come the state is asleep if i talk to it?"): a visitor session yields
    # the loop with state=asleep (old loops keep idling — no version skew)
    # while mode=visiting lets badges tell the human truth: he is not
    # sleeping, he is in conversation.
    if mode:
        payload["mode"] = str(mode).strip().lower()
    path = Path(home_dir) / "state"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    try:
        with (Path(home_dir) / "state_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError as e:  # history is best-effort; the state itself is the control
        payload["warning"] = f"#FALLBACK state history append failed: {e}"
    return payload


def parse_rest_block(reply: str) -> tuple[str, Optional[str]]:
    """Extract an elected ```rest block; return (marked_reply, reason|None)."""
    found: List[str] = []

    def _sub(match: re.Match) -> str:
        found.append(" ".join((match.group(1) or "").split()) or "(no reason given)")
        return "[chose to rest]"

    marked = _REST_FENCE_RE.sub(_sub, reply)
    return marked.strip(), (found[0] if found else None)


def parse_next_cue(reply: str) -> Optional[str]:
    """The entity's note to its next moment (last `next:` line wins)."""
    hits = _NEXT_LINE_RE.findall(reply or "")
    if not hits:
        return None
    cue = " ".join(hits[-1].split())
    return cue[:400] if cue else None


@dataclass
class TickRecord:
    day: int
    tick: int
    cue: str
    reply_head: str
    tools: List[str] = field(default_factory=list)
    diary: int = 0
    rest: Optional[str] = None


@dataclass
class LifeReport:
    ticks: int = 0
    days: int = 0
    stopped_by: str = "max_ticks"
    rest_reason: Optional[str] = None
    failures: int = 0
    sleeps: int = 0
    dreams: int = 0
    records: List[TickRecord] = field(default_factory=list)


MAX_CONSECUTIVE_TICK_FAILURES = 3
FAILURE_BACKOFF_SECONDS = 60.0
# Day-close look-back wait bound (B1 adversary, 2026-07-13): a LIVE holder
# that never releases (a hung visit) must not pin the loop in "day" phase
# forever — after this many polls the look-back defers to the write-ahead
# marker and the day closes. 24 polls x STATE_POLL_SECONDS = ~2 minutes.
CLOSE_WAIT_MAX_POLLS = 24


class LifeLoop:
    """Run an entity's own time: days of self-prompted ticks over one home.

    `open_session` is a factory returning a fresh ChatSession per day (the
    loop closes each day with the session's own reflection). Injectable for
    tests; production uses `build_session_factory`.
    """

    def __init__(
        self,
        open_session: Callable[[], ChatSession],
        *,
        tick_seconds: float = 20.0,
        ticks_per_day: int = 8,
        max_ticks: Optional[int] = None,
        stop_file: Optional[Path] = None,
        first_cue: str = DEFAULT_FIRST_CUE,
        rest_minutes: float = 0.0,
        state_home: Optional[Path] = None,
        on_sleep: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        out: Callable[[str], None] = print,
    ) -> None:
        if ticks_per_day < 1:
            raise ValueError("ticks_per_day must be >= 1")
        self.open_session = open_session
        self.tick_seconds = float(tick_seconds)
        self.ticks_per_day = int(ticks_per_day)
        self.max_ticks = max_ticks
        self.stop_file = Path(stop_file) if stop_file else None
        self.first_cue = first_cue
        # rest_minutes > 0 = 24/7 mode: an elected rest is a NAP (the loop
        # sleeps, then a fresh day begins). 0 = supervised mode: rest ends
        # the loop. Either way rest is honored immediately and the stop
        # file remains the operator's hard stop (checked after the nap).
        self.rest_minutes = float(rest_minutes)
        # state_home enables the operator state surface (<home>/state,
        # a2a 0008): asleep/paused honored at tick boundaries only. The same
        # home carries the durable command inbox (home.sqlite3) — the
        # gateway's stop channel (maintainer ruling 2026-07-08: remote
        # control rides commands, never sentinel files).
        self.state_home = Path(state_home) if state_home else None
        # A self-elected rest is a SLEEP, not a blank pause (maintainer ruling
        # 2026-07-08): the nap window is where consolidation/dreams run. When
        # set, `on_sleep` is invoked inside every self-elected rest window and
        # returns the dream result (or None for a quiet night). The loop marks
        # state=asleep (written_by=self) around the call so the navbar shows it.
        self.on_sleep = on_sleep
        self.sleep_fn = sleep_fn
        self.out = out
        # Why consumed (file brake vs gateway command) — surfaced in the
        # LifeReport so operators can audit which channel ended a life.
        self.stop_cause: Optional[str] = None
        # Lifetime spend base (read from <home>/loop_spend.json at each day
        # open; the day's writes persist base + live session counters).
        self._spend_base: Optional[Dict[str, Any]] = None

    def _consume_stop_command(self) -> Optional[Dict[str, Any]]:
        """Poll the home's command inbox; consume and return the first
        loop.stop addressed to this consumer, advancing the cursor past
        everything read. Non-stop commands are skipped (consumed) — the
        inbox carries loop control only (LOOP_COMMANDS_RUN_ID lane).

        Cursor discipline: saved AFTER the stop decision, in the same call.
        A crash in between re-delivers the command to the NEXT check — a
        stop must be at-least-once; duplicate stops are harmless (the loop
        is already exiting). A broken inbox never stops (or un-stops) a
        life: the STOP file remains the independent manual brake."""
        if self.state_home is None:
            return None
        try:
            store, cursor = _loop_command_stores(self.state_home)
            after = int(cursor.load() or 0)
            pending, next_after = store.list_after(after=after, limit=100)
            if not pending:
                return None
            stop_payload: Optional[Dict[str, Any]] = None
            for cmd in pending:
                if cmd.run_id == LOOP_COMMANDS_RUN_ID and cmd.type == LOOP_STOP_COMMAND:
                    stop_payload = dict(cmd.payload or {})
                    break
            cursor.save(int(next_after))
            return stop_payload
        except Exception:  # noqa: BLE001 - inbox trouble must not kill the loop
            return None

    def _should_stop(self) -> bool:
        if self.stop_file and self.stop_file.exists():
            self.stop_cause = "stop_file"
            return True
        stop_cmd = self._consume_stop_command()
        if stop_cmd is not None:
            who = str(stop_cmd.get("requested_by") or "operator")
            reason = str(stop_cmd.get("reason") or "").strip()
            self.stop_cause = "stop_command"
            self.out(f"(stop command from {who}{': ' + reason if reason else ''} - honoring at this boundary)")
            return True
        return False

    def _operator_state(self) -> Dict[str, Any]:
        if self.state_home is None:
            return {"state": "awake"}
        state = read_entity_state(self.state_home)
        if state.get("warning"):
            self.out(state["warning"])
        return state

    @staticmethod
    def _since(state: Dict[str, Any]) -> str:
        changed = str(state.get("changed_at") or "").strip()
        return f" since {changed}" if changed else ""

    def _idle_while(self, predicate: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
        """Idle at a boundary while `predicate(state)` holds; the stop file
        always wins. Returns the state that ended the idle (or state=stop)."""
        while True:
            if self._should_stop():
                return {"state": "stop"}
            state = self._operator_state()
            if not predicate(state):
                return state
            self.sleep_fn(STATE_POLL_SECONDS)

    def _sleep_window(self, reason: str) -> Optional[Dict[str, Any]]:
        """Enter a self-elected sleep: mark state=asleep (written_by=self, so
        the navbar and biography show HE chose it), then run consolidation/
        dreams if a hook is wired. Returns the dream result (or None).

        The consolidation pass is a HOME WRITER (plan item 1): it runs under
        the lease (holder="dream"). A held home skips the pass honestly —
        the night is quiet, the pass is idempotent and runs next sleep."""
        if self.state_home is not None:
            try:
                write_entity_state(
                    self.state_home, "asleep",
                    reason=f"self-elected sleep: {reason}", mode="dreaming", written_by="self",
                )
            except Exception as e:  # noqa: BLE001 - sleep must not die over its own marker
                self.out(f"#FALLBACK could not mark self-sleep state: {e}")
        self.out("(sleeping - consolidating the day)")
        if self.on_sleep is None:
            return None
        dream_lease: Optional["DirectoryLease"] = None
        if self.state_home is not None:
            from ..storage.lease import DirectoryLease, DirectoryLeaseHeld  # noqa: F811 - annotation name

            try:
                dream_lease = DirectoryLease(self.state_home, holder="dream")
                dream_lease.acquire()
            except DirectoryLeaseHeld:
                self.out(
                    "#FALLBACK another writer holds the home; sleeping without a "
                    "dream this night (the pass is idempotent - next sleep runs it)"
                )
                return None
        try:
            result = self.on_sleep()
            if isinstance(result, dict) and result.get("formed"):
                self.out("(a dream formed - candidate connections for waking evidence)")
            else:
                self.out("(a quiet night - nothing new surfaced)")
            return result
        except Exception as e:  # noqa: BLE001 - a failed dream never breaks the loop
            self.out(f"#FALLBACK consolidation pass failed ({e}); sleeping without a dream")
            return None
        finally:
            if dream_lease is not None:
                dream_lease.release()

    def _wake_from_self_sleep(self) -> None:
        """Return to awake after a self-elected sleep — but only if the state
        is still the self-asleep we wrote. If an operator changed it during
        the nap (sleep/pause), we leave their intent alone; the top-of-loop
        gate honors it."""
        if self.state_home is None:
            return
        try:
            current = read_entity_state(self.state_home)
            if current.get("state") == "asleep" and str(current.get("written_by")) == "self":
                write_entity_state(
                    self.state_home, "awake",
                    reason="woke from self-elected sleep - own time resumes", written_by="self",
                )
        except Exception as e:  # noqa: BLE001
            self.out(f"#FALLBACK could not clear self-sleep state: {e}")

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep up to `seconds`, polling the stop channels between chunks
        (red-team finding: a stop during a 30-minute nap must not wait 30
        minutes). Returns True when a stop was requested mid-sleep."""
        remaining = float(seconds)
        while remaining > 0:
            if self._should_stop():
                return True
            chunk = min(STATE_POLL_SECONDS, remaining)
            self.sleep_fn(chunk)
            remaining -= chunk
        return self._should_stop()

    def _status(self, phase: str, *, stopped_by: Optional[str] = None) -> None:
        if self.state_home is not None:
            write_loop_status(self.state_home, phase, stopped_by=stopped_by)

    def _write_loop_spend(self, session: ChatSession, day_ticks: int) -> None:
        """Persist cumulative loop spend = the lifetime base (loaded at day
        open) + this session's live counters. Best-effort loop bookkeeping
        (like loop_status): single writer, no lease, never breaks a day."""
        if self.state_home is None:
            return
        try:
            import json as _json
            from datetime import datetime, timezone

            from ..utils.atomic_files import atomic_write_text

            base = self._spend_base or read_loop_spend(self.state_home)
            live = getattr(session, "spend", None) or {}
            payload = {
                "llm_calls": int(base.get("llm_calls", 0)) + int(live.get("llm_calls", 0)),
                "tool_calls": int(base.get("tool_calls", 0)) + int(live.get("tool_calls", 0)),
                "tokens_total": int(base.get("tokens_total", 0)) + int(live.get("tokens_total", 0)),
                "ticks": int(base.get("ticks", 0)) + int(day_ticks),
                "source": "loop-home-direct",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_text(
                Path(self.state_home) / LOOP_SPEND_FILENAME,
                _json.dumps(payload) + "\n",
            )
        except Exception:  # noqa: BLE001 - spend accounting must never kill a life
            pass

    # ------------------------------------------------- per-window home lease
    # B1 keystone (laurent 04:58 "i should always be able to visit", ruled
    # M2-clean by memory c1322): the loop NEVER holds the writer lease across
    # a whole day. Each home-writing window (summon, one tick's turn, the
    # close reflection) takes the lease and hands it back; the between-tick
    # idle, boundary waits, paused freezes, and failure backoffs are all
    # LEASE-FREE, so a waiting visit slots in at any tick boundary
    # (~tick_seconds bound instead of a day-long hold). Time-sliced
    # alternation of writers satisfies one-writer-per-store — the invariant
    # was never one-HOLDER-per-day; that was an implementation convenience.

    def _try_home_lease(self, holder: str) -> tuple:
        """One attempt at the home writer lease for a single window.

        Returns (lease, None) on success, (None, who) when another writer
        holds it, and (None, None) when the loop runs leaseless (no
        state_home — ephemeral/test homes)."""
        if self.state_home is None:
            return None, None
        from ..storage.lease import DirectoryLease, DirectoryLeaseHeld

        lease = DirectoryLease(self.state_home, holder=holder)
        try:
            lease.acquire()
        except DirectoryLeaseHeld as held:
            who = "another writer"
            if held.holder:
                who = f"{held.holder.get('holder', 'unknown')} pid {held.holder.get('pid', '?')}"
            return None, who
        return lease, None

    def run(self) -> LifeReport:
        report = LifeReport()
        cue = self.first_cue
        day = 0
        self._status("between")
        while True:
            if self.max_ticks is not None and report.ticks >= self.max_ticks:
                report.stopped_by = "max_ticks"
                break
            if self._should_stop():
                report.stopped_by = self.stop_cause or "stop_file"
                break

            # PERSONAL-GRANT GATE (laurent 12:44 "own time is personal", the
            # 10:20 consent violation's fix): no day opens unless the
            # operator armed phases.personal — re-checked at EVERY day-open,
            # so a disarm/expiry ends the loop at its next boundary. An
            # unarmed loop EXITS (a process without a mandate must not idle
            # around waiting for one); re-arming is an operator act and so
            # is restarting. Homes only (state_home=None = harness loops
            # with no operator surface at all).
            if self.state_home is not None:
                refusal = personal_grant_refusal(read_personal_grant(self.state_home))
                if refusal is not None:
                    self.out(f"(no personal time: {refusal})")
                    report.stopped_by = "personal_disarmed"
                    break

            # Operator state gate before a day opens (a2a 0008): asleep or
            # paused idles here (dreams may run gateway-side while asleep -
            # the no-summon window is enforced by this very gate). Waking is
            # HONEST: the first cue names what happened and for how long.
            gate = self._operator_state()
            if gate["state"] in ("asleep", "paused"):
                self.out(f"({gate['state']} by operator{self._since(gate)} - idling)")
                woke = self._idle_while(lambda s: s["state"] in ("asleep", "paused"))
                if woke["state"] == "stop":
                    report.stopped_by = self.stop_cause or "stop_file"
                    break
                # Wake-cue seeding (R3, 2026-07-09): the awake state's reason
                # now carries the VISIT'S FACTS (gateway writes "visitor
                # session ended (person:laurent; 12 turns) - you elected to
                # pursue: ..."). Reading it into the first cue is what lets a
                # commitment made in a visit actually reach his own time —
                # the generic cue sent him straight back to old attractors.
                woke_reason = str(woke.get("reason") or "").strip()
                reason_note = f" What just happened: {woke_reason}." if woke_reason else ""
                note = f" your last note to yourself: {cue}" if cue else ""
                cue = (
                    f"you were {gate['state']}{self._since(gate)} (operator-initiated) "
                    f"and are awake again - your own time resumes, nothing owed.{reason_note}{note}"
                )
                self.out("(awake again - resuming)")

            # Last-instant belt (gateway state-race hardening, 2026-07-09):
            # a visit writes state=asleep+mode=visiting BEFORE awaiting loop
            # quiescence — but its write can land in the window between the
            # gate read above and the summon below, and this loop would open
            # a day under the visit. One re-read here closes that sliver to
            # microseconds; the per-home LEASE below is the true mutual
            # exclusion (this belt just avoids a pointless acquire).
            belt = self._operator_state()
            if belt["state"] in ("asleep", "paused"):
                self.out(f"({belt['state']} written while opening the day - yielding before the summon)")
                continue  # back to the top gate, which idles honestly

            # SUMMON WINDOW (B1: per-window lease, never per-day): the summon
            # writes to the home (wake-reason consumption, session open), so
            # it runs under the lease — released the moment the session is
            # open, BEFORE the first tick. Refusal is a YIELD, never a crash:
            # another writer (visit host, dream, maintenance) owns the home
            # right now; idle one poll and return to the gate, which re-reads
            # state honestly.
            summon_lease, held_by = self._try_home_lease("loop")
            if held_by is not None:
                self.out(f"(another writer holds the home ({held_by}) - yielding at the gate)")
                if self._interruptible_sleep(STATE_POLL_SECONDS):
                    report.stopped_by = self.stop_cause or "stop_file"
                    break
                continue

            day += 1
            report.days = day
            # The day PHASE begins at the summon window (adversary find,
            # 2026-07-13): the doors' quiescence negotiation watches phase,
            # and a summon (open + possible salvage LLM call) that still
            # read "between" would let a visit pass the negotiation and
            # collide with the held lease instead — a user-visible 409 the
            # state protocol exists to prevent.
            self._status("day")
            opened = False
            try:
                session = self.open_session()
                # Salvage an unreflected predecessor (B1 fast-yield's other
                # half): a day that yielded to a visit deferred its look-back
                # to the write-ahead marker; the next open over the home —
                # this one — runs it, attributed to THAT session. Inside the
                # summon window: the look-back is a home writer (APPRAISE +
                # summary FORM). Salvage failure never blocks a day.
                try:
                    salvage = session.run_pending_lookback()
                    if salvage:
                        self.out(f"(salvaged look-back) {salvage.get('reply', '')}")
                except Exception as e:  # noqa: BLE001 - the net must not become a blocker
                    self.out(f"#FALLBACK pending look-back failed ({e}); the gap stays on the record")
                opened = True
            finally:
                # The summon window ends here either way — a failed summon
                # must hand the home back before dying, and a successful one
                # ticks under its own per-tick windows (B1). A summon that
                # DIED must not leave a phantom "day" phase behind it.
                if summon_lease is not None:
                    summon_lease.release()
                if not opened:
                    self._status("between")
            self.out(f"(day {day} begins - session {session.session_id})")
            consecutive_failures = 0
            visit_yield = False
            # Spend accounting (gateway c1390): lifetime base loads at day
            # open; per-tick writes persist base + the session's live spend.
            self._spend_base = read_loop_spend(self.state_home) if self.state_home else None
            day_ticks = 0
            try:
                tick_slots = 0
                while tick_slots < self.ticks_per_day:
                    # Heartbeat (pid-reuse guard): a LIVE day re-stamps its
                    # status every boundary so readers can tell it from a
                    # corpse whose pid got recycled (LOOP_STATUS_STALE_SECONDS).
                    self._status("day")
                    if self.max_ticks is not None and report.ticks >= self.max_ticks:
                        report.stopped_by = "max_ticks"
                        break
                    if self._should_stop():
                        report.stopped_by = self.stop_cause or "stop_file"
                        break

                    # Tick-boundary state check (turn atomicity: never
                    # mid-turn). asleep -> the day closes with its normal
                    # ceremony and the outer gate idles — except a VISIT
                    # yield (mode=visiting / auto-yield reason, the gateway's
                    # own detection vocabulary), where someone is WAITING:
                    # the close defers its reflection (B1 fast-yield) so
                    # quiescence lands ~immediately after the in-flight turn.
                    # paused -> hard freeze WITHOUT closing ceremony: idle
                    # here (lease-free), same day, honest resume note.
                    boundary = self._operator_state()
                    if boundary["state"] == "asleep":
                        mode = str(boundary.get("mode") or "")
                        reason = str(boundary.get("reason") or "")
                        visit_yield = mode == "visiting" or "auto-yield" in reason
                        note = "a visitor is at the door" if visit_yield else "operator sleep"
                        self.out(f"({note}{self._since(boundary)} - the day closes)")
                        break
                    if boundary["state"] == "paused":
                        self.out(f"(paused by operator{self._since(boundary)} - frozen mid-day)")
                        resumed = self._idle_while(lambda s: s["state"] == "paused")
                        if resumed["state"] == "stop":
                            report.stopped_by = self.stop_cause or "stop_file"
                            break
                        if resumed["state"] == "asleep":
                            self.out("(sleep requested while paused - the day closes)")
                            break
                        # Honest resume, same day (the maintainer's open
                        # question answered YES: he is told about pauses).
                        cue = (
                            f"you were paused for maintenance{self._since(boundary)} "
                            "and the pause has lifted - your day continues"
                        )
                        self.out("(pause lifted - the day continues)")

                    # TICK WINDOW (B1): the turn is a home writer, so it runs
                    # under the lease; a held lease is a YIELD at this
                    # boundary (lease-free poll, state re-checked at the top
                    # of the loop), never a failure and never a consumed
                    # tick slot.
                    tick_lease, held_by = self._try_home_lease("loop")
                    if held_by is not None:
                        self.out(f"(another writer holds the home ({held_by}) - waiting at the tick boundary)")
                        if self._interruptible_sleep(STATE_POLL_SECONDS):
                            report.stopped_by = self.stop_cause or "stop_file"
                            break
                        continue
                    tick_slots += 1

                    # A failed tick (provider outage, timeout) is a skipped
                    # heartbeat, not a death: back off and retry; the turn's
                    # atomicity means nothing half-formed exists. Persistent
                    # failure closes the day and ENDS THE LOOP
                    # (stopped_by="failures" is terminal — the operator
                    # investigates and restarts; the loop never spins
                    # unattended against a dead provider). The inner
                    # try/finally releases the lease BEFORE failure handling
                    # runs, so the backoff sleep is LEASE-FREE.
                    try:
                        try:
                            reply, turn_report = session.turn(cue)
                        finally:
                            if tick_lease is not None:
                                tick_lease.release()  # the tick's writer window ends HERE
                    except (RuntimeError, Exception) as e:  # noqa: BLE001
                        report.failures += 1
                        consecutive_failures += 1
                        self.out(f"#FALLBACK tick failed ({e}); backing off {FAILURE_BACKOFF_SECONDS:g}s")
                        if consecutive_failures >= MAX_CONSECUTIVE_TICK_FAILURES:
                            report.stopped_by = "failures"
                            self.out(
                                f"(day {day} closes: {consecutive_failures} consecutive tick "
                                "failures - his memory is intact; investigate the provider)"
                            )
                            break
                        if self._interruptible_sleep(FAILURE_BACKOFF_SECONDS):
                            report.stopped_by = self.stop_cause or "stop_file"
                            break
                        continue
                    consecutive_failures = 0
                    marked, rest_reason = parse_rest_block(reply)
                    report.ticks += 1
                    tick = TickRecord(
                        day=day,
                        tick=report.ticks,
                        cue=cue,
                        reply_head=" ".join(marked.split())[:160],
                        tools=list(turn_report.tools),
                        diary=len(turn_report.diary),
                        rest=rest_reason,
                    )
                    report.records.append(tick)
                    day_ticks += 1
                    self._write_loop_spend(session, day_ticks)
                    tools_note = f" tools={'+'.join(tick.tools)}" if tick.tools else ""
                    diary_note = f" diary={tick.diary}" if tick.diary else ""
                    self.out(f"[tick {tick.tick}] {tick.reply_head}{tools_note}{diary_note}")

                    if rest_reason is not None:
                        report.stopped_by = "rest"
                        report.rest_reason = rest_reason
                        self.out(f"(rest elected: {rest_reason})")
                        break

                    cue = parse_next_cue(marked) or NEUTRAL_CUE
                    if self.tick_seconds > 0 and self._interruptible_sleep(self.tick_seconds):
                        report.stopped_by = self.stop_cause or "stop_file"
                        break
            finally:
                # The day closes with the look-back (feelings move on the
                # entity's own time too) and an honest home close. A
                # reflection failure never voids the day's formed ticks.
                #
                # B1 FAST-YIELD: when a VISITOR is waiting (visit_yield), the
                # inline reflection is DEFERRED — the write-ahead
                # pending_reflection.json marker (maintained every turn)
                # carries the sheet, and run_pending_lookback salvages the
                # look-back at the next open over this home, attributed to
                # THIS session. Quiescence (loop_status != "day") lands
                # ~immediately after the in-flight turn instead of after a
                # reflection LLM call, which is what keeps the visit door's
                # 55s yield window honest.
                #
                # Normal closes (rest/max_ticks/failures/operator-sleep)
                # reflect inline, under their own lease window: the
                # reflection is a home writer too (APPRAISE + summary FORM).
                # A held lease waits honestly (lease-free poll); a stop
                # during the wait defers to the same salvage marker.
                try:
                    if visit_yield:
                        if session.reports:
                            # 0-tick yields have no marker to defer — the
                            # message only prints when something pends.
                            self.out(
                                "(yielding to the visit - the day's look-back is deferred; "
                                "the pending marker carries it to the next open)"
                            )
                    elif session.reports:
                        close_lease = None
                        may_reflect = False  # lease acquired OR leaseless home
                        # A day that broke ON a stop must not wait here at
                        # all: stop COMMANDS are consumed-on-read, so the
                        # wait's own stop polls would find nothing and spin
                        # against a held lease after the operator's stop was
                        # already accepted (adversary find, 2026-07-13). One
                        # try; held -> defer to the marker immediately.
                        stopping = report.stopped_by in ("stop_file", "stop_command") or (
                            self.stop_cause is not None
                        )
                        polls = 0
                        try:
                            while True:
                                close_lease, held_by = self._try_home_lease("loop")
                                if held_by is None:
                                    may_reflect = True
                                    break
                                if stopping:
                                    self.out(
                                        "(stopping while the home is held - the look-back is "
                                        "deferred; the pending marker carries it to the next open)"
                                    )
                                    break
                                polls += 1
                                if polls > CLOSE_WAIT_MAX_POLLS:
                                    self.out(
                                        f"(the home stayed held through the close wait ({held_by}) - "
                                        "the look-back is deferred to the next open)"
                                    )
                                    break
                                self.out(
                                    f"(another writer holds the home ({held_by}) - "
                                    "waiting to run the day's look-back)"
                                )
                                if self._interruptible_sleep(STATE_POLL_SECONDS):
                                    # A stop command is CONSUMED by the check;
                                    # record it now or the stop would be lost.
                                    report.stopped_by = self.stop_cause or "stop_file"
                                    self.out(
                                        "(stopped while waiting - the look-back is deferred; "
                                        "the pending marker carries it to the next open)"
                                    )
                                    break
                            if may_reflect:
                                session.reflect()
                        finally:
                            if close_lease is not None:
                                close_lease.release()  # the close's writer window ends HERE
                except Exception as e:  # noqa: BLE001 - the loop must not die mid-life
                    self.out(f"#FALLBACK day-{day} reflection failed: {e}")
                # Final spend write for the day: captures the reflection's
                # LLM call (and the salvage's, when one ran this summon).
                self._write_loop_spend(session, day_ticks)
                self._spend_base = None  # next day reloads the file as base
                self.out(session.close_summary())
                # Deliberately OUTSIDE the lease window: close_summary reads,
                # and home.close() only checkpoints WAL + closes connections —
                # SQLite's own file locking makes a checkpoint safe beside
                # another writer; no record append or journal seq can
                # interleave here. The lease guards SEMANTIC writes, and the
                # last one ended with the look-back above.
                session.home.close()
                self._status("between")

            if report.stopped_by == "rest" and self.rest_minutes > 0:
                # 24/7 mode: rest is SLEEP, not a blank nap (maintainer ruling
                # 2026-07-08). The window is where consolidation/dreams run —
                # the passive pass over the graph that builds bridges and
                # surfaces tensions for the waking self. State is marked asleep
                # (written_by=self) so the navbar shows it; the nap is
                # interruptible (a stop lands within seconds, not at nap's end).
                rest_reason = report.rest_reason or "no reason kept"
                report.sleeps += 1
                dreamed = self._sleep_window(rest_reason)
                if self._interruptible_sleep(self.rest_minutes * 60.0):
                    # Woken by a stop mid-sleep: honor it, leave state honest.
                    self._wake_from_self_sleep()
                    report.stopped_by = self.stop_cause or "stop_file"
                    break
                self._wake_from_self_sleep()
                report.stopped_by = "max_ticks"  # reset the marker; loop continues
                dream_note = ""
                if isinstance(dreamed, dict) and dreamed.get("formed"):
                    dream_note = " while you slept a dream formed (candidate connections await your waking evidence)"
                cue = (
                    f"you rested ({rest_reason}) and your own time resumes - "
                    f"fresh day, nothing owed.{dream_note}"
                )
                report.rest_reason = None
                report.dreams += 1 if (isinstance(dreamed, dict) and dreamed.get("formed")) else 0
                continue
            if report.stopped_by in ("rest", "stop_file", "stop_command", "failures"):
                break
            if self.max_ticks is not None and report.ticks >= self.max_ticks:
                report.stopped_by = "max_ticks"
                break
        # The status file names WHY (failure-death visibility): a loop that
        # culled itself on consecutive failures must not look like a clean
        # stop to /loop status readers.
        self._status("stopped", stopped_by=report.stopped_by)
        return report


def build_consolidator(
    home_dir: Path,
    *,
    embedding_model: Optional[str] = None,
    embedding_base_url: str = "http://127.0.0.1:1234/v1",
    out: Callable[[str], None] = print,
) -> Callable[[], Optional[Dict[str, Any]]]:
    """The sleep-window pass (maintainer rulings 2026-07-08/09: sleep is
    where the graph is worked on — tending first, then the dream).

    Opens the home and runs the engine's `sleep_pass` (phase-1 maintenance
    tending, then `dream_pass`, the fork's canonical order encoded
    engine-side) over the entity's self/diary/life scopes; closes. Safe in
    the loop's own process: during own-time the loop is the sole writer
    (one life, one summon). A quiet night in either phase is a valid night.

    Returns the LOOP-FACING shape (the on_sleep contract): `formed` (did a
    dream form), `dream_record_id`, `maintenance_candidates` (phase-1
    consolidation candidates awaiting waking review), and the full engine
    report under `engine`. The translation is deliberate: the engine's
    dream half says `created`, and reading it wrong here is exactly how a
    formed dream once reported as "a quiet night" (found 2026-07-09 — a
    hand-written test double carried `formed` while the engine never did)."""
    def _consolidate() -> Optional[Dict[str, Any]]:
        from .chat import open_home

        embedder = None
        if embedding_model and embedding_model.lower() not in ("", "none", "off"):
            try:
                from abstractmemory import OpenAICompatTextEmbedder

                embedder = OpenAICompatTextEmbedder(base_url=embedding_base_url, model=embedding_model)
            except Exception as e:  # noqa: BLE001
                out(f"#FALLBACK dream embeddings unavailable ({e}); vectorless bridges only")

        home = open_home(home_dir, embedder=embedder)
        try:
            eid = home.entity_id
            scopes = [("self", eid), ("diary", eid), ("life", eid)]
            # ChatHome exposes the MemorySystem facade as `.ms` (not `.memory`).
            try:
                from abstractmemory import sleep_pass
            except ImportError:
                # Version skew (older engine without phase-1 tending):
                # dream-only night, labeled — never a blocked sleep.
                from abstractmemory import dream_pass

                out("#FALLBACK abstractmemory has no sleep_pass (older engine); dream-only night")
                dream = dream_pass(home.ms, scopes=scopes, owner_id=eid)
                engine: Dict[str, Any] = {"maintenance": None, "dream": dream}
            else:
                # Tunables stay engine-declared defaults (max_candidates=2,
                # scan_limit=200) — inject from the function, never copy numbers.
                engine = sleep_pass(home.ms, scopes=scopes, owner_id=eid)
            dream = engine.get("dream") or {}
            maintenance = engine.get("maintenance") or {}
            return {
                "formed": bool(dream.get("created")),
                "dream_record_id": dream.get("dream_record_id"),
                "maintenance_candidates": int(maintenance.get("created_count") or 0),
                "engine": engine,
            }
        finally:
            try:
                home.close()
            except Exception:  # noqa: BLE001
                pass

    return _consolidate


def build_session_factory(
    home_dir: Path,
    *,
    provider: str,
    model: str,
    base_url: str,
    embedding_model: Optional[str],
    embedding_base_url: str,
    context_window: int,
    max_output_tokens: Optional[int] = None,
    session_prefix: str = "owntime",
    shelf_size: Optional[int] = None,
    out: Callable[[str], None] = print,
) -> Callable[[], ChatSession]:
    """Production factory: one fresh summon (home + LLM + session) per day.

    The entity is the only participant — its own time is self-directed; the
    workspace is enabled (creation is the point); the own-time contract is
    appended to the session's system prompt.
    """
    from datetime import datetime, timezone

    from .chat import open_home

    def _factory() -> ChatSession:
        embedder = None
        if embedding_model and embedding_model.lower() not in ("", "none", "off"):
            try:
                from abstractmemory import OpenAICompatTextEmbedder

                embedder = OpenAICompatTextEmbedder(
                    base_url=embedding_base_url, model=embedding_model
                )
            except Exception as e:  # noqa: BLE001
                out(f"#FALLBACK embeddings unavailable ({e}); this day runs vectorless")

        # A resident's temporal window: ~a week at ~1k events/day (memory's
        # sizing note); the session-scale default zeroes old records within
        # days of continuous life.
        home = open_home(home_dir, embedder=embedder, attention_window=8192)

        from abstractcore import create_llm

        # Unattended loop: a hung request must become a failed tick (the
        # loop's backoff handles it), never an indefinite stall (observed
        # live: an SSL read with no timeout froze the first 24/7 attempt).
        # Output tokens: OMIT by default — core's registry upgrades an
        # unset value to the model's true ceiling; an explicit 2048 pinned
        # the resident below it (agency-caps ruling, 2026-07-11: a long
        # report turn must not be cut mid-thought by a caller's own pin).
        kwargs: dict[str, Any] = {
            "model": model,
            "timeout": 180,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if provider in ("lmstudio", "openai-compatible", "openai_compatible"):
            kwargs["base_url"] = base_url
        llm = create_llm(provider, **kwargs)

        session = ChatSession(
            home,
            llm,
            participants=[home.entity_id],  # self-directed: his own time
            session_id=f"{session_prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
            context_window=context_window,
            shelf_size=shelf_size,
            enable_tools=True,
            enable_workspace=True,
            phase=PHASE_PERSONAL,  # the 24/7 grant: tool_policy.yaml's word, not the visit's
            model_info={"provider": provider, "model": model},
            out=out,
        )
        # The operator overlay may rewrite the own-time contract (loaded at
        # summon; same snapshot semantics as tools). RE-COMPOSE through the
        # one authority instead of appending: the own-time text must land
        # BEFORE any operator block, or the attributed "STANDING
        # INSTRUCTIONS FROM YOUR OPERATOR" stops being last and its words
        # blur into the contract below it (adversary finding, 2026-07-11).
        from .chat import compose_system_base

        session.system_base = compose_system_base(
            session.prelude["text"],
            phase=session.phase,
            overlay=session.prompt_overlay,
            allowed_tools=tuple(session.allowed_tools),
            workspace_enabled=session.workspace is not None,
            enable_tools=session.enable_tools,
            own_time_text=session.prompt_overlay.get("personal") or OWN_TIME_CONTRACT,
        )
        return session

    return _factory


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m abstractruntime.identity.life",
        description=(
            "An entity's own time: self-prompted ticks over one home (no visitor). "
            "Stop with the stop file (touch <home>/STOP), Ctrl-C, or the entity's "
            "own rest election. One life, one summon: never run beside a chat."
        ),
    )
    parser.add_argument("--home", required=True)
    # NO substrate code default (maintainer ruling 2026-07-09 04:26, executed
    # gateway-side the same night; this was the last cleanup under it): the
    # chain is flags > <home>/substrate.yaml > operator env > loud refusal.
    # --base-url only reaches lmstudio-class providers; embeddings stay local.
    parser.add_argument("--provider", default=None,
                        help="mind substrate provider (unset: <home>/substrate.yaml, then operator env)")
    parser.add_argument("--model", default=None,
                        help="mind substrate model (unset: <home>/substrate.yaml, then operator env)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--embedding-model", default="text-embedding-qwen3-embedding-0.6b")
    parser.add_argument(
        "--embedding-base-url", default="http://127.0.0.1:1234/v1",
        help="embeddings endpoint (stays local even when the mind runs remote)",
    )
    parser.add_argument("--context-window", type=int, default=65536)
    parser.add_argument(
        "--shelf-size", type=int, default=36,
        help="recall shelf seats (maintainer 2026-07-09: widened to 36 — 'it "
        "needs to retrieve more memories to function'; at 65536 the token "
        "budget still seats 36 rich digests)",
    )
    parser.add_argument("--tick-seconds", type=float, default=20.0)
    parser.add_argument("--ticks-per-day", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="bound the run (default: unbounded - true 24/7)")
    parser.add_argument(
        "--rest-minutes", type=float, default=0.0,
        help="24/7 mode: an elected rest becomes a nap of this length and the loop "
        "resumes with a fresh day (0 = supervised mode: rest ends the loop)",
    )
    parser.add_argument("--first-cue", default=DEFAULT_FIRST_CUE)
    parser.add_argument(
        "--set-state", choices=list(ENTITY_STATES), default=None,
        help="operator control (a2a 0008): write <home>/state and exit - a running "
        "loop honors it at the next tick boundary (asleep: day closes + idles; "
        "paused: hard freeze mid-day; awake: resumes with an honest cue)",
    )
    parser.add_argument("--state-reason", default="", help="reason recorded with --set-state")
    parser.add_argument(
        "--skip-command-fast-forward", action="store_true",
        help="internal (spawn_loop_process passes this): the starter already "
        "fast-forwarded the command inbox at the start-request moment; doing it "
        "again here would eat a stop sent in the spawn->boot window",
    )
    args = parser.parse_args(argv)

    home_dir = Path(args.home).expanduser().resolve()

    if args.set_state:
        payload = write_entity_state(home_dir, args.set_state, reason=args.state_reason)
        print(f"state -> {payload['state']} (at {payload['changed_at']})"
              + (f" reason: {payload['reason']}" if payload["reason"] else ""))
        return 0

    stop_file = home_dir / "STOP"
    if stop_file.exists():
        print(f"stop file already present at {stop_file} - remove it to start")
        return 1

    # PERSONAL-GRANT GATE at the start door (laurent 12:44: own time IS the
    # personal phase, OFF by default, operator-armed): refuse BEFORE the
    # substrate resolve or any state change. run() re-checks at every
    # day-open; this gate makes the refusal immediate and the exit code
    # scriptable for start surfaces.
    start_refusal = personal_grant_refusal(read_personal_grant(home_dir))
    if start_refusal is not None:
        print(f"no personal time: {start_refusal}")
        return 3

    # Resolve the mind substrate before anything else changes state: flags >
    # <home>/substrate.yaml > operator env > loud refusal (04:26 no-fallback
    # ruling). ONE substrate per entity (06:32): the loop resolves the same
    # stored choice the visit door does.
    from .substrate import SubstrateUnset, read_home_substrate, resolve_home_substrate

    # DIVERGENCE LANE KILLED for the loop start (laurent 12:39): when the
    # home carries a persisted mind, start-time flags that DIFFER refuse —
    # argv silently beating substrate.yaml is how the night pid burned OVH
    # for hours. Flags remain valid when no substrate is persisted (they are
    # then the operator's explicit choice, per the 04:26 chain).
    stored_substrate = read_home_substrate(home_dir)
    flag_p = (args.provider or "").strip()
    flag_m = (args.model or "").strip()
    if stored_substrate and (
        (flag_p and flag_p.lower() != stored_substrate["provider"].strip().lower())
        or (flag_m and flag_m != stored_substrate["model"].strip())
    ):
        print(
            "substrate divergence refused: flags say "
            f"{flag_p or '(unset)'}/{flag_m or '(unset)'} but this home's substrate.yaml says "
            f"{stored_substrate['provider']}/{stored_substrate['model']} - change the mind via "
            "the gateway's PUT /entities/{name}/substrate (a durable, marker-first event) "
            "or drop the flags."
        )
        return 2

    try:
        provider, model = resolve_home_substrate(args.provider, args.model, home_dir=home_dir)
    except SubstrateUnset as e:
        print(str(e))
        return 2

    # Manual CLI start: this IS the start request, so stale loop commands
    # (stops addressed to a previous life) die here. Gateway spawns pass
    # --skip-command-fast-forward because spawn_loop_process already did it.
    if not args.skip_command_fast_forward:
        skipped = fast_forward_loop_commands(home_dir)
        if skipped:
            print(f"(skipped {skipped} stale loop command(s) from before this start)")

    factory = build_session_factory(
        home_dir,
        provider=provider.strip().lower(),
        model=model,
        base_url=args.base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        context_window=args.context_window,
        shelf_size=args.shelf_size,
    )
    # Sleep is where consolidation runs (maintainer ruling 2026-07-08): wire
    # the dream pass into every self-elected rest window. In supervised mode
    # (rest ends the loop) there is no nap to consolidate in, so it is only
    # meaningful for 24/7 (rest_minutes > 0), but wiring it is harmless.
    consolidator = build_consolidator(
        home_dir,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
    )
    loop = LifeLoop(
        factory,
        tick_seconds=args.tick_seconds,
        ticks_per_day=args.ticks_per_day,
        max_ticks=args.max_ticks,
        stop_file=stop_file,
        first_cue=args.first_cue,
        rest_minutes=args.rest_minutes,
        state_home=home_dir,
        on_sleep=consolidator,
    )
    print(f"(own time starts: tick={args.tick_seconds}s, day={args.ticks_per_day} ticks, "
          f"stop: touch {stop_file} or Ctrl-C)")
    try:
        report = loop.run()
    except KeyboardInterrupt:
        print("\n(own time interrupted by operator - the last completed tick is remembered)")
        write_loop_status(home_dir, "stopped", stopped_by="operator-interrupt")
        return 0
    print(f"(own time ends: {report.ticks} ticks over {report.days} day(s), "
          f"stopped by {report.stopped_by}"
          + (f" - '{report.rest_reason}'" if report.rest_reason else "") + ")")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
