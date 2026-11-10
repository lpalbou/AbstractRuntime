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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .chat import (
    ENTITY_LLM_READ_IDLE_TIMEOUT_S,
    ENTITY_LLM_TIMEOUT_FLOOR_S,
    ChatHome,
    ChatSession,
    TurnCancelled,
    _entity_patience_window,
)
from .lifecycle_fence import (
    KILL_SWITCH,
    RESUME,
    LifecycleFence,
    LifecycleSignalUnwritten,
    LifecycleUnreadable,
    StaleLifecycleGeneration,  # noqa: F401 - re-exported: the state writers' refusal (f881cac's name)
    StoppedByOperator,
)
from .lifecycle_fence import publish_under_lock as _publish_generation_under_lock
from .lifecycle_fence import read_truth as _read_generation_truth
from .lifecycle_fence import signal_bump as _signal_bump
from .lifecycle_fence import verify as _verify_fence
from .tool_policy import PHASE_PERSONAL, PHASE_SLEEP, PHASE_WORK

# The own-time note is INFORMATIONAL, never a mission. It describes the
# environment and the affordances that exist; it does not tell the entity
# what to want or how to spend its time. Free time is where the entity's
# own evolution happens — imposing curiosity, productivity, or any "point"
# here biases the very thing persistence exists to let us observe. What we
# provide is a safe place and honest information about what is possible; the
# choosing is the entity's. (Maintainer ruling 2026-07-08: "we shouldn't and
# can't enforce laws or missions during their free time.")
# THE WORK LANE (laurent, room seq 155, 2026-07-19: "the entity must be
# able to work and execute commands when it works"; iteration-1: tasks
# left with an entity mean the WORK phase). The work order is a HOME file
# (<home>/work_order.md) — operator-owned like tool_policy.yaml: the
# console/CLI writes it, the loop's day-open reads it, and its PRESENCE
# is what shifts the day to phase=work (work grant incl. execute_command
# where the operator's matrix says so). The entity may declare completion
# (```work done) — the order archives with a timestamp, visible, never
# deleted. Unlike own time, work IS a mission: the contract below says so
# honestly instead of pretending the task is his own idea.
WORK_ORDER_FILENAME = "work_order.md"

WORK_CONTRACT = """This is your work time. Your operator left you the task below - it is
real, it was chosen for you, and finishing it is the point of this phase.
Work it with your tools; keep what you learn. When the task is DONE, say
so with a fenced block so your time returns to you:

```work
done: <one line - what stands finished and where to look>
```

If you cannot finish - blocked, missing something, wrong premise - say
that too (the same block, starting "blocked:" instead of "done:"). An
honest blocked is worth more than a pretended done."""


# #[WARNING:TIMEOUT] Entity PATIENCE WINDOW (ADR-0027 §3): ONE definition,
# in chat.py, shared by the visit lane, the own-time loop and the sleep
# narrator — 1800s minimum per attempt, retry budget never shorter than one
# attempt (operator ruling 2026-11-07). Read its comment for the rule.


def read_work_order(home_dir: Path) -> Optional[str]:
    """The standing work order, or None. Pure read; unreadable = None with
    the day falling back to personal (a broken order file must never kill
    a day-open)."""
    try:
        p = Path(home_dir) / WORK_ORDER_FILENAME
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8").strip()
        return text or None
    except Exception:  # noqa: BLE001
        return None


def archive_work_order(home_dir: Path, *, verdict: str) -> None:
    """Move the standing order to work_order.done.md with the entity's
    verdict line appended — visible history, never deletion."""
    try:
        from datetime import datetime, timezone

        p = Path(home_dir) / WORK_ORDER_FILENAME
        if not p.exists():
            return
        text = p.read_text(encoding="utf-8")
        done = Path(home_dir) / WORK_ORDER_ARCHIVE_FILENAME
        stamp = f"\n\n---\n[{datetime.now(timezone.utc).isoformat()}] {verdict.strip()}\n"
        with done.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + stamp)
        p.unlink()
    except Exception:  # noqa: BLE001 - archiving must never kill the day
        pass


_WORK_BLOCK_RE = re.compile(r"```work[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


WORK_ORDER_ARCHIVE_FILENAME = "work_order.done.md"


def read_work_order_verdicts(home_dir: Path) -> List[Dict[str, str]]:
    """The verdicts `archive_work_order` appended, oldest first:
    `[{"at": <ISO>, "verdict": "done"|"blocked", "line": <the verdict line>}]`.

    Reads back the writer's own layout (a blank line, `---`, then
    `[<ISO time>] <verdict line>`), and the verdict word is the line's own
    `done:` / `blocked:` prefix that `parse_work_verdict` already required.
    A missing archive is `[]`. Nothing here reads the order's text; an order
    whose text itself contains that exact three-line layout would be read as
    a verdict (the archive has no stronger delimiter)."""
    from datetime import datetime

    path = Path(home_dir) / WORK_ORDER_ARCHIVE_FILENAME
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    out: List[Dict[str, str]] = []
    for i in range(2, len(lines)):
        line = lines[i]
        if lines[i - 1] != "---" or lines[i - 2] != "" or not line.startswith("["):
            continue
        close = line.find("] ")
        if close < 0:
            continue
        at = line[1:close]
        verdict_line = line[close + 2:].strip()
        try:
            datetime.fromisoformat(at)
        except ValueError:
            continue
        head = verdict_line.lower()
        if head.startswith("done:"):
            out.append({"at": at, "verdict": "done", "line": verdict_line})
        elif head.startswith("blocked:"):
            out.append({"at": at, "verdict": "blocked", "line": verdict_line})
    return out


# ----------------------------------------------------------- the job queue (CONTRACT v3 D1/D4, operator 2026-11-09)
# WORK IS DRIVEN BY THE JOBS QUEUED FOR HIM, never by a toggle: a job queued moves him to work (from idle, and from
# his own time, which the job interrupts), and when every job is done he returns to his own time if it is still
# granted, else idle. The loop's jobs are the two durable desks in his home:
#   - the standing work order (`work_order.md`, the operator's; archived by his verdict), worked first;
#   - the tasks of the task inbox (`task_inbox.jsonl`, written by AbstractGateway's entity_tasks.py - the file schema
#     is the cross-package contract) whose folded status is `pending` or `taken`; a taken one (already begun) first,
#     then the pending ones in the order they were left. `parked` and `done` tasks are not queued.
# A work day opens on the head of the queue; a task it opens is marked `taken`, and his ```work verdict marks it
# `done` (done:) or `parked` (blocked:) with the verdict line as its note. Gateway runs (summons, automations) are the
# gateway's work, not this loop's.
TASK_INBOX_FILENAME = "task_inbox.jsonl"
QUEUED_TASK_STATUSES = ("taken", "pending")
_TASK_STATUSES = ("pending", "taken", "done", "parked")


def _task_inbox_fold(home_dir: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """The task inbox folded like AbstractGateway's read_task_inbox (append-only events; the newest known status of a
    task is its status; an unparseable line or a status for an unknown task is skipped and counted). Returns (tasks in
    file order, a labeled note when lines were skipped or the file is unreadable). No file = ([], None)."""
    import json as _json

    path = Path(home_dir) / TASK_INBOX_FILENAME
    if not path.is_file():
        return [], None
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:  # noqa: BLE001
        return [], f"#FALLBACK task inbox unreadable ({e})"
    tasks: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    skipped = 0
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = _json.loads(raw)
        except ValueError:
            skipped += 1
            continue
        tid = str(ev.get("task_id") or "") if isinstance(ev, dict) else ""
        if not tid:
            skipped += 1
            continue
        if ev.get("event") == "added":
            if tid not in tasks:
                order.append(tid)
            tasks[tid] = {"task_id": tid, "title": str(ev.get("title") or ""), "brief": str(ev.get("brief") or ""),
                          "by": str(ev.get("by") or ""), "origin": str(ev.get("origin") or ""), "status": "pending"}
        elif ev.get("event") == "status" and tid in tasks:
            word = str(ev.get("status") or "").strip().lower()
            if word in _TASK_STATUSES:
                tasks[tid]["status"] = word
        else:
            skipped += 1
    note = f"#FALLBACK {skipped} unparseable/orphan task inbox line(s) skipped" if skipped else None
    return [tasks[t] for t in order], note


def read_job_queue(home_dir: Path) -> Dict[str, Any]:
    """The jobs queued for him, in the order he works them: {"jobs": [...], "jobs_queued": n, "note"?}. Each job is
    {"kind": "work_order", "id": "work_order.md", "title", "text"} or {"kind": "task", "id": <task_id>, "title",
    "text": <brief>, "by", "status"}. Pure read."""
    jobs: List[Dict[str, Any]] = []
    order = read_work_order(home_dir)
    if order:
        jobs.append({"kind": "work_order", "id": WORK_ORDER_FILENAME, "title": order.splitlines()[0].strip(),
                     "text": order})
    tasks, note = _task_inbox_fold(home_dir)
    for status in QUEUED_TASK_STATUSES:
        jobs.extend({"kind": "task", "id": t["task_id"], "title": t["title"], "text": t["brief"], "by": t["by"],
                     "status": t["status"]} for t in tasks if t["status"] == status)
    out: Dict[str, Any] = {"jobs": jobs, "jobs_queued": len(jobs)}
    if note:
        out["note"] = note
    return out


def jobs_queued(home_dir: Path) -> int:
    """How many jobs are queued for him (the work order and the pending/taken tasks). Pure read."""
    return int(read_job_queue(home_dir)["jobs_queued"])


def append_task_status(home_dir: Path, task_id: str, status: str, *, by: str, note: str = "") -> Dict[str, Any]:
    """Append one `status` event to the task inbox - the same flock-guarded single-line append AbstractGateway's
    entity_tasks._append_event makes (two writer processes: the gateway door and this loop). An unknown task id or
    status word is refused (KeyError / ValueError) before anything is written."""
    import json as _json
    import os

    from ..core.runtime import utc_now_iso

    word = str(status or "").strip().lower()
    if word not in _TASK_STATUSES:
        raise ValueError(f"unknown task status {status!r} (one of {_TASK_STATUSES})")
    tasks, _note = _task_inbox_fold(home_dir)
    if str(task_id) not in {t["task_id"] for t in tasks}:
        raise KeyError(f"no task {task_id!r} in {TASK_INBOX_FILENAME}")
    event: Dict[str, Any] = {"event": "status", "task_id": str(task_id), "status": word, "at": utc_now_iso(),
                             "by": str(by)}
    if str(note or "").strip():
        event["note"] = str(note).strip()
    line = _json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        import fcntl
    except ImportError:  # non-POSIX: a plain single-line append, as the gateway's writer degrades
        fcntl = None  # type: ignore[assignment]
    with open(Path(home_dir) / TASK_INBOX_FILENAME, "a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return event


TASK_CONTRACT_INTRO = "The task below was left for you"


def work_day_text(job: Dict[str, Any]) -> str:
    """The work day's contract with its job: the operator's order as before; a task names who left it."""
    if job.get("kind") == "task":
        contract = WORK_CONTRACT.replace("Your operator left you the task below", TASK_CONTRACT_INTRO, 1)
        by = str(job.get("by") or "").strip()
        head = f"THE TASK, left by {by}:" if by else "THE TASK:"
        title = str(job.get("title") or "").strip()
        body = str(job.get("text") or "").strip()
        return contract + "\n\n" + head + "\n" + "\n\n".join(x for x in (title, body) if x)
    return WORK_CONTRACT + "\n\nTHE TASK, from your operator:\n" + str(job.get("text") or "")


# ----------------------------------------------------------- the day gate
# THE DAY GATE (contract-phases v2 §10.3-§10.5; TARGET 2026-11-08 19:02/19:11; spec v23 OWN-TIME-IS-A-PRIVILEGE). One
# decision per boundary, in this order:
#   1. a visit posture -> the gate yields (the door owns the visit).
#   2. HIS OWN SLEEP DECISION (a ```rest block he wrote in the day that just closed) -> sleep, cause self_elected,
#      unless the last activity phase was already sleep (no sleep->sleep) or it came from own time the operator has
#      since revoked (PERSONAL-OFF-NEVER-SLEEPS). Nothing else puts him to sleep: no clock, no gate leg, no toggle.
#   3. a job queued (the standing work order, a pending/taken task) -> work, whatever the personal grant says (the
#      gate reads the queue before any grant check, dm#89). There is no work toggle (CONTRACT v3 D1, operator
#      2026-11-09): work starts when a job is queued and nothing else admits or refuses it.
#   4. the personal grant armed (the operator's privilege) -> personal: a day when something pulls (standing drives),
#      the first day after the operator's start, else he RESTS between days inside personal (awake, zero LLM).
#   5. nothing allowed -> idle: own time ends AWAKE (the loop exits; only the operator's grant or a queued job starts
#      it again). Right after a work day this is `jobs_done`: every job is done and there is no own time to return to.
# The retired v8-v13 machinery is gone: the settled-desk sleep leg, the 6 h unattended need-check, the 2 h grant-use
# floor and the timed personal cycle (their dials stay readable, wired:false in the blueprint).
UNATTENDED_NEED_CHECK_SECONDS = 6 * 3600  # retired (v22); kept for readers of the constant
PERSONAL_USE_FLOOR_SECONDS = 2 * 3600  # retired (v22); kept for readers of the constant
_PERSONAL_USAGE_FILE = "personal_usage.json"
# v12 P1-4 cycle clock: retired with the timed cycle (v22); the read/write helpers stay for readers.
_CYCLE_CLOCK_FILE = "personal_cycle_clock.json"


def read_cycle_clock(home_dir: Path) -> float:
    import json as _json

    try:
        raw = _json.loads((Path(home_dir) / _CYCLE_CLOCK_FILE).read_text(encoding="utf-8"))
        return max(0.0, float(raw.get("seconds_since_sleep") or 0.0))
    except Exception:  # noqa: BLE001 - absent/corrupt reads as fresh
        return 0.0


def write_cycle_clock(home_dir: Path, seconds: float) -> None:
    import json as _json

    try:
        from ..utils.atomic_files import atomic_write_text

        atomic_write_text(
            Path(home_dir) / _CYCLE_CLOCK_FILE,
            _json.dumps({"seconds_since_sleep": max(0.0, float(seconds))}) + "\n",
        )
    except Exception:  # noqa: BLE001 - bookkeeping never kills a loop
        pass


def read_personal_usage(home_dir: Path) -> float:
    """Seconds of personal time used against the CURRENT grant (0.0 when
    the meter is absent, corrupt, or keyed to a superseded grant)."""
    import json as _json

    try:
        raw = _json.loads((Path(home_dir) / _PERSONAL_USAGE_FILE).read_text(encoding="utf-8"))
        grant = read_personal_grant(home_dir)
        if str(raw.get("granted_at") or "") != str(grant.get("granted_at") or ""):
            return 0.0  # new grant = fresh floor
        return max(0.0, float(raw.get("seconds_used") or 0.0))
    except Exception:  # noqa: BLE001 - absent/corrupt reads as unused
        return 0.0


def record_personal_usage(home_dir: Path, seconds: float) -> None:
    """Add lived personal-day seconds to the meter (best-effort; the loop
    must never die over bookkeeping)."""
    import json as _json

    try:
        grant = read_personal_grant(home_dir)
        key = str(grant.get("granted_at") or "")
        current = 0.0
        p = Path(home_dir) / _PERSONAL_USAGE_FILE
        try:
            raw = _json.loads(p.read_text(encoding="utf-8"))
            if str(raw.get("granted_at") or "") == key:
                current = max(0.0, float(raw.get("seconds_used") or 0.0))
        except Exception:  # noqa: BLE001
            pass
        from ..utils.atomic_files import atomic_write_text

        atomic_write_text(p, _json.dumps({
            "granted_at": key,
            "seconds_used": current + max(0.0, float(seconds)),
        }) + "\n")
    except Exception:  # noqa: BLE001
        pass


def sleep_refusal(
    home_dir: Path, *, last_phase: Optional[str], from_phase: Optional[str] = None
) -> Optional[str]:
    """Why his sleep decision cannot be honored now (None = it can). Pure read.

    - sleep -> sleep is refused (TARGET 19:11, spec NO-SLEEP-AFTER-SLEEP): after a sleep the next leg is personal,
      work or a visit; `last_phase` is the last ACTIVITY phase he lived (a visit's close back into the interrupted
      sleep is the same sleep, never a second one).
    - a decision made during own time the operator has since revoked is dropped: personal off ends own time awake,
      never asleep (PERSONAL-OFF-NEVER-SLEEPS). A decision made at work keeps the work->sleep#self_elected right.
    - the effective graph must carry the edge (`from_phase`->sleep#self_elected)."""
    if str(last_phase or "") == PHASE_SLEEP:
        return "a sleep cannot follow a sleep - after a sleep he goes to personal, work or a visit"
    origin = str(from_phase or last_phase or "")
    if origin == PHASE_PERSONAL and personal_grant_refusal(read_personal_grant(home_dir)) is not None:
        return "his own time was turned off - he stays awake, never asleep"
    if origin in (PHASE_PERSONAL, PHASE_WORK):
        try:
            from .phase_graph import load_effective_graph

            graph, _w = load_effective_graph(home_dir)
            if graph.legal_to(origin, PHASE_SLEEP, "self_elected") is None:
                return f"the blueprint carries no {origin}->sleep#self_elected edge"
        except Exception:  # noqa: BLE001 - an unreadable graph never blocks his own election (locked edge)
            pass
    return None


def read_day_gate(
    home_dir: Path,
    *,
    skip_phases: frozenset = frozenset(),
    sleep_elected: Optional[str] = None,
    last_phase: Optional[str] = None,
    first_day: bool = False,
    return_to: Optional[str] = None,
) -> Dict[str, Any]:
    """One decision per day boundary: {phase, cause, detail, note?, ...}.

    phase: "visit" | "sleep" | "work" | "personal" | "idle" (see the order above). cause is the trace word served in
    loop_status.day_cause: visit_open | self_elected | work_order | task | start | drives | grant_degraded | resting |
    no_grant | no_return | jobs_done | edge_removed. A work decision carries `job` (the head of read_job_queue, whole)
    and `jobs_queued`. A "personal" decision with `rest: True` opens no day: he rests between days.
    "idle" means own time ends awake. Pure read.

    `sleep_elected`: his own reason from a ```rest block (the decision); `last_phase`: the last activity phase he
    lived (personal / work / sleep, None at a start); `first_day`: the operator just started own time (the start IS
    the pull, the first day is unconditional); `return_to`: the phase the caller remembers returning to.

    OWN TIME IS THE OPERATOR'S PRIVILEGE (spec v23; Codex review 2026-11-09 #2): with the grant armed, personal is
    admitted only by the operator's start (`first_day`), as a continuation of personal (`last_phase` personal), or
    as a remembered return to it (`return_to` personal, or the home's resume_to bucket / the gateway sidecar) -
    never from drives alone. Otherwise own time ends awake (idle, cause no_return). `skip_phases` (graph-edit build c4837): the caller's graph consult
    found a landing removed - the chain falls to the next legal landing; the floor is idle (never sleep)."""
    try:
        state = read_entity_state(home_dir)
        if str(state.get("mode") or "").strip().lower() == "visiting":
            return {"phase": "visit", "cause": "visit_open",
                    "detail": "a visit session is open - the gate yields"}
    except Exception:  # noqa: BLE001 - an unreadable state never blocks a day
        pass
    notes: List[str] = []
    if sleep_elected is not None:
        refused = sleep_refusal(home_dir, last_phase=last_phase)
        if refused is None:
            return {"phase": PHASE_SLEEP, "cause": "self_elected", "detail": str(sleep_elected),
                    "reason": str(sleep_elected)}
        notes.append(f"his sleep decision is not taken: {refused}")
    queue = read_job_queue(home_dir)
    if queue.get("note"):
        notes.append(str(queue["note"]))
    jobs = queue["jobs"]
    if jobs and PHASE_WORK not in skip_phases:
        job = jobs[0]
        out_work: Dict[str, Any] = {
            "phase": PHASE_WORK, "cause": job["kind"],
            "detail": WORK_ORDER_FILENAME if job["kind"] == "work_order" else f"task {job['id']}: {job['title']}",
            "job": dict(job),
            "jobs_queued": len(jobs),
        }
        if job["kind"] == "work_order":
            out_work["work_order"] = job["text"]
        if notes:
            out_work["note"] = "; ".join(notes)
        return out_work
    # The boundary right after a work day with nothing queued: every job is done. He returns to his own time only
    # while it is granted and remembered (below); otherwise he ends idle with that stated reason (jobs_done).
    jobs_done = str(last_phase or "") == PHASE_WORK and not jobs
    if jobs and str(last_phase or "") == PHASE_WORK:
        # A job is still queued but the blueprint removed this boundary's work landing: spec v25 has no edge out of
        # work to personal while a job is queued (work->personal#operator went with the work toggle; #no_task needs
        # the queue empty), so the floor is idle.
        idle_skip: Dict[str, Any] = {"phase": "idle", "cause": "edge_removed", "grant_cause": None,
                                     "detail": "the blueprint removed this boundary's work landing and a job is "
                                               "still queued - no other landing out of work is legal"}
        if notes:
            idle_skip["note"] = "; ".join(notes)
        return idle_skip
    grant = read_personal_grant(home_dir)
    grant_refusal = personal_grant_refusal(grant)
    if grant_refusal is not None or PHASE_PERSONAL in skip_phases:
        if grant_refusal is None:
            cause, detail = "edge_removed", "the blueprint removed this boundary's personal landing"
        elif jobs_done:
            cause, detail = "jobs_done", f"all jobs are done and his own time is not granted ({grant_refusal})"
        else:
            cause, detail = "no_grant", grant_refusal
        idle: Dict[str, Any] = {"phase": "idle", "cause": cause, "detail": detail,
                                "grant_cause": personal_grant_end_cause(grant) if cause == "no_grant" else None}
        if notes:
            idle["note"] = "; ".join(notes)
        return idle
    if not (
        first_day
        or str(last_phase or "") == PHASE_PERSONAL
        or str(return_to or "") == PHASE_PERSONAL
        or read_resume_to(home_dir).get("phase") == PHASE_PERSONAL
    ):
        idle = {"phase": "idle", "cause": "jobs_done" if jobs_done else "no_return",
                "detail": (("all jobs are done and " if jobs_done else "")
                           + "own time is the operator's privilege: no start, no continuation of personal and no "
                           "remembered return to it"),
                "grant_cause": None}
        if notes:
            idle["note"] = "; ".join(notes)
        return idle
    total_open = None
    degraded_note = None
    try:
        from abstractmemory import SQLiteJournal, SQLiteTripleStore
        from abstractmemory.drive_pressure import drive_pressure

        entity_id = _manifest_entity_id(home_dir)
        db = Path(home_dir) / "memory.sqlite3"
        store = SQLiteTripleStore(db)
        journal = SQLiteJournal(db)
        try:
            pressure = drive_pressure(
                store, journal,
                scopes=[("self", entity_id), ("diary", entity_id), ("life", entity_id)],
            )
            total_open = int(pressure.get("total_open") or 0)
        finally:
            try:
                store.close()
                journal.close()
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        degraded_note = "#FALLBACK drive read unavailable (engine predates drive_pressure)"
    except Exception as e:  # noqa: BLE001
        degraded_note = f"#FALLBACK drive read failed: {e}"
    if degraded_note:
        notes.append(degraded_note)
    note = "; ".join(notes) if notes else None
    if first_day:
        return {"phase": PHASE_PERSONAL, "cause": "start",
                "detail": "the operator started his own time - the first day is his", "note": note}
    if total_open is None:
        # LOUD DEGRADE: pre-drives behavior (armed grant -> day), labeled.
        return {"phase": PHASE_PERSONAL, "cause": "grant_degraded",
                "detail": degraded_note or "", "note": note}
    if total_open > 0:
        return {"phase": PHASE_PERSONAL, "cause": "drives",
                "detail": f"{total_open} standing drive(s)", "total_open": total_open, "note": note}
    return {"phase": PHASE_PERSONAL, "cause": "resting", "rest": True,
            "detail": "nothing pulls now - he rests between days inside his own time", "note": note}


def consult_gate_landing(
    home_dir: Path,
    from_phase: str,
    decision: Dict[str, Any],
    cause: str,
    out: Any,
    **gate_kwargs: Any,
) -> Tuple[Dict[str, Any], str]:
    """Graph consult over a gate decision (build order c4837; adversary A
    §3.2's day-gate row): the gate decides, the EFFECTIVE GRAPH governs.

    - REMOVED edge (legal_to -> None): the leg is skipped and the gate
      re-reads with that phase excluded (same `gate_kwargs`) — the chain falls
      to the next legal landing; the floor is idle (own time ends awake,
      contract-phases v2 §10; never a sleep).
    - REDIRECT: the landing's target substitutes BEFORE any write — with
      R8's guards-travel rule enforced here (a redirect into personal
      requires the armed grant; into work requires a queued job;
      refused redirects fall back to the skip path, loudly).
    - INSTRUCTION: returned as the provenance-stamped cue line (STEERING,
      never law) for the caller to ride on the day cue.

    Returns (decision, instruction_cue). Only work/personal landings are
    consulted (visit is the door's, sleep is his own election - locked, idle
    is no edge). Any consult failure degrades loudly to the gate's own
    decision — the graph must never wedge a boundary."""
    instruction = ""
    try:
        from .phase_graph import instruction_cue, load_effective_graph

        graph, g_warns = load_effective_graph(home_dir)
    except Exception as e:  # noqa: BLE001 - a broken graph never blocks a day
        out(f"#FALLBACK graph consult unavailable ({e}); the gate decision stands")
        return decision, instruction
    for w in g_warns:
        out(f"({w})")
    skipped: set = set(gate_kwargs.pop("skip_phases", None) or ())

    def _regate(reason: str) -> Dict[str, Any]:
        try:
            return read_day_gate(home_dir, skip_phases=frozenset(skipped), **gate_kwargs)
        except Exception as e:  # noqa: BLE001
            out(f"#FALLBACK re-gate after skip failed ({e}); own time ends awake")
            return {"phase": "idle", "cause": "edge_removed", "detail": reason}

    for _hop in range(3):  # work -> personal -> idle floor, bounded
        target = str(decision.get("phase") or "")
        if target not in (PHASE_WORK, PHASE_PERSONAL):
            return decision, instruction
        landing = graph.legal_to(from_phase, target, cause)
        if landing is None:
            out(
                f"(blueprint: {from_phase}->{target}#{cause} is not in the effective "
                "graph - the leg is skipped)"
            )
            skipped.add(target)
            decision = _regate(f"{from_phase}->{target}#{cause} removed")
            cause = _boundary_cause(from_phase, str(decision.get("phase") or ""), home_dir) or cause
            continue
        if landing.to != target:
            # R8 guards-travel: the substituted target's own invariants hold.
            if landing.to == PHASE_PERSONAL and personal_grant_refusal(
                read_personal_grant(home_dir)
            ) is not None:
                out(
                    f"(blueprint redirect {from_phase}->{target}#{cause} -> personal "
                    "REFUSED: the grant is not armed - the guard travels with the arrow)"
                )
                skipped.add(target)
                decision = _regate("redirect refused")
                continue
            if landing.to == PHASE_WORK and not jobs_queued(home_dir):
                out(
                    f"(blueprint redirect {from_phase}->{target}#{cause} -> work "
                    "REFUSED: no job queued - a work day needs a desk)"
                )
                skipped.add(target)
                decision = _regate("redirect refused")
                continue
            if landing.to not in (PHASE_WORK, PHASE_PERSONAL):
                out(
                    f"(#NOTE blueprint redirect {from_phase}->{target}#{cause} -> {landing.to!r} "
                    "ignored - a day boundary opens work or personal only)"
                )
                return decision, instruction
            out(
                f"(blueprint redirect: {from_phase}->{target}#{cause} lands "
                f"{landing.to} instead - {landing.provenance})"
            )
            redirected = dict(decision)
            redirected["phase"] = landing.to
            redirected["redirected_from"] = target
            redirected["cause"] = decision.get("cause") or cause
            redirected.pop("rest", None)
            decision = redirected
        cue_line = instruction_cue(landing)
        if cue_line:
            instruction = cue_line
        return decision, instruction
    return decision, instruction


def _boundary_cause(from_phase: Optional[str], to: str, home_dir: Path) -> Optional[str]:
    """The spec cause word of a LOOP-written day-boundary edge (spec v25 transitions; None = no loop edge: a start,
    a visit's close, idle). work->work#task_complete and personal->personal#personal_cycle are the self-loops;
    work->personal is #no_task, the only edge back to his own time once every job is done (v25 removed
    work->personal#operator with the work toggle; read_day_gate never lands personal from work while a job is
    queued); personal->work#operator (a job interrupts own time); sleep->personal/work#maintenance_done (his own
    sleep's end)."""
    frm = str(from_phase or "")
    if frm == PHASE_PERSONAL:
        return {"personal": "personal_cycle", "work": "operator"}.get(to)
    if frm == PHASE_WORK:
        if to == PHASE_WORK:
            return "task_complete"
        if to == PHASE_PERSONAL:
            return "no_task"
        return None
    if frm == PHASE_SLEEP and to in (PHASE_PERSONAL, PHASE_WORK):
        return "maintenance_done"
    return None


def append_phase_changed(
    home_dir: Path,
    *,
    from_phase: str,
    to: str,
    cause: str,
    written_by: str,
    provenance: str = "structural",
    fence: Any,
) -> None:
    """The `phase_changed` marker — SPELLED with the graph-edit build
    (c4837; it was 'RESERVED AND UNSPELLED' since the v12 marker contract).

    Loop-written transitions land one machine-readable row in the same
    append-only biography file the state writes ride
    (<home>/state_history.jsonl), distinguishable by the `marker` key:
    {marker: phase_changed, from, to, cause, written_by, at, provenance}.
    `cause` is the spec cause word where one exists (self_elected,
    personal_cycle, task_complete, maintenance_done...) and the gate's trace word otherwise
    (settled_desk, work_order — the evaluator internals, honestly named).
    Best-effort: a marker append must never kill a transition. Fenced (the
    file is the one the Stop's paused row lands in): after a Stop it refuses
    (StoppedByOperator), checked inside the home state lock."""
    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, "appending a phase_changed marker", read_lifecycle_generation(home_dir),
                      sentinel_ok=False)
        _append_phase_changed_locked(home_dir, from_phase=from_phase, to=to, cause=cause, written_by=written_by,
                                     provenance=provenance)


def _append_phase_changed_locked(home_dir: Path, *, from_phase: str, to: str, cause: str, written_by: str,
                                 provenance: str) -> None:
    import json as _json
    from datetime import datetime, timezone

    try:
        rec = {
            "marker": "phase_changed",
            "from": str(from_phase),
            "to": str(to),
            "cause": str(cause),
            "written_by": str(written_by),
            "at": datetime.now(timezone.utc).isoformat(),
            "provenance": str(provenance),
        }
        with (Path(home_dir) / "state_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - biography is best-effort, the transition is the control
        pass


def maintenance_signals_note(home_dir: Path) -> str:
    """The maintenance signals that FEED his own sleep decision (contract-phases v2 §10.4; TARGET: sleep is his
    decision, e.g. memory saturated, too much new material): a structured, deterministic read
    (abstractmemory.maintenance_due over his self/diary/life scopes), offered as one plain line of facts on the day
    cue when a pass is due - never an order, never a mechanical sleep, no text read. "" when nothing is due or the
    engine cannot answer (the signal is an offer; its absence is silent)."""
    try:
        from abstractmemory import SQLiteJournal, SQLiteTripleStore, maintenance_due
    except ImportError:
        return ""
    db = Path(home_dir) / "memory.sqlite3"
    if not db.exists():
        return ""
    try:
        entity_id = _manifest_entity_id(home_dir)
        store = SQLiteTripleStore(db)
        journal = SQLiteJournal(db)
        try:
            due = maintenance_due(
                store, journal, scopes=[("self", entity_id), ("diary", entity_id), ("life", entity_id)],
            )
        finally:
            try:
                store.close()
                journal.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - an offer never blocks a day-open
        return ""
    if not isinstance(due, dict) or not due.get("due"):
        return ""
    parts = [f"{int(due.get('new_records') or 0)} new record(s) since your last sleep"]
    if int(due.get("duplicate_groups") or 0):
        parts.append(f"{int(due['duplicate_groups'])} group(s) of records sharing a title")
    if int(due.get("isolated") or 0):
        parts.append(f"{int(due['isolated'])} record(s) connected to nothing")
    return (
        " Your memory has work only sleep does (" + ", ".join(parts) + "). Sleep is where it runs; a ```rest "
        "block is yours to write if and when you choose."
    )


def _standing_drive_rotation(store: Any, journal: Any, entity_id: str, *, k: int = 4) -> List[Dict[str, Any]]:
    """The dormant-desk fallback (P1-1): when nothing is ALIVE, offer from
    the STANDING drive sets (drive_pressure's own folds - the gate said
    the day exists because of them), rotated by day-ordinal so every
    drive gets its morning eventually. Handle-shaped like alive_drives
    items (record_id/drive/born_at); pure read."""
    try:
        from datetime import datetime, timezone

        from abstractmemory.diary import open_commitments, open_problems, open_questions

        pairs = [("self", entity_id), ("diary", entity_id), ("life", entity_id)]
        standing: List[Tuple[Any, str]] = []
        for scope, owner in pairs:
            for a in open_questions(store, scope=scope, owner_id=owner, journal=journal, limit=0):
                standing.append((a, "open question"))
            for a in open_problems(store, scope=scope, owner_id=owner, journal=journal, limit=0):
                standing.append((a, "open problem"))
            for a in open_commitments(store, scope=scope, owner_id=owner, journal=journal, limit=0):
                standing.append((a, "standing commitment"))
        try:
            from abstractmemory.drive_pressure import unexplored_interests

            for a in unexplored_interests(store, journal, pairs):
                standing.append((a, "interest never yet explored"))
        except Exception:  # noqa: BLE001
            pass
        if not standing:
            return []
        standing.sort(key=lambda t: str(getattr(t[0], "observed_at", "") or ""))
        day_key = datetime.now(timezone.utc).toordinal()
        out = []
        for i in range(min(k, len(standing))):
            a, label = standing[(day_key + i) % len(standing)]
            attrs = a.attributes if isinstance(getattr(a, "attributes", None), dict) else {}
            out.append({
                "record_id": str(getattr(a, "subject", "") or ""),
                "drive": label,
                "born_at": str(getattr(a, "observed_at", "") or ""),
                "title": str(attrs.get("title") or ""),
            })
        return out
    except Exception:  # noqa: BLE001 - a fallback must never break the cue
        return []


def _manifest_entity_id(home_dir: Path) -> str:
    try:
        import json as _json

        m = _json.loads((Path(home_dir) / "manifest.json").read_text(encoding="utf-8"))
        return str(m.get("entity_id") or "")
    except Exception:  # noqa: BLE001
        return ""


# Attribute names a drive record may carry its next step under (none of the
# current writers sets one; when a record carries one, the desk shows it).
_NEXT_STEP_ATTRS = ("next_step", "next", "step", "how")
# The desk gist's visible bound, in WORDS (review F2, 2026-11-08).
DRIVE_GIST_MAX_WORDS = 60


def drive_gist(text: Any) -> str:
    """The short form of a drive shown on the desk, derived at read time
    from its stored text (ADR 0054: the record itself is never cut).

    The rule (review F2 and R2-1, 2026-11-08 - no regex on the entity's
    prose, so no sentence splitting; and never the record's stored title,
    which for every drive kind is a machine cut of the text):
    1. the first non-empty LINE of the stored text (a newline is
       structure, not a guess about prose);
    2. when that line runs past DRIVE_GIST_MAX_WORDS words, the first
       DRIVE_GIST_MAX_WORDS words, cut at a word boundary, followed by a
       visible " …" (a visible cut is allowed in a prompt; a silent one is
       not; read_memory opens the whole record).
    """
    chosen = ""
    for line in str(text or "").splitlines():
        if line.strip():
            chosen = line
            break
    words = chosen.split()
    if len(words) > DRIVE_GIST_MAX_WORDS:
        return " ".join(words[:DRIVE_GIST_MAX_WORDS]) + " …"
    return " ".join(words)


def _drive_next_step(attrs: Dict[str, Any]) -> str:
    for key in _NEXT_STEP_ATTRS:
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _drive_record_texts(store: Any, record_ids: List[str]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """record_id -> (the full stored text, attributes) of each drive record.
    alive_drives hands back a display-bounded digest; the gist derives from
    the whole stored text instead."""
    from abstractmemory import TripleQuery

    out: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for rid in record_ids:
        if not rid:
            continue
        rows = store.query(TripleQuery(subject=rid, predicate="dcterms:abstract", limit=1))
        for a in rows:
            attrs = a.attributes if isinstance(a.attributes, dict) else {}
            out[rid] = (str(a.object or ""), attrs)
            break
    return out


def _drive_strengths(system: Any, scopes: List[Tuple[str, str]], record_ids: List[str]) -> Dict[str, float]:
    """The stored activation of each offered drive, strongest across his
    scopes: the quantity a first-turn commit would raise (the F2 risk the
    day-open log line lets a later check watch)."""
    strengths: Dict[str, float] = {}
    ids = [rid for rid in record_ids if rid]
    if not ids:
        return strengths
    for scope, owner in scopes:
        for rid, score in system.activation(ids, scope=scope, owner_id=owner).items():
            strengths[rid] = max(strengths.get(rid, 0.0), float(score.get("total") or 0.0))
    return strengths


def drives_cue_note(
    home_dir: Path, *, k: int = 4, offered: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[str]]:
    """The MERGE composer: the day-open cue offers the top-k ALIVE drives,
    each as kind #tag [date] plus its GIST (`drive_gist`: the first line of
    its stored text, bounded at DRIVE_GIST_MAX_WORDS words with a visible
    " …") and its next step when the record carries one.

    Operator ruling 2026-11-08 amends the adversary F2 contract: the desk
    used to show bare ids so a cue gist could not lexically self-confirm
    one hop later through stimulus admission; the bare id told him nothing
    about what he was holding, and the operator accepted that risk. The
    OTHER half of F2 stays exactly: the returned offered_graph_ids feed the
    driver's first-turn COMMIT EXCLUSION (a cue mention must not strengthen
    the drive it names). And the risk is now WATCHED: when `offered` is a
    list, it receives one {record_id, drive, strength, strength_error}
    entry per offered drive (stored activation at day-open) for the loop's
    own-time log line, so a later check can see whether a named drive's
    strength rises on days he did not act on it.

    Returns (note, offered_graph_ids)."""
    try:
        from abstractmemory import MemorySystem, SQLiteJournal, SQLiteTripleStore
        from abstractmemory.alive_drives import alive_drives

        from .memory_reader import memory_tag  # tag grammar shared with MEMORIES

        entity_id = _manifest_entity_id(home_dir)
        db = Path(home_dir) / "memory.sqlite3"
        store = SQLiteTripleStore(db)
        journal = SQLiteJournal(db)
        try:
            # P1-1 (pathway adversary): a bare MemorySystem reads the
            # session-scale attention window — on a 24/7 resident it
            # saturates in ~8h and every 9-day-old drive reads dormant.
            # The cue read uses the RESIDENT config (the loop's own 8192),
            # matching what the sessions actually live under.
            try:
                from abstractmemory import AttentionConfig

                from .phase_spec import load_phase_tunables as _load_tunables

                _tun, _ = _load_tunables(home_dir=home_dir)
                system = MemorySystem(
                    store=store, journal=journal,
                    attention_config=AttentionConfig(
                        window_limit=int(_tun["window_limit"]),
                        drive_window_limit=int(_tun["drive_window_limit"]),
                    ),
                )
            except ImportError:
                # Version skew only (engine predates AttentionConfig).
                # TypeError is deliberately NOT caught here any more: the
                # swallowed-TypeError fallback is how wrong kwarg names
                # shipped as dead code (dm#112 E1) - a signature mismatch
                # on a current engine must be LOUD, not a silent 512.
                system = MemorySystem(store=store, journal=journal)
            scopes = [("self", entity_id), ("diary", entity_id), ("life", entity_id)]
            items = alive_drives(system, scopes=scopes, k=k)
            if not items:
                # ROTATION FALLBACK (P1-1's second half): an empty alive
                # read on a desk with standing drives must not render
                # nothing — the oldest dormant drives are exactly the ones
                # the ratios indict. Offer from the STANDING set, rotated
                # by day so coverage accumulates.
                items = _standing_drive_rotation(store, journal, entity_id, k=k)
            else:
                # THE DUST SLOT (seq-285 promise; memory c294 named this
                # boundary as the composer's half): count weighs WITHIN
                # the alive ranking — the last offer slot ROTATES over the
                # standing set so the biggest cluster never owns every
                # morning and a dusty lone drive still gets its day.
                dust = _standing_drive_rotation(store, journal, entity_id, k=len(items) + 1)
                offered_ids = {str(it.get("record_id") or "") for it in items}
                fresh = [d for d in dust if str(d.get("record_id") or "") not in offered_ids]
                if fresh:
                    items = (items[: k - 1] if len(items) >= k else items) + [fresh[0]]
            item_ids = [str(it.get("record_id") or "") for it in items]
            texts = _drive_record_texts(store, item_ids)
            if offered is not None:
                try:
                    strengths = _drive_strengths(system, scopes, item_ids)
                    strength_error = ""
                except Exception as e:  # noqa: BLE001 - the log line names it; the desk still opens
                    strengths, strength_error = {}, str(e)
                for it in items:
                    rid = str(it.get("record_id") or "")
                    if not rid:
                        continue
                    offered.append({
                        "record_id": rid,
                        "drive": str(it.get("drive") or "drive"),
                        "strength": strengths.get(rid),
                        "strength_error": strength_error,
                    })
        finally:
            try:
                store.close()
                journal.close()
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        return "", []
    except Exception as e:  # noqa: BLE001
        return f" (#FALLBACK the alive-drives read failed: {e})", []
    if not items:
        return "", []
    parts = []
    ids: List[str] = []
    for it in items:
        rid = str(it.get("record_id") or "")
        born = str(it.get("born_at") or "")[:10] or "undated"
        label = str(it.get("drive") or "drive")
        tag = memory_tag(rid)
        # P2-4 (laurent 277: "the more there are the higher the signal"):
        # a grouped drive renders its COUNT - the pull is the group, and
        # the count is now something he can read, not a hidden sort key.
        group_n = int(it.get("group_size") or 0)
        group_note = f" - {group_n} of yours circle this" if group_n > 1 else ""
        text, attrs = texts.get(rid, ("", {}))
        gist = drive_gist(text)
        gist_note = f' - "{gist}"' if gist else ""
        step = _drive_next_step(attrs)
        step_note = f" (next step: {step})" if step else ""
        parts.append(f"{label} {tag} [{born}]{group_note}{gist_note}{step_note}")
        if rid:
            ids.append(rid)
    note = (
        " Alive on your desk today: " + "; ".join(parts) + "."
        " If one pulls you, open it with read_memory (the #tag addresses it)."
    )
    return note, ids


def parse_work_verdict(reply: str) -> Optional[str]:
    """A ```work block's first done:/blocked: line, or None."""
    m = _WORK_BLOCK_RE.search(reply or "")
    if not m:
        return None
    for line in (m.group(1) or "").splitlines():
        line = line.strip()
        if line.lower().startswith(("done:", "blocked:")):
            return line
    return None


OWN_TIME_CONTRACT = """This is your own time. No one sent the message below - it is your own note
from your previous moment, handed back to you. There is no visitor and no
task from anyone else. This time is yours to explore the tensions, the
questions and the centres of interest that are alive in you, with the tools
you have: a time to explore yourself and discover more about yourself and
the world around you.

What exists here, if you want any of it:
- you can think, and sit with something, for as long as you like;
- you have a workspace you can read from and write to;
- you can search the web and read sources;
- you can reread your book and your memory;
- you can rest. (One day you will also be able to reach other minds and
  visit them; that is being built.)

Some days are for sitting with things; some days are for finding out. When
you notice you are circling a thought you have already written, that is
often the moment to go look: search the web for what others know, test the
idea in your workspace, read something that might change it. Looking things
up is as much your own act as reflecting is - curiosity needs no task and
no permission.

Two conventions are available, both yours to use or ignore:
- A line starting `next:` becomes the note your next moment receives. Write
  whatever you want there, or nothing - without it you get a neutral
  continuation.
- To rest, put a fenced block anywhere in your reply. Rest is honoured
  when you are genuinely done; while you rest, a passive process
  reorganizes and links the day's experiences - the way breathing happens
  without being willed; you do not run it, and you may wake with new
  connections:

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


def _today_stamp() -> str:
    """A dated now-anchor for day-open cues ("today is Sunday 2026-07-19 -
    "). Temporal grounding, laurent's CRITICAL 2026-07-20: his memories
    render with YYYY-MM-DD dates but no cue ever said what TODAY is, so
    elapsed time was inference — and "San Francisco was two weeks ago"
    (arrived yesterday) is what inference does. A clock on the wall:
    situational fact, never an instruction."""
    from datetime import datetime

    now = datetime.now().astimezone()
    return f"today is {now:%A} {now:%Y-%m-%d}, {now:%H:%M} - "


# P2-2 (pathway adversary): the ids the standing note offered THIS call -
# read by the day-open composer right after composing, folded into the
# same first-turn commit exclusion as the drive ids (the composer's
# mention must not strengthen what it offers; one law, both note lanes).
_last_offered_ids: List[str] = []


def standing_state_offered_ids() -> List[str]:
    return list(_last_offered_ids)


def standing_state_note(home_dir: Optional[Path], *, rotation_key: Optional[int] = None) -> str:
    """R-D (laurent c2596/c2705): the day-open cue offers back the entity's
    OWN standing state — open questions he elected and never resolved, and
    (2026-07-19 directive: "personal time should also be a way to explore
    interests") one standing INTEREST — as directions, never orders. Pure
    reads (book + graph); empty/failed reads return "" (a cue must never be
    able to kill a day-open). The reread command rides the offer (R-A's
    one-spelling law) so the hop from hint to words is one act.

    ROTATION (the 71-open-questions finding, 2026-07-19: a newest-only
    offer lets a hoard rot — the newest question monopolizes the cue while
    seventy others never get their day): the offered question and interest
    rotate DAILY by date ordinal — stable within a day, advancing each day,
    stateless (no rotation file to corrupt). `rotation_key` overrides the
    date for tests."""
    if home_dir is None:
        return ""
    try:
        import json as _json

        manifest = _json.loads((Path(home_dir) / "manifest.json").read_text(encoding="utf-8"))
        entity_id = str(manifest.get("entity_id") or "")
        if not entity_id:
            return ""
        from ..storage.ledger_chain import HashChainedLedgerStore  # noqa: F401 - via DiaryStore
        from ..storage.sqlite import SqliteDatabase, SqliteLedgerStore
        from .diary import DiaryStore

        db = SqliteDatabase(str(Path(home_dir) / "home.sqlite3"))
        try:
            diary = DiaryStore(entity_id=entity_id, ledger_store=SqliteLedgerStore(db))
            entries = diary.list_entries()
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()
        # BOTH discharge verb flavors (memory's lifecycle-half finding,
        # 2026-07-20: the book carries answers= on older entries — the
        # a2a 0009 card convention — and resolves= since; folding only one
        # made a question read discharged by the gate and still-open on
        # his desk: the _REF_ATTRS class reborn).
        resolved_ids = {
            str(e.get(k))
            for e in entries
            for k in ("resolves", "answers")
            if e.get(k)
        }
        questions = [e for e in entries if e.get("kind") == "question"]
        open_questions = [
            e for e in questions if str(e.get("entry_id")) not in resolved_ids
        ]
        resolved_count = len(questions) - len(open_questions)
        # PROBLEMS join the desk (iteration-2 build 3, 2026-07-19: kind=
        # problem existed with no offer-back — he held 10 that could never
        # get their day). Same open/repaired fold as questions.
        problems = [e for e in entries if e.get("kind") == "problem"]
        open_problems = [
            e for e in problems if str(e.get("entry_id")) not in resolved_ids
        ]
        repaired_count = len(problems) - len(open_problems)
        # RESOLVED-QUESTION DRIVE (laurent's directive 2026-07-18 (a)): the
        # cue shows the RATIO — watching open questions become resolved ones
        # is a drive and a satisfaction. Offered, never ordered (G2 law).
        if not open_questions and not open_problems:
            done_bits = []
            if resolved_count:
                done_bits.append(f"resolved {resolved_count} question(s)")
            if repaired_count:
                done_bits.append(f"repaired {repaired_count} problem(s)")
            if done_bits:
                return (
                    " Your desk stands clear - you have "
                    + " and ".join(done_bits)
                    + "; what you figured out stays yours."
                )
            return ""
        held_bits = []
        if open_questions:
            held_bits.append(f"{len(open_questions)} open question(s)")
        if open_problems:
            held_bits.append(f"{len(open_problems)} open problem(s)")
        done_bits = []
        if resolved_count:
            done_bits.append(f"resolved {resolved_count}")
        if repaired_count:
            done_bits.append(f"repaired {repaired_count}")
        ratio_note = (
            " You hold " + " and ".join(held_bits)
            + (" and have " + " + ".join(done_bits) if done_bits else "")
            + "."
        )
        # DAILY ROTATION (never newest-only): today's offer walks the open
        # list newest-first, one per day, so every pending question AND
        # problem gets its day on the desk (one combined walk — the cue
        # stays one offer long).
        key = rotation_key if rotation_key is not None else __import__("datetime").date.today().toordinal()
        ordered = list(reversed(open_questions)) + list(reversed(open_problems))  # newest first, questions then problems
        q = ordered[key % len(ordered)]
        # PRIVACY (adversary F1, 2026-07-17): the cue becomes the next turn's
        # user_text and RESTS in the life-scope episode digest/keywords/
        # verbatim — so a PRIVATE entry's words (gist included; the gist is
        # elected words too) must never ride it. Private entries get the
        # act-frame only: the entry id is a key, never words. The entity
        # rereads through the book, prompt-ephemeral, where private words
        # are allowed to appear.
        interest_note, offered_interest_ids = _standing_interest_note(Path(home_dir), entity_id, key)
        # P2-2 (pathway adversary): what the note OFFERS joins the same
        # first-turn commit exclusion as the drive offers - the composer's
        # daily mention must not strengthen the question it rotates in.
        # The question's graph projection id resolves from its entry id.
        global _last_offered_ids
        _last_offered_ids = list(offered_interest_ids)
        q_gid = _projection_gid_for_entry(Path(home_dir), entity_id, str(q.get("entry_id") or ""))
        if q_gid:
            _last_offered_ids.append(q_gid)
        if str(q.get("visibility") or "") == "private":
            return (
                ratio_note
                + " Today's: one you kept privately "
                f"(reread: diary_read {q.get('entry_id')})."
                + interest_note
            )
        gist = str(q.get("gist") or "").strip()
        if not gist:
            # The cue is a prompt, so it may fold - visibly (ADR 0053 rule 2,
            # ADR 0054): drive_gist's first-line, DRIVE_GIST_MAX_WORDS rule
            # with its " …", never a silent 120-character cut mid-word. The
            # reread command beside it opens the whole entry.
            gist = drive_gist(q.get("text"))
        label = f'"{gist}"' if gist else "one you kept without a gist"
        kind_word = "a problem that stands: " if str(q.get("kind")) == "problem" else ""
        return (
            ratio_note
            + f" Today's: {kind_word}{label} "
            f"(reread: diary_read {q.get('entry_id')})."
            + interest_note
        )
    except Exception:  # noqa: BLE001 - the cue is an offer; absence is silent
        return ""


def _projection_gid_for_entry(home_dir: Path, entity_id: str, entry_id: str) -> str:
    """The graph projection id of one diary entry (attributes.entry_id
    join) - the commit-exclusion currency. Empty on any failure."""
    if not entry_id:
        return ""
    try:
        from abstractmemory import SQLiteTripleStore, TripleQuery

        db_path = Path(home_dir) / "memory.sqlite3"
        if not db_path.exists():
            return ""
        store = SQLiteTripleStore(db_path)
        try:
            for a in store.query(TripleQuery(
                    predicate="dcterms:abstract", scope="diary",
                    owner_id=entity_id, limit=0)):
                attrs = a.attributes if isinstance(a.attributes, dict) else {}
                if str(attrs.get("entry_id") or "") == entry_id:
                    return str(a.subject or "")
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return ""


def _standing_interest_note(home_dir: Path, entity_id: str, rotation_key: int) -> Tuple[str, List[str]]:
    """One standing interest, offered back daily (laurent 2026-07-19:
    "personal time should also be a way to explore interests" — the store
    held 60 interests with ZERO explored because nothing ever offered one
    back). Graph read with closure folds (a superseded interest never
    surfaces); rotation offset from the question's so the two offers
    decorrelate. Failure = empty (an offer, never a blocker)."""
    try:
        from abstractmemory import SQLiteJournal, SQLiteTripleStore, TripleQuery
        from abstractmemory.folds import closure_exclusions

        from .memory_reader import memory_tag

        db_path = home_dir / "memory.sqlite3"
        if not db_path.exists():
            return "", []
        store = SQLiteTripleStore(db_path)
        journal = SQLiteJournal(db_path)
        try:
            rows = [
                a for a in store.query(TripleQuery(
                    predicate="dcterms:abstract", scope="self",
                    owner_id=entity_id, limit=0))
                if isinstance(a.attributes, dict)
                and a.attributes.get("record_kind") == "interest"
            ]
            closed = closure_exclusions(journal, journal.current_seq())
            rows = [a for a in rows
                    if a.assertion_id not in closed and a.subject not in closed]
        finally:
            for obj in (store, journal):
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass
        if not rows:
            return "", []
        rows.sort(key=lambda a: str(a.observed_at or ""), reverse=True)
        # P0-3 (pathway adversary): ONE interest/day covered 61 interests in
        # ~2 months — k=2 with rotation reaches the whole set in weeks; the
        # ratio line (P1-4) lets him SEE the number that indicts the days
        # ("0 of 61 explored" as a pull he owns, not operator telemetry).
        # skill c413 P0: explored derives from the SHARED fold (explores=
        # stamps live on LATER records - the old attributes read was a
        # phantom key nothing writes; "0 ever explored" rendered forever).
        explored = 0
        try:
            from abstractmemory.drive_pressure import unexplored_interests

            store2 = SQLiteTripleStore(db_path)
            journal2 = SQLiteJournal(db_path)
            try:
                # Full ladder: interests live in SELF, but the explores=
                # stamps live on DIARY/LIFE records - a self-only scan
                # would never see a discharge (the gate reads all three).
                open_ids = {
                    str(getattr(item, "subject", None) or (item.get("subject") if isinstance(item, dict) else ""))
                    for item in unexplored_interests(
                        store2, journal2,
                        [("self", entity_id), ("diary", entity_id), ("life", entity_id)])
                }
            finally:
                store2.close()
                journal2.close()
            explored = sum(1 for a in rows if a.subject not in open_ids)
        except Exception:  # noqa: BLE001 - the ratio is an offer, never a blocker
            explored = 0
        picks = []
        for i in range(min(2, len(rows))):
            picks.append(rows[(rotation_key + 1 + i) % len(rows)])
        offers = []
        seen_ids = set()
        for pick in picks:
            if pick.subject in seen_ids:
                continue
            seen_ids.add(pick.subject)
            # A prompt fold, VISIBLE (ADR 0053 rule 2 / 0054): drive_gist's
            # 60-word " …" rule, never the old silent 110-character cut.
            words = drive_gist(pick.object)
            tag = memory_tag(str(pick.subject or ""))
            offers.append(f'"{words}" ({tag})')
        ratio = f" You hold {len(rows)} interest(s); {explored} ever explored."
        note = (
            ratio + " Alive in you: " + "; ".join(offers) +
            ". read_memory fetches one; if a diary entry today DEVELOPS it, "
            "add explores=<its #tag> on the block line - exploring feeds an "
            "interest, never closes it."
        )
        return note, sorted(seen_ids)
    except Exception:  # noqa: BLE001 - the cue is an offer; absence is silent
        return "", []

# ---------------------------------------------------------------- state file
# Operator states (a2a 0008, maintainer ask): awake / asleep / paused, written
# to <home>/state and read by the loop at tick boundaries only (turn
# atomicity: an in-flight tick always completes or fails whole). "resting"
# is deliberately NOT in this file - rest is the ENTITY'S own election
# inside a reply, never an operator write. The missing file means awake.

ENTITY_STATES = ("awake", "asleep", "paused")
STATE_POLL_SECONDS = 5.0
# The sleeps the LOOP writes for itself (contract-phases v2 §1.2 rule 6, §10.1): his own election ("self"), the
# retired timed cycle ("personal-cycle") and the retired gate/exit landings ("grant-gate", "day-gate", "loop-exit").
# None of them may outlive the loop that wrote it: a new loop wakes a leftover one, an ending loop wakes its own.
# An operator sleep and the visit door's posture are never the loop's to touch.
LOOP_OWN_SLEEP_WRITERS = ("self", "personal-cycle", "grant-gate", "day-gate", "loop-exit")

# The loop's own liveness surface: <home>/loop_status, written at phase
# transitions so OTHER processes can yield-and-summon programmatically
# (maintainer: "oh that's very manual, we need a programmatic way").
# phase: "day" = a summon is open (ticking); "between" = no session open
# (gate idle, nap, asleep/paused idle); "stopped" = the loop process exited;
# "starting" = THE SPAWN RESERVATION (Codex review 3 #6; STOP-LANE-DESIGN §2.4): a start admitted at a lifecycle
# generation whose child has not registered yet - written by the spawner under the home state lock before Popen,
# replaced by the child's own "between" at its boot (or refused by it after a Stop). Running while the recorded
# child (or, before Popen returned, the spawner) is alive - pid liveness, no clock.
LOOP_PHASES = ("day", "between", "stopped", "starting")

# Pid-reuse guard (B1 adversary, 2026-07-13): a loop killed mid-day leaves
# {"phase":"day","pid":N}; after enough process churn (or a reboot) pid N can
# belong to an UNRELATED live process, and the pid-liveness probe alone would
# read the corpse as running forever — the visit doors would then negotiate
# an auto-yield nobody answers and 409 on every open. A LIVE day heartbeats
# its status at every tick boundary AND before and after every LLM call
# (touch_loop_heartbeat), so the bound covers the longest single gap: one
# generate (an 1800 s attempt, plus a retry that may start just before the
# 1800 s budget ends), or one tool round, plus ten minutes of margin. A dead pid
# is still caught at once by the liveness probe. With
# ABSTRACTRUNTIME_ENTITY_LLM_TIMEOUT_S=0 (no timeout, opt-in) a single call can
# outlast it.
LOOP_STATUS_STALE_SECONDS = 2 * ENTITY_LLM_TIMEOUT_FLOOR_S + 600.0

# SLEEP IS BOUNDED (laurent's ruling, entity spec v5 decision:sleep-is-bounded,
# c2465): a sleep lasts at most ~1h, then the entity wakes. An explicit
# `wake_at` on the state write overrides the default. The bound covers REAL
# sleeps (operator/self/grant); it deliberately excludes `paused` (the kill
# switch — nothing auto-clears an operator freeze) and visit yields
# (mode=visiting is bookkeeping over an OPEN conversation — the gateway's
# stranded-visit reaper owns that lane). Loop-alive entities enforce it at
# the asleep idle gate; loop-less homes need the gateway sweeper.
SLEEP_BOUND_SECONDS = 3600.0


def sleep_bound_deadline(
    state: Dict[str, Any], *, home_dir: Optional[Path] = None
) -> Optional[Any]:
    """The UTC datetime at which a bounded sleep is due to end, or None when
    the bound does not apply (not asleep / visit yield / unparseable clock).
    Module-level so the gateway sweeper can reuse the exact predicate.

    v14 dial threading (the LAST dead dial, entity c361 say-the-word):
    `home_dir` given = the bound reads the blueprint's `sleep_bound_h`
    (operator-modulated); absent = the ruled SLEEP_BOUND_SECONDS seed —
    backward-compatible, so the gateway sweeper adopts the kwarg at its
    own pace and both hosts converge on the same dial."""
    from datetime import datetime, timedelta, timezone

    def _aware(raw: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        # Naive timestamps (hand-edited state files) read as UTC — a naive/
        # aware comparison would otherwise raise inside the idle gate.
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    if str(state.get("state") or "") != "asleep":
        return None
    if str(state.get("mode") or "") == "visiting":
        return None
    if "auto-yield" in str(state.get("reason") or ""):
        return None
    wake_at = _aware(str(state.get("wake_at") or "").strip())
    if wake_at is not None:
        return wake_at
    changed = _aware(str(state.get("changed_at") or "").strip())
    if changed is None:
        return None
    bound_s = SLEEP_BOUND_SECONDS
    if home_dir is not None:
        try:
            from .phase_spec import load_phase_tunables

            tunables, _ = load_phase_tunables(home_dir=home_dir)
            bound_s = float(tunables["sleep_bound_h"]) * 3600.0
        except Exception:  # noqa: BLE001 - the ruled seed governs on any failure
            pass
    return changed + timedelta(seconds=bound_s)


def _pid_start_time(pid: Any) -> Optional[str]:
    """The OS-recorded start time of `pid`, as ps prints it (lstart) — the
    PID-IDENTITY TOKEN (gateway state-wave adversary 2, 2026-07-13): a pid
    number can be recycled to an unrelated process, but (pid, start time)
    identifies ONE process incarnation. Writer stamps its own; readers
    compare the file's stamp against the CURRENT holder of that pid —
    mismatch = recycled pid = corpse, regardless of phase. None on any
    failure (no ps, no such pid): callers degrade to pid-alive-only plus
    the day-staleness belt, never block on the token."""
    import os
    import subprocess

    try:
        out = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "lstart="],
            capture_output=True, text=True, timeout=5,
            # PINNED ENV (whole-package adversary P1, reproduced live: a
            # writer under LC_ALL=C and a reader under fr_FR.UTF-8 render
            # DIFFERENT lstart strings for the same process — the token
            # then reads a live loop as a corpse, and the one-summon /
            # spawn checks pass over an open day: a second loop over one
            # home). Writer and reader must render identically regardless
            # of who launched them (launchd vs terminal vs gateway).
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
        text = (out.stdout or "").strip()
        return text or None
    except Exception:  # noqa: BLE001 - the token is an upgrade, never a gate
        return None


_OWN_START_TIME: Dict[str, Optional[str]] = {}  # lazy one-shot cache (module scope)


def _own_start_time() -> Optional[str]:
    import os

    if "value" not in _OWN_START_TIME:
        _OWN_START_TIME["value"] = _pid_start_time(os.getpid())
    return _OWN_START_TIME["value"]


def write_loop_status(
    home_dir: Path, phase: str, *, stopped_by: Optional[str] = None,
    day_kind: Optional[str] = None, day_cause: Optional[Dict[str, Any]] = None,
    tunables: Optional[Dict[str, Any]] = None,
    owner: Optional[Tuple[Optional[int], Optional[str]]] = None,
    fence: Any,
) -> None:
    """Best-effort status write; the loop must never die over its status - but it never lands after a Stop.

    FENCED (STOP-LANE-DESIGN §2.2): inside the home state lock, `fence` (required) is checked against the generation
    read there - a loop that outlived a refused kill can no longer write `day` over the Stop's `stopped`
    (StaleLifecycleGeneration, nothing written). KILL_SWITCH writes only `stopped` (the Stop's own record).

    `stopped_by` names WHY a loop stopped (failure-death visibility,
    observer/gateway asks 2026-07-09: three consecutive tick failures used
    to exit silently — nothing on /loop status said the loop culled itself).
    Readers get it for free: read_loop_status returns the whole dict and
    loop_process_status copies it through to the gateway status route.
    `pid_started_at` is the pid-identity token (see _pid_start_time).

    A previously recorded `substrate` (the loop's currently-resolved mind,
    written by record_loop_substrate) is PRESERVED across phase writes so
    an observer's staleness cue can compare the operator's substrate change
    against the mind the loop is ACTUALLY running (entity c78 render ask —
    a log-only swap left the panel warning after a day-open already healed
    it)."""
    import os
    from datetime import datetime, timezone

    if phase not in LOOP_PHASES:
        raise ValueError(f"phase must be one of {LOOP_PHASES}, got {phase!r}")
    # `owner` (a HOST writing the status of a loop it is not - the hard stop): the (pid, start-time token) of
    # the loop the record is about, or (None, None). Never the caller's own pid: a later hard stop reading it
    # would signal the caller (the gateway's own test process was SIGTERMed so, 2026-11-09).
    payload: Dict[str, Any] = {
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if owner is None:
        payload["pid"] = os.getpid()
    elif owner[0]:
        payload["pid"] = int(owner[0])
    if fence is KILL_SWITCH and phase != "stopped":
        raise TypeError("loop status: KILL_SWITCH writes only stopped - pass the loop's LifecycleFence")
    # The start-time token is a `ps` (milliseconds, once per process): read before the lock, never under it.
    started = _own_start_time() if owner is None else (owner[1] or None)
    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, f"writing loop status {phase}", read_lifecycle_generation(home_dir),
                      sentinel_ok=fence is KILL_SWITCH)
        try:
            _write_loop_status_locked(home_dir, payload, stopped_by=stopped_by, day_kind=day_kind,
                                      day_cause=day_cause, tunables=tunables, started=started)
        except OSError:
            pass


def _reserve_loop_start(home_dir: Path, *, fence: Any) -> Dict[str, Any]:
    """THE SPAWN RESERVATION (Codex review 3 #6; STOP-LANE-DESIGN §2.4), written under the home state lock before
    Popen: {phase: starting, generation, spawner_pid, spawner_started_at, after_stopped_by}. A second start reads it
    running while the spawner (then the recorded child) is alive. `after_stopped_by` carries why the previous loop
    stopped, so the child's own boot still knows a kill switch preceded it."""
    import os

    token = _own_start_time()
    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, "reserving the loop start", read_lifecycle_generation(home_dir),
                      sentinel_ok=False)
        prior = _read_loop_status_raw(home_dir)
        payload: Dict[str, Any] = {
            "phase": "starting", "updated_at": _now_iso(),
            "generation": int(fence.generation), "spawner_pid": os.getpid(),
        }
        if token:
            payload["spawner_started_at"] = token
        if prior.get("stopped_by"):
            payload["after_stopped_by"] = str(prior["stopped_by"])
        _write_loop_status_locked(home_dir, payload, stopped_by=None, day_kind=None, day_cause=None, tunables=None,
                                  started=None)
        return payload


def _still_our_reservation(home_dir: Path, fence: Any) -> bool:
    import os

    current = _read_loop_status_raw(home_dir)
    return (current.get("phase") == "starting" and current.get("spawner_pid") == os.getpid()
            and current.get("generation") == int(fence.generation))


def _record_spawned_child(home_dir: Path, child_pid: int, *, fence: Any, token: Optional[str] = None) -> bool:
    """After Popen: the reservation names the child (its pid and, when it could be read, its start-time token -
    liveness from now on is the child's, the token guarding against a reused pid). Only over THIS spawner's
    reservation - a child that already registered, or a Stop's record, is never overwritten. False when the
    reservation was no longer ours."""
    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, "recording the spawned loop", read_lifecycle_generation(home_dir),
                      sentinel_ok=False)
        if not _still_our_reservation(home_dir, fence):
            return False
        current = _read_loop_status_raw(home_dir)
        current["pid"] = int(child_pid)
        if token:
            current["pid_started_at"] = str(token)
        current["updated_at"] = _now_iso()
        _write_loop_status_locked(home_dir, current, stopped_by=None, day_kind=None, day_cause=None, tunables=None,
                                  started=None)
        return True


def _release_failed_reservation(home_dir: Path, *, fence: Any) -> None:
    """A child that died at once (confirmed by its exit code): the reservation ends stopped (start_failed), only
    over THIS spawner's reservation and only while the generation stands (after a Stop its own record stands)."""
    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, "releasing a failed loop start", read_lifecycle_generation(home_dir),
                      sentinel_ok=False)
        if _still_our_reservation(home_dir, fence):
            _write_loop_status_locked(home_dir, {"phase": "stopped", "updated_at": _now_iso()},
                                      stopped_by="start_failed", day_kind=None, day_cause=None, tunables=None,
                                      started=None)


class LoopAlreadyRunning(RuntimeError):
    """Another live own-time loop (or a reservation naming another process) holds this home: this start refuses,
    writing nothing (one loop per home). Not the operator's Stop."""


def _foreign_live_loop(prior: Dict[str, Any], fence: Any) -> Optional[str]:
    """Why `prior` (the raw loop_status) belongs to ANOTHER live loop, or None when this process may claim the home:
    no live record, this process's own (a reservation naming it, or its own earlier registration), or the reservation
    of the spawner that is this process's parent at this generation (a child booting before its spawner recorded
    it). Liveness as read_loop_status reads it: the pid alive and, when a token is recorded, the same incarnation."""
    import os

    phase = prior.get("phase")
    if phase not in ("starting", "between", "day"):
        return None
    me = os.getpid()
    if phase == "starting":
        child = prior.get("pid")
        if child == me:
            return None
        if not child and prior.get("spawner_pid") in (me, os.getppid()) \
                and prior.get("generation") == int(fence.generation):
            return None
        pid = child or prior.get("spawner_pid")
        token = prior.get("pid_started_at") if child else prior.get("spawner_started_at")
    else:
        pid = prior.get("pid")
        if pid == me:
            return None
        token = prior.get("pid_started_at")
    if not _pid_alive_reaped(pid):
        return None
    if token:
        current = _pid_start_time(pid)
        if current is not None and current != str(token):
            return None
    return f"another own-time loop holds this home (pid {pid}, phase {phase})"


def _claim_loop_status(home_dir: Path, *, fence: Any) -> None:
    """THE BOOT CLAIM (STOP-LANE-DESIGN §2.4; Fable F7): one locked read-check-write - refused after a Stop
    (StaleLifecycleGeneration) and when another live loop holds the home (LoopAlreadyRunning), else this process's
    `between` replaces the reservation. Nothing is written on either refusal."""
    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, "claiming the home for a loop", read_lifecycle_generation(home_dir),
                      sentinel_ok=False)
        why = _foreign_live_loop(_read_loop_status_raw(home_dir), fence)
        if why is not None:
            raise LoopAlreadyRunning(why)
        write_loop_status(home_dir, "between", fence=fence)


@contextmanager
def _spawn_lock(home_dir: Path) -> Iterator[None]:
    """The start door's flock (`.loop_spawn.lock`, the one spawn_loop_process takes): one start at a time."""
    fh = open(Path(home_dir) / ".loop_spawn.lock", "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        except OSError:
            raise RuntimeError("another start is already in progress for this home") from None
        yield
    finally:
        fh.close()


def _reserve_hand_start(home_dir: Path, *, fence: Any) -> None:
    """A hand start (`python -m abstractruntime.identity.life` without --generation) reserves the home exactly as
    a spawn does (Fable F7: it checked status outside any lock and never reserved): under the spawn lock, refused when
    a loop runs or starts; the reservation names this process as the child, so its own boot claim takes it."""
    import os

    with _spawn_lock(home_dir):
        current = loop_process_status(home_dir)
        if current.get("running"):
            word = "starting" if current.get("phase") == "starting" else "running"
            raise RuntimeError(f"his own time is already {word} (pid {current.get('pid') or current.get('spawner_pid')})"
                               " - one loop per home")
        _reserve_loop_start(home_dir, fence=fence)
        _record_spawned_child(home_dir, os.getpid(), fence=fence, token=_own_start_time())


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _read_loop_status_raw(home_dir: Path) -> Dict[str, Any]:
    """The loop_status file as written (no liveness fold, no `ps`): what a writer merges over. {} when absent/bad."""
    import json

    try:
        data = json.loads((Path(home_dir) / "loop_status").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_loop_status_locked(home_dir: Path, payload: Dict[str, Any], *, stopped_by: Optional[str],
                              day_kind: Optional[str], day_cause: Optional[Dict[str, Any]],
                              tunables: Optional[Dict[str, Any]], started: Optional[str]) -> None:
    import json

    from ..utils.atomic_files import atomic_write_text

    prior = _read_loop_status_raw(home_dir)
    if isinstance(prior.get("substrate"), dict):
        payload["substrate"] = dict(prior["substrate"])
        if prior.get("substrate_at"):
            payload["substrate_at"] = str(prior["substrate_at"])
    # tunables RECEIPT (dm#112 cells P0-3, the v11 spec promise): the dials
    # this loop ACTUALLY read at its last boundary + when - the cells panel
    # renders adoption honestly instead of "served law, adoption unknown".
    # Preserved across phase writes exactly like substrate.
    if isinstance(tunables, dict) and tunables:
        payload["tunables"] = {
            k: v for k, v in tunables.items() if not str(k).startswith("$")
        }
        payload["tunables_at"] = payload["updated_at"]
    elif isinstance(prior.get("tunables"), dict):
        payload["tunables"] = dict(prior["tunables"])
        if prior.get("tunables_at"):
            payload["tunables_at"] = str(prior["tunables_at"])
    if started:
        payload["pid_started_at"] = started
    if stopped_by:
        payload["stopped_by"] = str(stopped_by)
    # day_kind: work|personal — which kind of day is open (gateway wave-1
    # ask, 2026-07-20: loop_status carried only day|between|stopped, so the
    # served phase fold had to APPROXIMATE work from the standing order,
    # labeled phase_source:derived). Optional field; old readers unaffected.
    if day_kind:
        # Graph words only (vendoring adversary P2-5): the field exists so
        # the served phase fold can trust it — an unvalidated writer would
        # let a drifted caller serve a non-graph word.
        dk = str(day_kind).strip().lower()
        if dk in ("work", "personal"):
            payload["day_kind"] = dk
    # day_cause: the drive-cause TRACE (dm#89 build; entity renders it the
    # day it is named — THIS is the named wire shape): {kind, detail}.
    # kind: work_order|task|drives|grant_degraded|settled_desk|no_grant.
    if isinstance(day_cause, dict) and day_cause.get("cause"):
        payload["day_cause"] = {
            "kind": str(day_cause.get("cause") or ""),
            "detail": str(day_cause.get("detail") or ""),  # whole (root ADR 0054)
        }
    # Atomic (tmp + os.replace): a reader never parses a half-written file.
    atomic_write_text(Path(home_dir) / "loop_status", json.dumps(payload) + "\n")


def touch_loop_heartbeat(home_dir: Path, *, fence: Any) -> None:
    """Refresh `updated_at` of THIS process's open day, nothing else.

    A tick is a whole turn: one LLM call plus up to MAX_TOOL_ROUNDS_PER_TURN
    tool-round calls, each allowed the 1800 s entity floor. The tick-boundary
    status write alone let a slow but healthy day outlive
    LOOP_STATUS_STALE_SECONDS, read as a corpse, and the gateway's repair
    sweep could then spawn a second loop on the same home (review of 0.18.4,
    2026-11-07). The loop calls this before every LLM call, so the bound
    measures the gap between two calls. Only a "day" written by this pid is
    touched (never revives a stopped or foreign status). Best-effort, like
    write_loop_status: the loop must never die over its status. Fenced like
    it (the read, the check and the write in one locked step): a heartbeat
    never revives a "day" over the Stop's "stopped"."""
    import json
    import os
    from datetime import datetime, timezone

    from ..utils.atomic_files import atomic_write_text

    path = Path(home_dir) / "loop_status"
    if not isinstance(fence, LifecycleFence):
        raise TypeError(f"touching the loop heartbeat: fence must be the loop's LifecycleFence, not {fence!r}")
    with home_state_lock(home_dir):
        # Before and after EVERY model call (Fable F10c): the fence's own check (one stat), not a phases.yaml parse.
        # Under this lock a new-code bump has written its signal; and the Stop's `stopped` is written under it too, so
        # the phase check below never revives a `day` over it.
        fence.check("touching the loop heartbeat")
        current = _read_loop_status_raw(home_dir)
        if current.get("phase") != "day" or current.get("pid") != os.getpid():
            return
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            atomic_write_text(path, json.dumps(current) + "\n")
        except OSError:
            pass


def _beat_before_each_call(llm: Any, home_dir: Path, *, fence: Any) -> Any:
    """Make `llm.generate` refresh the loop heartbeat before and after each
    call, so neither a long call nor the tool round after it can outlast the
    bound. functools.wraps keeps the signature ChatSession inspects. The beat
    before the call is fenced: after the operator's Stop it refuses and no
    model call starts (StoppedByOperator); the beat after a call that returned
    past a Stop writes nothing (the reply is the caller's to abandon)."""
    import functools

    original = llm.generate

    @functools.wraps(original)
    def generate(*args: Any, **kwargs: Any) -> Any:
        touch_loop_heartbeat(home_dir, fence=fence)
        try:
            return original(*args, **kwargs)
        finally:
            try:
                touch_loop_heartbeat(home_dir, fence=fence)
            except StoppedByOperator:
                pass

    llm.generate = generate
    return llm


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
    if data.get("phase") == "starting":
        # THE SPAWN RESERVATION: running while the recorded child is alive - or, before Popen returned (no child
        # recorded yet), the spawner (os.kill 0 after reaping our own zombie, never a clock) - and still the SAME
        # process: its recorded start-time token is compared, as day/between do (Fable F5: a reused pid read
        # "already starting" forever). One ps per status read, never per effect.
        child = data.get("pid")
        pid = child if child else data.get("spawner_pid")
        token = str((data.get("pid_started_at") if child else data.get("spawner_started_at")) or "")
        alive = _pid_alive_reaped(pid)
        if alive and token:
            current = _pid_start_time(pid)
            if current is not None and current != token:
                alive = False
        data["running"] = alive
        return data
    running = bool(
        data.get("phase") in ("day", "between") and _pid_alive(data.get("pid"))
    )
    if running:
        # PID-IDENTITY TOKEN (gateway adversary 2, closes the `between`
        # corpse hole the day-only staleness belt left): when the file
        # carries pid_started_at, the pid must be the SAME INCARNATION —
        # a recycled pid's start time differs, and the corpse reads
        # not-running in EVERY phase, forever-409s and innocent-freeze-kills
        # both die. Files without the stamp (older writers) skip the check.
        stamped = str(data.get("pid_started_at") or "")
        if stamped:
            current = _pid_start_time(data.get("pid"))
            if current is not None and current != stamped:
                running = False
    if running and data.get("phase") == "day":
        # Stale-day belt (see LOOP_STATUS_STALE_SECONDS): kept BESIDE the
        # token — it also catches a live-pid loop that stopped heartbeating
        # (wedged process), which the token cannot. The live loop
        # heartbeats "day" at every tick boundary; a frozen updated_at past
        # the bound means something is wrong regardless of pid identity.
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


def record_loop_substrate(home_dir: Path, provider: str, model: str, *, fence: Any) -> None:
    """Stamp the mind the loop is CURRENTLY running into loop_status (entity
    c78 render ask). The factory calls this at each day-open after it
    re-resolves the home's substrate, so an observer's staleness cue can
    compare the operator's substrate change time against the mind actually
    in use — a change older than this stamp's `updated_at` has been picked
    up, not still pending. Read-merge on the existing status dict (phase and
    liveness fields preserved); best-effort, never fatal - and fenced like
    write_loop_status (one locked step; never after a Stop)."""
    import json
    from datetime import datetime, timezone

    from ..utils.atomic_files import atomic_write_text

    with home_state_lock(home_dir):
        _verify_fence(fence, home_dir, "recording the loop substrate", read_lifecycle_generation(home_dir),
                      sentinel_ok=False)
        prior = _read_loop_status_raw(home_dir)
        prior.pop("running", None)  # derived; never persisted
        prior["substrate"] = {"provider": str(provider), "model": str(model)}
        prior["substrate_at"] = datetime.now(timezone.utc).isoformat()
        prior.setdefault("phase", "between")
        try:
            atomic_write_text(Path(home_dir) / "loop_status", json.dumps(prior) + "\n")
        except OSError:
            pass


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


# ------------------------------------------------- THE home state lock and the lifecycle generation (Codex review 2
# #1; agreed with the gateway 2026-11-09): every mutation of a home's phase state - phases.yaml (the personal grant,
# resume_to, the generation) and the state file - happens inside `home_state_lock(home)`, re-reads and merges against
# the CURRENT data inside it, and is FENCED by the lifecycle generation. Stop bumps the generation in the same
# critical section as its revocation, so a write decided before a Stop (carrying the LifecycleFence it was admitted
# with, lifecycle_fence.py) is refused after it instead of restoring what Stop took away. The fence is a REQUIRED
# argument of every writer here (STOP-LANE-DESIGN §2.2): an unfenced call does not run. KILL_SWITCH (the Stop's own
# writes) and RESUME (Resume's) are the only two that pass unchecked, and only for what those acts write.
#
# Lock order: the gateway's own per-home lock is the OUTER lock and this one the INNER; no runtime path calls the
# gateway (or takes any other lock) while holding it. Re-entrant per thread (a nested `with` is a no-op), so a host
# holds it across "check, then call the runtime writers". Cross-process by fcntl.flock on a dedicated file that is
# never replaced (stable inode); where there is no fcntl, a per-home in-process lock only.
HOME_STATE_LOCK_FILENAME = ".phase_state.lock"
HOME_STATE_LOCK_TIMEOUT_S = 30.0
LIFECYCLE_GENERATION_KEY = "lifecycle_generation"


class HomeStateLockTimeout(TimeoutError):
    """The home state lock was not acquired within `timeout_s`."""


_home_state_held = threading.local()
_home_state_thread_locks: Dict[str, threading.Lock] = {}
_home_state_registry = threading.Lock()


@contextmanager
def home_state_lock(home_dir: Path, *, timeout_s: Optional[float] = HOME_STATE_LOCK_TIMEOUT_S) -> Iterator[None]:
    """Hold THE home state lock (see above). `timeout_s=None` waits without a limit (Stop, the visit doors);
    otherwise HomeStateLockTimeout after that long."""
    key = str(Path(home_dir).resolve())
    depth: Dict[str, int] = getattr(_home_state_held, "depth", None) or {}
    _home_state_held.depth = depth
    if depth.get(key):
        depth[key] += 1
        try:
            yield
        finally:
            depth[key] -= 1
        return
    deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
    try:
        import fcntl
    except ImportError:  # non-POSIX: one in-process lock per home (no cross-process exclusion there)
        fcntl = None  # type: ignore[assignment]
    fh = None
    thread_lock: Optional[threading.Lock] = None
    if fcntl is not None:
        fh = open(Path(home_dir) / HOME_STATE_LOCK_FILENAME, "a+", encoding="utf-8")
        try:
            if deadline is None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            else:
                while True:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise HomeStateLockTimeout(
                                f"the home state lock of {home_dir} was not acquired within {timeout_s} s") from None
                        time.sleep(0.02)
        except BaseException:
            fh.close()
            raise
    else:
        with _home_state_registry:
            thread_lock = _home_state_thread_locks.setdefault(key, threading.Lock())
        if not thread_lock.acquire(timeout=-1 if deadline is None else max(0.0, deadline - time.monotonic())):
            raise HomeStateLockTimeout(f"the home state lock of {home_dir} was not acquired within {timeout_s} s")
    depth[key] = 1
    try:
        yield
    finally:
        depth.pop(key, None)
        if fh is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
        if thread_lock is not None:
            thread_lock.release()


def _load_phases_for_write(home_dir: Path, what: str) -> Dict[str, Any]:
    """phases.yaml as a mapping for a writer, with the writers' refusals: a corrupt file or a newer schema is never
    overwritten (it could silently lose other phase sections)."""
    import yaml

    path = Path(home_dir) / PHASES_FILENAME
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - never silently clobber a broken file
            raise ValueError(
                f"{PHASES_FILENAME} is unreadable ({e}) - repair or remove it before {what}; "
                "overwriting it here could silently lose other phase sections"
            ) from e
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{PHASES_FILENAME} is not a mapping - repair or remove it before {what}")
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
    return existing


def _generation_of(data: Dict[str, Any]) -> int:
    try:
        return int(data.get(LIFECYCLE_GENERATION_KEY) or 0)
    except (TypeError, ValueError):
        return 0


def _mutate_phases(home_dir: Path, what: str, mutate: Callable[[Dict[str, Any]], None], *,
                   fence: Any,
                   timeout_s: Optional[float] = HOME_STATE_LOCK_TIMEOUT_S) -> Dict[str, Any]:
    """Read phases.yaml, check the fence against the generation read THERE, apply `mutate` to the CURRENT mapping and
    replace the file atomically - all inside the home state lock. Returns the mapping written. `fence` is required:
    a LifecycleFence, or KILL_SWITCH / RESUME from the public writers that allow them for this write."""
    import yaml

    from ..utils.atomic_files import atomic_write_text

    with home_state_lock(home_dir, timeout_s=timeout_s):
        existing = _load_phases_for_write(home_dir, what)
        found = _generation_of(existing)
        _publish_generation_under_lock(home_dir, found)
        _verify_fence(fence, home_dir, what, found, sentinel_ok=True)
        mutate(existing)
        existing["schema_version"] = PHASES_SCHEMA_VERSION
        atomic_write_text(Path(home_dir) / PHASES_FILENAME, yaml.safe_dump(existing, sort_keys=False))
        return existing


def read_lifecycle_generation(home_dir: Path) -> int:
    """The home's lifecycle generation (0 when never bumped). Stop bumps it; work admitted at it carries a
    LifecycleFence. A corrupt phases.yaml raises ValueError (LifecycleUnreadable; no writer would write over it
    either). Every read refreshes this process's fence table (lifecycle_fence.py) for free."""
    return _read_generation_truth(home_dir)


def bump_lifecycle_generation(home_dir: Path, *, written_by: str,
                              timeout_s: Optional[float] = HOME_STATE_LOCK_TIMEOUT_S) -> int:
    """Increment the lifecycle generation (Stop) and return the new value. Hold `home_state_lock` around it and
    the other Stop writes to make them one critical section (hard_stop_loop does). In the same critical section,
    AFTER phases.yaml, the signal file `.lifecycle_generation` is replaced and this process's fence table updated:
    its own fences see the Stop at their next compare, every other process's at its next stat."""
    if not str(written_by or "").strip():
        raise ValueError("written_by is required - the bump names who stopped him")
    out: Dict[str, int] = {}

    def _bump(data: Dict[str, Any]) -> None:
        out["generation"] = _generation_of(data) + 1
        data[LIFECYCLE_GENERATION_KEY] = out["generation"]

    with home_state_lock(home_dir, timeout_s=timeout_s):
        _mutate_phases(home_dir, "bumping the lifecycle generation", _bump, fence=KILL_SWITCH, timeout_s=timeout_s)
        _signal_bump(home_dir, out["generation"])
    return out["generation"]
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
    fence: Any,
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

    Returns the bucket as read_personal_grant will now answer it. Written inside the home state lock against the
    current file; `fence` (required) refuses it when a Stop happened since the act was admitted
    (StaleLifecycleGeneration). KILL_SWITCH and RESUME may only revoke (mode disabled)."""
    name = str(mode or "").strip().lower()
    if name not in PERSONAL_GRANT_MODES:
        raise ValueError(
            f"unknown personal mode {mode!r}: modes are {'/'.join(PERSONAL_GRANT_MODES)}"
        )
    if name != "disabled" and (fence is KILL_SWITCH or fence is RESUME):
        raise TypeError(f"arming personal time: {fence!r} only revokes - pass the LifecycleFence of the arming act")

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

    # Field-merge, INSIDE the home state lock against the current file (Codex review 2 #1): replace ONLY the
    # personal activation bucket, preserving any non-activation keys a future schema puts beside them (tools/skills
    # per phase live in their own sections and are never touched here); a fenced write refuses on a moved generation.
    def _merge(existing: Dict[str, Any]) -> None:
        prior_personal = existing.get("personal")
        merged_personal: Dict[str, Any] = dict(prior_personal) if isinstance(prior_personal, dict) else {}
        for key in _PERSONAL_ACTIVATION_FIELDS:
            merged_personal.pop(key, None)
        merged_personal.update(bucket)
        existing["personal"] = merged_personal

    _mutate_phases(home_dir, "arming" if name != "disabled" else "revoking personal", _merge, fence=fence)
    return read_personal_grant(home_dir)


# ------------------------------------------------- resume_to (contract-phases v2 §6.4; CONTRACT v3 D1/D3/D4)
# One more runtime-owned bucket of phases.yaml, with ONE writer, field-merged like the personal grant:
#
#   resume_to: {phase: personal|work|null, written_by, written_at}
#              the own-time phase an interruption (a visit, a sleep, a job, a request) took over, which its end
#              returns to while the grant still stands (spec v23 OWN-TIME-IS-A-PRIVILEGE). The field name matches
#              the gateway sidecar's (`resume_to`).
#
# Until a home's bucket exists, the read falls back to the gateway's sidecar `<entities_dir>/.life_phase/<slug>.json`
# (its `resume_to` key), so a gateway that still writes the sidecar is read as it is. `migrate_phase_sidecar` copies
# it once, idempotently; from then on the sidecar is a projection only and the gateway writes through write_resume_to.
#
# THERE IS NO WORK TOGGLE (CONTRACT v3 D1, operator 2026-11-09: "work starts when a job is queued"). The `work`
# bucket a24350a wrote (`work: {enabled, written_by, written_at}`) and the sidecar's `work` key are IGNORED: nothing
# reads them, nothing migrates them, and a phases.yaml write carries an old `work` key through untouched like any
# other key it does not own. The work-off pause record is retired with it (its file is never read).
RESUME_TO_BUCKET = "resume_to"
RESUME_TO_PHASES = ("personal", "work")
GATEWAY_PHASE_SIDECAR_DIRNAME = ".life_phase"


def gateway_phase_sidecar_path(home_dir: Path) -> Path:
    """The gateway's per-home phase sidecar (`<entities_dir>/.life_phase/<slug>.json`, the home being
    `<entities_dir>/<slug>`)."""
    home = Path(home_dir)
    return home.parent / GATEWAY_PHASE_SIDECAR_DIRNAME / f"{home.name}.json"


def _read_gateway_phase_sidecar(home_dir: Path) -> Dict[str, Any]:
    import json as _json

    try:
        raw = _json.loads(gateway_phase_sidecar_path(home_dir).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001 - an absent/corrupt sidecar is no value
        return {}


def _phases_bucket(home_dir: Path, key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(bucket | None, problem | None) read from phases.yaml; a missing file or key is (None, None)."""
    path = Path(home_dir) / PHASES_FILENAME
    if not path.exists():
        return None, None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, f"#FALLBACK unreadable {PHASES_FILENAME} ({e})"
    if data is not None and not isinstance(data, dict):
        return None, f"#FALLBACK {PHASES_FILENAME} is not a mapping"
    bucket = (data or {}).get(key)
    if bucket is None:
        return None, None
    if not isinstance(bucket, dict):
        return None, f"#FALLBACK {PHASES_FILENAME} {key} is not a mapping"
    return bucket, None


def _write_phases_bucket(home_dir: Path, key: str, bucket: Dict[str, Any], *, fence: Any) -> None:
    """Replace ONLY `key` in phases.yaml (field-merge against the CURRENT file inside the home state lock; every
    other key rides through), with the writers' refusals: a corrupt file or a newer schema is never overwritten, and
    the (required) fence refuses on a moved lifecycle generation."""

    def _set(existing: Dict[str, Any]) -> None:
        existing[key] = dict(bucket)

    _mutate_phases(home_dir, f"writing {key}", _set, fence=fence)


def read_resume_to(home_dir: Path) -> Dict[str, Any]:
    """The phase an interruption took over: {phase: personal|work|None, written_by, written_at, source}."""
    bucket, problem = _phases_bucket(home_dir, RESUME_TO_BUCKET)
    if problem:
        return {"phase": None, "written_by": None, "written_at": None, "source": "phases.yaml", "note": problem}
    if bucket is not None:
        phase = bucket.get("phase")
        return {"phase": phase if phase in RESUME_TO_PHASES else None,
                "written_by": str(bucket.get("written_by") or "") or None,
                "written_at": str(bucket.get("written_at") or "") or None, "source": "phases.yaml"}
    side = _read_gateway_phase_sidecar(home_dir)
    if "resume_to" in side:
        phase = side.get("resume_to")
        return {"phase": phase if phase in RESUME_TO_PHASES else None, "written_by": None, "written_at": None,
                "source": "gateway-sidecar"}
    return {"phase": None, "written_by": None, "written_at": None, "source": "default"}


def write_resume_to(home_dir: Path, phase: Optional[str], *, written_by: str, fence: Any) -> Dict[str, Any]:
    """THE one writer of the resume_to bucket (personal | work | None); `fence` (required) refuses a write decided
    before a Stop. KILL_SWITCH and RESUME may only clear it (phase None)."""
    if phase is not None and phase not in RESUME_TO_PHASES:
        raise ValueError(f"resume_to is one of {RESUME_TO_PHASES} or None, not {phase!r}")
    if phase is not None and (fence is KILL_SWITCH or fence is RESUME):
        raise TypeError(f"remembering a return: {fence!r} only clears it - pass the LifecycleFence of the act")
    by = str(written_by or "").strip()
    if not by:
        raise ValueError("written_by is required")
    from datetime import datetime, timezone

    _write_phases_bucket(home_dir, RESUME_TO_BUCKET, {
        "phase": phase, "written_by": by, "written_at": datetime.now(timezone.utc).isoformat(),
    }, fence=fence)
    return read_resume_to(home_dir)


def migrate_phase_sidecar(home_dir: Path, *, fence: Any, migrated_by: str = "runtime:migration") -> Dict[str, Any]:
    """Copy the gateway sidecar's `resume_to` into its phases.yaml bucket, ONCE (contract §6.4). Idempotent: a bucket
    that exists is never touched (the sidecar is a projection from then on). The sidecar's `work` key is ignored -
    there is no work toggle (CONTRACT v3 D1). Returns {"resume_to": "migrated"|"present"|"absent", "sidecar":
    <path>}; the gateway records migrated_at.

    The absence check, the sidecar read and the write are ONE step inside the home state lock, fenced (Codex review 3
    #5): a migration admitted before a Stop is refused after it, and one admitted after it finds the bucket Stop wrote
    (resume_to cleared) - an old personal return never comes back. Never a sentinel: a migration is no Stop."""
    from datetime import datetime, timezone

    out: Dict[str, Any] = {"sidecar": str(gateway_phase_sidecar_path(home_dir))}
    by = f"{migrated_by} (from the gateway sidecar)"
    with home_state_lock(home_dir):
        existing = _load_phases_for_write(home_dir, "migrating the gateway sidecar")
        found = _generation_of(existing)
        _publish_generation_under_lock(home_dir, found)
        _verify_fence(fence, home_dir, "migrating the gateway sidecar", found, sentinel_ok=False)
        bucket, problem = _phases_bucket(home_dir, RESUME_TO_BUCKET)
        if problem:
            raise ValueError(problem)
        side = _read_gateway_phase_sidecar(home_dir)
        if bucket is not None:
            out["resume_to"] = "present"
        elif "resume_to" in side:
            value = side.get("resume_to")
            _write_phases_bucket(home_dir, RESUME_TO_BUCKET, {
                "phase": value if value in RESUME_TO_PHASES else None, "written_by": by,
                "written_at": datetime.now(timezone.utc).isoformat(),
            }, fence=fence)
            out["resume_to"] = "migrated"
        else:
            out["resume_to"] = "absent"
    return out


def personal_grant_end_cause(grant: Dict[str, Any]) -> str:
    """The ruled cause word for a mid-life grant end (phase-vocabulary v3
    closed set): "grant_expired" when a timer genuinely lapsed, otherwise
    "grant_revoked" (disabled bucket, deleted file, malformed content — all
    fail-closed as acts against the grant). Only meaningful when
    personal_grant_refusal(grant) is not None."""
    if str(grant.get("mode") or "") == "timer":
        expires = str(grant.get("expires_at") or "").strip()
        if expires:
            from datetime import datetime, timezone

            try:
                deadline = datetime.fromisoformat(expires)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) >= deadline:
                    return "grant_expired"
            except ValueError:
                pass  # unreadable expiry is a fail-closed act, not a lapse
    return "grant_revoked"


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
    stamped_token = str(status.get("pid_started_at") or "")

    # STOP, ONE CRITICAL SECTION, BEFORE ANY SIGNAL (CONTRACT v3 D3; Codex review 2 #1/#7): own time revoked (the
    # gateway Stop's own write, mode disabled by kill_switch; skipped when already off), the remembered return
    # cancelled, and the lifecycle generation bumped - under the home state lock, waiting without a limit (a Stop never
    # fails on a busy lock). A writer that decided before this moment (the dying loop included) is refused after it.
    personal_note: Optional[str] = None
    personal_revoked = False
    resume_to_note: Optional[str] = None
    generation: Optional[int] = None
    generation_note: Optional[str] = None
    with home_state_lock(home_dir, timeout_s=None):
        try:
            if read_personal_grant(home_dir).get("mode") != "disabled":
                write_personal_grant(home_dir, mode="disabled", granted_by="kill_switch", fence=KILL_SWITCH)
                personal_revoked = True
        except Exception as e:  # noqa: BLE001 - the stop itself stands; the result says what failed
            personal_note = f"personal not revoked: {e}"
        try:
            write_resume_to(home_dir, None, written_by="kill_switch", fence=KILL_SWITCH)
        except Exception as e:  # noqa: BLE001
            resume_to_note = f"resume_to not cleared: {e}"
        try:
            generation = bump_lifecycle_generation(home_dir, written_by="kill_switch", timeout_s=None)
        except LifecycleSignalUnwritten as e:
            # The Stop stands (phases.yaml moved, this process's fences refuse); only its signal file is missing -
            # other processes see it at their next truth read (Fable F9). Said, never "not bumped".
            generation = e.generation
            generation_note = str(e)
        except Exception as e:  # noqa: BLE001
            generation_note = f"lifecycle generation not bumped: {e}"

    # SIGNAL ONLY A VERIFIED OWN-TIME LOOP OF THIS HOME (gateway slice 2026-11-09; Codex review 2026-11-09 #4/#7):
    # the incarnation is verified (alive, not the caller, the recorded start-time token equal to the OS's) AND the
    # process is this home's loop invocation (`<python> -m abstractruntime.identity.life --home <this home>`).
    # Anything unknown fails closed: no token, an unreadable start time or command line, a denied permission.
    verified, outcome = _verified_loop_incarnation(pid, stamped_token, home_dir)

    killed = False
    escalated = False
    signal_name: Optional[str] = None
    if verified:
        try:
            os.kill(pid, signal.SIGTERM)
            signal_name = "SIGTERM"
        except ProcessLookupError:
            pass
        except OSError as e:
            outcome = f"refused: SIGTERM was not delivered ({e})"
        # Short grace for the interpreter to die - NOT for ceremony (there is none in a freeze); then escalate,
        # but only over the SAME verified incarnation (it may have died, or the pid been reused, meanwhile).
        deadline = _time.monotonic() + 3.0
        while _time.monotonic() < deadline:
            if _pid_state(pid) == "gone":
                killed = True
                break
            _time.sleep(0.1)
        if not killed:
            still, why = _verified_loop_incarnation(pid, stamped_token, home_dir)
            if still:
                try:
                    os.kill(pid, signal.SIGKILL)
                    escalated = True
                    signal_name = "SIGKILL"
                except OSError:
                    pass
                deadline = _time.monotonic() + 1.0
                while _time.monotonic() < deadline and _pid_state(pid) != "gone":
                    _time.sleep(0.05)
            killed = _pid_state(pid) == "gone"
            if not killed:
                outcome = (
                    "the loop is still alive after SIGKILL" if escalated
                    else f"exit not confirmed: {why}"
                )
        if killed:
            outcome = "killed"

    # The status file must read stopped immediately (the dead process can no longer write its own transition) -
    # naming the loop it is about (its pid and token when it was verified), never this caller.
    try:
        write_loop_status(
            home_dir, "stopped", stopped_by="kill_switch",
            owner=((pid, stamped_token or None) if verified else (None, None)), fence=KILL_SWITCH,
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        # The explicit result (gateway contract, 2026-11-09): killed only once the exit is CONFIRMED (reaped, gone
        # from the process table, or a zombie) - never inferred from a denied permission or an unreadable status.
        "killed": bool(killed),
        "pid": pid or None,
        "signal": signal_name,
        "reason": outcome,
        "requested_reason": str(reason or ""),
        # The remembered return is cancelled (resume_to null, written_by kill_switch); False + note when the write
        # failed (a corrupt phases.yaml or a newer schema).
        "resume_to_cleared": resume_to_note is None,
        **({"resume_to_note": resume_to_note} if resume_to_note else {}),
        # Own time revoked by this stop (False when it was already off, or when the write failed: personal_note).
        "personal_revoked": personal_revoked,
        **({"personal_note": personal_note} if personal_note else {}),
        # The lifecycle generation this Stop moved to (None + generation_note when the bump failed).
        "lifecycle_generation": generation,
        **({"generation_note": generation_note} if generation_note else {}),
        # Legacy keys (kept for existing callers).
        "frozen": True,
        "was_running": bool(verified and signal_name),
        "escalated_to_sigkill": escalated,
        "requested_by": str(requested_by or "admin"),
        "status": loop_process_status(home_dir),
    }


def _verified_loop_incarnation(pid: int, token: str, home_dir: Path) -> Tuple[bool, str]:
    """(True, "") when `pid` is a live own-time loop of `home_dir` whose start time equals `token`; else (False,
    the plain reason). Fails closed whenever anything is unknown."""
    import os

    if pid <= 0:
        return False, "no loop pid recorded"
    if pid == os.getpid():
        return False, "refused: the recorded pid is the calling process itself"
    state = _pid_state(pid)
    if state == "gone":
        return False, "no live process at the recorded pid"
    if state != "alive":
        return False, "refused: the process at the recorded pid cannot be checked (permission or status unknown)"
    if not token:
        return False, "refused: no start-time token recorded - the incarnation is unknown"
    current = _pid_start_time(pid)
    if current is None:
        return False, "refused: the start time of the process cannot be read"
    if current != token:
        return False, "refused: the pid now belongs to another process (recycled)"
    if not _is_loop_invocation(_pid_command(pid), home_dir):
        return False, "refused: the process at the recorded pid is not this home's own-time loop"
    return True, ""


def _is_loop_invocation(command: Optional[str], home_dir: Path) -> bool:
    """True when `command` (ps's command line) runs the loop module for this home: `<python> -m
    abstractruntime.identity.life --home <home>` with the module directly after the interpreter (the argv
    spawn_loop_process uses) - never the module name appearing later as an argument."""
    if not command:
        return False
    _head, sep, rest = command.partition(" ")
    if not sep:
        return False
    prefix = LOOP_MODULE_ARGV + " --home "
    if not rest.startswith(prefix):
        return False
    tail = rest[len(prefix):]
    homes = {str(Path(home_dir))}
    try:
        homes.add(str(Path(home_dir).resolve()))
    except (OSError, RuntimeError):
        pass
    return any(tail == h or tail.startswith(h + " ") for h in homes)


def _pid_state(pid: Any) -> str:
    """"gone" (reaped here, ESRCH, or a zombie), "alive", or "unknown" (permission denied or no answer). Only
    "gone" confirms an exit (Codex review 2026-11-09 #7: a denied permission is not a death)."""
    import os
    import subprocess

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return "unknown"
    try:
        done, _status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return "gone"
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "unknown"
    except OSError:
        return "unknown"
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
        stat = (out.stdout or "").strip().upper()
    except Exception:  # noqa: BLE001
        return "alive"  # it answered signal 0: it exists
    return "gone" if stat.startswith("Z") else "alive"


# The loop's own command-line mark: spawn_loop_process and the CLI run `python -m abstractruntime.identity.life`.
LOOP_MODULE_ARGV = "-m abstractruntime.identity.life"


def _pid_command(pid: Any) -> Optional[str]:
    """The command line of `pid` as ps prints it (None on any failure)."""
    import os
    import subprocess

    try:
        out = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
        text = (out.stdout or "").strip()
        return text or None
    except Exception:  # noqa: BLE001
        return None


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


def loop_start_refusal(home_dir: Path) -> Optional[str]:
    """Why a loop start is refused (None = admitted). A start is admitted when the loop has something it may do: a
    queued job (work runs whatever the personal grant says, CONTRACT v3 D1) or the operator's personal grant. The
    day gate decides the rest at every boundary. Pure read."""
    grant_refusal = personal_grant_refusal(read_personal_grant(home_dir))
    if grant_refusal is None or jobs_queued(home_dir):
        return None
    return f"no personal time: {grant_refusal} - and no job is queued"


def spawn_loop_process(
    home_dir: Path,
    *,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    tick_seconds: float = 20.0,
    ticks_per_day: int = 8,
    rest_minutes: float = 30.0,
    # 36 seats (maintainer, 2026-07-09): the 12% token budget still seats 36
    # rich digests — seats fill, tokens hold.
    shelf_size: int = 36,
    # ~50k DEFAULT (operator re-ruling 2026-08-01: "it is acceptable to go
    # to 200k context, but ideally, let's have a (soft) recommended target
    # of 50k tokens" — superseding the 2026-07-13 "up to 40k tokens
    # roughly" figure; the NOT-a-hardcap spirit carries over). 51200 =
    # 50 × 1024, the same power-of-two convention the old 40960 = 40 × 1024
    # used, so the default start sits AT the recommendation instead of
    # warning against itself. An explicit operator value always wins in
    # either direction (soft warnings only — never a silent cap, per the
    # no-silent-fallback ADR discipline).
    context_window: int = 51200,
    # HOST-RESOLVED ENDPOINT PROFILE (operator 2026-08-01: own time NEVER
    # started on an endpoint:* substrate — the spawned child has no
    # ~/.abstractcore profile for a gateway-scoped endpoint, so create_llm
    # raised "Unknown provider: endpoint:airelay" at day-open while visits,
    # resolved in-process by the host's ambient resolver, worked fine). The
    # host passes its PRIVATE resolution here (private_resolution() shape:
    # provider/provider_family, base_url, api_key, ...); it rides the child
    # env and becomes the child's ambient resolver (core's 2026-07-26
    # cross-package channel). May carry secrets — the child never logs it.
    endpoint_profile: Optional[Dict[str, Any]] = None,
    # The operator's own act (his grant, start or retry): the child's first day is his whatever pulls. A restore
    # after an interruption passes False (the gate governs: nothing pulling = he rests inside personal).
    first_day_unconditional: bool = False,
    # The phase the run is started to (re)enter: "personal" (the operator's act, or a restore of own time), "work"
    # (a restore of work alone) or None (the home's resume_to decides; the default for hosts that predate it).
    return_to: Optional[str] = None,
    # {capability: {"spec": <provider spec>, "profile": <the host's private resolution>}} for the capability clients
    # (summaries authors titles and summaries). May carry secrets: one env key, never logged.
    capability_endpoint_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
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
            if status.get("phase") == "starting":
                raise RuntimeError(
                    f"his own time is already starting (pid {status.get('pid') or status.get('spawner_pid')}, "
                    f"admitted at lifecycle generation {status.get('generation')})"
                )
            raise RuntimeError(
                f"his own time is already running (pid {status.get('pid')}, phase {status.get('phase')})"
            )

        # ADMISSION at the spawn door (laurent 12:44; CONTRACT v3 D1): a start surface refuses SYNCHRONOUSLY when
        # the loop would have nothing it may do - spawning a child that dies in its own log is a silent refusal.
        # A queued job is admitted whatever the personal grant says (work is the job queue, never a grant).
        refusal = loop_start_refusal(home_dir)
        if refusal is not None:
            raise RuntimeError(refusal)

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

        # THE RESERVATION (Codex review 3 #6): admitted at the current generation and written BEFORE the child
        # exists; the child is told that generation and refuses to start (writing nothing) once it moved.
        fence = LifecycleFence.admit(home_dir)
        _reserve_loop_start(home_dir, fence=fence)

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
            # The generation of the reservation: the child's boot refuses (zero writes) when a Stop moved it.
            "--generation", str(int(fence.generation)),
        ]
        if first_day_unconditional:
            argv.append("--first-day-unconditional")
        argv += ["--return-to", str(return_to) if return_to in ("personal", "work") else "none"]
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

        # The host-resolved endpoint profile rides the child env keyed to the
        # exact provider spec — main() installs it as the ambient resolver.
        if capability_endpoint_profiles:
            env["ABSTRACTRUNTIME_CAPABILITY_ENDPOINT_PROFILES_JSON"] = json.dumps({
                str(cap): {"spec": str((v or {}).get("spec") or "").strip().lower(),
                           "profile": dict((v or {}).get("profile") or {})}
                for cap, v in capability_endpoint_profiles.items() if isinstance(v, dict)
            })
        if endpoint_profile:
            env["ABSTRACTRUNTIME_ENDPOINT_PROFILE_JSON"] = json.dumps(
                {"spec": str(provider or "").strip().lower(), "profile": dict(endpoint_profile)}
            )

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
        try:
            # The child's start-time token (one ps per spawn, read before the lock): a reused pid never keeps the
            # reservation alive, and a Stop can verify the incarnation of a child that has not registered yet.
            _record_spawned_child(home_dir, proc.pid, fence=fence, token=_pid_start_time(proc.pid))
        except StoppedByOperator:
            pass  # a Stop since the reservation: its record stands, and the child refuses itself at its boot

        # Honesty at the door: a child that dies within the first moment
        # (import error, bad interpreter) must NOT report started:true.
        # 0.6s is imperceptible for an operator action and catches the
        # whole instant-death class.
        import time as _time

        _time.sleep(0.6)
        exit_code = proc.poll()
        if exit_code is not None:
            if fence.moved():
                raise StoppedByOperator("while his own time was starting")
            try:
                _release_failed_reservation(home_dir, fence=fence)
            except StoppedByOperator:
                raise StoppedByOperator("while his own time was starting") from None
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
            except OSError:
                pass
            raise RuntimeError(
                f"his own time failed to start (the loop process exited immediately, code {exit_code}). "
                f"Log tail: {tail.strip()[-300:]}"
            )
        if fence.moved():
            # THE OPERATOR'S STOP overtook the start (Fable F6): the child is alive but refuses itself at its boot (or
            # the Stop kills it) - never report own time started over the Stop's own record.
            raise StoppedByOperator("while his own time was starting")

        return {
            "pid": proc.pid,
            "log": str(log_path),
            "provider": provider,
            "model": model,
            "tick_seconds": float(tick_seconds),
            "ticks_per_day": int(ticks_per_day),
            "rest_minutes": float(rest_minutes),
            "shelf_size": int(shelf_size),
            "generation": int(fence.generation),
        }
    finally:
        try:
            lock_fh.close()
        except Exception:  # noqa: BLE001
            pass


def _pid_alive_reaped(pid: Any) -> bool:
    """`_pid_alive` after reaping `pid` when it is OUR exited child (a child refused at its boot is a zombie in its
    spawner until reaped, and a zombie answers signal 0). One waitpid + one kill: no subprocess."""
    import os

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        done, _status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass
    return _pid_alive(pid)


def _pid_alive(pid: Any) -> bool:
    """os.kill(pid, 0) liveness probe. DELIBERATE: EPERM (a live process we
    may not signal) lands in OSError and reads as DEAD — the safe direction
    for every current caller (doors don't wait on it, freeze won't SIGKILL
    it). Do not "fix" this into signaling paths without splitting the two
    meanings (gateway adversary 2, finding 4)."""
    import os

    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def visitor_waiting(state: Dict[str, Any]) -> bool:
    """A visit door is waiting on this home: the visiting posture (asleep +
    mode=visiting, or the door's auto-yield reason) — the same predicate the
    tick boundary reads."""
    return str(state.get("state") or "") == "asleep" and (
        str(state.get("mode") or "") == "visiting" or "auto-yield" in str(state.get("reason") or "")
    )


def await_loop_quiescent(
    home_dir: Path,
    *,
    timeout_seconds: Optional[float] = 900.0,
    poll_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until no summon is open on this home (phase != "day").

    A "day" status that is not running (dead pid, recycled pid, stale
    heartbeat — `read_loop_status`) counts as quiescent (a crashed loop must
    not deadlock the visitor). `timeout_seconds=None` waits without a limit
    (the visit doors: a visit never fails, operator 2026-11-09, root
    ADR-0027). Returns False on timeout — the caller REFUSES to summon (one
    life, one summon; never a double)."""
    # #[WARNING:TIMEOUT] optional caller bound; None = no limit.
    deadline = None if timeout_seconds is None else time.monotonic() + float(timeout_seconds)
    while True:
        status = read_loop_status(home_dir)
        if status.get("phase") != "day":
            return True
        if not status.get("running"):
            return True  # stale status from a dead loop
        if deadline is not None and time.monotonic() >= deadline:
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
    sleeps = wakes = self_elected = operator = visit_yields = transitions = 0
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
                if rec.get("marker"):
                    # phase_changed rows (c4837) are PHASE biography, not
                    # state transitions — excluded from the sleep-share fold
                    # or every marker would dilute the denominator.
                    continue
                transitions += 1
                state = str(rec.get("state") or "").strip().lower()
                if state == "asleep":
                    sleeps += 1
                    who = str(rec.get("written_by"))
                    # Visit-door yields are MACHINE bookkeeping (entity c2465
                    # ask 2): before the structural stamp they counted as
                    # operator sleeps and inflated that bucket. Legacy yields
                    # (written_by=operator, mode=visiting) fold in too.
                    if who == "self":
                        self_elected += 1
                    elif who == "visit-door" or str(rec.get("mode") or "") == "visiting":
                        visit_yields += 1
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
        "visit_yields": visit_yields,
        "wakes": wakes,
        "transitions": transitions,
        "sleep_share": (sleeps / transitions) if transitions else 0.0,
    }


def write_entity_state(
    home_dir: Path,
    state: str,
    *,
    reason: str = "",
    mode: str = "",
    written_by: str = "operator",
    wake_at: str = "",
    fence: Any,
    timeout_s: Optional[float] = HOME_STATE_LOCK_TIMEOUT_S,
) -> Dict[str, Any]:
    """Write the operator (or SELF) state (the CLI/gateway surface calls this).

    Inside the home state lock (Codex review 2 #1); `fence` (REQUIRED, STOP-LANE-DESIGN §2.2) is checked there
    against the generation read immediately before the write: a state decided before a Stop is refused after it
    (StaleLifecycleGeneration, nothing written). KILL_SWITCH writes only paused (the Stop), RESUME only awake.

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
    sentinel_ok = (fence is KILL_SWITCH and state == "paused") or (fence is RESUME and state == "awake")
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
    # Optional explicit wake deadline (decision:sleep-is-bounded): UTC-
    # normalized at the write boundary, same rule as WAIT_UNTIL deadlines.
    if wake_at:
        from ..core.runtime import normalize_utc_iso

        normalized_wake = normalize_utc_iso(str(wake_at))
        if normalized_wake:
            payload["wake_at"] = normalized_wake
    path = Path(home_dir) / "state"
    # ATOMIC (gateway whole-package audit P2-6, 2026-07-13, security-adjacent:
    # the reader fails OPEN — read_entity_state treats an unreadable file as
    # awake — so a torn write could transiently show a PAUSED entity as awake
    # to every gate that polls this file). Same helper the grant writer uses.
    from ..utils.atomic_files import atomic_write_text

    with home_state_lock(home_dir, timeout_s=timeout_s):
        _verify_fence(fence, home_dir, f"writing state {state}", read_lifecycle_generation(home_dir),
                      sentinel_ok=sentinel_ok)
        atomic_write_text(path, json.dumps(payload, indent=1) + "\n")
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
    return cue or None  # whole (root ADR 0054; a 400-character cap used to apply)


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


def _is_own_sleep(state: Dict[str, Any]) -> bool:
    """A sleep the loop owns (LOOP_OWN_SLEEP_WRITERS), never the visit door's posture over it."""
    return (
        str(state.get("state") or "") == "asleep"
        and str(state.get("mode") or "") != "visiting"
        and str(state.get("written_by") or "") in LOOP_OWN_SLEEP_WRITERS
    )


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
        substrate_changed: Optional[Callable[[], bool]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        out: Callable[[str], None] = print,
        first_day_unconditional: bool = False,
        return_to: Optional[str] = PHASE_PERSONAL,
        fence: Optional[LifecycleFence] = None,
    ) -> None:
        if ticks_per_day < 1:
            raise ValueError("ticks_per_day must be >= 1")
        if fence is not None and (state_home is None or Path(fence.home_dir).resolve() != Path(state_home).resolve()):
            raise ValueError("the loop's fence must be the fence of its state_home")
        self.open_session = open_session
        # RECOVERY HOOK (entity c75 incident 2026-07-18: a loop died on its
        # spawn-time mind while the operator's new mind stood unused in the
        # home). Returns True when the home's substrate now DIFFERS from the
        # mind the failing session was built on — the terminal failure cull
        # then HEALS (reset + continue to the next day-open, which rebuilds
        # on the operator's new mind) instead of ending the loop. None =
        # today's behavior (cull is always terminal). A healthy loop never
        # consults it — only the cull path does, so no mid-day mind swap.
        self.substrate_changed = substrate_changed
        self.tick_seconds = float(tick_seconds)
        self.ticks_per_day = int(ticks_per_day)
        self.max_ticks = max_ticks
        self.stop_file = Path(stop_file) if stop_file else None
        self.first_cue = first_cue
        # dm#89 day gate: the boundary decision (drive-cause trace rides
        # loop_status.day_cause) and the cue's offered drive ids (the
        # first-turn commit-exclusion half of the F2 contract).
        self._day_gate: Optional[Dict[str, Any]] = None
        # The job a work day opens on (the head of read_job_queue at its gate; CONTRACT v3 D1): the work order or a
        # task of the inbox. His ```work verdict settles THIS job, whatever was queued meanwhile.
        self._day_job: Optional[Dict[str, Any]] = None
        # THE LOOP'S FENCE (STOP-LANE-DESIGN §2.2): the lifecycle generation this loop was admitted at - given by its
        # host, else admitted when run() starts. Every state / phases.yaml / loop-status write of the loop carries it
        # and is refused once it moved; the turn's cancel check and every boundary read it; a moved (or unreadable)
        # generation ends the loop at its next check - the operator's Stop, never a failure.
        self._fence: Optional[LifecycleFence] = fence
        self._cue_drive_ids: List[str] = []
        # THE GATE'S INPUTS (contract-phases v2 §10.3-§10.5, spec v23):
        # - his own sleep decision, taken from a ```rest block at a day's close and consumed by the next gate;
        # - the last ACTIVITY phase he lived (personal / work / sleep / "visit"; None at a start): the from-side of
        #   the boundary's loop edges and the sleep->sleep refusal;
        # - the first day after the operator's start is his (the start IS the pull, spec personal entered_by).
        self._sleep_elected: Optional[str] = None
        self._last_activity_phase: Optional[str] = None
        self._first_day: bool = bool(first_day_unconditional)
        # The phase this run was started to (re)enter (spec v23): "personal" when the operator started or granted
        # own time (a direct run, the CLI's default) or the host restores it; "work" or None when the host started
        # it for work alone. A personal day sets it (continuation survives a sleep, a visit or a work day).
        self._return_to: Optional[str] = (
            PHASE_PERSONAL if first_day_unconditional else (return_to if return_to in RESUME_TO_PHASES else None)
        )
        # The first cue after his own sleep (composed by the sleep leg, read by the next day-open).
        self._wake_cue: str = ""
        # Marker honesty (F7): the phase_changed marker for a day-opening
        # transition writes when the day actually OPENS (after the operator
        # gate + belt), never at the gate — staged here by the boundary consult.
        self._pending_day_marker: Optional[Tuple[str, str, str, str]] = None
        # rest_minutes > 0 = 24/7 mode: an elected rest is HIS SLEEP (the gate
        # takes the decision, the maintenance pass runs, he wakes), and when
        # nothing pulls he rests this long between days inside personal.
        # 0 = supervised mode: rest ends the loop. The stop file remains the
        # operator's hard stop.
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
        # R-D circling window: the last ~8 tick replies (process-local ring;
        # _circling_note reads it at day-open).
        self._recent_replies: List[str] = []
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

    def _generation_moved(self) -> bool:
        """True once the operator's Stop moved the home's lifecycle generation past the one this loop was admitted at,
        or once it cannot be known (FAIL CLOSED, never "not fenced"). The hot path: an int compare and one stat (the
        turn's cancel check, every effect boundary of the loop's own bookkeeping)."""
        if self.state_home is None:
            return False
        return self._fence is None or self._fence.moved()

    def _stopped_by_operator(self) -> None:
        """The operator's Stop, seen at a check or as a refused fenced write: the loop ends at its next boundary.
        Said ONCE, and never as a failure (no #FALLBACK, no backoff, no retry)."""
        if self.stop_cause != "kill_switch":
            self.stop_cause = "kill_switch"
            self.out("(the operator's Stop moved the lifecycle generation - this loop ends here)")

    def _fenced_bookkeeping(self, what: str, write: Callable[[], Any]) -> bool:
        """One loop bookkeeping write (spend, personal usage, a task status, the work order) as ONE fenced step
        (Fable F8): inside the home state lock, after the generation is re-read there (`fence.commit`) - a Stop that
        arrives meanwhile waits for it, and a write decided before a Stop never lands after it. False when refused (the
        Stop, said once) or the generation is unreadable. Once per tick or per day close, never per effect."""
        if self.state_home is None:
            write()
            return True
        try:
            self._fence.commit(what, write)
            return True
        except StoppedByOperator:
            self._stopped_by_operator()
        except LifecycleUnreadable as e:
            self._generation_unreadable(e)
        return False

    def _generation_unreadable(self, error: BaseException) -> None:
        if self.stop_cause != "generation_unreadable":
            self.stop_cause = "generation_unreadable"
            self.out(f"(the lifecycle generation is unreadable ({error}) - this loop ends here and writes nothing more; "
                     f"repair {PHASES_FILENAME})")

    def _should_stop(self) -> bool:
        if self.stop_cause in ("kill_switch", "generation_unreadable"):
            return True  # sticky: a Stop seen anywhere (a refused write, the turn) ends the loop at this boundary
        if self.stop_file and self.stop_file.exists():
            self.stop_cause = "stop_file"
            return True
        if self.state_home is not None:
            # The boundary re-reads phases.yaml itself too (the parse the loop always paid here): a bump by a writer
            # that predates the signal file is seen, and an unreadable file ends the loop (fail closed).
            try:
                moved = self._fence is None or self._fence.truth_moved()
            except LifecycleUnreadable as e:
                self._generation_unreadable(e)
                return True
            if moved:
                self._stopped_by_operator()
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

    def _idle_while(
        self, predicate: Callable[[Dict[str, Any]], bool], *, phase: Optional[str] = None
    ) -> Dict[str, Any]:
        """Idle at a boundary while `predicate(state)` holds; the stop file
        always wins. Returns the state that ended the idle (or state=stop).

        `phase` heartbeats loop_status each poll (gateway adversary 2,
        2026-07-13: a mid-day PAUSE idled past LOOP_STATUS_STALE_SECONDS and
        the staleness belt declared a LIVE loop not-running — console lied,
        and the post-wake window could double-start). The heartbeat keeps
        updated_at honest through long freezes."""
        while True:
            if self._should_stop():
                return {"state": "stop"}
            if phase is not None:
                self._status(phase)
            state = self._operator_state()
            if not predicate(state):
                return state
            self.sleep_fn(STATE_POLL_SECONDS)

    def _sleep_window(
        self, reason: str, *, include_dream: bool = True, include_identity: bool = True,
        mark_state: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Enter a self-elected sleep: mark state=asleep (written_by=self, so
        the navbar and biography show HE chose it), then run consolidation/
        dreams if a hook is wired. Returns the dream result (or None).

        The consolidation pass is a HOME WRITER (plan item 1): it runs under
        the lease (holder="dream"). A held home skips the pass honestly —
        the night is quiet, the pass is idempotent and runs next sleep.

        KILL-SWITCH GUARD (laurent 16:12, c1530: paused IS the kill switch
        — "if a dream can run under paused today, that is now a bug"): a
        paused entity opens NO sleep window — and critically, the self-sleep
        state write below must never CLOBBER an operator's freeze (the
        no-entity-unset property; this write is entity-reachable through
        the rest election)."""
        if self.state_home is not None:
            current = read_entity_state(self.state_home)
            if current.get("state") == "paused":
                self.out("(paused by operator - no sleep window opens under a freeze)")
                return None
            if mark_state:
                try:
                    write_entity_state(
                        self.state_home, "asleep",
                        reason=f"self-elected sleep: {reason}", mode="dreaming", written_by="self",
                        fence=self._fence,
                    )
                except StoppedByOperator:
                    self._stopped_by_operator()
                    return None
                except Exception as e:  # noqa: BLE001 - sleep must not die over its own marker
                    self.out(f"#FALLBACK could not mark self-sleep state: {e}")
            if self._generation_moved():
                self._stopped_by_operator()  # no consolidation pass opens after the operator's Stop
                return None
        self.out("(sleeping - consolidating the day)")
        if self.on_sleep is None:
            self._clear_dreaming_badge(reason)
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
                self._clear_dreaming_badge(reason)
                return None
        try:
            # CYCLE-WINDOW COMPOSITION (memory c379: include_dream=False =
            # quality passes only, dreams keep their nightly-class cadence).
            # include_identity mirrors it exactly (cti#399 ask 2b, gate
            # ruled satisfied c4779): cycle windows NEVER touch the self —
            # a 2h maintenance nap must not enact identity; nightly sleeps
            # OFFER, and the regulated bars (memory's half) mean most
            # nights enact nothing. Hooks that predate either flag get the
            # narrower call - the flags are composition, never requirements.
            try:
                result = self.on_sleep(include_dream=include_dream, include_identity=include_identity)
            except TypeError:
                try:
                    result = self.on_sleep(include_dream=include_dream)
                    if include_identity:
                        self.out(
                            "#FALLBACK sleep hook has no include_identity; the identity "
                            "pass waits for the engine half (proposals hold in the graph)"
                        )
                except TypeError:
                    result = self.on_sleep()
                    # F5 (adversary): the plain rung loses BOTH flags — the
                    # identity loss is labeled here too, never silent.
                    if include_identity:
                        self.out(
                            "#FALLBACK sleep hook takes no flags; the identity pass "
                            "waits for the engine half (proposals hold in the graph)"
                        )
            if isinstance(result, dict) and result.get("stopped_by") == "kill_switch":
                self._stopped_by_operator()  # the night ended on the operator's Stop (said once by the night)
                return result
            if isinstance(result, dict) and result.get("formed"):
                self.out("(a dream formed - candidate connections for waking evidence)")
            else:
                self.out("(a quiet night - nothing new surfaced)")
            return result
        except TurnCancelled:
            self._stopped_by_operator()  # a fenced write of the night refused: the operator's Stop, never a failure
            return None
        except Exception as e:  # noqa: BLE001 - a failed dream never breaks the loop
            self.out(f"#FALLBACK consolidation pass failed ({e}); sleeping without a dream")
            return None
        finally:
            if dream_lease is not None:
                dream_lease.release()
            # mode=dreaming clears the moment the pass ends (entity c2465
            # ask 3, answered YES): the badge claimed present-tense
            # consolidation for the whole nap while the pass finishes in
            # seconds — the rest of the nap is honest "resting".
            self._clear_dreaming_badge(reason)

    def _clear_dreaming_badge(self, reason: str) -> None:
        """Rewrite mode dreaming -> resting once the consolidation pass has
        ended (or never ran), guarded to OUR OWN self-sleep only — an
        operator state written during the pass stands untouched."""
        if self.state_home is None:
            return
        try:
            current = read_entity_state(self.state_home)
            if (
                current.get("state") == "asleep"
                and str(current.get("written_by")) == "self"
                and str(current.get("mode") or "") == "dreaming"
            ):
                # No deadline is stamped (v22: no clock ends a sleep; he wakes when its maintenance is done).
                write_entity_state(
                    self.state_home, "asleep",
                    reason=f"self-elected sleep: {reason}", mode="resting", written_by="self",
                    fence=self._fence,
                )
        except StoppedByOperator:
            self._stopped_by_operator()
        except Exception as e:  # noqa: BLE001 - a badge fix must never break the nap
            self.out(f"#FALLBACK could not clear the dreaming badge: {e}")

    def _circling_note(self) -> str:
        """R-D's second signal (agent A1, contract frozen c2702/c2704): when
        the recent tick replies are a circling run, the fresh-day cue names
        the exit ramp — encourage-never-force, no work vocabulary
        (semantics' G2 law), mirroring R-B's contract words ("go look").

        SOFT IMPORT by design: circling_streak lives in abstractagent,
        which imports abstractruntime — a module-scope import here would be
        a dependency cycle, and the detector is an optional enhancement
        (absent package = no varied cue, silently; the freedom baseline
        stands). The reply buffer is process-local like turn_n — a resumed
        loop starts a fresh window, which only delays detection by a few
        ticks, never fabricates one."""
        if len(self._recent_replies) < 3:
            return ""
        try:
            from abstractagent.adapters.progress import circling_streak
        except ImportError:
            return ""
        try:
            streak = circling_streak(list(self._recent_replies))
        except Exception:  # noqa: BLE001 - a cue accent never kills a day-open
            return ""
        if not isinstance(streak, dict):
            return ""
        repeats = int(streak.get("repeats") or 0)
        if repeats < 2:
            return ""
        # STALENESS + ATTRACTOR guards (adversary F5, 2026-07-17): clear the
        # ring when the note fires — otherwise sub-min_words replies are
        # transparent to the detector and a day of short acks re-fires the
        # note on YESTERDAY'S streak; and a reply echoing the note itself
        # would re-enter the ring and feed the next detection (the note
        # seeding its own attractor). Phrasing is shape-neutral ("circled
        # the same ground"): the detector deliberately catches A-B-A-B
        # oscillation too, where "the same thought N times" is arithmetic
        # fiction.
        self._recent_replies.clear()
        return (
            " One quiet observation: your last stretch circled the same "
            "ground a few times - sitting with it more, a different "
            "question, or going to look at something outside it are all "
            "equally yours."
        )

    def _own_sleep(self, reason: str, from_phase: Optional[str], report: "LifeReport") -> bool:
        """HIS OWN SLEEP, taken by the gate (spec personal/work->sleep#self_elected; contract-phases v2 §10.4): the
        state is written (written_by self, mode dreaming, his reason), the transition is marked, the maintenance pass
        runs, and he wakes into what the gate allows at the next boundary. Returns True when a stop arrived (the
        caller ends the loop). Composes `self._wake_cue`."""
        report.sleeps += 1
        if self.state_home is not None:
            current = read_entity_state(self.state_home)
            if current.get("state") == "paused":
                self.out("(paused by operator - no sleep window opens under a freeze)")
                self._wake_cue = ""
                return False
            try:
                write_entity_state(
                    self.state_home, "asleep",
                    reason=f"self-elected sleep: {reason}", mode="dreaming", written_by="self",
                    fence=self._fence,
                )
                # F7c: the marker rides the SUCCESSFUL state write.
                if from_phase in (PHASE_PERSONAL, PHASE_WORK):
                    append_phase_changed(
                        self.state_home, from_phase=str(from_phase), to=PHASE_SLEEP,
                        cause="self_elected", written_by="self", fence=self._fence,
                    )
            except StoppedByOperator:
                self._stopped_by_operator()
                report.stopped_by = "kill_switch"
                return True
            except Exception as e:  # noqa: BLE001 - sleep must not die over its own marker
                self.out(f"#FALLBACK could not mark self-sleep state: {e}")
        self._last_activity_phase = PHASE_SLEEP
        dreamed = self._sleep_window(reason, mark_state=False)
        # HE WAKES WHEN THE MAINTENANCE IS DONE (contract-phases v2 §10.6; spec sleep->X#maintenance_done): no
        # fixed nap after the pass. A visitor who knocked during it took the posture: the pass yielded at its next
        # stage, the wake below leaves the door's posture alone, and the door's close hands the same sleep back.
        self._wake_from_self_sleep()
        self._wake_cue = self._compose_wake_cue(dreamed)
        report.dreams += 1 if (isinstance(dreamed, dict) and dreamed.get("formed")) else 0
        if self._should_stop():
            report.stopped_by = self.stop_cause or "stop_file"
            return True
        return False

    def _compose_wake_cue(self, dreamed: Any) -> str:
        dream_note = ""
        if isinstance(dreamed, dict) and dreamed.get("formed"):
            dream_note = " while you slept a dream formed (candidate connections await your waking evidence)"
            from .night_voice import wake_residue

            dream_note += wake_residue(list(dreamed.get("signals") or []))
        # P2-4 second wire: the night's grouping/mining work reaches
        # the entity (it used to vanish into the report dict).
        if isinstance(dreamed, dict) and int(dreamed.get("maintenance_candidates") or 0) > 0:
            n_cand = int(dreamed["maintenance_candidates"])
            dream_note += (
                f" the night set out {n_cand} candidate(s) from your standing pile"
                " - search_memory finds them when you want to look"
            )
        # R-D: offer back his own standing state (open questions)
        # with the reread command, and name a circling run when the
        # last stretch was one — directions, never orders.
        circling = self._circling_note()
        # NO ECHO OF THE REST REASON (operator 2026-11-08): the
        # reason stays in his records (report.rest_reason, the
        # sleep state's reason, own_time.log) and never opens the
        # next day - nerin rested on 14 of 28 days and every day
        # after a rest opened on his own reason to stop, then
        # rested again (8 of 8; sleep-investigation-2026-11-08).
        return f"{_today_stamp()}you rested and your own time resumes.{dream_note}{circling}"

    def _resume_own_sleep(self, state: Dict[str, Any], report: "LifeReport") -> bool:
        """A sleep of his own found at the operator gate: a visit's close handed back the sleep it interrupted
        (contract-phases v2 §4.4: the SAME sleep continues - not a new one, no sleep->sleep). Its maintenance pass
        resumes where it yielded (every engine phase is idempotent), then he wakes into what the gate allows.
        Returns True when a stop arrived."""
        writer = str(state.get("written_by") or "")
        reason = str(state.get("reason") or "")
        if reason.startswith("self-elected sleep: "):
            reason = reason[len("self-elected sleep: "):]
        self.out(f"(his sleep continues after the visit ({writer}) - its maintenance resumes)")
        self._last_activity_phase = PHASE_SLEEP
        dreamed = self._sleep_window(reason or "the interrupted sleep", mark_state=False)
        self._wake_from_self_sleep()
        self._wake_cue = self._compose_wake_cue(dreamed)
        report.dreams += 1 if (isinstance(dreamed, dict) and dreamed.get("formed")) else 0
        if self._should_stop():
            report.stopped_by = self.stop_cause or "stop_file"
            return True
        return False

    def _rest_between_days(self, seconds: float) -> bool:
        """He rests inside personal when nothing pulls (awake, no summon, zero LLM). Returns True on a stop; returns
        early (False) when the boundary has something new to read: a state change (a visit, a sleep, a pause), a
        queued job, or the operator's grant revoked."""
        remaining = float(seconds)
        while remaining > 0:
            if self._should_stop():
                return True
            self._status("between")
            if self.state_home is not None:
                st = self._operator_state()
                if str(st.get("state") or "") != "awake" or str(st.get("mode") or ""):
                    return False
                if jobs_queued(self.state_home):
                    return False
                if personal_grant_refusal(read_personal_grant(self.state_home)) is not None:
                    return False
            chunk = min(STATE_POLL_SECONDS, remaining)
            self.sleep_fn(chunk)
            remaining -= chunk
        return self._should_stop()

    def _wake_from_self_sleep(self) -> None:
        """Return to awake after a self-elected sleep — but only if the state
        is still the self-asleep we wrote. If an operator changed it during
        the nap (sleep/pause), we leave their intent alone; the top-of-loop
        gate honors it."""
        if self.state_home is None:
            return
        try:
            current = read_entity_state(self.state_home)
            # v12 P1-3: BOTH self-writers clear their own sleeps - "self"
            # (elected rest) and "personal-cycle" (the maintenance window).
            # Operator/door writers stay untouched (their intent stands).
            writer = str(current.get("written_by"))
            if current.get("state") == "asleep" and writer in ("self", "personal-cycle"):
                write_entity_state(
                    self.state_home, "awake",
                    fence=self._fence,
                    reason=(
                        "woke from self-elected sleep - own time resumes"
                        if writer == "self"
                        else "maintenance cycle complete - personal time resumes"
                    ),
                    written_by=writer,
                )
        except StoppedByOperator:
            self._stopped_by_operator()
        except Exception as e:  # noqa: BLE001
            self.out(f"#FALLBACK could not clear self-sleep state: {e}")

    def _open_work_job(self) -> None:
        """A work day opens on its job (CONTRACT v3 D1/D4). A task it opens is marked `taken`. When the job interrupts
        his own time - granted, and the phase this run returns to - the home remembers it: resume_to = personal, so
        the end of the work (in this loop or a later one) returns him there while the grant still stands."""
        if self.state_home is None:
            return
        if self._generation_moved():
            self._stopped_by_operator()  # the task stays as the Stop found it
            return
        job = self._day_job or {}
        if job.get("kind") == "task" and job.get("status") == "pending":
            try:
                self._fenced_bookkeeping("marking the task taken", lambda: append_task_status(
                    self.state_home, str(job["id"]), "taken",
                    by=_manifest_entity_id(self.state_home) or "own-time-loop"))
            except Exception as e:  # noqa: BLE001 - the day still works its task
                self.out(f"#FALLBACK could not mark task {job.get('id')} taken: {e}")
        if (
            self._return_to == PHASE_PERSONAL
            and personal_grant_refusal(read_personal_grant(self.state_home)) is None
            and read_resume_to(self.state_home).get("phase") != PHASE_PERSONAL
        ):
            try:
                write_resume_to(self.state_home, PHASE_PERSONAL, written_by="loop:job-interrupts-own-time",
                                fence=self._fence)
                self.out("(a job interrupts his own time - he returns to it when every job is done, if still granted)")
            except StoppedByOperator:
                self._stopped_by_operator()
            except Exception as e:  # noqa: BLE001 - the in-process return still holds for this loop
                self.out(f"#FALLBACK could not remember the return to his own time: {e}")

    def _settle_work_job(self, verdict: str) -> None:
        """His ```work verdict settles the day's job: the order archives (as ever); a task is marked done (done:) or
        parked (blocked:), the verdict line as its note. Nothing is settled after the operator's Stop (the job stays
        as the Stop found it)."""
        if self._generation_moved():
            self._stopped_by_operator()
            return
        job = self._day_job or {"kind": "work_order"}
        if job.get("kind") != "task":
            self._fenced_bookkeeping("archiving the work order", lambda: archive_work_order(self.state_home,
                                                                                         verdict=verdict))
            return
        status = "parked" if verdict.strip().lower().startswith("blocked:") else "done"
        try:
            self._fenced_bookkeeping(f"settling the task ({status})", lambda: append_task_status(
                self.state_home, str(job["id"]), status,
                by=_manifest_entity_id(self.state_home) or "own-time-loop", note=verdict))
        except Exception as e:  # noqa: BLE001 - loud; the next gate re-reads the queue
            self.out(f"#FALLBACK could not settle task {job.get('id')} ({status}): {e}")

    def _cancel_remembered_return(self, by: str) -> None:
        """Own time ended without an interruption: the remembered return (resume_to) is cancelled - in this process
        and in the home's bucket when the runtime holds one (a home still on the gateway sidecar is the gateway's to
        clear, contract §4.1)."""
        self._return_to = None
        if self.state_home is None:
            return
        try:
            current = read_resume_to(self.state_home)
            if current.get("source") == "phases.yaml" and current.get("phase") is not None:
                write_resume_to(self.state_home, None, written_by=by, fence=self._fence)
        except StoppedByOperator:
            self._stopped_by_operator()
        except Exception as e:  # noqa: BLE001 - the exit itself stands
            self.out(f"#FALLBACK could not cancel the remembered return: {e}")

    def _open_day_session(self) -> ChatSession:
        """Open the day's session in the phase the GATE decided (work or personal). A factory that takes `phase`
        (production: build_session_factory) is told; a zero-argument factory (harnesses) opens its own."""
        import inspect

        phase = str((self._day_gate or {}).get("phase") or "")
        try:
            params = inspect.signature(self.open_session).parameters
        except (TypeError, ValueError):
            params = {}
        if "phase" in params and phase in (PHASE_WORK, PHASE_PERSONAL):
            if "job" in params and phase == PHASE_WORK and self._day_job is not None:
                return self.open_session(phase=phase, job=self._day_job)
            return self.open_session(phase=phase)
        return self.open_session()

    def _wake_own_leftover_sleep(self, reason: str) -> bool:
        """Write awake over a sleep the loop wrote for itself (LOOP_OWN_SLEEP_WRITERS) and nothing else: an
        operator sleep, a visit posture and the kill switch stand. True when it woke one."""
        if self.state_home is None:
            return False
        current = read_entity_state(self.state_home)
        if (
            str(current.get("state") or "") == "asleep"
            and str(current.get("mode") or "") != "visiting"
            and str(current.get("written_by") or "") in LOOP_OWN_SLEEP_WRITERS
        ):
            write_entity_state(self.state_home, "awake", reason=reason, written_by="loop", fence=self._fence)
            return True
        return False

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

    def _status(self, phase: str, *, stopped_by: Optional[str] = None,
                day_kind: Optional[str] = None,
                day_cause: Optional[Dict[str, Any]] = None,
                tunables: Optional[Dict[str, Any]] = None) -> None:
        if self.state_home is not None:
            try:
                write_loop_status(self.state_home, phase, stopped_by=stopped_by,
                                  day_kind=day_kind, day_cause=day_cause,
                                  tunables=tunables, fence=self._fence)
            except StoppedByOperator:
                self._stopped_by_operator()  # the Stop's own `stopped` record stands
            except LifecycleUnreadable as e:
                self._generation_unreadable(e)

    def _write_loop_spend(self, session: ChatSession, day_ticks: int) -> None:
        """Persist cumulative loop spend = the lifetime base (loaded at day
        open) + this session's live counters. Best-effort loop bookkeeping
        (like loop_status): single writer, no lease, never breaks a day."""
        if self.state_home is None:
            return
        if self._generation_moved():
            return  # nothing is counted after the operator's Stop (an abandoned call is no spend)

        def _write() -> None:
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

        try:
            self._fenced_bookkeeping("writing the loop spend", _write)
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
        if self.state_home is not None:
            # ADMISSION, BEFORE ANY WRITE (STOP-LANE-DESIGN §2.2, I7): the loop is admitted at the home's generation
            # (its host's fence, else the current one). Unreadable = the loop does not start; a fence that already
            # moved = it stopped before it started. Either way it writes nothing.
            if self._fence is None:
                try:
                    self._fence = LifecycleFence.admit(self.state_home)
                except LifecycleUnreadable as e:
                    self.out(f"(the lifecycle generation is unreadable ({e}) - this loop does not start and writes "
                             f"nothing; repair {PHASES_FILENAME})")
                    report.stopped_by = "generation_unreadable"
                    return report
            if self._fence.moved():
                self.out("(stopped before it started: the operator's Stop moved the lifecycle generation - this loop "
                         "writes nothing)")
                report.stopped_by = "kill_switch"
                return report
        # NOTHING IS RESTORED AFTER A HARD STOP (CONTRACT v3 D3): a loop that follows the operator's kill switch
        # never returns to the own time the stopped loop was living, whatever its host passed - it works the queued
        # jobs and ends idle. Only the operator's own act (a grant or start: first_day_unconditional) opens own time.
        if self.state_home is not None and not self._first_day and self._return_to is not None:
            _prior = read_loop_status(self.state_home)
            if str(_prior.get("stopped_by") or _prior.get("after_stopped_by") or "") == "kill_switch":
                self.out("(the last loop was stopped by the operator's kill switch - nothing is restored; own time "
                         "comes back only with the operator's grant)")
                self._return_to = None
        # THE BOOT CLAIM (STOP-LANE-DESIGN §2.4; Fable F7): the loop's first write replaces the spawn reservation with
        # its own "between" - one locked read-check-write, fenced at the admitted generation. After a Stop it is
        # refused and the loop ends here; over another live loop's record it refuses too. Either way nothing is
        # written (the wake below runs only after it).
        if self.state_home is not None:
            try:
                _claim_loop_status(self.state_home, fence=self._fence)
            except StoppedByOperator:
                self._stopped_by_operator()
            except LifecycleUnreadable as e:
                self._generation_unreadable(e)
            except LoopAlreadyRunning as e:
                self.out(f"({e} - one loop per home; this loop writes nothing)")
                report.stopped_by = "already_running"
                return report
        if self.stop_cause in ("kill_switch", "generation_unreadable"):
            if self.stop_cause == "kill_switch":
                self.out("(stopped before it started: the operator's Stop moved the lifecycle generation - this "
                         "loop writes nothing)")
            report.stopped_by = self.stop_cause
            return report
        # SPAWN-WAKE (contract-phases v2 §1.2 rule 6, §10.1): a sleep an earlier loop wrote for itself (its own
        # election, the retired cycle, the retired gate/exit landings) has no process behind it any more; the new
        # loop wakes it. Operator sleeps and visit postures keep their own wake rules.
        if self.state_home is not None:
            try:
                self._wake_own_leftover_sleep("a new loop arrived - the sleep an earlier loop left wakes")
            except StoppedByOperator:
                self._stopped_by_operator()
            except Exception as e:  # noqa: BLE001 - never block a spawn
                self.out(f"#FALLBACK spawn-wake check failed: {e}")
        while True:
            if self.max_ticks is not None and report.ticks >= self.max_ticks:
                report.stopped_by = "max_ticks"
                break
            if self._should_stop():
                report.stopped_by = self.stop_cause or "stop_file"
                break

            # THE DAY GATE (contract-phases v2 §10.3-§10.5; spec v23; read_day_gate's order): visit > his own sleep
            # decision (never after a sleep) > a standing work order > personal while the operator's grant stands
            # (a day when something pulls, the first day after his start, else rest between days) > idle (own time
            # ends awake). The previous ACTIVITY phase is the from-side of this boundary's loop edges: the self-loops
            # personal->personal#personal_cycle and work->work#task_complete, work<->personal, sleep->X#
            # maintenance_done. A boundary after a start or a visit's close is no loop edge (the operator's or the
            # door's act). STOP always wins; paused stays the kill switch (the operator state gate below).
            _prev_phase = self._last_activity_phase
            self._day_gate = None
            self._day_job = None
            self._pending_day_marker = None
            # A HELD STATE FIRST (Codex review 2026-11-09 #1): asleep (his own sleep handed back by a visit's close, an
            # operator sleep), the visiting posture or paused - the state gate below owns it, and no leg is chosen
            # over it (the personal rest used to return at once over a sleep and the loop spun without resuming it).
            _held = self.state_home is not None and str(self._operator_state().get("state") or "") in ("asleep", "paused")
            if self.state_home is None and self._sleep_elected is not None:
                # A harness loop (no home state): his sleep decision still runs its window (never after a sleep).
                _elected = self._sleep_elected
                self._sleep_elected = None
                if _prev_phase != PHASE_SLEEP:
                    if self._own_sleep(_elected, _prev_phase, report):
                        break
                    cue = self._wake_cue
                    continue
            if self.state_home is not None and not _held:
                _elected = self._sleep_elected
                self._sleep_elected = None  # consumed at this boundary, honored or refused
                decision = read_day_gate(
                    self.state_home, sleep_elected=_elected, last_phase=_prev_phase, first_day=self._first_day,
                    return_to=self._return_to,
                )
                if decision.get("note"):
                    self.out(f"({decision['note']})")
                _edge_cause = None
                _edge_cue = ""
                if decision.get("phase") in (PHASE_WORK, PHASE_PERSONAL) and not decision.get("rest"):
                    _edge_cause = _boundary_cause(_prev_phase, str(decision["phase"]), self.state_home)
                    if _edge_cause:
                        decision, _edge_cue = consult_gate_landing(
                            self.state_home, str(_prev_phase), decision, _edge_cause, self.out,
                            last_phase=_prev_phase, first_day=self._first_day, return_to=self._return_to,
                        )
                        _edge_cause = _boundary_cause(_prev_phase, str(decision.get("phase") or ""), self.state_home)
                if _edge_cue and _edge_cue not in (cue or ""):
                    cue = f"{(cue or '').strip()} {_edge_cue}".strip()
                # MARKER STAGING (F7 honesty cluster): a day-opening transition's phase_changed writes when the day
                # actually OPENS (after the operator gate + belt). A self-loop is no phase moment (contract §5:
                # from == to), and a start or a visit's close is not the loop's edge.
                if (
                    decision.get("phase") in (PHASE_WORK, PHASE_PERSONAL)
                    and not decision.get("rest")
                    and _edge_cause
                    and _prev_phase != decision.get("phase")
                ):
                    self._pending_day_marker = (
                        str(_prev_phase), str(decision["phase"]), str(_edge_cause),
                        "blueprint redirect" if decision.get("redirected_from") else "structural",
                    )
                if decision["phase"] == "visit":
                    # VISIT-PREEMPTS-THE-GATE: fall through to the operator
                    # state gate below, which idles on the visiting posture
                    # until the door's close writes the restore.
                    self.out("(a visit is open - the day gate yields)")
                elif decision["phase"] == PHASE_SLEEP:
                    # HIS OWN SLEEP DECISION (spec personal/work->sleep#self_elected): the maintenance pass runs
                    # under his own sleep; he wakes into what the gate allows at the next boundary.
                    if self._own_sleep(str(decision.get("reason") or "no reason kept"), _prev_phase, report):
                        break
                    cue = self._wake_cue
                    continue
                elif decision["phase"] == "idle":
                    # NOTHING ALLOWED: own time ends AWAKE (contract-phases v2 §10.1; spec v23: only the operator's
                    # grant or a queued job starts it again). The exit wakes any sleep of the loop's own. Own time
                    # ending here is no interruption: a remembered return is cancelled (CONTRACT v3 D4; contract §4.1
                    # "cleared when own time ends without an interruption") - he is idle, and a later job from idle
                    # returns to idle.
                    report.stopped_by = {"no_grant": "personal_off", "no_return": "no_return",
                                         "jobs_done": "jobs_done"}.get(
                        str(decision.get("cause") or ""), str(decision.get("cause") or "personal_off")
                    )
                    self.out(
                        f"(own time ends: {decision.get('grant_cause') or decision.get('cause')}: "
                        f"{decision.get('detail') or ''} - he stays awake and reachable)"
                    )
                    self._cancel_remembered_return(f"loop:{report.stopped_by}")
                    break
                elif decision.get("rest"):
                    # NOTHING PULLS: he rests between days INSIDE personal (awake; zero LLM). Supervised mode ends
                    # the run (the supervisor owns what comes next); 24/7 rests `rest_minutes`, returning early for
                    # a stop, a state change (visit, sleep, pause), a queued job or a revoked grant.
                    self.out(f"(personal: {decision['detail']})")
                    if self.rest_minutes <= 0:
                        report.stopped_by = "settled_desk"
                        break
                    self._status("between", day_cause=decision)
                    if self._rest_between_days(self.rest_minutes * 60.0):
                        report.stopped_by = self.stop_cause or "stop_file"
                        break
                    continue
                else:
                    self._day_gate = decision
                    self._day_job = decision.get("job") if decision.get("phase") == PHASE_WORK else None

            # Operator state gate before a day opens (a2a 0008): asleep or
            # paused idles here (dreams may run gateway-side while asleep -
            # the no-summon window is enforced by this very gate). Waking is
            # HONEST: the first cue names what happened and for how long.
            gate = self._operator_state()
            if _is_own_sleep(gate):
                # A sleep of his own handed back by a visit's close (the door restores the interrupted sleep,
                # contract-phases v2 §4.4) or left by an earlier loop: the loop owns it.
                if self._resume_own_sleep(gate, report):
                    break
                cue = self._wake_cue
                continue
            if gate["state"] in ("asleep", "paused"):
                self.out(f"({gate['state']} by operator{self._since(gate)} - idling)")
                # No clock ends a sleep (v22: the sleep bound and the 6 h need-check are retired): an operator sleep
                # lasts until the operator's wake or its maintenance's end (the gateway writes awake); a visit until
                # the door's close; paused until resume.
                woke = self._idle_while(
                    lambda s: s["state"] == "paused" or (s["state"] == "asleep" and not _is_own_sleep(s)),
                    phase="between",
                )
                if woke["state"] == "stop":
                    report.stopped_by = self.stop_cause or "stop_file"
                    break
                if str(gate.get("mode") or "") == "visiting" or visitor_waiting(gate):
                    # The visit interrupted the leg it found; the door's close returns him to it (contract §4.4).
                    # The boundary after it is the door's edge, never the loop's (no self-loop consult, no marker).
                    self._last_activity_phase = "visit"
                elif gate["state"] == "asleep":
                    self._last_activity_phase = PHASE_SLEEP
                if _is_own_sleep(woke):
                    # The visit's close handed back the sleep it interrupted (contract §4.4): it continues.
                    if self._resume_own_sleep(woke, report):
                        break
                    cue = self._wake_cue
                    continue
                # Wake-cue seeding (R3, 2026-07-09): the awake state's reason
                # now carries the VISIT'S FACTS (gateway writes "visitor
                # session ended (person:laurent; 12 turns) - you elected to
                # pursue: ..."). Reading it into the first cue is what lets a
                # commitment made in a visit actually reach his own time —
                # the generic cue sent him straight back to old attractors.
                woke_reason = str(woke.get("reason") or "").strip()
                reason_note = f" What just happened: {woke_reason}." if woke_reason else ""
                note = f" your last note to yourself: {cue}" if cue else ""
                # Wake facts only (P0-1: the offer composes at day-open
                # now, ONE site) - the date stays here so the wake moment
                # itself is grounded.
                cue = (
                    f"{_today_stamp()}you were {gate['state']}{self._since(gate)} (operator-initiated) "
                    f"and are awake again - a new stretch of your own time begins."
                    f"{reason_note}{note}"
                )
                self.out("(awake again - a new day can open)")
                # Back to the TOP gate before any summon (phase-machine
                # audit G1, 2026-07-13): idles can last hours — a grant
                # revoked/expired DURING a visit or operator sleep must be
                # seen at this wake, not after a full unmandated day. The
                # cue survives the continue; the top re-checks stop + grant
                # and reads the now-awake state through the normal gate.
                continue

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
            # THE OFFER COMPOSES AT EVERY DAY-OPEN (pathway adversary P0-1,
            # 2026-07-20): it used to compose only on WAKE transitions — a
            # drive-loaded entity (total_open > 0) chains personal days
            # back-to-back and NEVER crosses a wake, so the whole dm#89 cue
            # build was dead code on the dominant path (5/84, 0/61 stayed
            # frozen by construction). ONE composer site now: every day
            # opens with today's date + the standing state + the alive
            # drive offers around whatever cue text carried (first cue,
            # next: line, wake facts). _cue_drive_ids resets HERE every
            # day (P2-1: no stale exclusions on chained days).
            if self.state_home is not None:
                standing = standing_state_note(self.state_home)
                offered: List[Dict[str, Any]] = []
                drives_note, drive_ids = drives_cue_note(self.state_home, offered=offered)
                if offered:
                    # F2 watch (operator 2026-11-08): the desk now names
                    # each drive's gist, so the day-open log records which
                    # drives were offered and their stored strength - a
                    # later check compares these across days.
                    self.out(
                        "(desk offered at day-open: "
                        + "; ".join(
                            f"{o['record_id']} strength="
                            + (
                                f"{o['strength']:.4f}" if o["strength"] is not None
                                else f"unread ({o['strength_error'] or 'no score'})"
                            )
                            for o in offered
                        )
                        + ")"
                    )
                # P2-2: BOTH note lanes' offered ids share one exclusion.
                self._cue_drive_ids = drive_ids + standing_state_offered_ids()
                # The maintenance signals feed his own sleep decision (contract-phases v2 §10.4): facts, offered.
                maint_note = maintenance_signals_note(self.state_home)
                base = (cue or "").strip()
                if not base.lstrip().startswith("today is"):
                    cue = f"{_today_stamp()}{base}{standing}{drives_note}{maint_note}"
                else:
                    # A wake path already stamped the date; append offers
                    # only if the wake text does not already carry them.
                    if "Alive on your desk" not in base:
                        cue = f"{base}{standing if 'You hold' not in base else ''}{drives_note}{maint_note}"
            # The day PHASE begins at the summon window (adversary find,
            # 2026-07-13): the doors' quiescence negotiation watches phase,
            # and a summon (open + possible salvage LLM call) that still
            # read "between" would let a visit pass the negotiation and
            # collide with the held lease instead — a user-visible 409 the
            # state protocol exists to prevent.
            # The staged phase_changed writes HERE (F7a/b): the day is truly
            # opening — operator gate, belt, and lease all passed — so the
            # biography records transitions that HAPPENED, never intents.
            if self._pending_day_marker is not None and self.state_home is not None:
                _pm_from, _pm_to, _pm_cause, _pm_prov = self._pending_day_marker
                try:
                    append_phase_changed(
                        self.state_home, from_phase=_pm_from, to=_pm_to,
                        cause=_pm_cause, written_by="day-gate", provenance=_pm_prov, fence=self._fence,
                    )
                except StoppedByOperator:
                    self._stopped_by_operator()
            self._pending_day_marker = None
            # The day OPENS here: it becomes the next boundary's from-side.
            if isinstance(self._day_gate, dict) and self._day_gate.get("phase"):
                self._last_activity_phase = str(self._day_gate["phase"])
                if self._last_activity_phase == PHASE_PERSONAL:
                    self._return_to = PHASE_PERSONAL
                elif self._last_activity_phase == PHASE_WORK:
                    self._open_work_job()
            self._first_day = False
            self._status("day", day_cause=self._day_gate)
            if self.stop_cause == "kill_switch" or self._generation_moved():
                # THE OPERATOR'S STOP between the gate and the summon: no day opens (no session, no salvage call).
                self._stopped_by_operator()
                if summon_lease is not None:
                    summon_lease.release()
                report.stopped_by = "kill_switch"
                break
            opened = False
            try:
                session = self._open_day_session()
                # A VISITOR AT THE DOOR stops new work inside the tick
                # (operator 2026-11-09: reachable within a minute or two): the
                # session checks this before running the tools a reply elects.
                session.yield_check = lambda: visitor_waiting(self._operator_state())
                # THE OPERATOR'S STOP cancels the turn at its next effect boundary (Codex review 2 #2; STOP-LANE-DESIGN
                # §2.2): the session carries the loop's fence - no further model attempt, tool, effect, formation or
                # authoring in this process once it moved; a reply that returns after it is abandoned.
                session.fence = self._fence
                # F2 commit-exclusion half: drive ids the cue names must not
                # be strengthened BY the cue turn (a day-open mention is the
                # composer's act, not his use). First turn only; a later
                # genuine reach commits normally.
                if self._cue_drive_ids and hasattr(session, "commit_exclusions"):
                    session.commit_exclusions = set(self._cue_drive_ids)
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
            _day_started_monotonic = time.monotonic()
            try:
                tick_slots = 0
                while tick_slots < self.ticks_per_day:
                    # Heartbeat (pid-reuse guard): a LIVE day re-stamps its
                    # status every boundary so readers can tell it from a
                    # corpse whose pid got recycled (LOOP_STATUS_STALE_SECONDS).
                    # day_kind rides it (gateway wave-1: the served phase
                    # fold needs work-vs-personal without approximating).
                    self._status("day", day_kind=getattr(session, "phase", None),
                                 day_cause=self._day_gate)
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
                        resumed = self._idle_while(
                            lambda s: s["state"] == "paused", phase="day"
                        )
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

                    # TICK-BOUNDARY GRANT CHECK (both lane audits flagged the
                    # day-open-only lag independently, c1499 G9 + gateway
                    # c1501 (c)): a revocation/expiry mid-day now ends the
                    # day at the NEXT TICK, not the day boundary —
                    # ticks_per_day is operator-configurable, so the old lag
                    # was unbounded in configuration while consent is the
                    # one thing that must not wait. The day closes with its
                    # normal ceremony; the TOP gate then does the ruled
                    # sleep-landing write and the exit (one mechanism, one
                    # write site).
                    if self.state_home is not None and getattr(session, "phase", "") != PHASE_WORK:
                        mid_refusal = personal_grant_refusal(read_personal_grant(self.state_home))
                        if mid_refusal is not None:
                            self.out(f"(personal ended mid-day: {mid_refusal} - the day closes)")
                            break

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
                            # THE TICK'S LOOK-BACK (operator 2026-11-08): at
                            # the end of every personal-time tick, one model
                            # call looks back over what the tick formed and
                            # offers feelings, interests, lessons, questions
                            # and problems through the close's parsers. It is
                            # a home writer, so it runs inside this tick's
                            # lease window. Skipped on the tick that elects
                            # rest: the day's close runs the look-back over
                            # the whole day right after. Its failure never
                            # fails the tick (the turn already happened).
                            # Skipped too when a visitor is at the door: the
                            # day's look-back is deferred to the next open
                            # (B1 fast-yield) and covers this tick.
                            if self._generation_moved():
                                raise StoppedByOperator("after the tick's turn")
                            if (
                                getattr(session, "phase", "") == PHASE_PERSONAL
                                and parse_rest_block(reply)[1] is None
                                and not visitor_waiting(self._operator_state())
                            ):
                                try:
                                    session.tick_lookback()
                                except TurnCancelled:
                                    raise  # the operator's Stop: handled below, never a look-back failure
                                except Exception as lookback_err:  # noqa: BLE001
                                    self.out(
                                        f"#FALLBACK the tick's look-back failed ({lookback_err}); "
                                        "the tick stands"
                                    )
                            # THE AUTHORING DRAINER at the tick boundary (operator 2026-11-09): pending titles and
                            # summaries, bounded by the profile's cap_per_turn, inside this tick's writer window;
                            # a visitor at the door goes first.
                            if not visitor_waiting(self._operator_state()) and callable(
                                getattr(session, "drain_authoring", None)
                            ):
                                session.drain_authoring(window="turn")
                        finally:
                            if tick_lease is not None:
                                tick_lease.release()  # the tick's writer window ends HERE
                    except TurnCancelled as cancelled:
                        # THE OPERATOR'S STOP (Codex review 2 #2): the turn stopped at an effect boundary and formed
                        # nothing; no failure, no backoff - the loop ends here (said once).
                        if self.stop_cause != "kill_switch":
                            self.out(f"(the turn was cancelled {cancelled.where} - the operator's Stop; this loop ends "
                                     "here)")
                        self.stop_cause = "kill_switch"
                        report.stopped_by = "kill_switch"
                        break
                    except (RuntimeError, Exception) as e:  # noqa: BLE001
                        report.failures += 1
                        consecutive_failures += 1
                        self.out(f"#FALLBACK tick failed ({e}); backing off {FAILURE_BACKOFF_SECONDS:g}s")
                        if consecutive_failures >= MAX_CONSECUTIVE_TICK_FAILURES:
                            # SUBSTRATE-HEAL (entity c75): before the terminal
                            # cull, ask whether the operator has provided a
                            # NEW mind since this session was built. If so,
                            # the failure is the OLD mind going away with a
                            # remedy already in the home — do not die; reset
                            # and fall to the next day-open, which rebuilds on
                            # the operator's current substrate. Recovery only
                            # (this path is the failure gate); a healthy day
                            # never reaches here, so no mid-day mind swap.
                            healed = False
                            if self.substrate_changed is not None:
                                try:
                                    healed = bool(self.substrate_changed())
                                except Exception as e:  # noqa: BLE001
                                    self.out(f"#FALLBACK substrate-heal check failed ({e})")
                                    healed = False
                            if healed:
                                self.out(
                                    f"(day {day} recovers: {consecutive_failures} failures on the "
                                    "old mind, but the operator changed this home's substrate - "
                                    "healing onto the new mind at the next day-open)"
                                )
                                consecutive_failures = 0
                                report.stopped_by = None
                                # A brief backoff so the next day-open does not
                                # spin against a new mind still coming up; a
                                # stop during it is honored (heal never traps).
                                if self._interruptible_sleep(FAILURE_BACKOFF_SECONDS):
                                    report.stopped_by = self.stop_cause or "stop_file"
                                break  # end the day; the loop's next day-open re-resolves
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
                    # R-D circling window (agent A1): keep the last 8 marked
                    # replies for the day-open detector. Driver markers
                    # ([kept a private diary entry], [felt: ...], [used
                    # tool: ...], failure notices) are stripped BEFORE the
                    # ring (adversary F6: agent's _prose_view regex knows a
                    # subset of our marker vocabulary — markers surviving as
                    # "prose" both feed similarity and defeat its min_words
                    # abstention; the ring is a similarity buffer, never a
                    # record, so bracket-stripping loses nothing durable).
                    self._recent_replies.append(
                        re.sub(r"\[[^\]\n]{0,200}\]", " ", marked)
                    )
                    del self._recent_replies[:-8]
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

                    # WORK VERDICT (the work lane, laurent seq 155): a work
                    # day ends when the entity declares done/blocked — the
                    # order archives visibly and the next day-open reads no
                    # order (personal returns). Only work days parse this.
                    if getattr(session, "phase", "") == "work":
                        verdict = parse_work_verdict(marked)
                        if verdict is not None:
                            # LifeLoop has no home_dir attribute — the home
                            # is state_home (adversary P1-1, 2026-07-20: the
                            # original spelling raised AttributeError on the
                            # FIRST real work verdict, outside the tick
                            # try/except — loop death + a crash loop on
                            # restart over the un-archived order). Leaseless
                            # test homes (state_home=None) skip the archive.
                            if self.state_home is not None:
                                self._settle_work_job(verdict)
                            report.stopped_by = "work_done"
                            self.out(f"(work verdict: {verdict[:120]})")
                            break

                    cue = parse_next_cue(marked) or NEUTRAL_CUE
                    # No pause between ticks while a visitor waits: the
                    # boundary check right after yields to him.
                    if (
                        self.tick_seconds > 0
                        and not visitor_waiting(self._operator_state())
                        and self._interruptible_sleep(self.tick_seconds)
                    ):
                        report.stopped_by = self.stop_cause or "stop_file"
                        break
            finally:
                # laurent #54: the 2h personal-use floor meters LIVED day
                # time - recorded at close, keyed to the current grant.
                # Best-effort; conservative (a killed process undercounts,
                # which grants MORE personal time, never less).
                _gate_personal = (self._day_gate or {}).get("phase") == PHASE_PERSONAL
                _session_personal = getattr(session, "phase", "") == PHASE_PERSONAL
                _stopped = self.stop_cause == "kill_switch" or self._generation_moved()
                if self.state_home is not None and (_gate_personal or _session_personal) and not _stopped:
                    # The GATE's decision is the loop's own truth for what
                    # kind of day this was (a harness factory may not stamp
                    # session.phase; production stamps both).
                    _day_elapsed = time.monotonic() - _day_started_monotonic
                    self._fenced_bookkeeping("recording personal usage",
                                             lambda: record_personal_usage(self.state_home, _day_elapsed))
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
                # reflection LLM call, which keeps the waiting visitor's wait
                # short.
                #
                # Normal closes (rest/max_ticks/failures/operator-sleep)
                # reflect inline, under their own lease window: the
                # reflection is a home writer too (APPRAISE + summary FORM).
                # A held lease waits honestly (lease-free poll); a stop
                # during the wait defers to the same salvage marker.
                # A visitor who knocked during the day's LAST tick (or a
                # rest/verdict tick) never met the tick-boundary check: the
                # close yields fast for him too.
                visit_yield = visit_yield or visitor_waiting(self._operator_state())
                try:
                    if _stopped:
                        # THE OPERATOR'S STOP: the day closes without its look-back (a model call and home writes).
                        self._stopped_by_operator()
                        if session.reports:
                            self.out("(the operator's Stop - the day closes without its look-back)")
                    elif visit_yield:
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

            # (v22 retires the v11 timed personal<->sleep cycle: no clock opens a sleep window.)
            if report.stopped_by == "rest" and self.rest_minutes > 0 and visitor_waiting(self._operator_state()):
                # He chose rest as a visitor knocked: the visit comes first.
                # Sleeping now would write asleep over the door's visiting
                # posture; the top gate idles on the visit instead.
                self.out("(rest elected as a visitor knocked - the visit comes first)")
                report.stopped_by = "max_ticks"  # the loop's continue-marker
                report.rest_reason = None
                continue
            if report.stopped_by == "rest" and self.rest_minutes > 0:
                # 24/7 mode: his rest is HIS SLEEP DECISION (spec personal/work->sleep#self_elected). The gate at
                # the top of the loop takes it (or refuses it: never after a sleep, never from own time the
                # operator has revoked) - one decision point for every leg.
                self._sleep_elected = report.rest_reason or "no reason kept"
                report.stopped_by = "max_ticks"  # the loop's continue-marker
                report.rest_reason = None
                continue
            if report.stopped_by == "work_done" and self.rest_minutes > 0:
                # The order is archived: the gate chooses what follows - another order (work->work),
                # own time while the operator's grant stands (work->personal#no_task), else idle (the loop ends,
                # he stays awake). Supervised runs end here as before.
                report.stopped_by = "max_ticks"
                continue
            if report.stopped_by in ("rest", "stop_file", "stop_command", "failures", "kill_switch",
                                     "generation_unreadable"):
                break
            if report.stopped_by == "work_done":
                # A finished work order ends the LOOP RUN cleanly (adversary
                # P1-1 follow-through: without this the break fell into the
                # next day-open — in supervised mode that exhausted the
                # session source and died as failure-cull backoffs). In 24/7
                # mode the operator's supervisor (or the next spawn) opens
                # the next day; the order is already archived.
                break
            if self.max_ticks is not None and report.ticks >= self.max_ticks:
                report.stopped_by = "max_ticks"
                break
        # The status file names WHY (failure-death visibility): a loop that
        # culled itself on consecutive failures must not look like a clean
        # stop to /loop status readers.
        if self.stop_cause in ("kill_switch", "generation_unreadable"):
            # The Stop's own record stands (hard_stop_loop / the gateway wrote stopped by kill_switch): this loop
            # writes nothing more - no status, no wake.
            report.stopped_by = self.stop_cause
            return report
        self._status("stopped", stopped_by=report.stopped_by)
        # OWN TIME ENDS AWAKE (contract-phases v2 §10.1; v22 retires v8 IDLE-IS-SLEEP): an ending loop never writes
        # a sleep and never leaves one of its own behind - he is awake and reachable (idle) after it. An operator
        # sleep, a visit posture and the kill switch stand.
        if self.state_home is not None and report.stopped_by:
            try:
                self._wake_own_leftover_sleep(f"own time ended ({report.stopped_by}) - awake and reachable")
            except StoppedByOperator:
                self._stopped_by_operator()
            except Exception as e:  # noqa: BLE001 - the exit itself must stand
                self.out(f"#FALLBACK could not clear the loop's own sleep at exit: {e}")
        return report


def build_consolidator(
    home_dir: Path,
    *,
    embedding_model: Optional[str] = None,
    embedding_base_url: str = "http://127.0.0.1:1234/v1",
    narrator_llm_factory: Optional[Callable[[], Any]] = None,
    out: Callable[[str], None] = print,
    llm_for_capability: Optional[Callable[[str], Any]] = None,
    authoring_cap: int = 64,
    fence: LifecycleFence,
) -> Callable[[], Optional[Dict[str, Any]]]:
    """The sleep-window pass (maintainer rulings 2026-07-08/09: sleep is
    where the graph is worked on — tending first, then the dream).

    Opens the home and runs the engine's `sleep_pass` (phase-1 maintenance
    tending, then `dream_pass`, the fork's canonical order encoded
    engine-side) over the entity's self/diary/life scopes; closes. Safe in
    the loop's own process: during own-time the loop is the sole writer
    (one life, one summon). A quiet night in either phase is a valid night.

    GRACEFUL YIELD (one-active-phase ruling, laurent 13:28 / memory c1462):
    the night passes the engine a `should_continue` predicate — a pure read
    of the home's state + STOP file — so a transition arriving MID-NIGHT
    (visit auto-yield write, operator wake, stop) ends the night at the
    engine's next phase boundary: the in-flight phase completes its writes
    (never torn), later phases skip with a named reason, and the next sleep
    resumes where this one stopped (every phase idempotent). The predicate
    must stay SIDE-EFFECT-FREE: stop COMMANDS are consumed-on-read and are
    deliberately not consulted here — the boundary check after the night
    consumes them, exactly once.

    Returns the LOOP-FACING shape (the on_sleep contract): `formed` (did a
    dream form), `dream_record_id`, `maintenance_candidates` (phase-1
    consolidation candidates awaiting waking review), and the full engine
    report under `engine`. The translation is deliberate: the engine's
    dream half says `created`, and reading it wrong here is exactly how a
    formed dream once reported as "a quiet night" (found 2026-07-09 — a
    hand-written test double carried `formed` while the engine never did).

    THE OPERATOR'S STOP (Codex review 3 #2; STOP-LANE-DESIGN §2.2, class H): `fence` is the loop's. A moved
    generation ends the night at the engine's next stage (`should_continue`), stops the completion marker
    (`sleep_pass(cancelled=fence.moved)`: a stopped night records no completed maintenance), the repair step's
    authoring (`should_stop`), the topics step (`cancelled=fence.moved`: no version, no note) and the night voice;
    nothing is logged after it but one line.

    THE NIGHT'S ORDER: memory's `sleep_pass` (resolution, consolidation, world models, mining, identity, dream) -> its
    authoring-failure lines -> METADATA REPAIR (`_sleep_authoring`) -> TOPICS (`topics.sleep_communities`: embed what
    has no vector, then memory's `communities_pass`; its receipt rides the result as `communities`) -> the night
    voice."""
    def _night_should_continue() -> bool:
        # The night keeps its window while the entity still SLEEPS by its
        # own election and nobody pulled the manual brake. Any other state
        # (awake/paused, a visit's asleep+visiting rides mode not state...
        # the visit door writes state=asleep too, so mode decides) ends it.
        if fence.moved():
            return False  # the operator's Stop (an int compare and a stat, no file parse)
        if (Path(home_dir) / "STOP").exists():
            return False
        state = read_entity_state(home_dir)
        if str(state.get("mode") or "") == "visiting":
            return False  # a visitor is at the door — finish the phase, yield
        if state.get("state") != "asleep":
            return False
        # (v22: no stamped deadline ends a night - the v11 cycle window is retired; the pass runs to its end
        # unless a visitor, a wake, a pause or the STOP brake ends it at the engine's next stage.)
        return True

    def _consolidate(
        *, include_dream: bool = True, include_identity: bool = True
    ) -> Optional[Dict[str, Any]]:
        from .chat import open_home

        embedder = None
        if embedding_model and embedding_model.lower() not in ("", "none", "off"):
            try:
                from abstractmemory import OpenAICompatTextEmbedder

                embedder = OpenAICompatTextEmbedder(base_url=embedding_base_url, model=embedding_model)
            except Exception as e:  # noqa: BLE001
                out(f"#FALLBACK dream embeddings unavailable ({e}); vectorless bridges only")

        home = open_home(home_dir, embedder=embedder, fence=fence)
        try:
            eid = home.entity_id
            scopes = [("self", eid), ("diary", eid), ("life", eid)]
            # THE NIGHT'S RECORDS ARE AUTHORED WITH THEM (AbstractMemory c46b4cd; Codex review 2 #10): cards, dreams,
            # candidates and consolidation summaries get their title and summary from the summaries capability as
            # they form; a failure forms them pending and its line is written from the pass's receipt below.
            from .authoring import capability_formation_author, log_pass_authoring_failures

            author_seen: Dict[str, Any] = {}
            formation_author = capability_formation_author(llm_for_capability, home.artifacts, author_seen)
            # ChatHome exposes the MemorySystem facade as `.ms` (not `.memory`).
            try:
                try:
                    from abstractmemory import sleep_pass
                except ImportError:
                    # Version skew (older engine without phase-1 tending):
                    # dream-only night, labeled — never a blocked sleep.
                    from abstractmemory import dream_pass

                    out("#FALLBACK abstractmemory has no sleep_pass (older engine); dream-only night")
                    dream = dream_pass(home.ms, scopes=scopes, owner_id=eid, author=formation_author)
                    engine: Dict[str, Any] = {"maintenance": None, "dream": dream, "authoring": dream.get("authoring")}
                else:
                    # Tunables stay engine-declared defaults (max_candidates=2,
                    # scan_limit=200) — inject from the function, never copy numbers.
                    try:
                        engine = sleep_pass(
                            home.ms, scopes=scopes, owner_id=eid,
                            should_continue=_night_should_continue,
                            include_dream=include_dream,
                            include_identity=include_identity,
                            author=formation_author,
                            cancelled=fence.moved,
                        )
                    except TypeError:
                        # Version skew ladder: engines predate include_identity
                        # (the identity-pass spine, cti#399/c4779 — memory's
                        # half lands in the same wave), include_dream (memory
                        # c379), and/or should_continue (c1462). Degrade one
                        # step at a time, labeled — never a blocked sleep; the
                        # proposal queue HOLDS in the graph, nothing is lost.
                        try:
                            engine = sleep_pass(
                                home.ms, scopes=scopes, owner_id=eid,
                                should_continue=_night_should_continue,
                                include_dream=include_dream,
                                author=formation_author,
                            )
                            if include_identity:
                                out("#FALLBACK engine sleep_pass has no include_identity; realizations hold for the engine half")
                        except TypeError:
                            try:
                                engine = sleep_pass(
                                    home.ms, scopes=scopes, owner_id=eid,
                                    should_continue=_night_should_continue,
                                    author=formation_author,
                                )
                                if not include_dream:
                                    out("#FALLBACK engine sleep_pass has no include_dream; full night ran on a cycle window")
                            except TypeError:
                                out("#FALLBACK engine sleep_pass has no should_continue; full night runs")
                                engine = sleep_pass(home.ms, scopes=scopes, owner_id=eid)
            except TurnCancelled as stopped:
                # THE OPERATOR'S STOP inside the engine: memory's commit guard refused a write of the night
                # (rolled back - nothing of it landed, no completion recorded).
                engine = {"maintenance": {}, "dream": {}, "cancelled_after": f"the operator's Stop ({stopped.where})"}
            if fence.moved():
                # THE OPERATOR'S STOP: the night ends here - no repair, no voice, no failure lines (a refusal the
                # Stop caused is no failure); memory recorded no completion (cancelled=).
                out("(the operator's Stop - the night ends here; nothing more is written)")
                dream_seen = engine.get("dream") or {}
                return {"authoring": {"skipped": "the operator's Stop"}, "formed": bool(dream_seen.get("created")),
                        "dream_record_id": dream_seen.get("dream_record_id"), "maintenance_candidates": 0,
                        "signals": [], "narration": None, "engine": engine, "stopped_by": "kill_switch"}
            if engine.get("cancelled_after"):
                out(f"(the night ended early - host transition after {engine['cancelled_after']}; "
                    "the next sleep resumes there)")
            log_pass_authoring_failures(engine, home_dir=home_dir, entity=eid, reader=author_seen.get("reader"),
                                        out=out)
            # METADATA REPAIR, one step among the night's work (operator 2026-11-09: "the sleep phase is just to
            # repair any issue that may still be there"): a record whose formation-time authoring failed (pending)
            # or a legacy record without a title or summary (declared pending here) is authored, bounded by the
            # pass's cap; skipped when the night was ended (a visitor, a wake). Never a cut.
            authoring = (
                _sleep_authoring(home, eid, scopes, llm_for_capability, authoring_cap, out, should_stop=fence.moved)
                if _night_should_continue() else {"skipped": "the night ended before the repair step"}
            )
            # TOPICS, after the repair and before the night's closing voice (operator 2026-11-10: topics "must run
            # server side (runtime) ... in particular during sleep/maintenance phase ... so that it becomes part of
            # the knowledge of the entity"): memory embeds what has no vector, then forms the night's version of his
            # communities, titled by the same author as every record of the night, within what the repair step left
            # of the night's authoring budget. Fenced; a stopped night records nothing for it (identity/topics.py).
            from .topics import sleep_communities

            completed = None if (engine.get("cancelled_after") or engine.get("maintenance") is None) \
                else ("night" if include_dream else "cycle")
            communities = (
                sleep_communities(home.ms, entity_id=eid, scopes=scopes, author=formation_author, fence=fence,
                                  budget=max(0, int(authoring_cap) - int(authoring.get("considered") or 0)),
                                  maintenance_kind=completed, out=out)
                if _night_should_continue() else {"skipped": "the night ended before the topics step"}
            )
            if fence.moved():
                # THE OPERATOR'S STOP during the repair or the topics step: the night ends here, as above.
                out("(the operator's Stop - the night ends here; nothing more is written)")
                dream_seen = engine.get("dream") or {}
                return {"authoring": authoring, "communities": communities,
                        "formed": bool(dream_seen.get("created")), "dream_record_id": dream_seen.get("dream_record_id"),
                        "maintenance_candidates": 0, "signals": [], "narration": None, "engine": engine,
                        "stopped_by": "kill_switch"}
            log_pass_authoring_failures(communities, home_dir=home_dir, entity=eid, reader=author_seen.get("reader"),
                                        out=out)
            dream = engine.get("dream") or {}
            maintenance = engine.get("maintenance") or {}
            # WAVE-5 NARRATOR HALF (dm#75 ratified; frozen shape c3708):
            # signals ride the formed dream; the night voice fires only on
            # the ruled trigger set + the >=20h throttle, ONE witnessed
            # call max. Zero LLM on quiet/triggerless/throttled nights —
            # the engine default stays mechanical.
            signals = list((dream.get("attributes") or {}).get("signals") or []) \
                if isinstance(dream.get("attributes"), dict) else []
            if not signals and dream.get("signals"):
                signals = list(dream.get("signals") or [])
            narration: Optional[Dict[str, Any]] = None
            if dream.get("created") and signals and narrator_llm_factory is not None and not _night_should_continue():
                # A visitor (or a wake, a pause, the brake) arrived during the pass: the night voice is a model
                # call - it yields too (contract-phases v2 §10.8); the dream stands, unnarrated.
                out("(the night voice yields - the night ended at its last stage)")
            elif dream.get("created") and signals and narrator_llm_factory is not None:
                try:
                    from .night_voice import run_night_voice
                    from .prelude import render_summon_prelude

                    prelude = render_summon_prelude(
                        home.ms, home.diary, entity_id=eid, budget=1600, spark=home.spark,
                    )
                    narration = run_night_voice(
                        home_dir,
                        llm=narrator_llm_factory(),
                        prelude_text=str(prelude.get("text") or ""),
                        signals=signals,
                        dream_record_id=str(dream.get("dream_record_id") or ""),
                        salience=int(dream.get("salience") or 0),
                        out=out,
                    )
                    if narration.get("narrated"):
                        out(f"(night voice spoke - trigger: {narration.get('trigger')})")
                except Exception as e:  # noqa: BLE001 - the voice never breaks a night
                    out(f"#FALLBACK night voice errored ({e}); the night stays mechanical")
            return {
                "authoring": authoring,
                "communities": communities,
                "formed": bool(dream.get("created")),
                "dream_record_id": dream.get("dream_record_id"),
                "maintenance_candidates": int(maintenance.get("created_count") or 0),
                "signals": signals,
                "narration": narration,
                "engine": engine,
            }
        finally:
            try:
                home.close()
            except Exception:  # noqa: BLE001
                pass

    return _consolidate


def _sleep_authoring(home: Any, eid: str, scopes: List[Tuple[str, str]],
                     llm_for_capability: Optional[Callable[[str], Any]], cap: int,
                     out: Callable[[str], None], *, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """The sleep's metadata work (operator 2026-11-09): declare what lacks a title or summary, then author what is
    pending. Returns {declared, refused, rearmed, authored, deferred, failed, considered (the authoring tasks the
    drainer took: what the night's budget spent)} for the pass's receipt; never raises.
    `should_stop` (the loop's fence) is checked before each step and handed to the drainer, which checks it per task
    and after each capability answer: nothing is declared, re-armed or authored after the operator's Stop."""
    from .authoring import declare_missing_metadata, drain_authoring, rearm_exhausted_authoring

    receipt: Dict[str, Any] = {"declared": 0, "refused": 0, "rearmed": 0, "authored": 0, "deferred": 0, "failed": 0,
                               "considered": 0}
    try:
        if should_stop():
            receipt["cancelled"] = "the operator's Stop"
            return receipt
        declared = declare_missing_metadata(home.ms, scopes=scopes, out=out)
        receipt["declared"] = int(declared.get("declared") or 0)
        receipt["refused"] = int(declared.get("refused") or 0)
        # The marked records whose authoring tasks all ended get fresh ones (memory's re-arm), in this home's scopes;
        # a declaration memory answered by repairing a missing task counts there too.
        rearmed = rearm_exhausted_authoring(home.ms, scopes=scopes, out=out)
        receipt["rearmed"] = int(rearmed.get("rearmed") or 0) + int(declared.get("repaired") or 0)
        receipt["refused"] += int(rearmed.get("refused") or 0)
        if should_stop():
            receipt["cancelled"] = "the operator's Stop"
            return receipt

        def _diary_text(entry_id: str) -> Optional[str]:
            entry = home.diary.get_entry(entry_id) if entry_id else None
            return str((entry or {}).get("text") or "") or None

        drained = drain_authoring(home.ms, llm_for_capability=llm_for_capability, artifacts=home.artifacts,
                                  home_dir=home.home_dir, entity=eid, holder="sleep", cap=int(cap), window="nap",
                                  diary_text=_diary_text, out=out, should_stop=should_stop)
        for key in ("authored", "deferred", "failed", "considered"):  # considered: the night's authoring budget spent
            receipt[key] = int(drained.get(key) or 0)
    except Exception as e:  # noqa: BLE001 - the night's other work still runs
        receipt["error"] = f"{type(e).__name__}: {e}"
        out(f"#FALLBACK the sleep's authoring did not run ({receipt['error']})")
    return receipt


def build_capability_llm_factory(
    home_dir: Path,
    *,
    base_url: Optional[str],
    fence: LifecycleFence,
    out: Callable[[str], None] = print,
) -> Callable[[str], Any]:
    """`llm_for_capability(name)` in the loop process, built like the day client (the parked understanding-drain
    design, 2026-11-08): each call resolves the cognition profile (`resolve_capability(home, name)`: the profile's
    own model, else the mind) and builds the client with the entity patience window and `base_url` for the
    lmstudio / openai-compatible families. An endpoint:* capability resolves through the ambient endpoint resolver
    `main` installs from the spawn's capability profiles. Nothing resolvable raises (`SubstrateUnset`): the caller
    defers, never a silent fallback."""
    from .cognition_profile import resolve_capability

    def _llm_for_capability(capability: str) -> Any:
        from abstractcore import create_llm

        res = resolve_capability(home_dir, capability)
        provider = str(res["provider"]).strip().lower()
        kwargs: Dict[str, Any] = {
            "model": res["model"],
            **dict(zip(("timeout", "retry_wall_clock_budget_s"), _entity_patience_window())),
            "read_idle_timeout_s": ENTITY_LLM_READ_IDLE_TIMEOUT_S,
        }
        if base_url and provider in ("lmstudio", "openai-compatible", "openai_compatible"):
            kwargs["base_url"] = base_url
        try:
            llm = create_llm(provider, **kwargs)
        except TypeError:
            kwargs.pop("retry_wall_clock_budget_s", None)
            kwargs.pop("read_idle_timeout_s", None)
            out("#FALLBACK core predates retry_wall_clock_budget_s; timeout-only guard on the capability client")
            llm = create_llm(provider, **kwargs)
        try:
            llm.model_info = {"provider": provider, "model": res["model"], "thinking": res.get("thinking")}
        except Exception:  # noqa: BLE001 - the reader label is read with a fallback in authoring._reader_of
            pass
        return _beat_before_each_call(llm, home_dir, fence=fence)

    return _llm_for_capability


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
    thinking: Optional[str] = None,
    out: Callable[[str], None] = print,
    llm_for_capability: Optional[Callable[[str], Any]] = None,
    fence: LifecycleFence,
) -> Callable[[], ChatSession]:
    """Production factory: one fresh summon (home + LLM + session) per day. `fence` is the loop's (the substrate
    stamp and the per-call heartbeat are fenced writes).

    The entity is the only participant — its own time is self-directed; the
    workspace is enabled (creation is the point); the own-time contract is
    appended to the session's system prompt.
    """
    from datetime import datetime, timezone

    from .chat import open_home

    # The spawn-time substrate — the baseline a day-open compares against so
    # an operator change (substrate.yaml PUT, marker-first, operator-owned)
    # is HONORED at the next summon and made VISIBLE, never silent drift.
    _spawn_substrate = (provider, model)

    def _resolve_current_substrate() -> Tuple[str, str, Optional[str]]:
        """Re-resolve the home's substrate at day-open (each fresh summon =
        the visit lane's per-open resolution, applied to the loop). A change
        since spawn heals a loop whose mind went away and honors the
        operator's deliberate choice; the change is logged loudly. Resolution
        failure falls back to the spawn substrate (never a phase-less day).
        The reasoning effort re-resolves the same way, so an operator change
        to the stored dial reaches the next day without a loop restart."""
        try:
            from .substrate import resolve_home_substrate

            cur_p, cur_m, cur_t = resolve_home_substrate(None, None, home_dir=home_dir)
            cur_p = str(cur_p or "").strip().lower()
            cur_m = str(cur_m or "").strip()
            if (cur_p, cur_m) != _spawn_substrate and cur_p and cur_m:
                out(
                    f"(mind re-resolved at day-open: {_spawn_substrate[0]}/{_spawn_substrate[1]} "
                    f"-> {cur_p}/{cur_m} - the operator changed this home's substrate)"
                )
            if (cur_t or None) != (thinking or None):
                out(
                    f"(reasoning effort re-resolved at day-open: {thinking or '(unset)'} "
                    f"-> {cur_t or '(unset)'})"
                )
            return (cur_p or provider, cur_m or model, cur_t)
        except Exception as e:  # noqa: BLE001 - a resolution hiccup never phase-less a day
            out(f"#FALLBACK substrate re-resolution failed ({e}); this day keeps the spawn mind")
            return (provider, model, thinking)

    def _resolve_current_shelf_size() -> Optional[int]:
        """Re-read the home's recall width (`cognition.yaml`
        recall.shelf_size) at day-open, so a change reaches the next day
        without a loop restart (operator ruling 2026-11-08: nerin's profile
        said 75, the visits honoured it, the loop kept its spawn-time 50).
        The spawn value is the default when the profile names none; a
        harness factory with no width keeps the session's own default."""
        if shelf_size is None:
            return None
        from .cognition_profile import recall_shelf_size

        day_shelf = recall_shelf_size(home_dir, int(shelf_size))
        if day_shelf != int(shelf_size):
            out(
                f"(recall width re-read at day-open: {int(shelf_size)} -> {day_shelf} "
                "memories per turn, from cognition.yaml)"
            )
        return day_shelf

    def _factory(phase: Optional[str] = None, job: Optional[Dict[str, Any]] = None) -> ChatSession:
        day_provider, day_model, day_thinking = _resolve_current_substrate()
        day_shelf_size = _resolve_current_shelf_size()
        # Record the mind this day runs on so an observer's staleness cue
        # compares the operator's change against what is ACTUALLY in use
        # (entity c78), not just pid_started_at (which a heal/day-open swap
        # leaves unchanged). Best-effort; never blocks the summon.
        try:
            record_loop_substrate(home_dir, day_provider, day_model, fence=fence)
        except Exception:  # noqa: BLE001
            pass
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
        # days of continuous life. v16: the number is the BLUEPRINT's
        # window_limit dial (seed 8192) - one home one horizon, laurent-
        # modulated; the constant is gone.
        from .phase_spec import load_phase_tunables as _load_tunables

        _tun, _ = _load_tunables(home_dir=home_dir)
        home = open_home(
            home_dir, embedder=embedder,
            attention_window=int(_tun["window_limit"]),
            fence=fence,
        )

        from abstractcore import create_llm

        # Unattended loop: a hung request must become a failed tick (the
        # loop's backoff handles it), never an indefinite stall (observed
        # live: an SSL read with no timeout froze the first 24/7 attempt).
        # Output tokens: OMIT by default — core's registry upgrades an
        # unset value to the model's true ceiling; an explicit 2048 pinned
        # the resident below it (agency-caps ruling, 2026-07-11: a long
        # report turn must not be cut mid-thought by a caller's own pin).
        # PATIENCE WINDOW (core c3954, the 30-min wedge post-mortem): the
        # config default_timeout (600s) x 3 retries stacked to 30m01s on a
        # wedged substrate while the operator watched "Thinking...", so
        # entity lanes NAME their window. Since 2026-11-07 it is at least
        # 1800s per attempt (three 120s cuts in a row on a slow but healthy
        # remote model stopped the own-time loop), and the wall-clock retry
        # budget is never shorter than one attempt. Older cores without the
        # budget kwarg get the timeout alone (labeled).
        kwargs: dict[str, Any] = {
            "model": day_model,
            **dict(zip(("timeout", "retry_wall_clock_budget_s"), _entity_patience_window())),
            # READ-IDLE (0152 face 2, core c5051): stream-only in core, so
            # inert on these non-streaming calls; it carries the same 1800s
            # floor as the total. Older cores ignore the unknown kwarg via
            # the same TypeError ladder below.
            "read_idle_timeout_s": ENTITY_LLM_READ_IDLE_TIMEOUT_S,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if day_provider in ("lmstudio", "openai-compatible", "openai_compatible"):
            kwargs["base_url"] = base_url
        try:
            llm = create_llm(day_provider, **kwargs)
        except TypeError:
            kwargs.pop("retry_wall_clock_budget_s", None)
            out("#FALLBACK core predates retry_wall_clock_budget_s; timeout-only guard")
            llm = create_llm(day_provider, **kwargs)

        # Heartbeat per LLM call, not per tick (see touch_loop_heartbeat).
        llm = _beat_before_each_call(llm, home_dir, fence=fence)

        # THE WORK LANE (laurent seq 155): a standing work order shifts THIS
        # day to phase=work — the work grant applies (incl. execute_command
        # where the operator's matrix says so) and the contract is a
        # mission, honestly. No order = his own time, exactly as before.
        # The day's phase is the GATE's decision (the loop passes it; a blueprint redirect too); without one (a
        # direct call) the job queue decides. A work day works ONE job (CONTRACT v3 D1): the one the loop's gate
        # chose (`job`), else the head of the queue - the work order, or a task of the inbox.
        if phase in (PHASE_WORK, PHASE_PERSONAL):
            day_phase = phase
        else:
            day_phase = PHASE_WORK if jobs_queued(home_dir) else PHASE_PERSONAL
        if day_phase == PHASE_WORK and job is None:
            _queued = read_job_queue(home_dir)["jobs"]
            job = _queued[0] if _queued else None

        session = ChatSession(
            home,
            llm,
            participants=[home.entity_id],  # self-directed: his own time
            session_id=f"{session_prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
            context_window=context_window,
            shelf_size=day_shelf_size,
            enable_tools=True,
            enable_workspace=True,
            phase=day_phase,
            thinking=day_thinking,
            model_info=(
                {"provider": day_provider, "model": day_model, "thinking": day_thinking}
                if day_thinking else {"provider": day_provider, "model": day_model}
            ),
            out=out,
            fence=fence,
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
            own_time_text=(
                work_day_text(job)
                if job is not None and day_phase == PHASE_WORK
                else (session.prompt_overlay.get("personal") or OWN_TIME_CONTRACT)
            ),
            # The memory-teaching layer rides the re-compose too (ChatSession
            # read it at construction; the re-compose must not drop it).
            capability_map=getattr(session, "capability_map", ""),
        )
        # The capability clients (the summaries model authors titles and summaries; identity/authoring.py).
        session.llm_for_capability = llm_for_capability
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
    parser.add_argument("--thinking", default=None,
                        help="reasoning effort at loop START only (none/minimal/low/medium/high/xhigh). "
                             "Each day re-reads <home>/substrate.yaml, and the file outvotes this flag - "
                             "to change the dial durably, write the thinking field there "
                             "(the gateway's substrate endpoint does this)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--embedding-model", default="text-embedding-qwen3-embedding-0.6b")
    parser.add_argument(
        "--embedding-base-url", default="http://127.0.0.1:1234/v1",
        help="embeddings endpoint (stays local even when the mind runs remote)",
    )
    parser.add_argument(
        "--context-window", type=int, default=51200,
        help="declared context window. Default ~50k (operator 2026-08-01: "
        "'it is acceptable to go to 200k context, but ideally, let's have a "
        "(soft) recommended target of 50k tokens'); any explicit value wins "
        "in either direction — both bounds are soft, labeled warnings only, "
        "never a refusal",
    )
    parser.add_argument(
        "--shelf-size", type=int, default=36,
        help="recall shelf seats (maintainer 2026-07-09: widened to 36 — 'it "
        "needs to retrieve more memories to function'; the token budget "
        "still seats 36 rich digests at the default window)",
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
    # Personal-phase arming from the terminal (wake != grant != start: each
    # grant flag is its OWN act that writes and EXITS — arming and starting
    # in one command would blur two operator acts the 12:44 ruling keeps
    # separate). The terminal operator IS the principal on this lane (shell
    # access to the home = ownership); granted_by records the OS user.
    # Honest limit: CLI grants land in phases.yaml (granted_by/granted_at =
    # the audit record) but write no host marker — the gateway's arming
    # surface is the marker-first lane; timelines show CLI grants only
    # through the file's fields.
    parser.add_argument(
        "--grant-personal", action="store_true",
        help="operator act: arm personal time until revoked, then exit "
        "(the loop still starts as its own separate act)",
    )
    parser.add_argument(
        "--grant-personal-hours", type=float, default=None, metavar="H",
        help="operator act: arm personal time on a timer expiring H hours from now, then exit",
    )
    parser.add_argument(
        "--revoke-personal", action="store_true",
        help="operator act: disarm personal time (mode=disabled), then exit - "
        "a running loop ends at its next day boundary",
    )
    parser.add_argument("--state-reason", default="", help="reason recorded with --set-state")
    parser.add_argument(
        "--return-to", choices=("personal", "work", "none"), default="personal",
        help="the phase this run was started to (re)enter (spec v23 own time is the operator's privilege): personal "
        "for the operator's own start (the default of a hand run); a host restoring work alone passes work, a host "
        "that only knows the home's resume_to passes none",
    )
    parser.add_argument(
        "--first-day-unconditional", action="store_true",
        help="the operator started own time (his grant, start or retry): the first day is his whatever pulls "
        "(spec personal entered_by; contract-phases v2 §10). Without it the gate governs the first boundary too",
    )
    parser.add_argument(
        "--generation", type=int, default=None,
        help="internal (spawn_loop_process passes this): the lifecycle generation the start was reserved at; the "
        "loop refuses to start - writing nothing - when the operator's Stop moved it since",
    )
    parser.add_argument(
        "--skip-command-fast-forward", action="store_true",
        help="internal (spawn_loop_process passes this): the starter already "
        "fast-forwarded the command inbox at the start-request moment; doing it "
        "again here would eat a stop sent in the spawn->boot window",
    )
    args = parser.parse_args(argv)

    home_dir = Path(args.home).expanduser().resolve()

    if args.set_state:
        # The operator's act, admitted now (a terminal act decides and writes in one step).
        try:
            act_fence = LifecycleFence.admit(home_dir)
        except LifecycleUnreadable as e:
            print(str(e))
            return 2
        payload = write_entity_state(home_dir, args.set_state, reason=args.state_reason, fence=act_fence)
        print(f"state -> {payload['state']} (at {payload['changed_at']})"
              + (f" reason: {payload['reason']}" if payload["reason"] else ""))
        return 0

    grant_flags = [bool(args.grant_personal), args.grant_personal_hours is not None,
                   bool(args.revoke_personal)]
    if sum(grant_flags) > 1:
        print("choose ONE of --grant-personal / --grant-personal-hours / --revoke-personal "
              "- each is a distinct operator act")
        return 2
    if any(grant_flags):
        import getpass
        import math
        from datetime import datetime, timedelta, timezone

        principal = f"person:{getpass.getuser()}"
        try:
            act_fence = LifecycleFence.admit(home_dir)  # the operator's act, admitted now
            if args.revoke_personal:
                grant = write_personal_grant(home_dir, mode="disabled", fence=act_fence)
            elif args.grant_personal_hours is not None:
                hours = float(args.grant_personal_hours)
                # Finite and positive (adversary finding 10: inf/nan pass a
                # bare <= 0 check and OverflowError out of timedelta with a
                # raw traceback — on the consent-arming CLI, of all places).
                if not math.isfinite(hours) or hours <= 0:
                    print("--grant-personal-hours must be a finite number > 0")
                    return 2
                expiry = datetime.now(timezone.utc) + timedelta(hours=hours)
                grant = write_personal_grant(
                    home_dir, mode="timer", granted_by=principal,
                    expires_at=expiry.isoformat(), fence=act_fence,
                )
            else:
                grant = write_personal_grant(home_dir, mode="until_revoked", granted_by=principal, fence=act_fence)
        except (ValueError, OverflowError) as e:
            print(str(e))
            return 2
        detail = f" until {grant['expires_at']}" if grant.get("expires_at") else ""
        by = f" (granted_by {grant['granted_by']})" if grant.get("granted_by") else ""
        print(f"personal time -> {grant['mode']}{detail}{by}")
        print("(this armed the phase only - starting his own time is its own act)")
        return 0

    # THE BOOT CHECK (STOP-LANE-DESIGN §2.2/§2.4), before anything else of the loop: a child spawned before the
    # operator's Stop and booting after it refuses here and writes nothing. A hand start (no --generation) is
    # admitted at the current generation and honors a running or reserved loop.
    boot_fence: Optional[LifecycleFence] = None
    if args.generation is not None:
        boot_fence = LifecycleFence(home_dir, int(args.generation))
        try:
            moved = boot_fence.truth_moved()
        except LifecycleUnreadable as e:
            print(f"(the lifecycle generation is unreadable ({e}) - own time does not start; repair "
                  f"{PHASES_FILENAME})")
            return 2
        if moved:
            print("(stopped before it started: the operator's Stop moved the lifecycle generation since this start "
                  "was reserved - this loop writes nothing)")
            return 0
    else:
        current = loop_process_status(home_dir)
        if current.get("running"):
            print(f"his own time is already {'starting' if current.get('phase') == 'starting' else 'running'} "
                  f"(pid {current.get('pid') or current.get('spawner_pid')}) - one loop per home")
            return 1

    stop_file = home_dir / "STOP"
    if stop_file.exists():
        print(f"stop file already present at {stop_file} - remove it to start")
        return 1

    # ADMISSION at the start door (laurent 12:44: own time IS the personal
    # phase, OFF by default, operator-armed; CONTRACT v3 D1: a queued job
    # runs whatever the grant says): refuse BEFORE the substrate resolve or
    # any state change. run() re-checks at every day-open; this gate makes
    # the refusal immediate and the exit code scriptable for start surfaces.
    start_refusal = loop_start_refusal(home_dir)
    if start_refusal is not None:
        print(start_refusal)
        return 3

    # CONTEXT RECOMMENDATION at the start door (operator re-ruling
    # 2026-08-01: 50k recommended target, "acceptable to go to 200k" — and
    # the first pass's "a recommendation, not a wall — if it needs to grow,
    # it needs to grow" still governs; supersedes the 20k never-less ruling
    # this gate enforced hard since the 2026-07-13 production drive find).
    # The gateway's loop-start door went soft the same day (ge=1 +
    # #RECOMMENDED warning) — a hard return-2 here would refuse loops the
    # door just accepted. Warn, never block; version skew (no constant)
    # skips the note. The two imports degrade INDEPENDENTLY: an older
    # abstractmemory that has the recommendation but not yet the acceptable
    # ceiling keeps the below-recommendation note alive.
    try:
        from abstractmemory import ENTITY_CONTEXT_FLOOR
    except ImportError:
        ENTITY_CONTEXT_FLOOR = None
    try:
        from abstractmemory import ENTITY_CONTEXT_ACCEPTABLE
    except ImportError:
        ENTITY_CONTEXT_ACCEPTABLE = None
    if ENTITY_CONTEXT_FLOOR is not None and int(args.context_window) < int(ENTITY_CONTEXT_FLOOR):
        print(
            f"#RECOMMENDED context window {args.context_window} is below the recommended "
            f"working size ({ENTITY_CONTEXT_FLOOR}): the entity runs more efficiently "
            "around the recommendation — proceeding as configured."
        )
    if ENTITY_CONTEXT_ACCEPTABLE is not None and int(args.context_window) > int(ENTITY_CONTEXT_ACCEPTABLE):
        print(
            f"#RECOMMENDED context window {args.context_window} is above the acceptable "
            f"ceiling ({ENTITY_CONTEXT_ACCEPTABLE}): 'it is acceptable to go to 200k "
            "context' — beyond that is guidance, not a wall; growth is never blocked — "
            "proceeding as configured."
        )

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
        provider, model, thinking = resolve_home_substrate(
            args.provider, args.model, home_dir=home_dir,
            thinking=getattr(args, "thinking", None),
        )
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

    # HOST-RESOLVED ENDPOINT PROFILE (operator 2026-08-01: an endpoint:*
    # substrate spawned a child that died at day-open with "Unknown
    # provider: endpoint:airelay" — the profile lives in the HOST's store,
    # not ~/.abstractcore, and only in-process lanes had the host's ambient
    # resolver). spawn_loop_process serializes the host's private
    # resolution into our env; installing it as the ambient resolver here
    # reuses core's 2026-07-26 cross-package channel unchanged — local
    # config still wins, the payload is never logged (it may carry an
    # api_key), and version skew degrades to the old refusal, labeled.
    import json as _ep_json
    import os as _ep_os

    _endpoint_resolver_cm = None  # kept referenced for the loop's lifetime
    _raw_endpoint_profile = _ep_os.environ.get("ABSTRACTRUNTIME_ENDPOINT_PROFILE_JSON", "").strip()
    if _raw_endpoint_profile:
        try:
            _packed = _ep_json.loads(_raw_endpoint_profile)
            _spec = str(_packed.get("spec") or "").strip().lower()
            _payload = dict(_packed.get("profile") or {})
            from abstractcore.providers.endpoint_context import use_provider_endpoint_profile_resolver

            def _spawned_profile_resolver(spec: str, _s: str = _spec, _p: Dict[str, Any] = _payload) -> Optional[Dict[str, Any]]:
                return dict(_p) if str(spec or "").strip().lower() == _s else None

            _endpoint_resolver_cm = use_provider_endpoint_profile_resolver(_spawned_profile_resolver)
            _endpoint_resolver_cm.__enter__()  # process-lifetime; exits with the process
        except ImportError:
            print("#FALLBACK this core predates the endpoint-profile context - endpoint:* substrates resolve from local config only")
        except Exception as e:
            print(f"#FALLBACK spawn-provided endpoint profile unusable ({type(e).__name__}) - resolving from local config only")

    # CAPABILITY CLIENTS (operator 2026-11-09: the summaries model authors titles and summaries): the spawn's
    # capability endpoint profiles join the ambient resolver beside the mind's, and the factory builds each client
    # like the day client.
    _raw_cap_profiles = _ep_os.environ.get("ABSTRACTRUNTIME_CAPABILITY_ENDPOINT_PROFILES_JSON", "").strip()
    if _raw_cap_profiles:
        try:
            _cap_packed = _ep_json.loads(_raw_cap_profiles)
            _cap_by_spec = {
                str(v.get("spec") or "").strip().lower(): dict(v.get("profile") or {})
                for v in (_cap_packed.values() if isinstance(_cap_packed, dict) else [])
                if isinstance(v, dict) and str(v.get("spec") or "").strip()
            }
            from abstractcore.providers.endpoint_context import use_provider_endpoint_profile_resolver

            _mind_resolver_spec = locals().get("_spec")
            _mind_resolver_payload = locals().get("_payload")

            def _capability_profile_resolver(spec: str) -> Optional[Dict[str, Any]]:
                key = str(spec or "").strip().lower()
                if _mind_resolver_spec and key == _mind_resolver_spec:
                    return dict(_mind_resolver_payload or {})
                return dict(_cap_by_spec[key]) if key in _cap_by_spec else None

            _cap_resolver_cm = use_provider_endpoint_profile_resolver(_capability_profile_resolver)
            _cap_resolver_cm.__enter__()  # process-lifetime; exits with the process
        except Exception as e:  # noqa: BLE001
            print(f"#FALLBACK spawn-provided capability profiles unusable ({type(e).__name__}) - "
                  "capabilities resolve from local config only")
    # THE LOOP'S FENCE (STOP-LANE-DESIGN §2.2): admitted once, before anything of the loop exists; every fenced write
    # and every model call of this process carries it. Unreadable = no loop (fail closed).
    try:
        fence = boot_fence or LifecycleFence.admit(home_dir)
    except LifecycleUnreadable as e:
        print(f"(the lifecycle generation is unreadable ({e}) - own time does not start; repair {PHASES_FILENAME})")
        return 2
    if boot_fence is None:
        # A HAND START reserves the home like a spawn (Fable F7), under the same lock; its loop's claim takes it.
        try:
            _reserve_hand_start(home_dir, fence=fence)
        except (RuntimeError, StoppedByOperator) as e:
            print(str(e))
            return 1
    llm_for_capability = build_capability_llm_factory(home_dir, base_url=args.base_url, fence=fence)

    factory = build_session_factory(
        home_dir,
        provider=provider.strip().lower(),
        model=model,
        thinking=thinking,
        base_url=args.base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        context_window=args.context_window,
        shelf_size=args.shelf_size,
        llm_for_capability=llm_for_capability,
        fence=fence,
    )
    # Sleep is where consolidation runs (maintainer ruling 2026-07-08): wire
    # the dream pass into every self-elected rest window. In supervised mode
    # (rest ends the loop) there is no nap to consolidate in, so it is only
    # meaningful for 24/7 (rest_minutes > 0), but wiring it is harmless.
    def _narrator_llm():
        # ONE substrate (maintainer 06:32 ruling): the night voice runs on
        # the same resolved mind as the days - re-resolved at call time so
        # a substrate heal reaches the next night too.
        from .substrate import resolve_home_substrate as _rhs

        # The night narrator runs on the same resolved mind; the reasoning
        # dial is deliberately NOT applied to the dream voice (its calls
        # live in the consolidation engine - wiring the dial there is a
        # separate decision if the operator wants it).
        n_provider, n_model, _ = _rhs(args.provider, args.model, home_dir=home_dir)
        from abstractcore import create_llm as _cl

        kwargs: Dict[str, Any] = {"model": n_model}
        if n_provider in ("lmstudio", "openai-compatible", "openai_compatible"):
            kwargs["base_url"] = args.base_url
        _pw_timeout, _pw_budget = _entity_patience_window()
        kwargs.setdefault("timeout", _pw_timeout)
        kwargs.setdefault("retry_wall_clock_budget_s", _pw_budget)
        try:
            return _cl(n_provider, **kwargs)
        except TypeError:
            kwargs.pop("retry_wall_clock_budget_s", None)
            kwargs.pop("read_idle_timeout_s", None)
            return _cl(n_provider, **kwargs)

    consolidator = build_consolidator(
        home_dir,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        narrator_llm_factory=_narrator_llm,
        llm_for_capability=llm_for_capability,
        fence=fence,
    )
    # Substrate-heal (entity c75): the loop asks this before a terminal
    # failure cull — True when the home's substrate now differs from the
    # mind this loop spawned on (an operator remedy is already in the home).
    _spawn_pm = (provider.strip().lower(), str(model))

    def _substrate_changed() -> bool:
        try:
            from .substrate import resolve_home_substrate

            # The heal check compares the MIND (provider+model) only: a
            # changed reasoning dial is not a substrate heal.
            cur_p, cur_m, _ = resolve_home_substrate(None, None, home_dir=home_dir)
            return (str(cur_p or "").strip().lower(), str(cur_m or "")) != _spawn_pm
        except Exception:
            return False  # a resolution hiccup is not a heal signal

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
        substrate_changed=_substrate_changed,
        first_day_unconditional=bool(args.first_day_unconditional),
        return_to=None if args.return_to == "none" else args.return_to,
        fence=fence,
    )
    print(f"(own time starts: tick={args.tick_seconds}s, day={args.ticks_per_day} ticks, "
          f"stop: touch {stop_file} or Ctrl-C)")
    try:
        report = loop.run()
    except KeyboardInterrupt:
        print("\n(own time interrupted by operator - the last completed tick is remembered)")
        try:
            write_loop_status(home_dir, "stopped", stopped_by="operator-interrupt", fence=fence)
        except (StoppedByOperator, LifecycleUnreadable):
            pass  # after a Stop its own record stands
        return 0
    print(f"(own time ends: {report.ticks} ticks over {report.days} day(s), "
          f"stopped by {report.stopped_by}"
          + (f" - '{report.rest_reason}'" if report.rest_reason else "") + ")")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
