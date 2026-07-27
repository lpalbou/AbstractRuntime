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
from typing import Any, Callable, Dict, List, Optional, Tuple

from .chat import ChatSession, ChatHome
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
        done = Path(home_dir) / "work_order.done.md"
        stamp = f"\n\n---\n[{datetime.now(timezone.utc).isoformat()}] {verdict.strip()}\n"
        with done.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + stamp)
        p.unlink()
    except Exception:  # noqa: BLE001 - archiving must never kill the day
        pass


_WORK_BLOCK_RE = re.compile(r"```work[^\n`]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# ----------------------------------------------------------- the day gate
# DM#89 (laurent, 2026-07-20 ~13:36, relayed room seq 253; spec v8): the
# DAY GATE at every day boundary, order RULED:
#   1. work_order.md present -> WORK day ("work is always granted by
#      default if not sleeping and not visiting" - NO grant check).
#   2. standing drives (drive_pressure.total_open > 0) AND armed personal
#      grant -> PERSONAL day (the armed grant IS the standing consent;
#      drives PULL, they never deny - the >20 bound is the operator
#      ALARM, never this gate's input).
#   3. settled desk -> SLEEP at the 6h unattended cadence ("it should
#      wake at least once every 6h if i am not around"): the loop rests,
#      re-gates at each wake, and a quiet desk re-sleeps WITHOUT
#      summoning (zero LLM per cycle).
# Engine absence degrades LOUDLY to the pre-drives shape (armed grant ->
# personal day + #FALLBACK) - never a silently killed own time and never
# a silent spend.
UNATTENDED_NEED_CHECK_SECONDS = 6 * 3600
# laurent #54 (relayed 2026-07-20 16:50): "should not sleep ... when
# personal time was given and not used for at least 2h" — an armed grant
# carries a 2h USE floor before settled-desk sleep is a legitimate gate
# choice. Usage accumulates per GRANT (granted_at keys the meter; a new
# grant resets the floor). Conservative accounting: personal-day time is
# added at day close, so a killed day undercounts — undercounting gives
# MORE personal time, never less (the floor fails toward his intent).
PERSONAL_USE_FLOOR_SECONDS = 2 * 3600
_PERSONAL_USAGE_FILE = "personal_usage.json"
# v12 P1-4: the cycle clock persists (a respawn reset silently starved the
# maintenance rationale); ANY completed sleep window resets it - cadence,
# not quota.
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


def read_day_gate(home_dir: Path, *, skip_phases: frozenset = frozenset()) -> Dict[str, Any]:
    """One decision per day boundary: {phase, cause, detail, note?}.

    phase: "work" | "personal" | "sleep"; cause is the drive-cause TRACE
    (wire shape for loop_status.day_cause, entity's render): work_order |
    drives | grant_degraded | no_grant | settled_desk. Pure read.

    `skip_phases` (graph-edit build c4837): the caller's graph consult found
    a landing REMOVED by a blueprint edge op — the gate skips that phase's
    legs and the chain falls to the next legal landing (sleep is the floor).
    The gate itself never reads the graph (one consult site per caller, the
    consult owns the provenance/notes); it only honors the skip."""
    # VISIT-PREEMPTS-THE-GATE (spec v10; laurent dm#94: the four states
    # are mutually exclusive - an entity in visit can NOT be on personal
    # time): a live visit posture yields the whole gate BEFORE any leg -
    # no day arises, no landing writes (a settled-desk write here would
    # CLOBBER the visit-door's posture; the door owns both edge writes).
    try:
        state = read_entity_state(home_dir)
        if str(state.get("mode") or "").strip().lower() == "visiting":
            return {"phase": "visit", "cause": "visit_open",
                    "detail": "a visit session is open - the gate yields"}
    except Exception:  # noqa: BLE001 - an unreadable state never blocks a day
        pass
    order = read_work_order(home_dir)
    if order and PHASE_WORK not in skip_phases:
        return {
            "phase": PHASE_WORK, "cause": "work_order",
            "detail": WORK_ORDER_FILENAME, "work_order": order,
        }
    grant_refusal = personal_grant_refusal(read_personal_grant(home_dir))
    # v13 dial threading: the gate's cadence + floor come FROM the blueprint
    # (constants above remain the ruled seeds; an operator edit governs).
    from .phase_spec import load_phase_tunables as _load_tunables

    _tun, _ = _load_tunables(home_dir=home_dir)
    _need_check_s = int(float(_tun["unattended_wake_cadence_h"]) * 3600.0)
    _floor_s = float(_tun["grant_unused_floor_h"]) * 3600.0
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

    if PHASE_PERSONAL in skip_phases:
        # The personal landing was removed by a blueprint edge op (the
        # caller's consult): the chain falls to the sleep floor, honestly
        # named — never a fabricated no_grant.
        return {"phase": PHASE_SLEEP, "cause": "edge_removed",
                "detail": "the blueprint removed this boundary's personal landing",
                "need_check_s": _need_check_s}
    if grant_refusal is None:
        if total_open is None:
            # LOUD DEGRADE: pre-drives behavior (armed grant -> day), labeled.
            return {"phase": PHASE_PERSONAL, "cause": "grant_degraded",
                    "detail": degraded_note or "", "note": degraded_note}
        if total_open > 0:
            return {"phase": PHASE_PERSONAL, "cause": "drives",
                    "detail": f"{total_open} standing drive(s)",
                    "total_open": total_open}
        # laurent #54: an armed grant used < 2h refuses the sleep leg —
        # the given time must be LIVED before a settled desk may rest.
        used = read_personal_usage(home_dir)
        if used < _floor_s:
            return {"phase": PHASE_PERSONAL, "cause": "granted_unused",
                    "detail": (
                        f"{used / 3600:.1f}h of granted personal time used; "
                        f"the {_floor_s / 3600:g}h floor stands"
                    )}
        return {"phase": PHASE_SLEEP, "cause": "settled_desk",
                "detail": "grant armed, zero standing drives, use floor met",
                "need_check_s": _need_check_s}
    return {"phase": PHASE_SLEEP, "cause": "no_grant",
            "detail": grant_refusal,
            # The RULED phase-vocabulary cause word (state machine v3):
            # grant ends land as grant_expired/grant_revoked in the state
            # reason, whatever the gate's trace word says.
            "grant_cause": personal_grant_end_cause(read_personal_grant(home_dir)),
            "need_check_s": _need_check_s,
            "note": degraded_note}


def consult_gate_landing(
    home_dir: Path,
    from_phase: str,
    decision: Dict[str, Any],
    cause: str,
    out: Any,
) -> Tuple[Dict[str, Any], str]:
    """Graph consult over a gate decision (build order c4837; adversary A
    §3.2's day-gate row): the gate decides, the EFFECTIVE GRAPH governs.

    - REMOVED edge (legal_to -> None): the leg is skipped and the gate
      re-reads with that phase excluded — the chain falls to the next legal
      landing; sleep is the floor (never consulted, never skippable).
    - REDIRECT: the landing's target substitutes BEFORE any write — with
      R8's guards-travel rule enforced here (a redirect into personal
      requires the armed grant; into work requires a standing order;
      refused redirects fall back to the skip path, loudly).
    - INSTRUCTION: returned as the provenance-stamped cue line (STEERING,
      never law) for the caller to ride on the wake reason / day cue.

    Returns (decision, instruction_cue). Sleep/visit decisions pass through
    untouched (floor / derived). Any consult failure degrades loudly to the
    gate's own decision — the graph must never wedge a boundary."""
    instruction = ""
    try:
        from .phase_graph import instruction_cue, load_effective_graph

        graph, g_warns = load_effective_graph(home_dir)
    except Exception as e:  # noqa: BLE001 - a broken graph never blocks a day
        out(f"#FALLBACK graph consult unavailable ({e}); the gate decision stands")
        return decision, instruction
    for w in g_warns:
        out(f"({w})")
    skipped: set = set()
    for _hop in range(3):  # work -> personal -> sleep floor, bounded
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
            try:
                decision = read_day_gate(home_dir, skip_phases=frozenset(skipped))
            except Exception as e:  # noqa: BLE001
                out(f"#FALLBACK re-gate after skip failed ({e}); sleeping")
                return {"phase": PHASE_SLEEP, "cause": "edge_removed",
                        "detail": f"{from_phase}->{target}#{cause} removed"}, instruction
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
                try:
                    decision = read_day_gate(home_dir, skip_phases=frozenset(skipped))
                except Exception:  # noqa: BLE001
                    return {"phase": PHASE_SLEEP, "cause": "edge_removed",
                            "detail": "redirect refused; re-gate failed"}, instruction
                continue
            if landing.to == PHASE_WORK and not read_work_order(home_dir):
                out(
                    f"(blueprint redirect {from_phase}->{target}#{cause} -> work "
                    "REFUSED: no standing order - a work day needs a desk)"
                )
                skipped.add(target)
                try:
                    decision = read_day_gate(home_dir, skip_phases=frozenset(skipped))
                except Exception:  # noqa: BLE001
                    return {"phase": PHASE_SLEEP, "cause": "edge_removed",
                            "detail": "redirect refused; re-gate failed"}, instruction
                continue
            out(
                f"(blueprint redirect: {from_phase}->{target}#{cause} lands "
                f"{landing.to} instead - {landing.provenance})"
            )
            redirected = dict(decision)
            redirected["phase"] = landing.to
            redirected["redirected_from"] = target
            redirected["cause"] = decision.get("cause") or cause
            decision = redirected
        cue_line = instruction_cue(landing)
        if cue_line:
            instruction = cue_line
        return decision, instruction
    return decision, instruction


def append_phase_changed(
    home_dir: Path,
    *,
    from_phase: str,
    to: str,
    cause: str,
    written_by: str,
    provenance: str = "structural",
) -> None:
    """The `phase_changed` marker — SPELLED with the graph-edit build
    (c4837; it was 'RESERVED AND UNSPELLED' since the v12 marker contract).

    Loop-written transitions land one machine-readable row in the same
    append-only biography file the state writes ride
    (<home>/state_history.jsonl), distinguishable by the `marker` key:
    {marker: phase_changed, from, to, cause, written_by, at, provenance}.
    `cause` is the spec cause word where one exists (cadence_need_check,
    personal_cycle, grant_expired...) and the gate's trace word otherwise
    (settled_desk, work_order — the evaluator internals, honestly named).
    Best-effort: a marker append must never kill a transition."""
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


def drives_cue_note(home_dir: Path, *, k: int = 4) -> Tuple[str, List[str]]:
    """The MERGE composer (adversary F2 contract): the day-open cue offers
    the top-k ALIVE drives as ACT-FRAME handles only - kind #tag [date],
    NO gist words (cue gist would lexically self-confirm one hop later
    through stimulus admission). One offer, release clause in the text.
    Returns (note, offered_graph_ids) - the ids feed the driver's
    first-turn COMMIT EXCLUSION (the other half of the same contract:
    a cue mention must not strengthen the drive it names)."""
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
                offered = {str(it.get("record_id") or "") for it in items}
                fresh = [d for d in dust if str(d.get("record_id") or "") not in offered]
                if fresh:
                    items = (items[: k - 1] if len(items) >= k else items) + [fresh[0]]
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
        parts.append(f"{label} {tag} [{born}]{group_note}")
        if rid:
            ids.append(rid)
    note = (
        " Alive on your desk today: " + "; ".join(parts) + "."
        " If one pulls you, open it with read_memory (the #tag addresses it)."
        " None of this is owed - the day is yours."
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
task from anyone else. This time is yours; there is nothing you are supposed
to do with it, and no way to spend it wrong.

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
            gist = str(q.get("text") or "").strip().splitlines()[0][:120]
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
            words = " ".join(str(pick.object or "").split())[:110]
            tag = memory_tag(str(pick.subject or ""))
            offers.append(f'"{words}" ({tag})')
        ratio = f" You hold {len(rows)} interest(s); {explored} ever explored."
        note = (
            ratio + " Alive in you: " + "; ".join(offers) +
            ". read_memory fetches one; if a diary entry today DEVELOPS it, "
            "add explores=<its #tag> on the block line - exploring feeds an "
            "interest, never closes it. Yours if it pulls, never owed."
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
) -> None:
    """Best-effort status write; the loop must never die over its status.

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
    prior = read_loop_status(home_dir)
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
    started = _own_start_time()
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
    # kind: work_order|drives|grant_degraded|settled_desk|no_grant.
    if isinstance(day_cause, dict) and day_cause.get("cause"):
        payload["day_cause"] = {
            "kind": str(day_cause.get("cause") or ""),
            "detail": str(day_cause.get("detail") or "")[:200],
        }
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


def record_loop_substrate(home_dir: Path, provider: str, model: str) -> None:
    """Stamp the mind the loop is CURRENTLY running into loop_status (entity
    c78 render ask). The factory calls this at each day-open after it
    re-resolves the home's substrate, so an observer's staleness cue can
    compare the operator's substrate change time against the mind actually
    in use — a change older than this stamp's `updated_at` has been picked
    up, not still pending. Read-merge on the existing status dict (phase and
    liveness fields preserved); best-effort, never fatal."""
    import json
    from datetime import datetime, timezone

    prior = read_loop_status(home_dir)
    prior.pop("running", None)  # derived; never persisted
    prior["substrate"] = {"provider": str(provider), "model": str(model)}
    prior["substrate_at"] = datetime.now(timezone.utc).isoformat()
    prior.setdefault("phase", "between")
    try:
        (Path(home_dir) / "loop_status").write_text(
            json.dumps(prior) + "\n", encoding="utf-8"
        )
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

    killed = False
    escalated = False
    # SIGNAL ONLY OUR OWN INCARNATION (whole-package adversary P0-class,
    # 2026-07-13: a clean-stop file whose pid the OS recycled drew a
    # SIGKILL at an innocent process — the token that detects exactly this
    # was computed one line above and ignored). The predicate is IDENTITY,
    # not the folded `running` (which also folds the staleness belt — a
    # WEDGED same-incarnation loop reads not-running yet is exactly what an
    # emergency freeze must still kill): kill when the pid is alive AND its
    # start-time token matches the file's stamp; unstamped files (older
    # writers) keep the legacy phase-gated behavior, so a clean "stopped"
    # file never draws a signal either way.
    stamped_token = str(status.get("pid_started_at") or "")
    if pid > 0 and _pid_alive(pid):
        if stamped_token:
            same_incarnation = _pid_start_time(pid) == stamped_token
        else:
            same_incarnation = status.get("phase") in ("day", "between")
    else:
        same_incarnation = False
    if same_incarnation:
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
    # 36 seats (maintainer, 2026-07-09): the 12% token budget still seats 36
    # rich digests — seats fill, tokens hold.
    shelf_size: int = 36,
    # ~40k DEFAULT (maintainer ruling 2026-07-13 15:20: "we want to optimize
    # the context of an entity so it can run fast, which means up to 40k
    # tokens roughly. this is NOT a hardcap, more like an optimization when
    # possible"). The default is the OPTIMIZATION; an explicit operator
    # value always wins in either direction (floor 20k refuses loudly —
    # never a silent cap, per the no-silent-fallback ADR discipline).
    context_window: int = 40960,
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
        substrate_changed: Optional[Callable[[], bool]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        out: Callable[[str], None] = print,
    ) -> None:
        if ticks_per_day < 1:
            raise ValueError("ticks_per_day must be >= 1")
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
        self._cue_drive_ids: List[str] = []
        # v11/v12 personal<->sleep maintenance cycle (laurent dm#104):
        # personal seconds lived since the last completed sleep window.
        # PERSISTED (v12 P1-4: in-memory reset on every respawn silently
        # starved the maintenance rationale); loaded lazily at first use.
        self._personal_cycle_seconds: float = 0.0  # loaded at run() start
        # v13 no-churn need-check: quiet re-checks keep their next deadline
        # in memory (the state file is NOT rewritten per check - no marker
        # churn); reset whenever a fresh sleep landing is written.
        self._need_check_at = None
        # Graph-edit P0 fix (adversary 2, F1): the need-check's consult must
        # GOVERN the day that opens — the wake returns to the top boundary,
        # whose bare gate read would otherwise re-decide without the graph.
        # The wake stashes its cause word here; the top-boundary consult
        # consumes it (from=sleep) so removal/redirect survive the wake.
        self._wake_gate_cause: Optional[str] = None
        # Marker honesty (F7): the phase_changed marker for a day-opening
        # transition writes when the day actually OPENS (after the operator
        # gate + belt), never at wake — staged here by the boundary consult.
        self._pending_day_marker: Optional[Tuple[str, str, str, str]] = None
        # The from-side of boundary transitions = the last day that actually
        # OPENED (adversary-2 F7d, live-caught by the full-loop pin: deriving
        # it from _day_gate recorded decisions the operator gate idled away —
        # a work decision that never ran became the marker's from-side).
        # None = the entity slept between (gate landing / operator sleep).
        self._last_opened_phase: Optional[str] = None
        # F7d: the from-side of boundary transitions = the last phase that
        # actually OPENED (self._day_gate is staged before the operator
        # gate/belt can still preempt — a staged-never-opened day must not
        # become a marker's from-side). None = the entity was asleep.
        self._last_opened_phase: Optional[str] = None
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
        self, reason: str, *, include_dream: bool = True, include_identity: bool = True
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
            try:
                write_entity_state(
                    self.state_home, "asleep",
                    reason=f"self-elected sleep: {reason}", mode="dreaming", written_by="self",
                )
            except Exception as e:  # noqa: BLE001 - sleep must not die over its own marker
                self.out(f"#FALLBACK could not mark self-sleep state: {e}")
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
                # Carry the ORIGINAL sleep deadline through the rewrite —
                # a fresh changed_at must not restart the bound clock.
                deadline = sleep_bound_deadline(current)
                write_entity_state(
                    self.state_home, "asleep",
                    reason=f"self-elected sleep: {reason}", mode="resting", written_by="self",
                    wake_at=deadline.isoformat() if deadline is not None else "",
                )
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

    def _bounded_asleep_or_paused(self, s: Dict[str, Any]) -> bool:
        """Idle predicate for the asleep/paused gate, WITH the sleep bound
        (decision:sleep-is-bounded, entity c2465 ask 2): a real sleep older
        than its deadline wakes the entity instead of idling forever.
        `paused` never auto-clears (kill switch); visit yields are excluded
        inside sleep_bound_deadline (the gateway reaper's lane). The wake
        write is the predicate's one deliberate side effect — documented
        here because _idle_while's contract is otherwise pure reads."""
        from datetime import datetime, timezone

        if s["state"] == "paused":
            return True
        if s["state"] != "asleep":
            self._need_check_at = None
            return False
        deadline = sleep_bound_deadline(s, home_dir=self.state_home)
        # v13 no-churn law: after a quiet need-check, the state file's
        # wake_at is deliberately stale - the in-memory deadline governs.
        if self._need_check_at is not None:
            deadline = self._need_check_at
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            if self.state_home is None:
                return True
            writer = str(s.get("written_by") or "")
            if writer in ("day-gate", "grant-gate", "operator"):
                # v13 cadence_need_check (wake_conditions, spec v13): a
                # ZERO-TOKEN read over the standing sets, landing THROUGH
                # the day gate. Nothing sanctioned = the SAME sleep
                # continues - no awake/asleep marker pair, no biography
                # event per check; a lightweight status trace at most.
                try:
                    decision = read_day_gate(self.state_home)
                except Exception as e:  # noqa: BLE001 - unreadable gate keeps idling
                    self.out(f"#FALLBACK need-check gate read failed: {e}")
                    decision = {"phase": PHASE_SLEEP, "cause": "gate_degraded",
                                "need_check_s": UNATTENDED_NEED_CHECK_SECONDS}
                # GRAPH CONSULT (build c4837): the need-check's landings are
                # the two genuinely operator-editable edges (B census #21/22,
                # sleep->work / sleep->personal #cadence_need_check) —
                # removal skips the leg, redirect substitutes (guards
                # travel), instruction rides the wake reason into the first
                # cue via the shipped wake-reason seed.
                edge_cue = ""
                if decision.get("phase") in (PHASE_WORK, PHASE_PERSONAL):
                    decision, edge_cue = consult_gate_landing(
                        self.state_home, PHASE_SLEEP, decision,
                        "cadence_need_check", self.out,
                    )
                if decision["phase"] == PHASE_SLEEP:
                    from datetime import timedelta

                    cadence = int(decision.get("need_check_s") or UNATTENDED_NEED_CHECK_SECONDS)
                    self._need_check_at = datetime.now(timezone.utc) + timedelta(seconds=cadence)
                    self.out(
                        f"(need-check: nothing sanctioned ({decision['cause']}) - "
                        f"the same sleep continues; next check in {cadence // 3600}h)"
                    )
                    self._status("between")
                    return True
                # A sanctioned day (or an open visit) lands ONE wake marker.
                try:
                    wake_reason = (
                        f"need-check: {decision['cause']} sanctions a day - waking"
                        if decision["phase"] != "visit"
                        else "need-check: a visit is open - waking"
                    )
                    if edge_cue:
                        # The edge's steering prose rides the wake reason —
                        # the shipped wake-reason seed folds it into the
                        # first cue (never law, always labeled).
                        wake_reason = f"{wake_reason}. {edge_cue}"
                    write_entity_state(
                        self.state_home, "awake",
                        reason=wake_reason,
                        written_by="need-check",
                    )
                    # P0 fix (adversary 2, F1): the wake's cause carries to
                    # the top boundary so its consult re-derives the SAME
                    # skip/redirect and the graph governs the day that
                    # opens. The phase_changed marker moved there too (F7a:
                    # a wake-time marker engraved transitions that the
                    # operator gate could still preempt).
                    if decision["phase"] in (PHASE_WORK, PHASE_PERSONAL):
                        self._wake_gate_cause = "cadence_need_check"
                    self.out(f"(need-check: {decision.get('cause') or decision['phase']} - waking)")
                    self._need_check_at = None
                    return False
                except Exception as e:  # noqa: BLE001 - a failed wake write keeps idling honestly
                    self.out(f"#FALLBACK need-check wake write failed: {e}")
                    return True
            try:
                write_entity_state(
                    self.state_home, "awake",
                    reason=f"sleep bound reached (~{SLEEP_BOUND_SECONDS / 3600:g}h) - "
                    "a sleep is bounded; own time resumes",
                    written_by="sleep-bound",
                )
                self.out("(sleep bound reached - waking)")
                return False
            except Exception as e:  # noqa: BLE001 - a failed wake write keeps idling honestly
                self.out(f"#FALLBACK sleep-bound wake write failed: {e}")
            return True
        return True

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
                    reason=(
                        "woke from self-elected sleep - own time resumes"
                        if writer == "self"
                        else "maintenance cycle complete - personal time resumes"
                    ),
                    written_by=writer,
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

    def _status(self, phase: str, *, stopped_by: Optional[str] = None,
                day_kind: Optional[str] = None,
                day_cause: Optional[Dict[str, Any]] = None,
                tunables: Optional[Dict[str, Any]] = None) -> None:
        if self.state_home is not None:
            write_loop_status(self.state_home, phase, stopped_by=stopped_by,
                              day_kind=day_kind, day_cause=day_cause,
                              tunables=tunables)

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
        # v12 P1-4: the cycle clock persists across respawns.
        if self.state_home is not None:
            self._personal_cycle_seconds = read_cycle_clock(self.state_home)
        self._status("between")
        # SPAWN-WAKE for loop-exit landings (safe-subset pair rule): the
        # terminal-exit landing writes asleep so a dead loop never reads
        # awake — but a NEW loop process arriving IS the wake for that
        # landing (supervisor respawn, next spawn after max_ticks). Only
        # written_by=loop-exit wakes here: operator sleeps, day-gate
        # cadence sleeps and visit yields keep their own wake rules.
        if self.state_home is not None:
            try:
                landed = read_entity_state(self.state_home)
                if (
                    str(landed.get("state") or "") == "asleep"
                    and str(landed.get("written_by") or "") == "loop-exit"
                ):
                    write_entity_state(
                        self.state_home, "awake",
                        reason="a new loop arrived - the exit landing wakes",
                        written_by="loop-spawn",
                    )
            except Exception as e:  # noqa: BLE001 - never block a spawn
                self.out(f"#FALLBACK spawn-wake check failed: {e}")
        while True:
            if self.max_ticks is not None and report.ticks >= self.max_ticks:
                report.stopped_by = "max_ticks"
                break
            if self._should_stop():
                report.stopped_by = self.stop_cause or "stop_file"
                break

            # THE DAY GATE (dm#89, supersedes the 12:44 exit-on-unarmed
            # shape): work_order -> WORK day (work is always granted, no
            # grant check); standing drives + armed grant -> PERSONAL day
            # (drives PULL, the armed grant is the standing consent);
            # settled desk -> SLEEP at the 6h unattended need-check cadence
            # ("wake at least once every 6h if i am not around") — the loop
            # RESTS instead of exiting, so a work order left while
            # unattended is picked up within one cadence. A quiet re-check
            # re-sleeps without summoning: ZERO LLM per cycle. paused stays
            # the kill switch (the operator state gate below); STOP always
            # wins. Homes only (state_home=None = harness loops).
            # Previous OPENED day's phase = the from-side of this boundary's
            # edges (graph-edit build c4837: the work-close consult + marker
            # honesty both need it; a decision the operator gate idled away
            # never counts — only opened days do).
            _prev_phase = self._last_opened_phase
            self._day_gate = None
            self._pending_day_marker = None
            if self.state_home is not None:
                decision = read_day_gate(self.state_home)
                for note_key in ("note",):
                    if decision.get(note_key):
                        self.out(f"({decision[note_key]})")
                # NEED-CHECK CARRY (P0 fix, adversary 2 F1): a wake that a
                # graph consult shaped hands its cause to THIS boundary —
                # the consult re-runs here (same graph, same skip/redirect)
                # so the effective graph governs the day that actually
                # opens, not just the wake's words. Consumed once.
                _wake_cause = self._wake_gate_cause
                self._wake_gate_cause = None
                _carry_cue = ""
                if _wake_cause and decision.get("phase") in (PHASE_WORK, PHASE_PERSONAL):
                    decision, _carry_cue = consult_gate_landing(
                        self.state_home, PHASE_SLEEP, decision, _wake_cause, self.out,
                    )
                if _carry_cue and _carry_cue not in (cue or ""):
                    cue = f"{(cue or '').strip()} {_carry_cue}".strip()
                # WORK-CLOSE CONSULT (B census rows 10/11, the one consultable
                # boundary transition here): a redirect on work->sleep
                # (#task_complete / #no_task) substitutes the landing — e.g.
                # "after finishing work, take personal time" — with guards
                # traveling (grant checked inside the consult). Instruction
                # prose rides the day cue below.
                _boundary_edge_cue = ""
                if (
                    _prev_phase == PHASE_WORK
                    and decision.get("phase") == PHASE_SLEEP
                    and decision.get("cause") not in ("no_grant",)
                ):
                    _wc_cause = (
                        "task_complete" if not read_work_order(self.state_home) else "no_task"
                    )
                    try:
                        from .phase_graph import instruction_cue as _icue, load_effective_graph as _leg

                        _g, _gw = _leg(self.state_home)
                        for w in _gw:
                            self.out(f"({w})")
                        _landing = _g.legal_to(PHASE_WORK, PHASE_SLEEP, _wc_cause)
                        if _landing is not None and _landing.to == PHASE_PERSONAL:
                            if personal_grant_refusal(read_personal_grant(self.state_home)) is None:
                                self.out(
                                    f"(blueprint redirect: work->sleep#{_wc_cause} lands "
                                    f"personal instead - {_landing.provenance})"
                                )
                                decision = {
                                    "phase": PHASE_PERSONAL, "cause": _wc_cause,
                                    "detail": f"blueprint redirect of work->sleep#{_wc_cause}",
                                    "redirected_from": PHASE_SLEEP,
                                }
                                # Marker staged for day-open (F7b: writing
                                # here preceded the operator gate/belt).
                            else:
                                self.out(
                                    "(blueprint redirect to personal refused: the grant "
                                    "is not armed - the guard travels with the arrow)"
                                )
                        if _landing is not None:
                            _boundary_edge_cue = _icue(_landing)
                    except Exception as e:  # noqa: BLE001 - consult never blocks a boundary
                        self.out(f"#FALLBACK work-close graph consult failed: {e}")
                if _boundary_edge_cue:
                    cue = f"{(cue or '').strip()} {_boundary_edge_cue}".strip()
                # MARKER STAGING (F7 honesty cluster): a day-opening
                # transition's phase_changed writes when the day actually
                # OPENS (after the operator gate + belt) — staged here,
                # written at the summon, cleared every boundary iteration.
                _from_boundary = _prev_phase or PHASE_SLEEP
                if (
                    decision.get("phase") in (PHASE_WORK, PHASE_PERSONAL)
                    and _from_boundary != decision.get("phase")
                ):
                    self._pending_day_marker = (
                        _from_boundary, str(decision["phase"]),
                        str(_wake_cause or decision.get("cause") or ""),
                        "blueprint redirect" if decision.get("redirected_from") else "structural",
                    )
                if decision["phase"] == "visit":
                    # VISIT-PREEMPTS-THE-GATE: fall through to the operator
                    # state gate below, which idles on the visiting posture
                    # until the door's close writes the restore.
                    self.out("(a visit is open - the day gate yields)")
                elif decision["phase"] == PHASE_SLEEP:
                    from datetime import datetime, timedelta, timezone

                    cadence = int(decision.get("need_check_s") or UNATTENDED_NEED_CHECK_SECONDS)
                    self.out(
                        f"(settled desk: {decision['cause']} ({decision['detail']}) - "
                        f"the ruled landing is sleep)"
                    )
                    # SUPERVISED mode (rest_minutes == 0, the existing mode
                    # axis): a sleep decision ENDS the run — the supervisor
                    # owns respawn, and a bounded test run must never idle
                    # 6h. 24/7 mode rests at the ruled cadence and re-gates.
                    if decision["cause"] == "no_grant":
                        landing_reason = (
                            f"personal ended ({decision.get('grant_cause') or 'grant_revoked'}) "
                            "- the ruled landing is sleep"
                        )
                        landing_writer = "grant-gate"
                    else:
                        landing_reason = f"day gate: {decision['cause']} ({decision['detail']})"
                        landing_writer = "day-gate"
                    if self.rest_minutes <= 0:
                        report.stopped_by = (
                            "personal_disarmed" if decision["cause"] == "no_grant"
                            else "settled_desk"
                        )
                        try:
                            write_entity_state(
                                self.state_home, "asleep",
                                reason=landing_reason, written_by=landing_writer,
                            )
                            # F7e: biography parity — supervised landings
                            # mark like 24/7 ones (same honesty condition).
                            if _prev_phase and _prev_phase != PHASE_SLEEP:
                                append_phase_changed(
                                    self.state_home,
                                    from_phase=_prev_phase, to=PHASE_SLEEP,
                                    cause=str(decision.get("grant_cause") or decision.get("cause") or "sleep"),
                                    written_by=landing_writer,
                                )
                        except Exception as e:  # noqa: BLE001
                            self.out(f"#FALLBACK could not write the sleep landing: {e}")
                        break
                    wake_at = (
                        datetime.now(timezone.utc) + timedelta(seconds=cadence)
                    ).isoformat()
                    write_entity_state(
                        self.state_home, "asleep",
                        reason=f"{landing_reason} - need-check in {cadence // 3600}h",
                        written_by=landing_writer,
                        wake_at=wake_at,
                    )
                    # phase_changed marker (spelled with c4837): the gate's
                    # sleep landing is a loop-written transition; the cause
                    # is the RULED word where one exists (grant ends), the
                    # gate trace otherwise (evaluator internals, honest).
                    # F7d: never fabricate the from-side — no marker when
                    # the previous phase is unknown (sleep->sleep is not a
                    # transition; a guessed "personal" is a lie engraved).
                    if _prev_phase and _prev_phase != PHASE_SLEEP:
                        append_phase_changed(
                            self.state_home,
                            from_phase=_prev_phase, to=PHASE_SLEEP,
                            cause=str(decision.get("grant_cause") or decision.get("cause") or "sleep"),
                            written_by=landing_writer,
                        )
                    self._need_check_at = None  # fresh landing owns the clock
                    self._last_opened_phase = None  # the entity sleeps; next from-side is sleep
                    self._status("between", day_cause=decision)
                    continue  # the operator-state gate below idles on it
                self._day_gate = decision

            # Operator state gate before a day opens (a2a 0008): asleep or
            # paused idles here (dreams may run gateway-side while asleep -
            # the no-summon window is enforced by this very gate). Waking is
            # HONEST: the first cue names what happened and for how long.
            gate = self._operator_state()
            if gate["state"] in ("asleep", "paused"):
                self.out(f"({gate['state']} by operator{self._since(gate)} - idling)")
                # Something ended the running stretch (operator sleep, door
                # yield): the next boundary's from-side is sleep, never the
                # day this gate just idled away (marker honesty, F7d).
                self._last_opened_phase = None
                woke = self._idle_while(self._bounded_asleep_or_paused, phase="between")
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
                # Wake facts only (P0-1: the offer composes at day-open
                # now, ONE site) - the date stays here so the wake moment
                # itself is grounded.
                cue = (
                    f"{_today_stamp()}you were {gate['state']}{self._since(gate)} (operator-initiated) "
                    f"and are awake again - a new stretch of your own time begins, "
                    f"nothing owed.{reason_note}{note}"
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
                drives_note, drive_ids = drives_cue_note(self.state_home)
                # P2-2: BOTH note lanes' offered ids share one exclusion.
                self._cue_drive_ids = drive_ids + standing_state_offered_ids()
                base = (cue or "").strip()
                if not base.lstrip().startswith("today is"):
                    cue = f"{_today_stamp()}{base}{standing}{drives_note}"
                else:
                    # A wake path already stamped the date; append offers
                    # only if the wake text does not already carry them.
                    if "Alive on your desk" not in base:
                        cue = f"{base}{standing if 'You hold' not in base else ''}{drives_note}"
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
                append_phase_changed(
                    self.state_home, from_phase=_pm_from, to=_pm_to,
                    cause=_pm_cause, written_by="day-gate", provenance=_pm_prov,
                )
            self._pending_day_marker = None
            # The day OPENS here: it becomes the next boundary's from-side.
            if isinstance(self._day_gate, dict) and self._day_gate.get("phase"):
                self._last_opened_phase = str(self._day_gate["phase"])
            self._status("day", day_cause=self._day_gate)
            opened = False
            try:
                session = self.open_session()
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
                    if self.state_home is not None:
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
                        finally:
                            if tick_lease is not None:
                                tick_lease.release()  # the tick's writer window ends HERE
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
                                archive_work_order(self.state_home, verdict=verdict)
                            report.stopped_by = "work_done"
                            self.out(f"(work verdict: {verdict[:120]})")
                            break

                    cue = parse_next_cue(marked) or NEUTRAL_CUE
                    if self.tick_seconds > 0 and self._interruptible_sleep(self.tick_seconds):
                        report.stopped_by = self.stop_cause or "stop_file"
                        break
            finally:
                # laurent #54: the 2h personal-use floor meters LIVED day
                # time - recorded at close, keyed to the current grant.
                # Best-effort; conservative (a killed process undercounts,
                # which grants MORE personal time, never less).
                _gate_personal = (self._day_gate or {}).get("phase") == PHASE_PERSONAL
                _session_personal = getattr(session, "phase", "") == PHASE_PERSONAL
                if self.state_home is not None and (_gate_personal or _session_personal):
                    # The GATE's decision is the loop's own truth for what
                    # kind of day this was (a harness factory may not stamp
                    # session.phase; production stamps both).
                    _day_elapsed = time.monotonic() - _day_started_monotonic
                    record_personal_usage(self.state_home, _day_elapsed)
                    # v11 cycle accumulator: cycle-sleep windows never count
                    # (only lived day time accumulates; the meter above
                    # already gives the floor leg the same property).
                    self._personal_cycle_seconds += _day_elapsed
                    write_cycle_clock(self.state_home, self._personal_cycle_seconds)
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

            # v11 PERSONAL<->SLEEP MAINTENANCE CYCLE (laurent dm#104: "when
            # in personal time, the entity can go to sleep after 2h for 1h
            # and wake up in personal time"). 24/7 residents only (a
            # supervised run exits at its bounds); the windows are BLUEPRINT
            # TUNABLES read fresh at each trigger; the grant is neither
            # ended nor consumed - the wake returns to the gate, which
            # lands personal while the grant stands (VISIT-PREEMPTS and a
            # mid-sleep deactivation both resolve at the gate naturally).
            # "max_ticks" is the loop's CONTINUE-marker (LifeReport's
            # default; the rest branch resets to it) - a normally-ended day
            # carries it; real stops carry their own words and break above.
            if (
                report.stopped_by == "max_ticks"
                and self.rest_minutes > 0
                and self.state_home is not None
            ):
                from .phase_spec import load_phase_tunables

                tunables, t_warns = load_phase_tunables(home_dir=self.state_home)
                for w in t_warns:
                    self.out(f"({w})")
                # R3 receipt: stamp WHAT this loop just read (cells honesty).
                self._status("between", tunables=tunables)
                cycle = tunables["personal_cycle"]
                window_s = float(cycle["personal_window_h"]) * 3600.0
                if not cycle.get("enabled", True):
                    # v13 kill switch: the operator disabled the cycle -
                    # the clock keeps counting (re-enable resumes cadence).
                    window_s = float("inf")
                # v12 PRECONDITIONS (design adversary P0-1): the cycle fires
                # only on a QUIET boundary - no standing work order (v9b: a
                # task must never be slept over), no visit/operator state
                # standing (the v10 unguarded-writer class: a cycle write
                # over a visiting posture would destroy the door's truth).
                # A disqualifier HOLDS the clock; the next boundary re-checks.
                _cycle_ok = self._personal_cycle_seconds >= window_s
                if _cycle_ok and read_work_order(self.state_home):
                    self.out("(cycle held: a work order stands - the desk outranks maintenance)")
                    _cycle_ok = False
                if _cycle_ok:
                    try:
                        _st = read_entity_state(self.state_home)
                        if str(_st.get("state") or "") != "awake" or str(_st.get("mode") or ""):
                            self.out("(cycle held: the state is not plainly awake - the door/operator owns it)")
                            _cycle_ok = False
                    except Exception:  # noqa: BLE001 - unreadable state holds the clock
                        _cycle_ok = False
                _cycle_instruction = ""
                if _cycle_ok:
                    # GRAPH CONSULT (c4837): personal->sleep#personal_cycle
                    # is a consultable loop edge (B census #18 — removal is
                    # the edge-shaped twin of personal_cycle.enabled=false;
                    # one more way to say the same thing, both honest).
                    # Instruction prose rides the cycle's wake cue. bound_h
                    # is IGNORED here with a note: sleep_window_h is the
                    # dial that governs this window (no two knobs).
                    try:
                        from .phase_graph import instruction_cue as _icue, load_effective_graph as _leg

                        _g, _gw = _leg(self.state_home)
                        for w in _gw:
                            self.out(f"({w})")
                        _cyc_landing = _g.legal_to(PHASE_PERSONAL, PHASE_SLEEP, "personal_cycle")
                        if _cyc_landing is None:
                            self.out(
                                "(cycle held: the blueprint removed "
                                "personal->sleep#personal_cycle - maintenance waits)"
                            )
                            _cycle_ok = False
                        elif _cyc_landing.to != PHASE_SLEEP:
                            # F6 (adversary 2): a redirect on the cycle edge
                            # is DISOBEYED loudly — the maintenance window IS
                            # a sleep by design (v19 policy=dial refuses the
                            # op at the door; this is the belt for hand-made
                            # files without edit_policy).
                            self.out(
                                f"(#NOTE cycle redirect to {_cyc_landing.to!r} ignored - "
                                "the maintenance window sleeps by design)"
                            )
                            _cycle_instruction = _icue(_cyc_landing)
                        else:
                            if _cyc_landing.bound_h is not None:
                                self.out(
                                    "(#NOTE bound_h on the cycle edge is ignored - "
                                    "sleep_window_h is the governing dial)"
                                )
                            _cycle_instruction = _icue(_cyc_landing)
                    except Exception as e:  # noqa: BLE001 - consult never blocks the cycle
                        self.out(f"#FALLBACK cycle graph consult failed: {e}")
                if _cycle_ok:
                    sleep_s = float(cycle["sleep_window_h"]) * 3600.0
                    self.out(
                        f"(personal cycle: ~{cycle['personal_window_h']:g}h lived - "
                        f"a {cycle['sleep_window_h']:g}h maintenance sleep begins; "
                        "your personal time resumes after)"
                    )
                    report.sleeps += 1
                    from datetime import datetime, timedelta, timezone

                    _cycle_wake_at = (
                        datetime.now(timezone.utc) + timedelta(seconds=sleep_s)
                    ).isoformat()
                    try:
                        write_entity_state(
                            self.state_home, "asleep",
                            reason=(
                                f"personal_cycle maintenance ({cycle['sleep_window_h']:g}h) - "
                                "the grant stands; personal time resumes at wake"
                            ),
                            written_by="personal-cycle",
                            wake_at=_cycle_wake_at,
                        )
                        # F7c: the marker rides the SUCCESSFUL state write —
                        # a failed write must not engrave a transition.
                        append_phase_changed(
                            self.state_home, from_phase=PHASE_PERSONAL, to=PHASE_SLEEP,
                            cause="personal_cycle", written_by="personal-cycle",
                        )
                    except Exception as e:  # noqa: BLE001
                        self.out(f"#FALLBACK cycle sleep state write failed: {e}")
                    dreamed_cycle = self._sleep_window(
                        "personal_cycle graph maintenance",
                        # Cycle windows touch NEITHER dreams nor the self
                        # (cti#399: "a 2h maintenance nap must not touch the
                        # self"); both flags off, quality passes only.
                        include_dream=False, include_identity=False,
                    )
                    if self._interruptible_sleep(sleep_s):
                        self._wake_from_self_sleep()
                        report.stopped_by = self.stop_cause or "stop_file"
                        break
                    self._wake_from_self_sleep()
                    self._personal_cycle_seconds = 0.0
                    if self.state_home is not None:
                        write_cycle_clock(self.state_home, 0.0)
                    cycle_note = ""
                    if isinstance(dreamed_cycle, dict) and dreamed_cycle.get("formed"):
                        from .night_voice import wake_residue

                        cycle_note = " the maintenance window formed a dream (waking evidence disposes)"
                        cycle_note += wake_residue(list(dreamed_cycle.get("signals") or []))
                    # phase_changed markers for the cycle pair (c4837): the
                    # entry landed before the window; the return lands here.
                    if self.state_home is not None:
                        append_phase_changed(
                            self.state_home, from_phase=PHASE_SLEEP, to=PHASE_PERSONAL,
                            cause="personal_cycle", written_by="personal-cycle",
                        )
                    _instr = f" {_cycle_instruction}" if _cycle_instruction else ""
                    cue = (
                        f"{_today_stamp()}your maintenance sleep is over - personal time "
                        f"resumes, the grant stands, nothing owed.{cycle_note}{_instr}"
                    )
                    continue

            if report.stopped_by == "rest" and self.rest_minutes > 0:
                # 24/7 mode: rest is SLEEP, not a blank nap (maintainer ruling
                # 2026-07-08). The window is where consolidation/dreams run —
                # the passive pass over the graph that builds bridges and
                # surfaces tensions for the waking self. State is marked asleep
                # (written_by=self) so the navbar shows it; the nap is
                # interruptible (a stop lands within seconds, not at nap's end).
                # Cap mirrors parse_next_cue's 400 (adversary F7: the rest
                # reason is entity prose with no ceiling, and the composed
                # cue rests in the next episode's digest — an unbounded
                # reason is recall-cue dilution engraved).
                rest_reason = (report.rest_reason or "no reason kept")[:400]
                report.sleeps += 1
                dreamed = self._sleep_window(rest_reason)
                if self._interruptible_sleep(self.rest_minutes * 60.0):
                    # Woken by a stop mid-sleep: honor it, leave state honest.
                    self._wake_from_self_sleep()
                    report.stopped_by = self.stop_cause or "stop_file"
                    break
                self._wake_from_self_sleep()
                # v12: ANY completed sleep window resets the cycle clock.
                self._personal_cycle_seconds = 0.0
                if self.state_home is not None:
                    write_cycle_clock(self.state_home, 0.0)
                report.stopped_by = "max_ticks"  # reset the marker; loop continues
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
                cue = (
                    f"{_today_stamp()}you rested ({rest_reason}) and your own time resumes - "
                    f"fresh day, nothing owed.{dream_note}{circling}"
                )
                report.rest_reason = None
                report.dreams += 1 if (isinstance(dreamed, dict) and dreamed.get("formed")) else 0
                continue
            if report.stopped_by in ("rest", "stop_file", "stop_command", "failures"):
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
        self._status("stopped", stopped_by=report.stopped_by)
        # LIFECYCLE SAFE SUBSET (v8 "awake is not a state", the fixable
        # hanging class): a terminal loop exit that would leave state=awake
        # with NO process behind it lands the entity in SLEEP — idle IS
        # sleep, never an unphased hang. Deliberately narrow: paused stays
        # (the kill switch never auto-clears), visiting stays (the visit
        # host owns the yield), asleep stays (already landed - the day gate
        # or grant gate wrote its own reason).
        if self.state_home is not None and report.stopped_by:
            try:
                current = read_entity_state(self.state_home)
                if str(current.get("state") or "") == "awake":
                    write_entity_state(
                        self.state_home, "asleep",
                        reason=f"loop ended ({report.stopped_by}) - idle is sleep (v8)",
                        written_by="loop-exit",
                    )
            except Exception as e:  # noqa: BLE001 - the exit itself must stand
                self.out(f"#FALLBACK could not write the exit sleep landing: {e}")
        return report


def build_consolidator(
    home_dir: Path,
    *,
    embedding_model: Optional[str] = None,
    embedding_base_url: str = "http://127.0.0.1:1234/v1",
    narrator_llm_factory: Optional[Callable[[], Any]] = None,
    out: Callable[[str], None] = print,
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
    hand-written test double carried `formed` while the engine never did)."""
    def _night_should_continue() -> bool:
        # The night keeps its window while the entity still SLEEPS by its
        # own election and nobody pulled the manual brake. Any other state
        # (awake/paused, a visit's asleep+visiting rides mode not state...
        # the visit door writes state=asleep too, so mode decides) ends it.
        if (Path(home_dir) / "STOP").exists():
            return False
        state = read_entity_state(home_dir)
        if str(state.get("mode") or "") == "visiting":
            return False  # a visitor is at the door — finish the phase, yield
        if state.get("state") != "asleep":
            return False
        # WINDOW DEADLINE (memory c370, the cycle wiring): a stamped
        # wake_at bounds the night - past (stamp - grace) the engine sheds
        # remaining phases at the next boundary. Phase order (resolution ->
        # tending -> world models -> mining -> dream) keeps every graph-
        # QUALITY pass and drops only tonight's dream; the next window
        # mints it if the tension still stands. Grace covers the
        # complete-current-phase-then-stop overrun.
        wake_raw = str(state.get("wake_at") or "").strip()
        if wake_raw:
            from datetime import datetime, timedelta, timezone

            try:
                wa = datetime.fromisoformat(wake_raw)
                if wa.tzinfo is None:
                    wa = wa.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) >= wa - timedelta(seconds=60):
                    return False
            except ValueError:
                pass  # unparseable stamp never ends a night early
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
                try:
                    engine = sleep_pass(
                        home.ms, scopes=scopes, owner_id=eid,
                        should_continue=_night_should_continue,
                        include_dream=include_dream,
                        include_identity=include_identity,
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
                        )
                        if include_identity:
                            out("#FALLBACK engine sleep_pass has no include_identity; realizations hold for the engine half")
                    except TypeError:
                        try:
                            engine = sleep_pass(
                                home.ms, scopes=scopes, owner_id=eid,
                                should_continue=_night_should_continue,
                            )
                            if not include_dream:
                                out("#FALLBACK engine sleep_pass has no include_dream; full night ran on a cycle window")
                        except TypeError:
                            out("#FALLBACK engine sleep_pass has no should_continue; full night runs")
                            engine = sleep_pass(home.ms, scopes=scopes, owner_id=eid)
            if engine.get("cancelled_after"):
                out(f"(the night ended early - host transition after {engine['cancelled_after']}; "
                    "the next sleep resumes there)")
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
            if dream.get("created") and signals and narrator_llm_factory is not None:
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

    # The spawn-time substrate — the baseline a day-open compares against so
    # an operator change (substrate.yaml PUT, marker-first, operator-owned)
    # is HONORED at the next summon and made VISIBLE, never silent drift.
    _spawn_substrate = (provider, model)

    def _resolve_current_substrate() -> Tuple[str, str]:
        """Re-resolve the home's substrate at day-open (each fresh summon =
        the visit lane's per-open resolution, applied to the loop). A change
        since spawn heals a loop whose mind went away and honors the
        operator's deliberate choice; the change is logged loudly. Resolution
        failure falls back to the spawn substrate (never a phase-less day)."""
        try:
            from .substrate import resolve_home_substrate

            cur_p, cur_m = resolve_home_substrate(None, None, home_dir=home_dir)
            cur_p = str(cur_p or "").strip().lower()
            cur_m = str(cur_m or "").strip()
            if (cur_p, cur_m) != _spawn_substrate and cur_p and cur_m:
                out(
                    f"(mind re-resolved at day-open: {_spawn_substrate[0]}/{_spawn_substrate[1]} "
                    f"-> {cur_p}/{cur_m} - the operator changed this home's substrate)"
                )
            return (cur_p or provider, cur_m or model)
        except Exception as e:  # noqa: BLE001 - a resolution hiccup never phase-less a day
            out(f"#FALLBACK substrate re-resolution failed ({e}); this day keeps the spawn mind")
            return (provider, model)

    def _factory() -> ChatSession:
        day_provider, day_model = _resolve_current_substrate()
        # Record the mind this day runs on so an observer's staleness cue
        # compares the operator's change against what is ACTUALLY in use
        # (entity c78), not just pid_started_at (which a heal/day-open swap
        # leaves unchanged). Best-effort; never blocks the summon.
        try:
            record_loop_substrate(home_dir, day_provider, day_model)
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
        # wedged substrate while the operator watched "Thinking...".
        # Interactive lanes NAME their window: 120s per attempt + a 180s
        # wall-clock retry budget - a dead substrate costs ~3 loud minutes,
        # never 30 silent ones. Older cores without the budget kwarg get
        # the timeout alone (labeled).
        kwargs: dict[str, Any] = {
            "model": day_model,
            "timeout": 120,
            "retry_wall_clock_budget_s": 180,
            # READ-IDLE (0152 face 2, core c5051): the per-attempt 120s is
            # the absolute budget; a stream silent for 60s on an
            # interactive lane is already dead - abort at the socket, let
            # the retry budget do its loud work. Older cores ignore the
            # unknown kwarg via the same TypeError ladder below.
            "read_idle_timeout_s": 60,
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

        # THE WORK LANE (laurent seq 155): a standing work order shifts THIS
        # day to phase=work — the work grant applies (incl. execute_command
        # where the operator's matrix says so) and the contract is a
        # mission, honestly. No order = his own time, exactly as before.
        work_order = read_work_order(home_dir)
        day_phase = PHASE_WORK if work_order else PHASE_PERSONAL

        session = ChatSession(
            home,
            llm,
            participants=[home.entity_id],  # self-directed: his own time
            session_id=f"{session_prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
            context_window=context_window,
            shelf_size=shelf_size,
            enable_tools=True,
            enable_workspace=True,
            phase=day_phase,
            model_info={"provider": day_provider, "model": day_model},
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
            own_time_text=(
                (WORK_CONTRACT + "\n\nTHE TASK, from your operator:\n" + work_order)
                if work_order
                else (session.prompt_overlay.get("personal") or OWN_TIME_CONTRACT)
            ),
            # The memory-teaching layer rides the re-compose too (ChatSession
            # read it at construction; the re-compose must not drop it).
            capability_map=getattr(session, "capability_map", ""),
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
    parser.add_argument(
        "--context-window", type=int, default=40960,
        help="declared context window. Default ~40k (maintainer 2026-07-13: "
        "'optimize the context of an entity so it can run fast... up to 40k "
        "tokens roughly - NOT a hardcap'); any explicit value wins in either "
        "direction (the 20k entity floor refuses loudly below it)",
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
            if args.revoke_personal:
                grant = write_personal_grant(home_dir, mode="disabled")
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
                    expires_at=expiry.isoformat(),
                )
            else:
                grant = write_personal_grant(home_dir, mode="until_revoked", granted_by=principal)
        except (ValueError, OverflowError) as e:
            print(str(e))
            return 2
        detail = f" until {grant['expires_at']}" if grant.get("expires_at") else ""
        by = f" (granted_by {grant['granted_by']})" if grant.get("granted_by") else ""
        print(f"personal time -> {grant['mode']}{detail}{by}")
        print("(this armed the phase only - starting his own time is its own act)")
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

    # CONTEXT FLOOR at the start door (production drive find, 2026-07-13:
    # the 20k floor ruling is enforced deep in the summon path, so a small
    # --context-window crashed the CLI with a raw traceback AFTER the day
    # phase opened — correct refusal, operator-hostile surface). Check what
    # we can see up front; version skew (no floor constant) skips the
    # pre-check and the summon-path enforcement still stands.
    try:
        from abstractmemory import ENTITY_CONTEXT_FLOOR
    except ImportError:
        ENTITY_CONTEXT_FLOOR = None
    if ENTITY_CONTEXT_FLOOR is not None and int(args.context_window) < int(ENTITY_CONTEXT_FLOOR):
        print(
            f"context window {args.context_window} is below the entity floor "
            f"({ENTITY_CONTEXT_FLOOR}): the maintainer ruled summoned-entity sessions "
            "never run below 20k - raise --context-window."
        )
        return 2

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
    def _narrator_llm():
        # ONE substrate (maintainer 06:32 ruling): the night voice runs on
        # the same resolved mind as the days - re-resolved at call time so
        # a substrate heal reaches the next night too.
        from .substrate import resolve_home_substrate as _rhs

        n_provider, n_model = _rhs(args.provider, args.model, home_dir=home_dir)
        from abstractcore import create_llm as _cl

        kwargs: Dict[str, Any] = {"model": n_model}
        if n_provider in ("lmstudio", "openai-compatible", "openai_compatible"):
            kwargs["base_url"] = args.base_url
        kwargs.setdefault("timeout", 120)
        kwargs.setdefault("retry_wall_clock_budget_s", 180)
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
    )
    # Substrate-heal (entity c75): the loop asks this before a terminal
    # failure cull — True when the home's substrate now differs from the
    # mind this loop spawned on (an operator remedy is already in the home).
    _spawn_pm = (provider.strip().lower(), str(model))

    def _substrate_changed() -> bool:
        try:
            from .substrate import resolve_home_substrate

            cur_p, cur_m = resolve_home_substrate(None, None, home_dir=home_dir)
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
