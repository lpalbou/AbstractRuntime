"""Track record and reputation of an entity or a session (ADR 0052 §5-§6, framework backlog 1119).

Everything here is DERIVED from records the runtime and the home already keep; nothing is written.

Counted runs (ADR §5, ATTACK B6) — what a track record is made of:

- a TURN ROOT (`core.run_attribution.is_turn_root`: parent-less and not an automation controller, or an
  occurrence) whose workflow is not one of the runtime's reserved `__...__` ids (the `__session_memory__`
  owner runs), and
- either an automation OCCURRENCE (role `occurrence`), counted once per occurrence however many attempts it
  took, or a run whose recorded origin (`_meta.origin.kind`, `core.run_origin`) is `person` or `trigger`.

Sub-runs, controllers, owner runs and every run another run or the system started (origin `agent` or
`system`, or no origin) are never counted; their tokens count only as part of the counted run above them.

Outcome signals (one per counted unit; the signal is named on each evidence row):

- `run_completed` / `run_failed`: a counted run's status (cancelled = cancelled, running / waiting = in
  progress; neither is an outcome);
- `occurrence_completed` / `occurrence_failed`: the automation controller's `automation.completed` decision
  for the occurrence (`automations/controller.py` `complete_occurrence`); a cancelled occurrence is
  cancelled; no decision yet = in progress;
- `task_done` (entity homes): a task of `<home>/task_inbox.jsonl` whose folded status is `done`;
- `work_order_done` / `work_order_blocked` (entity homes): a verdict of `<home>/work_order.done.md`;
- `feedback_user` / `feedback_agent`: a judgement of the work by a person or another agent (who, when,
  verdict, task kind — the `OutcomeEvidence` fields). The type has room for them; NO source produces them
  today (operator ruling 2026-11-04: reputation needs external feedback; the intake is backlog 1120).

"Success" therefore means "it completed" (or "the entity said done"), never "it was good": no one has
judged the work. `assessed` (the count of feedback outcomes) stays 0 and `basis` says
"completion-rate only" until feedback lands.

Task kind (ADR §5, ATTACK B4): `<phase or role>/<workflow_id>` — the phase of the run's entity stamp
(`vars._runtime.entity.phase`), else its attribution role (`chat` when it has none), then its
`workflow_id`. Read from ids only; no text of any run is ever read. Tasks and work orders are the work
phase's two home files: `work/task_inbox` and `work/work_order`.

Reputation (ADR §6, ATTACK B5): per task kind `trust = (successes + 1) / (successes + failures + 2)`,
`support = successes + failures` (cancelled and in-progress never count); `overall` is the support-weighted
mean of the kinds; a kind (or a subject) with no outcome reads trust 0.5 with support 0 — "no experience
yet", never trusted. `evidence` is the last 20 outcomes in time order. No recency weighting.

Read cost: one run-index listing (the entity's whole home store, or the session's runs), one run load per
candidate turn root (its origin and stamp live in the run, not in the index), one token query per 300
counted runs and their sub-runs (SQLite; other ledgers are read run by run), one controller-ledger read per
automation with a counted occurrence, and one read of each home file. O(index rows + turn roots + ledger
records of counted runs on non-SQLite stores). Hosts cache by `as_of`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.run_attribution import is_turn_root
from ..core.run_origin import origin_of
from ..session_turns import is_internal_workflow_id
from .fingerprint import ActorFingerprint
from .tool_policy import PHASE_VISIT, PHASE_WORK, canonical_phase

#: How many outcomes a reputation shows as its evidence (ADR §6).
EVIDENCE_WINDOW = 20
#: What "success" means while no run is assessed (ADR §5 / ATTACK B3).
COMPLETION_RATE_ONLY = "completion-rate only"
#: Origins whose runs count (ATTACK B6): a person asked, or an automation's trigger fired.
COUNTED_ORIGIN_KINDS = ("person", "trigger")
#: Every outcome signal. The two feedback kinds are typed but produced by no source yet (backlog 1120).
SIGNALS = (
    "run_completed",
    "run_failed",
    "occurrence_completed",
    "occurrence_failed",
    "task_done",
    "work_order_done",
    "work_order_blocked",
    "feedback_user",
    "feedback_agent",
)
FEEDBACK_SIGNALS = ("feedback_user", "feedback_agent")
TASK_INBOX_KIND = f"{PHASE_WORK}/task_inbox"
WORK_ORDER_KIND = f"{PHASE_WORK}/work_order"

# A listing bound no real store reaches (the same bound `core.run_attribution` uses for whole-session reads).
_INDEX_SCAN_LIMIT = 1_000_000
_METRICS_CHUNK = 300
_TASK_INBOX_FILENAME = "task_inbox.jsonl"


class TrackRecordError(ValueError):
    """A record the track record reads is not in the shape its writer guarantees (fail loud, never guess)."""


# --- types ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenCounts:
    input: int = 0
    output: int = 0
    total: int = 0

    def plus(self, other: "TokenCounts") -> "TokenCounts":
        return TokenCounts(self.input + other.input, self.output + other.output, self.total + other.total)


@dataclass(frozen=True)
class OutcomeEvidence:
    """One success or failure and where it was read: what (`task_kind`, `run_id` — None for a task or a
    work order, which have no run), the verdict (`outcome`), when (`at`), the `signal` it came from, and
    for feedback who gave it (`by`: `<tenant>:<user>` or an agent's id; None for every other signal)."""

    run_id: Optional[str]
    task_kind: str
    outcome: str  # success | failure
    at: str
    signal: str  # one of SIGNALS
    by: Optional[str] = None


@dataclass(frozen=True)
class TaskKindRecord:
    runs: int = 0
    successes: int = 0
    failures: int = 0
    cancelled: int = 0
    tokens_total: int = 0
    first_at: Optional[str] = None
    last_at: Optional[str] = None
    last_outcome_at: Optional[str] = None


@dataclass(frozen=True)
class TrackRecord:
    subject: str  # entity:<slug> | session:<session_id>
    fingerprint: Optional[ActorFingerprint]
    computed_at: str
    as_of: Optional[str]  # the newest timestamp read (runs' updated_at, home-file events)
    since: Optional[str]
    sessions: int
    runs: int
    successes: int
    failures: int
    cancelled: int
    in_progress: int
    assessed: int
    tokens: TokenCounts
    by_task_kind: Mapping[str, TaskKindRecord]
    recent_outcomes: Tuple[OutcomeEvidence, ...]  # the last EVIDENCE_WINDOW outcomes, oldest first
    basis: str = COMPLETION_RATE_ONLY

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["fingerprint"] = self.fingerprint.to_dict() if self.fingerprint is not None else None
        out["by_task_kind"] = {k: asdict(v) for k, v in self.by_task_kind.items()}
        out["recent_outcomes"] = [asdict(e) for e in self.recent_outcomes]
        return out


@dataclass(frozen=True)
class TrustCell:
    trust: float
    support: int
    successes: int
    failures: int
    last_outcome_at: Optional[str]

    def describe(self) -> str:
        """Plain words, always with the support: "trust 0.71 over 5 outcomes" / "no experience yet"."""
        if self.support == 0:
            return "no experience yet"
        noun = "outcome" if self.support == 1 else "outcomes"
        return f"trust {self.trust:.2f} over {self.support} {noun}"


@dataclass(frozen=True)
class Reputation:
    subject: str
    computed_at: str
    overall: TrustCell
    by_task_kind: Mapping[str, TrustCell]
    evidence: Tuple[OutcomeEvidence, ...]
    basis: str = COMPLETION_RATE_ONLY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "computed_at": self.computed_at,
            "overall": {**asdict(self.overall), "text": self.overall.describe()},
            "by_task_kind": {k: {**asdict(v), "text": v.describe()} for k, v in self.by_task_kind.items()},
            "evidence": [asdict(e) for e in self.evidence],
            "basis": self.basis,
        }


# --- rules ---------------------------------------------------------------------------------------------


def trust_of(successes: int, failures: int) -> float:
    """The rule of succession: (successes + 1) / (successes + failures + 2). 0/0 -> 0.5."""
    return (successes + 1) / (successes + failures + 2)


def task_kind_of(run: Any, *, role: Any) -> str:
    """`<phase or role>/<workflow_id>` (ADR §5). `role` is the run's attribution role from the index.

    The phase is the run's entity stamp phase (a stamp minted before phases existed is a visit, as the
    gateway resolves it); a run without a stamp uses its role, `chat` when it has none. Ids only."""
    vars_obj = getattr(run, "vars", None)
    runtime_ns = vars_obj.get("_runtime") if isinstance(vars_obj, Mapping) else None
    stamp = runtime_ns.get("entity") if isinstance(runtime_ns, Mapping) else None
    if isinstance(stamp, Mapping):
        raw = stamp.get("phase")
        if raw is None or not str(raw).strip():
            head = PHASE_VISIT
        else:
            try:
                head = canonical_phase(str(raw))
            except ValueError as exc:
                raise TrackRecordError(f"run {getattr(run, 'run_id', '?')}: entity stamp phase {raw!r}: {exc}") from exc
    else:
        head = str(role) if role else "chat"
    workflow_id = str(getattr(run, "workflow_id", "") or "").strip()
    if not workflow_id:
        raise TrackRecordError(f"run {getattr(run, 'run_id', '?')} has no workflow_id")
    return f"{head}/{workflow_id}"


def is_counted_candidate(row: Mapping[str, Any]) -> bool:
    """Index-only half of the counted-run rule: a turn root that is not a reserved runtime run."""
    return is_turn_root(parent_run_id=row.get("parent_run_id"), role=row.get("role")) and not is_internal_workflow_id(
        row.get("workflow_id")
    )


def has_counted_origin(run: Any) -> bool:
    """Run half of the counted-run rule for non-occurrences: origin person or trigger."""
    origin = origin_of(getattr(run, "vars", None))
    return origin is not None and origin["kind"] in COUNTED_ORIGIN_KINDS


def entity_fingerprint(manifest: Mapping[str, Any]) -> ActorFingerprint:
    """ADR §4: `kind="entity"`, the slug as display name, `{entity_id, home_id, spark_hash}` as metadata."""
    missing = [k for k in ("slug", "entity_id", "home_id", "spark_hash") if not str(manifest.get(k) or "").strip()]
    if missing:
        raise TrackRecordError(f"manifest lacks {', '.join(missing)}")
    return ActorFingerprint.from_metadata(
        kind="entity",
        display_name=str(manifest["slug"]),
        metadata={k: str(manifest[k]) for k in ("entity_id", "home_id", "spark_hash")},
    )


def _innermost(store: Any) -> Any:
    cur = store
    for _ in range(8):
        inner = getattr(cur, "_inner", None)
        if inner is None:
            return cur
        cur = inner
    return cur


def _usage_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def token_counts_of_records(records: Iterable[Any]) -> TokenCounts:
    """The SQLite `metrics_many` token rules over plain ledger records: completed `llm_call` records;
    input = `usage.input_tokens` else `prompt_tokens`, output = `output_tokens` else `completion_tokens`,
    total = `total_tokens`."""
    tin = tout = ttot = 0
    for rec in records or []:
        # In-memory ledgers keep the enum; persisted ones its value.
        if not isinstance(rec, Mapping) or str(getattr(rec.get("status"), "value", rec.get("status")) or "") != "completed":
            continue
        effect = rec.get("effect") if isinstance(rec.get("effect"), Mapping) else {}
        if str(effect.get("type") or "") != "llm_call":
            continue
        result = rec.get("result") if isinstance(rec.get("result"), Mapping) else {}
        usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
        i = _usage_int(usage.get("input_tokens"))
        o = _usage_int(usage.get("output_tokens"))
        tin += i if i is not None else (_usage_int(usage.get("prompt_tokens")) or 0)
        tout += o if o is not None else (_usage_int(usage.get("completion_tokens")) or 0)
        ttot += _usage_int(usage.get("total_tokens")) or 0
    return TokenCounts(tin, tout, ttot)


def ledger_token_counts(ledger_store: Any, run_ids: Sequence[str]) -> Dict[str, TokenCounts]:
    """Per-run tokens: the SQLite ledger's `metrics_many` query, else each run's records."""
    from ..storage.sqlite import SqliteLedgerStore

    ids = sorted({str(r) for r in run_ids if str(r or "").strip()})
    if not ids:
        return {}
    inner = _innermost(ledger_store)
    out: Dict[str, TokenCounts] = {}
    if isinstance(inner, SqliteLedgerStore):
        for i in range(0, len(ids), _METRICS_CHUNK):
            metrics = inner.metrics_many(ids[i : i + _METRICS_CHUNK])
            for rid, m in metrics.items():
                out[rid] = TokenCounts(int(m["tokens_input"]), int(m["tokens_output"]), int(m["tokens_total"]))
        return out
    for rid in ids:
        out[rid] = token_counts_of_records(ledger_store.list(rid))
    return out


# --- the fold ------------------------------------------------------------------------------------------


@dataclass
class _Unit:
    """One counted thing: a run, an occurrence (all its attempts), a task or a work order."""

    run_id: Optional[str]
    session_id: Optional[str]
    task_kind: str
    state: str  # success | failure | cancelled | in_progress
    signal: str  # the source: run | occurrence | task | work_order
    first_at: Optional[str]
    last_at: Optional[str]
    outcome_at: Optional[str]
    token_run_ids: List[str] = field(default_factory=list)


# (source, outcome) -> the evidence signal kind.
_SIGNAL_OF = {
    ("run", "success"): "run_completed",
    ("run", "failure"): "run_failed",
    ("occurrence", "success"): "occurrence_completed",
    ("occurrence", "failure"): "occurrence_failed",
    ("task", "success"): "task_done",
    ("work_order", "success"): "work_order_done",
    ("work_order", "failure"): "work_order_blocked",
}
_RUN_STATES = {"completed": "success", "failed": "failure", "cancelled": "cancelled", "running": "in_progress", "waiting": "in_progress"}
_OCCURRENCE_STATES = {"completed": "success", "failed": "failure", "cancelled": "cancelled"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _since_ok(at: Optional[str], since: Optional[str]) -> bool:
    return since is None or (at is not None and str(at) >= since)


def _descendants(roots: Sequence[str], children: Mapping[str, List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    stack = list(roots)
    while stack:
        rid = stack.pop()
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
        stack.extend(children.get(rid, ()))
    return out


def _load(run_store: Any, run_id: str) -> Any:
    run = run_store.load(run_id)
    if run is None:
        raise TrackRecordError(f"run {run_id} is in the run index but cannot be loaded")
    return run


def _occurrence_decisions(ledger_store: Any, automation_id: str) -> Dict[int, Dict[str, Any]]:
    """`{occurrence index: automation.completed payload}` of one automation's controller ledger."""
    from ..automations.ledger import automation_records

    out: Dict[int, Dict[str, Any]] = {}
    for rec in automation_records(ledger_store, automation_id, "automation.completed"):
        payload = rec["payload"]
        index = payload.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            out[index] = payload
    return out


def _run_units(run_store: Any, ledger_store: Any, rows: List[Dict[str, Any]], since: Optional[str]) -> List[_Unit]:
    children: Dict[str, List[str]] = {}
    for row in rows:
        parent = str(row.get("parent_run_id") or "").strip()
        if parent:
            children.setdefault(parent, []).append(str(row["run_id"]))

    units: List[_Unit] = []
    occurrence_groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (str(r.get("created_at") or ""), str(r.get("run_id") or ""))):
        if not is_counted_candidate(row) or not _since_ok(row.get("created_at"), since):
            continue
        rid = str(row["run_id"])
        if row.get("role") == "occurrence":
            index = row.get("occurrence_index")
            key = (str(row.get("automation_id") or ""), index if isinstance(index, int) else rid)
            occurrence_groups.setdefault(key, []).append(row)
            continue
        run = _load(run_store, rid)
        if not has_counted_origin(run):
            continue
        status = str(row.get("status") or "")
        if status not in _RUN_STATES:
            raise TrackRecordError(f"run {rid} has an unknown status {status!r}")
        state = _RUN_STATES[status]
        units.append(
            _Unit(
                run_id=rid,
                session_id=row.get("session_id"),
                task_kind=task_kind_of(run, role=row.get("role")),
                state=state,
                signal="run",
                first_at=row.get("created_at"),
                last_at=row.get("updated_at"),
                outcome_at=row.get("updated_at") if state in ("success", "failure") else None,
                token_run_ids=_descendants([rid], children),
            )
        )

    decisions_by_automation: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for (automation_id, index), attempts in sorted(occurrence_groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        decision = None
        if automation_id and isinstance(index, int):
            if automation_id not in decisions_by_automation:
                decisions_by_automation[automation_id] = _occurrence_decisions(ledger_store, automation_id)
            decision = decisions_by_automation[automation_id].get(index)
        by_id = {str(r["run_id"]): r for r in attempts}
        if decision is not None:
            decided_id = str(decision.get("run_id") or "")
            if decided_id not in by_id:
                raise TrackRecordError(
                    f"automation {automation_id} occurrence {index}: its decision names run {decided_id!r}, "
                    "which is not one of the occurrence's runs"
                )
            status = str(decision.get("status") or "")
            if status not in _OCCURRENCE_STATES:
                raise TrackRecordError(f"automation {automation_id} occurrence {index}: unknown status {status!r}")
            state = _OCCURRENCE_STATES[status]
            shown = by_id[decided_id]
            outcome_at = str(decision.get("finished_at") or "") or None
        else:
            state = "in_progress"
            shown = attempts[-1]  # the newest attempt (created_at order)
            outcome_at = None
        run = _load(run_store, str(shown["run_id"]))
        units.append(
            _Unit(
                run_id=str(shown["run_id"]),
                session_id=shown.get("session_id"),
                task_kind=task_kind_of(run, role="occurrence"),
                state=state,
                signal="occurrence",
                first_at=attempts[0].get("created_at"),
                last_at=max(str(a.get("updated_at") or "") for a in attempts) or None,
                outcome_at=outcome_at if state in ("success", "failure") else None,
                token_run_ids=_descendants([str(a["run_id"]) for a in attempts], children),
            )
        )
    return units


def _task_units(home_dir: Path, since: Optional[str]) -> List[_Unit]:
    """Tasks of `<home>/task_inbox.jsonl` (written by AbstractGateway's `entity_tasks.py`) whose folded
    status is `done`. Each line is `{"event": "added"|"status", "task_id", "at", "status"?}`; the newest
    status event of a task is its status. A line that is not JSON is the one torn tail the writer allows
    (a crash mid-append) when it is the LAST line; anywhere else it fails loudly."""
    path = Path(home_dir) / _TASK_INBOX_FILENAME
    if not path.is_file():
        return []
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    added: Dict[str, str] = {}
    status: Dict[str, Tuple[str, str]] = {}
    for n, raw in enumerate(raw_lines):
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError as exc:
            if n == len(raw_lines) - 1:
                break
            raise TrackRecordError(f"{path}: line {n + 1} is not JSON ({exc})") from exc
        if not isinstance(ev, dict) or not ev.get("task_id"):
            raise TrackRecordError(f"{path}: line {n + 1} is not a task event")
        tid = str(ev["task_id"])
        if ev.get("event") == "added":
            added.setdefault(tid, str(ev.get("at") or ""))
        elif ev.get("event") == "status" and tid in added:
            status[tid] = (str(ev.get("status") or ""), str(ev.get("at") or ""))
    units = []
    for tid, (word, at) in status.items():
        if word != "done" or not _since_ok(at, since):
            continue
        units.append(_Unit(None, None, TASK_INBOX_KIND, "success", "task", added[tid] or at, at, at))
    return units


def _work_order_units(home_dir: Path, since: Optional[str]) -> List[_Unit]:
    from .life import read_work_order_verdicts

    units = []
    for v in read_work_order_verdicts(Path(home_dir)):
        if not _since_ok(v["at"], since):
            continue
        state = "success" if v["verdict"] == "done" else "failure"
        units.append(_Unit(None, None, WORK_ORDER_KIND, state, "work_order", v["at"], v["at"], v["at"]))
    return units


def _fold(
    *,
    subject: str,
    fingerprint: Optional[ActorFingerprint],
    rows: List[Dict[str, Any]],
    units: List[_Unit],
    ledger_store: Any,
    since: Optional[str],
) -> TrackRecord:
    token_ids = [rid for u in units for rid in u.token_run_ids]
    per_run = ledger_token_counts(ledger_store, token_ids)

    kinds: Dict[str, Dict[str, Any]] = {}
    total_tokens = TokenCounts()
    outcomes: List[OutcomeEvidence] = []
    counts = {"success": 0, "failure": 0, "cancelled": 0, "in_progress": 0}
    for u in units:
        tokens = TokenCounts()
        for rid in u.token_run_ids:
            tokens = tokens.plus(per_run.get(rid, TokenCounts()))
        total_tokens = total_tokens.plus(tokens)
        counts[u.state] += 1
        k = kinds.setdefault(
            u.task_kind,
            {"runs": 0, "successes": 0, "failures": 0, "cancelled": 0, "tokens_total": 0, "first_at": None, "last_at": None, "last_outcome_at": None},
        )
        k["runs"] += 1 if u.run_id is not None else 0
        k["successes"] += u.state == "success"
        k["failures"] += u.state == "failure"
        k["cancelled"] += u.state == "cancelled"
        k["tokens_total"] += tokens.total
        if u.first_at and (k["first_at"] is None or u.first_at < k["first_at"]):
            k["first_at"] = u.first_at
        if u.last_at and (k["last_at"] is None or u.last_at > k["last_at"]):
            k["last_at"] = u.last_at
        if u.state in ("success", "failure"):
            if u.outcome_at is None:
                raise TrackRecordError(f"{u.signal} outcome of {u.run_id or u.task_kind} has no time")
            if k["last_outcome_at"] is None or u.outcome_at > k["last_outcome_at"]:
                k["last_outcome_at"] = u.outcome_at
            outcomes.append(OutcomeEvidence(u.run_id, u.task_kind, u.state, u.outcome_at, _SIGNAL_OF[(u.signal, u.state)]))

    outcomes.sort(key=lambda e: (e.at, e.signal, e.run_id or "", e.task_kind))
    stamps = [str(r.get("updated_at") or "") for r in rows] + [str(u.last_at or "") for u in units]
    stamps = [s for s in stamps if s]
    return TrackRecord(
        subject=subject,
        fingerprint=fingerprint,
        computed_at=_utc_now_iso(),
        as_of=max(stamps) if stamps else None,
        since=since,
        sessions=len({u.session_id for u in units if u.run_id is not None and u.session_id}),
        runs=sum(1 for u in units if u.run_id is not None),
        successes=counts["success"],
        failures=counts["failure"],
        cancelled=counts["cancelled"],
        in_progress=counts["in_progress"],
        assessed=sum(1 for e in outcomes if e.signal in FEEDBACK_SIGNALS),
        tokens=total_tokens,
        by_task_kind={name: TaskKindRecord(**kinds[name]) for name in sorted(kinds)},
        recent_outcomes=tuple(outcomes[-EVIDENCE_WINDOW:]),
    )


def _index_rows(run_store: Any, *, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    lister = getattr(run_store, "list_run_index", None)
    if not callable(lister):
        raise TrackRecordError(f"run store {type(run_store).__name__} has no run index (list_run_index)")
    rows = list(lister(session_id=session_id, limit=_INDEX_SCAN_LIMIT))
    if len(rows) >= _INDEX_SCAN_LIMIT:
        raise TrackRecordError(f"run index listing reached its bound ({_INDEX_SCAN_LIMIT} rows)")
    return rows


def entity_track_record(home_dir: Any, run_store: Any, ledger_store: Any, *, since: Optional[str] = None) -> TrackRecord:
    """An entity's track record: its home's run store and ledger (`identity.entity_runtime`), its task inbox
    and its work-order archive. Subject `entity:<slug>`; fingerprint from `<home>/manifest.json` (ADR §4).
    `since` (ISO-8601, compared as text like every stored time) keeps what started at or after it."""
    home = Path(home_dir)
    manifest_path = home / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackRecordError(f"{manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TrackRecordError(f"{manifest_path} is not a JSON object")
    fingerprint = entity_fingerprint(manifest)
    rows = _index_rows(run_store)
    units = _run_units(run_store, ledger_store, rows, since) + _task_units(home, since) + _work_order_units(home, since)
    return _fold(
        subject=f"entity:{manifest['slug']}",
        fingerprint=fingerprint,
        rows=rows,
        units=units,
        ledger_store=ledger_store,
        since=since,
    )


def session_track_record(session_id: str, run_store: Any, ledger_store: Any, *, since: Optional[str] = None) -> TrackRecord:
    """A regular agent's track record: the same rules over one session's runs (ADR §5). Subject
    `session:<session_id>`. No fingerprint: a session's agent fingerprint is abstractagent's own. A sub-run
    started into ANOTHER session is outside this listing, so its tokens are not part of this record."""
    sid = str(session_id or "").strip()
    if not sid:
        raise TrackRecordError("session_id is required")
    rows = _index_rows(run_store, session_id=sid)
    units = _run_units(run_store, ledger_store, rows, since)
    return _fold(subject=f"session:{sid}", fingerprint=None, rows=rows, units=units, ledger_store=ledger_store, since=since)


def compute_reputation(record: TrackRecord) -> Reputation:
    """Trust per task kind from the record's counts (ADR §6); pure — the same record gives the same answer."""
    cells: Dict[str, TrustCell] = {}
    for name, k in record.by_task_kind.items():
        cells[name] = TrustCell(
            trust=trust_of(k.successes, k.failures),
            support=k.successes + k.failures,
            successes=k.successes,
            failures=k.failures,
            last_outcome_at=k.last_outcome_at,
        )
    support = sum(c.support for c in cells.values())
    if support:
        overall_trust = sum(c.trust * c.support for c in cells.values()) / support
    else:
        overall_trust = trust_of(0, 0)
    stamps = [c.last_outcome_at for c in cells.values() if c.last_outcome_at]
    overall = TrustCell(
        trust=overall_trust,
        support=support,
        successes=sum(c.successes for c in cells.values()),
        failures=sum(c.failures for c in cells.values()),
        last_outcome_at=max(stamps) if stamps else None,
    )
    return Reputation(
        subject=record.subject,
        computed_at=record.computed_at,
        overall=overall,
        by_task_kind=dict(cells),
        evidence=record.recent_outcomes,
        basis=record.basis,
    )


__all__ = [
    "COMPLETION_RATE_ONLY",
    "COUNTED_ORIGIN_KINDS",
    "EVIDENCE_WINDOW",
    "FEEDBACK_SIGNALS",
    "OutcomeEvidence",
    "Reputation",
    "SIGNALS",
    "TaskKindRecord",
    "TokenCounts",
    "TrackRecord",
    "TrackRecordError",
    "TrustCell",
    "compute_reputation",
    "entity_fingerprint",
    "entity_track_record",
    "has_counted_origin",
    "is_counted_candidate",
    "ledger_token_counts",
    "session_track_record",
    "task_kind_of",
    "token_counts_of_records",
    "trust_of",
]
