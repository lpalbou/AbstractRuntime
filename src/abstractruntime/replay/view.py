"""The replay reducer: one deterministic fold of a run tree's ledger into a view (ADR-0009).

`replay(records_by_run, runs_by_id)` folds the ledger records of a run tree (the root run and
the runs it delegated to) plus their durable snapshots into a `ReplayView`; `apply(view,
records)` folds further records into an existing view and returns the same view, whose
`last_delta` names what changed. Both are the same code path: `replay` is `apply` over every
record, so `replay(all) == fold(apply)` holds by construction and is pinned by tests.

Decisions this module owns (framework backlog 0993; each was a divergence between clients):

- Run tree: a run's children are found through its delegation waits (`subworkflow:<id>`
  wait keys, `sub_run_id` in a wait's details or a `start_subworkflow` result) and through the
  snapshots' `parent_run_id`.
- A wait is over when its run records a `resume` for its wait key OR any later step of the
  same run exists (a timer wait often has no resume record; the runtime also appends the
  resume before it saves the run without its wait, so a snapshot read in between still says
  "waiting"). Observability events (`abstract.status`, `abstract.progress`) are not steps.
- The person-facing interaction is the newest open tool approval or question anywhere in the
  tree; a delegation wait (`subworkflow:`) or a timer is never one. A snapshot's wait is
  adopted only when the run's ledger never mentioned that wait key (a windowed history).
- Status: a terminal snapshot wins; then the ledger's terminal record (`system:completion:*`,
  `system:terminal_status:<status>`, a run-level failure); then an open wait; then running.
- `$slim` dedup pointers are resolved here against the STARTED record of the same step, so a
  client never sees one; a pointer that cannot be resolved is dropped and named in
  `warnings`.
- Executions (one activity per node execution) follow `executions.ExecutionTracker`.

The view is JSON (`to_json()`), versioned `replay_view_v1`. Records carry their per-run `seq`
(the durable cursor, `LedgerStore.list` / `list_after`); a record without one is numbered
after the previous record of its run. A record at or below a run's folded seq is ignored, so
folding is idempotent.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from ..automations.attention import ANSWER_PAYLOADS
from ..history_bundle import _extract_usage_from_ledger_record, _extract_user_prompt_from_input
from ..storage.ledger_slim import is_slim_marker, resolve_slim_value
from .delta import token_run_names
from .executions import ExecutionTracker, is_completion_record, is_progress_record

REPLAY_VIEW_VERSION = "replay_view_v1"

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# The STARTED-payload fields a view reads (and so the only ones a `$slim` pointer is resolved
# for). Conversation-sized fields (`messages`, `system_prompt`) are never read by the view.
_VIEW_PAYLOAD_FIELDS = ("tool_calls", "prompt", "message", "text", "content", "provider", "model", "workflow_id")
_TEXT_KEYS = ("answer", "response", "message", "text", "content", "output")
_OBSERVABILITY_EVENTS = frozenset({"abstract.status", "abstract.progress", "abstractcode.status"})
_MAX_WALK_DEPTH = 12


# --------------------------------------------------------------------------- helpers


def _s(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _enum(value: Any) -> str:
    return _s(getattr(value, "value", value)).lower()


def _obj(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_of(value: Any) -> str:
    """The display text of a value: itself when a string, else the first string under the usual keys."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict) and key == "output":
                t = _text_of(v)
                if t:
                    return t
    return ""


def _output_text(output: Any) -> str:
    """A run output's display text. `{result: {...}}` beside other keys (`success`, ...) and no
    `response`/`answer`/`notify` of its own is read through `result`: the runtime's own rule
    (`automations.attention.normalize_occurrence_output`)."""
    if isinstance(output, dict) and isinstance(output.get("result"), dict) and not any(
            k in output for k in ("response", "answer", "notify")):
        return _text_of(output["result"]) or _text_of(output)
    return _text_of(output)


def _artifact_refs(value: Any, path: str = "", depth: int = 0) -> List[Tuple[str, str]]:
    """Every `{"$artifact": id}` reference under `value` as (artifact_id, json path)."""
    if depth > _MAX_WALK_DEPTH:
        return []
    out: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        ref = value.get("$artifact")
        if isinstance(ref, str) and ref.strip():
            out.append((ref.strip(), path or "$"))
            return out
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                out.extend(_artifact_refs(v, f"{path}.{k}" if path else str(k), depth + 1))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, (dict, list)):
                out.extend(_artifact_refs(v, f"{path}[{i}]", depth + 1))
    return out


def _snapshot(run: Any) -> Dict[str, Any]:
    """A run snapshot (RunState or its dict) as the plain fields the view reads."""
    if run is None:
        return {}
    if isinstance(run, dict):
        d = run
    else:
        d = {k: getattr(run, k, None) for k in ("run_id", "workflow_id", "status", "parent_run_id", "waiting",
                                                 "vars", "output", "created_at", "updated_at", "error", "session_id")}
    waiting = d.get("waiting")
    if waiting is not None and not isinstance(waiting, dict):
        waiting = {k: getattr(waiting, k, None) for k in ("reason", "wait_key", "until", "resume_to_node",
                                                           "result_key", "prompt", "choices", "allow_free_text",
                                                           "details")}
    if isinstance(waiting, dict):
        waiting = copy.deepcopy(dict(waiting))
        waiting["reason"] = _enum(waiting.get("reason"))
    source = d.get("input_data") if isinstance(d.get("input_data"), dict) else d.get("vars")
    prompt = _extract_user_prompt_from_input(source)[0] if isinstance(source, dict) else ""
    out = d.get("output")
    # Only what the view reads is kept (a snapshot's vars can be conversation-sized).
    return {
        "run_id": _s(d.get("run_id")),
        "workflow_id": _s(d.get("workflow_id")),
        "status": _enum(d.get("status")),
        "session_id": _s(d.get("session_id")) or None,
        "parent_run_id": _s(d.get("parent_run_id")) or None,
        "waiting": waiting if isinstance(waiting, dict) else None,
        "prompt": prompt,
        "output_text": _output_text(out),
        "output_artifacts": [a for a, _ in _artifact_refs(out)],
        "created_at": _s(d.get("created_at")),
        "updated_at": _s(d.get("updated_at")),
    }


def _child_from_wait(wait: Dict[str, Any]) -> str:
    key = _s(wait.get("wait_key"))
    if key.startswith("subworkflow:"):
        return key[len("subworkflow:"):].strip()
    details = _obj(wait.get("details"))
    for k in ("sub_run_id", "child_run_id", "subRunId"):
        if _s(details.get(k)):
            return _s(details.get(k))
    return ""


def workflow_ref(workflow_id: Any) -> Optional[Dict[str, Any]]:
    """What a run executes, named: `{id, bundle_id, bundle_version, flow_id}`.

    Workflows are the framework's native unit (agents, automations and trigger responses are
    workflows). A bundled workflow's id is `bundle_id@version:flow_id` (the grammar of
    `WorkflowBundleRegistry.resolve_entrypoint`; `bundle_id:flow_id` without a version is
    accepted too); any other id (a visual flow, a host-registered workflow) is its own flow
    with no bundle."""
    wid = _s(workflow_id)
    if not wid:
        return None
    bundle_id: Optional[str] = None
    version: Optional[str] = None
    flow_id = wid
    left, sep, right = wid.partition(":")
    if sep and left.strip() and right.strip() and "/" not in left and "\\" not in left:
        bundle, at, ver = left.partition("@")
        bundle_id = bundle.strip() or None
        version = ver.strip() or None if at else None
        flow_id = right.strip()
    return {"id": wid, "bundle_id": bundle_id, "bundle_version": version, "flow_id": flow_id}


def wait_kind(wait: Dict[str, Any], effect_type: str = "", run_id: str = "") -> str:
    """tool_approval | ask_user | event | delegation | timer | job | pause | automation_wake | signal

    Agrees with the runtime's attention classifier (`automations.attention.is_interactive_wait` /
    `wait_kind`, which the gateway's attention stream uses), so a view and the attention stream
    never disagree on whether a person is asked something:
    - `tool_approval`: `details.mode == "approval_required"` (also `details.kind` tool_approval or
      a `tool_approval:` key);
    - a USER wait is `ask_user`, except a pause (`pause:<run id>` key or `details.kind == "pause"`):
      `pause`;
    - an automation controller's wake (`automation:<id>:wake`) is `automation_wake`;
    - an EVENT wait with a prompt or choices is `event` (a person answers it); without one it is
      `signal` (a machine wakes it);
    - `delegation` (`subworkflow:`), `timer` (UNTIL), `job` (JOB).
    Person-facing: tool_approval, ask_user, event (`PERSON_FACING_KINDS`)."""
    details = _obj(wait.get("details"))
    reason = _enum(wait.get("reason"))
    key = _s(wait.get("wait_key"))
    if _s(details.get("mode")) == "approval_required" or _s(details.get("kind")) == "tool_approval" or key.startswith("tool_approval:"):
        return "tool_approval"
    if reason == "subworkflow" or key.startswith("subworkflow:"):
        return "delegation"
    if key.startswith("automation:") and key.endswith(":wake"):
        return "automation_wake"
    if reason == "until":
        return "timer"
    if reason == "job":
        return "job"
    if reason == "user" or effect_type == "ask_user" or (not reason and _s(details.get("kind")) in ("ask_user", "ask")):
        if _s(details.get("kind")) == "pause" or key == f"pause:{run_id}" or (not run_id and key.startswith("pause:")):
            return "pause"
        return "ask_user"
    if reason == "event" and (wait.get("prompt") or wait.get("choices")):
        return "event"
    return "signal"


PERSON_FACING_KINDS = frozenset({"tool_approval", "ask_user", "event"})


def _tool_call_id(call: Dict[str, Any], index: int) -> str:
    for k in ("runtime_call_id", "call_uid"):
        if _s(call.get(k)):
            return _s(call.get(k))
    for k in ("call_id", "id"):
        if _s(call.get(k)):
            return _s(call.get(k))
    return f"#{index}"


def _match_results(calls: List[Dict[str, Any]], results: List[Any]) -> Dict[int, Dict[str, Any]]:
    """Call index -> its result: runtime_call_id/call_uid, then a unique call_id/id, then position."""
    out: Dict[int, Dict[str, Any]] = {}
    rows = [r for r in results if isinstance(r, dict)]
    used: Set[int] = set()
    for ci, call in enumerate(calls):
        for key_names in (("runtime_call_id", "call_uid"), ("call_id", "id")):
            want = next((_s(call.get(k)) for k in key_names if _s(call.get(k))), "")
            if not want:
                continue
            hits = [ri for ri, r in enumerate(rows) if ri not in used and want in (_s(r.get(k)) for k in key_names)]
            if len(hits) == 1:
                out[ci] = rows[hits[0]]
                used.add(hits[0])
                break
    if len(out) < len(calls) and len(rows) == len(calls):
        for ci in range(len(calls)):
            if ci not in out and ci not in used:
                out[ci] = rows[ci]
    return out


# The runtime's denial: `Runtime.resume` turns `{"approved": false}` on a tool-approval wait into
# `{"mode": "executed", "results": [{success: false, output: None, error: <reason>}]}` (default
# reason "Denied by user", core/runtime.py) and THAT is what the ledger records (the approval
# step's results and the resume payload); `approved: false` itself is never written. Only the
# default reason is recognisable: a custom deny reason reads as a failed call (follow-up: the
# runtime should stamp `denied: true` on those rows).
_RUNTIME_DENIAL_ERROR = "denied by user"


def _is_denial(out: Any) -> bool:
    if not isinstance(out, dict) or out.get("success") is not False:
        return False
    return out.get("output") is None and _s(out.get("error")).lower() == _RUNTIME_DENIAL_ERROR


# --------------------------------------------------------------------------- the view


class _Run:
    __slots__ = ("run_id", "parent_run_id", "workflow_id", "snapshot", "last_seq", "first_seq", "record_count",
                 "ledger_terminal", "open_wait", "seen_wait_keys", "resolved_wait_keys", "display_status",
                 "discover_key", "first_ts")

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.parent_run_id: Optional[str] = None
        self.workflow_id = ""
        self.snapshot: Dict[str, Any] = {}
        self.last_seq = 0
        self.first_seq = 0
        self.record_count = 0
        self.ledger_terminal: Optional[str] = None
        self.open_wait: Optional[Dict[str, Any]] = None
        self.seen_wait_keys: Set[str] = set()
        self.resolved_wait_keys: Set[str] = set()
        self.display_status = ""
        self.discover_key: Tuple[str, str, int] = ("", "", 0)
        self.first_ts = ""


class ReplayView:
    """A run tree's replay: fold records with `apply`, read it with `to_json`.

    Not thread-safe; a host serialises `apply` per view (the gateway holds one lock per cached
    view). `apply` returns the delta of that call; `to_json` is the whole view.
    """

    def __init__(self, root_run_id: str) -> None:
        self.root_run_id = _s(root_run_id)
        self._runs: Dict[str, _Run] = {}
        self._tracker = ExecutionTracker()
        self._started: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._activities: Dict[str, Dict[str, Any]] = {}
        self._messages: Dict[str, Dict[str, Any]] = {}
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._usage_steps: Set[Tuple[str, str]] = set()
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        self._warnings: Dict[str, str] = {}
        self._wait_activity: Dict[Tuple[str, str], str] = {}
        self._dirty: Dict[str, Set[str]] = {"runs": set(), "activities": set(), "messages": set(), "artifacts": set()}
        self._interaction: Optional[Dict[str, Any]] = None
        self._statuses: Dict[str, str] = {}  # run id -> the status the last delta reported
        self._pending_changes: List[Dict[str, Any]] = []  # interaction opened/closed since the last delta
        self.last_delta: Optional[Dict[str, Any]] = None
        self._run(self.root_run_id)

    # ----------------------------------------------------------------- public API

    def apply(self, records: Iterable[Dict[str, Any]] = (), *, runs_by_id: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Fold `records` (each run's in its ledger order) and any newer snapshots; return the delta."""
        for snap in (runs_by_id or {}).values():
            self._fold_snapshot(_snapshot(snap))
        for rec in records or ():
            if isinstance(rec, dict):
                self._fold_record(rec)
        self._finalize()
        delta = self._take_delta()
        self.last_delta = delta
        return delta

    @property
    def cursor(self) -> Dict[str, Any]:
        runs = {rid: r.last_seq for rid, r in self._runs.items() if r.last_seq > 0 or rid == self.root_run_id}
        order = self._run_order()
        # One header-safe token (an ETag, an SSE id): no comma, no quote, no space. Each run is
        # named by the shortest prefix (8 characters at least) that no other run of the tree
        # shares; `resolve_cursor_token` maps it back.
        names = token_run_names(order)
        token = ".".join(f"{names[rid]}:{self._runs[rid].last_seq}" for rid in order)
        return {"token": token, "records": sum(r.record_count for r in self._runs.values()), "runs": runs}

    @property
    def run_ids(self) -> List[str]:
        """The tree's runs, root first (the runs whose ledgers this view folds)."""
        return self._run_order()

    def last_seq(self, run_id: str) -> int:
        r = self._runs.get(_s(run_id))
        return r.last_seq if r is not None else 0

    def status(self) -> str:
        return self._run_status(self._runs[self.root_run_id])

    def run_statuses(self) -> Dict[str, str]:
        """run id -> its derived status (no serialization)."""
        return {rid: self._run_status(r) for rid, r in self._runs.items()}

    def is_terminal(self) -> bool:
        return self.status() in TERMINAL_STATUSES

    def to_json(self) -> Dict[str, Any]:
        order = self._run_order()
        root = self._runs[self.root_run_id]
        return {
            "version": REPLAY_VIEW_VERSION,
            "root_run_id": self.root_run_id,
            "workflow": self._root_workflow(),
            "session_id": root.snapshot.get("session_id"),
            "status": self._run_status(root),
            "display_status": root.display_status or None,
            "cursor": self.cursor,
            "runs": [self._run_row(rid, order) for rid in order],
            "interaction": copy.deepcopy(self._interaction),
            "open_waits": self._open_waits(),
            "messages": _sorted_items(self._messages),
            "activities": _sorted_items(self._activities),
            "artifacts": _sorted_items(self._artifacts),
            "usage": dict(self._usage),
            "warnings": [self._warnings[k] for k in sorted(self._warnings)],
        }

    # ----------------------------------------------------------------- runs

    def _root_workflow(self) -> Optional[Dict[str, Any]]:
        """The root run's workflow, with `entrypoint` (the flow the run was started at)."""
        ref = workflow_ref(self._runs[self.root_run_id].workflow_id)
        if ref is not None:
            ref["entrypoint"] = ref["flow_id"]
        return ref

    def _run(self, run_id: str) -> _Run:
        r = self._runs.get(run_id)
        if r is None:
            r = self._runs[run_id] = _Run(run_id)
            self._dirty["runs"].add(run_id)
        return r

    def _link(self, child_id: str, parent_id: str, discover_key: Tuple[str, str, int]) -> None:
        if not child_id or child_id == parent_id or child_id == self.root_run_id:
            return
        child = self._run(child_id)
        # The ledger's link (a seq > 0) outranks a snapshot's (seq 0), whichever is folded first,
        # so a run's place in the tree does not depend on when its snapshot was read.
        if child.parent_run_id is None or (child.discover_key[2] == 0 and discover_key[2] > 0):
            if child.parent_run_id != parent_id or child.discover_key != discover_key:
                child.parent_run_id = parent_id
                child.discover_key = discover_key
                self._dirty["runs"].add(child_id)

    def _ledger_workflow(self, run_id: str, workflow_id: str) -> None:
        """A run's workflow as its parent's ledger names it (its own snapshot, when read, wins)."""
        r = self._runs.get(run_id)
        if r is not None and workflow_id and not r.snapshot.get("workflow_id") and r.workflow_id != workflow_id:
            r.workflow_id = workflow_id
            self._dirty["runs"].add(run_id)

    def _fold_snapshot(self, snap: Dict[str, Any]) -> None:
        rid = snap.get("run_id") or ""
        if not rid:
            return
        r = self._run(rid)
        if r.snapshot != snap:
            r.snapshot = snap
            self._dirty["runs"].add(rid)
        if snap.get("workflow_id") and r.workflow_id != snap["workflow_id"]:
            r.workflow_id = snap["workflow_id"]
            self._dirty["runs"].add(rid)
        parent = snap.get("parent_run_id")
        if parent and rid != self.root_run_id:
            self._run(parent)
            self._link(rid, parent, (snap.get("created_at") or "", rid, 0))
        waiting = snap.get("waiting")
        if isinstance(waiting, dict):
            child = _child_from_wait(waiting)
            if child:
                self._link(child, rid, (snap.get("created_at") or "", child, 0))

    def _run_order(self) -> List[str]:
        """The tree's runs in tree order: root, then each child (by discovery) followed by its own
        subtree; runs whose ancestry does not reach the root (a windowed or partial read) after.

        Defined as the sort of each row's `order` (`_path_order`), so a client that sorts rows by
        `order` (`apply_delta`) gets the same order as `to_json`. (A tree-walk order with a
        per-level `order` disagreed for a grandchild discovered after its uncle and for runs
        whose parent is not yet known; found by the replay invariants.)"""
        return sorted(self._runs, key=lambda rid: (_order_tuple(self._path_order(rid)), rid))

    def _path_order(self, rid: str) -> List[Any]:
        """A run row's `order`: the root's `["", "", 0, root]`, then one `[discovered at, run id,
        seq, run id]` level per run on the path from the root down to `rid`. A run whose ancestry
        does not reach the root starts with `["~", "", 0, ""]` (after the root's tree)."""
        levels: List[Any] = []
        seen: Set[str] = set()
        cur = rid
        while cur not in seen and len(seen) < 64:
            if cur == self.root_run_id:
                return ["", "", 0, self.root_run_id] + levels
            seen.add(cur)
            r = self._runs[cur]
            levels = [r.discover_key[0], r.discover_key[1], r.discover_key[2], cur] + levels
            if not r.parent_run_id or r.parent_run_id not in self._runs:
                break
            cur = r.parent_run_id
        return ["~", "", 0, ""] + levels

    def _run_status(self, r: _Run) -> str:
        snap_status = r.snapshot.get("status") or ""
        if snap_status in TERMINAL_STATUSES:
            return snap_status
        if r.ledger_terminal:
            return r.ledger_terminal
        if r.open_wait is not None:
            return "waiting"
        if r.record_count:
            return "running"
        if snap_status == "waiting":
            return "running" if r.resolved_wait_keys else "waiting"
        return snap_status or "unknown"

    def _run_row(self, rid: str, order: List[str]) -> Dict[str, Any]:
        r = self._runs[rid]
        depth = 0
        p = r.parent_run_id
        guard = 0
        while p and p in self._runs and guard < 64:
            depth += 1
            p = self._runs[p].parent_run_id if p != self.root_run_id else None
            guard += 1
        w = r.open_wait
        return {
            "run_id": rid,
            "parent_run_id": r.parent_run_id if rid != self.root_run_id else None,
            "workflow_id": r.workflow_id or None,
            "workflow": workflow_ref(r.workflow_id),
            "status": self._run_status(r),
            "display_status": r.display_status or None,
            "depth": depth if rid != self.root_run_id else 0,
            "order": self._path_order(rid),
            "waiting": ({"kind": w["kind"], "wait_key": w["wait_key"], "child_run_id": w.get("child_run_id")} if w else None),
            "holds_interaction": bool(self._interaction and self._interaction.get("run_id") == rid),
            "records": r.record_count,
            "first_seq": r.first_seq,
            "last_seq": r.last_seq,
        }

    # ----------------------------------------------------------------- records

    def _resolve(self, value: Any, run_id: str, step_id: str, field: str) -> Any:
        if not is_slim_marker(value):
            return value
        # The marker names the STARTED record it points at. An approved tool call completes under a
        # NEW step id whose `$slim` points back at the waiting step's STARTED record, so the
        # record's own step id is not the target (found by the a2/e2 fixtures).
        target = _s(_obj(value.get("$slim")).get("step_id")) or step_id
        payload = self._started.get((run_id, target))
        resolved = resolve_slim_value(value, {target: payload}) if isinstance(payload, dict) else value
        if is_slim_marker(resolved):
            self._warnings[f"slim:{run_id}:{step_id}:{field}"] = (
                f"run {run_id} step {step_id}: `{field}` points at a STARTED record this view did not fold "
                "(windowed history); the field is omitted")
            return None
        return resolved

    def _payload(self, rec: Dict[str, Any], run_id: str, step_id: str) -> Dict[str, Any]:
        eff = _obj(rec.get("effect"))
        raw = _obj(eff.get("payload"))
        return {k: self._resolve(raw.get(k), run_id, step_id, k) for k in _VIEW_PAYLOAD_FIELDS if k in raw}

    def _fold_record(self, rec: Dict[str, Any]) -> None:
        run_id = _s(rec.get("run_id"))
        if not run_id:
            return
        r = self._run(run_id)
        seq_raw = rec.get("seq")
        try:
            seq = int(seq_raw) if seq_raw is not None and not isinstance(seq_raw, bool) else r.last_seq + 1
        except (TypeError, ValueError):
            seq = r.last_seq + 1
        if seq <= r.last_seq:
            return  # already folded: folding is idempotent
        if r.record_count == 0:
            r.first_seq = seq
            if seq > 1:
                self._warnings[f"window:{run_id}"] = (
                    f"run {run_id}: records before seq {seq} were not folded (windowed history)")
        r.last_seq = seq
        r.record_count += 1
        self._dirty["runs"].add(run_id)

        status = _enum(rec.get("status"))
        step_id = _s(rec.get("step_id"))
        node_id = _s(rec.get("node_id"))
        eff = _obj(rec.get("effect"))
        eff_type = _s(eff.get("type"))
        raw_payload = _obj(eff.get("payload"))
        result = rec.get("result")
        res = _obj(result)
        idem = _s(rec.get("idempotency_key"))
        ts = _s(rec.get("ended_at")) or _s(rec.get("started_at"))
        if not r.first_ts:
            r.first_ts = _s(rec.get("started_at")) or ts

        if status == "started" and step_id and eff_type and eff_type != "emit_event":
            started = {k: raw_payload[k] for k in _VIEW_PAYLOAD_FIELDS if k in raw_payload}
            self._started.setdefault((run_id, step_id), started)

        event_name = _s(raw_payload.get("name")) if eff_type == "emit_event" else ""
        observability = eff_type == "emit_event" and (event_name in _OBSERVABILITY_EVENTS or is_progress_record(rec))

        # A wait is over when a later step of the same run exists (or its resume, below).
        if r.open_wait is not None and not observability and seq > r.open_wait["seq"]:
            if not (eff_type == "resume" and _s(raw_payload.get("wait_key")) == r.open_wait["wait_key"]):
                self._close_wait(r, "later_step", ts, step_id, seq)

        # Run-level terminal records.
        if idem.startswith("system:terminal_status:"):
            t = idem[len("system:terminal_status:"):].split(":", 1)[0].strip().lower()
            if t in TERMINAL_STATUSES:
                r.ledger_terminal = t
                if r.open_wait is not None:
                    self._close_wait(r, "terminal", ts, step_id, seq)
            return
        if eff_type == "emit_event":
            self._fold_event(r, rec, raw_payload, event_name, ts, seq, step_id)
            return
        if is_completion_record(rec):
            if status == "completed":
                r.ledger_terminal = r.ledger_terminal or "completed"
                self._tracker.execution_id(rec)
                if run_id == self.root_run_id:
                    output = res.get("output") if "output" in res else res
                    self._final_message(run_id, step_id, seq, ts, _output_text(output),
                                        [a for a, _ in _artifact_refs(output)], "ledger")
                self._note_artifacts(run_id, step_id, seq, res)
            return
        if not eff_type and status == "failed":
            r.ledger_terminal = "failed"
            self._tracker.execution_id(rec)
            return

        exec_id = self._tracker.execution_id(rec) or step_id or f"seq{seq}"
        aid = f"{run_id}:{exec_id}"

        if eff_type == "resume":
            self._fold_resume(r, rec, raw_payload, aid, ts, seq, step_id)
            return

        payload = self._payload(rec, run_id, step_id)
        act = self._activity(aid, run_id, node_id, eff_type, exec_id, ts, seq)
        act["seqs"].append(seq)
        if step_id and step_id not in act["step_ids"]:
            act["step_ids"].append(step_id)
        started_at = _s(rec.get("started_at"))
        if started_at and (not act["started_at"] or started_at < act["started_at"]):
            act["started_at"] = started_at
        prev_status = act["status"]
        if status == "started":
            if prev_status not in TERMINAL_STATUSES and prev_status not in ("waiting", "denied"):
                act["status"] = "running"
        else:
            act["status"] = status if status in ("completed", "failed", "cancelled", "waiting") else (status or prev_status)
            if status != "waiting":
                act["ended_at"] = _s(rec.get("ended_at")) or act["ended_at"]
        if rec.get("error"):
            act["error"] = str(rec.get("error"))
        self._fill_activity(act, eff_type, payload, res, status, rec, run_id, step_id)
        if status in ("completed", "failed", "cancelled") and step_id:
            self._started.pop((run_id, step_id), None)

        if status == "waiting":
            wait = _obj(res.get("wait"))
            if wait:
                # The STARTED payload of a waiting step is kept: the record that ends the wait
                # (an approved call's completion, under a new step id) points back at it.
                self._open_wait(r, rec, wait, payload, eff_type, aid, ts, seq, step_id)
        elif eff_type == "start_subworkflow":
            child = ""
            for src in (res, raw_payload):
                for k in ("sub_run_id", "child_run_id", "subRunId"):
                    if _s(src.get(k)):
                        child = _s(src.get(k))
                        break
                if child:
                    break
            if child:
                act["child_run_id"] = child
                self._link(child, run_id, (ts, child, seq))
                self._ledger_workflow(child, _s(payload.get("workflow_id")))

        if eff_type == "answer_user" and status == "completed":
            text = _text_of(res) or _text_of(payload)
            if text:
                self._message(f"{run_id}:{step_id}:answer_user", role="assistant", kind="answer", text=text,
                              run_id=run_id, step_id=step_id, seq=seq, ts=ts, source="answer_user",
                              artifact_ids=[a for a, _ in _artifact_refs(result)])
        self._note_artifacts(run_id, step_id, seq, result)
        self._dirty["activities"].add(aid)

    def _activity(self, aid: str, run_id: str, node_id: str, eff_type: str, exec_id: str, ts: str, seq: int) -> Dict[str, Any]:
        act = self._activities.get(aid)
        if act is None:
            kind = {"llm_call": "llm", "tool_calls": "tool", "tool_invoke": "tool", "start_subworkflow": "subworkflow",
                    "wait_until": "timer", "wait_event": "event_wait", "ask_user": "ask_user",
                    "answer_user": "answer"}.get(eff_type, eff_type or "step")
            act = self._activities[aid] = {
                "id": aid, "run_id": run_id, "execution_id": exec_id, "node_id": node_id, "kind": kind,
                "effect_type": eff_type or None, "name": None, "status": "running", "started_at": None,
                "ended_at": None, "seqs": [], "step_ids": [], "order": [ts, run_id, seq],
            }
        return act

    def _fill_activity(self, act: Dict[str, Any], eff_type: str, payload: Dict[str, Any], res: Dict[str, Any],
                       status: str, rec: Dict[str, Any], run_id: str, step_id: str) -> None:
        if eff_type == "llm_call":
            for k in ("provider", "model"):
                if _s(payload.get(k)) and not act.get(k):
                    act[k] = _s(payload[k])
            if not act.get("model") and _s(res.get("model")):
                act["model"] = _s(res.get("model"))
            if act["kind"] == "llm" or not act.get("name"):
                # An Agent execution (start_subworkflow + its structured-output llm_call) keeps
                # the name its delegation gave it.
                act["name"] = act.get("model") or act.get("provider") or "llm_call"
            if status == "completed" and (run_id, step_id) not in self._usage_steps:
                usage = _extract_usage_from_ledger_record(rec)
                self._usage_steps.add((run_id, step_id))
                self._usage["llm_calls"] += 1
                if usage:
                    act["usage"] = usage
                    for k in ("input_tokens", "output_tokens", "total_tokens"):
                        self._usage[k] += int(usage.get(k) or 0)
        elif eff_type in ("tool_calls", "tool_invoke"):
            calls_raw = payload.get("tool_calls")
            if not isinstance(calls_raw, list) and status == "waiting":
                calls_raw = _obj(_obj(res.get("wait")).get("details")).get("tool_calls")
            if eff_type == "tool_invoke" and not isinstance(calls_raw, list):
                raw = _obj(_obj(rec.get("effect")).get("payload"))
                name = _s(raw.get("name")) or _s(raw.get("tool"))
                calls_raw = [{"name": name, "arguments": raw.get("arguments") or raw.get("args")}] if name else None
            calls = [c for c in calls_raw if isinstance(c, dict)] if isinstance(calls_raw, list) else []
            rows = act.setdefault("tool_calls", [])
            by_id = {row["id"]: row for row in rows}
            results = res.get("results") if isinstance(res.get("results"), list) else []
            matched = _match_results(calls, results) if status in ("completed", "failed") else {}
            approval = isinstance(act.get("wait"), dict) and act["wait"].get("kind") == "tool_approval"
            for i, call in enumerate(calls):
                cid = _tool_call_id(call, i)
                row = by_id.get(cid)
                if row is None:
                    row = {"id": cid, "name": _s(call.get("name")) or None, "arguments": call.get("arguments"),
                           "status": "running"}
                    rows.append(row)
                    by_id[cid] = row
                if row["status"] == "denied":
                    continue
                if status == "waiting":
                    row["status"] = "waiting"
                elif status in ("completed", "failed"):
                    out = matched.get(i)
                    if approval and _is_denial(out):
                        row["status"] = "denied"
                        row["error"] = _s(out.get("error"))
                        continue
                    failed = status == "failed" or (isinstance(out, dict) and (out.get("success") is False or bool(out.get("error"))))
                    row["status"] = "failed" if failed else "completed"
                    if isinstance(out, dict):
                        if "output" in out or "result" in out:
                            row["output"] = out.get("output", out.get("result"))
                        if out.get("error"):
                            row["error"] = str(out.get("error"))
                elif status == "cancelled":
                    row["status"] = "cancelled"
            if rows and all(row.get("status") == "denied" for row in rows) and act["status"] != "denied":
                # The ledger records a denial as the approval step's results, never as
                # `approved: false` (see `_is_denial`): every call denied = the batch was denied.
                act["status"] = "denied"
                act["denied_reason"] = next((row["error"] for row in rows if row.get("error")), "") or "Denied by user"
            names = [row["name"] for row in rows if row.get("name")]
            act["name"] = ", ".join(dict.fromkeys(names)) or act.get("name")
        elif eff_type == "start_subworkflow":
            if _s(payload.get("workflow_id")):
                act["name"] = _s(payload["workflow_id"])
        elif not act.get("name"):
            act["name"] = eff_type or None

    def _open_wait(self, r: _Run, rec: Dict[str, Any], wait: Dict[str, Any], payload: Dict[str, Any], eff_type: str,
                   aid: str, ts: str, seq: int, step_id: str) -> None:
        kind = wait_kind(wait, eff_type, r.run_id)
        key = _s(wait.get("wait_key"))
        details = wait.get("details") if isinstance(wait.get("details"), dict) else None
        tool_calls = None
        if kind == "tool_approval":
            calls = payload.get("tool_calls")
            if not isinstance(calls, list):
                calls = _obj(details).get("tool_calls")
            tool_calls = [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
        w = {
            "run_id": r.run_id, "step_id": step_id, "seq": seq, "ts": ts, "kind": kind, "wait_key": key or None,
            "reason": _enum(wait.get("reason")) or None, "until": wait.get("until"), "prompt": wait.get("prompt"),
            "choices": wait.get("choices"), "allow_free_text": wait.get("allow_free_text"),
            "details": copy.deepcopy(details), "tool_calls": copy.deepcopy(tool_calls), "activity_id": aid,
        }
        child = _child_from_wait(wait)
        if child:
            w["child_run_id"] = child
            act = self._activities.get(aid)
            if act is not None:
                act["child_run_id"] = child
            self._link(child, r.run_id, (ts, child, seq))
            self._ledger_workflow(child, _s(payload.get("workflow_id")))
        if key:
            r.seen_wait_keys.add(key)
            self._wait_activity[(r.run_id, key)] = aid
        r.open_wait = w
        act = self._activities.get(aid)
        if act is not None:
            act["wait"] = {"kind": kind, "wait_key": key or None, "until": wait.get("until"), "ended_by": None}
        if kind == "ask_user" and _s(wait.get("prompt")):
            self._message(f"{r.run_id}:{step_id}:question", role="assistant", kind="question",
                          text=_s(wait.get("prompt")), run_id=r.run_id, step_id=step_id, seq=seq, ts=ts,
                          source="wait", extra={"choices": wait.get("choices"), "wait_key": key or None})
        self._dirty["runs"].add(r.run_id)

    def _close_wait(self, r: _Run, ended_by: str, ts: str, step_id: str, seq: int) -> None:
        w = r.open_wait
        if w is None:
            return
        r.open_wait = None
        if w.get("wait_key"):
            r.resolved_wait_keys.add(w["wait_key"])
        act = self._activities.get(w.get("activity_id") or "")
        if act is not None:
            if act["status"] == "waiting":
                act["status"] = "completed"
                act["ended_at"] = ts or act["ended_at"]
                for row in act.get("tool_calls") or []:
                    if row.get("status") == "waiting":
                        row["status"] = "running"
            if isinstance(act.get("wait"), dict) and not act["wait"].get("ended_by"):
                act["wait"]["ended_by"] = ended_by
            self._dirty["activities"].add(act["id"])
        self._dirty["runs"].add(r.run_id)

    def _fold_resume(self, r: _Run, rec: Dict[str, Any], raw_payload: Dict[str, Any], aid: str, ts: str, seq: int,
                     step_id: str) -> None:
        key = _s(raw_payload.get("wait_key"))
        answer = raw_payload.get("payload")
        if key:
            r.seen_wait_keys.add(key)
            r.resolved_wait_keys.add(key)
        target_aid = self._wait_activity.get((r.run_id, key)) if key else None
        wait = r.open_wait if (r.open_wait is not None and (not key or r.open_wait.get("wait_key") == key)) else None
        kind = (wait or {}).get("kind")
        act = self._activities.get(target_aid or aid)
        if act is not None:
            if kind is None and isinstance(act.get("wait"), dict):
                kind = act["wait"].get("kind")
            act["seqs"].append(seq)
            if step_id and step_id not in act["step_ids"]:
                act["step_ids"].append(step_id)
            approved = _obj(answer).get("approved") if isinstance(answer, dict) else None
            if kind == "tool_approval" and approved is False:
                act["status"] = "denied"
                act["ended_at"] = ts or act["ended_at"]
                for row in act.get("tool_calls") or []:
                    row["status"] = "denied"
                act["denied_reason"] = _s(_obj(answer).get("reason")) or "Denied by user"
            elif act["status"] == "waiting":
                act["status"] = "completed"
                act["ended_at"] = ts or act["ended_at"]
                for row in act.get("tool_calls") or []:
                    if row.get("status") == "waiting":
                        row["status"] = "running"
            if isinstance(act.get("wait"), dict) and not act["wait"].get("ended_by"):
                act["wait"]["ended_by"] = "resume"
            self._dirty["activities"].add(act["id"])
        if wait is not None:
            r.open_wait = None
            self._dirty["runs"].add(r.run_id)
        if kind == "ask_user":
            text = _text_of(answer) if not isinstance(answer, str) else answer.strip()
            if not text and isinstance(answer, dict):
                text = _s(answer.get("response")) or _s(answer.get("value"))
            if text:
                self._message(f"{r.run_id}:{step_id}:answer", role="user", kind="answer", text=text, run_id=r.run_id,
                              step_id=step_id, seq=seq, ts=ts, source="resume", extra={"wait_key": key or None})
        self._note_artifacts(r.run_id, step_id, seq, answer)

    def _fold_event(self, r: _Run, rec: Dict[str, Any], payload: Dict[str, Any], name: str, ts: str, seq: int,
                    step_id: str) -> None:
        body = payload.get("payload")
        if name in ("abstract.status", "abstractcode.status"):
            b = _obj(body)
            text = _s(b.get("status")) or _s(b.get("text")) or _s(b.get("value"))
            if text and text.lower() not in TERMINAL_STATUSES and r.display_status != text:
                r.display_status = text
                self._dirty["runs"].add(r.run_id)
            return
        if is_progress_record(rec):
            # A progress tick updates the execution it reports on (executions rule 5).
            exec_id = self._tracker.execution_id(rec)
            act = self._activities.get(f"{r.run_id}:{exec_id}") if exec_id else None
            if act is not None:
                b = _obj(body) if body is not None else _obj(_obj(rec.get("result")).get("payload"))
                act["progress"] = copy.deepcopy(b)
                self._dirty["activities"].add(act["id"])
            return
        if name in ("abstract.message", "abstractcode.message") and _s(rec.get("status")) != "started":
            b = _obj(body)
            text = _s(b.get("message")) or _s(b.get("text")) or (_s(body) if isinstance(body, str) else "")
            if text:
                self._message(f"{r.run_id}:{step_id}:event", role="system", kind="event", text=text, run_id=r.run_id,
                              step_id=step_id, seq=seq, ts=ts, source="emit_event",
                              extra={"level": _s(b.get("level")) or None})

    # ----------------------------------------------------------------- messages, artifacts

    def _message(self, mid: str, *, role: str, kind: str, text: str, run_id: str, step_id: str, seq: int, ts: str,
                 source: str, artifact_ids: Optional[List[str]] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        msg = {
            "id": mid, "role": role, "kind": kind, "text": text, "run_id": run_id, "step_ids": [step_id] if step_id else [],
            "seqs": [seq] if seq else [], "source": source, "artifact_ids": sorted(set(artifact_ids or [])),
            "order": [ts, run_id, seq],
        }
        for k, v in (extra or {}).items():
            if v is not None:
                msg[k] = v
        if self._messages.get(mid) != msg:
            self._messages[mid] = msg
            self._dirty["messages"].add(mid)

    def _final_message(self, run_id: str, step_id: str, seq: int, ts: str, text: str, refs: List[str], source: str) -> None:
        """The run's final output: always the one `<run_id>:final` message (the ledger's completion
        record, else the snapshot's output). When it repeats an answer the run already gave, the
        final message names that answer in `repeats` (set in `_finalize`) and a client shows one.

        (Merging the final into the answer message instead, as before, made the message list
        depend on read order: a snapshot read before the ledger's answer produced a `final`
        message that a later answer could no longer absorb, and items are never removed. Found
        by the replay invariants, fixture l_answer_then_completion.)"""
        if not text:
            return
        self._message(f"{run_id}:final", role="assistant", kind="final", text=text, run_id=run_id, step_id=step_id,
                      seq=seq, ts=ts, source=source, artifact_ids=refs)

    def _note_artifacts(self, run_id: str, step_id: str, seq: int, value: Any) -> None:
        if not isinstance(value, (dict, list)):
            return
        for art_id, path in _artifact_refs(value):
            if art_id in self._artifacts:
                continue
            self._artifacts[art_id] = {"id": art_id, "run_id": run_id, "step_id": step_id or None, "seq": seq,
                                       "path": path, "order": [run_id, seq, art_id]}
            self._dirty["artifacts"].add(art_id)

    # ----------------------------------------------------------------- finalize, delta

    def _finalize(self) -> None:
        root = self._runs[self.root_run_id]
        # The run's input is the first user message (the snapshot holds it, the ledger does not).
        snap = root.snapshot
        if snap:
            if snap.get("prompt"):
                self._message(f"{self.root_run_id}:input", role="user", kind="prompt", text=snap["prompt"],
                              run_id=self.root_run_id, step_id="", seq=0, ts=snap.get("created_at") or root.first_ts,
                              source="run_input")
            if snap.get("status") == "completed" and f"{self.root_run_id}:final" not in self._messages:
                # No terminal record folded (a legacy or windowed ledger): the snapshot's output.
                self._final_message(self.root_run_id, "", 0, snap.get("updated_at") or "~", snap.get("output_text") or "",
                                    list(snap.get("output_artifacts") or []), "run_output")
        # A final message that repeats an answer of its run names it (the first such answer).
        final = self._messages.get(f"{self.root_run_id}:final")
        if final is not None:
            same = sorted((m for m in self._messages.values()
                           if m["run_id"] == self.root_run_id and m["kind"] == "answer" and m["role"] == "assistant"
                           and m["text"] == final["text"]), key=_order_key)
            repeats = same[0]["id"] if same else None
            if final.get("repeats") != repeats:
                if repeats:
                    final["repeats"] = repeats
                else:
                    final.pop("repeats", None)
                self._dirty["messages"].add(final["id"])
        # Snapshot waits the ledger never mentioned (a windowed history, an empty ledger).
        for r in self._runs.values():
            w = r.snapshot.get("waiting") if r.snapshot.get("status") == "waiting" else None
            if r.open_wait is None and isinstance(w, dict):
                key = _s(w.get("wait_key"))
                if key and key not in r.seen_wait_keys and not r.ledger_terminal:
                    kind = wait_kind(w, "", r.run_id)
                    details = w.get("details") if isinstance(w.get("details"), dict) else None
                    r.open_wait = {
                        "run_id": r.run_id, "step_id": None, "seq": r.last_seq, "ts": r.snapshot.get("created_at") or "",
                        "kind": kind, "wait_key": key, "reason": w.get("reason") or None, "until": w.get("until"),
                        "prompt": w.get("prompt"), "choices": w.get("choices"),
                        "allow_free_text": w.get("allow_free_text"), "details": copy.deepcopy(details),
                        "tool_calls": copy.deepcopy([c for c in _obj(details).get("tool_calls") or [] if isinstance(c, dict)])
                        if kind == "tool_approval" else None,
                        "activity_id": None, "source": "snapshot",
                    }
                    r.seen_wait_keys.add(key)
                    child = _child_from_wait(w)
                    if child:
                        r.open_wait["child_run_id"] = child
                        self._link(child, r.run_id, (r.snapshot.get("created_at") or "", child, 0))
                    self._dirty["runs"].add(r.run_id)
        # The person-facing interaction: the newest open approval or question in the tree.
        interaction = None
        if self._run_status(root) not in TERMINAL_STATUSES:
            best = None
            for r in self._runs.values():
                w = r.open_wait
                if w is None or w["kind"] not in PERSON_FACING_KINDS or self._run_status(r) in TERMINAL_STATUSES:
                    continue
                key = (w.get("ts") or "", w.get("seq") or 0, r.run_id)
                if best is None or key > best[0]:
                    best = (key, w)
            if best is not None:
                interaction = self._interaction_of(best[1])
        if interaction != self._interaction:
            old = self._interaction
            self._interaction = interaction
            for x in (old, interaction):
                if x:
                    self._dirty["runs"].add(x["run_id"])
            if _interaction_identity(old) != _interaction_identity(interaction):
                # Typed changes (root backlog 1058): the closed one first, then the opened one.
                if old:
                    r_old = self._runs.get(old["run_id"])
                    self._pending_changes.append(self._interaction_change(old, "closed", r_old.last_seq if r_old else 0))
                if interaction:
                    self._pending_changes.append(self._interaction_change(interaction, "opened", int(interaction.get("seq") or 0)))

    def _interaction_of(self, w: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "kind": w["kind"], "run_id": w["run_id"], "wait_key": w.get("wait_key"), "step_id": w.get("step_id"),
            "seq": w.get("seq"), "prompt": w.get("prompt"), "choices": w.get("choices"),
            "allow_free_text": w.get("allow_free_text"), "details": copy.deepcopy(w.get("details")),
            "tool_calls": copy.deepcopy(w.get("tool_calls")), "activity_id": w.get("activity_id"),
            "command": {"type": "resume", "run_id": w["run_id"], "payload": {"wait_key": w.get("wait_key")}},
            # What the resume payload carries for this kind (the runtime's attention contract).
            "answer_payload": ANSWER_PAYLOADS[w["kind"]],
        }
        if isinstance(out["details"], dict) and "tool_calls" in out["details"]:
            out["details"] = {k: v for k, v in out["details"].items() if k != "tool_calls"}
        return out

    @staticmethod
    def _interaction_change(it: Dict[str, Any], change: str, seq: int) -> Dict[str, Any]:
        return {"kind": "interaction", "id": it.get("wait_key") or f"{it['run_id']}:{it.get('step_id') or ''}",
                "run_id": it["run_id"], "seq": seq, "change": change, "interaction_kind": it["kind"],
                "wait_key": it.get("wait_key")}

    def _open_waits(self) -> List[Dict[str, Any]]:
        rows = []
        for rid in self._run_order():
            w = self._runs[rid].open_wait
            if w is None or self._run_status(self._runs[rid]) in TERMINAL_STATUSES:
                continue
            rows.append({"run_id": rid, "kind": w["kind"], "wait_key": w.get("wait_key"), "step_id": w.get("step_id"),
                         "seq": w.get("seq"), "until": w.get("until"), "child_run_id": w.get("child_run_id"),
                         "person_facing": w["kind"] in PERSON_FACING_KINDS})
        return rows

    def _take_delta(self) -> Dict[str, Any]:
        order = self._run_order()
        delta = {
            "version": REPLAY_VIEW_VERSION,
            "root_run_id": self.root_run_id,
            "workflow": self._root_workflow(),
            "session_id": self._runs[self.root_run_id].snapshot.get("session_id"),
            "cursor": self.cursor,
            "status": self._run_status(self._runs[self.root_run_id]),
            "display_status": self._runs[self.root_run_id].display_status or None,
            "interaction": copy.deepcopy(self._interaction),
            "open_waits": self._open_waits(),
            "usage": dict(self._usage),
            "warnings": [self._warnings[k] for k in sorted(self._warnings)],
            "runs": [self._run_row(rid, order) for rid in order if rid in self._dirty["runs"]],
            "messages": [copy.deepcopy(self._messages[i]) for i in sorted(self._dirty["messages"]) if i in self._messages],
            "activities": [copy.deepcopy(self._activities[i]) for i in sorted(self._dirty["activities"]) if i in self._activities],
            "artifacts": [copy.deepcopy(self._artifacts[i]) for i in sorted(self._dirty["artifacts"]) if i in self._artifacts],
        }
        delta["changes"] = self._changes(order, delta)
        for k in self._dirty:
            self._dirty[k] = set()
        # A run row's `order`/`depth` can change when another run is linked; send every row then.
        if delta["runs"]:
            delta["runs"] = [self._run_row(rid, order) for rid in order]
        return delta


    def _changes(self, order: List[str], delta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The delta's typed changes (root backlog 1058: a push is a ledger event at its seq).

        Each is `{kind, id, run_id, seq, ...}`; `seq` is the newest seq of that run behind the
        change (0 when only a snapshot is behind it). Order: status (tree order), interaction
        (closed, then opened), run (tree order), message, activity, artifact (each by id).
        `apply_delta` ignores them: the view itself is carried by the scalars and lists."""
        out: List[Dict[str, Any]] = []
        for rid in order:
            r = self._runs[rid]
            st = self._run_status(r)
            if self._statuses.get(rid) != st:
                out.append({"kind": "status", "id": rid, "run_id": rid, "seq": r.last_seq, "status": st,
                            "previous": self._statuses.get(rid)})
                self._statuses[rid] = st
        out.extend(self._pending_changes)
        self._pending_changes = []
        for rid in order:
            if rid in self._dirty["runs"]:
                out.append({"kind": "run", "id": rid, "run_id": rid, "seq": self._runs[rid].last_seq})
        for m in delta["messages"]:
            out.append({"kind": "message", "id": m["id"], "run_id": m["run_id"], "seq": max(m["seqs"] or [0]),
                        "message_kind": m["kind"], "role": m["role"]})
        for a in delta["activities"]:
            out.append({"kind": "activity", "id": a["id"], "run_id": a["run_id"], "seq": max(a["seqs"] or [0]),
                        "activity_kind": a["kind"], "status": a["status"]})
        for a in delta["artifacts"]:
            out.append({"kind": "artifact", "id": a["id"], "run_id": a["run_id"], "seq": int(a.get("seq") or 0)})
        return out


def _interaction_identity(it: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str, str]]:
    if not it:
        return None
    return (str(it.get("run_id") or ""), str(it.get("wait_key") or ""), str(it.get("step_id") or ""))


def _order_tuple(order: Iterable[Any]) -> Tuple:
    # The same formatting as `delta._order_key` (a client sorts by it): ints zero-padded.
    return tuple(str(x) if not isinstance(x, int) else f"{x:012d}" for x in order or ())


def _order_key(item: Dict[str, Any]) -> Tuple:
    # `delta._order_key`: the order tuple compared whole (a prefix first), then the id.
    return (_order_tuple(item.get("order") or ()), str(item.get("id") or item.get("run_id")))


def _sorted_items(items: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [copy.deepcopy(v) for v in sorted(items.values(), key=_order_key)]


# --------------------------------------------------------------------------- module API


def replay(
    records_by_run: Mapping[str, Iterable[Dict[str, Any]]],
    runs_by_id: Optional[Mapping[str, Any]] = None,
    *,
    root_run_id: Optional[str] = None,
) -> ReplayView:
    """Fold a run tree's ledgers and snapshots into a new `ReplayView` (pure, deterministic).

    `records_by_run` maps run id -> that run's records in ledger order; the order between runs
    does not matter. `root_run_id` defaults to the one snapshot without a parent among
    `runs_by_id`, else the first key of `records_by_run`.
    """
    runs_by_id = dict(runs_by_id or {})
    root = _s(root_run_id)
    if not root:
        snaps = [_snapshot(v) for v in runs_by_id.values()]
        ids = {s["run_id"] for s in snaps}
        roots = [s["run_id"] for s in snaps if not s.get("parent_run_id") or s["parent_run_id"] not in ids]
        root = roots[0] if roots else next(iter(records_by_run), "")
    if not root:
        raise ValueError("replay needs a root run id")
    view = ReplayView(root)
    records: List[Dict[str, Any]] = []
    for rid in sorted(records_by_run):
        records.extend(r for r in records_by_run[rid] or () if isinstance(r, dict))
    view.apply(records, runs_by_id=runs_by_id)
    return view


def apply(view: ReplayView, new_records: Iterable[Dict[str, Any]], *, runs_by_id: Optional[Mapping[str, Any]] = None) -> ReplayView:
    """Fold `new_records` into `view` (in place) and return it; `view.last_delta` is the change."""
    view.apply(new_records, runs_by_id=runs_by_id)
    return view
