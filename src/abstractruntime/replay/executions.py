"""Which execution of its node a ledger record belongs to (framework backlog 1016, 0993).

The runtime writes several ledger steps for one node execution and names no execution id:
an Agent node is `start_subworkflow` started/waiting, a `resume`, then its structured-output
`llm_call`; an approved tool call is a fresh step id with the same idempotency key. This is
the one place the phases are recognised, ported from AbstractFlow's `executionIdFor`
(`abstractflow/src/utils/ledgerEvents.ts`, checked over the 9,476 ledgers of the operator's
gateway on 2026-10-11: 1,668 duplicate rows removed, none added). Rules, in order:

1. a step already seen belongs to its execution;
2. a `resume` belongs to the execution whose wait it answers (same wait key);
3. a record whose idempotency key is already in the node's latest execution belongs to it
   (an approved tool call's execution, a retry, a crash replay);
4. an Agent's structured-output call (result key `_temp.agent.<node>.structured`) and the
   run's terminal record (`system:completion:*`) belong to the node's latest execution;
5. a progress tick (`abstract.progress`, keyed `system:progress:<step id>:<n>`) belongs to the
   execution of the step it reports on, else to the node's latest execution, and never
   becomes the node's latest execution itself;
6. anything else starts a new execution, whose id is the record's step id.

"Same node with nothing recorded in between" is NOT a rule: loops, If/Else and Set Variable
write no records, so a node run twice in a row would merge.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Set

_PROGRESS_KEY = re.compile(r"^system:progress:([^:]+):")


def _s(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _status(rec: Dict[str, Any]) -> str:
    raw = rec.get("status")
    raw = getattr(raw, "value", raw)
    return _s(raw).lower()


def is_progress_record(rec: Dict[str, Any]) -> bool:
    res = rec.get("result")
    if isinstance(res, dict) and _s(res.get("name")) == "abstract.progress":
        return True
    eff = rec.get("effect")
    payload = eff.get("payload") if isinstance(eff, dict) else None
    return isinstance(payload, dict) and _s(payload.get("name")) == "abstract.progress"


class _RunExecutions:
    __slots__ = ("by_step", "by_wait_key", "latest_by_node", "idem_by_exec")

    def __init__(self) -> None:
        self.by_step: Dict[str, str] = {}
        self.by_wait_key: Dict[str, str] = {}
        self.latest_by_node: Dict[str, str] = {}
        self.idem_by_exec: Dict[str, Set[str]] = {}


class ExecutionTracker:
    """Stateful, per-run execution identity; feed records in each run's ledger order."""

    def __init__(self) -> None:
        self._runs: Dict[str, _RunExecutions] = {}

    def execution_id(self, rec: Dict[str, Any]) -> Optional[str]:
        run_id = _s(rec.get("run_id"))
        node_id = _s(rec.get("node_id"))
        step_id = _s(rec.get("step_id"))
        if not run_id or not node_id or not step_id:
            return None
        ex = self._runs.get(run_id)
        if ex is None:
            ex = self._runs[run_id] = _RunExecutions()
        effect = rec.get("effect") if isinstance(rec.get("effect"), dict) else {}
        payload = effect.get("payload") if isinstance(effect.get("payload"), dict) else {}
        idem = _s(rec.get("idempotency_key"))
        latest = ex.latest_by_node.get(node_id)

        exec_id = ex.by_step.get(step_id)
        if not exec_id and effect.get("type") == "resume":
            exec_id = ex.by_wait_key.get(_s(payload.get("wait_key")))
        if not exec_id and idem and latest and idem in ex.idem_by_exec.get(latest, ()):
            exec_id = latest
        if not exec_id and latest:
            result_key = _s(effect.get("result_key"))
            if result_key == f"_temp.agent.{node_id}.structured" or idem.startswith("system:completion:"):
                exec_id = latest
        progress = is_progress_record(rec)
        if not exec_id and progress:
            m = _PROGRESS_KEY.match(idem)
            parent = ex.by_step.get(m.group(1)) if m else None
            exec_id = parent or latest
        if not exec_id:
            exec_id = step_id

        ex.by_step[step_id] = exec_id
        if not progress:
            ex.latest_by_node[node_id] = exec_id
        if idem:
            ex.idem_by_exec.setdefault(exec_id, set()).add(idem)
        res = rec.get("result") if isinstance(rec.get("result"), dict) else {}
        wait = res.get("wait") if isinstance(res.get("wait"), dict) else None
        wait_key = _s((wait or {}).get("wait_key"))
        if _status(rec) == "waiting" and wait_key:
            ex.by_wait_key[wait_key] = exec_id
        return exec_id
