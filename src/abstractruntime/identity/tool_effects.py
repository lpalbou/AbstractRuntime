"""Entity tool surface as effects (flow c5285 ask 3 — the P0 tools gap).

The flow-brain cognition turn is an observable graph: the LLM call is a real
LLM_CALL node and the tool loop must stay IN the graph, not inside a nested
effect. Two effects, flow's shape (a), and the decisive argument is stronger
than observability: a single loop-owning effect would nest provider calls
inside a tool handler, re-routing LLM traffic around every LLM_CALL
invariant (prompt-cache fingerprinting, patience windows, params lane,
ledger LLM records) — shape (b) was rejected as wrong, not just opaque.

- ``ENTITY_TOOLS_QUERY``   -> ``resolve_tool_grant`` + ``native_tool_specs``:
  the run's phase grant and its native declaration payloads. Pure read.
- ``ENTITY_TOOLS_EXECUTE`` -> ``native_tool_elections`` +
  ``execute_tool_elections``: ONE batch of wire-shape tool_calls executed
  under the grant. The grant is RE-RESOLVED here — the executor never
  trusts a caller-carried list (one authority across lanes, 2026-07-11).
  Rounds live in the FLOW graph (the flow's own loop budget; the RULED
  bounds are tools.MAX_TOOL_BLOCKS_PER_TURN = 20 calls/turn threaded by the
  caller, rounds <= 20 — flow c5323 corrected this docstring's stale "<=3"
  words); this handler executes exactly one batch per dispatch.

Compose-not-reimplement: both handlers are thin folds over the chat
driver's own machinery (identity/tools.py + tool_policy.py) — the same
grant resolver, the same election fold, the same executor, the same
HomeMemoryReader. Registered ONLY through ``open_home`` (workplaces stay
structurally handler-less, the DIARY_* law); results are prompt-currency
for the NEXT LLM round and rest in the ledger like any effect result
(tool results were ruled operator-visible, never truncated).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.models import Effect, EffectType, RunState
from ..core.runtime import EffectHandler, EffectOutcome

# One batch is one model reply's worth of calls; the flow's loop owns rounds.
# DEFAULT when the payload names no cap. The CEILING is the ruled per-turn
# budget (tools.MAX_TOOL_BLOCKS_PER_TURN = 20, maintainer 2026-07-11): the
# ruled shape is callers THREADING the remaining turn budget through
# max_calls, so the clamp must admit what the budget allows in one round —
# a hard 6 ceiling would refuse a legitimate 8-call round (consolidation
# sanity-check, flow c5318).
MAX_CALLS_PER_BATCH = 6


def build_entity_tool_effect_handlers(*, home: Any) -> Dict[EffectType, EffectHandler]:
    """Build the two tool-surface handlers over an OPEN home.

    ``home`` is the ChatHome (entity_id, home_dir, ms, diary, handlers) —
    everything the execution context needs is constructible from it.
    """

    def _grant(payload: Dict[str, Any], run: Optional[RunState] = None):
        from .tool_policy import canonical_phase, resolve_tool_grant

        phase_raw = str(payload.get("phase") or "").strip()
        if not phase_raw:
            raise ValueError("payload.phase is required (visit | work | personal | sleep)")
        # DOOR-LANE DEFENSE (adversary F1): on a stamped VISIT run the phase
        # is visit BY CONSTRUCTION (the door minted `_visit` vars) — a
        # payload claiming another phase must not widen/narrow the grant.
        # The gateway's payload gates are the primary wall; this is the
        # runtime's own belt for the lane it can see structurally.
        notes: List[str] = []
        vars_ = getattr(run, "vars", None) if run is not None else None
        if isinstance(vars_, dict) and isinstance(vars_.get("_visit"), dict):
            if canonical_phase(phase_raw) != "visit":
                notes.append(
                    f"#FALLBACK payload claimed phase {phase_raw!r} on a stamped visit run; "
                    "the visit phase governs the grant"
                )
            phase_raw = "visit"
        # Pass the RAW phase down (adversary F7): resolve_tool_grant
        # normalizes legacy spellings LOUDLY into grant.notes; canonicalize
        # only for the reported phase. `enable_workspace` is deliberately
        # NOT part of this payload contract (adversary F5: the kwarg no
        # longer subtracts from defaults — workspace access follows the
        # GRANT's own tool list, one authority).
        grant = resolve_tool_grant(home.home_dir, phase_raw)
        return canonical_phase(phase_raw), grant, notes

    def _handle_query(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})
        try:
            phase, grant, gate_notes = _grant(payload, run)
        except Exception as e:  # noqa: BLE001
            return EffectOutcome.failed(f"ENTITY_TOOLS_QUERY: {e}")

        from .tools import native_tool_specs

        return EffectOutcome.completed({
            "phase": phase,
            "tools": list(grant.tools),
            "specs": native_tool_specs(tuple(grant.tools)),
            "source": getattr(grant, "source", ""),
            "workspace_enabled": bool(getattr(grant, "workspace_enabled", False)),
            "notes": gate_notes + list(grant.notes),
        })

    def _handle_execute(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})
        calls = payload.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return EffectOutcome.failed(
                "ENTITY_TOOLS_EXECUTE requires payload.tool_calls (the model reply's native "
                "tool_calls list, wire shape)"
            )
        # Degenerate-batch wall (adversary F6): the fold caps EXECUTION at
        # MAX_CALLS_PER_BATCH but iterates the whole list, minting one
        # marker per over-cap entry — a runaway reply with thousands of
        # entries would rest thousands of marker lines in run vars. A batch
        # this oversized is a broken caller, not a bigger election.
        if len(calls) > 4 * MAX_CALLS_PER_BATCH:
            return EffectOutcome.failed(
                f"ENTITY_TOOLS_EXECUTE: {len(calls)} tool_calls in one batch "
                f"(cap {MAX_CALLS_PER_BATCH}; refusing a degenerate batch outright)"
            )
        try:
            phase, grant, gate_notes = _grant(payload, run)
        except Exception as e:  # noqa: BLE001
            return EffectOutcome.failed(f"ENTITY_TOOLS_EXECUTE: {e}")

        from .memory_reader import HomeMemoryReader, feelings_about_text
        from .tools import (
            WorkspaceRoot,
            execute_tool_elections,
            native_tool_elections,
        )

        from .tools import MAX_TOOL_BLOCKS_PER_TURN

        cap = payload.get("max_calls")
        cap_n = int(cap) if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else MAX_CALLS_PER_BATCH
        # Ceiling = the RULED turn budget, so callers can thread their
        # remaining budget (the 2026-07-11 ruling's shape); default stays
        # conservative when unspecified.
        cap_n = min(cap_n, MAX_TOOL_BLOCKS_PER_TURN)

        # The election fold IS the refusal surface: ungranted names come back
        # as marker lines (shown to the model verbatim), never execute.
        elections, markers, notices = native_tool_elections(
            calls, tuple(grant.tools), max_elections=cap_n
        )

        reader = HomeMemoryReader(home)

        def _diary_read(entry_id: str) -> Dict[str, Any]:
            handler = home.handlers.get(EffectType.DIARY_READ)
            if handler is None:  # pragma: no cover - homes always carry it
                raise RuntimeError("home has no DIARY_READ handler")
            out = handler(run, Effect(type=EffectType.DIARY_READ, payload={"entry_id": entry_id}), None)
            status = getattr(out.status, "value", out.status)
            if status != "completed":
                raise RuntimeError(str(out.error or "diary read failed"))
            return out.result or {}

        workspace = WorkspaceRoot(home.home_dir) if getattr(grant, "workspace_enabled", False) else None

        results_message = ""
        exec_notices: List[str] = []
        if elections:
            try:
                results_message, exec_notices = execute_tool_elections(
                    elections,
                    diary_store=home.diary,
                    diary_read_effect=_diary_read,
                    workspace=workspace,
                    read_memory_fn=reader.read_memory,
                    search_memory_fn=reader.search_memory,
                    recent_memories_fn=reader.recent_memories,
                    # Adversary F2: granted+declared but unwired = the exact
                    # incident class this surface closes. ONE implementation
                    # with the chat driver (memory_reader.feelings_about_text).
                    feelings_about_fn=lambda t: feelings_about_text(home, t),
                    # Adversary F4: on this lane results REST (effect results
                    # are ledger truth in the home's own store) — the header/
                    # trailer must not tell the entity its words vanish.
                    results_rest_durably=True,
                )
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"ENTITY_TOOLS_EXECUTE executor failed: {e}")

        # tools_ran: unique names actually executed (set semantics — the
        # announcement-dedupe lesson); refusals are visible in markers.
        seen: List[str] = []
        for e in elections:
            if e.name not in seen:
                seen.append(e.name)
        return EffectOutcome.completed({
            "phase": phase,
            "tools_ran": seen,
            "results": [{"name": e.name, "result": e.result} for e in elections],
            "results_message": results_message,
            "markers": markers,
            "notices": gate_notes + list(notices) + list(exec_notices),
        })

    return {
        EffectType.ENTITY_TOOLS_QUERY: _handle_query,
        EffectType.ENTITY_TOOLS_EXECUTE: _handle_execute,
    }
