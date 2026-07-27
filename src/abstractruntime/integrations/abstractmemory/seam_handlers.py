"""Runtime handlers for the usage-weighted-graph memory seam (v1).

These bridge Runtime effects to the AbstractMemory `MemorySystem` facade
negotiated in `a2a/threads/0001-runtime-memory-orchestration/004-memory--to--runtime.md`:

- `MEMORY_RECALL`  -> `MemorySystem.reconstruct(view="working_set")` : stimulus-driven
  reconstruction of the emergent working set. The runtime owns WHEN to call
  (turn start, or budget pressure); the memory package owns the graph/activation.
- `MEMORY_ACCESS`  -> `MemorySystem.commit_selection(...)` : the Hebbian trail
  deposit for the records that ACTUALLY entered the prompt. This is the memory
  seam's only strengthening path — the runtime never posts `selected` events
  directly.
- `MEMORY_FORM`    -> `MemorySystem.remember_many(...)` : per-turn formation of
  the `verbatim <-> digest` records that feed selection (the graph-feeder that
  replaces compaction-as-sole-writer). Full verbatim goes to the RUNTIME
  ArtifactStore; memory receives only a `payload_ref` (a2a 0001/008 ask 1).
  Idempotent by `(idempotency_key, position)` — replays return the same
  record_ids and write nothing, so the runtime's at-least-once execution is
  safe against the append-only graph.
- `MEMORY_ADJUST`  -> `MemorySystem.reinforce/attenuate/refocus/close_record` :
  ACTIVE remembering — the agent deliberately reshapes salience or retracts a
  record. One parameterized effect (`op`) with a mandatory `reason` (audited).
  Replay-safe: a `turn_id`-derived `event_id` rides memory's supplied-id journal
  dedup so an at-least-once replay of an additive salience write is a no-op
  (a2a 0001/016). NOTE: `op=close` is a destructive belief-revision act and is
  intended for host/flow use, not the model tool surface in v1 (builtin effect
  tools bypass tool approval — active-reconstruction charter finding).
- `MEMORY_APPRAISE` -> `MemorySystem.appraise/heal_scar/break_bond/gradation` :
  the affect/valence seam (a2a 0003 identity wave). Signed appraisals of
  experience against the entity's values, standing peaks (scar/bond), their
  append-only resolutions, and the derived dual-channel gradation read.
  Valence is orthogonal to attention and never gates recall.

The SITUATION contract (co-signed seam note, a2a 0001/20260707T014610Z +
co-sign): grounding — when/where/on-what/WITH WHOM — flows as STIMULUS,
never identity. Hosts stamp verified participants into
`payload.participants` (the door verifies WHO; payloads never claim it);
the participants channel scores shared context ("what WE lived together").
Situation stimuli deposit like any stimulus — the situation shapes the
trail without ever occupying the identity's reserved seats: it moves you,
it does not define you. Prompt-side situation rendering stays guard-gated
(the 2026-06-10 language-flip finding).

Placement rationale (boundary charter, ADR-0001): these are host-provided
integration handlers, NOT kernel effects. The kernel only owns the `EffectType`
names.

Determinism/replay: `reconstruct` results are JSON-safe and returned verbatim as
the effect result, so the runtime persists them in its ledger and replays from
that recorded result (memory's `as_of` anchors journal signals only; the truth
store has no seq axis — see `MemorySystem` docstring and a2a 0001/011 ask 5).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...core.models import Effect, EffectType, RunState
from ...core.runtime import EffectHandler, EffectOutcome
from ...storage.base import RunStore
from .effect_handlers import resolve_scope_owner_id


def _import_seam():
    """Import the frozen seam dataclasses lazily (keeps the module import-light
    and gives an actionable error when AbstractMemory is absent)."""
    try:
        from abstractmemory.seam import RecallBudget, Stimulus  # type: ignore

        return Stimulus, RecallBudget
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "AbstractMemory seam is not available. Install and ensure it is importable "
            "(e.g. `pip install -e abstractmemory`) so MEMORY_RECALL/MEMORY_ACCESS can bind."
        ) from e


# Effort -> RecallBudget shortcut. The maintainer's model (a2a 0001/007) makes the
# budget effort-adaptive on (time, importance): under pressure select a few and
# deep-dive later; with time/importance/sleep use a larger budget to discover
# links. This is the minimal, overridable map; an explicit `budget` dict in the
# payload always wins. Full (time, importance) policy is a later runtime concern.
#
# `stm_fraction` (seam v1.1, maintainer's union model) is the STM continuity dial:
# how much of the working set the stimulus-independent trail-hot component may
# claim. Higher effort => more room for continuity + link discovery (memory's
# 0001/012 ask 2 suggested URGENT 0.15 / STANDARD 0.25 / DEEP 0.35). It and
# `decay_window` (engine-side) are the two load-bearing dials.
_EFFORT_BUDGET: Dict[str, Dict[str, Any]] = {
    "urgent": {"max_candidates": 24, "shelf_size": 5, "token_budget": 800, "max_hops": 1, "max_edges": 40, "stm_fraction": 0.15},
    "standard": {"max_candidates": 64, "shelf_size": 12, "token_budget": 2400, "max_hops": 2, "max_edges": 100, "stm_fraction": 0.25},
    "deep": {"max_candidates": 128, "shelf_size": 24, "token_budget": 6000, "max_hops": 2, "max_edges": 200, "stm_fraction": 0.35},
}


def _coerce_str_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        v = value.strip()
        return (v,) if v else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(x).strip() for x in value if isinstance(x, (str,)) and str(x).strip())
    return ()


# Sentinel distinguishing "field absent" from "field present but uncoercible".
_INVALID = object()


def _coerce_bool(value: Any):
    """Boolean coercion for tool-call args. In Python non-empty strings are
    truthy, so a payload `scar="false"` would silently mark a standing trauma
    (AGENTS.md 2026-02-20). Returns the bool, or `_INVALID` when
    present-but-uncoercible; None/absent coerces to False."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no", ""):
            return False
    return _INVALID


def _coerce_number(value: Any, *, kind: type):
    """Schema-aware numeric coercion for tool-call args (which frequently arrive
    as strings — see AGENTS.md 2026-02-20). Returns None when absent, the coerced
    number when valid, or `_INVALID` when present-but-uncoercible. NEVER silently
    substitutes a default or drops a bad value: for salience `weight` a wrong
    value skews the graph, and for `ttl_activity` a dropped string would invert to
    'never expires' — both must fail loudly, not degrade silently."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        return _INVALID
    if isinstance(value, (int, float)):
        return kind(value)
    if isinstance(value, str) and value.strip():
        try:
            return kind(float(value.strip())) if kind is int else kind(value.strip())
        except (TypeError, ValueError):
            return _INVALID
    return _INVALID


def _default_scope(run: RunState) -> str:
    """The default memory scope when a caller does not specify one.

    Sessions are the natural memory horizon: a gateway runs ONE run per incoming
    message, so a `run`-scoped default would silo every turn's memory (per-message
    amnesia — consolidated-seam-review finding). Defaulting to `session` when the
    run carries a session_id makes FORM/ADJUST/RECALL agree on where memory lives;
    a standalone long-lived run with no session_id keeps `run` scope. Callers
    (e.g. a host that wants global memory) still pass scope explicitly."""
    sid = getattr(run, "session_id", None)
    return "session" if isinstance(sid, str) and sid.strip() else "run"


# Entity-plane scope names: on a home-bound seam (open_home passes the home's
# entity id), BARE names resolve to the home owner — the channel fills
# authorship so callers (notably entity-brain VisualFlows) never carry an
# entity id in their payloads (the deposit-gate rule applied to scoping).
_ENTITY_SCOPES = ("self", "diary", "life")


def _resolve_scope_pairs(
    run: RunState, raw_scopes: Any, *, run_store: RunStore,
    entity_scope_owner: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Turn a JSON scope spec into concrete (scope, owner_id) pairs.

    Accepts either scope names (["run", "session"]) whose owner_id the runtime
    resolves from the run, or explicit [scope, owner_id] pairs (used for
    cross-run recall). Bare ENTITY scope names (self/diary/life) resolve to
    `entity_scope_owner` when the seam is home-bound. Defaults to the current
    run scope.
    """

    def _owner_for(scope: str) -> str:
        if entity_scope_owner and scope in _ENTITY_SCOPES:
            return entity_scope_owner
        return resolve_scope_owner_id(run, scope=scope, run_store=run_store)

    if not raw_scopes:
        default = _default_scope(run)
        return [(default, _owner_for(default))]
    pairs: List[Tuple[str, str]] = []
    for entry in raw_scopes:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            scope = str(entry[0]).strip().lower()
            owner = str(entry[1]).strip()
            if scope:
                pairs.append((scope, owner))
        elif isinstance(entry, str) and entry.strip():
            scope = entry.strip().lower()
            pairs.append((scope, _owner_for(scope)))
    if not pairs:
        default = _default_scope(run)
        return [(default, _owner_for(default))]
    return pairs


def _build_budget(
    RecallBudget: Any, payload: Dict[str, Any], *, entity_scope_owner: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """RecallBudget from an explicit `budget` dict, else an `effort` shortcut,
    else the seam default. Returns (budget, warnings).

    Loud, never silent (keystone audit G1): an explicit budget the constructor
    rejects raises ValueError naming the failure — a silently-defaulted budget
    zeroes `self_fraction` and would make a re-adoption probe read "identity
    lost" for a reason that isn't identity. Unknown keys are dropped for
    forward-compatibility but each drop is a labeled warning."""
    explicit = payload.get("budget")
    fields: Dict[str, Any] = {}
    if isinstance(explicit, dict):
        fields = dict(explicit)
    else:
        effort = str(payload.get("effort") or "standard").strip().lower()
        fields = dict(_EFFORT_BUDGET.get(effort, _EFFORT_BUDGET["standard"]))
        if entity_scope_owner:
            # Identity present by right (maintainer law): a home-bound recall
            # without an explicit budget seats the self core at the summon
            # posture — effort presets alone left self_fraction at 0.0, so
            # every flow-lane recall ran identity-blind (adversary-2 P0-1).
            fields.setdefault("self_fraction", 0.5)

    warnings: List[str] = []
    allowed = getattr(RecallBudget, "__dataclass_fields__", {})
    filtered: Dict[str, Any] = {}
    for k, v in fields.items():
        if k in allowed:
            filtered[k] = v
        else:
            warnings.append(f"#FALLBACK budget field {k!r} is not a RecallBudget field; dropped")
    try:
        return RecallBudget(**filtered), warnings
    except Exception as e:
        if isinstance(explicit, dict):
            raise ValueError(f"invalid recall budget {explicit!r}: {e}") from e
        # Effort presets are code-owned; a failure here is drift, not caller error.
        warnings.append(f"#FALLBACK effort preset budget rejected ({e}); using seam defaults")
        return RecallBudget(), warnings


def _import_record_input():
    try:
        from abstractmemory.records import MemoryRecordInput  # type: ignore

        return MemoryRecordInput
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "AbstractMemory records API is not available (needs abstractmemory with "
            "remember_many/MemoryRecordInput; see a2a 0001/008). Update abstractmemory."
        ) from e


def build_memory_seam_effect_handlers(
    *,
    memory_system: Any,
    run_store: RunStore,
    now_iso: Callable[[], str],
    artifact_store: Any = None,
    strict: bool = True,
    entity_scope_owner: Optional[str] = None,
) -> Dict[EffectType, EffectHandler]:
    """Build `MEMORY_RECALL` / `MEMORY_ACCESS` / `MEMORY_FORM` / `MEMORY_ADJUST`
    handlers over a `MemorySystem`.

    `memory_system` is duck-typed to the seam facade (`reconstruct`,
    `commit_selection`, `remember_many`, deliberate acts). `artifact_store` is
    the RUNTIME-owned store for full-verbatim payloads; when absent, MEMORY_FORM
    still works but cannot attach `payload_ref` for inline verbatim text (a
    labeled degradation).

    `strict` (default True) controls the blast radius of a memory failure:
    - `strict=True` (dev/tests/flows that need correctness): any failure returns
      an `EffectOutcome.failed` — loud, so bugs surface.
    - `strict=False` (the live-agent posture the gateway wiring sets, R1 from the
      wiring plan): a memory failure (locked seam db, engine exception, even a
      malformed effect) is DOWNGRADED to a labeled `#FALLBACK` *completed*
      outcome with an empty/degraded result, so a memory hiccup can never abort
      the flagship agent's turn. The degradation is always visible in the
      result's `warnings` + `degraded: true`.
    """
    del now_iso  # reserved for future formation-run wiring; unused in v1
    if not callable(getattr(memory_system, "reconstruct", None)) or not callable(
        getattr(memory_system, "commit_selection", None)
    ):
        raise TypeError(
            "memory_system must provide reconstruct(...) and commit_selection(...) "
            "(abstractmemory MemorySystem seam v1 contract)"
        )

    Stimulus, RecallBudget = _import_seam()

    def _scope_owner(run: RunState, scope: str) -> str:
        # Home-bound seams resolve bare entity scopes to the home owner
        # (channel authority: payloads never carry the entity id).
        if entity_scope_owner and scope in _ENTITY_SCOPES:
            return entity_scope_owner
        return resolve_scope_owner_id(run, scope=scope, run_store=run_store)

    def _handle_recall(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})

        try:
            budget, budget_warnings = _build_budget(
                RecallBudget, payload, entity_scope_owner=entity_scope_owner
            )
        except ValueError as e:
            return EffectOutcome.failed(f"MEMORY_RECALL: {e}")

        cue_text = str(payload.get("cue_text") or "").strip()
        has_cue = bool(cue_text or payload.get("anchor_record_ids") or payload.get("patterns"))
        # A cue-free read is legal when the SELF component is enabled: the
        # identity core admits by binding STATE, not stimulus (keystone audit
        # G3 — "render the self" is exactly a blank-cue reconstruct).
        if not has_cue and not float(getattr(budget, "self_fraction", 0.0) or 0.0) > 0.0:
            return EffectOutcome.failed(
                "MEMORY_RECALL requires a cue: payload.cue_text (or anchor_record_ids / patterns) — "
                "unless budget.self_fraction > 0 (a cue-free self-core read)"
            )

        view = str(payload.get("view") or "working_set").strip().lower()
        if view not in ("working_set", "shelf"):
            return EffectOutcome.failed(f"MEMORY_RECALL: unknown view {view!r} (use 'working_set' or 'shelf')")

        embedding = payload.get("embedding")
        embedding_tuple = tuple(float(x) for x in embedding) if isinstance(embedding, (list, tuple)) else None
        as_of = payload.get("as_of")
        turn_id = payload.get("turn_id")

        stimulus = Stimulus(
            cue_text=cue_text,
            patterns=tuple(p for p in (payload.get("patterns") or ()) if isinstance(p, dict)),
            anchor_record_ids=_coerce_str_tuple(payload.get("anchor_record_ids")),
            participants=_coerce_str_tuple(payload.get("participants")),
            embedding=embedding_tuple,
            as_of=int(as_of) if isinstance(as_of, int) else None,
            turn_id=str(turn_id).strip() if isinstance(turn_id, str) and turn_id.strip() else None,
        )
        scope_pairs = _resolve_scope_pairs(run, payload.get("scopes"), run_store=run_store, entity_scope_owner=entity_scope_owner)
        escalation_reason = payload.get("escalation_reason")
        journal = payload.get("journal", True)

        # Replay safety (keystone audit G4): an at-least-once replay of this
        # effect must not journal a second trace. Derive a deterministic
        # trace_id when the caller gave a turn_id; an explicit payload
        # trace_id always wins; without either, memory generates one (legacy).
        trace_id = str(payload.get("trace_id") or "").strip() or None
        if trace_id is None and stimulus.turn_id:
            run_id = str(getattr(run, "run_id", "") or "")
            basis = f"recall|{run_id}|{stimulus.turn_id}|{cue_text}|{view}|{bool(journal)}"
            trace_id = "trace_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]

        try:
            result = memory_system.reconstruct(
                stimulus,
                scopes=scope_pairs,
                budget=budget,
                view=view,
                escalation_reason=str(escalation_reason).strip() if isinstance(escalation_reason, str) else None,
                journal=bool(journal),
                trace_id=trace_id,
            )
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_RECALL reconstruct failed: {e}")

        # JSON-safe result -> ledger truth (runtime replays from this recorded
        # result; it does not re-call reconstruct on replay, per a2a 0001/011 #5).
        out = result.to_dict() if hasattr(result, "to_dict") else result
        if isinstance(out, dict) and budget_warnings:
            out["warnings"] = list(out.get("warnings") or []) + budget_warnings
        return EffectOutcome.completed(out)

    def _handle_access(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node, run
        payload = dict(effect.payload or {})

        trace_id = str(payload.get("trace_id") or "").strip()
        if not trace_id:
            return EffectOutcome.failed("MEMORY_ACCESS requires payload.trace_id (from the MEMORY_RECALL result)")

        used = payload.get("used_record_ids")
        used_ids = [str(x).strip() for x in used if isinstance(x, str) and str(x).strip()] if isinstance(used, (list, tuple)) else []
        if not used_ids:
            # An empty selection deposits no trail; treat as an explicit no-op success
            # rather than an error, so callers can commit unconditionally.
            return EffectOutcome.completed({"committed": 0, "trace_id": trace_id, "skipped": "no used_record_ids"})

        prompt_token_estimate = payload.get("prompt_token_estimate")
        try:
            snapshot = memory_system.commit_selection(
                trace_id,
                used_ids,
                prompt_token_estimate=int(prompt_token_estimate) if isinstance(prompt_token_estimate, int) else None,
            )
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_ACCESS commit_selection failed: {e}")

        out = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
        if isinstance(out, dict):
            out.setdefault("committed", len(used_ids))
        return EffectOutcome.completed(out)

    def _handle_form(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        if not callable(getattr(memory_system, "remember_many", None)):
            return EffectOutcome.failed(
                "MEMORY_FORM requires MemorySystem.remember_many (update abstractmemory; a2a 0001/008)"
            )
        payload = dict(effect.payload or {})

        raw_records = payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            return EffectOutcome.failed("MEMORY_FORM requires payload.records (non-empty list of record dicts)")

        scope = str(payload.get("scope") or "").strip().lower() or _default_scope(run)
        owner_override = payload.get("owner_id")
        if isinstance(owner_override, str) and owner_override.strip():
            owner_id = owner_override.strip()
        else:
            try:
                owner_id = _scope_owner(run, scope)
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_FORM could not resolve owner for scope {scope!r}: {e}")

        turn_id = str(payload.get("turn_id") or "").strip()
        run_id = str(getattr(run, "run_id", "") or "").strip()
        if not turn_id:
            return EffectOutcome.failed(
                "MEMORY_FORM requires payload.turn_id (it derives the idempotency key that makes "
                "at-least-once replays safe against the append-only graph)"
            )
        # Default idempotency key is CONTENT-AWARE: `run:turn` alone would alias
        # two different formation batches in the same turn (the second batch
        # would silently dedup against the first — records lost, dangling ids
        # returned as success). Hashing the raw batch keeps at-least-once replays
        # of the SAME effect payload idempotent while distinct batches in one
        # turn write independently. An explicit payload.idempotency_key wins.
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key:
            batch_fp = hashlib.sha256(
                json.dumps(raw_records, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()[:16]
            idempotency_key = f"{run_id}:{turn_id}:{batch_fp}"

        MemoryRecordInput = _import_record_input()
        warnings: List[str] = []
        inputs = []
        for i, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                return EffectOutcome.failed(f"MEMORY_FORM records[{i}] must be an object")
            rec = dict(raw)

            # Full verbatim goes to the RUNTIME artifact store; memory gets a ref.
            payload_ref = rec.get("payload_ref")
            verbatim = rec.pop("verbatim", None)
            if payload_ref is None and isinstance(verbatim, str) and verbatim.strip():
                if artifact_store is not None and callable(getattr(artifact_store, "store_text", None)):
                    meta = artifact_store.store_text(
                        verbatim,
                        run_id=run_id or None,
                        tags={"kind": "memory_verbatim", "turn_id": turn_id},
                    )
                    payload_ref = getattr(meta, "artifact_id", None) or (
                        meta.get("artifact_id") if isinstance(meta, dict) else None
                    )
                else:
                    warnings.append(
                        f"#FALLBACK records[{i}]: no artifact_store wired; verbatim dropped, digest-only formation"
                    )

            provenance = dict(rec.get("provenance") or {})
            provenance.setdefault("run_id", run_id)
            provenance.setdefault("turn_id", turn_id)

            try:
                inputs.append(
                    MemoryRecordInput(
                        kind=str(rec.get("kind") or "memory"),
                        title=str(rec.get("title") or ""),
                        digest=str(rec.get("digest") or ""),
                        intents=_coerce_str_tuple(rec.get("intents")),
                        outcomes=_coerce_str_tuple(rec.get("outcomes")),
                        keywords=_coerce_str_tuple(rec.get("keywords")),
                        edges=tuple(
                            (str(e[0]).strip(), str(e[1]).strip())
                            for e in (rec.get("edges") or ())
                            if isinstance(e, (list, tuple)) and len(e) == 2
                        ),
                        payload_ref=str(payload_ref) if payload_ref else None,
                        topic=str(rec["topic"]).strip() if isinstance(rec.get("topic"), str) and rec["topic"].strip() else None,
                        confidence=float(rec["confidence"]) if isinstance(rec.get("confidence"), (int, float)) else None,
                        attributes=dict(rec.get("attributes") or {}),
                        provenance=provenance,
                    )
                )
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_FORM records[{i}] invalid: {e}")

        try:
            record_ids = memory_system.remember_many(
                inputs, scope=scope, owner_id=owner_id, idempotency_key=idempotency_key, turn_id=turn_id
            )
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_FORM remember_many failed: {e}")

        result: Dict[str, Any] = {
            "record_ids": list(record_ids),
            "formed": len(record_ids),
            "scope": scope,
            "owner_id": owner_id,
            "turn_id": turn_id,
            "idempotency_key": idempotency_key,
        }
        if warnings:
            result["warnings"] = warnings
        return EffectOutcome.completed(result)

    def _handle_adjust(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = dict(effect.payload or {})

        op = str(payload.get("op") or "").strip().lower()
        if op not in ("reinforce", "attenuate", "refocus", "close"):
            return EffectOutcome.failed(
                "MEMORY_ADJUST requires op in {reinforce, attenuate, refocus, close}"
            )

        # Deliberate acts are audited: reason is mandatory (memory-side contract).
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            return EffectOutcome.failed(f"MEMORY_ADJUST op={op} requires a non-empty reason (deliberate acts are audited)")

        # turn_id derives the replay-idempotency key: runtime effects are
        # at-least-once, and reinforce/attenuate are ADDITIVE salience writes —
        # a replayed boost without a stable event_id would skew activation
        # permanently. close_record is already deterministic (memory 0001/016),
        # but we still require turn_id for a uniform, auditable contract.
        turn_id = str(payload.get("turn_id") or "").strip()
        if not turn_id:
            return EffectOutcome.failed(
                f"MEMORY_ADJUST op={op} requires turn_id (it derives the idempotency key that makes "
                "at-least-once replays safe against additive salience writes)"
            )

        record_id = str(payload.get("record_id") or "").strip()
        if op != "refocus" and not record_id:
            return EffectOutcome.failed(f"MEMORY_ADJUST op={op} requires record_id")

        scope = str(payload.get("scope") or "").strip().lower() or _default_scope(run)
        owner_override = payload.get("owner_id")
        if isinstance(owner_override, str) and owner_override.strip():
            owner_id = owner_override.strip()
        else:
            try:
                owner_id = _scope_owner(run, scope)
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_ADJUST could not resolve owner for scope {scope!r}: {e}")

        # Salience/ttl coercion must be LOUD, not silent (consolidated-seam-review):
        # a wrong weight skews the graph; a dropped ttl string inverts to
        # "never expires". Tool-call args often arrive as strings (AGENTS.md).
        weight = _coerce_number(payload.get("weight", 8), kind=float)
        if weight is _INVALID:
            return EffectOutcome.failed("MEMORY_ADJUST weight must be a number (1..25)")
        if weight is None:
            weight = 8.0
        ttl = _coerce_number(payload.get("ttl_activity"), kind=int)
        if ttl is _INVALID:
            return EffectOutcome.failed("MEMORY_ADJUST ttl_activity must be an integer (activity units)")

        method = getattr(memory_system, op if op != "close" else "close_record", None)
        if not callable(method):
            return EffectOutcome.failed(
                f"MEMORY_ADJUST op={op} requires MemorySystem.{op if op != 'close' else 'close_record'} "
                "(update abstractmemory; a2a 0001/016)"
            )

        run_id = str(getattr(run, "run_id", "") or "").strip()
        provenance = {"run_id": run_id, "turn_id": turn_id}
        # Stable event id so a replayed effect is a journal no-op (memory dedups
        # by supplied id). The key includes scope+owner_id because memory's
        # supplied-id dedup is GLOBAL (consolidated-seam-review): without them,
        # two same-turn adjusts to the same record in DIFFERENT scopes would
        # derive the same event_id and the second would be silently swallowed.
        # close_record ignores event_id (its closure ids are deterministic).
        event_id = hashlib.sha256(
            f"{op}|{scope}|{owner_id}|{record_id}|{run_id}|{turn_id}|{reason}".encode("utf-8")
        ).hexdigest()

        try:
            if op == "reinforce" or op == "attenuate":
                out_id = method(
                    record_id, reason=reason, weight=weight, ttl_activity=ttl,
                    scope=scope, owner_id=owner_id,
                    event_id=event_id, actor="runtime", provenance=provenance,
                )
                result: Dict[str, Any] = {"op": op, "event_id": out_id, "record_id": record_id}
            elif op == "refocus":
                out_id = method(
                    reason=reason, scope=scope, owner_id=owner_id,
                    event_id=event_id, actor="runtime", provenance=provenance,
                )
                result = {"op": op, "event_id": out_id}
            else:  # close
                kind = str(payload.get("kind") or "retract").strip().lower() or "retract"
                repl = payload.get("replacement_ids")
                replacement_ids = tuple(str(x).strip() for x in repl if isinstance(x, str) and str(x).strip()) if isinstance(repl, (list, tuple)) else ()
                closure_ids = method(record_id, reason=reason, kind=kind, replacement_ids=replacement_ids)
                result = {"op": op, "record_id": record_id, "kind": kind, "closure_ids": list(closure_ids)}
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_ADJUST op={op} failed: {e}")

        result.update({"turn_id": turn_id, "scope": scope, "owner_id": owner_id})
        return EffectOutcome.completed(result)

    def _handle_appraise(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """MEMORY_APPRAISE — the affect/valence seam (a2a 0003 identity wave).

        Ops: `appraise` (signed deposit, optionally with an explicit standing
        peak: scar=- / bond=+), `heal_scar` / `break_bond` (append-only
        resolution acts), `gradation` (pure read of the derived dual-channel
        standing). Valence is orthogonal to attention and NEVER gates recall
        (memory enforces this by construction; this handler only deposits).

        Actor honesty: `actor` is payload-supplied until the gateway stamps
        channels (same documented fail-open as tool-tier run_mode). The
        engine's amplitude authority still bounds the damage: magnitude > 3
        with a non-privileged actor fails loudly memory-side.
        """
        del default_next_node
        payload = dict(effect.payload or {})

        op = str(payload.get("op") or "appraise").strip().lower()
        if op not in ("appraise", "heal_scar", "break_bond", "gradation"):
            return EffectOutcome.failed(
                "MEMORY_APPRAISE requires op in {appraise, heal_scar, break_bond, gradation}"
            )

        scope = str(payload.get("scope") or "").strip().lower() or _default_scope(run)
        owner_override = payload.get("owner_id")
        if isinstance(owner_override, str) and owner_override.strip():
            owner_id = owner_override.strip()
        else:
            try:
                owner_id = _scope_owner(run, scope)
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_APPRAISE could not resolve owner for scope {scope!r}: {e}")

        if op == "gradation":
            if not callable(getattr(memory_system, "gradation", None)):
                return EffectOutcome.failed(
                    "MEMORY_APPRAISE op=gradation requires MemorySystem.gradation (update abstractmemory)"
                )
            raw_targets = payload.get("target_ids")
            target_ids = None
            if isinstance(raw_targets, (list, tuple)):
                target_ids = [str(t).strip() for t in raw_targets if str(t).strip()]
            at_seq = _coerce_number(payload.get("at_seq"), kind=int)
            if at_seq is _INVALID:
                return EffectOutcome.failed("MEMORY_APPRAISE at_seq must be an integer")
            try:
                grades = memory_system.gradation(target_ids, scope=scope, owner_id=owner_id, at_seq=at_seq)
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_APPRAISE op=gradation failed: {e}")
            return EffectOutcome.completed(
                {"op": op, "gradations": grades, "scope": scope, "owner_id": owner_id}
            )

        # All write ops are deliberate acts: reason is mandatory (audited).
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            return EffectOutcome.failed(f"MEMORY_APPRAISE op={op} requires a non-empty reason (appraisals are audited)")

        if op in ("heal_scar", "break_bond"):
            key = "scar_event_id" if op == "heal_scar" else "bond_event_id"
            marker_id = str(payload.get(key) or "").strip()
            if not marker_id:
                return EffectOutcome.failed(f"MEMORY_APPRAISE op={op} requires {key}")
            method = getattr(memory_system, op, None)
            if not callable(method):
                return EffectOutcome.failed(
                    f"MEMORY_APPRAISE op={op} requires MemorySystem.{op} (update abstractmemory)"
                )
            kwargs: Dict[str, Any] = {"reason": reason, "scope": scope, "owner_id": owner_id}
            if op == "heal_scar":
                lesson = str(payload.get("lesson_record_id") or "").strip()
                if lesson:
                    kwargs["lesson_record_id"] = lesson
            try:
                # Resolution ids are deterministic memory-side (heal:{id} /
                # break:{id}), so at-least-once replays are journal no-ops.
                out_id = method(marker_id, **kwargs)
            except Exception as e:
                return EffectOutcome.failed(f"MEMORY_APPRAISE op={op} failed: {e}")
            return EffectOutcome.completed(
                {"op": op, "event_id": out_id, key: marker_id, "scope": scope, "owner_id": owner_id}
            )

        # op == "appraise"
        if not callable(getattr(memory_system, "appraise", None)):
            return EffectOutcome.failed(
                "MEMORY_APPRAISE requires MemorySystem.appraise (update abstractmemory; a2a 0003 identity wave)"
            )

        target_id = str(payload.get("target_id") or "").strip()
        if not target_id:
            return EffectOutcome.failed("MEMORY_APPRAISE op=appraise requires target_id (record id or identity string)")

        turn_id = str(payload.get("turn_id") or "").strip()
        if not turn_id:
            return EffectOutcome.failed(
                "MEMORY_APPRAISE op=appraise requires turn_id (it derives the event id that makes "
                "at-least-once replays safe against the append-only valence journal)"
            )

        sign = _coerce_number(payload.get("sign"), kind=int)
        if sign is _INVALID or sign not in (-1, 1):
            return EffectOutcome.failed("MEMORY_APPRAISE sign must be -1 or +1")

        magnitude = _coerce_number(payload.get("magnitude", 1), kind=float)
        if magnitude is _INVALID or magnitude is None or magnitude <= 0:
            return EffectOutcome.failed("MEMORY_APPRAISE magnitude must be a positive number (1..10)")

        scar = _coerce_bool(payload.get("scar"))
        bond = _coerce_bool(payload.get("bond"))
        if scar is _INVALID or bond is _INVALID:
            return EffectOutcome.failed("MEMORY_APPRAISE scar/bond must be booleans")

        actor = str(payload.get("actor") or "runtime").strip() or "runtime"
        run_id = str(getattr(run, "run_id", "") or "").strip()
        provenance = dict(payload.get("provenance") or {})
        provenance.setdefault("run_id", run_id)
        provenance.setdefault("turn_id", turn_id)

        # Stable event id (same collision reasoning as MEMORY_ADJUST: memory's
        # supplied-id dedup is global, so scope+owner participate). Memory
        # derives the paired marker id as "{event_id}:scar"/":bond" itself.
        event_id = hashlib.sha256(
            f"appraise|{scope}|{owner_id}|{target_id}|{run_id}|{turn_id}|{reason}".encode("utf-8")
        ).hexdigest()

        try:
            event_ids = memory_system.appraise(
                target_id,
                sign=int(sign),
                magnitude=float(magnitude),
                reason=reason,
                scope=scope,
                owner_id=owner_id,
                value_refs=_coerce_str_tuple(payload.get("value_refs")),
                scar=bool(scar),
                bond=bool(bond),
                event_id=event_id,
                actor=actor,
                provenance=provenance,
            )
        except Exception as e:
            return EffectOutcome.failed(f"MEMORY_APPRAISE op=appraise failed: {e}")

        return EffectOutcome.completed(
            {
                "op": op,
                "event_ids": list(event_ids),
                "target_id": target_id,
                "sign": int(sign),
                "magnitude": float(magnitude),
                "scar": bool(scar),
                "bond": bool(bond),
                "turn_id": turn_id,
                "scope": scope,
                "owner_id": owner_id,
            }
        )

    # R1 (wiring plan): a memory failure must never kill a live turn. In
    # non-strict mode, wrap each handler so a failed outcome (or a raised
    # exception) becomes a labeled, degraded *completed* outcome. The degraded
    # payload is shaped so the agent loop can proceed as if memory returned
    # nothing this turn. In strict mode this wrapper is transparent.
    def _degraded_for(etype: EffectType, effect: Effect) -> Dict[str, Any]:
        payload = effect.payload or {}
        if etype == EffectType.MEMORY_RECALL:
            return {
                "view": str(payload.get("view") or "working_set"),
                "trace_id": "", "as_of_seq": -1, "handles": [], "edges": [],
                "dropped": [], "selector_route": "degraded", "stop_reason": "degraded",
                "budget_spent": {},
            }
        if etype == EffectType.MEMORY_ACCESS:
            return {"committed": 0, "trace_id": str(payload.get("trace_id") or "")}
        if etype == EffectType.MEMORY_FORM:
            return {"formed": 0, "record_ids": []}
        if etype == EffectType.MEMORY_ADJUST:
            return {"op": str(payload.get("op") or "")}
        if etype == EffectType.MEMORY_APPRAISE:
            op = str(payload.get("op") or "appraise")
            if op == "gradation":
                return {"op": op, "gradations": {}}
            return {"op": op, "event_ids": []}
        return {}

    def _resilient(etype: EffectType, handler: EffectHandler) -> EffectHandler:
        def wrapped(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
            try:
                outcome = handler(run, effect, default_next_node)
            except Exception as e:  # defense in depth: a handler must not raise into the runtime
                outcome = EffectOutcome.failed(f"{etype.value} raised: {e}")
            if strict or getattr(outcome, "status", None) != "failed":
                return outcome
            degraded = _degraded_for(etype, effect)
            warns = list(degraded.get("warnings") or [])
            warns.append(f"#FALLBACK {etype.value} degraded (strict=False): {getattr(outcome, 'error', 'unknown error')}")
            degraded["warnings"] = warns
            degraded["degraded"] = True
            return EffectOutcome.completed(degraded)

        return wrapped

    raw = {
        EffectType.MEMORY_RECALL: _handle_recall,
        EffectType.MEMORY_ACCESS: _handle_access,
        EffectType.MEMORY_FORM: _handle_form,
        EffectType.MEMORY_ADJUST: _handle_adjust,
        EffectType.MEMORY_APPRAISE: _handle_appraise,
    }
    return {etype: _resilient(etype, handler) for etype, handler in raw.items()}


__all__ = ["build_memory_seam_effect_handlers"]
