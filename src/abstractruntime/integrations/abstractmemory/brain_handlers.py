"""Entity-brain effect handlers (flow's build-split ask, commons c5163/c5169).

Three HOME-ONLY effects wrapping EXISTING facade calls so the entity-life
master flow can animate the sleep window, the personal day-gate, and active
recall as VisualFlow nodes — compose-not-reimplement:

- ``MEMORY_CONSOLIDATE`` -> ``abstractmemory.sleep_pass(...)`` : the six-phase
  night (resolution / maintenance / world models / mining / identity / dream)
  in ONE call. The handler — not the flow — enforces the two constraints that
  keep animation safe: the HOME LEASE (holder="dream", one writer per home)
  and the operator PAUSED kill-switch. Both refuse as honest *results*
  (``{"ran": False, "reason": ...}``), never crashes: the pass is idempotent,
  the next window runs it.
- ``MEMORY_PROBE``       -> ``MemorySystem.probe / probe_expand / familiarity`` :
  the deliberate reach (shelf-race-exempt, reason-mandatory) plus its two
  siblings, one parameterized effect (``op``) per the MEMORY_ADJUST precedent.
- ``LIFE_QUERY``         -> ``alive_drives / cognition_health / entity_card`` :
  the day-gate OFFER/GATE reads and the identity view. Pure reads.

Registration is deliberately NOT part of ``build_memory_seam_effect_handlers``:
these bind only through ``open_home`` (chat.py), so workplaces stay
structurally handler-less — dispatching MEMORY_CONSOLIDATE on a workplace
runtime fails with "no effect handler registered", the same deposit-gate law
that keeps DIARY_* home-only. Naming note: LIFE_QUERY is deliberately not
MEMORY_QUERY — that EffectType already exists as a built-in for the old
workflow memory API; collision avoided at birth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.models import Effect, EffectType, RunState
from ...core.runtime import EffectHandler, EffectOutcome

# The home ladder every read/pass defaults to (the driver's own scopes).
_LADDER = ("self", "diary", "life")

# THE canonical entity-brain effect set (flow c5237: the door's routing was
# hand-counted at 7 while the brain grew to 11 — "import the set from one
# place so the NEXT effect type cannot re-open this gap", the diary_type-
# clamp drift class). Every effect type an entity HOME registers: the seam
# five + the diary pair + the four composition effects. `open_home` is
# pinned to compose EXACTLY this set (test derives from the real handler
# dict, never a hand count); door routing (gateway install_entity_routing)
# imports it instead of enumerating.
ENTITY_HOME_EFFECT_TYPES = frozenset({
    EffectType.MEMORY_RECALL,
    EffectType.MEMORY_ACCESS,
    EffectType.MEMORY_FORM,
    EffectType.MEMORY_ADJUST,
    EffectType.MEMORY_APPRAISE,
    EffectType.DIARY_WRITE,
    EffectType.DIARY_READ,
    EffectType.MEMORY_CONSOLIDATE,
    EffectType.MEMORY_PROBE,
    EffectType.LIFE_QUERY,
    EffectType.MEMORY_TEND,
    # Tool surface (flow c5285 ask 3): grant query + one-batch execution —
    # identity/tool_effects.py, registered via open_home like the rest.
    EffectType.ENTITY_TOOLS_QUERY,
    EffectType.ENTITY_TOOLS_EXECUTE,
})


def _json_safe(value: Any) -> Any:
    """Ledger truth must be JSON-safe (results replay from the recorded
    ledger row). Engine dicts are JSON-clean today; the round-trip guards
    against a future tuple/dataclass leaf without masking the shape."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:  # noqa: BLE001 - a result we cannot serialize is a bug surface
        return {"repr": repr(value)}


def _scope_pairs(payload: Dict[str, Any], entity_id: str) -> List[Tuple[str, str]]:
    """Payload scopes (validated shape) or the home ladder. Bare scope names
    resolve to the home owner — payloads never carry the entity id (the
    deposit-gate scoping rule, same as the seam handlers)."""
    raw = payload.get("scopes")
    if not raw:
        return [(s, entity_id) for s in _LADDER]
    pairs: List[Tuple[str, str]] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                pairs.append((item.strip().lower(), entity_id))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                scope = str(item[0] or "").strip().lower()
                owner = str(item[1] or "").strip() or entity_id
                if scope:
                    pairs.append((scope, owner))
    return pairs or [(s, entity_id) for s in _LADDER]


def build_entity_brain_effect_handlers(
    *,
    memory_system: Any,
    entity_id: str,
    home_dir: Path,
) -> Dict[EffectType, EffectHandler]:
    """Build the three entity-brain handlers over an OPEN home's facade.

    ``memory_system`` is the home's own MemorySystem (writes land in the
    home's memory.sqlite3); ``home_dir`` grounds the lease and the operator
    state reads for MEMORY_CONSOLIDATE's guards.
    """
    home_dir = Path(home_dir)

    def _night_should_continue() -> bool:
        # The flow-driven night ends at the engine's next phase boundary when
        # the operator pulls the brake (STOP/paused), a visitor arrives, or a
        # stamped wake_at deadline passes (sleep-is-bounded law). Deliberately
        # NOT state != "asleep": the animated lifecycle owns phase semantics —
        # requiring the legacy gate write here would break the master flow.
        if (home_dir / "STOP").exists():
            return False
        state = _read_state()
        if str(state.get("state") or "") == "paused":
            return False
        if str(state.get("mode") or "") == "visiting":
            return False
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

    def _read_state() -> Dict[str, Any]:
        try:
            from ...identity.life import read_entity_state

            return read_entity_state(home_dir)
        except Exception:  # noqa: BLE001 - a missing/corrupt state file is not a crash
            return {}

    def _handle_consolidate(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node, run
        payload = dict(effect.payload or {})
        report_only = bool(payload.get("report_only", False))
        include_dream = bool(payload.get("include_dream", True))
        include_identity = bool(payload.get("include_identity", True))
        scopes = _scope_pairs(payload, entity_id)

        try:
            from abstractmemory import sleep_pass
        except ImportError:
            return EffectOutcome.failed(
                "MEMORY_CONSOLIDATE requires abstractmemory.sleep_pass (older engine; "
                "update abstractmemory)"
            )

        # Write passes carry the two host constraints; a report_only pass is a
        # pure structural read — no lease, no paused gate (observability under
        # an operator freeze stays legal, same as every other read).
        lease = None
        if not report_only:
            state = _read_state()
            if str(state.get("state") or "") == "paused":
                return EffectOutcome.completed({
                    "ran": False,
                    "reason": "operator paused the entity - no pass runs under a freeze",
                })
            if (home_dir / "STOP").exists():
                return EffectOutcome.completed({
                    "ran": False,
                    "reason": "STOP file present - no pass runs under the manual brake",
                })
            from ...storage.lease import DirectoryLease, DirectoryLeaseHeld

            try:
                lease = DirectoryLease(home_dir, holder="dream")
                lease.acquire()
            except DirectoryLeaseHeld:
                return EffectOutcome.completed({
                    "ran": False,
                    "reason": "another writer holds the home; the pass is idempotent - "
                              "the next window runs it",
                })

        kwargs: Dict[str, Any] = {
            "scopes": scopes,
            "owner_id": entity_id,
            "report_only": report_only,
            "include_dream": include_dream,
            "include_identity": include_identity,
        }
        for key in ("max_candidates", "scan_limit", "salience_floor"):
            if isinstance(payload.get(key), int):
                kwargs[key] = int(payload[key])
        if not report_only:
            kwargs["should_continue"] = _night_should_continue

        warnings: List[str] = []
        try:
            try:
                engine = sleep_pass(memory_system, **kwargs)
            except TypeError:
                # Version-skew ladder: drop the newest kwargs one step at a
                # time, labeled. report_only is NEVER dropped (adversary C1,
                # 2026-07-25): the lease + paused/STOP gates are conditioned
                # on `not report_only`, so silently dropping it would run an
                # UNGUARDED WRITE PASS that the result still labels a pure
                # read — the read->write inversion. An engine that cannot do
                # a pure-read pass fails HONESTLY here, never degrades a read
                # into a write. Each retry drops from the ORIGINAL kwargs so
                # an engine rejecting only one kwarg keeps the others.
                droppable = [k for k in ("include_identity", "include_dream", "should_continue") if k in kwargs]
                engine = None
                for i in range(len(droppable)):
                    narrowed = dict(kwargs)
                    for k in droppable[: i + 1]:
                        narrowed.pop(k, None)
                    try:
                        engine = sleep_pass(memory_system, **narrowed)
                        warnings.extend(
                            f"#FALLBACK engine sleep_pass has no {k}; narrower call"
                            for k in droppable[: i + 1]
                        )
                        break
                    except TypeError:
                        continue
                if engine is None:
                    return EffectOutcome.failed(
                        "MEMORY_CONSOLIDATE: engine sleep_pass rejected every supported call shape"
                        + (" (report_only pure-read is required but unsupported; upgrade abstractmemory)"
                           if report_only else "")
                    )
        except Exception as e:  # noqa: BLE001
            return EffectOutcome.failed(f"MEMORY_CONSOLIDATE sleep_pass failed: {e}")
        finally:
            if lease is not None:
                try:
                    lease.release()
                except Exception:  # noqa: BLE001 - release failure never masks the pass result
                    pass

        engine = engine if isinstance(engine, dict) else {"engine": engine}
        dream = engine.get("dream") if isinstance(engine.get("dream"), dict) else {}
        maint = engine.get("maintenance") if isinstance(engine.get("maintenance"), dict) else {}
        # The loop-facing translation (the consolidator's contract): read the
        # ENGINE'S REAL KEYS — dream_record_id (consolidation.py:537/765) and
        # maintenance `created` (candidate_miner.py: the created-candidates
        # LIST, created_count beside it). The first fold here read
        # `record_id`/`candidates` — keys that never existed — so EVERY
        # settlement reported a formed dream as "a quiet night" (wave-4
        # adversary E, F1: 10/10 nights false-negative; the 2026-07-09
        # formed/created class again, on both sides of one comment that
        # cited the lesson). Pinned against the real engine shape.
        out = {
            "ran": True,
            "report_only": report_only,
            "formed": bool(dream.get("created")),
            "dream_record_id": dream.get("dream_record_id") or dream.get("record_id"),
            "maintenance_candidates": maint.get("created") if isinstance(maint.get("created"), list) else None,
            "engine": engine,
        }
        if engine.get("cancelled_after"):
            out["cancelled_after"] = engine["cancelled_after"]
        if warnings:
            out["warnings"] = warnings
        return EffectOutcome.completed(_json_safe(out))

    def _handle_probe(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node, run
        payload = dict(effect.payload or {})
        op = str(payload.get("op") or "probe").strip().lower()
        reason = str(payload.get("reason") or "").strip()
        scopes = _scope_pairs(payload, entity_id)

        from .seam_handlers import _import_seam

        Stimulus, _ = _import_seam()

        if op == "probe":
            cue = str(payload.get("cue") or payload.get("cue_text") or "").strip()
            if not cue:
                return EffectOutcome.failed("MEMORY_PROBE op=probe requires payload.cue")
            if not reason:
                return EffectOutcome.failed(
                    "MEMORY_PROBE op=probe requires payload.reason (the deliberate-reach law: "
                    "an audited escalation names why it goes looking)"
                )
            effort = str(payload.get("effort") or "standard").strip().lower()
            try:
                result = memory_system.probe(
                    Stimulus(cue_text=cue), scopes=scopes, reason=reason, effort=effort,
                    journal=bool(payload.get("journal", True)),
                )
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"MEMORY_PROBE probe failed: {e}")
        elif op == "expand":
            record_ids = [str(x).strip() for x in (payload.get("record_ids") or ()) if str(x or "").strip()]
            if not record_ids:
                return EffectOutcome.failed("MEMORY_PROBE op=expand requires payload.record_ids")
            if not reason:
                return EffectOutcome.failed("MEMORY_PROBE op=expand requires payload.reason")
            kwargs: Dict[str, Any] = {"reason": reason}
            for key in ("depth", "max_records", "token_budget"):
                if isinstance(payload.get(key), int):
                    kwargs[key] = int(payload[key])
            if payload.get("scopes"):
                kwargs["scopes"] = scopes
            try:
                result = memory_system.probe_expand(record_ids, **kwargs)
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"MEMORY_PROBE expand failed: {e}")
        elif op == "familiarity":
            cue = str(payload.get("cue") or payload.get("cue_text") or "").strip()
            if not cue:
                return EffectOutcome.failed("MEMORY_PROBE op=familiarity requires payload.cue")
            try:
                result = memory_system.familiarity(
                    Stimulus(cue_text=cue), scopes=scopes,
                    effort=str(payload.get("effort") or "quick").strip().lower(),
                )
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"MEMORY_PROBE familiarity failed: {e}")
        else:
            return EffectOutcome.failed(
                f"MEMORY_PROBE: unknown op {op!r} (use 'probe', 'expand' or 'familiarity')"
            )

        out = result.to_dict() if hasattr(result, "to_dict") else result
        if isinstance(out, dict):
            out.setdefault("op", op)
        else:
            out = {"op": op, "result": out}
        return EffectOutcome.completed(_json_safe(out))

    def _resolve_tend_key(token: str):
        """(resolved_or_None, refusal_reason_or_None) for one record key —
        8-hex #tags resolve against the WHOLE home ladder (the one-spelling
        law: the taught key is the shown key), refuse-on-ambiguity because a
        tend verb is an audited act; ':'-bearing full ids pass through.
        Mirrors the chat driver's resolver exactly (chat.py _resolve_tend_key)
        so the two election surfaces cannot drift on key semantics."""
        from ...identity.memory_reader import memory_tag

        t = str(token or "").lstrip("#").strip()
        if not t or ":" in str(token) or len(t) > 12:
            return token, None
        try:
            from abstractmemory import TripleQuery

            matches: List[str] = []
            seen: set = set()
            for scope in _LADDER:
                for a in memory_system.store.query(
                    TripleQuery(predicate="dcterms:abstract", scope=scope, owner_id=entity_id, limit=0)
                ):
                    subject = str(a.subject or "")
                    if subject and subject not in seen and memory_tag(subject) == t:
                        seen.add(subject)
                        matches.append(subject)
        except Exception as e:  # noqa: BLE001 - resolution failure is a refusal, never a crash
            return None, f"#{t} could not be resolved ({e})"
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"#{t} matches {len(matches)} records - quote the full id"
        return None, f"#{t} matches nothing in the home graph"

    def _handle_tend(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        """MEMORY_TEND: the ONE tend-election route shared with the chat
        driver (flow c5208 ask 5 — dream disposal reached dispose_dream only
        through chat's fence). The payload carries the fence BODY verbatim;
        grammar and verbs stay ENGINE-OWNED (parse_tend_block +
        apply_tend_elections — the consent-vocabulary lesson: no second
        spelling is minted here). Refusals return as DATA in the result,
        shown to the author unedited; a failed election never fails the
        effect."""
        del default_next_node
        payload = dict(effect.payload or {})
        body = str(payload.get("body") or "").strip()
        if not body:
            return EffectOutcome.failed("MEMORY_TEND requires payload.body (the tend fence body verbatim)")
        scope = str(payload.get("scope") or "life").strip().lower()
        # THE CHANNEL, forwarded NEVER defaulted (memory tend.py, entity-seat
        # fable5 P0, 2026-07-25): apply_tend_elections refuses unless the
        # caller states the verified channel — and "an engine privilege
        # check may never be satisfied by its own default", so this handler
        # must NOT mint a privileged constant (that was the exact hole memory
        # closed: a workplace-stamped run tending as the entity's own
        # reflection). The DOOR forwards the verified channel into the
        # payload (like actor/participants); a home-direct in-process caller
        # states it where true by construction. Absent -> None -> memory
        # refuses loudly, never silently self-authorizes.
        channel = str(payload.get("channel") or "").strip() or None

        try:
            from abstractmemory import apply_tend_elections, parse_tend_block
        except ImportError:
            return EffectOutcome.failed(
                "MEMORY_TEND requires abstractmemory tending (parse_tend_block/apply_tend_elections); "
                "update abstractmemory"
            )

        try:
            parsed = parse_tend_block(body)
            elections: List[Any] = []
            for el in parsed.get("elections", []):
                refused_reason = None
                tgt = str(el.get("target") or "")
                if tgt:
                    resolved, refused_reason = _resolve_tend_key(tgt)
                    if refused_reason is None and resolved != tgt:
                        el = dict(el)
                        el["target"] = resolved
                if refused_reason is None and isinstance(el.get("args"), dict) and el["args"]:
                    new_args = dict(el["args"])
                    for key in ("source_id", "target_id"):
                        if key in new_args:
                            resolved, refused_reason = _resolve_tend_key(str(new_args[key]))
                            if refused_reason is not None:
                                break
                            new_args[key] = resolved
                    if refused_reason is None and "evidence_ids" in new_args:
                        ev = []
                        for item in tuple(new_args.get("evidence_ids") or ()):
                            resolved, refused_reason = _resolve_tend_key(str(item))
                            if refused_reason is not None:
                                break
                            ev.append(resolved)
                        if refused_reason is None:
                            new_args["evidence_ids"] = tuple(ev)
                    if refused_reason is None:
                        el = dict(el)
                        el["args"] = new_args
                if refused_reason is not None:
                    parsed.setdefault("refusals", []).append({
                        "line": el.get("line", ""), "reason": refused_reason,
                    })
                    continue
                elections.append(el)
            _tend_kwargs: Dict[str, Any] = dict(
                scope=scope, owner_id=entity_id, actor=entity_id, channel=channel,
            )
            try:
                report = apply_tend_elections(memory_system, elections, **_tend_kwargs)
            except TypeError:
                # Version skew: engine predates the channel kwarg.
                _tend_kwargs.pop("channel", None)
                report = apply_tend_elections(memory_system, elections, **_tend_kwargs)
        except Exception as e:  # noqa: BLE001
            return EffectOutcome.failed(f"MEMORY_TEND failed: {e}")

        out = {
            "applied": report.get("applied", []),
            "refused": list(parsed.get("refusals", [])) + list(report.get("refused", [])),
            "revisit_paths": report.get("revisit_paths", []),
        }
        return EffectOutcome.completed(_json_safe(out))

    def _handle_life_query(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node, run
        payload = dict(effect.payload or {})
        op = str(payload.get("op") or "").strip().lower()
        scopes = _scope_pairs(payload, entity_id)

        if op == "alive_drives":
            try:
                from abstractmemory.alive_drives import alive_drives

                k = int(payload.get("k")) if isinstance(payload.get("k"), int) else 5
                items = alive_drives(memory_system, scopes=scopes, k=k)
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"LIFE_QUERY alive_drives failed: {e}")
            return EffectOutcome.completed(_json_safe({"op": op, "items": items}))
        if op == "cognition_health":
            try:
                from abstractmemory.cognition_health import cognition_health

                health = cognition_health(
                    memory_system.store, memory_system.journal, scopes=scopes
                )
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"LIFE_QUERY cognition_health failed: {e}")
            out = dict(health) if isinstance(health, dict) else {"result": health}
            out["op"] = op
            return EffectOutcome.completed(_json_safe(out))
        if op == "entity_card":
            try:
                kwargs: Dict[str, Any] = {"scope_pairs": scopes, "owner_id": entity_id}
                if isinstance(payload.get("as_of"), int):
                    kwargs["as_of"] = int(payload["as_of"])
                card = memory_system.entity_card(**kwargs)
            except Exception as e:  # noqa: BLE001
                return EffectOutcome.failed(f"LIFE_QUERY entity_card failed: {e}")
            out = dict(card) if isinstance(card, dict) else {"result": card}
            out["op"] = op
            return EffectOutcome.completed(_json_safe(out))
        return EffectOutcome.failed(
            f"LIFE_QUERY: unknown op {op!r} (use 'alive_drives', 'cognition_health' or 'entity_card')"
        )

    return {
        EffectType.MEMORY_CONSOLIDATE: _handle_consolidate,
        EffectType.MEMORY_PROBE: _handle_probe,
        EffectType.LIFE_QUERY: _handle_life_query,
        EffectType.MEMORY_TEND: _handle_tend,
    }
