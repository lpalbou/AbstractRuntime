"""The visit as ONE DURABLE RUN — the frozen seam spec's node cycle, real.

Plan items 7-10 (phase 3, the centerpiece): the entity turn loop mapped
onto runtime effects so a gateway restart RESUMES a visit instead of
killing it. This module builds the WorkflowSpec; `open_entity_runtime`
(item 8) provides the per-home Runtime it runs on; the door (gateway GW-C)
stamps and serves it.

Node cycle (frozen spec §A + the 0014 node-graph draft):

    OPEN -> PARK -> ROUTE -> RECALL -> REASON -> ELECT* -> COMMIT -> FORM
              ^                                                       |
              +----------------------- ANSWER <----------------------+
    ROUTE(close/timeout) -> REFLECT -> APPLY* -> DONE

- OPEN renders the summon prelude ONCE into run vars (`_visit.system_base`
  — the stable head; byte-stable across every resume because it lives in
  the run state, which is exactly the cache-stability property the spec
  names). A REFUSED prelude completes the run with `{"refused": true}` —
  the door translates it to its 4xx; identity is never truncated.
- PARK is WAIT_EVENT `visitor_input` with the D3 idle deadline; resume
  payloads: `{text, speaker}` (turn), `{kind: "close"}` (explicit close),
  `{timed_out: true}` (deadline). `details.kind="visitor_message"` makes
  the wait self-describing to clients.
- The TURN chain is the driver's seam, per-effect: MEMORY_RECALL ->
  LLM_CALL -> DIARY_WRITE per election -> MEMORY_ACCESS (same-trace) ->
  MEMORY_FORM (episode + lossless verbatim) -> ANSWER_USER -> park again.
- REFLECT is the look-back: one LLM_CALL over the session sheet, then the
  staged APPLY loop (summary -> interests -> diary -> feelings) using the
  same reflection machinery the in-process driver uses.

v0 SCOPE, stated honestly: the REASON node is ONE LLM_CALL — the
abstractagent ReAct adapter replaces exactly that node in the full
integration (the boundary the agent seat asked about); tier-1 tool rounds
and the honesty guards (marker imitation / speak-now / liveness) ride that
replacement. Elections, recall, commit, form, reflection, and durability
are REAL here — this is the A/B fixture's durable side for criteria 1
(memory-plane equivalence), 5 (restart-mid-visit), and 6 (diary privacy).

Replay safety: every effect is idempotency-keyed by (run, node, payload);
memory/diary writes dedup on turn-derived ids; vars folds are guarded by
turn ids so a crash-replayed node never doubles history.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Effect, EffectType, RunState, StepPlan
from ..core.spec import WorkflowSpec
from .chat import (
    CONTRACT_PARAGRAPH,
    SUMMON_POSTURE_SELF_FRACTION,
    VISIT_OWN_TIME_PARAGRAPH,
    ChatHome,
    _memories_block,
    _serialize_under_budget,
    clean_model_reply,
    parse_diary_blocks,
)
from .digest import mechanical_digest_v2
from .prelude import render_summon_prelude
from .reflection import (
    FeelingElection,
    build_reflection_prompt,
    parse_feel_blocks,
    parse_interest_blocks,
    resolve_feeling_targets,
)

VISIT_WORKFLOW_ID = "entity-visit@1"
VISITOR_WAIT_KEY = "visitor_input"
DEFAULT_IDLE_SECONDS = 30 * 60  # a visit left silent this long closes with reflection
DEFAULT_HISTORY_TURNS = 10


def _ns(run: RunState, key: str) -> Dict[str, Any]:
    ns = run.vars.get(key)
    if not isinstance(ns, dict):
        ns = {}
        run.vars[key] = ns
    return ns


def _deadline_iso(idle_seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=float(idle_seconds))).isoformat()


def build_visit_workflow(
    home: ChatHome,
    *,
    participants: Optional[List[str]] = None,
    budget_profile: Optional[Dict[str, Any]] = None,
    prelude_budget: int = 1600,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    history_turns: int = DEFAULT_HISTORY_TURNS,
    model_info: Optional[Dict[str, str]] = None,
) -> WorkflowSpec:
    """Build the visit workflow over one OPEN home.

    `participants`/`budget_profile` are the STAMP-TIME facts (the door
    writes them at run creation; the home-direct caller passes them here):
    participants are door-verified — the entity itself is appended, door
    parity with the driver. The budget profile defaults to the summon
    posture over the entity floor when not supplied."""

    stamp_participants = [p for p in (participants or []) if str(p).strip()] or ["person:operator"]
    if home.entity_id not in stamp_participants:
        stamp_participants.append(home.entity_id)

    if budget_profile is None:
        from abstractmemory import ENTITY_CONTEXT_FLOOR, entity_recall_budget

        budget = entity_recall_budget(int(ENTITY_CONTEXT_FLOOR))
        budget_profile = (
            dataclasses.asdict(budget) if dataclasses.is_dataclass(budget) else dict(budget)
        )
        budget_profile["self_fraction"] = SUMMON_POSTURE_SELF_FRACTION

    ladder = [["self", home.entity_id], ["diary", home.entity_id], ["life", home.entity_id]]
    model_info_clean = {k: str(v) for k, v in (model_info or {}).items() if v}

    # ------------------------------------------------------------- nodes

    def open_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        if not visit.get("system_base"):
            prelude = render_summon_prelude(
                home.ms, home.diary,
                entity_id=home.entity_id, budget=prelude_budget, spark=home.spark,
            )
            if prelude.get("refused"):
                # Identity is never truncated: the visit ends before it
                # begins, with the reasons on the durable record.
                return StepPlan(
                    node_id="OPEN",
                    complete_output={
                        "refused": True,
                        "reasons": list(prelude.get("warnings", [])),
                    },
                )
            visit["system_base"] = (
                prelude["text"] + "\n\n" + CONTRACT_PARAGRAPH + "\n\n" + VISIT_OWN_TIME_PARAGRAPH
            )
            visit["prelude_warnings"] = list(prelude.get("warnings", []))
            visit.setdefault("participants", list(stamp_participants))
            visit.setdefault("budget", dict(budget_profile))
            visit.setdefault("history", [])
            visit.setdefault("sheet", [])
            visit.setdefault("turn_n", 0)
            visit.setdefault("model_info", dict(model_info_clean))
        return StepPlan(node_id="OPEN", next_node="PARK")

    def park_node(run: RunState, ctx: Any) -> StepPlan:
        return StepPlan(
            node_id="PARK",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={
                    "wait_key": VISITOR_WAIT_KEY,
                    "until": _deadline_iso(idle_seconds),
                    "details": {"kind": "visitor_message"},
                    "resume_to_node": "ROUTE",
                },
                result_key="_turn.resume",
            ),
            next_node="ROUTE",
        )

    def route_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        resume = turn.get("resume") or {}
        if resume.get("timed_out"):
            visit["close_reason"] = "idle_timeout"
            return StepPlan(node_id="ROUTE", next_node="REFLECT")
        if str(resume.get("kind") or "") == "close":
            visit["close_reason"] = "closed"
            return StepPlan(node_id="ROUTE", next_node="REFLECT")
        text = str(resume.get("text") or "").strip()
        if not text:
            # An empty message is not a turn; park again (honest no-op).
            return StepPlan(node_id="ROUTE", next_node="PARK")
        turn_n = int(visit.get("turn_n") or 0) + 1
        visit["turn_n"] = turn_n
        turn.clear()
        turn["text"] = text
        turn["speaker"] = str(resume.get("speaker") or "") or (visit["participants"][0])
        turn["turn_id"] = f"t-{turn_n:04d}"
        return StepPlan(node_id="ROUTE", next_node="RECALL")

    def recall_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        return StepPlan(
            node_id="RECALL",
            effect=Effect(
                type=EffectType.MEMORY_RECALL,
                payload={
                    "cue_text": turn["text"],
                    "scopes": ladder,
                    "view": "working_set",
                    "turn_id": turn["turn_id"],
                    "participants": list(visit["participants"]),
                    "budget": dict(visit["budget"]),
                },
                result_key="_turn.recall",
            ),
            next_node="REASON",
        )

    def reason_node(run: RunState, ctx: Any) -> StepPlan:
        """v0 reason: ONE LLM_CALL. The ReAct adapter replaces THIS node
        (and only this node) in the full integration — the seam boundary."""
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        recall = turn.get("recall") or {}
        rendered = _serialize_under_budget(
            list(recall.get("handles") or []), int(visit["budget"].get("token_budget") or 0)
        )
        displayed = [h for h in rendered if h.get("admission") != "self"]
        turn["displayed"] = displayed
        block = _memories_block(displayed, recall.get("as_of_seq"))
        others = [p for p in visit["participants"] if p != home.entity_id]
        presence = f"(present with you: {', '.join(others)})" if others else ""
        system_prompt = visit["system_base"] + ("\n\n" + presence if presence else "") + (
            "\n\n" + block if block else ""
        )
        messages = list(visit.get("history") or []) + [{"role": "user", "content": turn["text"]}]
        # turn_id + word-free anchors ride the payload: the act-only wrapper
        # captures diary elections at the RESULT boundary and writes the book
        # through DIARY_WRITE before anything persists (G1 write direction —
        # the A/B privacy grep found the raw reply resting in the run store
        # when elections were parsed a node later).
        anchor_graph_ids = [
            str((h.get("provenance") or {}).get("record_id") or "")
            for h in displayed
            if (h.get("provenance") or {}).get("record_id")
        ]
        return StepPlan(
            node_id="REASON",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "turn_id": turn["turn_id"],
                    "anchor_record_ids": [h.get("record_id") for h in displayed],
                    "anchor_graph_ids": anchor_graph_ids,
                },
                result_key="_turn.llm",
            ),
            next_node="ELECT",
        )

    def elect_node(run: RunState, ctx: Any) -> StepPlan:
        """Fold the wrapper-captured elections (pure node): the book was
        already written at the result boundary; only word-free metadata and
        the MARKED reply arrive here."""
        turn = _ns(run, "_turn")
        llm = turn.get("llm") or {}
        marked = clean_model_reply(str(llm.get("content") or ""))
        notices = list(llm.get("act_only_warnings") or [])
        if not marked:
            marked = "…"  # an empty reply still closes the turn honestly
            notices.append("#FALLBACK the model returned no words this turn")
        turn["marked_reply"] = marked
        turn["diary_meta"] = list(llm.get("diary_entries") or [])
        turn["notices"] = notices
        return StepPlan(node_id="ELECT", next_node="COMMIT")

    def commit_node(run: RunState, ctx: Any) -> StepPlan:
        turn = _ns(run, "_turn")
        displayed = list(turn.get("displayed") or [])
        recall = turn.get("recall") or {}
        if not displayed or not recall.get("trace_id"):
            return StepPlan(node_id="COMMIT", next_node="FORM")
        return StepPlan(
            node_id="COMMIT",
            effect=Effect(
                type=EffectType.MEMORY_ACCESS,
                payload={
                    # Same-trace contract (frozen spec): THIS turn's recall.
                    "trace_id": recall["trace_id"],
                    "used_record_ids": [h.get("record_id") for h in displayed],
                    "prompt_token_estimate": sum(
                        int(h.get("token_estimate") or 0) for h in displayed
                    ),
                },
                result_key="_turn.committed",
            ),
            next_node="FORM",
        )

    def form_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        speaker = str(turn.get("speaker") or visit["participants"][0])
        title, digest, keywords = mechanical_digest_v2(
            turn["text"], turn["marked_reply"], home.name, speaker=speaker
        )
        turn["digest"] = digest
        verbatim = f"{speaker}:\n{turn['text']}\n\n{home.name}:\n{turn['marked_reply']}"
        attributes: Dict[str, Any] = {
            "participants": list(visit["participants"]),
            "digest_method": "mechanical-v2",
        }
        if visit.get("model_info"):
            attributes["mind_substrate"] = dict(visit["model_info"])
        edges: List[List[str]] = []
        if visit.get("last_episode_id"):
            edges.append(["continues", str(visit["last_episode_id"])])
        # reflected_in per non-private projection (private projections carry
        # no edges — the containment rule); metadata is word-free.
        for meta in list(turn.get("diary_meta") or []):
            projected = (meta or {}).get("projected_record_id")
            if projected and meta.get("visibility") != "private":
                edges.append(["reflected_in", str(projected)])
        return StepPlan(
            node_id="FORM",
            effect=Effect(
                type=EffectType.MEMORY_FORM,
                payload={
                    "records": [{
                        "kind": "episode",
                        "title": title,
                        "digest": digest,
                        "keywords": keywords,
                        "verbatim": verbatim,
                        "edges": edges,
                        "attributes": attributes,
                        "provenance": {"source": "entity-visit-run-v0"},
                    }],
                    "scope": "life",
                    "owner_id": home.entity_id,
                    "turn_id": turn["turn_id"],
                },
                result_key="_turn.formed",
            ),
            next_node="ANSWER",
        )

    def answer_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        # Fold the turn into the visit ONCE (replay guard: a crash between
        # ANSWER and the next park re-runs this node; the fold must not
        # double history).
        if visit.get("last_folded_turn") != turn["turn_id"]:
            history = list(visit.get("history") or [])
            history.append({"role": "user", "content": turn["text"]})
            history.append({"role": "assistant", "content": turn["marked_reply"]})
            visit["history"] = history[-2 * int(history_turns):]
            sheet = list(visit.get("sheet") or [])
            formed_ids = list((turn.get("formed") or {}).get("record_ids") or [])
            for rid in formed_ids:
                sheet.append([str(rid), str(turn.get("digest") or "")[:160]])
                visit["last_episode_id"] = str(rid)
            for meta in list(turn.get("diary_meta") or []):
                projected = (meta or {}).get("projected_record_id")
                if projected:
                    gist = str(meta.get("gist") or "a private entry")
                    sheet.append([str(projected), f"you kept a diary entry ({meta.get('kind')}): {gist}"])
            visit["sheet"] = sheet
            visit["last_folded_turn"] = turn["turn_id"]
        return StepPlan(
            node_id="ANSWER",
            effect=Effect(
                type=EffectType.ANSWER_USER,
                payload={"message": turn["marked_reply"], "turn_id": turn["turn_id"]},
                result_key="_turn.answered",
            ),
            next_node="PARK",
        )

    def reflect_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        sheet = list(visit.get("sheet") or [])
        if not sheet:
            return StepPlan(node_id="REFLECT", next_node="DONE")
        sheet_lines = [f"{i}. {desc}" for i, (_rid, desc) in enumerate(sheet, start=1)]
        prompt = build_reflection_prompt(sheet_lines)
        session_graph_ids = [rid for rid, _ in sheet if rid][-4:]
        return StepPlan(
            node_id="REFLECT",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "messages": list(visit.get("history") or []) + [{"role": "user", "content": prompt}],
                    "system_prompt": visit["system_base"],
                    # The look-back's diary elections are captured at the
                    # result boundary too (same wrapper, same G1 rule).
                    "turn_id": "t-reflect",
                    "anchor_record_ids": session_graph_ids,
                    "anchor_graph_ids": session_graph_ids,
                },
                result_key="_reflect.llm",
            ),
            next_node="APPLY",
        )

    def apply_node(run: RunState, ctx: Any) -> StepPlan:
        """Staged look-back application: summary -> interests -> diary ->
        feelings (summary first — target=session needs its record id)."""
        visit = _ns(run, "_visit")
        refl = _ns(run, "_reflect")
        if "marked_reply" not in refl:
            raw = clean_model_reply(str((refl.get("llm") or {}).get("content") or ""))
            marked, feelings, notices = parse_feel_blocks(raw)
            marked, interests, i_notes = parse_interest_blocks(marked)
            marked, diary_elections, d_notes = parse_diary_blocks(marked)
            refl["marked_reply"] = marked
            refl["feelings"] = [dataclasses.asdict(f) for f in feelings]
            refl["interests"] = list(interests)
            refl["diary"] = [dataclasses.asdict(e) for e in diary_elections]
            refl["notices"] = list(notices) + list(i_notes) + list(d_notes)
            refl["stage"] = "summary"
            refl["i"] = 0

        sheet = list(visit.get("sheet") or [])
        stage = str(refl.get("stage") or "summary")

        if stage == "summary":
            refl["stage"] = "interest"
            refl["i"] = 0
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "summary",
                            "title": f"session reflection: {run.session_id}",
                            "digest": " ".join(str(refl["marked_reply"]).split())[:280],
                            "keywords": [],
                            "verbatim": str(refl["marked_reply"]),
                            "edges": [["summarizes", rid] for rid, _ in sheet if rid],
                            "attributes": {
                                "participants": list(visit["participants"]),
                                "session_id": str(run.session_id or ""),
                            },
                            "provenance": {"source": "entity-visit-run-reflection-v0"},
                        }],
                        "scope": "life",
                        "owner_id": home.entity_id,
                        "turn_id": "t-reflect",
                    },
                    result_key="_reflect.summary_out",
                ),
                next_node="APPLY",
            )

        if stage == "interest":
            interests = list(refl.get("interests") or [])
            i = int(refl.get("i") or 0)
            if i >= len(interests):
                refl["stage"] = "diary"
                refl["i"] = 0
                return StepPlan(node_id="APPLY", next_node="APPLY")
            refl["i"] = i + 1
            session_record_id = next(
                iter((refl.get("summary_out") or {}).get("record_ids") or []), None
            )
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "interest",
                            "title": "interest: " + " ".join(str(interests[i]).split()[:8]),
                            "digest": str(interests[i]),
                            "keywords": [],
                            "edges": (
                                [["from_session", str(session_record_id)]] if session_record_id else []
                            ),
                            "attributes": {"session_id": str(run.session_id or "")},
                            "provenance": {
                                "source": "entity-visit-run-reflection-v0",
                                "actor": "entity-reflection",
                            },
                        }],
                        "scope": "self",
                        "owner_id": home.entity_id,
                        "turn_id": f"t-reflect-interest-{i}",
                    },
                    result_key="_reflect.interest_out",
                ),
                next_node="APPLY",
            )

        if stage == "diary":
            entries = list(refl.get("diary") or [])
            i = int(refl.get("i") or 0)
            if i >= len(entries):
                refl["stage"] = "feel"
                refl["i"] = 0
                return StepPlan(node_id="APPLY", next_node="APPLY")
            refl["i"] = i + 1
            e = entries[i]
            session_graph_ids = [rid for rid, _ in sheet if rid][-4:]
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.DIARY_WRITE,
                    payload={
                        "text": e.get("text"),
                        "gist": e.get("gist"),
                        "kind": e.get("kind"),
                        "visibility": e.get("visibility"),
                        "resolves": e.get("resolves"),
                        "turn_id": f"t-reflect-diary-{i}",
                        "anchor_record_ids": session_graph_ids,
                        "anchor_graph_ids": session_graph_ids,
                    },
                    result_key="_reflect.diary_out",
                ),
                next_node="APPLY",
            )

        # stage == "feel"
        if "resolved_feelings" not in refl:
            feelings = [FeelingElection(**f) for f in (refl.get("feelings") or [])]
            session_record_id = next(
                iter((refl.get("summary_out") or {}).get("record_ids") or []), None
            )
            resolved, notes = resolve_feeling_targets(
                feelings,
                sheet_record_ids=[rid for rid, _ in sheet],
                session_record_id=str(session_record_id) if session_record_id else None,
                self_id=home.entity_id,
            )
            refl["resolved_feelings"] = [
                {**dataclasses.asdict(f), "record_id": rid} for f, rid in resolved
            ]
            refl["notices"] = list(refl.get("notices") or []) + list(notes)
            refl["i"] = 0
        resolved = list(refl.get("resolved_feelings") or [])
        i = int(refl.get("i") or 0)
        if i >= len(resolved):
            return StepPlan(node_id="APPLY", next_node="DONE")
        refl["i"] = i + 1
        f = resolved[i]
        target = str(f["record_id"])
        return StepPlan(
            node_id="APPLY",
            effect=Effect(
                type=EffectType.MEMORY_APPRAISE,
                payload={
                    "op": "appraise",
                    "target_id": target,
                    "sign": f.get("sign"),
                    "magnitude": f.get("magnitude"),
                    "reason": f.get("reason"),
                    "scar": bool(f.get("scar")),
                    "bond": bool(f.get("bond")),
                    "turn_id": f"t-reflect-feel-{i}",
                    "scope": "life" if target.startswith("ex:") else "self",
                    "owner_id": home.entity_id,
                    "actor": "entity-reflection",
                },
                result_key="_reflect.feel_out",
            ),
            next_node="APPLY",
        )

    def done_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        refl = _ns(run, "_reflect")
        return StepPlan(
            node_id="DONE",
            complete_output={
                "ok": True,
                "turns": int(visit.get("turn_n") or 0),
                "close_reason": str(visit.get("close_reason") or "closed"),
                "reflection_notices": list(refl.get("notices") or []),
            },
        )

    return WorkflowSpec(
        workflow_id=VISIT_WORKFLOW_ID,
        entry_node="OPEN",
        nodes={
            "OPEN": open_node,
            "PARK": park_node,
            "ROUTE": route_node,
            "RECALL": recall_node,
            "REASON": reason_node,
            "ELECT": elect_node,
            "COMMIT": commit_node,
            "FORM": form_node,
            "ANSWER": answer_node,
            "REFLECT": reflect_node,
            "APPLY": apply_node,
            "DONE": done_node,
        },
    )
