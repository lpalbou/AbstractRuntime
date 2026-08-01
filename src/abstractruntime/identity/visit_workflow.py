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
    SUMMON_POSTURE_SELF_FRACTION,
    ChatHome,
    _memories_block,
    _serialize_under_budget,
    clean_model_reply,
    compose_system_base,
    floored_reflection_digest,
    parse_diary_blocks,
)
from .digest import mechanical_digest_v2
from .prelude import render_summon_prelude
from .prompt_overlay import overlay_note, read_prompt_overlay
from .reflection import (
    FeelingElection,
    build_reflection_prompt,
    parse_feel_blocks,
    parse_interest_blocks,
    parse_lesson_blocks,
    parse_realize_blocks,
    parse_topic_blocks,
    resolve_feeling_targets,
)

VISIT_WORKFLOW_ID = "entity-visit@1"
VISITOR_WAIT_KEY = "visitor_input"
DEFAULT_IDLE_SECONDS = 30 * 60  # a visit left silent this long closes with reflection
DEFAULT_HISTORY_TURNS = 10
HARVEST_NODE = "HARVEST"  # the react middle's exit contract (final_next_node)

# HISTORY REPLAY DISCIPLINE (operator incident 2026-08-01, this very lane:
# entity ephemeral ran read_file on a 5MB attached screenshot; the tool
# result entered the durable react transcript as a 494,932-char role="tool"
# message and rode EVERY subsequent LLM call — final request 48 messages /
# 722,453 chars, refused upstream over the model's context window; the
# session was permanently wedged because context.messages is durable and
# was replayed whole, unbounded). These caps bound what a single HISTORY
# message may contribute when RE-SENT: the react adapter's payload hook
# (abstractagent adapters/react_runtime._sanitize_llm_messages) reads them
# from `_limits` and elides overages to a labeled stub — durable history is
# never mutated (ADR-0026 marked, payload boundary only), which is exactly
# why an already-poisoned STORED session recovers on its next packing pass
# with no manual surgery.
#
# Derivation, from the seam arithmetic (abstractmemory/seam.py; chars<->
# tokens at the repo's own 4-chars/token heuristic, memory/token_budget.py).
# SIZED AGAINST THE 40k-ERA RECOMMENDATION and deliberately KEPT at those
# values when the same day's re-ruling moved the target to 50k
# (ENTITY_CONTEXT_RECOMMENDED = 50_000 ~= 200_000 chars; operator: "it is
# acceptable to go to 200k context, but ideally, let's have a (soft)
# recommended target of 50k tokens") — these are poison guards from the
# ephemeral incident, not attention sizing; loosening them was no part of
# the re-ruling, and both still satisfy their governing bounds at 50k:
# - TOOL RESULTS (the aimed-at class — the poison was a tool message):
#   32_000 chars = 8k tokens (20% of the 40k target it was derived
#   against; 16% at 50k). Chosen as the smallest round bound that still
#   admits every honestly-capped walled tool result WHOLE with framing to
#   spare (the largest are execute_command output and a read_file text
#   slice, both 24_000 chars — identity/tools.py), so the clamp never
#   touches honest work; it exists for the monster class (pre-fix
#   poisoned transcripts, defective tools). That admit-honest-work-whole
#   basis is window-independent — the cap stands.
# - ENTITY PROSE / VISITOR WORDS are never sliced except at the EXTREME
#   bound, by the seam's own starvation arithmetic (RecallBudget validates
#   token_fraction <= 0.5: "a recall payload beyond half the context
#   starves generation — an ARITHMETIC bound, not a fear one"): 80_000
#   chars was half of the 40k-era 160_000-char working context, and at
#   the 50k target sits BELOW the 100_000-char half — still on the safe
#   side of the starvation bound. The largest honest message in the
#   poisoned session was 28,925 chars (a MEMORIES-decorated visitor
#   turn) — 2.8x headroom; a message past half the recommended context
#   is starvation by one voice, whoever speaks it.
VISIT_HISTORY_TOOL_RESULT_CAP_CHARS = 32_000
VISIT_HISTORY_MESSAGE_CAP_CHARS = 80_000


@dataclasses.dataclass
class ReactMiddle:
    """The adapter's reason/act cycle, passed IN as data (merge ownership
    ruling, a2a 0014: ONE owner for the visit graph — this package — with
    ZERO import of the adapter package; abstractagent depends on
    abstractruntime, so `build_visit_workflow(react_logic=...)` importing
    agent's API would be a dependency cycle. The CALLER — the door, the
    harness — builds the middle from agent's public API and hands it over).

    Contract (agent's proven merge, tests/test_react_visit_merge.py):
    - `nodes`: the adapter workflow's node map (lowercase ids — must not
      collide with this graph's UPPERCASE ids; refused loudly).
    - `entry`: where BRIDGE enters the cycle (adapter's `reason`).
    - The middle MUST exit to `HARVEST_NODE` (the caller builds it with
      `create_react_workflow(final_next_node="HARVEST")`).
    - `reset_turn`: the adapter's per-turn state reset, called by BRIDGE.
    - The cycle reads `_runtime.system_prompt` (the byte-stable head),
      `_runtime.turn_id` + `_runtime.llm_payload_extras` (word-free
      anchors — the act-only wrapper's capture keys), the durable
      transcript at `context.messages`, and `_limits.max_iterations`; it
      leaves `_temp.final_answer` + `_temp.turn_captures`
      ({diary_entries, act_only_warnings} accumulated across iterations)
      for HARVEST to fold.
    """

    nodes: Dict[str, Any]
    entry: str = "reason"
    reset_turn: Optional[Any] = None
    # Reason-cycle bound per turn. 20 matches the ruled 20-tool-call turn
    # budget (maintainer 2026-07-11: caps bound RUN-AWAY, not ambition) —
    # the original 6 was the tightest agency bound in the stack (agent's
    # caps audit): six cycles cannot spend a research turn's budget.
    # BRIDGE seeds `_limits.max_iterations` with setdefault, so a door- or
    # operator-supplied value always wins over this default.
    max_iterations: int = 20


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
    visit_id: Optional[str] = None,
    react_middle: Optional[ReactMiddle] = None,
) -> WorkflowSpec:
    """Build the visit workflow over one OPEN home.

    `react_middle` (merge ownership, a2a 0014): when provided, the
    adapter's multi-iteration reason/act cycle replaces the v0 single-call
    REASON node — BRIDGE wires the entity dress into the cycle's inputs and
    HARVEST folds its outcome back into `_turn.llm`; every node downstream
    (ELECT → COMMIT → FORM → ANSWER → REFLECT) runs unchanged. None = the
    v0 path, byte-identical to before this parameter existed.

    `participants`/`budget_profile`/`visit_id` are the STAMP-TIME facts
    (the door writes them at run creation; the home-direct caller passes
    them here): participants are door-verified — the entity itself is
    appended, door parity with the driver. The budget profile defaults to
    the summon posture over the entity floor when not supplied. `visit_id`
    is the door-stamped correlation key for two-sided visits (item 14):
    when present it is stamped as `attributes.visit_id` on every episode
    and the reflection summary — both legs of a cross-runtime visit carry
    the SAME string, correlating as DATA, never as shared rows."""

    stamp_participants = [p for p in (participants or []) if str(p).strip()] or ["person:operator"]
    if home.entity_id not in stamp_participants:
        stamp_participants.append(home.entity_id)

    if budget_profile is None:
        from abstractmemory import ENTITY_CONTEXT_FLOOR, entity_recall_budget

        # Home-direct fallback sized at the RECOMMENDATION itself
        # (ENTITY_CONTEXT_FLOOR aliases ENTITY_CONTEXT_RECOMMENDED — 50k
        # since the 2026-08-01 re-ruling), which keeps it consistent with
        # the door's derivation by construction: the gateway now budgets
        # from min(window, recommendation), and min(x, rec) == rec for
        # every window at/above the target — the recommendation sizes
        # ATTENTION; the window sizes growth. One derivation, two doors,
        # same number.
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
            # Operator prompt overlay: behavioral layers only (identity and
            # tools text stay machine-owned) — same file, same snapshot
            # semantics as the in-process driver (<home>/system_prompt.yaml).
            # ONE composition authority (compose_system_base); the durable
            # arm passes no tools because its tool grant threads through
            # the react middle, not the head text (Phase 0 lane).
            overlay = read_prompt_overlay(home.home_dir)
            from .chat import read_capability_map

            visit["system_base"] = compose_system_base(
                prelude["text"], phase="visit", overlay=overlay,
                capability_map=read_capability_map(home.home_dir),
            )
            visit["prelude_warnings"] = list(prelude.get("warnings", []))
            note = overlay_note(overlay)
            if note:
                visit["prelude_warnings"].append(note)
            visit.setdefault("participants", list(stamp_participants))
            visit.setdefault("budget", dict(budget_profile))
            visit.setdefault("history", [])
            visit.setdefault("sheet", [])
            visit.setdefault("turn_n", 0)
            visit.setdefault("model_info", dict(model_info_clean))
            # Door config wins (gateway shape 1, 0014/083823Z): the door
            # seeds `_visit.idle_seconds` at run creation; the build kwarg
            # is the home-direct default. setdefault = seeded value stands.
            visit.setdefault("idle_seconds", float(idle_seconds))
        return StepPlan(node_id="OPEN", next_node="PARK")

    def park_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        return StepPlan(
            node_id="PARK",
            effect=Effect(
                type=EffectType.WAIT_EVENT,
                payload={
                    "wait_key": VISITOR_WAIT_KEY,
                    "until": _deadline_iso(float(visit.get("idle_seconds") or idle_seconds)),
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
            # Door-authored close payload (gateway shape, 0014/083823Z,
            # superseded by 0014/094354Z: THREE closed_by kinds — operator
            # and sleep close gracefully; PAUSE is a hard freeze that must
            # not run the reflection LLM call; the look-back debt is the
            # door's pending-look-back pattern at the next open).
            visit["close_reason"] = "closed"
            if resume.get("closed_by"):
                visit["closed_by"] = str(resume.get("closed_by"))
            if resume.get("reason"):
                visit["close_note"] = str(resume.get("reason"))
            if resume.get("skip_reflection"):
                visit["skip_reflection"] = True
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
            next_node="RENDER",
        )

    def render_node(run: RunState, ctx: Any) -> StepPlan:
        """HEAD DISCIPLINE (agent's ruling ask, adopted — frozen spec §4:
        the head is byte-stable for the WHOLE visit; volatile
        MEMORIES/presence ride the MESSAGE LANE). This pure node renders
        the turn's user message: presence + MEMORIES + the visitor's words.
        The decoration is PER-TURN PROMPT CURRENCY: the fold (ANSWER)
        stores the RAW visitor text in history, so stale MEMORIES blocks
        never accumulate in the transcript presenting themselves as
        current — memories are re-recalled fresh every turn."""
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
        decoration = "\n\n".join(s for s in (presence, block) if s)
        # The decorated message is APPEND-ONCE: it enters the transcript at
        # the fold and STAYS (all-but-last message byte-identical across
        # turns = the cross-turn cache property; each block is dated +
        # as_of-labeled, so an old block reads as the honest record of what
        # that moment reminded him of, never as current recall). The RECORD
        # keeps the human's raw words: formation verbatim uses turn.text.
        turn["rendered_user"] = (
            (decoration + "\n\n" if decoration else "") + turn["text"]
        )
        return StepPlan(node_id="RENDER", next_node="REASON")

    def bridge_node(run: RunState, ctx: Any) -> StepPlan:
        """RENDER -> the adapter cycle (replaces v0 REASON when a
        react_middle is supplied). Body lifted verbatim from agent's proven
        merge (abstractagent tests/test_react_visit_merge.py) — zero
        adapter internals, only the documented vars contract."""
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        runtime_ns = run.vars.setdefault("_runtime", {})
        limits = run.vars.setdefault("_limits", {})
        # Entity dress: prelude head, turn identity, word-free anchors.
        runtime_ns["system_prompt"] = str(visit.get("system_base") or "")
        runtime_ns["turn_id"] = str(turn.get("turn_id") or "")
        # r-rt-1 (Ephemeral incident, laurent c2447 / agent F1): loop tails
        # are TASK-agent chrome — "[loop] iteration N of 20" merged into the
        # visitor's message read to the entity as "something automated is
        # running". BRIDGE is the one place that knows this cycle is an
        # entity visit; the adapters' tail block gates on this flag (agent's
        # knob — harmless until it ships, honored the moment it does).
        runtime_ns["suppress_loop_tail"] = True
        displayed = list(turn.get("displayed") or [])
        runtime_ns["llm_payload_extras"] = {
            "anchor_record_ids": [h.get("record_id") for h in displayed],
            "anchor_graph_ids": [
                str((h.get("provenance") or {}).get("record_id") or "")
                for h in displayed
                if (h.get("provenance") or {}).get("record_id")
            ],
        }
        limits.setdefault("max_iterations", int(react_middle.max_iterations))
        # r-rt-4 (ephemeral 2026-08-01): the entity lane DECLARES its replay
        # caps here — BRIDGE is the one place that knows this cycle is an
        # entity visit (the suppress_loop_tail precedent, c2447/c2453). The
        # adapter's payload hook honors the same `_limits` knobs CodeAct
        # documents; seeding is soft (a door/operator-configured int wins,
        # including an explicit <= 0 "unbounded by choice" — the adapter's
        # shared monster guard still floors every lane at 200k).
        for _key, _cap in (
            ("max_tool_message_chars", VISIT_HISTORY_TOOL_RESULT_CAP_CHARS),
            ("max_message_chars", VISIT_HISTORY_MESSAGE_CAP_CHARS),
        ):
            _existing = limits.get(_key)
            if not isinstance(_existing, int) or isinstance(_existing, bool):
                limits[_key] = int(_cap)
        # Fresh per-turn adapter state; append the decorated turn message to
        # the durable transcript (the ONE source of truth under the merge).
        if callable(react_middle.reset_turn):
            react_middle.reset_turn(run.vars)
        context = run.vars.setdefault("context", {})
        msgs = context.setdefault("messages", [])
        msgs.append({"role": "user", "content": str(turn.get("rendered_user") or turn.get("text") or "")})
        return StepPlan(node_id="REASON", next_node=str(react_middle.entry))

    def harvest_node(run: RunState, ctx: Any) -> StepPlan:
        """The adapter cycle's exit -> ELECT: fold the turn's outcome into
        `_turn.llm` so every downstream node runs byte-unchanged."""
        temp = run.vars.get("_temp") or {}
        turn = _ns(run, "_turn")
        captures = temp.get("turn_captures") or {}
        turn["llm"] = {
            "content": str(temp.get("final_answer") or ""),
            "diary_entries": list(captures.get("diary_entries") or []),
            "act_only_warnings": list(captures.get("act_only_warnings") or []),
        }
        # tools_ran parity (hooks plan H7a): DRIVER-AUTHORED tool truth, folded
        # from the middle's captures when the adapter reports it (never parsed
        # from reply prose — the marker-imitation lesson). Absent = honestly
        # empty; the ledger's TOOL_CALLS records remain the deep audit trail.
        turn["tools_ran"] = [str(t) for t in (captures.get("tools_ran") or []) if str(t or "").strip()]
        # W5: intermediate phases + at-rest results, when the middle reports
        # them (adapter-owned capture; absent = honestly empty).
        turn["lookup_phases"] = [str(x) for x in (captures.get("lookup_phases") or []) if str(x or "").strip()]
        turn["tool_results_at_rest"] = [str(x) for x in (captures.get("tool_results_at_rest") or []) if str(x or "").strip()]
        return StepPlan(node_id=HARVEST_NODE, next_node="ELECT")

    def reason_node(run: RunState, ctx: Any) -> StepPlan:
        """v0 reason: ONE LLM_CALL with the BYTE-STABLE head (system_base
        only — never mutated after OPEN). The ReAct adapter replaces THIS
        node (and only this node) in the full integration."""
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        displayed = list(turn.get("displayed") or [])
        messages = list(visit.get("history") or []) + [
            {"role": "user", "content": turn["rendered_user"]}
        ]
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
                    "system_prompt": visit["system_base"],
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
        the MARKED reply arrive here.

        Mid-turn identity elections (Veya deep check, 2026-07-27): the
        contract teaches feel/interest/lesson/realize in every lane, but
        this lane parsed them only in the CLOSE reflection — a fence
        written during a turn formed nothing, silently, and its raw text
        reached the visitor. Now every fence kind is parsed here, staged on
        the visit (run vars are durable — staging survives crashes on this
        lane), and the existing close APPLY stages form them. Realization
        evidence resolves against what she saw THIS turn (this turn's
        recall plus the session sheet so far), because the close-time sheet
        is not what she was looking at when she wrote the fence."""
        turn = _ns(run, "_turn")
        visit = _ns(run, "_visit")
        llm = turn.get("llm") or {}
        marked = clean_model_reply(str(llm.get("content") or ""))
        notices = list(llm.get("act_only_warnings") or [])
        if marked:
            sheet_now = [(str(r), str(d)) for r, d in (visit.get("sheet") or [])]
            sheet_lines = [f"{i}. {d}" for i, (_r, d) in enumerate(sheet_now, start=1)]
            marked, feelings, f_notes = parse_feel_blocks(marked, sheet_lines)
            marked, interests, i_notes = parse_interest_blocks(marked)
            marked, lessons, l_notes = parse_lesson_blocks(marked)
            marked, realize_elections, r_notes = parse_realize_blocks(marked)
            marked, topics, t_notes = parse_topic_blocks(marked)
            notices.extend(f_notes + i_notes + l_notes + r_notes + t_notes)

            pending = visit.setdefault(
                "pending_elections",
                {"feelings": [], "interests": [], "lessons": [], "realizations": [], "topics": []},
            )
            if feelings:
                # Resolve targets NOW: numbered targets index the sheet the
                # entity saw this turn, and those numbers shift as the visit
                # grows — close-time resolution would point at the wrong
                # records. "session" targets wait for the close (the summary
                # record does not exist yet).
                now_feelings = [f for f in feelings if str(f.target_token).strip().lower() != "session"]
                later_feelings = [f for f in feelings if str(f.target_token).strip().lower() == "session"]
                resolved_now, res_notes = resolve_feeling_targets(
                    now_feelings,
                    sheet_record_ids=[r for r, _ in sheet_now],
                    session_record_id=None,
                    self_id=home.entity_id,
                )
                notices.extend(res_notes)
                pending["feelings"].extend(
                    {**dataclasses.asdict(f), "record_id": rid} for f, rid in resolved_now
                )
                pending.setdefault("session_feelings", []).extend(
                    dataclasses.asdict(f) for f in later_feelings
                )
            if interests:
                pending["interests"].extend(str(x) for x in interests)
            if lessons:
                pending["lessons"].extend(str(x) for x in lessons)
            if topics:
                pending["topics"].extend(str(x) for x in topics)
            if realize_elections:
                # Evidence space = this turn's recalled records + the sheet
                # so far (both genuinely in front of her when she wrote).
                seen_rids = [r for r, _ in sheet_now]
                for h in list(turn.get("displayed") or []):
                    rid = str((h or {}).get("record_id") or "")
                    if rid and rid not in seen_rids:
                        seen_rids.append(rid)
                for _re in realize_elections:
                    gids: List[str] = []
                    for tok in _re.evidence:
                        bare = str(tok).strip().lstrip("#")
                        if not bare:
                            continue
                        if ":" in bare:
                            if bare in seen_rids and bare not in gids:
                                gids.append(bare)
                            continue
                        tails = [r for r in seen_rids if r.endswith(bare)]
                        if len(tails) == 1 and tails[0] not in gids:
                            gids.append(tails[0])
                    if gids:
                        pending["realizations"].append(
                            {"text": _re.text, "touches": _re.touches, "gids": gids}
                        )
                    else:
                        notices.append(
                            "#FALLBACK realization refused (no evidence resolved against "
                            f'what you saw this turn): "{_re.text[:60]}"'
                        )
            staged_count = (
                len(pending["feelings"]) + len(pending.get("session_feelings") or [])
                + len(pending["interests"]) + len(pending["lessons"])
                + len(pending["realizations"]) + len(pending["topics"])
            )
            if staged_count and (feelings or interests or lessons or realize_elections or topics):
                notices.append("[elections noted - they form at visit close]")
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
        # Speaker attribution in permanent prose (Veya deep check P2):
        # payload-claims-dropped held for stamps but not for digest text — a
        # visitor could engrave any name into the record. The prose label
        # now derives from the VERIFIED participants; a claimed label that
        # matches nobody verified is kept as a claim in attributes, never
        # written as fact.
        verified = [str(p) for p in (visit.get("participants") or []) if str(p).strip()]
        claimed = str(turn.get("speaker") or "").strip()
        speaker_claimed_label: Optional[str] = None
        if claimed and claimed in verified:
            speaker = claimed
        else:
            speaker = verified[0] if verified else home.entity_id
            if claimed:
                speaker_claimed_label = claimed
        title, digest, keywords = mechanical_digest_v2(
            turn["text"], turn["marked_reply"], home.name, speaker=speaker
        )
        turn["digest"] = digest
        # W5 (the visit half of the one verbatim edit): adapter-reported
        # intermediate rounds rest as inner speech + returned results.
        verbatim = f"{speaker}:\n{turn['text']}\n\n"
        phases = list(turn.get("lookup_phases") or [])
        results = list(turn.get("tool_results_at_rest") or [])
        for i, phase in enumerate(phases):
            verbatim += f"{home.name} (thinking, unspoken):\n{phase}\n\n"
            if i < len(results) and results[i].strip():
                verbatim += f"(what the tools returned:)\n{results[i]}\n\n"
        verbatim += f"{home.name}:\n{turn['marked_reply']}"
        attributes: Dict[str, Any] = {
            "participants": list(visit["participants"]),
            "digest_method": "mechanical-v2",
            # r-rt-3: awake-phase provenance for the origin labels.
            "phase": "visit",
        }
        if speaker_claimed_label:
            # The claim is recorded AS a claim (honesty both directions:
            # not engraved as fact, not silently thrown away).
            attributes["speaker_label_claimed"] = speaker_claimed_label
        if visit_id:
            attributes["visit_id"] = str(visit_id)  # item-14 correlation key
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
            # Append-once decorated user message (head discipline: the
            # transcript IS what was sent — all-but-last stays byte-stable
            # across turns); the raw words live in the formed verbatim.
            history.append({"role": "user", "content": turn.get("rendered_user") or turn["text"]})
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
                    # G1 at-rest rule, sheet edition (memory's e-s 233 rider):
                    # the sheet rests in run vars/ledger and rides the
                    # reflection prompt whose reply persists graph-ward — a
                    # private entry's line is the act-frame ONLY (the "private"
                    # word named, never the gist; capture_diary_elections
                    # already omits private gists from meta — the proximity
                    # pin in act_only.py guards that omission).
                    if str(meta.get("visibility") or "") == "private":
                        sheet.append([str(projected), "you kept a private diary entry"])
                    else:
                        gist = str(meta.get("gist") or "(no gist elected)")
                        sheet.append([str(projected), f"you kept a diary entry ({meta.get('kind')}): {gist}"])
            visit["sheet"] = sheet
            visit["last_folded_turn"] = turn["turn_id"]
        return StepPlan(
            node_id="ANSWER",
            effect=Effect(
                type=EffectType.ANSWER_USER,
                payload={
                    "message": turn["marked_reply"],
                    "turn_id": turn["turn_id"],
                    # H7a tools_ran parity: the door serves this list as the
                    # turn's tool truth (driver-authored; [] on the v0
                    # single-call path where no tool can run by construction).
                    "tools_ran": list(turn.get("tools_ran") or []),
                },
                result_key="_turn.answered",
            ),
            next_node="PARK",
        )

    def reflect_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        sheet = list(visit.get("sheet") or [])
        if visit.get("skip_reflection"):
            # closed_by=pause is a HARD FREEZE (gateway 0014/094354Z):
            # nothing runs — no reflection LLM call. The look-back debt is
            # honored by the door's pending-look-back at the next open.
            return StepPlan(node_id="REFLECT", next_node="DONE")
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
        feelings (summary first — target=session needs its record id).

        REFUSAL TOLERANCE (gateway c709 interim, the fdf01e0 rule class):
        every staged effect opts into `_absorb_failure` — a door refusal of
        ONE election (e.g. the close-reflection channel collision agency
        found: interest FORM into self refused under the visit's workplace
        stamp) lands as a loud #FALLBACK notice in the reflection output
        and the REMAINING stages still apply; it never terminal-fails the
        close. A refusal is the gate doing its job — the workflow dying on
        it converts a policy refusal into a dead visit and silently drops
        the diary + feelings queued behind it."""
        visit = _ns(run, "_visit")
        refl = _ns(run, "_reflect")

        def _note_absorbed(stage_key: str, label: str) -> None:
            out = refl.get(stage_key)
            if isinstance(out, dict) and out.get("absorbed_failure") and not out.get("_noted"):
                refl["notices"] = list(refl.get("notices") or []) + [
                    f"#FALLBACK reflection {label} was refused and skipped: "
                    f"{out['absorbed_failure']}"
                ]
                out["_noted"] = True

        _note_absorbed("summary_out", "summary")
        _note_absorbed("interest_out", "interest election")
        _note_absorbed("lesson_out", "lesson election")
        _note_absorbed("diary_out", "diary election")
        _note_absorbed("feel_out", "feeling election")
        if "marked_reply" not in refl:
            raw = clean_model_reply(str((refl.get("llm") or {}).get("content") or ""))
            _sheet = list(visit.get("sheet") or [])
            _sheet_lines = [f"{i}. {desc}" for i, (_rid, desc) in enumerate(_sheet, start=1)]
            marked, feelings, notices = parse_feel_blocks(raw, _sheet_lines)
            marked, interests, i_notes = parse_interest_blocks(marked)
            marked, lessons, l_notes = parse_lesson_blocks(marked)
            # REALIZATIONS (identity-pass spine, adversary F2: the contract
            # teaches the fence in every lane, so the durable visit lane
            # must parse it — an untaught-lane raw fence delivered to the
            # visitor was the finding). Evidence resolves against the VISIT
            # SHEET (what he actually saw this visit): full graph ids must
            # BE sheet rids; #hex-tails match a unique sheet rid tail.
            marked, realize_elections, r_notes = parse_realize_blocks(marked)
            _sheet_rids = [str(rid) for rid, _desc in _sheet if rid]
            _realizations: list = []
            for _re in realize_elections:
                _gids: list = []
                for _tok in _re.evidence:
                    _bare = str(_tok).strip().lstrip("#")
                    if not _bare:
                        continue
                    if ":" in _bare:
                        if _bare in _sheet_rids and _bare not in _gids:
                            _gids.append(_bare)
                        continue
                    _tails = [rid for rid in _sheet_rids if rid.endswith(_bare)]
                    if len(_tails) == 1 and _tails[0] not in _gids:
                        _gids.append(_tails[0])
                if _gids:
                    _realizations.append(
                        {"text": _re.text, "touches": _re.touches, "gids": _gids}
                    )
                else:
                    r_notes.append(
                        "#FALLBACK realization refused (no evidence resolved against "
                        f'this visit\'s sheet): "{_re.text[:60]}"'
                    )
            # Elected topics (operator directive 2026-07-19): parsed here,
            # stamped as attributes.topics on the summary stage below — the
            # engine's card-evidence seam; no extra APPLY stage needed (the
            # in-day topic card update is the chat lane's; visits ride the
            # sleep pass's full evidence scan).
            marked, topics, t_notes = parse_topic_blocks(marked)
            marked, diary_elections, d_notes = parse_diary_blocks(marked)
            refl["marked_reply"] = marked
            # MERGE the mid-turn staged elections (Veya deep check): turns
            # parsed and resolved them as they happened; the close stages
            # form them alongside the reflection's own. Turn-time items go
            # first (they were elected first).
            pending = dict(visit.get("pending_elections") or {})
            refl["feelings"] = (
                [dict(f) for f in (pending.get("session_feelings") or [])]
                + [dataclasses.asdict(f) for f in feelings]
            )
            refl["pending_resolved_feelings"] = [dict(f) for f in (pending.get("feelings") or [])]
            refl["interests"] = [str(x) for x in (pending.get("interests") or [])] + list(interests)
            refl["lessons"] = [str(x) for x in (pending.get("lessons") or [])] + list(lessons)
            refl["topics"] = [str(x) for x in (pending.get("topics") or [])] + list(topics)
            refl["diary"] = [dataclasses.asdict(e) for e in diary_elections]
            refl["realizations"] = (
                [dict(r) for r in (pending.get("realizations") or [])] + list(_realizations)
            )
            refl["notices"] = (
                list(notices) + list(i_notes) + list(l_notes) + list(r_notes)
                + list(t_notes) + list(d_notes)
            )
            refl["stage"] = "summary"
            refl["i"] = 0

        sheet = list(visit.get("sheet") or [])
        stage = str(refl.get("stage") or "summary")
        # RUN-SCOPED reflection ids (the chat-lane adversary's live-verified
        # P1, same class here: APPRAISE/DIARY event-ids derive from turn_id
        # with NO run component, so constant ids collided ACROSS VISITS and
        # a second visit's identical genuine feeling/entry was silently
        # swallowed by the at-least-once dedup). run_id is stable within a
        # run — crash-replay still re-derives identically — and unique
        # across runs, which is exactly the dedup boundary wanted.
        rid_scope = str(run.run_id)

        if stage == "summary":
            refl["stage"] = "interest"
            refl["i"] = 0
            # MECHANICAL FLOOR (r-rt-2, Ephemeral incident): never a
            # marker-only digest — the sheet narrates when the look-back
            # reply carried no prose. Floored digests self-identify via
            # digest_method (memory co-sign: the redigestion poverty scan
            # keys on it).
            refl_digest, refl_floored = floored_reflection_digest(
                str(refl["marked_reply"]), [(r, d) for r, d in sheet]
            )
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "summary",
                            "title": f"session reflection: {run.session_id}",
                            "digest": refl_digest,
                            "keywords": [],
                            "verbatim": str(refl["marked_reply"]),
                            "edges": [["summarizes", rid] for rid, _ in sheet if rid],
                            "attributes": {
                                "participants": list(visit["participants"]),
                                "session_id": str(run.session_id or ""),
                                "phase": "visit",  # r-rt-3
                                # Elected topics fan to topic:<words> card
                                # targets in memory's evidence scan.
                                **({"topics": [str(t) for t in refl.get("topics") or []]}
                                   if refl.get("topics") else {}),
                                **({"digest_method": "mechanical-floor-v1"} if refl_floored else {}),
                                **({"visit_id": str(visit_id)} if visit_id else {}),
                            },
                            "provenance": {"source": "entity-visit-run-reflection-v0"},
                        }],
                        "scope": "life",
                        "owner_id": home.entity_id,
                        "turn_id": f"t-reflect-{rid_scope}",
                        "_absorb_failure": True,
                    },
                    result_key="_reflect.summary_out",
                ),
                next_node="APPLY",
            )

        if stage == "interest":
            interests = list(refl.get("interests") or [])
            i = int(refl.get("i") or 0)
            if i >= len(interests):
                refl["stage"] = "lesson"
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
                        "turn_id": f"t-reflect-interest-{rid_scope}-{i}",
                        "_absorb_failure": True,
                    },
                    result_key="_reflect.interest_out",
                ),
                next_node="APPLY",
            )

        if stage == "lesson":
            # LESSONS — semantic knowledge (laurent's directive 2026-07-18;
            # same staged shape as interests, LIFE scope: knowledge is
            # recallable world-stuff, not identity core).
            lessons = list(refl.get("lessons") or [])
            i = int(refl.get("i") or 0)
            if i >= len(lessons):
                refl["stage"] = "realize"
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
                            "kind": "lesson",
                            "title": "lesson: " + " ".join(str(lessons[i]).split()[:8]),
                            "digest": str(lessons[i]),
                            "keywords": [],
                            "edges": (
                                [["from_session", str(session_record_id)]] if session_record_id else []
                            ),
                            "attributes": {
                                "session_id": str(run.session_id or ""),
                                "phase": "visit",
                            },
                            "provenance": {
                                "source": "entity-visit-run-reflection-v0",
                                "actor": "entity-reflection",
                            },
                        }],
                        "scope": "life",
                        "owner_id": home.entity_id,
                        "turn_id": f"t-reflect-lesson-{rid_scope}-{i}",
                        "_absorb_failure": True,
                    },
                    result_key="_reflect.lesson_out",
                ),
                next_node="APPLY",
            )

        if stage == "realize":
            # REALIZATIONS (identity-pass spine, F2 wiring): PROPOSALS about
            # the self — kind=realization, SELF scope, derived_from edges to
            # sheet-resolved evidence; INERT on formation, the sleep pass is
            # the only enactor. _absorb_failure carries the lane's standing
            # posture: a gate/engine refusal (kind vocabulary lands with
            # memory's half; workplace-channel rules are the door's) lands
            # loudly in the result and the close never dies over an election.
            realizations = list(refl.get("realizations") or [])
            i = int(refl.get("i") or 0)
            if i >= len(realizations):
                refl["stage"] = "diary"
                refl["i"] = 0
                return StepPlan(node_id="APPLY", next_node="APPLY")
            refl["i"] = i + 1
            r = realizations[i] if isinstance(realizations[i], dict) else {}
            r_text = str(r.get("text") or "")
            r_attrs: Dict[str, Any] = {
                "session_id": str(run.session_id or ""),
                "phase": "visit",
            }
            if r.get("touches"):
                r_attrs["touches"] = str(r["touches"])
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "realization",
                            "title": "realization: " + " ".join(r_text.split()[:8]),
                            "digest": r_text,
                            "keywords": [],
                            "edges": [["derived_from", str(g)] for g in (r.get("gids") or [])],
                            "attributes": r_attrs,
                            "provenance": {
                                "source": "entity-visit-run-reflection-v0",
                                "actor": "entity-reflection",
                            },
                        }],
                        "scope": "self",
                        "owner_id": home.entity_id,
                        "turn_id": f"t-reflect-realize-{rid_scope}-{i}",
                        "_absorb_failure": True,
                    },
                    result_key="_reflect.realize_out",
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
                        "explores": e.get("explores"),
                        "turn_id": f"t-reflect-diary-{rid_scope}-{i}",
                        "anchor_record_ids": session_graph_ids,
                        "anchor_graph_ids": session_graph_ids,
                        "_absorb_failure": True,
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
            # Turn-time feelings were resolved when they were elected (their
            # numbered targets meant THAT turn's sheet); they apply first.
            # One session cap across both sources, refused loudly.
            from .reflection import MAX_FEELINGS_PER_SESSION

            merged = [dict(f) for f in (refl.get("pending_resolved_feelings") or [])] + [
                {**dataclasses.asdict(f), "record_id": rid} for f, rid in resolved
            ]
            if len(merged) > MAX_FEELINGS_PER_SESSION:
                notes = list(notes) + [
                    f"#FALLBACK {len(merged) - MAX_FEELINGS_PER_SESSION} feeling(s) refused "
                    f"(cap {MAX_FEELINGS_PER_SESSION}/session across the visit)"
                ]
                merged = merged[:MAX_FEELINGS_PER_SESSION]
            refl["resolved_feelings"] = merged
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
                    "turn_id": f"t-reflect-feel-{rid_scope}-{i}",
                    "scope": "life" if target.startswith("ex:") else "self",
                    "owner_id": home.entity_id,
                    "actor": "entity-reflection",
                    "_absorb_failure": True,
                },
                result_key="_reflect.feel_out",
            ),
            next_node="APPLY",
        )

    def done_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        refl = _ns(run, "_reflect")
        out: Dict[str, Any] = {
            "ok": True,
            "turns": int(visit.get("turn_n") or 0),
            "close_reason": str(visit.get("close_reason") or "closed"),
            "reflection_notices": list(refl.get("notices") or []),
        }
        if visit.get("closed_by"):
            out["closed_by"] = str(visit["closed_by"])
        if visit.get("close_note"):
            out["close_note"] = str(visit["close_note"])
        sheet = list(visit.get("sheet") or [])
        if visit.get("skip_reflection") and sheet:
            # The look-back DEBT is explicit on the run output (a paused
            # visit's reflection is owed, not forgotten): the door's next
            # open runs the pending look-back over this sheet. The sheet is
            # word-free by construction — episode digests + non-private
            # gists; private entries appear as their act label only.
            out["reflection_pending"] = True
            out["sheet"] = sheet
        return StepPlan(node_id="DONE", complete_output=out)

    nodes: Dict[str, Any] = {
        "OPEN": open_node,
        "PARK": park_node,
        "ROUTE": route_node,
        "RECALL": recall_node,
        "RENDER": render_node,
        "REASON": reason_node,
        "ELECT": elect_node,
        "COMMIT": commit_node,
        "FORM": form_node,
        "ANSWER": answer_node,
        "REFLECT": reflect_node,
        "APPLY": apply_node,
        "DONE": done_node,
    }
    if react_middle is not None:
        collisions = sorted(set(react_middle.nodes.keys()) & set(nodes.keys()))
        if collisions:
            raise ValueError(
                "react_middle node ids collide with the visit graph "
                f"({', '.join(collisions)}) - adapter ids must not shadow seam nodes"
            )
        if str(react_middle.entry) not in react_middle.nodes:
            raise ValueError(
                f"react_middle.entry {react_middle.entry!r} is not in its own node map"
            )
        nodes.update(react_middle.nodes)
        nodes["REASON"] = bridge_node  # RENDER routes in unchanged
        nodes[HARVEST_NODE] = harvest_node  # the middle's declared exit
    return WorkflowSpec(
        workflow_id=VISIT_WORKFLOW_ID,
        entry_node="OPEN",
        nodes=nodes,
    )
