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
  payloads: `{text, speaker, attachments?}` (turn; attachments = the
  door's structured records, see `_attachment_records`), `{kind: "close"}` (explicit close),
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
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Effect, EffectType, RunState, StepPlan
from ..core.spec import WorkflowSpec
from ..core.vars import DEFAULT_MAX_ITERATIONS
from ..memory.token_budget import estimate_message_tokens
from ..session_history import HISTORY_REPLAY_MAX_TOKENS, window_report, window_transcript
from .chat import (
    SUMMON_POSTURE_SELF_FRACTION,
    ChatHome,
    _memories_block,
    _serialize_under_budget,
    authored_digest_attributes,
    clean_model_reply,
    compose_system_base,
    diary_notice,
    floored_reflection_digest,
    parse_diary_blocks,
    served_turn,
    sheet_prompt_line,
)
from .digest import mechanical_digest_v2
from .prelude import render_summon_prelude
from .prompt_overlay import overlay_note, read_prompt_overlay
from .reflection import (
    FeelingElection,
    build_reflection_prompt,
    elected_title_and_words,
    interest_marker,
    lesson_marker,
    parse_elected_title,
    parse_feel_blocks,
    parse_interest_blocks,
    parse_lesson_blocks,
    parse_realize_blocks,
    parse_topic_blocks,
    realization_marker,
    resolve_feeling_targets,
)

logger = logging.getLogger(__name__)

VISIT_WORKFLOW_ID = "entity-visit@1"
VISITOR_WAIT_KEY = "visitor_input"
DEFAULT_IDLE_SECONDS = 30 * 60  # a visit left silent this long closes with reflection
HARVEST_NODE = "HARVEST"  # the react middle's exit contract (final_next_node)
# A VISIT NEVER FAILS (operator 2026-11-09): a turn whose model call fails (after the effect policy's retries) answers
# the error for THAT turn and parks - the visit continues. Every model call of a turn is routed through MODEL_GATE
# (the failure is absorbed into its result key) and a failed one goes to TURN_FAILED.
MODEL_GATE_NODE = "VISIT_MODEL_GATE"
TURN_FAILED_NODE = "VISIT_TURN_FAILED"

# HISTORY WINDOW (operator ruling 2026-09-28, ADR-0026): the visit's
# history rides every prompt through THE runtime history window
# (`session_history.window_transcript` over `fold_history_window`): the most
# recent 50,000 tokens of whole turns, never a cut message, never a count cap,
# the receipt recorded at `_runtime.session_history`. On the react arm (the
# gateway's only arm) the transcript is the durable `context.messages`, kept
# whole; BRIDGE asks the adapter to send each request through the window
# (`_runtime.history_window_tokens`). An adapter too old to honor the key
# (abstractagent < 0.3.17) sends the whole transcript: the runtime cannot
# require its own dependent, so HARVEST keeps the turn, logs a warning and
# records `window_applied: false, reason: "agent_too_old"` in the run. The earlier 10-turn
# slice and the per-message re-send caps this lane seeded into `_limits`
# (32,000 chars per tool result, 80,000 per message) are gone: the model uses
# its full context. The 2026-08-01 poison class (a 494,932-char tool result
# replayed whole every turn) stays bounded by the adapter's own always-on
# guard (abstractagent adapters/transcripts.py OVERSIZED_MESSAGE_CLAMP_CHARS).


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
    # Reason-cycle bound per turn: the framework default (DEFAULT_MAX_ITERATIONS,
    # operator 2026-11-08 "50 everywhere"; caps bound RUN-AWAY, not ambition).
    # BRIDGE seeds `_limits.max_iterations` with setdefault, so a door- or
    # operator-supplied value always wins over this default.
    max_iterations: int = DEFAULT_MAX_ITERATIONS


def _ns(run: RunState, key: str) -> Dict[str, Any]:
    ns = run.vars.get(key)
    if not isinstance(ns, dict):
        ns = {}
        run.vars[key] = ns
    return ns


def _windowed_history(run: RunState, visit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The visit's history as the prompt carries it: THE history window
    (`session_history.window_transcript` — the most recent 50k tokens of
    whole turns, a labeled notice when older turns were dropped), its receipt
    recorded in the run at `_runtime.session_history` like session replay."""
    window = window_transcript(list(visit.get("history") or []))
    run.vars.setdefault("_runtime", {})["session_history"] = dict(window.report)
    return list(window)


# ATTACHMENTS (operator 2026-11-08: images "already work normally with every
# other client — this must work" for entities too). A visit turn's resume
# payload may carry `attachments`: the door's STRUCTURED records, one dict per
# file the visitor attached, `path` workspace-relative (the same string the
# workspace write returned) plus whatever the door recorded (bytes, sha256,
# content_type). They are never parsed from the visitor's words. The runtime
# keeps the records whole on the turn, says one honest line per file in the
# rendered message, shows this turn's IMAGES to the model on every call of
# the turn (`context.attachments`, cleared at HARVEST — the next turn does
# not re-send them; a re-look is read_file, the sight lane), and folds the
# records into the user message and the durable history entry.


def _attachment_records(raw: Any) -> List[Dict[str, Any]]:
    """The door's attachment records, kept whole: dicts with a non-empty
    `path`. Anything else is not an attachment."""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict) and str(item.get("path") or "").strip():
            rec = dict(item)
            rec["path"] = str(item["path"]).strip()
            out.append(rec)
    return out


def _attachment_facts(home_dir: Any, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the file itself says about each attachment (content, never the
    name): `{path, label, bytes, image, abs_path, problem}`. `image` is True
    for a PNG/JPEG/GIF/WEBP within the workspace acceptance cap — the one
    case the model is shown the file; `problem` names why a record could
    not be opened (the file vanished, left the workspace, is over the cap)
    so the rendered line stays honest instead of silently dropping it."""
    from .tools import WORKSPACE_FILE_CAP_BYTES, WorkspaceRoot, sniff_binary, sniff_image

    ws = WorkspaceRoot(home_dir)
    facts: List[Dict[str, Any]] = []
    for rec in records:
        rel = str(rec.get("path") or "")
        fact: Dict[str, Any] = {
            "path": rel, "label": "", "bytes": 0, "image": False, "abs_path": None, "problem": "",
        }
        try:
            path = ws.resolve(rel)
            if not path.is_file():
                raise FileNotFoundError(f"no file at {rel!r} in the workspace")
            size = int(path.stat().st_size)
            with open(path, "rb") as fh:
                head = fh.read(8000)
        except (PermissionError, FileNotFoundError) as exc:
            fact["problem"] = str(exc)  # the entity's own workspace words
            facts.append(fact)
            continue
        except (OSError, ValueError, RuntimeError):
            # An embedded NUL (ValueError), a name too long (OSError), a
            # symlink loop (RuntimeError before Python 3.13): never a crash
            # out of ROUTE, never a host path echoed into the prompt.
            fact["problem"] = "not a readable file path"
            facts.append(fact)
            continue
        fact["bytes"] = size
        image = sniff_image(head)
        if image is not None:
            fact["label"] = image[0]
            if size <= WORKSPACE_FILE_CAP_BYTES:
                fact["image"] = True
                fact["abs_path"] = str(path)
            else:
                fact["problem"] = f"over the {WORKSPACE_FILE_CAP_BYTES}-byte workspace cap"
        else:
            binary = sniff_binary(head)
            fact["label"] = binary if binary is not None else "a text file"
        facts.append(fact)
    return facts


def _attachment_lines(facts: List[Dict[str, Any]]) -> List[str]:
    """One runtime-authored line per attachment for the rendered message
    (prompt currency: never on `turn.text`, the human's raw words)."""
    lines: List[str] = []
    for f in facts:
        if f.get("problem"):
            lines.append(f"(attached: {f['path']} — could not be opened: {f['problem']})")
        elif f.get("image"):
            # The fact of the call (the host attaches it), not a claim about
            # perception — a text-only mind may be given nothing by its provider.
            lines.append(
                f"(attached: {f['path']} — {f['label']}, {int(f['bytes']):,} bytes; "
                "attached to this message as an image; read_file re-opens it)"
            )
        else:
            lines.append(f"(attached: {f['path']} — {f['label']}, {int(f['bytes']):,} bytes; read_file opens it)")
    return lines


def _deadline_iso(idle_seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=float(idle_seconds))).isoformat()


AUTHOR_EPISODE_NODE = "VISIT_AUTHOR"


def _captured_diary_marker(marked: str, meta: Dict[str, Any]) -> str:
    """The exact marker the act-only capture left for one diary entry (act_only.capture_diary_elections: the plain
    "[kept in diary - <kind>" / "[kept a private diary entry" head, the verified resolution note, then
    " - reread: diary_read <entry_id>]"), read back from the marked reply by its entry id; "" when it is not there."""
    entry_id = str((meta or {}).get("entry_id") or "")
    if not entry_id:
        return ""
    tail = f" - reread: diary_read {entry_id}]"
    end = marked.find(tail)
    if end < 0:
        return ""
    head = ("[kept a private diary entry" if str(meta.get("visibility") or "") == "private"
            else f"[kept in diary - {meta.get('kind')}")
    start = marked.rfind(head, 0, end + 1)
    return marked[start:end + len(tail)] if start >= 0 else ""


def _staged_notice(kind: str, body: str) -> Dict[str, Any]:
    """A visit-lane election staged for the close (it forms there, so it has no record yet): record_id None,
    staged true, its elected title ("" without one: the close authors it) and his words, whole."""
    title, words, _reason = parse_elected_title(body)
    return {"kind": kind, "record_id": None, "title": str(title or ""), "text": words, "staged": True}


def _authoring_step(node_id: str, kind: str, body: str, *, result_key: str, next_node: str) -> StepPlan:
    """The formation's authoring call as a durable LLM_CALL effect tagged with the summaries capability
    (identity/authoring.py: authoring_llm_call_payload; the host routes `payload.capability`)."""
    from .authoring import authoring_llm_call_payload

    return StepPlan(
        node_id=node_id,
        effect=Effect(type=EffectType.LLM_CALL, payload=authoring_llm_call_payload(kind, body), result_key=result_key),
        next_node=next_node,
    )


def _authored_fields(result: Any, *, need: List[str], run_id: str, title: Optional[str] = None,
                     digest: str = "") -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Record fields from one authoring call: (fields, failure). Authored = the title/summary set and nothing
    pending; failed = the needed fields PENDING (memory queues the task) and the failure to log once the record
    has an id. A call no handler served (no capability echo) is a wiring failure, logged as an error."""
    from .authoring import CapabilityUnrouted, authoring_failure_fields, parse_authored_result

    try:
        authored = parse_authored_result(result)
    except Exception as e:  # noqa: BLE001
        if isinstance(e, CapabilityUnrouted):
            logger.error("entity visit run %s: %s", run_id, e)
        why = authoring_failure_fields(e)
        fields: Dict[str, Any] = {"title": None if "title" in need else title, "digest": digest, "pending": list(need)}
        # The capability call's own provider/model when its result names them (the visit's mind is not the model
        # that was asked: never stamped in their place).
        res = result if isinstance(result, dict) else {}
        return fields, {"reason": why["reason"], "error": f"{type(e).__name__}: {e}",
                        "error_class": why["error_class"], "error_message": why["error_message"],
                        "provider": res.get("provider"), "model": res.get("model"),
                        "missing": "both" if len(need) == 2 else need[0]}
    fields = {
        "title": authored["title"] if "title" in need else title,
        "digest": authored["summary"] if "summary" in need else digest,
        "_metadata_authored": {"fields": list(need), "capability": "summaries", "model": authored.get("model"),
                               "run_id": run_id},
    }
    return fields, None


def _log_visit_authoring_failure(home: Any, visit: Dict[str, Any], failure: Dict[str, Any],
                                 formed: Any) -> None:
    from .title_failures import record_authoring_failure

    ids = list((formed or {}).get("record_ids") or []) if isinstance(formed, dict) else []
    record_authoring_failure(
        getattr(home, "home_dir", None), entity=getattr(home, "entity_id", None),
        record_id=str(ids[0]) if ids else None, kind=str(failure.get("kind")),
        raw_block=str(failure.get("raw_block") or ""), template_id=str(failure.get("template_id") or ""),
        provider=failure.get("provider"), model=failure.get("model"), reason=str(failure.get("reason")),
        missing=str(failure.get("missing") or "title"),
        extra={"capability": "summaries",
               **{k: failure[k] for k in ("error", "error_class", "error_message", "turn_id", "block_reason")
                  if failure.get(k)}},
    )


def _get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in [p for p in str(dotted or "").split(".") if p]:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _guard_model_call(run: RunState, plan: StepPlan) -> StepPlan:
    """Route one planned model call through MODEL_GATE: its final failure is absorbed into its result key (the
    runtime's `_absorb_failure`) instead of failing the visit run, and the gate sends the turn to TURN_FAILED."""
    eff = plan.effect
    if eff is None or eff.type != EffectType.LLM_CALL or not plan.next_node or not eff.result_key:
        return plan
    gate = _ns(run, "_visit_model_gate")
    gate["after"] = str(plan.next_node)
    gate["result_key"] = str(eff.result_key)
    payload = dict(eff.payload or {})
    payload["_absorb_failure"] = True
    return dataclasses.replace(
        plan, effect=dataclasses.replace(eff, payload=payload), next_node=MODEL_GATE_NODE,
    )


def _guarded_node(handler: Callable[[RunState, Any], StepPlan]) -> Callable[[RunState, Any], StepPlan]:
    def _node(run: RunState, ctx: Any) -> StepPlan:
        return _guard_model_call(run, handler(run, ctx))

    return _node


def build_visit_workflow(
    home: ChatHome,
    *,
    participants: Optional[List[str]] = None,
    budget_profile: Optional[Dict[str, Any]] = None,
    prelude_budget: int = 1600,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    model_info: Optional[Dict[str, str]] = None,
    visit_id: Optional[str] = None,
    react_middle: Optional[ReactMiddle] = None,
    identity_since: Optional[str] = None,
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
    the SAME string, correlating as DATA, never as shared rows.

    `identity_since` is the previous summon's time (aware ISO-8601) for the
    identity cue (ADR 0052 section 7); the door seeds it as
    `_visit.identity_since` at run creation like `_visit.idle_seconds`, and
    the seeded value wins. None tells the change that made the current
    identity."""

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
                home_dir=home.home_dir,
                identity_since=visit.get("identity_since", identity_since),
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
        # The door's structured attachment records, kept whole (never
        # parsed from `text`); RENDER says them, BRIDGE shows the images.
        records = _attachment_records(resume.get("attachments"))
        if not text and not records:
            # An empty message is not a turn; park again (honest no-op).
            return StepPlan(node_id="ROUTE", next_node="PARK")
        turn_n = int(visit.get("turn_n") or 0) + 1
        visit["turn_n"] = turn_n
        turn.clear()
        # An image-only turn (files, no words) is a turn: `text` stays "",
        # the rendered message is the decoration alone.
        turn["text"] = text
        turn["speaker"] = str(resume.get("speaker") or "") or (visit["participants"][0])
        turn["turn_id"] = f"t-{turn_n:04d}"
        turn["attachments"] = records
        # What the files themselves say, read once here: the lines RENDER
        # says (and RECALL cues on when there are no words), the images
        # BRIDGE/REASON show.
        facts = _attachment_facts(home.home_dir, records)
        turn["attachment_lines"] = _attachment_lines(facts)
        turn["attachment_media"] = [str(f["abs_path"]) for f in facts if f.get("image") and f.get("abs_path")]
        return StepPlan(node_id="ROUTE", next_node="RECALL")

    def recall_node(run: RunState, ctx: Any) -> StepPlan:
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        return StepPlan(
            node_id="RECALL",
            effect=Effect(
                type=EffectType.MEMORY_RECALL,
                payload={
                    # No words (an image-only turn): the cue is what was
                    # shown — the attachment lines (file names), never "".
                    "cue_text": turn["text"] or "\n".join(turn.get("attachment_lines") or []),
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
        # Attachments: one honest line per file (ROUTE read the files), INSIDE
        # the decoration, before the visitor's words — never after them.
        attached = "\n".join(turn.get("attachment_lines") or [])
        decoration = "\n\n".join(s for s in (presence, block, attached) if s)
        # The decorated message is APPEND-ONCE: it enters the transcript at
        # the fold and STAYS (all-but-last message byte-identical across
        # turns = the cross-turn cache property; each block is dated +
        # as_of-labeled, so an old block reads as the honest record of what
        # that moment reminded him of, never as current recall). The RECORD
        # keeps the human's raw words: formation verbatim uses turn.text.
        # An image-only turn has no words: the message is the decoration.
        text = str(turn.get("text") or "")
        turn["rendered_user"] = (decoration + "\n\n" + text) if (decoration and text) else (decoration or text)
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
        # THE history window on every model request of this cycle: the adapter
        # sends `window_transcript(context.messages)` and records the receipt
        # at `_runtime.session_history` (cleared here so HARVEST sees this
        # turn's receipt, never a stale one). The stored transcript stays whole.
        runtime_ns["history_window_tokens"] = HISTORY_REPLAY_MAX_TOKENS
        runtime_ns.pop("session_history", None)
        # Fresh per-turn adapter state; append the decorated turn message to
        # the durable transcript (the ONE source of truth under the merge).
        if callable(react_middle.reset_turn):
            react_middle.reset_turn(run.vars)
        context = run.vars.setdefault("context", {})
        # THIS turn's images ride every model call of the turn exactly as an
        # Assistant attachment rides its run (the adapter reads
        # `context.attachments` on each LLM_CALL); the previous turn's are
        # replaced, and HARVEST clears them — never a silent re-attach.
        media = [str(p) for p in (turn.get("attachment_media") or []) if p]
        if media:
            context["attachments"] = media
        else:
            context.pop("attachments", None)
        msgs = context.setdefault("messages", [])
        user_message: Dict[str, Any] = {
            "role": "user",
            "content": str(turn.get("rendered_user") or turn.get("text") or ""),
        }
        records = list(turn.get("attachments") or [])
        if records:
            # The durable transcript keeps the reference (path, bytes, sha256,
            # type); the file itself stays in the workspace.
            user_message["metadata"] = {"attachments": records}
        msgs.append(user_message)
        # The turn in progress starts at the visitor's message: the window keeps
        # it whole whatever user-role messages the loop adds inside it (an
        # ask_user answer, operator guidance).
        runtime_ns["history_window_turn_start"] = len(msgs) - 1
        return StepPlan(node_id="REASON", next_node=str(react_middle.entry))

    def harvest_node(run: RunState, ctx: Any) -> StepPlan:
        """The adapter cycle's exit -> ELECT: fold the turn's outcome into
        `_turn.llm` so every downstream node runs byte-unchanged."""
        runtime_ns = run.vars.setdefault("_runtime", {})
        # The turn's images were shown on this turn's calls only.
        context_ns = run.vars.get("context")
        if isinstance(context_ns, dict):
            context_ns.pop("attachments", None)
        report = runtime_ns.get("session_history")
        if isinstance(report, dict):
            report["window_applied"] = True
        else:
            # An adapter that ignores `_runtime.history_window_tokens`
            # (abstractagent < 0.3.17) sent the WHOLE visit transcript. The
            # runtime cannot require its own dependent (an upgraded runtime
            # under an old agent must not fail every visit turn), so the gap
            # is stated instead of silent: one warning, and the run says so.
            # What the last request carried: the transcript up to its last
            # non-assistant message (a request ends on a user or tool message;
            # what follows is the model's reply to it).
            stored = list((run.vars.get("context") or {}).get("messages") or [])
            end = len(stored)
            while end and isinstance(stored[end - 1], dict) and stored[end - 1].get("role") == "assistant":
                end -= 1
            sent = stored[:end]
            logger.warning(
                "abstractagent < 0.3.17 does not apply the history window; upgrade abstractagent "
                "(entity visit run %s sent its whole transcript: %d messages)",
                run.run_id, len(sent),
            )
            runtime_ns["session_history"] = {
                **window_report(
                    max_tokens=int(runtime_ns.get("history_window_tokens") or HISTORY_REPLAY_MAX_TOKENS),
                    replayed_messages=len(sent),
                    replayed_tokens=sum(estimate_message_tokens(m) for m in sent),
                ),
                "window_applied": False,
                "reason": "agent_too_old",
            }
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
        messages = _windowed_history(run, visit) + [
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
        payload: Dict[str, Any] = {
            "messages": messages,
            "system_prompt": visit["system_base"],
            "turn_id": turn["turn_id"],
            "anchor_record_ids": [h.get("record_id") for h in displayed],
            "anchor_graph_ids": anchor_graph_ids,
        }
        media = [str(p) for p in (turn.get("attachment_media") or []) if p]
        if media:
            # This turn's images, as the runtime's LLM handler reads them
            # (`payload.media`: file paths) — the same contract as BRIDGE's
            # `context.attachments` on the react arm.
            payload["media"] = media
        return _guard_model_call(run, StepPlan(
            node_id="REASON",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_turn.llm",
            ),
            next_node="ELECT",
        ))

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
        # (notice, marker) per act of this turn, served BESIDE the reply (turn.notices, the Entity app owner's
        # contract 2026-11-09); the marker leaves the served reply and stays in the stored one (ADR 0054).
        served: List[Tuple[Dict[str, Any], str]] = []
        for meta in list(llm.get("diary_entries") or []):
            marker = _captured_diary_marker(marked, meta)
            notice = diary_notice(kind=str(meta.get("kind") or ""), visibility=str(meta.get("visibility") or ""),
                                  record_id=meta.get("projected_record_id"), title=meta.get("title"))
            if marker and notice:
                served.append((notice, marker))
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
                served.extend((_staged_notice("interest", str(x)), interest_marker(str(x))) for x in interests)
            if lessons:
                pending["lessons"].extend(str(x) for x in lessons)
                served.extend((_staged_notice("lesson", str(x)), lesson_marker(str(x))) for x in lessons)
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
                            {"text": _re.text, "touches": _re.touches, "title": _re.title, "gids": gids, "title_reason": _re.title_reason, "raw": _re.raw}
                        )
                        served.append(({"kind": "realization", "record_id": None, "title": str(_re.title or ""),
                                        "text": _re.text, "staged": True}, realization_marker(_re.text)))
                    else:
                        notices.append(
                            "#FALLBACK realization refused (no evidence resolved against "
                            f'what you saw this turn): "{_re.text}"'
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
        # The SERVED reply and its notices (ANSWER serves them; the visit history and the episode keep `marked`).
        turn["reply"], turn["reply_notices"] = served_turn(marked, served)
        turn["diary_meta"] = list(llm.get("diary_entries") or [])
        turn["notices"] = notices
        return StepPlan(node_id="ELECT", next_node="COMMIT")

    def commit_node(run: RunState, ctx: Any) -> StepPlan:
        turn = _ns(run, "_turn")
        displayed = list(turn.get("displayed") or [])
        recall = turn.get("recall") or {}
        if not displayed or not recall.get("trace_id"):
            return StepPlan(node_id="COMMIT", next_node=AUTHOR_EPISODE_NODE)
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
            next_node=AUTHOR_EPISODE_NODE,
        )

    def author_episode_node(run: RunState, ctx: Any) -> StepPlan:
        """The episode's title and summary are AUTHORED in its formation (operator 2026-11-09): one LLM_CALL
        tagged capability=summaries over the exchange; FORM reads `_turn.authored`."""
        turn = _ns(run, "_turn")
        said = str(turn.get("text") or "") or "\n".join(turn.get("attachment_lines") or [])
        body = f"{turn.get('speaker') or 'visitor'}: {said}\n\n{home.name}: {turn.get('marked_reply') or ''}"
        return _authoring_step(AUTHOR_EPISODE_NODE, "episode", body, result_key="_turn.authored", next_node="FORM")

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
        # The human's raw words — or, on an image-only turn, what they showed
        # (the attachment lines), so the episode never records an empty
        # utterance against a turn that happened.
        said = str(turn.get("text") or "") or "\n".join(turn.get("attachment_lines") or [])
        title, digest, keywords = mechanical_digest_v2(
            said, turn["marked_reply"], home.name, speaker=speaker
        )
        turn["digest"] = digest
        # W5 (the visit half of the one verbatim edit): adapter-reported
        # intermediate rounds rest as inner speech + returned results.
        verbatim = f"{speaker}:\n{said}\n\n"
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
        # The authoring call's result (VISIT_AUTHOR): authored = no pending; failed = pending + a failure line
        # (written by ANSWER, once the record has its id).
        fields, failure = _authored_fields(turn.get("authored"), need=["title", "summary"], run_id=str(run.run_id),
                                           digest=digest)
        if fields.get("_metadata_authored"):
            attributes["metadata_authored"] = fields.pop("_metadata_authored")
            # The authored summary is the digest: its label says authored (memory queues a mechanical-v2 digest
            # for sleep repair - rewriting his authored words for nothing).
            attributes = authored_digest_attributes(attributes, attributes["metadata_authored"].get("fields"))
            digest = str(fields["digest"])
            turn["digest"] = digest
        if failure:
            turn["authoring_failure"] = {**failure, "kind": "episode", "raw_block": turn["marked_reply"],
                                         "template_id": "visit_workflow.episode", "turn_id": turn["turn_id"]}
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
                        # Authored metadata (operator 2026-11-09): the title and summary VISIT_AUTHOR wrote; only
                        # a failed call leaves them pending (the mechanical digest the provisional summary).
                        "title": fields.get("title"),
                        **({"pending": fields["pending"]} if fields.get("pending") else {}),
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
        failure = turn.pop("authoring_failure", None)
        if isinstance(failure, dict):
            _log_visit_authoring_failure(home, visit, failure, turn.get("formed"))
        # Fold the turn into the visit ONCE (replay guard: a crash between
        # ANSWER and the next park re-runs this node; the fold must not
        # double history).
        if visit.get("last_folded_turn") != turn["turn_id"]:
            history = list(visit.get("history") or [])
            # Append-once decorated user message (head discipline: the
            # transcript IS what was sent — all-but-last stays byte-stable
            # across turns); the raw words live in the formed verbatim.
            user_entry: Dict[str, Any] = {"role": "user", "content": turn.get("rendered_user") or turn["text"]}
            if turn.get("attachments"):
                # The durable visit transcript keeps the attachment reference
                # (path + bytes + sha256 + type); the file stays in the workspace.
                user_entry["attachments"] = list(turn["attachments"])
            history.append(user_entry)
            history.append({"role": "assistant", "content": turn["marked_reply"]})
            # The WHOLE visit stays here; the prompt reads it through the
            # history window (`_windowed_history`), never a count slice.
            visit["history"] = history
            sheet = list(visit.get("sheet") or [])
            formed_ids = list((turn.get("formed") or {}).get("record_ids") or [])
            for rid in formed_ids:
                sheet.append([str(rid), str(turn.get("digest") or "")])  # whole (root ADR 0054)
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
                        # The record's title as formed (Codex review 3 #9), never his first line standing in.
                        title = str(meta.get("title") or "").strip()
                        sheet.append([str(projected), f"you kept a diary entry ({meta.get('kind')}): {title}" if title
                                      else f"you kept a diary entry ({meta.get('kind')}) whose title is still being "
                                           "written"])
            visit["sheet"] = sheet
            visit["last_folded_turn"] = turn["turn_id"]
        return StepPlan(
            node_id="ANSWER",
            effect=Effect(
                type=EffectType.ANSWER_USER,
                payload={
                    # The SERVED reply (no bracket markers; a turn parked before 2026-11-09 has only its marked one).
                    "message": turn.get("reply", turn["marked_reply"]),
                    "turn_id": turn["turn_id"],
                    # H7a tools_ran parity: the door serves this list as the
                    # turn's tool truth (driver-authored; [] on the v0
                    # single-call path where no tool can run by construction).
                    "tools_ran": list(turn.get("tools_ran") or []),
                    # turn.notices (2026-11-09): what the turn formed (a diary entry: its graph projection) or
                    # staged for the visit's close (record_id None, staged true), as data beside the reply.
                    "notices": [dict(n) for n in (turn.get("reply_notices") or [])],
                },
                result_key="_turn.answered",
            ),
            next_node="PARK",
        )

    def model_gate_node(run: RunState, ctx: Any) -> StepPlan:
        """After a guarded model call: a failure (absorbed into its result key) ends THIS turn in TURN_FAILED;
        success continues where the call was going."""
        gate = _ns(run, "_visit_model_gate")
        after = str(gate.get("after") or "")
        result = _get_path(run.vars, str(gate.get("result_key") or "")) if gate.get("result_key") else None
        if isinstance(result, dict) and result.get("absorbed_failure") and result.get("ok") is False:
            _ns(run, "_turn")["model_error"] = str(result.get("absorbed_failure") or "the model call failed")
            gate.clear()
            return StepPlan(node_id=MODEL_GATE_NODE, next_node=TURN_FAILED_NODE)
        gate.clear()
        return StepPlan(node_id=MODEL_GATE_NODE, next_node=after or "PARK")

    def turn_failed_node(run: RunState, ctx: Any) -> StepPlan:
        """THIS turn could not be answered: answer the error for it (level error) and park - the visit continues
        (Codex review 2026-11-09 #5). What the turn already did STAYS: its completed tool results and any operator
        guidance remain in the transcript, an explicit failed-turn outcome closes the turn there (so the next turn
        never stacks two user messages), the visit history folds the turn with that outcome, and the answer reports
        the tools that really ran. Nothing is formed for the failed turn."""
        visit = _ns(run, "_visit")
        turn = _ns(run, "_turn")
        err = str(turn.get("model_error") or "the model call failed")
        outcome = f"[this turn could not be answered: {err}]"
        captures = ((run.vars.get("_temp") or {}).get("turn_captures") or {}) if isinstance(run.vars.get("_temp"), dict) else {}
        tools_ran = [str(t) for t in (captures.get("tools_ran") or []) if str(t or "").strip()]
        turn["tools_ran"] = tools_ran
        context_ns = run.vars.get("context")
        if isinstance(context_ns, dict):
            msgs = context_ns.get("messages")
            if isinstance(msgs, list) and msgs and _ns(run, "_runtime").get("history_window_turn_start") is not None:
                msgs.append({"role": "assistant", "content": outcome,
                             "metadata": {"kind": "turn_failed", "turn_id": turn.get("turn_id"), "error": err}})
            context_ns.pop("attachments", None)
        if visit.get("last_folded_turn") != turn.get("turn_id"):
            history = list(visit.get("history") or [])
            user_entry: Dict[str, Any] = {"role": "user", "content": turn.get("rendered_user") or turn.get("text") or ""}
            if turn.get("attachments"):
                user_entry["attachments"] = list(turn["attachments"])
            history.append(user_entry)
            history.append({"role": "assistant", "content": outcome})
            visit["history"] = history
            visit["last_folded_turn"] = turn.get("turn_id")
        return StepPlan(
            node_id=TURN_FAILED_NODE,
            effect=Effect(
                type=EffectType.ANSWER_USER,
                payload={
                    "message": f"This turn could not be answered: {err}",
                    "level": "error",
                    "turn_id": turn.get("turn_id"),
                    "turn_error": err,
                    "tools_ran": tools_ran,
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
        # The prompt lists the sheet folded visibly (an index beside the whole
        # history); the marker renderer in the apply node reads the whole lines.
        prompt_sheet_lines = [
            f"{i}. {sheet_prompt_line(desc)}" for i, (_rid, desc) in enumerate(sheet, start=1)
        ]
        prompt = build_reflection_prompt(prompt_sheet_lines)
        session_graph_ids = [rid for rid, _ in sheet if rid][-4:]
        return StepPlan(
            node_id="REFLECT",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "messages": _windowed_history(run, visit) + [{"role": "user", "content": prompt}],
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
        # AUTHORED AT FORMATION (operator 2026-11-09): a record of the close whose title / summary is not the
        # entity's own gets ONE capability=summaries LLM_CALL before it forms (`_reflect.authored.<key>`); a failed
        # call forms it pending and its failure line is written here, at the next APPLY, once the record has its id.
        authored_ns = refl.setdefault("authored", {})
        failed = refl.pop("authoring_failure_pending", None)
        if isinstance(failed, dict):
            _log_visit_authoring_failure(home, visit, failed, refl.get(str(failed.get("result_key") or "")))

        def _author_first(key: str, kind: str, body: str) -> Optional[StepPlan]:
            if key in authored_ns:
                return None
            authored_ns[key] = None  # requested; the call's result lands here
            return _authoring_step("APPLY", kind, body, result_key=f"_reflect.authored.{key}", next_node="APPLY")

        def _fields_for(key: str, *, kind: str, need: List[str], raw: str, result_key: str, title: Optional[str] = None,
                        digest: str = "", block_reason: Optional[str] = None) -> Dict[str, Any]:
            fields, failure = _authored_fields(authored_ns.get(key), need=need, run_id=str(run.run_id),
                                               title=title, digest=digest)
            if failure:
                refl["authoring_failure_pending"] = {
                    **failure, "kind": kind, "raw_block": raw, "template_id": "visit_workflow.close",
                    "result_key": result_key, **({"block_reason": block_reason} if block_reason else {}),
                }
            return fields

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
                        {"text": _re.text, "touches": _re.touches, "title": _re.title, "gids": _gids, "title_reason": _re.title_reason, "raw": _re.raw}
                    )
                else:
                    r_notes.append(
                        "#FALLBACK realization refused (no evidence resolved against "
                        f'this visit\'s sheet): "{_re.text}"'
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
            plan = _author_first("summary", "summary", str(refl["marked_reply"]))
            if plan is not None:
                return plan
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
            _sum = _fields_for("summary", kind="summary", need=["title", "summary"], raw=str(refl["marked_reply"]),
                               result_key="summary_out", digest=refl_digest)
            _sum_authored = _sum.pop("_metadata_authored", None)
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "summary",
                            "title": _sum.get("title"),
                            **({"pending": _sum["pending"]} if _sum.get("pending") else {}),
                            "digest": _sum["digest"],
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
                                **({"digest_method": "mechanical-floor-v1"}
                                   if refl_floored and not _sum_authored else {}),
                                **({"visit_id": str(visit_id)} if visit_id else {}),
                                **({"metadata_authored": _sum_authored} if _sum_authored else {}),
                                # An authored summary says so (digest_method authored:summaries).
                                **(authored_digest_attributes({}, _sum_authored.get("fields"))
                                   if _sum_authored else {}),
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
            _t, _w, _why = parse_elected_title(str(interests[i]))
            if _why:
                plan = _author_first(f"interest-{i}", "interest", _w)
                if plan is not None:
                    return plan
            refl["i"] = i + 1
            session_record_id = next(
                iter((refl.get("summary_out") or {}).get("record_ids") or []), None
            )
            # Title and words WHOLE (operator 2026-11-08; root ADR 0054): his `title:` first line; without one, no
            # title and a recorded failure (operator 2026-11-09: no fallback). A plain pending string reads too.
            _tf = _fields_for(f"interest-{i}", kind="interest", need=["title"], raw=str(interests[i]),
                              result_key="interest_out", title=_t, digest=_w, block_reason=_why) if _why else {"title": _t}
            _t_authored = _tf.pop("_metadata_authored", None)
            # No usable `title:` line: the record forms with its title PENDING (memory queues title.author) and the
            # authoring drainer writes it with the summaries capability model - never a guess from his words.
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "interest",
                            "title": _tf.get("title"),
                            **({"pending": _tf["pending"]} if _tf.get("pending") else {}),
                            "digest": _w,
                            "keywords": [],
                            "edges": (
                                [["from_session", str(session_record_id)]] if session_record_id else []
                            ),
                            "attributes": {"session_id": str(run.session_id or ""),
                                           **({"metadata_authored": _t_authored} if _t_authored else {})},
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
            _t, _w, _why = parse_elected_title(str(lessons[i]))
            if _why:
                plan = _author_first(f"lesson-{i}", "lesson", _w)
                if plan is not None:
                    return plan
            refl["i"] = i + 1
            session_record_id = next(
                iter((refl.get("summary_out") or {}).get("record_ids") or []), None
            )
            _tf = _fields_for(f"lesson-{i}", kind="lesson", need=["title"], raw=str(lessons[i]),
                              result_key="lesson_out", title=_t, digest=_w, block_reason=_why) if _why else {"title": _t}
            _t_authored = _tf.pop("_metadata_authored", None)
            # No usable `title:` line: the record forms with its title PENDING (memory queues title.author) and the
            # authoring drainer writes it with the summaries capability model - never a guess from his words.
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "lesson",
                            "title": _tf.get("title"),
                            **({"pending": _tf["pending"]} if _tf.get("pending") else {}),
                            "digest": _w,
                            "keywords": [],
                            "edges": (
                                [["from_session", str(session_record_id)]] if session_record_id else []
                            ),
                            "attributes": {
                                "session_id": str(run.session_id or ""),
                                "phase": "visit",
                                **({"metadata_authored": _t_authored} if _t_authored else {}),
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
            r = realizations[i] if isinstance(realizations[i], dict) else {}
            r_text = str(r.get("text") or "")
            _pre_title = str(r.get("title") or "").strip() or None
            _pre_why = r.get("title_reason")
            if _pre_title is None and not _pre_why and "title_reason" not in r:
                _pre_title, _unused_w, _pre_why = parse_elected_title(r_text)
            if _pre_why:
                plan = _author_first(f"realize-{i}", "realization", r_text)
                if plan is not None:
                    return plan
            refl["i"] = i + 1
            # His `title:` first line (parsed at election); without one, no title and a recorded failure (operator
            # 2026-11-09: no fallback). A pending election of an older run carries no reason: it is re-read here.
            _r_title = str(r.get("title") or "").strip() or None
            _r_why = r.get("title_reason")
            if _r_title is None and not _r_why and "title_reason" not in r:
                _r_title, _unused, _r_why = parse_elected_title(r_text)
            # No usable `title:` line: authored by the capability call above; pending only if it failed.
            _rf = _fields_for(f"realize-{i}", kind="realization", need=["title"], raw=str(r.get("raw") or r_text),
                              result_key="realize_out", title=_r_title, digest=r_text,
                              block_reason=str(_r_why)) if _r_why else {"title": _r_title}
            _r_authored = _rf.pop("_metadata_authored", None)
            r_attrs: Dict[str, Any] = {
                "session_id": str(run.session_id or ""),
                "phase": "visit",
            }
            if r.get("touches"):
                r_attrs["touches"] = str(r["touches"])
            if _r_authored:
                r_attrs["metadata_authored"] = _r_authored
            return StepPlan(
                node_id="APPLY",
                effect=Effect(
                    type=EffectType.MEMORY_FORM,
                    payload={
                        "records": [{
                            "kind": "realization",
                            "title": _rf.get("title"),
                            **({"pending": _rf["pending"]} if _rf.get("pending") else {}),
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
        AUTHOR_EPISODE_NODE: author_episode_node,
        MODEL_GATE_NODE: model_gate_node,
        TURN_FAILED_NODE: turn_failed_node,
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
        actor_reason = nodes[str(react_middle.entry)]

        def guidance_boundary(run: RunState, ctx: Any) -> StepPlan:
            pending = _ns(run, "_runtime").get("visit_guidance_pending")
            if not isinstance(pending, list) or not pending:
                return actor_reason(run, ctx)
            # Pop into a durable effect stage; the queue watermark already
            # belongs to this tick, never to a concurrent host writer.
            stage = _ns(run, "_visit_guidance")
            stage.clear()
            stage["messages"] = list(pending)
            _ns(run, "_runtime")["visit_guidance_pending"] = []
            turn = _ns(run, "_turn")
            fresh = [m for m in pending if m.get("turn_id") == turn.get("turn_id")]
            stage["fresh"] = fresh
            if not fresh:
                return StepPlan(node_id=str(react_middle.entry), effect=Effect(
                    type=EffectType.EMIT_EVENT, payload={"name": "abstract.visit_guidance_retained",
                    "payload": {"messages": pending, "reason": "turn_ended"}}),
                    next_node=str(react_middle.entry))
            return StepPlan(node_id=str(react_middle.entry), effect=Effect(
                type=EffectType.MEMORY_RECALL, payload={
                    "cue_text": "\n\n".join(str(m["content"]) for m in fresh),
                    "scopes": ladder, "view": "working_set", "participants": list(_ns(run, "_visit")["participants"]),
                    "budget": dict(_ns(run, "_visit")["budget"]),
                    "turn_id": f"{turn['turn_id']}:guidance:{fresh[-1]['seq']}",
                    "channel": "workplace",
                }, result_key="_visit_guidance.recall"), next_node="VISIT_GUIDANCE_RENDER")

        def render_guidance(run: RunState, ctx: Any) -> StepPlan:
            stage, turn, visit = _ns(run, "_visit_guidance"), _ns(run, "_turn"), _ns(run, "_visit")
            fresh = list(stage.get("fresh") or [])
            recall = stage.get("recall") or {}
            # Merge by stable memory identity, favoring this fresh recall.
            handles = list(recall.get("handles") or []) + list(turn.get("displayed") or [])
            merged, seen = [], set()
            for handle in handles:
                key = handle.get("record_id")
                if key and key not in seen:
                    seen.add(key)
                    merged.append(handle)
            displayed = [h for h in _serialize_under_budget(merged, int(visit["budget"].get("token_budget") or 0))
                         if h.get("admission") != "self"]
            turn["displayed"] = displayed
            block = _memories_block(displayed, recall.get("as_of_seq"))
            parts = [f"{m['speaker']} (operator-visit, correction):\n{m['content']}" for m in fresh]
            raw = "\n\n".join(parts)
            content = raw + ("\n\n" + block if block else "")
            _ns(run, "context").setdefault("messages", []).append({
                "role": "user", "content": content,
                "metadata": {"kind": "visit_guidance", "channel": "operator-visit",
                             "seqs": [m["seq"] for m in fresh]},
            })
            # Formation and next-open reconstruction retain the attributed raw
            # words, not just a transient instruction to the actor.
            turn["text"] = str(turn.get("text") or "") + "\n\n" + raw
            turn["rendered_user"] = str(turn.get("rendered_user") or "") + "\n\n" + content
            runtime_ns = _ns(run, "_runtime")
            runtime_ns["last_visit_guidance"] = {"status": "delivered", "seqs": [m["seq"] for m in fresh]}
            runtime_ns["llm_payload_extras"] = {
                "anchor_record_ids": [h.get("record_id") for h in displayed],
                "anchor_graph_ids": [str((h.get("provenance") or {}).get("record_id") or "") for h in displayed
                                     if (h.get("provenance") or {}).get("record_id")],
            }
            retained = [m for m in stage.get("messages", []) if m not in fresh]
            stage.clear()
            return StepPlan(node_id="VISIT_GUIDANCE_RENDER", effect=Effect(type=EffectType.EMIT_EVENT,
                payload={"name": "abstract.visit_guidance_delivered", "payload": {
                    "messages": fresh, "retained_messages": retained, "channel": "operator-visit",
                    "memory_ids": [h.get("record_id") for h in displayed]}}),
                next_node=str(react_middle.entry))

        nodes[str(react_middle.entry)] = guidance_boundary
        # Every model call the middle plans is guarded (a visit never fails on one failed call).
        for _mid in list(react_middle.nodes.keys()):
            nodes[_mid] = _guarded_node(nodes[_mid])
        nodes["VISIT_GUIDANCE_RENDER"] = render_guidance
        nodes["REASON"] = bridge_node  # RENDER routes in unchanged
        nodes[HARVEST_NODE] = harvest_node  # the middle's declared exit
    return WorkflowSpec(
        workflow_id=VISIT_WORKFLOW_ID,
        entry_node="OPEN",
        nodes=nodes,
    )
