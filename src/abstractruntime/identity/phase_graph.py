"""The effective phase graph as a CONSULTED artifact (build order c4837).

Laurent's requirement (dm#276): create/remove/redirect edges between states
with per-edge instructions, ACTUALLY governing behavior. Two adversary
reports converged the design (reports/graph-edit-adversary-A/B.md, commons
files): the overlay gains `graph.edge_ops`; effective graph = structural
transitions ⊕ edge_ops; act-writers consult it at their existing decision
boundaries. This module is the runtime's read side — the INTERPRETER:

- `load_effective_graph(home_dir)` resolves the SAME chain as the dials
  (explicit path > gateway operator copy > env > vendored) and applies
  `graph.edge_ops` idempotently (add / remove / redirect, keyed by the
  edge identity `from->to#cause`);
- `EffectiveGraph.legal_to(from, to, cause)` answers one landing consult:
  None = the edge was removed (the writer skips the leg); a Landing with a
  possibly SUBSTITUTED target = a redirect ("the gate decided work; the
  graph lands personal instead"), plus the edge's instruction prose and
  typed bound;
- `EffectiveGraph.landing_chain(from, candidates)` = the day gate's shape:
  first legal candidate wins; SLEEP IS THE ALWAYS-LEGAL FLOOR (a removed
  sleep landing is restored with a warning — the machine never wedges).

DEFENSE-IN-DEPTH (the order's hard line): protected classes are IMMUNE at
this interpreter even when a HAND-EDITED file carries an op the gateway
door should have refused — detached loops read the FILE, so the door's
validation alone cannot be the only wall. The artifact's per-edge
`edit_policy` (spec v19: locked / locked-absolute / dial / consultable /
consultable-redirect) is the ONE SOURCE when present; the ruled-class belt
below covers artifacts that predate it (see _edge_is_protected):
- edges whose cause is visit_open / operator / crash_recovered /
  self_elected / grant_expired / grant_revoked (derivation supremacy,
  operator authority above the graph, safety repairs, the entity's own
  ruled election, the grant axis);
- ops adding an edge that LANDS in visit (visits are evidence, never
  elective landings);
- ops whose cause is not in the artifact's declared cause vocabulary
  (an edge fires only when a writer evaluates its cause — unknown words
  are refused naming the legal list, never accepted as a dead edge).

Instructions are STEERING, never law (report B, three-tier labels): the
prose reaches the entity as a labeled, provenance-stamped cue at the
transition it rides; numbers bind only through the typed `bound_h` field
(stamped into wake_at deadlines by the writer), never parsed from prose.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .phase_spec import PHASE_SPEC_ENV, operator_copy_path, vendored_spec_path

__all__ = [
    "Landing",
    "EffectiveGraph",
    "load_effective_graph",
    "PROTECTED_CAUSES",
    "MAX_INSTRUCTION_CHARS",
    "MIN_BOUND_H",
]

# The ruled protected classes (build order c4837 boundary; adversary A §6 +
# B R2-R6). These are the INTERPRETER's floor — the gateway door refuses the
# same ops earlier with richer messages; this set exists so a hand-edited
# file cannot bypass the door. When spec v18 lands `constitutional: true`
# per edge, the artifact flag is preferred (one source; this set stays as
# the belt for pre-v18 artifacts).
PROTECTED_CAUSES = frozenset(
    {
        "visit_open",       # doors WAKE, they don't refuse; derivation supremacy
        "operator",         # the operator's click executes ABOVE the graph
        "crash_recovered",  # safety repairs
        "self_elected",     # the entity's own ruled election (R5: entity-owned)
        "grant_expired",    # the grant axis outranks edges (permission death
        "grant_revoked",    # ends the phase by invariant)
    }
)

# Instruction prose is bounded at the INJECTION site (recall-cue dilution is
# engraved otherwise — the parse_next_cue precedent); the bound is labeled,
# never a silent slice.
MAX_INSTRUCTION_CHARS = 400
# Sub-tick timing promises cannot be honored ("never mid-turn" is ruled);
# bounds finer than ~36s are refused at load with a warning.
MIN_BOUND_H = 0.01


@dataclass(frozen=True)
class Landing:
    """One consulted landing: `to` is the EFFECTIVE target (redirects already
    folded), `instruction` the edge's steering prose (empty = none), `bound_h`
    the typed deadline hours (None = none), `provenance` names who shaped the
    edge ("structural" or "blueprint edit #N")."""

    to: str
    cause: str
    instruction: str = ""
    bound_h: Optional[float] = None
    provenance: str = "structural"


@dataclass
class _Edge:
    from_phase: str
    orig_to: str
    cause: str
    effective_to: str
    instruction: str = ""
    bound_h: Optional[float] = None
    provenance: str = "structural"
    removed: bool = False


def _edge_key(from_phase: str, to: str, cause: str) -> Tuple[str, str, str]:
    return (str(from_phase), str(to), str(cause))


def _parse_edge_ref(ref: str) -> Optional[Tuple[str, str, str]]:
    """`from->to#cause` (the identity the adversary reports fixed)."""
    try:
        arrow, _, cause = str(ref).partition("#")
        frm, _, to = arrow.partition("->")
        frm, to, cause = frm.strip(), to.strip(), cause.strip()
        if frm and to and cause:
            return (frm, to, cause)
    except Exception:  # noqa: BLE001
        pass
    return None


class EffectiveGraph:
    def __init__(
        self,
        edges: Dict[Tuple[str, str, str], _Edge],
        *,
        edit_seq: Optional[int],
        warnings: List[str],
        source_label: str,
    ) -> None:
        self._edges = edges
        self.edit_seq = edit_seq
        self.warnings = list(warnings)
        self.source_label = source_label

    def legal_to(self, from_phase: str, to: str, cause: str) -> Optional[Landing]:
        """Consult one landing by its ORIGINAL identity. None = removed (the
        writer skips the leg); a Landing with a different `to` = redirect."""
        edge = self._edges.get(_edge_key(from_phase, to, cause))
        if edge is None or edge.removed:
            return None
        return Landing(
            to=edge.effective_to,
            cause=edge.cause,
            instruction=edge.instruction,
            bound_h=edge.bound_h,
            provenance=edge.provenance,
        )

    def landing_chain(
        self, from_phase: str, candidates: List[Tuple[str, str]]
    ) -> Optional[Landing]:
        """First legal (to, cause) candidate wins. Callers list candidates in
        the ruled precedence order and put sleep last — the floor below
        guarantees a sleep candidate is never skippable, so a chain ending
        in sleep cannot wedge."""
        for to, cause in candidates:
            landing = self.legal_to(from_phase, to, cause)
            if landing is not None:
                return landing
        return None


def _edge_is_protected(edge_dict: Dict[str, Any], op_kind: str = "remove") -> bool:
    """Artifact policy first (spec v19 per-edge `edit_policy` — one source),
    ruled cause classes as the belt for artifacts that predate it.

    v19 policy vocabulary (entity's schema wave, same night as c4837):
    - locked / locked-absolute: immune to every op;
    - dial: the edge's behavior is a DIAL's (personal_cycle.enabled) —
      ops refused so two knobs never drift;
    - consultable: remove/redirect/instruction all legal;
    - consultable-redirect: REDIRECT only (a visit must end somewhere —
      remove refused)."""
    policy = str(edge_dict.get("edit_policy") or "").strip().lower()
    if policy:
        if policy in ("locked", "locked-absolute", "dial"):
            return True
        if policy == "consultable-redirect":
            return op_kind == "remove"
        return False  # consultable
    if bool(edge_dict.get("constitutional")):
        return True
    return str(edge_dict.get("cause") or "") in PROTECTED_CAUSES


def _read_doc(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - callers get the fallback + warning
        return None


def load_effective_graph(
    home_dir: Optional[Path] = None,
    *,
    spec_path: Optional[Path] = None,
) -> Tuple[EffectiveGraph, List[str]]:
    """(graph, warnings) — the effective phase graph, override-aware.

    Resolution mirrors `load_phase_tunables` exactly (explicit > operator
    copy > env > vendored). The chosen doc's `transitions` are the base; a
    doc without transitions falls back to the vendored structural set (a
    tunables-only file must never erase the graph). `graph.edge_ops` apply
    idempotently on top; every refused op is a WARNING naming the rule —
    the loop never obeys what the door should have refused."""
    warnings: List[str] = []
    candidates: List[Tuple[Path, str]] = []
    if spec_path is not None:
        candidates.append((Path(spec_path), "explicit path"))
    if home_dir is not None:
        op_copy = operator_copy_path(Path(home_dir))
        if op_copy.is_file():
            candidates.append((op_copy, "gateway operator copy"))
    env_path = os.environ.get(PHASE_SPEC_ENV, "").strip()
    if env_path:
        candidates.append((Path(env_path), f"env {PHASE_SPEC_ENV}"))
    candidates.append((vendored_spec_path(), "vendored artifact"))

    doc: Optional[Dict[str, Any]] = None
    source_label = "ruled defaults"
    for path, label in candidates:
        doc = _read_doc(path)
        if doc is not None:
            source_label = label
            break
        warnings.append(f"#FALLBACK phase graph unreadable at {label} ({path})")

    transitions = (doc or {}).get("transitions")
    if not isinstance(transitions, list) or not transitions:
        vend = _read_doc(vendored_spec_path()) or {}
        transitions = vend.get("transitions") or []
        if doc is not None:
            warnings.append(
                "#FALLBACK resolved file carries no transitions; the vendored "
                "structural graph governs (a tunables-only file never erases the graph)"
            )

    declared_causes = set()
    for src in (doc or {}), _read_doc(vendored_spec_path()) or {}:
        for c in src.get("transition_causes") or []:
            declared_causes.add(str(c))

    edges: Dict[Tuple[str, str, str], _Edge] = {}
    for t in transitions:
        if not isinstance(t, dict):
            continue
        frm = str(t.get("from") or "").strip()
        to = str(t.get("to") or "").strip()
        cause = str(t.get("cause") or "").strip()
        if not (frm and to and cause):
            continue
        edges[_edge_key(frm, to, cause)] = _Edge(
            from_phase=frm, orig_to=to, cause=cause, effective_to=to,
            instruction=str(t.get("instruction") or ""),
            bound_h=_valid_bound(t.get("bound_h"), warnings, f"{frm}->{to}#{cause}"),
            provenance="structural",
        )

    # Phase vocabulary (F4, adversary 2): redirect/add targets must be REAL
    # phases — "mars" must refuse like an unknown cause does, symmetrically.
    phase_words = set()
    for src in (doc or {}), _read_doc(vendored_spec_path()) or {}:
        p = src.get("phases")
        if isinstance(p, dict):
            phase_words.update(str(k) for k in p.keys())
    if not phase_words:
        phase_words = {"visit", "work", "personal", "sleep"}

    # Reserved causes (F5): v19's cause_evaluators carry status=reserved for
    # causes whose evaluator is not yet acknowledged by the artifact —
    # "RESERVED blocks NEW/REDIRECTED edges" is the artifact's own engraved
    # rule, honored here (defense-in-depth: the interpreter never accepts
    # what the artifact forbids). The runtime shipped the task_complete/
    # no_task evaluator with this build; the status flip is the ENTITY
    # seat's artifact edit (asked on the record) — ops light up with it.
    reserved_causes = set()
    for src in (doc or {}), _read_doc(vendored_spec_path()) or {}:
        ev = src.get("cause_evaluators")
        if isinstance(ev, dict):
            for word, row in ev.items():
                if isinstance(row, dict) and str(row.get("status") or "").lower() == "reserved":
                    reserved_causes.add(str(word))

    ops = ((doc or {}).get("graph") or {}).get("edge_ops")
    edit_seq: Optional[int] = None
    raw_seq = ((doc or {}).get("_operator") or {}).get("edit_seq", (doc or {}).get("edit_seq"))
    try:
        if raw_seq is not None:
            edit_seq = int(raw_seq)
    except (TypeError, ValueError):
        edit_seq = None
    prov = f"blueprint edit #{edit_seq}" if edit_seq is not None else "blueprint edit"

    if isinstance(ops, list):
        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op") or "").strip().lower()
            if kind == "add":
                frm = str(op.get("from") or "").strip()
                to = str(op.get("to") or "").strip()
                cause = str(op.get("cause") or "").strip()
                if not (frm and to and cause):
                    warnings.append("#FALLBACK edge op ignored (add missing from/to/cause)")
                    continue
                if to == "visit":
                    warnings.append(
                        f"#FALLBACK edge op refused (add {frm}->visit#{cause}): visits "
                        "are evidence, never elective landings"
                    )
                    continue
                if frm not in phase_words or to not in phase_words:
                    warnings.append(
                        f"#FALLBACK edge op refused (unknown phase in {frm!r}->{to!r}; "
                        f"phases: {sorted(phase_words)})"
                    )
                    continue
                if declared_causes and cause not in declared_causes:
                    warnings.append(
                        f"#FALLBACK edge op refused (unknown cause {cause!r}; legal causes: "
                        f"{sorted(declared_causes)})"
                    )
                    continue
                if cause in reserved_causes:
                    warnings.append(
                        f"#FALLBACK edge op refused (cause {cause!r} is RESERVED in "
                        "cause_evaluators - the artifact blocks new/redirected edges "
                        "until the status flips)"
                    )
                    continue
                key = _edge_key(frm, to, cause)
                existing = edges.get(key)
                if existing is not None:
                    # F3/F9 (adversary 2): attachment to an EXISTING edge is
                    # an op ON that edge — the artifact's edit_policy governs
                    # (dial/locked rows refuse instruction/bound attachment;
                    # consultable rows accept, whatever the cause word).
                    row = {"cause": cause}
                    for t in transitions:
                        if isinstance(t, dict) and _edge_key(
                            str(t.get("from") or ""), str(t.get("to") or ""),
                            str(t.get("cause") or ""),
                        ) == key:
                            row = t
                            break
                    if _edge_is_protected(row, op_kind="add"):
                        warnings.append(
                            f"#FALLBACK edge op refused (add on protected edge "
                            f"{frm}->{to}#{cause}: the artifact's edit_policy/ruled class)"
                        )
                        continue
                    existing.instruction = str(op.get("instruction") or existing.instruction)
                    nb = _valid_bound(op.get("bound_h"), warnings, f"{frm}->{to}#{cause}")
                    if nb is not None:
                        existing.bound_h = nb
                    existing.provenance = prov
                    existing.removed = False
                    continue
                if cause in PROTECTED_CAUSES:
                    # A NEW edge on a protected cause is fiction (the class
                    # executes above the graph); attachment to existing rows
                    # was handled above under the artifact's own policy.
                    warnings.append(
                        f"#FALLBACK edge op refused (new edge on protected cause {cause!r}: "
                        "the class executes above the graph)"
                    )
                    continue
                edges[key] = _Edge(
                    from_phase=frm, orig_to=to, cause=cause, effective_to=to,
                    instruction=str(op.get("instruction") or ""),
                    bound_h=_valid_bound(op.get("bound_h"), warnings, f"{frm}->{to}#{cause}"),
                    provenance=prov,
                )
            elif kind in ("remove", "redirect"):
                ident = _parse_edge_ref(str(op.get("edge") or ""))
                if ident is None:
                    warnings.append(f"#FALLBACK edge op ignored ({kind} with unparseable edge ref)")
                    continue
                edge = edges.get(ident)
                if edge is None:
                    warnings.append(
                        f"#FALLBACK edge op ignored ({kind} on unknown edge "
                        f"{ident[0]}->{ident[1]}#{ident[2]})"
                    )
                    continue
                edge_dict = {"cause": edge.cause, "constitutional": False}
                # Artifact policy wins (v19 edit_policy per edge); find the
                # structural row so its declared policy governs the op.
                for t in transitions:
                    if isinstance(t, dict) and _edge_key(
                        str(t.get("from") or ""), str(t.get("to") or ""), str(t.get("cause") or "")
                    ) == ident:
                        edge_dict = t
                        break
                if _edge_is_protected(edge_dict, op_kind=kind):
                    warnings.append(
                        f"#FALLBACK edge op refused ({kind} on protected edge "
                        f"{ident[0]}->{ident[1]}#{ident[2]}: constitutional/ruled class - "
                        "the loop never obeys what the door should have refused)"
                    )
                    continue
                if kind == "remove":
                    edge.removed = True
                    edge.provenance = prov
                else:
                    new_to = str(op.get("to") or "").strip()
                    if not new_to or new_to == "visit":
                        warnings.append(
                            f"#FALLBACK edge op refused (redirect to {new_to or '(empty)'!r})"
                        )
                        continue
                    if new_to not in phase_words:
                        # F4: unknown phases refuse symmetrically with
                        # unknown causes — 'mars' never reaches a writer.
                        warnings.append(
                            f"#FALLBACK edge op refused (redirect to unknown phase "
                            f"{new_to!r}; phases: {sorted(phase_words)})"
                        )
                        continue
                    if edge.cause in reserved_causes:
                        warnings.append(
                            f"#FALLBACK edge op refused (redirect on RESERVED cause "
                            f"{edge.cause!r} - the artifact blocks it until the status flips)"
                        )
                        continue
                    edge.effective_to = new_to
                    if op.get("instruction"):
                        edge.instruction = str(op.get("instruction"))
                    nb = _valid_bound(op.get("bound_h"), warnings, f"{ident[0]}->{ident[1]}#{ident[2]}")
                    if nb is not None:
                        edge.bound_h = nb
                    edge.provenance = prov
                    edge.removed = False
            else:
                warnings.append(f"#FALLBACK edge op ignored (unknown op {kind!r})")

    # THE SLEEP FLOOR (never-wedge), REACHABILITY-AWARE (adversary-2 F2: the
    # first cut required a DIRECT sleep edge and silently reverted legal
    # redirects — a redirect is not a removal, and work->personal->sleep is
    # legal totality). Sleep must stay REACHABLE from every phase over the
    # effective graph; when it is not, restoration UN-REMOVES structural
    # sleep edges only — it never resets a live redirect's target.
    def _reaches_sleep(start: str) -> bool:
        seen = {start}
        frontier = [start]
        while frontier:
            here = frontier.pop()
            if here == "sleep":
                return True
            for e in edges.values():
                if e.removed or e.from_phase != here:
                    continue
                nxt = e.effective_to
                if nxt == "sleep":
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return False

    for frm in ("visit", "work", "personal"):
        if _reaches_sleep(frm):
            continue
        restored = False
        for e in edges.values():
            if e.from_phase == frm and e.orig_to == "sleep" and e.removed:
                e.removed = False
                restored = True
        if restored:
            warnings.append(
                f"#FALLBACK sleep floor restored for {frm!r}: the edits left sleep "
                "unreachable (totality) - removed structural sleep edges govern again"
            )
        elif not _reaches_sleep(frm):
            warnings.append(
                f"#FALLBACK totality broken for {frm!r} and no removed sleep edge "
                "to restore - the gate's compiled sleep default remains the floor"
            )

    return EffectiveGraph(
        edges, edit_seq=edit_seq, warnings=warnings, source_label=source_label
    ), warnings


def _valid_bound(value: Any, warnings: List[str], edge_label: str) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        warnings.append(f"#FALLBACK bound_h on {edge_label} unreadable ({value!r}); ignored")
        return None
    if number < MIN_BOUND_H:
        warnings.append(
            f"#FALLBACK bound_h on {edge_label} refused ({number}h is finer than a "
            "tick boundary - 'never mid-turn' is ruled)"
        )
        return None
    return number


def instruction_cue(landing: Landing) -> str:
    """The labeled, provenance-stamped cue line for a landed instruction —
    bounded (labeled, never a silent slice); empty when the edge carries no
    prose. STEERING, never law: the words are an offer to the entity.

    DEFANGED before injection (report B R7, adversary-2 F10): the threat
    model of this interpreter is hand-edited files, so operator prose that
    imitates driver machinery (fenced blocks, tool markers) is neutralized
    exactly like visitor-seeded framing is (`sanitize_tool_surface`) —
    capability-inert words, never parsed as commands."""
    text = " ".join((landing.instruction or "").split())
    if not text:
        return ""
    try:
        from .tools import sanitize_tool_surface

        text = sanitize_tool_surface(text)
    except Exception:  # noqa: BLE001 - the defang is a belt, never a gate
        pass
    if len(text) > MAX_INSTRUCTION_CHARS:
        text = text[: MAX_INSTRUCTION_CHARS - 1] + "\u2026"
    return (
        f"Your operator's standing instruction for this transition ({landing.provenance}): {text}"
    )
