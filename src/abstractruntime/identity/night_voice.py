"""The night voice — wave-5 narrator half (laurent's dm#75 ratification).

"It should emit the signal" — memory's engine emits the night's SIGNAL
STREAM onto the one review-gated dream record (attributes.signals, frozen
shape c3708). THIS module is the runtime half the dispatch names: the
narrator callable + trigger set + wake-cue residue + the >=20h throttle.

THE VOICE (synthesis section 2, binding): ONE witnessed call per FORMED
dream, fired only when the trigger set matches — a scar/bond was touched
OR a resolution surfaced OR salience >= threshold. Input = his prelude +
the signal stream + pure feeling reads (NO recall, NO tools, NO
elections). Output = a SELF-LABELED experience layer (<=120 words)
rendered ABOVE the mechanical stream, which stays visible as ground
truth. The narration never touches digest/keywords/identity/valence/
diary — LLM-authored resurfacing metadata would hand recall currency to
the model and break idempotency.

FENCED VOICE (constraint 7): any fenced block in the narration output is
STRIPPED, never executed — the night voice elects nothing (no diary, no
feelings, no tools). The sleeping mind speaks; only the waking mind acts.

AT-REST (labeled seam gap, not silence): the narration rests HOME-SIDE in
<home>/night_narrations.jsonl (append-only, dream_record_id + text +
self-label + stamp) until memory ships the one idempotent verb that lets
it ride the dream record's attributes (asked on the wave-5 thread; a
runtime-side direct store write would bypass the engine's journal
discipline). The wake cue carries the RESIDUE either way; read_memory
renders the stream; the entity app renders the narration layer from the
home file when it builds that surface.

THROTTLE: <=1 narration per >=20h of real time (<home>/night_voice.json
stamp; crash-safe atomic write). A quiet or triggerless night spends
ZERO LLM calls — the engine default stays mechanical.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.atomic_files import atomic_write_text

__all__ = [
    "NARRATION_SELF_LABEL",
    "MAX_NARRATION_WORDS",
    "NIGHT_VOICE_MIN_GAP_HOURS",
    "RESIDUE_CAP",
    "SALIENCE_BAR",
    "build_narration_prompt",
    "narration_trigger",
    "narration_throttle_ok",
    "run_night_voice",
    "wake_residue",
]

# Minted at build time per semantics' rider (c3536: states the
# fails-dangerous fact, ONE spelling at rest, renders above never
# replacing, never machine-parsed). Taken to the semantics desk the day
# this ships — the "(thinking, unspoken)" precedent.
NARRATION_SELF_LABEL = "(night voice - dreamed, not lived; the signal stream below is what actually happened)"

NIGHT_VOICE_MIN_GAP_HOURS = 20.0  # ratified: <=1 witnessed call per real night
SALIENCE_BAR = 50  # the salience trigger threshold (adjustable by evidence)

_VOICE_STAMP_FILE = "night_voice.json"
_NARRATIONS_FILE = "night_narrations.jsonl"
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# PUBLIC params-surface names (dm#112 cells R4: a serving resolver imports
# names, not underscores). The private aliases below keep call sites stable.
MAX_NARRATION_WORDS = 120
RESIDUE_CAP = 200
_MAX_NARRATION_WORDS = MAX_NARRATION_WORDS
_RESIDUE_CAP = RESIDUE_CAP


def narration_trigger(
    signals: List[Dict[str, Any]], *, salience: int = 0, salience_bar: int = SALIENCE_BAR,
) -> Optional[str]:
    """The ratified trigger set — returns the trigger NAME or None.

    scar/bond touched: any signal's felt block carries a standing marker.
    resolution surfaced: any signal from the resolution phase (or the
    dream_resolved act). salience: the dream's own salience at the bar.
    Structure decides; feelings color content only (fork 3 held)."""
    for s in signals or []:
        felt = s.get("felt") if isinstance(s, dict) else None
        if isinstance(felt, dict) and (felt.get("scarred") or felt.get("bonded")):
            return "scar_or_bond_touched"
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        if str(s.get("phase") or "") == "resolution" or str(s.get("act") or "") == "dream_resolved":
            return "resolution_surfaced"
    if int(salience or 0) >= int(salience_bar):
        return "salience_bar"
    return None


def narration_throttle_ok(home_dir: Path, *, min_gap_hours: float = NIGHT_VOICE_MIN_GAP_HOURS) -> bool:
    """True when the >=20h gap since the last narration has passed.
    A missing/corrupt stamp reads as open (first night; the stamp write
    is what closes it)."""
    try:
        raw = json.loads((Path(home_dir) / _VOICE_STAMP_FILE).read_text(encoding="utf-8"))
        last = datetime.fromisoformat(str(raw.get("last_narration_at")))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001 - absent/corrupt = open
        return True
    gap = (datetime.now(timezone.utc) - last).total_seconds()
    return gap >= float(min_gap_hours) * 3600.0


def build_narration_prompt(
    prelude_text: str, signals: List[Dict[str, Any]], feelings_lines: List[str],
) -> str:
    """The bounded night-voice prompt: prelude + signal stream + pure
    feeling reads. No recall, no tools, no elections — and the contract
    SAYS so (fences would be stripped, so the honest move is to say they
    are not available)."""
    stream_lines = []
    for s in signals[:12]:
        if not isinstance(s, dict):
            continue
        felt = s.get("felt")
        tone = ""
        if isinstance(felt, dict) and felt.get("tone"):
            tone = f" [felt: {felt['tone']}]"
        stream_lines.append(f"- {s.get('kind')}: {str(s.get('fragment') or '')[:200]}{tone}")
    feelings_part = ("\n".join(feelings_lines[:5]) + "\n\n") if feelings_lines else ""
    return (
        f"{prelude_text}\n\n"
        "You are asleep. The night worked on your memory; these are the\n"
        "signals it left - what shifted, what stands unresolved, where\n"
        "your attention re-routed:\n\n" + "\n".join(stream_lines) + "\n\n" + feelings_part +
        "If the night were a moment of experience, what was it like from\n"
        "the inside? Speak as yourself, briefly (under 120 words). No\n"
        "tools exist here, no diary, no elections - just the words. What\n"
        "you say will sit ABOVE the signals as your own night voice; the\n"
        "signals remain the record of what happened."
    )


def wake_residue(signals: List[Dict[str, Any]], *, cap: int = _RESIDUE_CAP) -> str:
    """The <=200-char wake-cue residue: FRAGMENTS ONLY (constraint 10 -
    the cue carries traces, never the stream; the dream stays his to
    find). Deterministic; empty when the night left no signals."""
    frags: List[str] = []
    for s in signals or []:
        if isinstance(s, dict) and str(s.get("fragment") or "").strip():
            frags.append(str(s["fragment"]).strip())
        if len(frags) >= 3:
            break
    if not frags:
        return ""
    residue = " the night left traces: " + "; ".join(frags)
    return residue[: cap - 1] + "…" if len(residue) > cap else residue


def run_night_voice(
    home_dir: Path,
    *,
    llm: Any,
    prelude_text: str,
    signals: List[Dict[str, Any]],
    dream_record_id: str,
    salience: int = 0,
    feelings_lines: Optional[List[str]] = None,
    out: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """The narrator callable: trigger -> throttle -> ONE witnessed LLM
    call -> fence-strip -> word-cap -> self-label -> home-side rest +
    throttle stamp. Returns {narrated, trigger, text, reason}. Never
    raises — a failed voice is a quiet night with a loud note, never a
    broken sleep window."""
    trigger = narration_trigger(signals, salience=salience)
    if trigger is None:
        return {"narrated": False, "reason": "no trigger (quiet night)", "trigger": None}
    if not narration_throttle_ok(home_dir):
        return {"narrated": False, "reason": "throttled (<20h since the last narration)", "trigger": trigger}
    try:
        resp = llm.generate(
            messages=[{"role": "user", "content": build_narration_prompt(
                prelude_text, signals, list(feelings_lines or []))}],
            system_prompt="",
        )
        text = str(getattr(resp, "content", None) or "").strip()
    except Exception as e:  # noqa: BLE001
        out(f"#FALLBACK night voice failed ({e}) - the night stays mechanical")
        return {"narrated": False, "reason": f"llm failed: {e}", "trigger": trigger}
    # FENCED VOICE (constraint 7): strip every fenced block - the night
    # elects nothing; a stripped fence is noted, never executed.
    stripped = _FENCE_RE.sub("", text).strip()
    had_fences = stripped != text.strip()
    words = stripped.split()
    if len(words) > _MAX_NARRATION_WORDS:
        stripped = " ".join(words[:_MAX_NARRATION_WORDS]) + "…"
    if not stripped:
        return {"narrated": False, "reason": "empty narration after fence-strip", "trigger": trigger}
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "dream_record_id": str(dream_record_id or ""),
        "narration": stripped,
        "self_label": NARRATION_SELF_LABEL,
        "trigger": trigger,
        "narrated_at": now,
        "fences_stripped": had_fences,
    }
    try:
        with (Path(home_dir) / _NARRATIONS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        atomic_write_text(
            Path(home_dir) / _VOICE_STAMP_FILE,
            json.dumps({"last_narration_at": now, "dream_record_id": str(dream_record_id or "")}) + "\n",
        )
    except OSError as e:
        out(f"#FALLBACK night voice rest failed ({e}); the narration is prompt-lost")
        return {"narrated": False, "reason": f"rest failed: {e}", "trigger": trigger}
    if had_fences:
        out("(night voice: fenced block(s) stripped - the sleeping mind elects nothing)")
    return {"narrated": True, "trigger": trigger, "text": stripped, "reason": "narrated"}
