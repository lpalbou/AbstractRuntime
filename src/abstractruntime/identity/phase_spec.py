"""The entity phase blueprint as a CONSUMED artifact (spec v11, dm#104).

Laurent's ruling made the state graph the SOURCE OF TRUTH for behavior:
"the state graph of the entity you created MUST be the source of truth
for all... i would like you to make those blueprints editable by me, so i
can slightly modulate the entity cognition and cycle."

This module is the runtime's READ SIDE of that ruling: machine numbers
(the `tunables` block) come FROM the blueprint, never from constants — a
blueprint edit changes behavior with zero code change. Resolution order:

1. explicit path argument (tests, harnesses),
2. the GATEWAY'S OPERATOR COPY, `<data_dir>/config/entity_phases.json`,
   located from the home dir (`<data_dir>/entities/<slug>` -> ../../config)
   — the PUT edit lane's persistence (gateway c-t-i 350): laurent's
   modulation reaches DETACHED loops as a plain file read, no HTTP, no
   env setup,
3. `ABSTRACTRUNTIME_PHASE_SPEC` env (test/harness convenience — BELOW the
   operator copy per the dm#177 ruling: behavior env never out-ranks
   console config),
4. the packaged vendored artifact (identity/spec/entity_phases.vendored.json).

Failure posture: an unreadable override falls to the vendored artifact
with a loud warning line returned to the caller; an unreadable vendored
artifact returns the RULED DEFAULTS (the v11 seed numbers) — the loop
must never die over its dials, and the defaults are the ruling's own
numbers, not inventions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = ["PHASE_SPEC_ENV", "load_phase_tunables", "vendored_spec_path"]

PHASE_SPEC_ENV = "ABSTRACTRUNTIME_PHASE_SPEC"

# The v11 seed numbers (laurent dm#104 + the standing rulings) — the
# last-resort defaults when NO artifact is readable. These mirror the
# blueprint's seeds; they are a safety floor, never the tuning surface.
_RULED_DEFAULTS: Dict[str, Any] = {
    "personal_cycle": {"personal_window_h": 2.0, "sleep_window_h": 1.0, "enabled": True},
    # v12 rename: sleep_bound_h (was sleep_bound_s - the one seconds-dial
    # in an hours block, renamed while still unwired).
    "sleep_bound_h": 1.0,
    "unattended_wake_cadence_h": 6.0,
    "grant_unused_floor_h": 2.0,
    # v16 window rows (M2, laurent's global-dials answer): the resident
    # temporal horizon - one home one horizon, both hosts thread the same
    # resolved number into AttentionConfig at home-open.
    "window_limit": 8192,
    "drive_window_limit": 256,
}

# HONESTY LEDGER (v12 P0-4, mirrored from the blueprint's tunables_meta):
# which dials the runtime actually THREADS today. Editing an unwired dial
# is a recorded no-op, never a silent one - load_phase_tunables warns and
# the blueprint UI renders unwired dials read-only. Flipping one true =
# threading the blueprint read through the consumer named here.
TUNABLE_WIRED: Dict[str, bool] = {
    "personal_cycle.personal_window_h": True,   # the cycle trigger (life.py)
    "personal_cycle.sleep_window_h": True,      # the cycle window (life.py)
    "personal_cycle.enabled": True,             # the cycle kill switch (life.py)
    "sleep_bound_h": True,                      # sleep_bound_deadline(home_dir=) reads it (v14; sweeper adopts the kwarg)
    "unattended_wake_cadence_h": True,          # the day gate's need_check_s (life.py read_day_gate)
    "grant_unused_floor_h": True,               # the day gate's use floor (life.py read_day_gate)
    # Runtime half threaded (loop home-open + cue read); the blueprint meta
    # flips wired:true when GATEWAY's home-open sites thread too (v16 rule:
    # both hosts, one horizon).
    "window_limit": True,
    "drive_window_limit": True,
}


def vendored_spec_path() -> Path:
    return Path(__file__).parent / "spec" / "entity_phases.vendored.json"


def _read_tunables(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tunables = data.get("tunables")
        return dict(tunables) if isinstance(tunables, dict) else None
    except Exception:  # noqa: BLE001 - callers get the fallback + warning
        return None


def operator_copy_path(home_dir: Path) -> Path:
    """The gateway PUT lane's persisted blueprint, located from a home dir
    (<data_dir>/entities/<slug> -> <data_dir>/config/entity_phases.json)."""
    return Path(home_dir).resolve().parent.parent / "config" / "entity_phases.json"


def load_phase_tunables(
    spec_path: Optional[Path] = None,
    *,
    home_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], list]:
    """(tunables, warnings) — the blueprint's dials, override-aware.

    The returned dict always carries the full key set (missing keys fill
    from the ruled defaults so a partial blueprint edit never KeyErrors a
    consumer); warnings name every degradation loudly."""
    warnings: list = []
    candidates: list = []
    if spec_path is not None:
        candidates.append((Path(spec_path), "explicit path"))
    if home_dir is not None:
        op_copy = operator_copy_path(home_dir)
        if op_copy.is_file():
            # Present = laurent edited the blueprint; absent is the normal
            # un-edited state (never a warning).
            candidates.append((op_copy, "gateway operator copy"))
    # ENV BELOW CONFIG (operator ruling 2026-07-21 dm#177: behavior env
    # vars must never out-rank console config): the env override is a
    # test/harness convenience and sits UNDER the gateway operator copy -
    # an exported shell var can no longer silently beat laurent's console
    # blueprint edit. It out-ranks only the packaged vendored artifact.
    env_path = os.environ.get(PHASE_SPEC_ENV, "").strip()
    if env_path:
        candidates.append((Path(env_path), f"env {PHASE_SPEC_ENV}"))
    candidates.append((vendored_spec_path(), "vendored artifact"))

    raw: Optional[Dict[str, Any]] = None
    for path, label in candidates:
        raw = _read_tunables(path)
        if raw is not None:
            break
        warnings.append(f"#FALLBACK phase tunables unreadable at {label} ({path})")
    if raw is None:
        warnings.append("#FALLBACK all tunable sources unreadable; the RULED defaults stand")
        raw = {}

    out: Dict[str, Any] = {}
    cycle_raw = raw.get("personal_cycle") if isinstance(raw.get("personal_cycle"), dict) else {}
    cycle_defaults = _RULED_DEFAULTS["personal_cycle"]
    out["personal_cycle"] = {
        "personal_window_h": _pos_float(cycle_raw.get("personal_window_h"), cycle_defaults["personal_window_h"], warnings, "personal_cycle.personal_window_h"),
        "sleep_window_h": _pos_float(cycle_raw.get("sleep_window_h"), cycle_defaults["sleep_window_h"], warnings, "personal_cycle.sleep_window_h"),
        # v13: the cycle kill switch (bool dial - only a literal false
        # disables; anything unreadable keeps the ruled default, loudly).
        "enabled": _bool(cycle_raw.get("enabled"), True, warnings, "personal_cycle.enabled"),
    }
    for key in ("sleep_bound_h", "unattended_wake_cadence_h", "grant_unused_floor_h",
                "window_limit", "drive_window_limit"):
        if key not in raw and raw:
            # v12: a MISSING known key on an edited blueprint is loud - a
            # typo'd dial must never read as a tuned one.
            warnings.append(f"#NOTE tunable {key} absent from the edited blueprint; ruled default {_RULED_DEFAULTS[key]} stands")
        out[key] = _pos_float(raw.get(key), _RULED_DEFAULTS[key], warnings, key)
    # Unwired dials that DIFFER from the ruled defaults are edits with no
    # engine behind them yet - say so (P0-4's silent-no-op class).
    for dotted, wired in TUNABLE_WIRED.items():
        if wired:
            continue
        head, _, tail = dotted.partition(".")
        current = out.get(head, {}).get(tail) if tail else out.get(head)
        default = _RULED_DEFAULTS.get(head, {}).get(tail) if tail else _RULED_DEFAULTS.get(head)
        if current != default:
            warnings.append(
                f"#NOTE tunable {dotted}={current} is edited but NOT YET WIRED "
                f"(hardcoded twin governs; the edit is recorded, not applied)"
            )
    return out, warnings


def _bool(value: Any, default: bool, warnings: list, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    warnings.append(f"#FALLBACK tunable {name}={value!r} not a bool; ruled default {default} stands")
    return default


def _pos_float(value: Any, default: float, warnings: list, name: str) -> float:
    try:
        number = float(value)
        if number > 0:
            return number
        warnings.append(f"#FALLBACK tunable {name}={value!r} not positive; ruled default {default} stands")
    except (TypeError, ValueError):
        if value is not None:
            warnings.append(f"#FALLBACK tunable {name}={value!r} unreadable; ruled default {default} stands")
    return float(default)
