"""An entity's cognition profile: which model each part of its thinking uses,
how many memories a turn brings to mind, and how readily it revisits what it
already understood.

The profile lives in the home, beside the mind (`substrate.yaml`):

    <home>/cognition.yaml
        schema_version: 1
        capabilities:                       # dialogue IS the mind (substrate.yaml)
          summaries:     {inherit: mind}    # or {provider, model, thinking?, max_input_tokens?}
          understanding: {inherit: mind}
          extraction:    {inherit: mind}
        recall:
          shelf_size: 50                    # memories retrieved per turn, visits and own time; 1..400
        rebuild:
          sensitivity: 0.5                  # 0..1, how easily new memories or surprises re-open what is understood
          cap_per_turn: 3                   # 0..10 memory tasks a turn may run at most; 0 = only while resting
        lookback:
          context: day                      # what the end-of-tick look-back sees: "day" (the whole day so far) or "tick"
        sleep:
          after_hours: 24                   # idle -> sleep once this long since the last completed sleep; 1..720
          after_events: 500                 # ... or once this many new memories formed since it; >= 1
          after_fragmented_share: 0.15      # ... or once more than this share of his memories is fragmented; 0 < x <= 1

Rules:

- Every capability defaults to the entity's mind (`inherit: mind`). There is
  no code default for a model anywhere in this module: a capability resolves
  to the profile's own choice, else to the mind (`resolve_home_substrate`),
  else it raises `SubstrateUnset`. `summaries` covers digests, titles,
  keywords and cards; `understanding` covers ripples, dream narration and the
  night's "what this changes"; `extraction` is reserved for the memory
  extraction lane.
- `max_input_tokens` defaults to 140000 and may only be lowered (to 4096 at
  the least).
- The recall width has no default here: the host passes its own
  (`recall_shelf_size(home, default)`); the gateway passes
  `DEFAULT_ENTITY_CHAT_SHELF_SIZE`.
- `lookback.context` says what the look-back at the end of a tick of its own
  time sees: "day" (the default) = the whole day so far, which a prefix cache
  keeps at a roughly constant cost of new tokens; "tick" = this tick's
  exchange plus a short recap of the day's earlier moments.
- `sleep` holds the thresholds after which an IDLE entity goes to sleep
  (CONTRACT v3 D2: hours since the last completed maintenance pass, memories
  formed since it; operator 2026-11-10: the share of his live memories that are
  FRAGMENTED - placed in a topic only provisionally, without a vector, or with
  a title or summary pending, which memory's sleep-cadence read counts); the
  gateway's idle sweep reads them (`sleep_thresholds`). Defaults 24 hours, 500
  memories and 0.15 (sleep once more than 15% are fragmented).
- Absent fields read as their defaults. A malformed file, or a malformed
  field, reads as unset and logs a warning naming the field; a read never
  raises. Writes are validated (`CognitionProfileInvalid`, with a sentence a
  person can act on) and atomic. The gateway is the only writer: it records a
  `cognition_profile_changed` marker before it calls `write_cognition_profile`.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .substrate import resolve_home_substrate

logger = logging.getLogger(__name__)

COGNITION_FILENAME = "cognition.yaml"
COGNITION_SCHEMA_VERSION = 1

CAPABILITIES: Tuple[str, ...] = ("summaries", "understanding", "extraction")
CAPABILITY_LABELS: Dict[str, str] = {
    "summaries": "Summaries and memory details",
    "understanding": "Understanding and dreams",
    "extraction": "Extraction",
}

INHERIT_KEY = "inherit"
INHERIT_MIND = "mind"

# The reasoning-effort words the gateway advertises (the same closed set its
# substrate route validates).
THINKING_VOCABULARY = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "auto", "on", "off"})

MAX_INPUT_TOKENS_CEILING = 140_000
MAX_INPUT_TOKENS_FLOOR = 4_096

SHELF_SIZE_MIN = 1
SHELF_SIZE_MAX = 400

DEFAULT_REBUILD_SENSITIVITY = 0.5
DEFAULT_REBUILD_CAP_PER_TURN = 3
CAP_PER_TURN_MIN = 0
CAP_PER_TURN_MAX = 10

LOOKBACK_CONTEXTS: Tuple[str, ...] = ("day", "tick")
DEFAULT_LOOKBACK_CONTEXT = "day"

DEFAULT_SLEEP_AFTER_HOURS = 24
DEFAULT_SLEEP_AFTER_EVENTS = 500
SLEEP_AFTER_HOURS_MIN = 1
SLEEP_AFTER_HOURS_MAX = 720
SLEEP_AFTER_EVENTS_MIN = 1
# The share of the home's live records that are fragmented (provisional community placement, no vector, or a title /
# summary pending) above which an idle entity goes to sleep: 0 < share <= 1 (operator 2026-11-10).
DEFAULT_SLEEP_AFTER_FRAGMENTED_SHARE = 0.15

SOURCE_PROFILE = "profile"
SOURCE_DEFAULT = "default"
SOURCE_MIND = "mind"

_CHOICE_KEYS = ("provider", "model", "thinking", "max_input_tokens")
_TOP_KEYS = ("schema_version", "capabilities", "recall", "rebuild", "lookback", "sleep")
_RECALL_KEYS = ("shelf_size",)
_REBUILD_KEYS = ("sensitivity", "cap_per_turn")
_LOOKBACK_KEYS = ("context",)
_SLEEP_KEYS = ("after_hours", "after_events", "after_fragmented_share")


class CognitionProfileInvalid(ValueError):
    """A profile value outside the schema. `field` names it (`capabilities.summaries`,
    `recall.shelf_size`, ...); the message is the sentence to show."""

    def __init__(self, field: str, sentence: str) -> None:
        super().__init__(sentence)
        self.field = field
        self.sentence = sentence


def _label(capability: str) -> str:
    return CAPABILITY_LABELS.get(capability, capability)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ------------------------------------------------------------------ validators


def check_capability_name(capability: Any) -> str:
    cap = str(capability or "").strip()
    if cap not in CAPABILITIES:
        raise CognitionProfileInvalid(
            "capabilities",
            f"There is no part of thinking called {capability!r}. Choose one of: {', '.join(CAPABILITIES)}.",
        )
    return cap


def validate_capability_choice(capability: str, value: Any) -> Dict[str, Any]:
    """`{"inherit": "mind"}` or `{provider, model[, thinking][, max_input_tokens]}` as stored.

    `thinking` is lower-cased and present only when chosen; `max_input_tokens`
    is present only when given (4096..140000; absent means 140000)."""
    cap = check_capability_name(capability)
    field = f"capabilities.{cap}"
    label = _label(cap)
    if not isinstance(value, Mapping):
        raise CognitionProfileInvalid(
            field, f"{label} needs either {{\"inherit\": \"mind\"}} or a provider and a model."
        )
    if INHERIT_KEY in value:
        if set(value.keys()) != {INHERIT_KEY} or str(value.get(INHERIT_KEY) or "").strip() != INHERIT_MIND:
            raise CognitionProfileInvalid(
                field, f"{label} can only inherit from the mind: send {{\"inherit\": \"mind\"}} and nothing else."
            )
        return {INHERIT_KEY: INHERIT_MIND}
    unknown = sorted(str(k) for k in value.keys() if k not in _CHOICE_KEYS)
    if unknown:
        raise CognitionProfileInvalid(
            field,
            f"{label} has an unknown field {unknown[0]!r}. Allowed: {', '.join(_CHOICE_KEYS)}.",
        )
    provider = value.get("provider")
    model = value.get("model")
    provider_s = provider.strip() if isinstance(provider, str) else ""
    model_s = model.strip() if isinstance(model, str) else ""
    if not provider_s or not model_s:
        raise CognitionProfileInvalid(
            field, f"{label} needs both a provider and a model, or the same model as the mind."
        )
    out: Dict[str, Any] = {"provider": provider_s, "model": model_s}
    thinking = value.get("thinking")
    # YAML 1.1 reads a hand-written `thinking: off` as False (and `on` as True).
    if thinking is True:
        thinking = "on"
    elif thinking is False:
        thinking = "off"
    if thinking is not None:
        t = thinking.strip().lower() if isinstance(thinking, str) else None
        if t is None or (t and t not in THINKING_VOCABULARY):
            raise CognitionProfileInvalid(
                f"{field}.thinking",
                f"Unknown reasoning effort {thinking!r} for {label}. Choose one of: "
                + ", ".join(sorted(THINKING_VOCABULARY))
                + ".",
            )
        if t:
            out["thinking"] = t
    mit = value.get("max_input_tokens")
    if mit is not None:
        if not _is_int(mit) or not (MAX_INPUT_TOKENS_FLOOR <= mit <= MAX_INPUT_TOKENS_CEILING):
            raise CognitionProfileInvalid(
                f"{field}.max_input_tokens",
                f"The input limit for {label} must be a whole number from {MAX_INPUT_TOKENS_FLOOR} "
                f"to {MAX_INPUT_TOKENS_CEILING}.",
            )
        out["max_input_tokens"] = int(mit)
    return out


def validate_shelf_size(value: Any) -> int:
    if not _is_int(value) or not (SHELF_SIZE_MIN <= value <= SHELF_SIZE_MAX):
        raise CognitionProfileInvalid(
            "recall.shelf_size",
            f"Memories per turn must be a whole number from {SHELF_SIZE_MIN} to {SHELF_SIZE_MAX}.",
        )
    return int(value)


def validate_sensitivity(value: Any) -> float:
    ok = (_is_int(value) or isinstance(value, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    if not ok:
        raise CognitionProfileInvalid("rebuild.sensitivity", "Rebuild sensitivity must be a number from 0 to 1.")
    return float(value)


def validate_cap_per_turn(value: Any) -> int:
    if not _is_int(value) or not (CAP_PER_TURN_MIN <= value <= CAP_PER_TURN_MAX):
        raise CognitionProfileInvalid(
            "rebuild.cap_per_turn",
            f"Rebuilds per turn must be a whole number from {CAP_PER_TURN_MIN} to {CAP_PER_TURN_MAX}.",
        )
    return int(value)


def validate_lookback_context(value: Any) -> str:
    context = value.strip().lower() if isinstance(value, str) else None
    if context not in LOOKBACK_CONTEXTS:
        raise CognitionProfileInvalid("lookback.context", "Look-back context must be day or tick.")
    return context


def validate_sleep_after_hours(value: Any) -> int:
    if not _is_int(value) or not (SLEEP_AFTER_HOURS_MIN <= value <= SLEEP_AFTER_HOURS_MAX):
        raise CognitionProfileInvalid(
            "sleep.after_hours",
            f"Hours before he sleeps must be a whole number from {SLEEP_AFTER_HOURS_MIN} to {SLEEP_AFTER_HOURS_MAX}.",
        )
    return int(value)


def validate_sleep_after_events(value: Any) -> int:
    if not _is_int(value) or value < SLEEP_AFTER_EVENTS_MIN:
        raise CognitionProfileInvalid(
            "sleep.after_events",
            f"New memories before he sleeps must be a whole number of at least {SLEEP_AFTER_EVENTS_MIN}.",
        )
    return int(value)


def validate_sleep_after_fragmented_share(value: Any) -> float:
    ok = (_is_int(value) or isinstance(value, float)) and math.isfinite(float(value)) and 0.0 < float(value) <= 1.0
    if not ok:
        raise CognitionProfileInvalid(
            "sleep.after_fragmented_share",
            "The share of fragmented memories before he sleeps must be a number above 0 and at most 1 "
            "(0.15 = 15%).",
        )
    return float(value)


# The sleep block's checks, ONE table for the writer and the reader.
_SLEEP_CHECKS = {"after_hours": validate_sleep_after_hours, "after_events": validate_sleep_after_events,
                 "after_fragmented_share": validate_sleep_after_fragmented_share}


def validate_cognition_profile(profile: Any) -> Dict[str, Any]:
    """The stored form of `profile`, strictly validated (raises `CognitionProfileInvalid`).

    The stored form keeps only what was chosen: a capability that inherits the
    mind, and a block whose fields all follow their defaults, are left out."""
    if not isinstance(profile, Mapping):
        raise CognitionProfileInvalid("profile", "The cognition profile must be a mapping.")
    unknown = sorted(str(k) for k in profile.keys() if k not in _TOP_KEYS)
    if unknown:
        raise CognitionProfileInvalid(
            unknown[0], f"Unknown field {unknown[0]!r}. Send capabilities, recall, rebuild, lookback or sleep."
        )
    version = profile.get("schema_version", COGNITION_SCHEMA_VERSION)
    if version != COGNITION_SCHEMA_VERSION or isinstance(version, bool):
        raise CognitionProfileInvalid(
            "schema_version", f"This runtime reads cognition profile schema {COGNITION_SCHEMA_VERSION}, not {version!r}."
        )
    out: Dict[str, Any] = {"schema_version": COGNITION_SCHEMA_VERSION}
    caps_in = profile.get("capabilities")
    if caps_in is not None:
        if not isinstance(caps_in, Mapping):
            raise CognitionProfileInvalid("capabilities", "capabilities must map a part of thinking to its model.")
        caps: Dict[str, Any] = {}
        for cap, choice in caps_in.items():
            name = check_capability_name(cap)
            if choice is None:
                continue
            normalized = validate_capability_choice(name, choice)
            if normalized != {INHERIT_KEY: INHERIT_MIND}:
                caps[name] = normalized
        if caps:
            out["capabilities"] = caps
    for block, keys, checks in (
        ("recall", _RECALL_KEYS, {"shelf_size": validate_shelf_size}),
        ("rebuild", _REBUILD_KEYS, {"sensitivity": validate_sensitivity, "cap_per_turn": validate_cap_per_turn}),
        ("lookback", _LOOKBACK_KEYS, {"context": validate_lookback_context}),
        ("sleep", _SLEEP_KEYS, _SLEEP_CHECKS),
    ):
        raw = profile.get(block)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise CognitionProfileInvalid(block, f"{block} must be a mapping of {', '.join(keys)}.")
        extra = sorted(str(k) for k in raw.keys() if k not in keys)
        if extra:
            raise CognitionProfileInvalid(
                f"{block}.{extra[0]}", f"{block} has an unknown field {extra[0]!r}. Allowed: {', '.join(keys)}."
            )
        values = {k: checks[k](raw[k]) for k in keys if raw.get(k) is not None}
        if values:
            out[block] = values
    return out


# ------------------------------------------------------------------- reading


def _load_file(home_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    """The file's mapping (or {}) and the warnings about it. Never raises."""
    path = Path(home_dir) / COGNITION_FILENAME
    if not path.exists():
        return {}, []
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - a broken file reads as unset, never a crash
        return {}, [f"{path}: not readable YAML ({e}); every field reads as its default"]
    if data is None:
        return {}, []
    if not isinstance(data, dict):
        return {}, [f"{path}: not a mapping; every field reads as its default"]
    version = data.get("schema_version", COGNITION_SCHEMA_VERSION)
    if version != COGNITION_SCHEMA_VERSION or isinstance(version, bool):
        return {}, [
            f"{path}: schema_version {version!r} is not {COGNITION_SCHEMA_VERSION}; every field reads as its default"
        ]
    return data, []


def stored_cognition_profile(home_dir: Path) -> Dict[str, Any]:
    """What the file chose, in the stored form (`validate_cognition_profile`'s
    shape), with every malformed field left out (each logged). A writer's
    keep-what-is-not-sent starts from this."""
    stored, _warnings = _stored_with_warnings(home_dir)
    return stored


def _stored_with_warnings(home_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    data, warnings = _load_file(Path(home_dir))
    out: Dict[str, Any] = {"schema_version": COGNITION_SCHEMA_VERSION}
    where = Path(home_dir) / COGNITION_FILENAME

    def _bad(e: CognitionProfileInvalid) -> None:
        warnings.append(f"{where}: {e.field} reads as its default ({e.sentence})")

    caps_in = data.get("capabilities")
    if caps_in is not None and not isinstance(caps_in, Mapping):
        warnings.append(f"{where}: capabilities is not a mapping; every capability inherits the mind")
        caps_in = None
    caps: Dict[str, Any] = {}
    for cap, choice in (caps_in or {}).items():
        try:
            name = check_capability_name(cap)
        except CognitionProfileInvalid as e:
            _bad(e)
            continue
        if choice is None:
            continue
        try:
            normalized = validate_capability_choice(name, choice)
        except CognitionProfileInvalid as e:
            _bad(e)
            continue
        if normalized != {INHERIT_KEY: INHERIT_MIND}:
            caps[name] = normalized
    if caps:
        out["capabilities"] = caps
    for block, checks in (
        ("recall", {"shelf_size": validate_shelf_size}),
        ("rebuild", {"sensitivity": validate_sensitivity, "cap_per_turn": validate_cap_per_turn}),
        ("lookback", {"context": validate_lookback_context}),
        ("sleep", _SLEEP_CHECKS),
    ):
        raw = data.get(block)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            warnings.append(f"{where}: {block} is not a mapping; its fields read as their defaults")
            continue
        values: Dict[str, Any] = {}
        for key, check in checks.items():
            if raw.get(key) is None:
                continue
            try:
                values[key] = check(raw[key])
            except CognitionProfileInvalid as e:
                _bad(e)
        if values:
            out[block] = values
    for w in warnings:
        logger.warning("cognition profile: %s", w)
    return out, warnings


def read_cognition_profile(home_dir: Path) -> Dict[str, Any]:
    """The profile normalized to the full schema, every field with its source.

        {"schema_version": 1,
         "capabilities": {cap: {"choice": {"inherit": "mind"} | {provider, model, thinking, max_input_tokens},
                                "source": "profile" | "default"}},
         "recall": {"shelf_size": int | None, "source": "profile" | "default"},
         "rebuild": {"sensitivity": float, "cap_per_turn": int,
                     "sources": {"sensitivity": ..., "cap_per_turn": ...}},
         "lookback": {"context": "day" | "tick", "source": "profile" | "default"},
         "sleep": {"after_hours": int, "after_events": int, "after_fragmented_share": float,
                   "sources": {"after_hours": ..., "after_events": ..., "after_fragmented_share": ...}},
         "warnings": [sentences about malformed fields]}

    `recall.shelf_size` is None when the profile does not choose one: the width
    is the host's default then (`recall_shelf_size`). A custom choice carries
    `thinking` (None when not chosen) and `max_input_tokens` (140000 when not
    lowered). Never raises."""
    stored, warnings = _stored_with_warnings(Path(home_dir))
    caps_stored = stored.get("capabilities") or {}
    capabilities: Dict[str, Any] = {}
    for cap in CAPABILITIES:
        choice = caps_stored.get(cap)
        if choice is None:
            capabilities[cap] = {"choice": {INHERIT_KEY: INHERIT_MIND}, "source": SOURCE_DEFAULT}
        else:
            capabilities[cap] = {
                "choice": {
                    "provider": choice["provider"],
                    "model": choice["model"],
                    "thinking": choice.get("thinking"),
                    "max_input_tokens": int(choice.get("max_input_tokens") or MAX_INPUT_TOKENS_CEILING),
                },
                "source": SOURCE_PROFILE,
            }
    recall_stored = stored.get("recall") or {}
    rebuild_stored = stored.get("rebuild") or {}
    lookback_stored = stored.get("lookback") or {}
    sleep_stored = stored.get("sleep") or {}
    return {
        "schema_version": COGNITION_SCHEMA_VERSION,
        "capabilities": capabilities,
        "recall": {
            "shelf_size": recall_stored.get("shelf_size"),
            "source": SOURCE_PROFILE if "shelf_size" in recall_stored else SOURCE_DEFAULT,
        },
        "rebuild": {
            "sensitivity": float(rebuild_stored.get("sensitivity", DEFAULT_REBUILD_SENSITIVITY)),
            "cap_per_turn": int(rebuild_stored.get("cap_per_turn", DEFAULT_REBUILD_CAP_PER_TURN)),
            "sources": {
                key: (SOURCE_PROFILE if key in rebuild_stored else SOURCE_DEFAULT) for key in _REBUILD_KEYS
            },
        },
        "lookback": {
            "context": str(lookback_stored.get("context", DEFAULT_LOOKBACK_CONTEXT)),
            "source": SOURCE_PROFILE if "context" in lookback_stored else SOURCE_DEFAULT,
        },
        "sleep": {
            "after_hours": int(sleep_stored.get("after_hours", DEFAULT_SLEEP_AFTER_HOURS)),
            "after_events": int(sleep_stored.get("after_events", DEFAULT_SLEEP_AFTER_EVENTS)),
            "after_fragmented_share": float(sleep_stored.get("after_fragmented_share",
                                                             DEFAULT_SLEEP_AFTER_FRAGMENTED_SHARE)),
            "sources": {key: (SOURCE_PROFILE if key in sleep_stored else SOURCE_DEFAULT) for key in _SLEEP_KEYS},
        },
        "warnings": list(warnings),
    }


# ------------------------------------------------------------------- writing


def write_cognition_profile(home_dir: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate `profile` and replace `<home>/cognition.yaml` atomically
    (temporary file in the home, fsync, rename). Returns the stored form.

    The gateway is the only caller, after its `cognition_profile_changed`
    marker landed. Raises `CognitionProfileInvalid` before touching the file."""
    stored = validate_cognition_profile(profile)
    import yaml

    home = Path(home_dir)
    target = home / COGNITION_FILENAME
    text = yaml.safe_dump(stored, sort_keys=False, default_flow_style=False)
    fd, tmp_name = tempfile.mkstemp(prefix=".cognition.", suffix=".yaml.tmp", dir=str(home))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return stored


# ----------------------------------------------------------------- resolution


def resolve_capability(
    home_dir: Path,
    capability: str,
    *,
    mind: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The model a capability runs on: `{provider, model, thinking, max_input_tokens, source}`.

    `source` is "profile" (the profile's own choice) or "mind". Inheriting goes
    through `resolve_home_substrate`; `mind` is the host's already-resolved
    mind (`{provider, model, thinking}`), its explicit rung, so a host whose
    mind chain differs (the gateway's text route) resolves the same mind its
    visits use. Nothing resolvable raises `SubstrateUnset`; an unknown
    capability raises `CognitionProfileInvalid`."""
    cap = check_capability_name(capability)
    choice = read_cognition_profile(Path(home_dir))["capabilities"][cap]["choice"]
    if choice.get(INHERIT_KEY) == INHERIT_MIND:
        given = mind or {}
        provider, model, thinking = resolve_home_substrate(
            given.get("provider"),
            given.get("model"),
            home_dir=Path(home_dir),
            thinking=given.get("thinking"),
        )
        return {
            "provider": provider,
            "model": model,
            "thinking": thinking,
            "max_input_tokens": MAX_INPUT_TOKENS_CEILING,
            "source": SOURCE_MIND,
        }
    return {
        "provider": choice["provider"],
        "model": choice["model"],
        "thinking": choice.get("thinking"),
        "max_input_tokens": int(choice["max_input_tokens"]),
        "source": SOURCE_PROFILE,
    }


def recall_shelf_size(home_dir: Path, default: int) -> int:
    """Memories retrieved per turn (visits and own time): the profile's
    `recall.shelf_size`, else the host's `default`."""
    chosen = read_cognition_profile(Path(home_dir))["recall"]["shelf_size"]
    return int(chosen) if chosen is not None else int(default)


def rebuild_policy(home_dir: Path) -> Dict[str, Any]:
    """`{sensitivity, cap_per_turn}`: the profile's values, else 0.5 and 3."""
    rebuild = read_cognition_profile(Path(home_dir))["rebuild"]
    return {"sensitivity": float(rebuild["sensitivity"]), "cap_per_turn": int(rebuild["cap_per_turn"])}


def lookback_context(home_dir: Path) -> str:
    """What the end-of-tick look-back sees: "day" (the whole day so far; the
    default) or "tick" (this tick's exchange plus a recap of earlier moments)."""
    return str(read_cognition_profile(Path(home_dir))["lookback"]["context"])


def sleep_thresholds(home_dir: Path) -> Dict[str, Any]:
    """`{after_hours, after_events, after_fragmented_share}`: when an idle entity goes to sleep (CONTRACT v3 D2;
    operator 2026-11-10) - the profile's values, else 24 hours, 500 new memories since the last completed sleep, or
    more than 15% of his live memories fragmented (memory's sleep-cadence read counts them)."""
    sleep = read_cognition_profile(Path(home_dir))["sleep"]
    return {"after_hours": int(sleep["after_hours"]), "after_events": int(sleep["after_events"]),
            "after_fragmented_share": float(sleep["after_fragmented_share"])}


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_LABELS",
    "COGNITION_FILENAME",
    "COGNITION_SCHEMA_VERSION",
    "CognitionProfileInvalid",
    "LOOKBACK_CONTEXTS",
    "DEFAULT_LOOKBACK_CONTEXT",
    "DEFAULT_REBUILD_CAP_PER_TURN",
    "DEFAULT_REBUILD_SENSITIVITY",
    "DEFAULT_SLEEP_AFTER_EVENTS",
    "DEFAULT_SLEEP_AFTER_FRAGMENTED_SHARE",
    "DEFAULT_SLEEP_AFTER_HOURS",
    "MAX_INPUT_TOKENS_CEILING",
    "MAX_INPUT_TOKENS_FLOOR",
    "SHELF_SIZE_MAX",
    "SHELF_SIZE_MIN",
    "SLEEP_AFTER_HOURS_MAX",
    "SLEEP_AFTER_HOURS_MIN",
    "SLEEP_AFTER_EVENTS_MIN",
    "THINKING_VOCABULARY",
    "check_capability_name",
    "lookback_context",
    "rebuild_policy",
    "recall_shelf_size",
    "read_cognition_profile",
    "resolve_capability",
    "sleep_thresholds",
    "stored_cognition_profile",
    "validate_cap_per_turn",
    "validate_capability_choice",
    "validate_cognition_profile",
    "validate_lookback_context",
    "validate_shelf_size",
    "validate_sleep_after_events",
    "validate_sleep_after_fragmented_share",
    "validate_sleep_after_hours",
    "validate_sensitivity",
]
