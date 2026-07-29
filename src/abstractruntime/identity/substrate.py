"""The entity's ONE persisted mind substrate (provider + model).

Maintainer rulings, 2026-07-09:
- 04:26 "I decide which provider and model is used ... NO FALLBACK": no code
  constant may elect a substrate anywhere — a default quietly spending the
  operator's money is exactly the fallback class the ADR forbids.
- 06:32 "i don't see the point in having potentially different models for
  visit and own time": the entity carries ONE persisted choice — an operator
  file in his home (`substrate.yaml`, beside tool_policy.yaml) — set once,
  used by visits AND his own time.

Resolution chain (every rung an explicit operator act, never a code default):

    explicit request (CLI flags / HTTP body) > home substrate.yaml >
    operator env (ABSTRACTGATEWAY_ENTITY_CHAT_PROVIDER/_MODEL) > LOUD REFUSAL

The env names are gateway-spelled on purpose: they are the operator's ONE
knob (the serve script's exports ARE the operator's choice); a runtime-spelled
twin would split that knob in two. The gateway's entity_chat.resolve_substrate
implements the same chain HTTP-side; this module is the home-direct half so
the runtime CLIs (identity.chat / identity.life) obey the same rulings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

SUBSTRATE_FILENAME = "substrate.yaml"

SUBSTRATE_ENV_PROVIDER = "ABSTRACTGATEWAY_ENTITY_CHAT_PROVIDER"
SUBSTRATE_ENV_MODEL = "ABSTRACTGATEWAY_ENTITY_CHAT_MODEL"


class SubstrateUnset(RuntimeError):
    """No complete provider+model choice exists on any rung of the chain."""


def read_home_substrate(home_dir: Path) -> Dict[str, str]:
    """The entity's persisted substrate choice: {provider, model[, thinking]} or {}.

    Both-or-nothing applies to the provider+model PAIR (a half choice is no
    choice); a malformed file reads as unset so the resolve site stays
    works-or-loud — same semantics as the gateway's reader, one contract.

    `thinking` (reasoning effort) is OPTIONAL and rides along only when the
    pair is set: one mind = one substrate, and a reasoning knob without a
    chosen mind is meaningless. Absent means unset — never a refusal. The
    key is spelled `thinking` at rest (the reasoning plan's one-spelling
    decision, 2026-07-26); readers that predate this field simply ignore
    it, and this reader returning it is what lets the gateway's writer
    store it without the field being silently dropped on read."""
    path = Path(home_dir) / SUBSTRATE_FILENAME
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        p = str(data.get("provider") or "").strip()
        m = str(data.get("model") or "").strip()
        if not (p and m):
            return {}
        out = {"provider": p, "model": m}
        raw_t = data.get("thinking")
        # YAML 1.1 gotcha (adversary P1): a hand-written `thinking: off`
        # (or on/true/false/no/yes) parses as a BOOLEAN, and str(False or "")
        # would silently swallow it — inverting the operator's intent (they
        # said "reasoning off"; the model would run its own default). Map
        # booleans to the words core understands.
        if raw_t is True:
            t = "on"
        elif raw_t is False:
            t = "off"
        else:
            t = str(raw_t or "").strip()
        if t:
            out["thinking"] = t
        return out
    except Exception:  # noqa: BLE001 - unset beats a crashed summon; resolve refuses loudly
        return {}


def normalize_thinking_or_drop(value: Optional[str], *, source: str) -> Optional[str]:
    """Check a reasoning-effort value against core's own vocabulary.

    A bad value stored in a file would otherwise fail EVERY model call in
    the session (core refuses unknown values loudly). At this edge we drop
    the bad value with a labeled warning instead — the session runs without
    the dial, and the warning names what to fix. The check calls core's own
    parser so the vocabulary can never drift; if that parser is missing
    (older core), the value passes through and core's own guard decides.
    """
    v = (value or "").strip()
    if not v:
        return None
    try:
        from abstractcore.providers.base import BaseProvider

        checker = getattr(BaseProvider, "_normalize_thinking_request", None)
        if callable(checker):
            checker(v)  # raises ValueError on junk
    except ValueError:
        print(
            f"#FALLBACK ignoring invalid thinking value {v!r} from {source} "
            "(valid: none, minimal, low, medium, high, xhigh, auto, on, off); "
            "this session runs without the reasoning dial"
        )
        return None
    except Exception:
        # Core absent or its private parser moved: pass through; core's own
        # loud guard at the call site stays the final check.
        pass
    return v


def resolve_home_substrate(
    provider: Optional[str],
    model: Optional[str],
    *,
    home_dir: Path,
    thinking: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """Explicit > substrate.yaml > operator env > SubstrateUnset (loud).

    Each field fills independently from successive rungs (the gateway's
    approved semantics); an incomplete pair after all rungs refuses with
    every fix named — the operator decides, the code never does.

    Returns (provider, model, thinking). The third field is the OPTIONAL
    reasoning effort (reasoning plan, 2026-07-26): explicit argument wins,
    else the home's stored choice, else None — absent is a fine answer,
    never a refusal (the dial is optional; the mind is not). There is no
    env rung for thinking today; adding one is an operator-knob decision
    to coordinate with the gateway, not a runtime default."""
    p = (provider or "").strip()
    m = (model or "").strip()
    t = (thinking or "").strip()
    stored: Dict[str, str] = {}
    if not (p and m) or not t:
        stored = read_home_substrate(home_dir)
    if not (p and m):
        p = p or stored.get("provider", "")
        m = m or stored.get("model", "")
    t = t or stored.get("thinking", "")
    p = p or (os.getenv(SUBSTRATE_ENV_PROVIDER) or "").strip()
    m = m or (os.getenv(SUBSTRATE_ENV_MODEL) or "").strip()
    if not p or not m:
        raise SubstrateUnset(
            "no mind substrate chosen for this entity - the operator decides, "
            "never a code default. Fix one of: pass --provider AND --model; "
            f"write {Path(home_dir) / SUBSTRATE_FILENAME} (the gateway's "
            "PUT /api/gateway/entities/{name}/substrate or the UI's mind picker "
            f"writes it); or export {SUBSTRATE_ENV_PROVIDER} + {SUBSTRATE_ENV_MODEL}."
        )
    return p, m, normalize_thinking_or_drop(t, source=f"{Path(home_dir) / SUBSTRATE_FILENAME} or flags")
