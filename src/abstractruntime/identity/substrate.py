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
        t = str(data.get("thinking") or "").strip()
        if t:
            out["thinking"] = t
        return out
    except Exception:  # noqa: BLE001 - unset beats a crashed summon; resolve refuses loudly
        return {}


def resolve_home_substrate(
    provider: Optional[str], model: Optional[str], *, home_dir: Path
) -> Tuple[str, str]:
    """Explicit > substrate.yaml > operator env > SubstrateUnset (loud).

    Each field fills independently from successive rungs (the gateway's
    approved semantics); an incomplete pair after all rungs refuses with
    every fix named — the operator decides, the code never does."""
    p = (provider or "").strip()
    m = (model or "").strip()
    if not (p and m):
        stored = read_home_substrate(home_dir)
        p = p or stored.get("provider", "")
        m = m or stored.get("model", "")
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
    return p, m
