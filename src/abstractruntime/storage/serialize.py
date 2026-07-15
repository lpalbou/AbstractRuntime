"""abstractruntime.storage.serialize

Durable-write serialization discipline (backlog 0067, operator-signed
2026-07-13): the run/ledger hot path used `dataclasses.asdict`, which walks
and REBUILDS the entire vars container tree (context messages, visit
sheets, event inboxes) on every checkpoint save and every ledger append — a
structural copy (strings are shared, every dict/list reconstructed; ~0.5-
0.9ms CPU per save on realistic states) — and the JSON run files were
written with `indent=2` (+75% serializer CPU on token-dense states,
measured; the win shrinks toward 1.0x on very string-heavy states). At ~5
saves per agent cycle this was the most frequent durable cost in the
system.

Under the single-writer ownership contract (see JsonFileRunStore's
docstring; ruling 2026-07-09) only the owning tick/host thread may mutate a
loaded RunState, and that is the same thread performing the save — so the
deep copy protects nothing. These helpers serialize BY REFERENCE:

- top-level dataclass fields are enumerated dynamically (drift-proof: a new
  RunState field can never be silently dropped from durable bytes);
- the big payload fields (`vars`, `output`, ledger `result`/`effect`) pass
  by reference — no copy;
- nested dataclasses are converted lazily via the `default=` hook, which
  preserves the exact output `asdict` produced for the rare
  dataclass-inside-vars case instead of crashing on it.

Byte compatibility: for any state the old path could serialize, the new
path produces the same JSON values (str-subclass enums serialize as their
values either way); only inter-token whitespace changes (compact
separators). Readers parse JSON and are whitespace-blind; the hash chain
canonicalizes independently (`ledger_chain._canonical_json`), so chain
verification is unaffected.

Contract-violation honesty (2026-07-14 durability audit): if a non-owner
thread mutates the shared vars tree DURING a save — a violation of the
single-writer contract either way — the old `asdict` path sometimes raised
loudly (net-growth resize -> RuntimeError mid-copy) while the C encoder
here can emit parseable-but-torn JSON silently. Balanced insert+delete
tore silently under BOTH paths, so loudness was never a guarantee; but a
debugger of a torn snapshot should know the loud subcase is gone.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict

from ..core.models import RunState, StepRecord

__all__ = [
    "dataclass_json_default",
    "runstate_to_dict",
    "steprecord_to_dict",
    "dumps_compact",
]


def dataclass_json_default(obj: Any) -> Any:
    """`json.dumps(default=...)` hook: convert nested dataclass instances.

    `dataclasses.asdict` used to convert dataclasses nested anywhere inside
    vars/results before serialization; passing containers by reference means
    the encoder can now meet one directly. Converting here (lazily, only
    when actually encountered) keeps the durable bytes identical to the old
    behavior without paying the recursive deep copy on every write.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def runstate_to_dict(run: RunState) -> Dict[str, Any]:
    """Shallow, field-complete dict for durable RunState serialization.

    `vars`/`output` are passed BY REFERENCE (the whole point); `waiting` is
    the only nested dataclass field and is converted eagerly (it is tiny).
    Field enumeration via `dataclasses.fields` keeps this drift-proof.
    """
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(run):
        value = getattr(run, f.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        out[f.name] = value
    return out


def steprecord_to_dict(record: StepRecord) -> Dict[str, Any]:
    """Shallow, field-complete dict for durable StepRecord serialization.

    StepRecord has no nested dataclass fields (`effect`/`result` are plain
    dicts by the ledger-safety invariant); anything exotic nested inside
    them is handled by `dataclass_json_default` at encode time.
    """
    return {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}


def dumps_compact(obj: Any) -> str:
    """Compact JSON for durable writes (no indent, no separator spaces)."""
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        default=dataclass_json_default,
    )
