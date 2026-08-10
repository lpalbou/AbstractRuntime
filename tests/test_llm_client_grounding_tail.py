"""Prompt-prefix cache stability for the runtime grounding envelope (backlog 0212).

The `<runtime_metadata>` envelope carries a per-second timestamp. Before 0212 it was
re-injected into the LAST `user` message on every LLM call; in an agent tool loop the
last user message is the task at index 0, so byte 0 of the message list changed every
iteration and provider prompt caching could never hit.

Contract under test:
- Chat shape (last message IS the user turn): envelope still rides the head of that
  final user turn (fresh bytes anyway; earlier messages untouched).
- Tool-loop shape (messages continue past the last user turn): the envelope rides a
  TRAILING, envelope-only user message; earlier messages stay byte-identical.
- Double normalization (runtime ledger pass + LLM client pass) is idempotent.
- Language safety from the 2026-06-10 fix is preserved: the envelope stays
  temporal-only by default (no country/locale leakage) — enforced by existing tests in
  test_llm_client_system_context.py and re-checked here for the trailing placement.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

import abstractruntime.integrations.abstractcore.llm_client as llm_client
from abstractruntime.core.models import Effect, EffectType, RunState, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


_ENVELOPE_RE = re.compile(r"<runtime_metadata>.*?</runtime_metadata>", re.DOTALL)


def _grounding(second: int) -> Dict[str, Any]:
    return llm_client._mark_grounding_prompt_injected(
        {
            "local_datetime": f"2000-01-01T00:00:{second:02d}+01:00",
            "country": "FR",
            "display": f"[2000-01-01 00:00:{second:02d} FR]",
        },
        True,
    )


def _tool_loop_messages() -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "Create a project folder"},
        {
            "role": "assistant",
            "content": "Checking the workspace.",
            "tool_calls": [
                {"type": "function", "id": "call_1", "function": {"name": "list_files", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "[list_files]: a.txt", "tool_call_id": "call_1"},
    ]


def test_tool_loop_shape_appends_trailing_envelope_and_keeps_prefix_stable() -> None:
    original = _tool_loop_messages()
    baseline = json.dumps(original, sort_keys=True, ensure_ascii=False)

    _, out1 = llm_client._normalize_turn_grounding(prompt="", messages=original, grounding=_grounding(1))
    _, out2 = llm_client._normalize_turn_grounding(prompt="", messages=original, grounding=_grounding(2))

    assert isinstance(out1, list) and isinstance(out2, list)
    # One trailing message added; all original messages byte-identical (input not mutated either).
    assert len(out1) == len(original) + 1
    assert json.dumps(out1[: len(original)], sort_keys=True, ensure_ascii=False) == baseline
    assert json.dumps(original, sort_keys=True, ensure_ascii=False) == baseline

    # The envelope appears ONLY in the trailing message; message[0] carries none.
    assert "<runtime_metadata>" not in str(out1[0].get("content") or "")
    tail = out1[-1]
    assert tail.get("role") == "user"
    assert _ENVELOPE_RE.fullmatch(str(tail.get("content") or "").strip())

    # Per-call timestamp entropy is confined to the trailing message: the prefix of two
    # consecutive normalizations is byte-identical, only the tails differ.
    assert json.dumps(out1[:-1], sort_keys=True) == json.dumps(out2[:-1], sort_keys=True)
    assert out1[-1]["content"] != out2[-1]["content"]
    assert "00:00:01" in out1[-1]["content"]
    assert "00:00:02" in out2[-1]["content"]


def test_tool_loop_trailing_envelope_stays_temporal_only() -> None:
    """Language safety (2026-06-10 grounding fix): relocating the envelope must not
    reintroduce locale leakage — country/timezone/user stay out of the prompt."""
    _, out = llm_client._normalize_turn_grounding(
        prompt="",
        messages=_tool_loop_messages(),
        grounding={
            "local_datetime": "2026-06-10T15:29:16+02:00",
            "timezone": "Europe/Paris",
            "country": "FR",
            "user": "albou",
            "display": "[2026-06-10 15:29:16 FR]",
        },
    )
    assert isinstance(out, list)
    payload_match = re.search(r"<runtime_metadata>(?P<p>.*?)</runtime_metadata>", str(out[-1]["content"]), re.DOTALL)
    assert payload_match
    payload = json.loads(payload_match.group("p"))
    assert "local_datetime" in payload
    assert "country" not in payload
    assert "timezone" not in payload
    assert "user" not in payload
    assert payload.get("display") == "[2026-06-10 15:29:16]"


def test_chat_shape_still_injects_into_final_user_turn() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "what time is it?"},
    ]
    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(5))

    assert isinstance(out, list)
    assert len(out) == 3, "chat shape must not grow the message list"
    assert out[0] == {"role": "user", "content": "hello"}
    assert out[1] == {"role": "assistant", "content": "hi"}
    final = str(out[2]["content"])
    assert final.startswith("<runtime_metadata>")
    assert final.rstrip().endswith("what time is it?")


def test_double_normalization_is_idempotent_for_tool_loop_shape() -> None:
    """The runtime ledger pass and the LLM client pass both normalize the same payload;
    the second pass must refresh (not duplicate) the trailing envelope message."""
    original = _tool_loop_messages()
    _, once = llm_client._normalize_turn_grounding(prompt="", messages=original, grounding=_grounding(1))
    assert isinstance(once, list)
    _, twice = llm_client._normalize_turn_grounding(prompt="", messages=once, grounding=_grounding(2))
    assert isinstance(twice, list)

    assert len(twice) == len(original) + 1
    envelope_msgs = [m for m in twice if llm_client._is_runtime_grounding_only_user_message(m)]
    assert len(envelope_msgs) == 1
    assert "00:00:02" in envelope_msgs[0]["content"]


def test_grounding_none_removes_envelope_artifacts_and_strips_prefixes() -> None:
    """Media-only calls (grounding=None) must clean both envelope forms: the legacy
    head-of-user-turn prefix and the trailing envelope-only message."""
    messages = [
        {"role": "user", "content": '<runtime_metadata>{"local_datetime":"x"}</runtime_metadata>\nreal task'},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "[t]: out", "tool_call_id": "c1"},
        {"role": "user", "content": '<runtime_metadata>{"local_datetime":"y"}</runtime_metadata>'},
    ]
    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=None)

    assert isinstance(out, list)
    assert len(out) == 3
    assert all("<runtime_metadata>" not in str(m.get("content") or "") for m in out)
    assert out[0]["content"] == "real task"


def test_messages_without_user_turn_still_get_trailing_envelope() -> None:
    messages = [{"role": "assistant", "content": "hi"}]
    _, out = llm_client._normalize_turn_grounding(prompt="", messages=messages, grounding=_grounding(3))
    assert isinstance(out, list)
    assert len(out) == 2
    assert llm_client._is_runtime_grounding_only_user_message(out[-1])


def test_prompt_path_unchanged() -> None:
    prompt, messages = llm_client._normalize_turn_grounding(prompt="hello", messages=None, grounding=_grounding(9))
    assert prompt.startswith("<runtime_metadata>")
    assert prompt.rstrip().endswith("hello")
    assert messages is None


class _FakeCacheProvider:
    """Minimal local control-plane provider for `_maybe_prepare_prompt_cache`.

    `prompt_cache_key_meta` / `prompt_cache_update_key_meta` and the `forked_from`
    stamp inside `prompt_cache_fork` are NOT test conveniences: every real provider
    inherits them from `BaseProvider` (abstractcore `providers/base.py`, and
    `prompt_cache_fork` there does `meta.setdefault("forked_from", src)` itself). A
    fake without them reports "never forked" forever, which makes the fork-once
    contract untestable and would let a per-call clear pass unnoticed.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []
        self.meta: Dict[str, Dict[str, Any]] = {}

    def prompt_cache_key_meta(self, key: Any) -> Dict[str, Any]:
        return dict(self.meta.get(str(key)) or {})

    def prompt_cache_update_key_meta(self, key: Any, **updates: Any) -> bool:
        entry = self.meta.setdefault(str(key), {})
        for name, value in updates.items():
            if value is not None:
                entry[str(name)] = value
        return True

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_operation(self, operation: str) -> bool:
        return True

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Dict[str, Any]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        _ = (make_default, ttl_s, version)
        derived: List[Dict[str, Any]] = []
        prefix_seed = "seed"
        for m in modules:
            raw = json.dumps(m, sort_keys=True, ensure_ascii=False, default=str)
            module_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            prefix_seed = hashlib.sha256((prefix_seed + module_hash).encode("utf-8")).hexdigest()
            derived.append({"module_id": m.get("module_id"), "module_hash": module_hash, "cache_key": f"{namespace}:{prefix_seed[:16]}"})
        self.calls.append(("prepare_modules",))
        return {"supported": True, "namespace": namespace, "modules": derived, "final_cache_key": derived[-1]["cache_key"]}

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        self.calls.append(("clear", key))
        if key is None:
            self.meta.clear()
        else:
            self.meta.pop(str(key), None)
        return True

    def prompt_cache_fork(self, from_key: str, to_key: str, *, make_default: bool = False, ttl_s: Optional[float] = None, **kwargs: Any) -> bool:
        _ = (make_default, ttl_s, kwargs)
        self.calls.append(("fork", from_key, to_key))
        self.meta.setdefault(str(to_key), {}).setdefault("forked_from", str(from_key))
        return True

    def prompt_cache_update(
        self,
        key: str,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> bool:
        _ = (prompt, system_prompt, tools, add_generation_prompt, kwargs)
        self.calls.append(("update", key, [dict(m) for m in (messages or [])]))
        return True


def _cache_client(provider: _FakeCacheProvider) -> Any:
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient

    client = LocalAbstractCoreLLMClient.__new__(LocalAbstractCoreLLMClient)
    client._llm = provider  # type: ignore[attr-defined]
    client._prompt_cache_state_lock = threading.Lock()  # type: ignore[attr-defined]
    client._prompt_cache_state = {}  # type: ignore[attr-defined]
    return client


_SYS = "SYSTEM"
_TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]


def _grown_tool_loop() -> List[Dict[str, Any]]:
    return _tool_loop_messages() + [
        {
            "role": "assistant",
            "content": "Creating the folder.",
            "tool_calls": [{"type": "function", "id": "call_2", "function": {"name": "execute_command", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "[execute_command]: ok", "tool_call_id": "call_2"},
    ]


def test_prompt_cache_prepare_never_feeds_messages_into_the_durable_cache() -> None:
    """The per-call envelope must never enter the durable per-session KV cache.

    This method's whole message lane was removed (2026-08-02), which makes that
    guarantee STRUCTURAL rather than filtered: no message of any kind is handed to
    the control plane here, so no per-call envelope, no volatile adapter tail and no
    generation scaffolding can be baked into the session key by this seam. The
    transcript reaches the provider exactly once, through `generate()`, where
    `mlx_provider._prepare_cache_delta_feed` does token-level LCP against the key's
    fed-token record and feeds only the true suffix.
    """
    provider = _FakeCacheProvider()
    client = _cache_client(provider)

    key = "sess:loop"
    _, msgs1 = llm_client._normalize_turn_grounding(prompt="", messages=_tool_loop_messages(), grounding=_grounding(1))
    client._maybe_prepare_prompt_cache(prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=msgs1)

    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]
    assert not any(c[0] == "update" for c in provider.calls), (
        "the session key must be built from the (system, tools) bloc chain only; "
        "any transcript pushed here is per-call bytes entering durable KV"
    )


def test_prompt_cache_prepare_forks_once_and_never_reclears_the_session_key() -> None:
    """A growing tool-loop transcript must cost ONE fork per session, not one per turn.

    `prompt_cache_clear(key)` is not free on a local control plane: the MLX provider
    drops `_hybrid_snapshots[key]` with it, which is the exact state the untrimmable /
    hybrid lane restores from. Re-clearing mid-session therefore forces a full cold
    prefill on the next turn. The identity check reads the PROVIDER's `forked_from`
    meta, so it holds even when a later turn runs on a different client instance.
    """
    provider = _FakeCacheProvider()
    client = _cache_client(provider)
    key = "sess:loop"

    _, msgs1 = llm_client._normalize_turn_grounding(prompt="", messages=_tool_loop_messages(), grounding=_grounding(1))
    client._maybe_prepare_prompt_cache(prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=msgs1)
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]

    # Turn 2: transcript grew at the tail, a fresh envelope replaced the old one.
    _, msgs2 = llm_client._normalize_turn_grounding(prompt="", messages=_grown_tool_loop(), grounding=_grounding(2))
    provider.calls.clear()
    client._maybe_prepare_prompt_cache(prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=msgs2)
    assert [c[0] for c in provider.calls] == ["prepare_modules"]

    # Turn 3 on a FRESH client (the shape that defeated the old per-instance state
    # dict): still no clear, because the provider's own meta carries the identity.
    fresh_client = _cache_client(provider)
    _, msgs3 = llm_client._normalize_turn_grounding(prompt="", messages=_grown_tool_loop(), grounding=_grounding(3))
    provider.calls.clear()
    fresh_client._maybe_prepare_prompt_cache(prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=msgs3)
    assert [c[0] for c in provider.calls] == ["prepare_modules"]


def test_prompt_cache_prepare_reforks_when_the_prefix_identity_changes() -> None:
    """Fork-once is conditional on IDENTITY, not unconditional: a genuinely different
    (system, tools) prefix produces a different final bloc key, and the session key
    must be re-forked from it. Without this the shortcut above would silently serve a
    session off a stale persona/tool set."""
    provider = _FakeCacheProvider()
    client = _cache_client(provider)
    key = "sess:reprefix"

    client._maybe_prepare_prompt_cache(prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=_tool_loop_messages())
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]

    provider.calls.clear()
    client._maybe_prepare_prompt_cache(
        prompt_cache_key=key, system_prompt="A DIFFERENT SYSTEM", tools=_TOOLS, messages=_tool_loop_messages()
    )
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]


def test_prompt_cache_prepare_is_indifferent_to_a_changed_volatile_tail() -> None:
    """B1 (code seat c971, runtime half): an adapter tail marked `volatile: true`
    changes every cycle. It must never force a clear+fork+re-prefill per iteration.

    The original structural EXCLUSION lived in this method's message lane; with that
    lane gone the guarantee is stronger — the tail is not merely filtered out here,
    it never had a path into the durable key at all, so no tail rewrite can be
    mistaken for a prefix divergence at this seam. The rewritten-tail case is
    exercised end-to-end against the real delta feed by abstractcore's
    `tests/providers/test_mlx_hybrid_snapshot_lane.py`.
    """
    provider = _FakeCacheProvider()
    client = _cache_client(provider)
    key = "sess:volatile"

    transcript = _tool_loop_messages()
    tail1 = {"role": "user", "content": "[loop] iteration 1 of 20.", "volatile": True}
    client._maybe_prepare_prompt_cache(
        prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=transcript + [tail1]
    )
    assert [c[0] for c in provider.calls] == ["prepare_modules", "clear", "fork"]

    tail2 = {"role": "user", "content": "[loop] iteration 2 of 20.", "volatile": True}
    provider.calls.clear()
    client._maybe_prepare_prompt_cache(
        prompt_cache_key=key, system_prompt=_SYS, tools=_TOOLS, messages=_grown_tool_loop() + [tail2]
    )
    assert [c[0] for c in provider.calls] == ["prepare_modules"]


def test_volatile_marker_is_stripped_before_the_provider() -> None:
    """The `volatile` key is runtime-internal: the MESSAGE still reaches the
    provider (it carries real per-call text) but the marker key never does
    (unknown fields 400 on strict provider SDKs); durable caller structures
    are never mutated by the strip."""
    strip = llm_client._strip_volatile_markers
    tail = {"role": "user", "content": "[loop] iteration 3 of 20.", "volatile": True}
    msgs = [{"role": "user", "content": "task"}, tail]
    out = strip(msgs)
    assert out is not None
    assert len(out) == 2
    assert out[1]["content"] == "[loop] iteration 3 of 20."
    assert "volatile" not in out[1]
    assert tail["volatile"] is True  # caller's dict untouched
    # No markers -> the exact same list object (zero-copy fast path).
    plain = [{"role": "user", "content": "task"}]
    assert strip(plain) is plain


def _run_tool_loop_workflow_once(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Drive one LLM_CALL with tool-loop shaped messages through the real Runtime so the
    ledger grounding pass (`_maybe_inject_llm_call_grounding_for_ledger`) applies."""
    monkeypatch.setattr(
        llm_client,
        "_runtime_grounding_metadata",
        lambda trace_metadata=None: {
            "local_datetime": "2000-01-01T00:00:00+00:00",
            "country": "FR",
            "display": "[2000-01-01 00:00:00 FR]",
            "source": "abstractruntime",
            "prompt_injected": False,
        },
    )

    captured: Dict[str, Any] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        _ = (run, default_next_node)
        captured["payload"] = dict(effect.payload or {})
        return EffectOutcome.completed({"content": "ok", "tool_calls": []})

    def node(run: RunState, ctx: Any) -> StepPlan:
        _ = (run, ctx)
        return StepPlan(
            node_id="n1",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={"prompt": "", "messages": _tool_loop_messages()},
            ),
            next_node=None,
        )

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    wf = WorkflowSpec(workflow_id="wf-grounding-tail", entry_node="n1", nodes={"n1": node})
    run_id = runtime.start(workflow=wf)
    runtime.tick(workflow=wf, run_id=run_id)
    return captured


def test_runtime_ledger_grounding_rides_trailing_message_for_tool_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _run_tool_loop_workflow_once(monkeypatch)

    payload = captured.get("payload")
    assert isinstance(payload, dict)
    msgs = payload.get("messages")
    assert isinstance(msgs, list)
    assert len(msgs) == len(_tool_loop_messages()) + 1

    # message[0] (the task) is byte-identical to the durable transcript: no envelope prefix.
    assert msgs[0] == _tool_loop_messages()[0]
    envelope_positions = [i for i, m in enumerate(msgs) if "<runtime_metadata>" in str(m.get("content") or "")]
    assert envelope_positions == [len(msgs) - 1]
    assert llm_client._is_runtime_grounding_only_user_message(msgs[-1])
