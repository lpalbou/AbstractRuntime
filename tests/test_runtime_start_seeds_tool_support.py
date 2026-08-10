from __future__ import annotations

from abstractruntime.core.config import RuntimeConfig
from abstractruntime.core.models import RunState, StepPlan
from abstractruntime.core.runtime import Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_runtime_start_seeds_tool_support_and_supports_native_tools_from_model_capabilities() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        config=RuntimeConfig(
            provider="lmstudio",
            model="qwen/qwen3-next-80b",
            model_capabilities={"tool_support": "native"},
        ),
    )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="seed_tool_support", entry_node="done", nodes={"done": done_node})
    run_id = rt.start(workflow=wf, vars={"context": {}, "_runtime": {}})

    state = run_store.load(run_id)
    assert state is not None
    runtime_ns = state.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    assert runtime_ns.get("tool_support") == "native"
    assert runtime_ns.get("supports_native_tools") is True
    # Capability-bit provenance (abstractagent wave-F P1, commons c1801):
    # the bits describe the CONFIG model; stamping WHICH model lets routed
    # runs (`_runtime.model` pointing elsewhere) detect staleness and fail
    # toward the safe posture instead of trusting a mismatched bit.
    assert runtime_ns.get("tool_support_model") == "qwen/qwen3-next-80b"


def test_capability_bits_are_withheld_when_they_describe_another_model() -> None:
    """Provenance only helps a consumer that checks it.

    The config model is native; the run routes to a different model whose
    tool support is unknown here. Stamping `supports_native_tools=True` would
    hand every consumer a confident bit derived from a model this run will
    never call -- the exact shape that made CodeAct complete silently with
    code-as-prose. Withhold the bits so capabilities get resolved for the REAL
    model (or fall back to the safe prompted posture), and say plainly which
    model the stamp does not cover.
    """
    run_store = InMemoryRunStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        config=RuntimeConfig(
            provider="openai-compatible",
            model="gpt-5.4",
            model_capabilities={"tool_support": "native"},
        ),
    )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="seed_caps_mismatch", entry_node="done", nodes={"done": done_node})
    run_id = rt.start(workflow=wf, vars={"provider": "lmstudio", "model": "qwen/qwen3-4b"})

    state = run_store.load(run_id)
    assert state is not None
    runtime_ns = state.vars.get("_runtime")
    assert runtime_ns.get("model") == "qwen/qwen3-4b"
    # Provenance is still recorded: the bits came from gpt-5.4.
    assert runtime_ns.get("tool_support_model") == "gpt-5.4"
    # ...but the bits themselves are NOT asserted for the routed model.
    assert "tool_support" not in runtime_ns
    assert "supports_native_tools" not in runtime_ns
    assert runtime_ns.get("tool_support_stale_for_model") == "qwen/qwen3-4b"


def test_capability_bits_are_seeded_when_the_route_matches_the_config_model() -> None:
    """The common case is unchanged: same model, bits apply."""
    run_store = InMemoryRunStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        config=RuntimeConfig(
            provider="lmstudio",
            model="qwen/qwen3.6-35b-a3b",
            model_capabilities={"tool_support": "native"},
        ),
    )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="seed_caps_match", entry_node="done", nodes={"done": done_node})
    run_id = rt.start(workflow=wf, vars={"provider": "lmstudio", "model": "qwen/qwen3.6-35b-a3b"})

    state = run_store.load(run_id)
    runtime_ns = state.vars.get("_runtime")
    assert runtime_ns.get("tool_support") == "native"
    assert runtime_ns.get("supports_native_tools") is True
    assert "tool_support_stale_for_model" not in runtime_ns


def test_caller_supplied_tool_support_metadata_is_never_clobbered() -> None:
    """setdefault semantics: a host that already routed/stamped its own
    tool-support metadata (e.g. the gateway resolving per-run models) keeps
    its values; start() only fills gaps."""
    run_store = InMemoryRunStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=InMemoryLedgerStore(),
        config=RuntimeConfig(
            provider="lmstudio",
            model="config-model",
            model_capabilities={"tool_support": "native"},
        ),
    )

    def done_node(run: RunState, ctx) -> StepPlan:
        return StepPlan(node_id="done", complete_output={"ok": True})

    wf = WorkflowSpec(workflow_id="seed_no_clobber", entry_node="done", nodes={"done": done_node})
    run_id = rt.start(
        workflow=wf,
        vars={
            "context": {},
            "_runtime": {
                "tool_support": "prompted",
                "supports_native_tools": False,
                "tool_support_model": "routed-model",
            },
        },
    )

    state = run_store.load(run_id)
    assert state is not None
    runtime_ns = state.vars.get("_runtime")
    assert runtime_ns.get("tool_support") == "prompted"
    assert runtime_ns.get("supports_native_tools") is False
    assert runtime_ns.get("tool_support_model") == "routed-model"



