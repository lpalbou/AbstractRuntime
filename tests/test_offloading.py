from __future__ import annotations

from pathlib import Path

import pytest

from abstractruntime import (
    FileArtifactStore,
    InMemoryArtifactStore,
    JsonFileRunStore,
    JsonlLedgerStore,
    OffloadingLedgerStore,
    OffloadingRunStore,
    Runtime,
    StepPlan,
    WorkflowSpec,
    get_artifact_id,
    is_artifact_ref,
    offload_large_values,
    resolve_artifact,
)


@pytest.mark.basic
def test_offload_large_values_offloads_large_string() -> None:
    store = InMemoryArtifactStore()
    large = "x" * 10_000

    out = offload_large_values(
        {"a": large},
        artifact_store=store,
        run_id="run-1",
        max_inline_bytes=1024,
        root_path="root",
        allow_offload=lambda path, v: True,
        allow_root_replace=False,
    )

    assert isinstance(out, dict)
    assert is_artifact_ref(out["a"])
    art = resolve_artifact(out["a"], store)
    assert art is not None
    assert art.as_text() == large


@pytest.mark.basic
def test_offload_large_values_does_not_replace_root_by_default() -> None:
    store = InMemoryArtifactStore()
    large = "x" * 10_000

    out = offload_large_values(
        large,
        artifact_store=store,
        run_id="run-1",
        max_inline_bytes=1024,
        root_path="root",
        allow_offload=lambda path, v: True,
        allow_root_replace=False,
    )

    assert out == large
    assert store.list_all() == []


@pytest.mark.integration
def test_offloading_stores_restart_simulation(tmp_path: Path) -> None:
    base = tmp_path / "runtime"
    base.mkdir(parents=True, exist_ok=True)

    max_inline_bytes = 1024
    large = "x" * 20_000

    artifact_store = FileArtifactStore(base)
    run_store = OffloadingRunStore(JsonFileRunStore(base), artifact_store=artifact_store, max_inline_bytes=max_inline_bytes)
    ledger_store = OffloadingLedgerStore(JsonlLedgerStore(base), artifact_store=artifact_store, max_inline_bytes=max_inline_bytes)

    def node(run, ctx):
        return StepPlan(node_id="n", complete_output={"text": large})

    workflow = WorkflowSpec(workflow_id="w", entry_node="n", nodes={"n": node})
    rt = Runtime(run_store=run_store, ledger_store=ledger_store, artifact_store=artifact_store)

    run_id = rt.start(workflow=workflow)
    state = rt.tick(workflow=workflow, run_id=run_id)
    assert state.output is not None

    run_path = base / f"run_{run_id}.json"
    ledger_path = base / f"ledger_{run_id}.jsonl"
    run_bytes = run_path.read_bytes()
    ledger_bytes = ledger_path.read_bytes()
    assert large.encode("utf-8") not in run_bytes
    assert large.encode("utf-8") not in ledger_bytes

    # Restart simulation: new store instances, load persisted refs, resolve artifacts.
    artifact_store2 = FileArtifactStore(base)
    run_store2 = OffloadingRunStore(JsonFileRunStore(base), artifact_store=artifact_store2, max_inline_bytes=max_inline_bytes)
    ledger_store2 = OffloadingLedgerStore(JsonlLedgerStore(base), artifact_store=artifact_store2, max_inline_bytes=max_inline_bytes)
    rt2 = Runtime(run_store=run_store2, ledger_store=ledger_store2, artifact_store=artifact_store2)

    persisted = rt2.get_state(run_id)
    assert isinstance(persisted.output, dict)
    text_val = persisted.output.get("text")
    assert is_artifact_ref(text_val)

    art_id = get_artifact_id(text_val)
    art = artifact_store2.load(art_id)
    assert art is not None
    assert art.as_text() == large

