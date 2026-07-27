"""The 401-incident chain's two runtime links (code-tui c4978, ledger-verified).

R1 — leaf-wise offload: a 4KB answer must never offload because it shares
the output dict with a 215KB scratchpad. The offloader reduces LARGEST
CHILDREN FIRST (by size, never a name list); whole-subtree replacement is
the last resort only.

R2 — bounded answer resolve: root-replaced outputs from the EXISTING corpus
(the all-or-nothing era) resolve at session-history extraction — offload-
minted refs only, size-capped, answer extraction only — so server-side
session replay stops forgetting exactly the turns that most need replaying.
"""

import json
from typing import Any, Dict

from abstractruntime.core.models import RunState, RunStatus
from abstractruntime.session_history import session_chat_messages
from abstractruntime.storage.artifacts import InMemoryArtifactStore, is_artifact_ref
from abstractruntime.storage.in_memory import InMemoryRunStore
from abstractruntime.storage.offloading import offload_large_values


CAP = 256 * 1024


def _incident_output() -> Dict[str, Any]:
    """The ledger-verified shape: answer 4KB, report 101KB, messages 130KB,
    scratchpad 215KB — no single leaf over the 256KB cap, whole dict 445KB."""
    return {
        "response": "A" * 4_000,
        "report": "R" * 101_000,
        "messages": ["m" * 1_000] * 130,
        "scratchpad": "S" * 215_000,
    }


class TestLeafwiseOffload:
    def test_small_answer_stays_inline_when_dict_crosses_cap(self):
        store = InMemoryArtifactStore()
        out = offload_large_values(
            _incident_output(),
            artifact_store=store,
            run_id="run-1",
            max_inline_bytes=CAP,
            base_tags={"source": "run_output_offload"},
            root_path="output",
            allow_root_replace=True,
        )
        # The root survives as a dict; the ANSWER is inline verbatim.
        assert isinstance(out, dict) and not is_artifact_ref(out)
        assert out["response"] == "A" * 4_000
        # The biggest child offloaded; the remainder fits the cap.
        assert is_artifact_ref(out["scratchpad"])
        assert len(json.dumps(out, separators=(",", ":")).encode()) <= CAP

    def test_reduction_is_by_size_never_by_name(self):
        """A huge value named `response` offloads; a small `scratchpad`
        stays — the discriminator is bytes, not vocabulary."""
        store = InMemoryArtifactStore()
        doc = {"response": "A" * 300_000, "scratchpad": "s" * 100}
        out = offload_large_values(
            doc, artifact_store=store, run_id="run-2",
            max_inline_bytes=CAP, root_path="output", allow_root_replace=True,
        )
        assert isinstance(out, dict)
        # The 300KB string crossed the cap as a LEAF already; either way the
        # small field survives inline and the root is not replaced.
        assert out["scratchpad"] == "s" * 100
        assert is_artifact_ref(out["response"])

    def test_root_replace_remains_the_last_resort(self):
        """When no child may be offloaded, the whole subtree still replaces
        (the machine never persists an over-cap document silently)."""
        store = InMemoryArtifactStore()
        doc = {f"k{i}": "x" * 10_000 for i in range(60)}  # ~600KB, all denied

        def deny_children(path: str, v: Any) -> bool:
            return path == "output"  # only the root itself may offload

        out = offload_large_values(
            doc, artifact_store=store, run_id="run-3",
            max_inline_bytes=CAP, root_path="output",
            allow_offload=deny_children, allow_root_replace=True,
        )
        assert is_artifact_ref(out)


def _seed_completed_run(run_store, run_id: str, session_id: str, output: Any) -> None:
    run = RunState.new(
        workflow_id="chat",
        entry_node="START",
        vars={"prompt": "what is the plan?", "_meta": {"session_id": session_id}},
        actor_id="user",
        session_id=session_id,
    )
    run.run_id = run_id
    run.status = RunStatus.COMPLETED
    run.output = output
    run_store.save(run)


class TestBoundedAnswerResolve:
    def test_root_replaced_output_resolves_to_its_answer(self):
        artifact_store = InMemoryArtifactStore()
        run_store = InMemoryRunStore()
        # The all-or-nothing era: the WHOLE output rests as one artifact.
        doc = _incident_output()
        meta = artifact_store.store(
            json.dumps(doc).encode("utf-8"),
            content_type="application/json",
            run_id="run-old",
            tags={"source": "run_output_offload", "kind": "json"},
        )
        _seed_completed_run(
            run_store, "run-old", "sess-401", {"$artifact": meta.artifact_id}
        )
        messages = session_chat_messages(
            run_store=run_store, artifact_store=artifact_store, session_id="sess-401"
        )
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"].startswith("AAAA")

    def test_foreign_refs_stay_skipped(self):
        """Handler-authored artifact currency is NOT resolved — only refs
        the run-output offloader minted."""
        artifact_store = InMemoryArtifactStore()
        run_store = InMemoryRunStore()
        meta = artifact_store.store(
            json.dumps({"response": "secret"}).encode("utf-8"),
            content_type="application/json",
            run_id="run-f",
            tags={"source": "media_handoff", "kind": "json"},
        )
        _seed_completed_run(run_store, "run-f", "sess-f", {"$artifact": meta.artifact_id})
        assert session_chat_messages(
            run_store=run_store, artifact_store=artifact_store, session_id="sess-f"
        ) == []

    def test_oversized_artifacts_stay_skipped(self):
        artifact_store = InMemoryArtifactStore()
        run_store = InMemoryRunStore()
        big = {"response": "B" * (5 * 1024 * 1024)}
        meta = artifact_store.store(
            json.dumps(big).encode("utf-8"),
            content_type="application/json",
            run_id="run-big",
            tags={"source": "run_output_offload", "kind": "json"},
        )
        _seed_completed_run(run_store, "run-big", "sess-big", {"$artifact": meta.artifact_id})
        assert session_chat_messages(
            run_store=run_store, artifact_store=artifact_store, session_id="sess-big"
        ) == []
