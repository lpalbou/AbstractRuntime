from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from abstractruntime import Runtime, RunStatus
from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor
from abstractruntime.integrations.abstractmemory import build_memory_kg_effect_handlers
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.artifacts import InMemoryArtifactStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore
from abstractruntime.visualflow_compiler.compiler import compile_visualflow


def _repo_root() -> Path:
    # .../abstractruntime/tests/<file>.py -> repo root is 2 levels up.
    return Path(__file__).resolve().parents[2]


def _load_flow(flow_id: str) -> dict:
    p = _repo_root() / "abstractflow" / "web" / "flows" / f"{flow_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


class _ConstantEmbedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Constant, non-zero vector => deterministic cosine similarity (all ~1.0).
        return [[1.0, 0.0, 0.0] for _ in texts]


class _StubLLM:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        params = kwargs.get("params") if isinstance(kwargs, dict) else None
        params = params if isinstance(params, dict) else {}

        # Structured-output call: `ltm-ai-kg-extract-triples`
        if "response_model" in params:
            ctx = "Ada's goal is to find the City of Glass."
            return {
                "data": {
                    "assertions": [
                        {
                            "subject": "ex:person-ada",
                            "predicate": "schema:name",
                            "object": "Ada",
                            "attributes": {"evidence_quote": "Ada", "original_context": ctx},
                        },
                        {
                            "subject": "ex:claim-ada-goal",
                            "predicate": "rdf:type",
                            "object": "cito:Claim",
                            "attributes": {"evidence_quote": ctx, "original_context": ctx},
                        },
                        {
                            "subject": "ex:claim-ada-goal",
                            "predicate": "skos:definition",
                            "object": ctx,
                            "attributes": {"evidence_quote": ctx, "original_context": ctx},
                        },
                        {
                            "subject": "ex:concept-city-of-glass",
                            "predicate": "rdf:type",
                            "object": "skos:Concept",
                            "attributes": {"evidence_quote": "City of Glass", "original_context": ctx},
                        },
                        {
                            "subject": "ex:concept-city-of-glass",
                            "predicate": "skos:prefLabel",
                            "object": "City of Glass",
                            "attributes": {"evidence_quote": "City of Glass", "original_context": ctx},
                        },
                        {
                            "subject": "ex:claim-ada-goal",
                            "predicate": "schema:about",
                            "object": "ex:concept-city-of-glass",
                            "attributes": {"evidence_quote": ctx, "original_context": ctx},
                        },
                    ]
                },
                "content": "",
                "metadata": {},
            }

        # Normal chat call (the contract we care about): KG Active Memory is injected.
        return {"content": "ok", "metadata": {}}


def test_kg_ingest_is_idempotent_and_recall_injects_active_memory() -> None:
    from abstractmemory.in_memory_store import InMemoryTripleStore
    from abstractmemory.store import TripleQuery
    from abstractruntime.core.models import RunState, RunStatus as _RS, WaitReason

    llm = _StubLLM()
    kg_store = InMemoryTripleStore(embedder=_ConstantEmbedder())
    artifacts = InMemoryArtifactStore()

    # Workflows: ingest-turn (calls extract-triples) + a minimal llm_call with KG recall enabled.
    extract_spec = compile_visualflow(_load_flow("ltm-ai-kg-extract-triples"))
    ingest_spec = compile_visualflow(_load_flow("ltm-ai-kg-ingest-turn"))

    kg_chat_spec = compile_visualflow(
        {
            "id": "kg-chat-test",
            "name": "kg-chat-test",
            "entryNode": "start",
            "nodes": [
                {
                    "id": "start",
                    "type": "on_flow_start",
                    "data": {
                        "outputs": [
                            {"id": "exec-out", "type": "execution"},
                            {"id": "prompt", "type": "string"},
                            {"id": "provider", "type": "provider"},
                            {"id": "model", "type": "model"},
                        ]
                    },
                },
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {
                        "effectConfig": {"provider": "lmstudio", "model": "unit-test-model", "temperature": 0.0},
                        "pinDefaults": {"use_kg_memory": True, "memory_scope": "session", "recall_level": "standard"},
                    },
                },
            ],
            "edges": [
                {"source": "start", "sourceHandle": "exec-out", "target": "call", "targetHandle": "exec-in"},
                {"source": "start", "sourceHandle": "prompt", "target": "call", "targetHandle": "prompt"},
                {"source": "start", "sourceHandle": "provider", "target": "call", "targetHandle": "provider"},
                {"source": "start", "sourceHandle": "model", "target": "call", "targetHandle": "model"},
            ],
        }
    )

    reg = WorkflowRegistry()
    reg.register(extract_spec)
    reg.register(ingest_spec)
    reg.register(kg_chat_spec)

    with tempfile.TemporaryDirectory() as d:
        run_store = JsonFileRunStore(Path(d))
        ledger_store = JsonlLedgerStore(Path(d))

        handlers = {}
        handlers.update(build_effect_handlers(llm=llm, tools=MappingToolExecutor.from_tools([]), artifact_store=artifacts, run_store=run_store))
        handlers.update(build_memory_kg_effect_handlers(store=kg_store, run_store=run_store, now_iso=lambda: "2026-01-25T00:00:00Z"))

        rt = Runtime(
            run_store=run_store,
            ledger_store=ledger_store,
            workflow_registry=reg,
            artifact_store=artifacts,
            effect_handlers=handlers,
        )

        def _drive_to_completion(*, workflow, run_id: str, max_steps: int = 500) -> RunState:
            """Drive a run to completion, including async subworkflow waits (FlowRunner-style)."""
            cur = rt.get_state(run_id)
            safety = 0
            while safety < 50:
                safety += 1
                cur = rt.tick(workflow=workflow, run_id=run_id, max_steps=max_steps)
                if cur.status != _RS.WAITING:
                    return cur
                if cur.waiting is None or cur.waiting.reason != WaitReason.SUBWORKFLOW:
                    return cur

                details = cur.waiting.details if isinstance(cur.waiting.details, dict) else {}
                sub_run_id = details.get("sub_run_id")
                sub_wf_id = details.get("sub_workflow_id")
                if not isinstance(sub_run_id, str) or not sub_run_id.strip():
                    return cur
                if not isinstance(sub_wf_id, str) or not sub_wf_id.strip():
                    return cur
                sub_spec = reg.get_or_raise(sub_wf_id.strip())

                child = _drive_to_completion(workflow=sub_spec, run_id=sub_run_id.strip(), max_steps=max_steps)
                payload = {"sub_run_id": sub_run_id, "output": child.output, "node_traces": rt.get_node_traces(sub_run_id)}
                cur = rt.resume(
                    workflow=workflow,
                    run_id=run_id,
                    wait_key=cur.waiting.wait_key,
                    payload=payload,
                    max_steps=max_steps,
                )
                if cur.status != _RS.WAITING:
                    return cur

            return cur

        session_id = "session-test-kg"
        transcript = (
            "USER: Please remember this.\n\n"
            "Ada's goal is to find the City of Glass.\n\n"
            "ASSISTANT: Understood. Ada's goal is to find the City of Glass."
        )

        # Turn 1: ingest transcript -> assertions stored.
        run1 = rt.start(workflow=ingest_spec, session_id=session_id, vars={"text": transcript})
        state1 = _drive_to_completion(workflow=ingest_spec, run_id=run1)
        assert state1.status == RunStatus.COMPLETED

        owner_id = f"session_memory_{session_id}"
        items1 = kg_store.query(TripleQuery(scope="session", owner_id=owner_id, limit=0))
        assert len(items1) >= 3

        # Turn 1 replay: ingestion is idempotent (no duplicate triples inserted).
        run1b = rt.start(workflow=ingest_spec, session_id=session_id, vars={"text": transcript})
        state1b = _drive_to_completion(workflow=ingest_spec, run_id=run1b)
        assert state1b.status == RunStatus.COMPLETED

        items2 = kg_store.query(TripleQuery(scope="session", owner_id=owner_id, limit=0))
        assert len(items2) == len(items1)

        # Turn 2: a KG-aware LLM call should receive KG Active Memory in the system prompt.
        run2 = rt.start(
            workflow=kg_chat_spec,
            session_id=session_id,
            vars={"prompt": "What is Ada's goal?", "provider": "lmstudio", "model": "unit-test-model"},
        )
        state2 = _drive_to_completion(workflow=kg_chat_spec, run_id=run2)
        assert state2.status == RunStatus.COMPLETED

        chat_calls = []
        for c in llm.calls:
            params = c.get("params") if isinstance(c, dict) else None
            params = params if isinstance(params, dict) else {}
            if "response_model" in params:
                continue
            chat_calls.append(c)

        assert chat_calls, "Expected at least one non-structured (chat) LLM call"
        sys_prompt = chat_calls[-1].get("system_prompt") or ""
        assert isinstance(sys_prompt, str)
        assert "## KG ACTIVE MEMORY" in sys_prompt
        assert "city of glass" in sys_prompt.lower()
