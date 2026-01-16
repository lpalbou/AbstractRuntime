from __future__ import annotations

from typing import Any, Dict, List, Optional


def test_llm_call_media_resolves_artifact_refs_to_temp_files() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    store = InMemoryArtifactStore()
    meta = store.store(b"hello world", content_type="text/plain", run_id="r1")

    class DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate(
            self,
            *,
            prompt: str,
            messages: Optional[List[Dict[str, str]]] = None,
            system_prompt: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            media: Optional[List[Any]] = None,
            params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            self.calls.append(
                {
                    "prompt": prompt,
                    "messages": messages,
                    "system_prompt": system_prompt,
                    "tools": tools,
                    "media": media,
                    "params": dict(params or {}),
                }
            )
            assert isinstance(params, dict)
            assert params.get("glyph_compression") == "never"
            assert isinstance(media, list) and media, "Expected media to be passed"
            p = media[0]
            assert isinstance(p, str) and p, "Expected media item to be a temp file path"
            with open(p, "rb") as f:
                assert f.read() == b"hello world"
            return {"content": "OK", "metadata": {}}

    llm = DummyLLM()
    handler = make_llm_call_handler(llm=llm, artifact_store=store)

    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "Hello",
            "media": [{"$artifact": meta.artifact_id, "filename": "notes.txt"}],
            "params": {},
        },
        result_key="_temp.llm",
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"
    assert llm.calls, "Expected LLM to be called"

    # Ensure we persist only the JSON-safe attachment refs (not temp paths).
    assert isinstance(outcome.result, dict)
    meta_out = outcome.result.get("metadata")
    assert isinstance(meta_out, dict)
    obs = meta_out.get("_runtime_observability")
    assert isinstance(obs, dict)
    kwargs = obs.get("llm_generate_kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("media") == [{"$artifact": meta.artifact_id, "filename": "notes.txt"}]


def test_llm_call_media_preserves_explicit_glyph_compression_param() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    store = InMemoryArtifactStore()
    meta = store.store(b"hello world", content_type="text/plain", run_id="r1")

    class DummyLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate(self, *, params=None, media=None, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"params": dict(params or {}), "media": media})
            assert isinstance(params, dict)
            assert params.get("glyph_compression") == "auto"
            assert isinstance(media, list) and media
            return {"content": "OK", "metadata": {}}

    handler = make_llm_call_handler(llm=DummyLLM(), artifact_store=store)
    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={
            "prompt": "Hello",
            "media": [{"$artifact": meta.artifact_id, "filename": "notes.txt"}],
            "params": {"glyph_compression": "auto"},
        },
        result_key="_temp.llm",
    )

    outcome = handler(run, effect, None)
    assert outcome.status == "completed"


def test_llm_call_media_missing_artifact_fails_cleanly() -> None:
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_llm_call_handler
    from abstractruntime.storage.artifacts import InMemoryArtifactStore

    store = InMemoryArtifactStore()

    class DummyLLM:
        def generate(self, **kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

    handler = make_llm_call_handler(llm=DummyLLM(), artifact_store=store)
    run = RunState.new(workflow_id="wf", entry_node="node-1", vars={})
    run.current_node = "node-1"

    effect = Effect(
        type=EffectType.LLM_CALL,
        payload={"prompt": "Hello", "media": [{"$artifact": "missing"}], "params": {}},
        result_key="_temp.llm",
    )
    outcome = handler(run, effect, None)
    assert outcome.status == "failed"
    assert isinstance(outcome.error, str)
    assert "not found" in outcome.error.lower()
