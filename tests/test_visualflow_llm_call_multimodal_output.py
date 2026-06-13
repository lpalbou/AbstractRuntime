from __future__ import annotations


def test_visualflow_llm_call_uses_effect_config_output_request() -> None:
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "vf-mm-output",
            "name": "vf-mm-output",
            "entryNode": "call",
            "nodes": [
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {
                        "effectConfig": {
                            "provider": "lmstudio",
                            "model": "unit-test-model",
                            "temperature": 0.0,
                            "output": {"modality": "image", "format": "png"},
                        }
                    },
                }
            ],
            "edges": [],
        }
    )

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="call",
        vars={"_last_output": {"prompt": "Draw a red cube."}},
    )

    plan = spec.get_node("call")(run, None)

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    assert payload.get("prompt") == "Draw a red cube."
    assert payload.get("output") == {"modality": "image", "format": "png"}


def test_visualflow_llm_call_output_pin_overrides_effect_config_output_request() -> None:
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler import compile_visualflow

    spec = compile_visualflow(
        {
            "id": "vf-mm-output-override",
            "name": "vf-mm-output-override",
            "entryNode": "call",
            "nodes": [
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {
                        "pinDefaults": {"output": {"modality": "voice", "voice": "coral", "format": "wav"}},
                        "effectConfig": {
                            "provider": "lmstudio",
                            "model": "unit-test-model",
                            "output": {"modality": "image", "format": "png"},
                        },
                    },
                }
            ],
            "edges": [],
        }
    )

    run = RunState.new(
        workflow_id=spec.workflow_id,
        entry_node="call",
        vars={"_last_output": {"prompt": "Say hello."}},
    )

    plan = spec.get_node("call")(run, None)

    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    assert payload.get("output") == {"modality": "voice", "voice": "coral", "format": "wav"}


def test_visualflow_llm_call_media_result_sync_exposes_generated_output_fields() -> None:
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-mm-sync",
            "name": "vf-mm-sync",
            "entryNode": "call",
            "nodes": [
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {
                        "effectConfig": {"provider": "lmstudio", "model": "unit-test-model"},
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "response", "label": "response", "type": "string"},
                            {"id": "outputs", "label": "outputs", "type": "object"},
                            {"id": "resources", "label": "resources", "type": "object"},
                            {"id": "artifact_ref", "label": "artifact_ref", "type": "object"},
                            {"id": "artifact_id", "label": "artifact_id", "type": "string"},
                        ],
                    },
                }
            ],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="call")
    artifact_ref = {"$artifact": "art-img", "artifact_id": "art-img", "content_type": "image/png"}
    generated = {
        "content": None,
        "outputs": {
            "image": [
                {
                    "modality": "image",
                    "task": "image_generation",
                    "artifact_id": "art-img",
                    "artifact_ref": artifact_ref,
                    "content_type": "image/png",
                }
            ]
        },
        "resources": {"image": [{"resource_type": "artifact", "artifact_ref": artifact_ref}]},
        "metadata": {"trace_id": "tr-mm"},
    }
    run.vars["_temp"] = {"effects": {"call": generated}}

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["call"]  # type: ignore[attr-defined]
    assert out["response"] == ""
    assert out["outputs"] == generated["outputs"]
    assert out["resources"] == generated["resources"]
    assert out["artifact_id"] == "art-img"
    assert out["artifact_ref"] == artifact_ref
    assert out["meta"]["output_mode"] == "media"
    assert out["meta"]["trace"] == {"trace_id": "tr-mm"}


def test_visualflow_llm_call_structured_result_sync_exposes_data_pin() -> None:
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-llm-structured-sync",
            "name": "vf-llm-structured-sync",
            "entryNode": "call",
            "nodes": [
                {
                    "id": "call",
                    "type": "llm_call",
                    "data": {
                        "nodeType": "llm_call",
                        "pinDefaults": {
                            "resp_schema": {
                                "type": "object",
                                "properties": {
                                    "choice": {"type": "string", "enum": ["summarize", "classify"]},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["choice"],
                            }
                        },
                        "outputs": [
                            {"id": "exec-out", "label": "", "type": "execution"},
                            {"id": "response", "label": "response", "type": "string"},
                            {"id": "data", "label": "data", "type": "object"},
                            {"id": "meta", "label": "meta", "type": "object"},
                        ],
                    },
                }
            ],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="call")
    structured_data = {"choice": "summarize", "confidence": 0.95}
    run.vars["_temp"] = {
        "effects": {
            "call": {
                "content": "ignored because data is authoritative",
                "data": structured_data,
                "metadata": {"trace_id": "tr-structured"},
            }
        }
    }

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["call"]  # type: ignore[attr-defined]
    assert out["response"] == '{"choice":"summarize","confidence":0.95}'
    assert out["data"] == structured_data
    assert out["meta"]["output_mode"] == "structured"
    assert out["meta"]["trace"] == {"trace_id": "tr-structured"}


def test_visualflow_generate_music_result_sync_exposes_music_artifact_aliases() -> None:
    from abstractruntime.core.models import RunState
    from abstractruntime.visualflow_compiler.compiler import _sync_effect_results_to_node_outputs
    from abstractruntime.visualflow_compiler.visual.executor import visual_to_flow
    from abstractruntime.visualflow_compiler.visual.models import load_visualflow_json

    vf = load_visualflow_json(
        {
            "id": "vf-music-sync",
            "name": "vf-music-sync",
            "entryNode": "music",
            "nodes": [
                {
                    "id": "music",
                    "type": "generate_music",
                    "data": {"nodeType": "generate_music", "effectConfig": {"prompt": "lo-fi"}},
                }
            ],
            "edges": [],
        }
    )
    flow = visual_to_flow(vf)
    run = RunState.new(workflow_id=flow.flow_id, entry_node="music")
    artifact_ref = {"$artifact": "art-music", "artifact_id": "art-music", "content_type": "audio/wav"}
    generated = {
        "outputs": {
            "music": [
                {
                    "modality": "music",
                    "task": "music_generation",
                    "artifact_id": "art-music",
                    "artifact_ref": artifact_ref,
                    "content_type": "audio/wav",
                }
            ]
        },
        "provider": "acemusic",
        "model": "ace-step",
    }
    run.vars["_temp"] = {"effects": {"music": generated}}

    _sync_effect_results_to_node_outputs(run, flow)

    out = flow._node_outputs["music"]  # type: ignore[attr-defined]
    assert out["music_artifact"] == artifact_ref
    assert out["audio_artifact"] == artifact_ref
    assert out["artifact_ref"] == artifact_ref
    assert out["artifact_id"] == "art-music"
    assert out["content_type"] == "audio/wav"
    assert out["outputs"] == generated["outputs"]
    assert out["success"] is True
    assert out["meta"]["schema"] == "abstractflow.generate_music.v1.meta"
    assert out["meta"]["modality"] == "music"
