"""write_chart effect node + charts renderer (operator ruling 2026-07-20).

A deterministic workflow must never stall on a tool-approval prompt while
following its process: figure rendering moved from the diagram-render
workflow's write-script-then-execute_command lane (approval-gated) into a
first-class in-process effect node in the write_pdf trust class. These
tests pin the adversarial-review hardening checklist: resource caps,
mathtext inertness, NaN rejection, honest ok:false degradation, workspace
path containment, and the deterministic bytes-exist gate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from abstractruntime.documents import render_chart


def _layered_spec() -> dict:
    return {
        "kind": "layered",
        "title": "Architecture",
        "caption": "caption",
        "layers": [
            {"label": "Input", "nodes": [{"id": "a", "label": "Source"}]},
            {"label": "Output", "nodes": [{"id": "b", "label": "Sink"}]},
        ],
        "edges": [{"from": "a", "to": "b", "label": "flow"}],
    }


def test_layered_renders_png_and_pdf(tmp_path: Path) -> None:
    result = render_chart(_layered_spec(), tmp_path / "arch.png")
    assert result.ok and result.rendered
    assert result.png_path is not None and result.png_path.stat().st_size > 128
    assert result.pdf_path is not None and result.pdf_path.stat().st_size > 128


def test_line_renders_with_y_min_anchor(tmp_path: Path) -> None:
    spec = {
        "kind": "line",
        "title": "Elo",
        "x_label": "cycle",
        "y_label": "elo",
        "y_min": 1200,
        "series": [{"label": "top", "points": [[1, 1210], [2, 1240], [3, 1260]]}],
    }
    result = render_chart(spec, tmp_path / "elo.png")
    assert result.ok and result.rendered


def test_mathtext_dollars_in_labels_are_inert(tmp_path: Path) -> None:
    # matplotlib parses $...$ as mathtext by default; an odd number of $
    # raises inside text layout. Labels must render literally.
    spec = _layered_spec()
    spec["title"] = "cost is $5 and $x"
    spec["layers"][0]["nodes"][0]["label"] = "price $ tag"
    spec["caption"] = "odd $ count"
    result = render_chart(spec, tmp_path / "d.png")
    assert result.ok, result.error


def test_nan_and_inf_points_are_rejected_not_plotted(tmp_path: Path) -> None:
    spec = {
        "kind": "line",
        "series": [{"label": "s", "points": [[1, float("nan")], [2, float("inf")], [3, 10]]}],
    }
    result = render_chart(spec, tmp_path / "n.png")
    assert result.ok  # the one finite point renders


def test_all_nonfinite_points_is_an_honest_refusal(tmp_path: Path) -> None:
    spec = {"kind": "line", "series": [{"label": "s", "points": [[1, float("nan")]]}]}
    result = render_chart(spec, tmp_path / "n2.png")
    assert not result.ok and "no points" in (result.error or "")


@pytest.mark.parametrize(
    "spec, needle",
    [
        ({"kind": "pie"}, "unknown kind"),
        ({"kind": "layered", "layers": []}, "no layers"),
        (
            {"kind": "layered", "layers": [{"label": "L", "nodes": [{"id": "a"}, {"id": "a"}]}]},
            "duplicate node id",
        ),
        (
            {
                "kind": "layered",
                "layers": [{"label": "L", "nodes": [{"id": "a"}]}],
                "edges": [{"from": "a", "to": "ghost"}],
            },
            "unknown node",
        ),
    ],
)
def test_invalid_specs_return_ok_false_never_raise(tmp_path: Path, spec: dict, needle: str) -> None:
    result = render_chart(spec, tmp_path / "bad.png")
    assert not result.ok and not result.rendered
    assert needle in (result.error or "")
    assert any("#FALLBACK" in w for w in result.warnings)


def test_resource_caps_refuse_oversized_specs(tmp_path: Path) -> None:
    too_many_points = {
        "kind": "line",
        "series": [{"label": "s", "points": [[i, i] for i in range(5001)]}],
    }
    result = render_chart(too_many_points, tmp_path / "big.png")
    assert not result.ok and "too many points" in (result.error or "")

    too_many_layers = {
        "kind": "layered",
        "layers": [{"label": str(i), "nodes": [{"id": f"n{i}"}]} for i in range(40)],
    }
    result = render_chart(too_many_layers, tmp_path / "big2.png")
    assert not result.ok and "too many layers" in (result.error or "")


def test_non_dict_spec_is_reported(tmp_path: Path) -> None:
    result = render_chart("not a spec", tmp_path / "x.png")
    assert not result.ok and "must be an object" in (result.error or "")


def _chart_flow() -> dict:
    return {
        "id": "chart-smoke",
        "name": "chart-smoke",
        "nodes": [
            {
                "id": "start",
                "type": "on_flow_start",
                "position": {"x": 0, "y": 0},
                "data": {
                    "nodeType": "on_flow_start",
                    "label": "Start",
                    "inputs": [],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "spec", "label": "spec", "type": "object"},
                    ],
                },
            },
            {
                "id": "chart",
                "type": "write_chart",
                "position": {"x": 300, "y": 0},
                "data": {
                    "nodeType": "write_chart",
                    "label": "Write Chart",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "file_path", "label": "file_path", "type": "string"},
                        {"id": "spec", "label": "spec", "type": "object"},
                    ],
                    "outputs": [
                        {"id": "exec-out", "label": "", "type": "execution"},
                        {"id": "ok", "label": "ok", "type": "boolean"},
                        {"id": "rendered", "label": "rendered", "type": "boolean"},
                        {"id": "file_path", "label": "file_path", "type": "string"},
                        {"id": "pdf_path", "label": "pdf_path", "type": "string"},
                        {"id": "error", "label": "error", "type": "string"},
                        {"id": "warnings", "label": "warnings", "type": "array"},
                    ],
                    "pinDefaults": {"file_path": "reports/figures/out.png"},
                },
            },
            {
                "id": "end",
                "type": "on_flow_end",
                "position": {"x": 600, "y": 0},
                "data": {
                    "nodeType": "on_flow_end",
                    "label": "End",
                    "inputs": [
                        {"id": "exec-in", "label": "", "type": "execution"},
                        {"id": "result", "label": "result", "type": "object"},
                    ],
                    "outputs": [],
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "sourceHandle": "exec-out", "target": "chart", "targetHandle": "exec-in"},
            {"id": "e2", "source": "start", "sourceHandle": "spec", "target": "chart", "targetHandle": "spec"},
            {"id": "e3", "source": "chart", "sourceHandle": "exec-out", "target": "end", "targetHandle": "exec-in"},
            {"id": "e4", "source": "chart", "sourceHandle": "ok", "target": "end", "targetHandle": "result"},
        ],
        "entryNode": "start",
    }


def _run_chart_flow(tmp_path: Path, file_path: str) -> object:
    from abstractruntime import Runtime
    from abstractruntime.scheduler.registry import WorkflowRegistry
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
    from abstractruntime.visualflow_compiler.compiler import compile_visualflow

    flow = _chart_flow()
    flow["nodes"][1]["data"]["pinDefaults"]["file_path"] = file_path
    workflow = compile_visualflow(flow)
    registry = WorkflowRegistry()
    registry.register(workflow)
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        workflow_registry=registry,
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={
            "spec": _layered_spec(),
            "workspace_root": str(tmp_path),
            "workspace_access_mode": "workspace_only",
        },
        actor_id="test",
    )
    return runtime.tick(workflow=workflow, run_id=run_id, max_steps=50)


def test_write_chart_handler_contains_paths_and_writes_both_files(tmp_path: Path) -> None:
    """The effect handler resolves through the workspace scope like write_pdf:
    a contained relative path renders both artifacts inside the workspace."""
    run = _run_chart_flow(tmp_path, "reports/figures/out.png")
    assert run.status.value == "completed", getattr(run, "error", None)
    png = tmp_path / "reports" / "figures" / "out.png"
    assert png.is_file() and png.stat().st_size > 128
    assert (tmp_path / "reports" / "figures" / "out.pdf").is_file()


def test_write_chart_handler_refuses_escaping_path(tmp_path: Path) -> None:
    """An absolute path outside the workspace refuses loudly — never rendered.

    The refusal rides the shared resolver (write_file parity). The node
    wrapper converts the raise into a success:false node output (the
    framework's resilient write-node semantics), so the pinned contract is:
    loud error in the node output, and NO file outside the workspace.
    """
    run = _run_chart_flow(tmp_path, "/private/tmp/definitely-outside-evil.png")
    node_outputs = ((run.vars or {}).get("_temp") or {}).get("node_outputs") or {}
    chart_out = node_outputs.get("chart") or {}
    assert chart_out.get("success") is False
    assert "outside workspace" in str(chart_out.get("error") or "")
    assert not Path("/private/tmp/definitely-outside-evil.png").exists()
    # Nothing rendered anywhere inside the workspace either.
    assert not list(tmp_path.rglob("*.png"))
