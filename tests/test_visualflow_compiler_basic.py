import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../abstractruntime/tests/<file> -> .../abstractframework
    return Path(__file__).resolve().parents[2]


def test_visualflow_compiler_compiles_real_flow_without_abstractflow_import() -> None:
    assert "abstractflow" not in sys.modules

    from abstractruntime.visualflow_compiler import compile_visualflow

    path = (
        _repo_root()
        / "docs"
        / "assessments"
        / "examples"
        / "workflowartifact-abstraction"
        / "ac-advanced-agent.visualflow.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))

    spec = compile_visualflow(raw)
    assert spec.workflow_id == "ac-advanced-agent"
    assert spec.entry_node == "node-1"
    assert isinstance(spec.nodes, dict)
    assert len(spec.nodes) == 4

    # The compiler must not import AbstractFlow (single semantics engine lives in runtime).
    assert "abstractflow" not in sys.modules


def test_visualflow_compiler_accepts_model_dump_like_objects() -> None:
    from abstractruntime.visualflow_compiler import compile_visualflow

    path = (
        _repo_root()
        / "docs"
        / "assessments"
        / "examples"
        / "workflowartifact-abstraction"
        / "ac-advanced-agent.visualflow.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))

    class Dummy:
        def model_dump(self):
            return raw

    spec = compile_visualflow(Dummy())
    assert spec.workflow_id == "ac-advanced-agent"


def test_visualflow_compiler_tree_compiles_bundle_flows() -> None:
    assert "abstractflow" not in sys.modules

    from abstractruntime.visualflow_compiler import compile_visualflow_tree

    flows_dir = (
        _repo_root()
        / "docs"
        / "assessments"
        / "examples"
        / "workflowartifact-abstraction"
        / "ac-advanced-agent.bundle"
        / "flows"
    )
    flows = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in flows_dir.glob("*.json")}

    specs = compile_visualflow_tree(root_id="ac-advanced-agent", flows_by_id=flows)
    assert set(specs.keys()) == {"ac-advanced-agent", "4ed3b340", "60a97e4d", "15f19f7f"}

    assert "abstractflow" not in sys.modules

