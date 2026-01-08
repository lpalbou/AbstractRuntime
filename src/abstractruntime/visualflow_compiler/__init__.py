"""abstractruntime.visualflow_compiler

VisualFlow JSON → WorkflowSpec compiler (single semantics engine).

This module compiles the VisualFlow authoring DSL (JSON) into an in-memory
`WorkflowSpec` (Python callables) that AbstractRuntime can execute durably.

Note: `WorkflowSpec` is not a portable artifact format (it contains callables).
"""

from .flow import Flow, FlowEdge, FlowNode
from .compiler import compile_flow, compile_visualflow, compile_visualflow_tree
from .visual.models import VisualEdge, VisualFlow, VisualNode, load_visualflow_json
from .visual.executor import visual_to_flow

__all__ = [
    "Flow",
    "FlowNode",
    "FlowEdge",
    "VisualFlow",
    "VisualNode",
    "VisualEdge",
    "load_visualflow_json",
    "visual_to_flow",
    "compile_flow",
    "compile_visualflow",
    "compile_visualflow_tree",
]

