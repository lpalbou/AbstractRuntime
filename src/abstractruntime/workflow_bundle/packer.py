"""WorkflowBundle pack/unpack tooling (stdlib-only).

This module is intentionally host-agnostic:
- packing bundles is a pure filesystem/content operation
- hosts decide where bundles live and how they're distributed (disk, gateway upload, etc.)
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import WORKFLOW_BUNDLE_FORMAT_VERSION_V1, WorkflowBundleEntrypoint, WorkflowBundleError, WorkflowBundleManifest, workflow_bundle_manifest_to_dict
from .reader import open_workflow_bundle


@dataclass(frozen=True)
class PackedWorkflowBundle:
    path: Path
    manifest: WorkflowBundleManifest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _load_visualflow_dict_from_bytes(raw: bytes) -> Dict[str, Any]:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise WorkflowBundleError("VisualFlow JSON must be an object")
    return data


def _node_type(node: Any) -> str:
    if isinstance(node, dict):
        t = node.get("type")
        if isinstance(t, str) and t.strip():
            return t.strip()
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        t2 = data.get("nodeType")
        return str(t2 or "").strip()
    return ""


def _pins_from_node(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    data = node.get("data") if isinstance(node.get("data"), dict) else {}
    pins_in = data.get("inputs") if isinstance(data.get("inputs"), list) else node.get("inputs") if isinstance(node.get("inputs"), list) else []
    pins_out = data.get("outputs") if isinstance(data.get("outputs"), list) else node.get("outputs") if isinstance(node.get("outputs"), list) else []
    for p in list(pins_in) + list(pins_out):
        if isinstance(p, dict):
            yield p


def _reachable_exec_node_ids(flow: Dict[str, Any]) -> set[str]:
    """Return exec-reachable node ids (Blueprint-style; ignores disconnected exec nodes)."""
    nodes = flow.get("nodes")
    if not isinstance(nodes, list):
        return set()

    exec_ids: set[str] = set()
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_id = str(n.get("id") or "").strip()
        if not node_id:
            continue
        for p in _pins_from_node(n):
            if p.get("type") == "execution":
                exec_ids.add(node_id)
                break

    if not exec_ids:
        return set()

    edges = flow.get("edges")
    edges_list = edges if isinstance(edges, list) else []
    incoming_exec = {str(e.get("target") or "").strip() for e in edges_list if isinstance(e, dict) and e.get("targetHandle") == "exec-in"}

    roots: list[str] = []
    entry = flow.get("entryNode")
    if isinstance(entry, str) and entry in exec_ids:
        roots.append(entry)
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_id = str(n.get("id") or "").strip()
        if not node_id or node_id not in exec_ids:
            continue
        if _node_type(n) == "on_event":
            roots.append(node_id)
    if not roots:
        for node_id in exec_ids:
            if node_id not in incoming_exec:
                roots.append(node_id)
                break
    if not roots:
        roots.append(next(iter(exec_ids)))

    adj: Dict[str, list[str]] = {}
    for e in edges_list:
        if not isinstance(e, dict):
            continue
        if e.get("targetHandle") != "exec-in":
            continue
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if not src or not tgt:
            continue
        if src not in exec_ids or tgt not in exec_ids:
            continue
        adj.setdefault(src, []).append(tgt)

    reachable: set[str] = set()
    stack = list(dict.fromkeys([r for r in roots if isinstance(r, str) and r]))
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                stack.append(nxt)
    return reachable


def _collect_reachable_flows(
    *,
    root_flow: Dict[str, Any],
    root_bytes: bytes,
    flows_dir: Path,
) -> Tuple[List[Tuple[str, Dict[str, Any], bytes]], List[str]]:
    """Return [(flow_id, flow_dict, raw_bytes)] in discovery order + list of missing subflow ids."""
    ordered: list[Tuple[str, Dict[str, Any], bytes]] = []
    visited: set[str] = set()
    missing: list[str] = []

    root_id = str(root_flow.get("id") or "").strip()
    if not root_id:
        raise WorkflowBundleError("Root flow is missing 'id'")

    cache: Dict[str, Tuple[Dict[str, Any], bytes]] = {root_id: (root_flow, root_bytes)}

    def _load_by_id(flow_id: str) -> Optional[Tuple[Dict[str, Any], bytes]]:
        fid = str(flow_id or "").strip()
        if not fid:
            return None
        if fid in cache:
            return cache[fid]
        p = (flows_dir / f"{fid}.json").resolve()
        if not p.exists():
            return None
        raw = _read_json_bytes(p)
        vf = _load_visualflow_dict_from_bytes(raw)
        cache[fid] = (vf, raw)
        return cache[fid]

    def _dfs(vf: Dict[str, Any], raw: bytes) -> None:
        fid = str(vf.get("id") or "").strip()
        if not fid:
            missing.append("<missing-flow-id>")
            return
        if fid in visited:
            return
        visited.add(fid)
        ordered.append((fid, vf, raw))

        nodes = vf.get("nodes")
        if not isinstance(nodes, list):
            return
        reachable = _reachable_exec_node_ids(vf)
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if _node_type(n) != "subflow":
                continue
            nid = str(n.get("id") or "").strip()
            if reachable and nid and nid not in reachable:
                continue
            data = n.get("data") if isinstance(n.get("data"), dict) else {}
            sub_id = data.get("subflowId") or data.get("flowId")
            if not isinstance(sub_id, str) or not sub_id.strip():
                missing.append(f"<missing-subflow-id:{fid}:{nid or '?'}>")
                continue
            sub_id2 = sub_id.strip()
            child = _load_by_id(sub_id2)
            if child is None:
                if sub_id2 == fid:
                    _dfs(vf, raw)
                    continue
                missing.append(sub_id2)
                continue
            _dfs(child[0], child[1])

    _dfs(root_flow, root_bytes)
    return ordered, missing


def pack_workflow_bundle(
    *,
    root_flow_json: str | Path,
    out_path: str | Path,
    bundle_id: Optional[str] = None,
    bundle_version: str = "0.0.0",
    flows_dir: Optional[str | Path] = None,
    entrypoints: Optional[List[str]] = None,
    default_entrypoint: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PackedWorkflowBundle:
    """Pack a `.flow` bundle from a root VisualFlow JSON file."""
    root_path = Path(root_flow_json).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"root flow not found: {root_path}")
    root_bytes = _read_json_bytes(root_path)
    root_flow = _load_visualflow_dict_from_bytes(root_bytes)

    flows_base = Path(flows_dir).expanduser().resolve() if flows_dir is not None else root_path.parent
    if not flows_base.exists() or not flows_base.is_dir():
        raise FileNotFoundError(f"flows_dir does not exist: {flows_base}")

    ordered, missing = _collect_reachable_flows(root_flow=root_flow, root_bytes=root_bytes, flows_dir=flows_base)
    if missing:
        uniq = sorted(set(missing))
        raise WorkflowBundleError(f"Missing referenced subflows in flows_dir: {uniq}")

    root_id = str(root_flow.get("id") or "").strip()
    if not root_id:
        raise WorkflowBundleError("Root flow is missing 'id'")

    entry_ids = list(entrypoints) if isinstance(entrypoints, list) and entrypoints else [root_id]
    entry_ids = [str(x).strip() for x in entry_ids if isinstance(x, str) and str(x).strip()]
    if not entry_ids:
        raise WorkflowBundleError("No valid entrypoints specified")

    de_param = str(default_entrypoint).strip() if isinstance(default_entrypoint, str) and str(default_entrypoint).strip() else ""
    if de_param and de_param not in entry_ids:
        raise WorkflowBundleError(f"default_entrypoint '{de_param}' must be one of: {entry_ids}")
    default_ep = de_param or (root_id if root_id in entry_ids else entry_ids[0])

    flows_json: Dict[str, bytes] = {}
    interfaces_by_flow: Dict[str, list[str]] = {}
    name_by_flow: Dict[str, str] = {}
    desc_by_flow: Dict[str, str] = {}

    for fid, vf, raw in ordered:
        flows_json[fid] = raw
        name_by_flow[fid] = str(vf.get("name") or "")
        desc_by_flow[fid] = str(vf.get("description") or "")
        ifaces = vf.get("interfaces")
        interfaces_by_flow[fid] = [str(x).strip() for x in list(ifaces) if isinstance(x, str) and x.strip()] if isinstance(ifaces, list) else []

    bid = str(bundle_id or "").strip() or root_id
    created_at = _now_iso()

    eps: list[WorkflowBundleEntrypoint] = []
    for fid in entry_ids:
        fid2 = str(fid or "").strip()
        if not fid2:
            continue
        eps.append(
            WorkflowBundleEntrypoint(
                flow_id=fid2,
                name=name_by_flow.get(fid2) or fid2,
                description=desc_by_flow.get(fid2, ""),
                interfaces=list(interfaces_by_flow.get(fid2, [])),
            )
        )
    if not eps:
        raise WorkflowBundleError("No valid entrypoints specified")

    manifest = WorkflowBundleManifest(
        bundle_format_version=WORKFLOW_BUNDLE_FORMAT_VERSION_V1,
        bundle_id=bid,
        bundle_version=str(bundle_version or "0.0.0"),
        created_at=created_at,
        entrypoints=eps,
        default_entrypoint=default_ep,
        flows={fid: f"flows/{fid}.json" for fid in sorted(flows_json.keys())},
        artifacts={},
        assets={},
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )
    manifest.validate()

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(workflow_bundle_manifest_to_dict(manifest), indent=2, ensure_ascii=False))
        for fid in sorted(flows_json.keys()):
            zf.writestr(f"flows/{fid}.json", flows_json[fid])

    return PackedWorkflowBundle(path=out, manifest=manifest)


def inspect_workflow_bundle(*, bundle_path: str | Path) -> WorkflowBundleManifest:
    b = open_workflow_bundle(bundle_path)
    return b.manifest


def unpack_workflow_bundle(*, bundle_path: str | Path, out_dir: str | Path) -> Path:
    src = Path(bundle_path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Bundle not found: {src}")
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(out)
    return out

