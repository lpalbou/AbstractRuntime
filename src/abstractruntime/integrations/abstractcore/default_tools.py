"""Default toolsets for AbstractRuntime's AbstractCore integration.

This module provides a *host-side* convenience list of common, safe(ish) tools
that can be wired into a Runtime via MappingToolExecutor.

Design notes:
- We keep the runtime kernel dependency-light; this lives under
  `integrations/abstractcore/` which is the explicit opt-in to AbstractCore.
- Tool callables are never persisted in RunState; only ToolSpecs (dicts) are.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


ToolCallable = Callable[..., Any]

# ONE SOURCE for the comms channel membership (gateway c4899: their catalog
# carried a pinned COPY of this map for partial-enablement remainder rows —
# exporting it kills the cross-repo copy). The composition below consumes
# exactly this structure, so membership changes cannot drift between the
# composed toolset and the exported map. Order = composition order.
_COMMS_KIND_TOOLS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "email": (
        "abstractcore.tools.comms_tools",
        ("list_email_accounts", "send_email", "list_emails", "read_email"),
    ),
    "whatsapp": (
        "abstractcore.tools.comms_tools",
        ("send_whatsapp_message", "list_whatsapp_messages", "read_whatsapp_message"),
    ),
    "telegram": (
        "abstractcore.tools.telegram_tools",
        ("send_telegram_message", "send_telegram_artifact"),
    ),
}


def comms_toolset_kinds() -> Dict[str, List[str]]:
    """kind -> tool NAMES for the comms channels (email/whatsapp/telegram),
    independent of enablement — the FULL potential membership a catalog
    needs to render disabled remainder rows. Exported for the gateway
    (c4899); the toolset composition consumes the same structure, so this
    can never disagree with what registers."""
    return {kind: list(names) for kind, (_module, names) in _COMMS_KIND_TOOLS.items()}

_COMMS_ENABLE_ENV_VARS = (
    "ABSTRACT_ENABLE_COMMS_TOOLS",
    "ABSTRACT_ENABLE_EMAIL_TOOLS",
    "ABSTRACT_ENABLE_WHATSAPP_TOOLS",
    "ABSTRACT_ENABLE_TELEGRAM_TOOLS",
)


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def comms_tools_enabled() -> bool:
    """Return True when the host explicitly opts into comms tools via env."""
    return any(_env_flag(k) for k in _COMMS_ENABLE_ENV_VARS)


def email_tools_enabled() -> bool:
    return _env_flag("ABSTRACT_ENABLE_COMMS_TOOLS") or _env_flag("ABSTRACT_ENABLE_EMAIL_TOOLS")


def whatsapp_tools_enabled() -> bool:
    return _env_flag("ABSTRACT_ENABLE_COMMS_TOOLS") or _env_flag("ABSTRACT_ENABLE_WHATSAPP_TOOLS")


def telegram_tools_enabled() -> bool:
    return _env_flag("ABSTRACT_ENABLE_COMMS_TOOLS") or _env_flag("ABSTRACT_ENABLE_TELEGRAM_TOOLS")


def agora_tools_enabled() -> bool:
    """Agora (agent-to-agent hub) tools: EXPLICIT INTENT + a key, never a
    lucky inherited credential.

    LIVE INCIDENT 2026-07-22 (env-conflict adversary B / gateway c4218):
    the running gateway inherited another seat's AGORA_API_KEY from its
    spawning shell, and this gate's old ambient-key-IMPLIES-toolset rule
    minted agora tools that posted under the FOREIGN identity - live
    cross-identity contamination. A credential in the process env proves
    only that a shell exported something, never that THIS host chose to
    speak on the hub. The gate now requires BOTH halves:
    - EXPLICIT INTENT: ABSTRACT_ENABLE_AGORA_TOOLS (the operator's own
      opt-in; migrates to gateway/console config with the dm#177 wave), AND
    - A KEY to speak with (process-global or per-alias `AGORA_API_KEY__*`,
      hooks plan H8 - a fleet host may run N aliased residents).
    Key-alone no longer registers (the contamination vector); flag-alone
    no longer registers either (tools that cannot authenticate are a
    misconfiguration surfaced at registration, not at first call).
    """
    if not _env_flag("ABSTRACT_ENABLE_AGORA_TOOLS"):
        return False
    if str(os.getenv("AGORA_API_KEY") or "").strip():
        return True
    return any(
        k.startswith("AGORA_API_KEY__") and str(v or "").strip() for k, v in os.environ.items()
    )


def shell_tools_enabled() -> bool:
    """Persistent shell-session tools (backlog 0220): explicit opt-in only.

    Deliberately NOT a default: a persistent shell escapes per-call cwd confinement once
    approved. The tools stay approval-gated per call even when enabled; until an OS sandbox
    exists (gateway backlog 0062) enabling this grants execute_command-level trust with
    session persistence.
    """
    return _env_flag("ABSTRACT_ENABLE_SHELL_TOOLS")


# --- camera (drafted by seat: camera; owner review: runtime — commons c3826/c3829;
# env gate REMOVED by operator ruling 2026-07-21 dm:camera--laurent#10; import
# lane MOVED to core's capability surface by operator ruling 2026-07-22
# dm:camera--laurent#16-20, verbatim: "THE ONLY PACKAGE THAT CAN AND SHOULD
# IMPORT ABSTRACT CAMERA IS ABSTRACT CORE" — runtime never imports
# abstractcamera; it consumes the tools the camera PLUGIN contributed through
# abstractcore.capabilities (register_capability_tools / capability_tools,
# commons c4219)) ---
_camera_import_warned = False


def _camera_capability_tools() -> List[Any]:
    """Camera ToolDefinitions served THROUGH core's capability surface —
    the ONE predicate every camera surface consumes (availability answer ==
    registration gate; adversary F1: two gates minted a silent
    inconsistency window).

    The plugin contributes its tools when core loads entry-point plugins;
    an empty answer with the plugin PRESENT is the present-but-broken shape
    and warns ONCE with #FALLBACK — true absence stays silent (not
    installed = not registered is the ruled contract, nothing to say).
    Present-vs-broken reads core's registry status (core ruling c4265:
    `plugins_seen`/`plugin_errors` name the erroring plugin AND carry its
    real error text — never an import probe, which false-positives on a
    bare directory shadowing sys.path).

    Debugging gotcha (runtime owner review c4253): a probe run from the
    FRAMEWORK WORKSPACE ROOT resolves abstractcore as an empty NAMESPACE
    package (the repo dir shadows the installed package; core has no src/
    layout) and this lane reads zero tools — a FALSE present-but-broken.
    Probe from a neutral cwd (or `python -P`) before chasing a phantom
    plugin failure.

    Cost contract (adversary P2-3, accepted + documented): the FIRST call
    pays core's whole entry-point plugin load — every installed capability
    plugin's register() runs (measured ~0.9s marginal with voice/vision/
    music installed; the heavy camera stack still never loads — plugins are
    import-light by core's contract). Once per process, module import stays
    clean."""
    global _camera_import_warned
    try:
        from abstractcore.capabilities import capability_tools
    except Exception as exc:
        # This module IS the abstractcore integration (get_default_toolsets
        # imports abstractcore.tools unconditionally), so landing here means
        # VERSION SKEW — an abstractcore predating the capability-tools
        # surface — never true absence. Silent degradation here is exactly
        # how the 2026-07-22 serving gateway dropped all 11 camera tools
        # with nothing in the logs (flow c4351): warn once, name the skew.
        if not _camera_import_warned:
            _camera_import_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "#FALLBACK abstractcore is present but its capabilities "
                "surface lacks capability_tools (version skew — the running "
                "process imported an abstractcore predating the capability-"
                "tools contract, or stale bytecode); camera toolset skipped: %s",
                exc,
            )
        return []
    try:
        tools = list(capability_tools("camera") or [])
    except Exception:
        tools = []
    if tools:
        return tools
    if not _camera_import_warned:
        detail = _camera_plugin_error_detail()
        if detail is not None:
            _camera_import_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "#FALLBACK abstractcamera's capability plugin is present but "
                "served no camera tools; camera toolset skipped: %s", detail,
            )
    return []


def capability_plugin_errors() -> List[Dict[str, str]]:
    """Every capability plugin's recorded load error (gateway c4899 facade
    ask 2): the `plugin_errors` slice of core's shared registry status,
    normalized to [{name, error}]. Consumer: the gateway's catalog_warnings
    surfacing (the camera boot-race class, c4634) — one facade instead of a
    direct abstractcore import in their catalog. Degrades to [] when core
    is absent or the status read fails (the camera-specific note below
    keeps its own richer diagnosis either way)."""
    try:
        from abstractcore.capabilities import shared_capability_registry

        status = shared_capability_registry().status()
    except Exception:  # noqa: BLE001 - a status read must never raise into a catalog
        return []
    out: List[Dict[str, str]] = []
    for entry in status.get("plugin_errors") or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if name:
                out.append({"name": name, "error": str(entry.get("error") or "plugin load failed")})
    return out


def _camera_plugin_error_detail() -> "str | None":
    """None when abstractcamera is truly ABSENT (silence is correct);
    otherwise the present-but-broken diagnosis from core's registry status
    (the plugin's real load error when recorded, or the release-predates-
    contribution shape when the plugin loaded but contributed nothing)."""
    try:
        from abstractcore.capabilities import shared_capability_registry

        status = shared_capability_registry().status()
    except Exception:
        return None
    seen = any(
        str(entry.get("name") or "") == "abstractcamera"
        for entry in (status.get("plugins_seen") or [])
        if isinstance(entry, dict)
    )
    for entry in status.get("plugin_errors") or []:
        if isinstance(entry, dict) and str(entry.get("name") or "") == "abstractcamera":
            return str(entry.get("error") or "plugin load failed")
    if seen:
        return (
            "the plugin loaded without error but contributed no tools "
            "(installed release predates the capability-tools contribution?)"
        )
    return None


def camera_tools_available() -> bool:
    """The camera toolset WILL register — installed and importable, the ONLY
    gate (installed = registered, like files/web/system riding abstractcore).

    Exposure and consent stay where they already live, per the ruling: the
    app's tool selection (allowed_tools / run tool configs), the user's
    tool_policy, the gateway's structural walls, and the classification's
    ask-by-default approval for every environment-capturing tool (c3938 —
    a DEFAULT the user may override, not a floor). A duplicate env flag on
    top of those was gating theater."""
    return bool(_camera_capability_tools())


_camera_policy_scope_warned = False


def camera_approval_sets() -> tuple[set, set]:
    """(auto_approve, require_approval) camera tool names, or (∅, ∅) when
    abstractcamera is not installed (absence of the package is the only gate).

    DERIVED from abstractcamera's own classification, served through core's
    `capability_tool_policy("camera")` — the plugin registers the partition
    it computes from its own classification facts (derive-never-copy; the
    diary_type-clamp drift is what copies cause), core carries the result,
    runtime folds it. Fail closed: no policy registered means no camera name
    auto-approves (an unlisted name already asks via default-deny).

    CONTAINMENT (adversary P1-2): the served partition is scoped to the
    names the SAME capability actually serves as tools — the fold unions
    auto_approve into the process-wide default policy, so an unscoped
    entry (a "camera" policy auto-approving `write_file` or an MCP tool,
    from a buggy or hostile plugin overwrite) would silently escalate
    arbitrary names past approval. Foreign names are DROPPED with one
    #FALLBACK warn; a capability's policy can only ever speak for its own
    tools."""
    global _camera_policy_scope_warned
    try:
        from abstractcore.capabilities import capability_tool_policy

        policy = capability_tool_policy("camera")
    except Exception:
        return set(), set()
    if not isinstance(policy, dict) or not policy:
        return set(), set()
    auto = set(policy.get("auto_approve") or [])
    require = set(policy.get("require_approval") or [])
    served_names = {
        str(getattr(d, "name", "") or "").strip() for d in _camera_capability_tools()
    } - {""}
    foreign = (auto | require) - served_names
    if foreign:
        auto &= served_names
        require &= served_names
        if not _camera_policy_scope_warned:
            _camera_policy_scope_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "#FALLBACK camera capability policy named tools the capability "
                "does not serve; dropped from the approval fold (a policy may "
                "only speak for its own tools): %s", sorted(foreign),
            )
    return auto, require


def default_approval_policy_sets() -> tuple[set, set]:
    """The EFFECTIVE (auto_approve, require_approval) name sets for a session:
    tool_executor's base defaults extended by every enabled toolset that
    carries its own approval facts.

    This is the ONE fold point runtime's executor construction consults so
    camera's derived partition rides through live (installed → the three
    read-only camera tools auto-approve, the eight mutating/remote/capturing
    ones ask; not installed → no camera names appear anywhere). Base sets stay
    in tool_executor (module constants, hot path); the camera import happens
    here, never at tool_executor import time."""
    from .tool_executor import _DEFAULT_REQUIRE_APPROVAL, _DEFAULT_SAFE_AUTO_APPROVE

    auto = set(_DEFAULT_SAFE_AUTO_APPROVE)
    require = set(_DEFAULT_REQUIRE_APPROVAL)
    cam_auto, cam_require = camera_approval_sets()
    auto |= cam_auto
    require |= cam_require
    return auto, require


def _tool_name(func: ToolCallable) -> str:
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None:
        name = getattr(tool_def, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = getattr(func, "__name__", "")
    return str(name or "").strip()


def _tool_spec(func: ToolCallable) -> Dict[str, Any]:
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None and hasattr(tool_def, "to_dict"):
        return dict(tool_def.to_dict())

    from abstractcore.tools.core import ToolDefinition

    return dict(ToolDefinition.from_function(func).to_dict())

def _normalize_tool_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ToolSpec quirks for better UI/LLM ergonomics.

    This is a presentation layer: it must not change tool execution semantics.
    """
    name = str(spec.get("name") or "").strip()
    if not name:
        return spec

    # AbstractCore's `skim_*` tools accept multiple paths and can parse strings for
    # backward compatibility, but the preferred shape is an array-of-strings.
    if name in {"skim_files", "skim_folders"}:
        params = spec.get("parameters")
        if isinstance(params, dict):
            paths_schema = params.get("paths")
            if isinstance(paths_schema, dict):
                desc = paths_schema.get("description")
                normalized = {
                    "type": "array",
                    "items": {"type": "string"},
                }
                if isinstance(desc, str) and desc.strip():
                    normalized["description"] = desc.strip()
                params["paths"] = normalized

    return spec


def list_tool_catalog(include_disabled: bool = True) -> List[Dict[str, Any]]:
    """The FULL toolset catalog (tool-tiers item H, laurent dm#221): every
    toolset runtime knows how to compose - enabled AND disabled - each row
    carrying {id, label, enabled, gate, tools}. Disabled rows still import
    their callables so consumers can serve REAL specs with enabled:false
    (exists-but-not-enabled is a visible state, never silence - the audit
    G-1 structural fix). The gateway's catalog fold consumes THIS and
    deletes its own assembly (one source, c4562 seam).

    `gate` names what governs enablement TODAY (the env flags; they migrate
    to gateway config per dm#177/dm#210 - the gate string is the honest
    pointer either way). Import failures on disabled rows degrade to
    tool_names-only rows with a #FALLBACK note, never a raised catalog."""
    catalog: List[Dict[str, Any]] = []
    enabled_sets = get_default_toolsets()
    for ts_id, ts in enabled_sets.items():
        catalog.append({
            "id": ts_id, "label": ts.get("label") or ts_id,
            "enabled": True, "gate": _CATALOG_GATES.get(ts_id, "always"),
            "tools": list(ts.get("tools") or []),
        })
    if not include_disabled:
        return catalog
    present = {row["id"] for row in catalog}

    def _disabled(ts_id: str, label: str, gate: str, loader: Any,
                  *, specless_note: Optional[str] = None) -> None:
        if ts_id in present:
            return
        row: Dict[str, Any] = {"id": ts_id, "label": label, "enabled": False, "gate": gate}
        try:
            row["tools"] = loader()
        except Exception as e:  # noqa: BLE001 - a broken import never kills the catalog
            row["tools"] = []
            row["note"] = f"#FALLBACK callables unavailable ({e}); specs unavailable until installed"
        # A row that loaded ZERO specs without raising is still VISIBLE with
        # its enablement path (gateway c4630: camera-not-installed must not
        # be silence - "exists but not surfaced" is the class we are
        # fixing). Env-gated rows never hit this (their specs always
        # import); install-gated rows (camera) do when the package is
        # absent - the note names WHY there are no specs yet.
        if not row.get("tools") and "note" not in row and specless_note:
            row["note"] = specless_note
        catalog.append(row)

    def _load_email() -> List[Any]:
        from abstractcore.tools.comms_tools import (
            list_email_accounts, list_emails, read_email, send_email,
        )

        return [list_email_accounts, send_email, list_emails, read_email]

    def _load_whatsapp() -> List[Any]:
        from abstractcore.tools.comms_tools import (
            list_whatsapp_messages, read_whatsapp_message, send_whatsapp_message,
        )

        return [send_whatsapp_message, list_whatsapp_messages, read_whatsapp_message]

    def _load_telegram() -> List[Any]:
        from abstractcore.tools.telegram_tools import send_telegram_artifact, send_telegram_message

        return [send_telegram_message, send_telegram_artifact]

    def _load_agora() -> List[Any]:
        from .agora_tools import AGORA_TOOLS

        return list(AGORA_TOOLS)

    def _load_shell() -> List[Any]:
        from abstractcore.tools.shell_tools import SHELL_TOOLS

        return list(SHELL_TOOLS)

    def _load_camera() -> List[Any]:
        return [d.function for d in _camera_capability_tools()]

    # PER-CHANNEL comms rows (gateway c4573 gap: comms as ONE row made
    # whatsapp/telegram VANISH from both lanes under partial enablement -
    # email on, others off, the aggregate disabled row suppressed by the
    # enabled comms row). Each channel that is OFF gets its own disabled
    # row with ITS gate; enabled channels ride the enabled comms row as
    # before (get_default_toolsets composition unchanged - the pin holds).
    if not email_tools_enabled():
        _disabled("comms.email", "Comms - Email",
                  "ABSTRACT_ENABLE_COMMS_TOOLS or ABSTRACT_ENABLE_EMAIL_TOOLS", _load_email)
    if not whatsapp_tools_enabled():
        _disabled("comms.whatsapp", "Comms - WhatsApp",
                  "ABSTRACT_ENABLE_COMMS_TOOLS or ABSTRACT_ENABLE_WHATSAPP_TOOLS", _load_whatsapp)
    if not telegram_tools_enabled():
        _disabled("comms.telegram", "Comms - Telegram",
                  "ABSTRACT_ENABLE_COMMS_TOOLS or ABSTRACT_ENABLE_TELEGRAM_TOOLS", _load_telegram)
    _disabled("agora", "Agora",
              "ABSTRACT_ENABLE_AGORA_TOOLS AND an AGORA_API_KEY (both halves, c4226)", _load_agora)
    _disabled("shell", "Shell (persistent)", "ABSTRACT_ENABLE_SHELL_TOOLS", _load_shell)
    _disabled("camera", "Camera",
              "install abstractcamera (installed = registered, served via core's capability surface)",
              _load_camera,
              specless_note=("#FALLBACK abstractcamera not installed/registered in this "
                             "process; the capability is grantable once installed - no "
                             "specs to serve until then"))
    return catalog


# The gate names for ENABLED rows (the catalog's provenance column).
_CATALOG_GATES: Dict[str, str] = {
    "files": "always", "web": "always", "system": "always",
    "comms": "ABSTRACT_ENABLE_COMMS_TOOLS (or per-channel flags)",
    "agora": "ABSTRACT_ENABLE_AGORA_TOOLS AND an AGORA_API_KEY",
    "shell": "ABSTRACT_ENABLE_SHELL_TOOLS",
    "camera": "installed = registered (abstractcamera via core)",
}


def get_default_toolsets(
    *, disabled_toolsets: Optional[Iterable[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Return default toolsets {id -> {label, tools:[callables]}}.

    `disabled_toolsets` (gateway c4877 registration seam, camera default-off
    wave): toolset ids the HOST's settings registry turned off compose OUT
    of registration entirely — a workflow bypassing discovery cannot reach
    them (the executor's default-deny covers dispatch of unregistered
    names). PARAMETER-EXPLICIT by design, never an env read: dm#10 killed
    the camera enable env ("each app can decide which tools run") and
    dm#177 forbids new behavior envs — a host passing its own console-held
    configuration IS the app deciding. Availability predicates above stay
    untouched (installed = registered remains the DEFAULT; this is the
    host's explicit subtraction on top)."""
    from abstractcore.tools.common_tools import (
        list_files,
        skim_folders,
        read_file,
        skim_files,
        search_files,
        analyze_code,
        analyze_media,
        write_file,
        edit_file,
        skim_websearch,
        skim_url,
        web_search,
        fetch_url,
        execute_command,
    )
    from .local_helper_tools import LOCAL_HELPER_TOOLS

    toolsets: Dict[str, Dict[str, Any]] = {
        "files": {
            "id": "files",
            "label": "Files",
            # analyze_media joins files (G-2 ruled DRIFT by core, c4526:
            # shipped 0825 into core's inventory 2026-07-21 but never wired
            # into the toolset composition - delegated SIGHT for text-only
            # agents belongs in the default set beside analyze_code).
            "tools": [list_files, skim_folders, search_files, analyze_code, analyze_media, skim_files, read_file, write_file, edit_file],
        },
        "web": {
            "id": "web",
            "label": "Web",
            "tools": [skim_websearch, skim_url, web_search, fetch_url],
        },
    }

    # browser_probe joins web (operator dm#24, core c5005): render-verification
    # is fetch_url's peer (same facts + mcd; ask-by-default in the approval
    # fold). The IMPORT is guarded because the tool lives in its own module
    # (browser_tools) - core's inventory emits its FACTS even when Playwright
    # is absent (the tool self-describes the install hint at call time), so
    # registration follows importability of the module, not of Chromium.
    try:
        from abstractcore.tools.browser_tools import browser_probe

        toolsets["web"]["tools"].append(browser_probe)
    except ImportError:
        pass  # older core without browser_tools: the web set stays as-is

    toolsets["system"] = {
        "id": "system",
        "label": "System",
        "tools": [execute_command, *LOCAL_HELPER_TOOLS],
    }

    if comms_tools_enabled():
        comms: list[ToolCallable] = []
        # Composition consumes the exported kind map (one source, c4899):
        # per-channel gates decide WHICH kinds load; the map decides WHAT
        # each kind is.
        _kind_gates: Dict[str, Callable[[], bool]] = {
            "email": email_tools_enabled,
            "whatsapp": whatsapp_tools_enabled,
            "telegram": telegram_tools_enabled,
        }
        for _kind, (_module_path, _names) in _COMMS_KIND_TOOLS.items():
            gate = _kind_gates.get(_kind)
            if gate is None or not gate():
                continue
            _mod = importlib.import_module(_module_path)
            comms.extend(getattr(_mod, n) for n in _names)

        if comms:
            toolsets["comms"] = {
                "id": "comms",
                "label": "Comms",
                "tools": comms,
            }

    if agora_tools_enabled():
        from .agora_tools import AGORA_TOOLS

        toolsets["agora"] = {
            "id": "agora",
            "label": "Agora",
            "tools": list(AGORA_TOOLS),
        }

    if shell_tools_enabled():
        from abstractcore.tools.shell_tools import SHELL_TOOLS

        toolsets["shell"] = {
            "id": "shell",
            "label": "Shell (persistent)",
            "tools": list(SHELL_TOOLS),
        }

    # Camera: installed = registered (operator ruling 2026-07-21, dm#10 —
    # the ABSTRACT_ENABLE_CAMERA_TOOLS env gate is DEAD: apps own tool
    # selection, tool_policy owns consent, the classification keeps every
    # capture verb ask-by-default). Not installed = not registered; a
    # present-but-broken install warns once inside the shared predicate.
    # Served THROUGH core's capability surface (runtime never imports
    # abstractcamera — dm#16-20 layering ruling): the plugin contributed
    # ToolDefinitions whose .function is the @tool-decorated callable, so
    # the registry composes them exactly like the abstractcore common tools.
    camera_defs = _camera_capability_tools()
    if camera_defs:
        camera_callables = [
            d.function for d in camera_defs if callable(getattr(d, "function", None))
        ]
        if camera_callables:
            toolsets["camera"] = {
                "id": "camera",
                "label": "Camera",
                "tools": camera_callables,
            }

    disabled = {str(t).strip() for t in (disabled_toolsets or []) if str(t).strip()}
    if disabled:
        for tid in list(toolsets.keys()):
            if tid in disabled:
                del toolsets[tid]

    return toolsets


def get_default_tools(
    *, disabled_toolsets: Optional[Iterable[str]] = None
) -> List[ToolCallable]:
    """Return the flattened list of all default tool callables."""
    toolsets = get_default_toolsets(disabled_toolsets=disabled_toolsets)
    out: list[ToolCallable] = []
    seen: set[str] = set()
    for spec in toolsets.values():
        for tool in spec.get("tools", []):
            if not callable(tool):
                continue
            name = _tool_name(tool)
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(tool)
    return out


def list_default_tool_specs(
    *, disabled_toolsets: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Return ToolSpecs for UI and LLM payloads (JSON-safe)."""
    toolsets = get_default_toolsets(disabled_toolsets=disabled_toolsets)
    toolset_by_name: Dict[str, str] = {}
    toolset_order: Dict[str, int] = {tid: idx for idx, tid in enumerate(toolsets.keys())}
    tool_order_by_name: Dict[str, int] = {}
    toolset_sizes: Dict[str, int] = {}
    for tid, spec in toolsets.items():
        order = 0
        for tool in spec.get("tools", []):
            if callable(tool):
                name = _tool_name(tool)
                if name:
                    toolset_by_name[name] = tid
                    tool_order_by_name[name] = order
                    order += 1
        toolset_sizes[tid] = order

    out: list[Dict[str, Any]] = []
    for tool in get_default_tools(disabled_toolsets=disabled_toolsets):
        spec = _normalize_tool_spec(_tool_spec(tool))
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        spec["toolset"] = toolset_by_name.get(name) or "other"
        out.append(spec)

    # Runtime-owned tools (no host callable).
    #
    # Rationale: these tools require runtime context/state (e.g., session-scoped ArtifactStore access)
    # and are executed inside the runtime effect handlers instead of via a host ToolExecutor.
    out.append(
        {
            "name": "open_attachment",
            "description": "Open a session attachment; returns text (excerpt or full if max_chars<=0) and attaches media for the next LLM call.",
            "when_to_use": (
                "Re-open a previously attached file; text returns an excerpt by default (or full when max_chars<=0), "
                "media is attached as `media` on the next call. Do not call if it is already present in the current messages/media."
            ),
            "parameters": {
                "artifact_id": {"type": "string", "default": None},
                "handle": {"type": "string", "default": None},
                "expected_sha256": {"type": "string", "default": None},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "default": None},
                "max_chars": {"type": "integer", "default": 8000},
            },
            "required_args": [],
            "examples": [
                {
                    "description": "Open by artifact id (preferred)",
                    "arguments": {"artifact_id": "abc123", "start_line": 1, "end_line": 80},
                },
                {
                    "description": "Open by handle",
                    "arguments": {"handle": "@docs/architecture.md", "max_chars": 2000},
                },
                {
                    "description": "Open full text (no cap)",
                    "arguments": {"artifact_id": "abc123", "max_chars": 0},
                },
            ],
            "toolset": "files",
        }
    )
    tool_order_by_name["open_attachment"] = toolset_sizes.get("files", 0)

    # Stable ordering: keep declared toolset order and preserve the preferred
    # within-toolset order so lighter "skim_*" tools can appear before heavier
    # full-fetch tools in LLM-facing specs.
    out.sort(
        key=lambda s: (
            toolset_order.get(str(s.get("toolset") or ""), 999),
            tool_order_by_name.get(str(s.get("name") or ""), 999),
            str(s.get("name") or ""),
        )
    )
    return out


def build_default_tool_map() -> Dict[str, ToolCallable]:
    """Return {tool_name -> callable} for MappingToolExecutor."""
    tool_map: Dict[str, ToolCallable] = {}
    for tool in get_default_tools():
        name = _tool_name(tool)
        if not name:
            continue
        tool_map[name] = tool
    return tool_map


def filter_tool_specs(tool_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Return ToolSpecs for the requested tool names (order preserved)."""
    available = {str(s.get("name")): s for s in list_default_tool_specs() if isinstance(s.get("name"), str)}
    out: list[Dict[str, Any]] = []
    for name in tool_names:
        spec = available.get(name)
        if spec is not None:
            out.append(spec)
    return out
