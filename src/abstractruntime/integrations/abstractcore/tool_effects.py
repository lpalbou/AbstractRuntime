"""What each exposable tool can do to the host, as ONE table (automations contract C4).

A discussion forked from an automation occurrence runs on the occurrence's
workspace mounted READ-ONLY (`workspace_read_only`). Enforcement asks this
table, never the tool's name or description text:

- `read`         reads files/web/devices; no workspace or host mutation
- `write`        writes files (workspace or host-local)
- `exec`         runs arbitrary commands or code (can write anything)
- `delegate`     starts child work (the child inherits the read-only scope)
- `comms`        talks to people/other systems (email, chat, agora, asking the user)
- `memory-write` writes the runtime's own memory/plan state, not files

Under read-only, `write` and `exec` are refused, and so is every tool NOT in
this table (an unclassified capability fails closed). The table covers every
tool the runtime's default toolsets can expose (abstractcore common, web,
system, comms, agora, shell and camera tools, the runtime-owned
`open_attachment`) plus the abstractagent tools the gateway exposes
(`execute_python`, `self_improve` and the agent builtins).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

READ = "read"
WRITE = "write"
EXEC = "exec"
DELEGATE = "delegate"
COMMS = "comms"
MEMORY_WRITE = "memory-write"

EFFECT_CLASSES = (READ, WRITE, EXEC, DELEGATE, COMMS, MEMORY_WRITE)
READ_ONLY_REFUSED_CLASSES = frozenset({WRITE, EXEC})

TOOL_EFFECT_CLASSES: Dict[str, str] = {
    # abstractcore files toolset (+ the runtime-owned attachment reader)
    "list_files": READ,
    "skim_folders": READ,
    "search_files": READ,
    "analyze_code": READ,
    "analyze_media": READ,
    "skim_files": READ,
    "read_file": READ,
    "open_attachment": READ,
    "write_file": WRITE,
    "edit_file": WRITE,
    # web toolset (network reads; browser_probe screenshots go to a temp dir)
    "skim_websearch": READ,
    "skim_url": READ,
    "web_search": READ,
    "fetch_url": READ,
    "browser_probe": READ,
    # system toolset
    "execute_command": EXEC,
    "local_helper_start": EXEC,
    "local_helper_status": READ,
    "local_helper_stop": EXEC,
    # persistent shell toolset
    "shell_exec": EXEC,
    "shell_write_stdin": EXEC,
    "shell_close": EXEC,
    # comms toolset
    "list_email_accounts": COMMS,
    "list_email_folders": COMMS,
    "send_email": COMMS,
    "reply_email": COMMS,
    "list_emails": COMMS,
    "search_emails": COMMS,
    "read_email": COMMS,
    # Saves an attachment into a (workspace-walled) local folder: a local write.
    "get_email_attachment": WRITE,
    "send_whatsapp_message": COMMS,
    "list_whatsapp_messages": COMMS,
    "read_whatsapp_message": COMMS,
    "send_telegram_message": COMMS,
    "send_telegram_artifact": COMMS,
    # agora toolset (remote hub state, never the local workspace)
    "agora_whoami": COMMS,
    "agora_check_inbox": COMMS,
    "agora_ack_inbox": COMMS,
    "agora_read_channel": COMMS,
    "agora_read_message": COMMS,
    "agora_post_message": COMMS,
    "agora_send_dm": COMMS,
    "channel_fs_write": COMMS,
    "channel_fs_read": COMMS,
    "channel_fs_list": COMMS,
    "channel_store_set": COMMS,
    "channel_store_get": COMMS,
    # camera toolset (device input; captures land in the artifact store)
    "camera_list_devices": READ,
    "camera_open": READ,
    "camera_close": READ,
    "camera_status": READ,
    "camera_preview_photo": READ,
    "camera_capture_photo": READ,
    "camera_capture_video": READ,
    "camera_stop_recording": READ,
    "camera_start_detection": READ,
    "camera_stop_detection": READ,
    "camera_get_events": READ,
    # abstractagent tools exposed by the gateway
    "execute_python": EXEC,
    "self_improve": WRITE,
    "ask_user": COMMS,
    "delegate_agent": DELEGATE,
    "inspect_vars": READ,
    "read_skill": READ,
    "recall_memory": READ,
    "remember": MEMORY_WRITE,
    "remember_note": MEMORY_WRITE,
    "compact_memory": MEMORY_WRITE,
    "update_plan": MEMORY_WRITE,
}


# --- Network reach and write scope (framework backlog 0992: untrusted-input grants) -----------
#
# Two more facts per tool, owned by the runtime because AbstractCore's inventory rows do not
# carry them (`skim_url` and `web_search` declare no destination fact; the agora, camera and
# agent tools have no core row at all). An automation whose trigger delivers text written by
# other people (`email.received@1`) grants, under "allow all tools", only what these facts
# prove harmless (`untrusted_input_grantable`). A tool missing from a table is treated as the
# worst value (fail closed).

NET_NONE = "none"
"""No network access."""
NET_CONFIGURED = "configured"
"""Reaches only services the user or the administrator configured (the model provider, the
user's own mailbox, the agora hub), at no address the model chooses, and changes nothing there."""
NET_OPEN = "open"
"""Reaches a destination the model or its input chooses (a URL, a host, a search engine fed
model-written queries), runs arbitrary code that can reach anything, or changes remote state."""
NET_SEND = "send"
"""Delivers a message that people or agents receive, to a recipient or channel the model chooses."""

NETWORK_REACHES = (NET_NONE, NET_CONFIGURED, NET_OPEN, NET_SEND)

TOOL_NETWORK_REACH: Dict[str, str] = {
    "list_files": NET_NONE,
    "skim_folders": NET_NONE,
    "search_files": NET_NONE,
    "analyze_code": NET_NONE,
    # Sends the file to the configured vision provider (never to an address the model picks).
    "analyze_media": NET_CONFIGURED,
    "skim_files": NET_NONE,
    "read_file": NET_NONE,
    "open_attachment": NET_NONE,
    "write_file": NET_NONE,
    "edit_file": NET_NONE,
    "skim_websearch": NET_OPEN,
    "skim_url": NET_OPEN,
    "web_search": NET_OPEN,
    "fetch_url": NET_OPEN,
    "browser_probe": NET_OPEN,
    "execute_command": NET_OPEN,
    "local_helper_start": NET_OPEN,
    "local_helper_status": NET_NONE,
    "local_helper_stop": NET_NONE,
    "shell_exec": NET_OPEN,
    "shell_write_stdin": NET_OPEN,
    "shell_close": NET_NONE,
    "list_email_accounts": NET_CONFIGURED,
    "list_email_folders": NET_CONFIGURED,
    "send_email": NET_SEND,
    "reply_email": NET_SEND,
    "list_emails": NET_CONFIGURED,
    "search_emails": NET_CONFIGURED,
    "read_email": NET_CONFIGURED,
    "get_email_attachment": NET_CONFIGURED,
    "send_whatsapp_message": NET_SEND,
    "list_whatsapp_messages": NET_CONFIGURED,
    "read_whatsapp_message": NET_CONFIGURED,
    "send_telegram_message": NET_SEND,
    "send_telegram_artifact": NET_SEND,
    "agora_whoami": NET_CONFIGURED,
    "agora_check_inbox": NET_CONFIGURED,
    "agora_ack_inbox": NET_OPEN,  # changes hub state
    "agora_read_channel": NET_CONFIGURED,
    "agora_read_message": NET_CONFIGURED,
    "agora_post_message": NET_SEND,
    "agora_send_dm": NET_SEND,
    "channel_fs_write": NET_OPEN,  # writes a file every channel member reads
    "channel_fs_read": NET_CONFIGURED,
    "channel_fs_list": NET_CONFIGURED,
    "channel_store_set": NET_OPEN,
    "channel_store_get": NET_CONFIGURED,
    "camera_list_devices": NET_NONE,
    "camera_open": NET_NONE,
    "camera_close": NET_NONE,
    "camera_status": NET_NONE,
    "camera_preview_photo": NET_NONE,
    "camera_capture_photo": NET_NONE,
    "camera_capture_video": NET_NONE,
    "camera_stop_recording": NET_NONE,
    "camera_start_detection": NET_NONE,
    "camera_stop_detection": NET_NONE,
    "camera_get_events": NET_NONE,
    "execute_python": NET_OPEN,
    "self_improve": NET_NONE,
    "ask_user": NET_NONE,
    "delegate_agent": NET_NONE,  # the child's own calls are judged on their own facts
    "inspect_vars": NET_NONE,
    "read_skill": NET_NONE,
    "recall_memory": NET_CONFIGURED,
    "remember": NET_NONE,
    "remember_note": NET_NONE,
    "compact_memory": NET_CONFIGURED,
    "update_plan": NET_NONE,
}

WRITES_WORKSPACE = "workspace"
"""Every path it writes is confined to the run's workspace, in every workspace mode."""
WRITES_WORKSPACE_ONLY_MODE = "workspace_only_mode"
"""Confined to the run's workspace only when `workspace_access_mode` is `workspace_only` (the
default); the other modes let absolute paths reach allowed or non-ignored folders."""
WRITES_HOST = "host"
"""Can write outside the run's workspace."""

# The scope of every `write`-class tool (`exec` tools can write anything and have no entry).
TOOL_WRITE_SCOPE: Dict[str, str] = {
    "write_file": WRITES_WORKSPACE_ONLY_MODE,
    "edit_file": WRITES_WORKSPACE_ONLY_MODE,
    "get_email_attachment": WRITES_WORKSPACE,
    "self_improve": WRITES_HOST,
}

# Effect classes an untrusted-input grant never covers, whatever the other facts say.
UNTRUSTED_REFUSED_CLASSES = frozenset({EXEC, DELEGATE})


def untrusted_input_grantable(tool_name: str, *, workspace_access_mode: Any = None) -> bool:
    """True when "allow all tools" may pre-approve `tool_name` for an occurrence whose input was
    written by other people: no network egress beyond configured services, no code or command
    execution, no messaging, no writes outside the run's workspace, no delegation.

    Decided on this module's facts only (effect class, network reach, write scope); a tool
    missing from any table it needs is refused. The caller adds AbstractCore's row facts on top.
    """
    name = str(tool_name or "").strip()
    effect = TOOL_EFFECT_CLASSES.get(name)
    reach = TOOL_NETWORK_REACH.get(name)
    if effect is None or reach is None:
        return False
    if effect in UNTRUSTED_REFUSED_CLASSES or reach not in (NET_NONE, NET_CONFIGURED):
        return False
    if effect == WRITE:
        scope = TOOL_WRITE_SCOPE.get(name)
        mode = str(workspace_access_mode or "workspace_only").strip() or "workspace_only"
        return scope == WRITES_WORKSPACE or (scope == WRITES_WORKSPACE_ONLY_MODE and mode == "workspace_only")
    return True


def sends_messages(tool_name: str) -> bool:
    """True for a tool whose reach is `send` (see `NET_SEND`)."""
    return TOOL_NETWORK_REACH.get(str(tool_name or "").strip()) == NET_SEND


def tool_effect_class(tool_name: str) -> Optional[str]:
    """The tool's effect class, or None when it is unclassified."""
    return TOOL_EFFECT_CLASSES.get(str(tool_name or "").strip())


def read_only_refusal(tool_name: str, *, path: Optional[str] = None) -> Optional[str]:
    """Why `tool_name` is refused, or None when allowed.

    Without `path`: the whole workspace is read-only (`workspace_read_only`).
    With `path`: the call writes `path`, which lies under a read-only mount
    (`workspace_read_only_paths`); only `write`-class tools are refused there
    (reads and exec tools are allowed — the shell is not sandboxed by mounts).
    """
    name = str(tool_name or "").strip()
    if path is not None:
        if TOOL_EFFECT_CLASSES.get(name) != WRITE:
            return None
        return (
            f"Tool '{name}' is refused: '{path}' is inside a read-only mount (a discussion can read "
            "the automation's workspace but never change it; write in your own workspace instead)."
        )
    effect = TOOL_EFFECT_CLASSES.get(name)
    if effect is None:
        return (
            f"Tool '{name}' is refused: this workspace is read-only and the tool's filesystem/"
            "execution effects are not classified (unclassified tools are refused)."
        )
    if effect in READ_ONLY_REFUSED_CLASSES:
        return (
            f"Tool '{name}' is refused: this workspace is read-only (a discussion forked from an "
            f"automation run cannot {'write files' if effect == WRITE else 'run commands or code'})."
        )
    return None


__all__ = [
    "COMMS",
    "DELEGATE",
    "EFFECT_CLASSES",
    "EXEC",
    "MEMORY_WRITE",
    "NETWORK_REACHES",
    "NET_CONFIGURED",
    "NET_NONE",
    "NET_OPEN",
    "NET_SEND",
    "READ",
    "READ_ONLY_REFUSED_CLASSES",
    "TOOL_EFFECT_CLASSES",
    "TOOL_NETWORK_REACH",
    "TOOL_WRITE_SCOPE",
    "UNTRUSTED_REFUSED_CLASSES",
    "WRITE",
    "WRITES_HOST",
    "WRITES_WORKSPACE",
    "WRITES_WORKSPACE_ONLY_MODE",
    "read_only_refusal",
    "sends_messages",
    "tool_effect_class",
    "untrusted_input_grantable",
]
