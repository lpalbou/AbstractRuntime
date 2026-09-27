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

from typing import Dict, Optional

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
    "send_email": COMMS,
    "list_emails": COMMS,
    "read_email": COMMS,
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


def tool_effect_class(tool_name: str) -> Optional[str]:
    """The tool's effect class, or None when it is unclassified."""
    return TOOL_EFFECT_CLASSES.get(str(tool_name or "").strip())


def read_only_refusal(tool_name: str) -> Optional[str]:
    """Why `tool_name` is refused in a read-only workspace, or None when allowed."""
    name = str(tool_name or "").strip()
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
    "READ",
    "READ_ONLY_REFUSED_CLASSES",
    "TOOL_EFFECT_CLASSES",
    "WRITE",
    "read_only_refusal",
    "tool_effect_class",
]
