"""Runtime-owned local-helper tools for long-lived local processes.

Narrower than persistent shell sessions:
- purpose-built for starting / observing / stopping one long-lived helper;
- run-scoped via a hidden `_registry_namespace` trust-boundary stamp;
- default-toolset eligible without exposing a general persistent shell.

The helper registry is process-local and non-durable. Helpers are torn down
by the runtime's terminal-hook seam when the owning run ends.
"""

import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from abstractcore.tools.core import tool

_DEFAULT_HELPER_ID = "main"
_TAIL_CHARS = 4000
_MAX_READY_TIMEOUT_S = 120.0
_READY_POLL_S = 0.1
_SHELL_TOKENS = ("&&", "||", ";", "|", ">", ">>", "<", "<<", "&")


def namespaced_helper_id(namespace: str, helper_id: str) -> str:
    ns = str(namespace or "").strip()
    hid = str(helper_id or "").strip() or _DEFAULT_HELPER_ID
    return f"{ns}::{hid}" if ns else hid


def _tail_text(path: Path, *, max_chars: int = _TAIL_CHARS) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            window = min(size, max_chars * 4)
            fh.seek(max(0, size - window), os.SEEK_SET)
            text = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    text = text[-max_chars:]
    return text.strip()


def _coerce_timeout(value: Any, default: float, *, cap: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not (0 < out < float("inf")):
        return float(default)
    return min(out, cap)


def _port_ready(port: int) -> bool:
    for host in ("127.0.0.1", "localhost"):
        try:
            with socket.create_connection((host, int(port)), timeout=0.25):
                return True
        except Exception:
            pass
    return False


def _url_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(str(url).strip(), timeout=1.0) as resp:
            return int(getattr(resp, "status", 200) or 200) < 500
    except urllib.error.HTTPError as exc:
        try:
            return int(exc.code) < 500
        except Exception:
            return False
    except Exception:
        return False


@dataclass
class LocalHelperProcess:
    key: str
    helper_id: str
    process: subprocess.Popen
    log_path: Path
    log_handle: Any
    working_directory: Optional[str]
    ready_port: Optional[int]
    ready_url: Optional[str]
    ready_text: Optional[str]
    local_url: Optional[str]
    created_at: float

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def returncode(self) -> Optional[int]:
        return self.process.poll()

    def output_tail(self, *, max_chars: int = _TAIL_CHARS) -> str:
        try:
            self.log_handle.flush()
        except Exception:
            pass
        return _tail_text(self.log_path, max_chars=max_chars)

    def ready(self) -> bool:
        if not self.is_alive():
            return False
        checks = []
        if self.ready_port is not None:
            checks.append(_port_ready(int(self.ready_port)))
        if self.ready_url:
            checks.append(_url_ready(self.ready_url))
        if self.ready_text:
            checks.append(self.ready_text in self.output_tail())
        if not checks:
            return True
        return all(checks)

    def terminate(self) -> None:
        if not self.is_alive():
            try:
                self.log_handle.close()
            except Exception:
                pass
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=2)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            except Exception:
                pass
            try:
                self.process.wait(timeout=2)
            except Exception:
                pass
        finally:
            try:
                self.log_handle.close()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            "success": True,
            "helper_id": self.helper_id,
            "alive": self.is_alive(),
            "ready": self.ready(),
            "pid": int(self.process.pid),
            "returncode": self.returncode(),
            "working_directory": self.working_directory,
            "port": self.ready_port,
            "url": self.local_url,
            "output_tail": self.output_tail(),
        }


class LocalHelperRegistry:
    def __init__(self) -> None:
        self._helpers: Dict[str, LocalHelperProcess] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[LocalHelperProcess]:
        with self._lock:
            return self._helpers.get(str(key or ""))

    def put(self, helper: LocalHelperProcess) -> None:
        with self._lock:
            self._helpers[helper.key] = helper

    def close(self, key: str) -> bool:
        helper: Optional[LocalHelperProcess]
        with self._lock:
            helper = self._helpers.pop(str(key or ""), None)
        if helper is None:
            return False
        helper.terminate()
        return True

    def close_namespace(self, namespace: str) -> int:
        prefix = f"{str(namespace or '').strip()}::"
        with self._lock:
            keys = [k for k in self._helpers.keys() if k.startswith(prefix)]
        count = 0
        for key in keys:
            if self.close(key):
                count += 1
        return count

    def close_all(self) -> None:
        with self._lock:
            keys = list(self._helpers.keys())
        for key in keys:
            self.close(key)


_REGISTRY = LocalHelperRegistry()


def get_local_helper_registry() -> LocalHelperRegistry:
    return _REGISTRY


def _parse_command(command: str) -> tuple[Optional[list[str]], Optional[str]]:
    raw = str(command or "").strip()
    if not raw:
        return None, "command must be a non-empty string"
    if "`" in raw or "$(" in raw:
        return None, "shell substitution is not available here"
    try:
        argv = shlex.split(raw)
    except ValueError as exc:
        return None, f"could not parse command ({exc})"
    if not argv:
        return None, "command must be a non-empty string"
    for tok in argv:
        if tok in _SHELL_TOKENS:
            return None, f"shell operator {tok!r} is not available here"
    return argv, None


def _wait_for_ready(helper: LocalHelperProcess, *, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if helper.ready():
            return True
        if not helper.is_alive():
            return False
        time.sleep(_READY_POLL_S)
    return helper.ready()


def _start_result(helper: LocalHelperProcess, *, ready_timeout_s: float) -> Dict[str, Any]:
    ready = _wait_for_ready(helper, timeout_s=ready_timeout_s)
    payload = helper.snapshot()
    payload["success"] = bool(payload.get("alive")) and ready
    if not ready:
        payload["error"] = (
            f"helper did not satisfy its readiness contract within {ready_timeout_s:.1f}s"
            if payload.get("alive")
            else "helper exited before becoming ready"
        )
    return payload


@tool(
    description=(
        "Start one long-lived local helper process for this run (dev server, watcher, background helper) "
        "with an optional readiness contract. NOT a sandbox; helper is auto-cleaned when the run ends."
    ),
    when_to_use=(
        "When the task needs a local process that must keep running across tool calls: start a dev server, "
        "wait for it to become ready, then inspect or stop it later."
    ),
    tags=["mutating"],
    hide_args=["_registry_namespace"],
)
def local_helper_start(
    command: str,
    helper_id: str = _DEFAULT_HELPER_ID,
    working_directory: Optional[str] = None,
    ready_port: Optional[int] = None,
    ready_url: Optional[str] = None,
    ready_text: Optional[str] = None,
    ready_timeout: float = 20.0,
    _registry_namespace: str = "",
) -> Dict[str, Any]:
    argv, error = _parse_command(command)
    if error:
        return {"success": False, "error": error}
    hid = str(helper_id or "").strip() or _DEFAULT_HELPER_ID
    key = namespaced_helper_id(_registry_namespace, hid)
    registry = get_local_helper_registry()
    existing = registry.get(key)
    if existing is not None and existing.is_alive():
        return {
            "success": False,
            "error": f"helper '{hid}' is already running for this run; stop it first or use another helper_id",
            "helper_id": hid,
        }
    if existing is not None:
        registry.close(key)

    wd = str(working_directory).strip() if isinstance(working_directory, str) and working_directory.strip() else None
    log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        prefix="abstractruntime-helper-",
        suffix=".log",
        encoding="utf-8",
        delete=False,
    )
    try:
        proc = subprocess.Popen(
            argv or [],
            cwd=wd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError:
        try:
            log_file.close()
        except Exception:
            pass
        return {"success": False, "error": f"{argv[0]!r} is not a program available here"}
    except Exception as exc:
        try:
            log_file.close()
        except Exception:
            pass
        return {"success": False, "error": f"helper could not start ({exc})"}

    port_value: Optional[int]
    try:
        port_value = int(ready_port) if ready_port is not None else None
    except Exception:
        port_value = None
    url_value = str(ready_url).strip() if isinstance(ready_url, str) and ready_url.strip() else None
    text_value = str(ready_text) if ready_text is not None and str(ready_text) else None
    local_url = url_value or (f"http://127.0.0.1:{port_value}" if port_value is not None else None)
    helper = LocalHelperProcess(
        key=key,
        helper_id=hid,
        process=proc,
        log_path=Path(log_file.name),
        log_handle=log_file,
        working_directory=wd,
        ready_port=port_value,
        ready_url=url_value,
        ready_text=text_value,
        local_url=local_url,
        created_at=time.time(),
    )
    registry.put(helper)
    return _start_result(
        helper,
        ready_timeout_s=_coerce_timeout(ready_timeout, 20.0, cap=_MAX_READY_TIMEOUT_S),
    )


@tool(
    description=(
        "Check the current state of a previously started local helper process for this run."
    ),
    when_to_use=(
        "After local_helper_start, to see whether the helper is still alive, whether its readiness "
        "contract still holds, and what its recent output says."
    ),
    hide_args=["_registry_namespace"],
)
def local_helper_status(
    helper_id: str = _DEFAULT_HELPER_ID,
    _registry_namespace: str = "",
) -> Dict[str, Any]:
    hid = str(helper_id or "").strip() or _DEFAULT_HELPER_ID
    key = namespaced_helper_id(_registry_namespace, hid)
    helper = get_local_helper_registry().get(key)
    if helper is None:
        return {
            "success": False,
            "error": f"no active local helper '{hid}' for this run",
            "helper_id": hid,
        }
    return helper.snapshot()


@tool(
    description=(
        "Stop a previously started local helper process for this run, reaping its whole process group."
    ),
    when_to_use=(
        "When you are done with a helper or want to cleanly stop a server/watcher before the run ends."
    ),
    tags=["mutating"],
    hide_args=["_registry_namespace"],
)
def local_helper_stop(
    helper_id: str = _DEFAULT_HELPER_ID,
    _registry_namespace: str = "",
) -> Dict[str, Any]:
    hid = str(helper_id or "").strip() or _DEFAULT_HELPER_ID
    key = namespaced_helper_id(_registry_namespace, hid)
    helper = get_local_helper_registry().get(key)
    if helper is None:
        return {
            "success": False,
            "error": f"no active local helper '{hid}' for this run",
            "helper_id": hid,
        }
    snapshot = helper.snapshot()
    get_local_helper_registry().close(key)
    snapshot["success"] = True
    snapshot["alive"] = False
    snapshot["ready"] = False
    return snapshot


LOCAL_HELPER_TOOLS = [local_helper_start, local_helper_status, local_helper_stop]

__all__ = [
    "LOCAL_HELPER_TOOLS",
    "get_local_helper_registry",
    "local_helper_start",
    "local_helper_status",
    "local_helper_stop",
    "namespaced_helper_id",
]
