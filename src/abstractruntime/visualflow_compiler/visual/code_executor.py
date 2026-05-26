"""Sandboxed Python code execution for visual `Code` nodes.

This is a host-side utility used by the visual workflow compiler. It is kept in
the `abstractflow` package so visual workflows can be executed from other hosts
without importing the web backend implementation.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Callable, Dict

# Try to import RestrictedPython, fall back to basic execution if not available
try:
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import guarded_iter_unpack_sequence

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    RESTRICTED_PYTHON_AVAILABLE = False


class CodeExecutionError(Exception):
    """Error during code execution."""


def normalize_code_permissions(permissions: Any = "sandbox") -> str:
    """Return the canonical Code node execution permission mode."""
    raw = str(permissions or "sandbox").strip().lower().replace("-", "_")
    if raw in {"", "safe", "protected", "restricted", "sandboxed"}:
        return "sandbox"
    if raw in {"sandbox", "full_access"}:
        return raw
    if raw in {"full", "unrestricted"}:
        return "full_access"
    raise CodeExecutionError(f"Unsupported code execution permissions: {permissions!r}")


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _full_access_enabled() -> bool:
    return _truthy_env("ABSTRACTRUNTIME_CODE_FULL_ACCESS")


def ensure_code_permissions_allowed(permissions: Any = "sandbox") -> str:
    """Return the canonical mode after enforcing the current host policy."""
    mode = normalize_code_permissions(permissions)
    if mode == "full_access" and not _full_access_enabled():
        raise CodeExecutionError(
            "full_access code execution is disabled by host policy. "
            "Set ABSTRACTRUNTIME_CODE_FULL_ACCESS=1 only for trusted local deployments."
        )
    return mode


def get_code_execution_policy() -> Dict[str, Any]:
    """Return the effective Code-node execution policy for thin-client discovery."""
    full_access = _full_access_enabled()
    return {
        "contract": "code_execution_policy_v1",
        "version": 1,
        "available": True,
        "default_mode": "sandbox",
        "modes": [
            {
                "id": "sandbox",
                "label": "Sandbox",
                "available": True,
                "default": True,
                "safety": "restricted_python",
                "description": "Protected Python execution with imports and unsafe constructs disabled.",
            },
            {
                "id": "full_access",
                "label": "Full access",
                "available": full_access,
                "requires_host_policy": True,
                "safety": "trusted_host",
                **(
                    {}
                    if full_access
                    else {
                        "disabled_reason": "Disabled by execution-host policy.",
                        "config_hint": "Set ABSTRACTRUNTIME_CODE_FULL_ACCESS=1 only for trusted local deployments.",
                    }
                ),
            },
        ],
    }


def validate_code(code: str) -> None:
    """Validate Python code for safety.

    Raises:
        CodeExecutionError: If code contains disallowed constructs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeExecutionError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        # Disallow imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeExecutionError("Imports are not allowed")

        # Disallow exec/eval
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("exec", "eval", "compile", "__import__"):
                raise CodeExecutionError(f"'{node.func.id}' is not allowed")

        # Disallow dunder attributes
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise CodeExecutionError(f"Access to dunder attributes ('{node.attr}') is not allowed")


def create_code_handler(code: str, function_name: str = "transform", *, permissions: Any = "sandbox") -> Callable[[Any], Any]:
    """Create a handler function from user-provided Python code.

    The code should define a function that takes input data and returns a result.
    """
    mode = ensure_code_permissions_allowed(permissions)
    if mode == "full_access":
        return _create_full_access_handler(code, function_name)

    validate_code(code)

    if RESTRICTED_PYTHON_AVAILABLE:
        return _create_restricted_handler(code, function_name)
    return _create_basic_handler(code, function_name)


def _create_full_access_handler(code: str, function_name: str) -> Callable[[Any], Any]:
    """Create a full Python handler for explicitly trusted local deployments."""
    try:
        byte_code = compile(code, filename="<user_code>", mode="exec")
    except SyntaxError as e:
        raise CodeExecutionError(f"Syntax error: {e}") from e

    def handler(input_data: Any) -> Any:
        module_globals: Dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(byte_code, module_globals, module_globals)
        except Exception as e:
            raise CodeExecutionError(f"Execution error: {e}") from e

        func = module_globals.get(function_name)
        if func is None:
            raise CodeExecutionError(f"Function '{function_name}' not defined in code")
        if not callable(func):
            raise CodeExecutionError(f"'{function_name}' is not a callable function")

        try:
            return func(input_data)
        except Exception as e:
            raise CodeExecutionError(f"Runtime error: {e}") from e

    return handler


def _create_restricted_handler(code: str, function_name: str) -> Callable[[Any], Any]:
    """Create handler using RestrictedPython for sandboxed execution."""
    byte_code = compile_restricted(code, filename="<user_code>", mode="exec")

    if getattr(byte_code, "errors", None):
        errors = getattr(byte_code, "errors", None)
        if isinstance(errors, list) and errors:
            raise CodeExecutionError(f"Compilation errors: {'; '.join(errors)}")

    def handler(input_data: Any) -> Any:
        restricted_globals = {
            "__builtins__": safe_builtins,
            "_getiter_": default_guarded_getiter,
            "_getitem_": default_guarded_getitem,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            # Allow some safe built-ins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
            "type": type,
            "print": lambda *args, **kwargs: None,  # Silent print
        }

        local_vars: Dict[str, Any] = {}

        try:
            exec(byte_code, restricted_globals, local_vars)
        except Exception as e:
            raise CodeExecutionError(f"Execution error: {e}") from e

        # `exec(..., globals, locals)` stores definitions in `locals`, but functions
        # resolve globals against the `globals` dict. Make user-defined helpers
        # (and other top-level values) available to `transform`.
        reserved = {"__builtins__", "_getiter_", "_getitem_", "_iter_unpack_sequence_"}
        for name, value in local_vars.items():
            if name in reserved:
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if name not in restricted_globals:
                restricted_globals[name] = value

        func = local_vars.get(function_name)
        if func is None:
            raise CodeExecutionError(f"Function '{function_name}' not defined in code")
        if not callable(func):
            raise CodeExecutionError(f"'{function_name}' is not a callable function")

        try:
            return func(input_data)
        except Exception as e:
            raise CodeExecutionError(f"Runtime error: {e}") from e

    return handler


def _create_basic_handler(code: str, function_name: str) -> Callable[[Any], Any]:
    """Create handler with basic (less secure) execution.

    Used as fallback when RestrictedPython is not available.
    """
    try:
        byte_code = compile(code, filename="<user_code>", mode="exec")
    except SyntaxError as e:
        raise CodeExecutionError(f"Syntax error: {e}") from e

    def handler(input_data: Any) -> Any:
        limited_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "isinstance": isinstance,
                "type": type,
                "print": lambda *args, **kwargs: None,
                "True": True,
                "False": False,
                "None": None,
            }
        }

        local_vars: Dict[str, Any] = {}

        try:
            exec(byte_code, limited_globals, local_vars)
        except Exception as e:
            raise CodeExecutionError(f"Execution error: {e}") from e

        # Keep the same semantics as normal Python modules: helper functions and
        # top-level constants defined alongside `transform()` should be visible
        # at runtime. Avoid letting user code replace `__builtins__`.
        reserved = {"__builtins__"}
        for name, value in local_vars.items():
            if name in reserved:
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if name not in limited_globals:
                limited_globals[name] = value

        func = local_vars.get(function_name)
        if func is None:
            raise CodeExecutionError(f"Function '{function_name}' not defined in code")
        if not callable(func):
            raise CodeExecutionError(f"'{function_name}' is not a callable function")

        try:
            return func(input_data)
        except Exception as e:
            raise CodeExecutionError(f"Runtime error: {e}") from e

    return handler
