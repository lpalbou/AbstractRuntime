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
    from RestrictedPython import RestrictingNodeTransformer, compile_restricted, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import (
        full_write_guard,
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
        safer_getattr,
    )

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    RESTRICTED_PYTHON_AVAILABLE = False


class CodeExecutionError(Exception):
    """Error during code execution."""


# Names the sandbox machinery itself binds into the execution namespace.
# User code may never define or reference them: shadowing a guard would let
# code replace the sandbox's own access checks.
_GUARD_NAMES = frozenset(
    {
        "_getattr_",
        "_getitem_",
        "_getiter_",
        "_write_",
        "_print_",
        "_print",
        "_inplacevar_",
        "_unpack_sequence_",
        "_iter_unpack_sequence_",
        "_apply_",
        "__builtins__",
    }
)


if RESTRICTED_PYTHON_AVAILABLE:

    class _CodeNodePolicy(RestrictingNodeTransformer):
        """RestrictedPython policy for visual Code nodes.

        The generated wrapper binds the node's input as ``_input`` and flow
        authors legitimately use private (leading-underscore) helper names,
        which the default policy refuses outright. This policy allows
        single-leading-underscore identifiers while keeping the base rules
        for everything else: guard helper names stay reserved (shadowing a
        guard would disable the sandbox's own access checks), and dunder /
        ``__roles__`` names remain refused by the base check.

        Restored 2026-07-17: this leading-underscore allowance existed in
        the original abstractflow executor (2026-02-20 note) but was lost
        when the compiler moved into abstractruntime. The gap was invisible
        while RestrictedPython was absent (the ImportError fallback ran the
        basic handler); the moment any package installed RestrictedPython,
        every generated ``def transform(_input)`` wrapper failed to compile.
        """

        def check_name(self, node, name, allow_magic_methods=False):  # type: ignore[override]
            if name is None:
                return
            if name in _GUARD_NAMES:
                self.error(node, f'"{name}" is a reserved sandbox name')
                return
            if name.startswith("_") and not name.startswith("__") and not name.endswith("__roles__"):
                return
            super().check_name(node, name, allow_magic_methods)


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


def _inplacevar(op: str, x: Any, y: Any) -> Any:
    """Guard for augmented assignment on plain names (``total += 1``).

    RestrictedPython compiles ``x <op>= y`` into ``_inplacevar_('<op>=', x, y)``
    but ships no implementation (hosts supply their own). Delegating to the
    operator module keeps Python semantics exactly (including ``+=`` on lists).
    """
    import operator as _op

    table = {
        "+=": _op.iadd,
        "-=": _op.isub,
        "*=": _op.imul,
        "/=": _op.itruediv,
        "//=": _op.ifloordiv,
        "%=": _op.imod,
        "**=": _op.ipow,
        "@=": _op.imatmul,
        "&=": _op.iand,
        "|=": _op.ior,
        "^=": _op.ixor,
        "<<=": _op.ilshift,
        ">>=": _op.irshift,
    }
    fn = table.get(op)
    if fn is None:
        raise CodeExecutionError(f"Unsupported in-place operator: {op}")
    return fn(x, y)


def _apply(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Guard for calls with starred arguments (``f(*args, **kwargs)``)."""
    return func(*args, **kwargs)


def _sandbox_parse_json(text: Any) -> Any:
    """``parse_json(text)`` — the sandbox's JSON reader.

    The sandbox has no imports, so JSON parsing was impossible in code nodes
    and would have been impossible in pin expressions (the design verdict's
    amendment 4: "parser" helpers were unbuildable inline). Errors stay in
    the CodeExecutionError family so they surface loudly with attribution.
    """
    import json

    if isinstance(text, (dict, list)):
        # Already-parsed payloads pass through: upstream nodes frequently
        # hand structured data to bodies written for the string case.
        return text
    try:
        return json.loads(str(text))
    except Exception as e:
        head = str(text)[:80]
        raise CodeExecutionError(f"parse_json: invalid JSON ({e}) — input head: {head!r}") from e


def _sandbox_to_json(obj: Any, *, indent: Any = None) -> str:
    """``to_json(obj)`` — the sandbox's JSON writer.

    ``default=str`` is a DELIBERATE, documented trade-off (cycle-2 decision,
    kept): the writer is TOTAL — it never raises on an exotic value (datetime,
    a domain object), stringifying it instead. A raising ``to_json`` inside a
    pin expression would fail the consumer's step with a confusing error over
    what is usually a best-effort "render this as text" call; a total writer is
    the right posture for a one-liner helper shared by both sandboxes. The cost
    is that exotic values are LOSSY (their ``str()``, not a structured form) —
    authors needing an exact round-trip must hand JSON-native types. No
    security angle: the object is the author's own run data, ``str()`` leaks
    nothing the author could not already read.
    """
    import json

    return json.dumps(obj, default=str, indent=indent if isinstance(indent, int) else None)


def _sandbox_shq(value: Any) -> str:
    """``shq(value)`` — POSIX single-quote escape for shell composition.

    A flow that composes a command for ``execute_command`` must quote every
    interpolated value: ``"cd '" + shq(path) + "'"``. The escape itself is
    three characters of ``str.replace`` and needs no imports, so this is NOT
    the ``parse_json`` case (impossible inside the sandbox). It is promoted
    for a different and stronger reason: the runtime ships the dangerous
    primitive (``execute_command``), so it owes callers the safe quoting
    primitive next to it. Hand-rolled per flow, the escape gets copied — it
    was inlined NINE times in one bundle before it was factored — and ONE
    divergent copy is a command injection on a path containing a quote.

    Semantics, deliberately: ``None`` becomes ``""``, but a falsy NON-None
    value keeps its text (``shq(0)`` is ``"0"``; ``s or ""`` would erase it).
    Total — never raises, so a composer cannot fail its step on odd input.
    """
    text = "" if value is None else str(value)
    return text.replace("'", "'\\''")


# Depth cap for text_of's fold. Envelopes nest 2-3 levels; the cap only makes
# the walk TOTAL on a pathological (or cyclic-by-reference) payload so it can
# never spin the tick thread.
_TEXT_OF_MAX_STEPS = 400


def _sandbox_text_of(raw: Any) -> str:
    """``text_of(envelope)`` — the text inside any tool-result envelope.

    Promoted because the shapes it decodes are the RUNTIME'S OWN, not any
    flow's, and they have changed under flows before:

    - the direct lane delivers a bare output dict (``{stdout, stderr, ...}``)
    - older tools return a plain string
    - durable compaction leaves only the ``*_preview`` keys
    - the approval-resume lane nests ``{mode, results: [...]}``

    A flow-local reader written against three of those four silently drops
    text the day the fourth appears — which is exactly how this function grew
    its shapes. Knowledge of an envelope belongs to whoever defines it. Same
    architecture as the runtime-owned ``stringify_json`` /
    ``render_agent_trace_markdown`` the visual layer already delegates to.

    Total by construction: never raises, returns ``""`` when there is no text.
    NOT overridable — ``function_library`` refuses a flow function that shadows
    a helper name, so a bundle cannot silently re-fork this reader; that
    refusal is deliberate (a second copy is how the shapes drifted apart in the
    first place) and it fails LOUDLY at flow build, naming the collision.
    """
    parts: list[str] = []
    stack: list[Any] = [raw]
    steps = 0
    while stack and steps < _TEXT_OF_MAX_STEPS:
        steps += 1
        cur = stack.pop()
        if isinstance(cur, str):
            if cur:
                parts.append(cur)
            continue
        if isinstance(cur, dict):
            for key in ("stdout", "stderr", "stdout_preview", "stderr_preview"):
                value = cur.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
            for key in ("output", "result", "results", "payload"):
                if key in cur:
                    stack.append(cur.get(key))
            continue
        if isinstance(cur, list):
            for item in cur:
                stack.append(item)
    return "\n".join(parts)


def sandbox_helper_globals() -> Dict[str, Any]:
    """The convenience names granted to BOTH code-node bodies and pin
    expressions — one source so the two sandboxes can never drift (the
    four-copy-contract lesson applied preemptively).

    NOTE on precedence (unchanged, stated so the next reader does not have to
    rediscover it): a name here wins over a same-named helper defined INSIDE a
    code-node body, because ``_create_restricted_handler`` only copies a local
    definition into the execution globals ``if name not in restricted_globals``.
    Keep additions here rare, generic, and unlikely to collide.
    """
    return {
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
        "parse_json": _sandbox_parse_json,
        "to_json": _sandbox_to_json,
        "shq": _sandbox_shq,
        "text_of": _sandbox_text_of,
        "print": lambda *args, **kwargs: None,  # Silent print
    }


def _create_restricted_handler(code: str, function_name: str) -> Callable[[Any], Any]:
    """Create handler using RestrictedPython for sandboxed execution."""
    try:
        byte_code = compile_restricted(
            code, filename="<user_code>", mode="exec", policy=_CodeNodePolicy
        )
    except SyntaxError as e:
        # compile_restricted raises SyntaxError carrying the policy errors
        # (RestrictedPython >= 7 behavior); surface them as our error type so
        # callers keep one exception surface.
        raise CodeExecutionError(f"Compilation errors: {e}") from e

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
            "_unpack_sequence_": guarded_unpack_sequence,
            "_getattr_": safer_getattr,
            "_write_": full_write_guard,
            "_inplacevar_": _inplacevar,
            "_apply_": _apply,
            # Convenience builtins + parse_json/to_json — shared with the pin
            # expression sandbox (one source, see sandbox_helper_globals).
            **sandbox_helper_globals(),
        }

        local_vars: Dict[str, Any] = {}

        try:
            exec(byte_code, restricted_globals, local_vars)
        except Exception as e:
            raise CodeExecutionError(f"Execution error: {e}") from e

        # `exec(..., globals, locals)` stores definitions in `locals`, but functions
        # resolve globals against the `globals` dict. Make user-defined helpers
        # (and other top-level values) available to `transform`.
        reserved = set(_GUARD_NAMES)
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
                # Same convenience set as the restricted lane (one source),
                # plus the literal names bare exec needs.
                **sandbox_helper_globals(),
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
