"""Flow-level function library — tier 2 of the helper-function redesign.

A VisualFlow may carry named, reusable helper FUNCTIONS at the flow level:

    "functions": [
        {"name": "build_again", "code": "def build_again(state, max_fix):\n    ...",
         "kind": "checker", "description": "..."}
    ]

Pin expressions (tier 1, ``pin_expressions.py``) may then CALL these functions:
``node.data.pinExpressions = {"condition": "build_again(vars.state, 3)"}``.
Tier 2 is deliberately pure sugar over tier 1 — ONE runtime lane:

- The library compiles ONCE at flow build into a single shared namespace
  (RestrictedPython exec mode, the SAME ``_CodeNodePolicy`` and guard set as
  code-node bodies — one sandbox, one rule). Every declared function's
  ``__globals__`` IS that shared namespace, so functions can call each other
  (late binding) and shared private ``_helpers`` defined beside them.
- Expressions receive the DECLARED names only (privates stay library-internal).
- Failures are LOUD and ATTRIBUTED at build time (``flow function 'x' ...``),
  exactly like an unparseable pin expression — the compile boundary is the
  honesty boundary.

Skew safety (same story as ``pinExpressions``): ``functions`` is a top-level
flow field an older runtime's parser never reads. Old runtimes also never
read ``pinExpressions``, so the pins that would have CALLED these functions
fall back to their defaults — bounded, falsy conditions; the bundle-level
``metadata.min_runtime`` gate is the loud refusal for bundles that depend on
this tier (fail-dangerous skew is the gate's job, not this module's).

Name rules (refusals, never silent fixes):
- names must be valid Python identifiers, non-dunder, and may not collide
  with the expression environment's reserved names (``vars``, ``value``),
  the sandbox helper set (``parse_json`` …), or safe builtins (``len`` …);
- ONE namespace: duplicate declared names refuse, and so does any entry
  touching a name another entry already owns (including private ``_helper``
  collisions — two entries silently last-write-winning each other's helper
  is the drift class this module exists to prevent);
- each entry's code MUST define its declared name as a callable.

PURITY IS ENFORCED, NOT ASSUMED (adversary finding, 2026-07-26). The shared
namespace is compiled ONCE and its callables' ``__globals__`` persist for the
whole cached-spec lifetime (a gateway compiles a spec once and reuses it
across every run/tick/resume, and across tenants when one bundle serves many).
A library function that acquired persistent writable state would therefore
LEAK it across runs — silently. In a no-import sandbox, persistent writable
state has exactly three sources, and ``_validate_library_purity`` refuses all
three at build (loud, attributed):
  1. ``global`` / ``nonlocal`` — the only way to STORE into the shared
     namespace / an enclosing scope (a rebind that corrupts other functions);
  2. a top-level statement other than ``def`` (a module-level mutable binding
     like ``_cache = {}`` that call-time subscript-mutation then persists);
  3. a MUTABLE default argument (``def f(x, _c={})`` — the classic trap: the
     default object is created once at def time and shared across calls).
Closing these three closes the CLASS: every remaining name a call binds is a
fresh local / per-call closure (run-scoped, exactly like the ``get_var`` lane
and tier-1 expressions), and the def objects + guard globals are immutable.
This makes the module's "one namespace, no silent overwrites" true at CALL
time, not only at build. Top-level constants are deliberately disallowed
(put them inside a function) so the rule stays a simple, complete closure.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List, Mapping, Optional

from .code_executor import (
    RESTRICTED_PYTHON_AVAILABLE,
    CodeExecutionError,
    _apply,
    _inplacevar,
    sandbox_helper_globals,
    validate_code,
)

if RESTRICTED_PYTHON_AVAILABLE:  # pragma: no branch
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import (
        full_write_guard,
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
        safer_getattr,
    )

    from .code_executor import _CodeNodePolicy


class FunctionLibraryError(CodeExecutionError):
    """A flow function failed to validate or compile (build-time, attributed)."""


def normalize_flow_functions(raw: Any) -> List[Dict[str, str]]:
    """``flow.functions`` → [{name, code}], dropping junk entries.

    Tolerant on shape (hosts may carry nulls / extra fields) but strict on
    type: only entries with a non-empty string name AND code survive. A
    dropped entry degrades exactly like the old-runtime skew path — the
    expressions that call it fail loudly at build ("name ... is not defined"),
    never silently misbehave.
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        code = entry.get("code")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(code, str) or not code.strip():
            continue
        out.append({"name": name.strip(), "code": code})
    return out


def reserved_function_names() -> set[str]:
    """Names a library function may not claim.

    ``vars``/``value`` are the expression environment's own bindings; the
    sandbox helpers and safe builtins are the shared vocabulary both lanes
    (code nodes + expressions) already teach — shadowing ``len`` or
    ``parse_json`` flow-locally would make the same expression mean different
    things in different flows.
    """
    reserved = {"vars", "value"}
    reserved.update(sandbox_helper_globals().keys())
    if RESTRICTED_PYTHON_AVAILABLE:
        # safe_builtins is a mapping; read its keys directly. A silent
        # try/except here would NARROW the reserved set on any failure —
        # letting a function be named `repr`/`hex`/… and shadow that builtin
        # flow-locally, the exact drift this set exists to prevent. Fail loud
        # instead (adversary finding 3).
        reserved.update(safe_builtins.keys())
    return reserved


# Default-argument literal nodes that are MUTABLE: created once at def time
# and shared across every call (the classic mutable-default trap). Immutable
# literals (int/str/tuple/None via ast.Constant) are fine as defaults.
_MUTABLE_DEFAULT_NODES = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)


def _validate_library_purity(name: str, code: str) -> None:
    """Refuse the three sources of persistent cross-run writable state.

    See the module docstring: in a no-import sandbox those are ``global`` /
    ``nonlocal``, top-level non-``def`` statements, and mutable default
    arguments. Refusing all three makes every library function stateless by
    construction, so the process-lifetime shared namespace can never leak
    state across runs/tenants. Build-time, loud, attributed by function name.

    FIRST runs the SAME ``validate_code`` the code-node lane uses (imports,
    exec/eval/compile/__import__, dunder-attribute access) — one source, no
    drift ("one sandbox, one rule"). The library lane previously skipped it
    and let in-body imports fail only at CALL time; now it refuses at build
    exactly like a code node. The purity rules below are the tier-2 ADDITION
    on top of that shared baseline.
    """
    try:
        validate_code(code)
    except CodeExecutionError as e:
        raise FunctionLibraryError(f"flow function {name!r}: {e}") from e

    # validate_code already parsed successfully; re-parse for the purity walk
    # (cheap, and keeps this function self-contained).
    tree = ast.parse(code)

    # (2) Top level may hold only `def`s (and a module docstring). Anything
    # else could bind a module-level mutable that call-time mutation persists.
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue  # a bare string (docstring) binds nothing
        raise FunctionLibraryError(
            f"flow function {name!r}: only `def` statements are allowed at the top level "
            f"(got {type(stmt).__name__}) — flow functions must be pure (no module-level "
            "state that would leak across runs); put constants inside a function"
        )

    # (1) no `global`/`nonlocal` anywhere; (3) no mutable default arguments
    # anywhere (including nested defs and lambdas).
    for sub in ast.walk(tree):
        if isinstance(sub, (ast.Global, ast.Nonlocal)):
            raise FunctionLibraryError(
                f"flow function {name!r}: `global`/`nonlocal` is not allowed — a flow "
                "function must not write persistent state (it would leak across runs)"
            )
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = getattr(sub, "args", None)
            if args is None:
                continue
            defaults = list(getattr(args, "defaults", []) or [])
            defaults += [d for d in (getattr(args, "kw_defaults", []) or []) if d is not None]
            for d in defaults:
                if isinstance(d, _MUTABLE_DEFAULT_NODES):
                    raise FunctionLibraryError(
                        f"flow function {name!r}: a mutable default argument "
                        f"({type(d).__name__}) is not allowed — it is created once and "
                        "shared across calls/runs; use None and build the value inside"
                    )


def compile_function_library(
    functions: Any,
) -> Dict[str, Callable[..., Any]]:
    """Compile the flow's function entries into {declared name: callable}.

    All entries execute into ONE shared namespace (so functions can call each
    other and shared private helpers); only DECLARED names are exported to
    expressions. Raises :class:`FunctionLibraryError` on the first invalid
    entry — flow build fails loudly naming the function.
    """
    entries = normalize_flow_functions(functions)
    if not entries:
        return {}
    if not RESTRICTED_PYTHON_AVAILABLE:
        raise FunctionLibraryError(
            "flow functions require RestrictedPython (the expression tier's sandbox)"
        )

    reserved = reserved_function_names()

    # The shared namespace doubles as every def's __globals__: the guard set
    # mirrors the code-node sandbox EXACTLY (one sandbox, one rule) or bodies
    # die at call time with bare sandbox-internal NameErrors.
    shared_ns: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "_getiter_": default_guarded_getiter,
        "_getitem_": default_guarded_getitem,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_getattr_": safer_getattr,
        "_write_": full_write_guard,
        "_inplacevar_": _inplacevar,
        "_apply_": _apply,
        **sandbox_helper_globals(),
    }
    baseline_keys = set(shared_ns.keys())

    declared: List[str] = []
    owner_of: Dict[str, str] = {}  # non-baseline key -> declaring entry name

    for entry in entries:
        name = entry["name"]
        if not name.isidentifier() or name.startswith("__"):
            raise FunctionLibraryError(
                f"flow function name {name!r} is not a plain identifier"
            )
        if name in reserved:
            raise FunctionLibraryError(
                f"flow function name {name!r} shadows a sandbox builtin/helper — pick another name"
            )
        if name in owner_of:
            raise FunctionLibraryError(
                f"duplicate flow function name {name!r} (already defined by entry {owner_of[name]!r})"
            )

        # Purity BEFORE compile/exec: refuse persistent-state constructs so a
        # library function can never leak across runs (see module docstring).
        _validate_library_purity(name, entry["code"])

        try:
            byte_code = compile_restricted(
                entry["code"], filename=f"<flowfn:{name}>", mode="exec", policy=_CodeNodePolicy
            )
        except SyntaxError as e:
            raise FunctionLibraryError(f"flow function {name!r} does not parse: {e}") from e
        except Exception as e:
            raise FunctionLibraryError(
                f"flow function {name!r} could not be compiled ({type(e).__name__}: {e})"
            ) from e
        errors = getattr(byte_code, "errors", None)
        if isinstance(errors, list) and errors:
            raise FunctionLibraryError(
                f"flow function {name!r} refused: {'; '.join(str(x) for x in errors)}"
            )

        # Snapshot identities so overwrites of another entry's names are
        # detectable (a plain after-minus-before set misses redefinitions).
        before = {k: id(v) for k, v in shared_ns.items()}
        try:
            exec(byte_code, shared_ns)  # noqa: S102 - RestrictedPython bytecode + guarded globals
        except Exception as e:
            raise FunctionLibraryError(
                f"flow function {name!r} failed at definition time: {type(e).__name__}: {e}"
            ) from e

        touched = [
            k
            for k, v in shared_ns.items()
            if (k not in before or before[k] != id(v)) and not (k.startswith("__") and k.endswith("__"))
        ]
        for key in touched:
            if key in baseline_keys or key in reserved:
                raise FunctionLibraryError(
                    f"flow function entry {name!r} redefines {key!r}, which shadows a sandbox builtin/helper"
                )
            prior_owner = owner_of.get(key)
            if prior_owner is not None and prior_owner != name:
                raise FunctionLibraryError(
                    f"flow function entry {name!r} redefines {key!r}, already owned by entry {prior_owner!r} — "
                    "one namespace, no silent overwrites"
                )
            owner_of[key] = name

        fn = shared_ns.get(name)
        if fn is None or not callable(fn):
            raise FunctionLibraryError(
                f"flow function entry {name!r} must define a callable named {name!r} "
                "(the code's `def` name must match the declared name)"
            )
        declared.append(name)

    return {name: shared_ns[name] for name in declared}
