"""Inline pin expressions — tier 1 of the helper-function redesign.

A data input pin may carry a small Python EXPRESSION instead of (or on top
of) a wire. The expression is stored on the node as
``node.data.pinExpressions: {pin_id: "vars.fix_cycles < 3"}`` — a field of
its own, deliberately OUTSIDE ``pinDefaults``.

WHY the separate field is load-bearing (the verdict's amendment 1, the hard
GO condition): an older compiler that has never heard of pin expressions
simply never reads the key — the pin resolves to its default/absent value,
a while-condition stays falsy, loops stay bounded. Encoding expressions as
a sentinel dict inside ``pinDefaults`` was rejected because an old compiler
would pass the dict through as a TRUTHY literal — a while(condition) would
spin to the iteration cap EXECUTING ITS BODY (agent calls in this corpus):
fail-dangerous, not merely stale.

Evaluation model (adversary-settled, 2026-07-25):
- Expressions are evaluated AT INPUT RESOLUTION on the consuming node — the
  two resolver sites in ``executor.py`` (``_ensure_node_output`` for pure
  consumers, ``_create_data_aware_handler`` for exec consumers). No synthetic
  nodes are injected: volatility comes free (input resolution happens per
  pull / per execution, so a while-condition expression re-reads
  ``vars.*`` fresh each iteration, exactly like the ``get_var`` chains it
  replaces), and there is no graph surgery to keep honest.
- The environment is deliberately tiny: ``vars`` (READ-ONLY view of run
  vars, attribute or subscript access), ``value`` (whatever the pin would
  have resolved to without the expression), plus the shared sandbox helpers
  (``parse_json`` / ``to_json``) and the code-node builtin set. Full
  RestrictedPython, eval mode — statements and imports cannot appear
  (expressions, not programs). Expression-shaped sub-forms the eval policy
  permits (lambdas, comprehensions, conditional/walrus expressions) remain
  available: ``sorted(vars.items, key=lambda r: r["ts"])`` is still an
  expression.
- Failures are LOUD and ATTRIBUTED: a raising expression fails the
  consumer's step with a message naming ``<node>.<pin>`` and an expression
  preview — strictly better than today's zero-trace pure lane.

Precedence rule (one sentence): an expression REPLACES the pin's resolved
value; whatever the pin would have resolved to without it (wire value,
cloned default, or a same-named key from the ambient exec payload on the
exec lane) is exposed to the expression as ``value``. This holds UNIFORMLY
for multi-entry nodes: ``apply_pin_expressions`` runs strictly LAST, after
wires, per-path route overrides, and defaults have all resolved — so
whatever a route override selected arrives as ``value`` and the expression
replaces it. Precedence is well-defined without any expression↔route
special-casing.

Posture on run-state exposure (adversary cycle 2, 2026-07-25) — ONE posture,
stated once:
- The ``vars`` VIEW is an evaluation-only artifact and NEVER escapes into a
  node output / persisted run state. A view returned bare OR nested in a
  container (``[vars]``, ``{"v": vars}``, ``(vars,)``) is stripped to a plain
  shallow dict before the result leaves ``evaluate`` (``_strip_views``). This
  is the one leak unique to the expression tier and it is closed here: a view
  in a persisted output would alias every run var and is not JSON
  serializable (node outputs are saved into ``run.vars["_temp"]`` across
  pause/resume).
- NESTED values are handed out LIVE and method-call mutation of them
  (``vars.items.append(...)``, ``vars.state.update(...)``) is REACHABLE in
  eval mode. This is deliberate PARITY with the pre-existing ``get_var`` ->
  code-node lane, which returns the same live objects (``_create_get_var_handler``
  returns ``run.vars[name]`` uncopied) and lets a code body mutate them
  identically. The expression tier adds NO new mutation surface beyond that
  lane; blanket deep-copying results was rejected (it diverges from the
  code-node posture — two sandboxes, one rule is the standing law — prices
  every hot-loop condition, and would newly CRASH on the non-deepcopyable
  objects run vars may legitimately hold). Expressions SHOULD be pure reads;
  that discipline is the author's, exactly as for code bodies. The top-level
  add/remove/replace guards on ``_ReadOnlyVars`` still hold (you cannot
  ``vars.x = 1`` or delete a run var) — a guarantee get_var cannot even
  express — but they are NOT a claim of deep immutability.

Resource posture (shared, known, no watchdog): an expression runs on the tick
thread with the same builtins the code-node sandbox grants (``range``,
``sum``, ``**`` …), so ``sum(range(10**9))`` blocks the tick exactly as the
identical code body would. This is IDENTICAL exposure to code nodes — same
lane, same one-sandbox rule — and is deliberately left to a future shared
watchdog rather than an expression-only guard (a per-lane timeout would be
the drift the one-sandbox rule forbids). Literal blacklisting
(``refuse range(`` over big constants) is theater — trivially evaded by
computing the bound — and is not added.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from .code_executor import (
    RESTRICTED_PYTHON_AVAILABLE,
    CodeExecutionError,
    _apply,
    sandbox_helper_globals,
)

if RESTRICTED_PYTHON_AVAILABLE:  # pragma: no branch
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
        safer_getattr,
    )

    # ONE sandbox policy for both lanes (runtime seat's rule, dm 2026-07-25):
    # expressions compile under the SAME RestrictedPython policy object as
    # code-node bodies — two sandboxes with independently-drifting rules is
    # the diary_type-clamp class of bug. The policy's exec-lane allowances
    # (leading-underscore identifiers minus reserved guard names) apply
    # identically in eval mode.
    from .code_executor import _CodeNodePolicy

# One name for the compiled-evaluator shape so the executor's signatures and
# this module's tables spell the same contract: (value, run_vars) -> Any.
PinExpressionEvaluator = Callable[[Any, Optional[Mapping[str, Any]]], Any]


class PinExpressionError(CodeExecutionError):
    """A pin expression failed to compile or evaluate.

    Subclasses CodeExecutionError so every existing caller that treats code
    failures loudly treats expression failures the same way.
    """


class _ReadOnlyVars:
    """Read-only ``vars`` view for expressions: ``vars.build_state`` or
    ``vars["build_state"]``.

    Only the TOP mapping is wrapped: an expression cannot ADD, REMOVE, or
    REPLACE a run var (``vars.x = 1`` / ``vars["x"] = 1`` both raise here) —
    a guarantee the ``get_var`` node cannot even express, kept because writing
    top-level state is ``set_var``'s job, on the exec plane, where order is
    visible.

    This is NOT deep immutability, and the docstring says so plainly (cycle 2
    honesty): nested values pass through LIVE, so ``vars.items.append(...)``
    or ``vars.state.update(...)`` DO mutate run state. That is the SAME
    exposure the ``get_var`` → code-node chain already has (get_var returns
    the live object uncopied), so the expression tier adds no new mutation
    surface. Deep-freezing/deep-copying every read was priced and rejected
    (hot-loop conditions read per iteration; it would also diverge from the
    code-node posture and crash on non-copyable vars). Expressions SHOULD be
    pure reads — the author's discipline, as for code bodies. The one hard
    guarantee beyond parity is that the VIEW OBJECT itself never escapes a
    result (``_strip_views`` in ``compile_pin_expression``).
    """

    __slots__ = ("_vars",)

    def __init__(self, run_vars: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_vars", run_vars)

    def __getattr__(self, name: str) -> Any:
        v = object.__getattribute__(self, "_vars")
        try:
            return v[name]
        except KeyError:
            raise KeyError(
                f"unknown run var '{name}' (known: {', '.join(sorted(k for k in v.keys() if not k.startswith('_')))[:200] or 'none yet'})"
            ) from None

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(str(name))

    def get(self, name: str, default: Any = None) -> Any:
        v = object.__getattribute__(self, "_vars")
        return v.get(name, default)

    def __contains__(self, name: str) -> bool:
        v = object.__getattribute__(self, "_vars")
        return name in v

    def __setattr__(self, name: str, _value: Any) -> None:
        raise PinExpressionError("vars is read-only inside a pin expression — use a Set Variable node to write state")

    def __setitem__(self, name: str, _value: Any) -> None:
        raise PinExpressionError("vars is read-only inside a pin expression — use a Set Variable node to write state")

    def __repr__(self) -> str:  # keep error payloads readable, never dump values
        v = object.__getattribute__(self, "_vars")
        return f"<vars: {', '.join(sorted(v.keys()))}>"


# Depth cap for the view-strip walk. Expressions construct results 1-2 levels
# deep at most (``[vars]``, ``{"v": vars}``); the cap only exists to make the
# walk TOTAL on a pathological cyclic/deep result so it can never hang the
# tick thread. A view deeper than this is not a realistic authoring vector.
_STRIP_MAX_DEPTH = 25


def _strip_views(obj: Any, _depth: int = 0) -> Any:
    """Remove any ``_ReadOnlyVars`` view from an expression result.

    The view aliases every live run var and is not JSON serializable, so it
    must never reach a node output / persisted run state — and a view nested
    inside a container (``[vars]``, ``{"v": vars}``, ``(vars,)``, ``{vars}``)
    is the exact leak the bare top-level ``isinstance`` unwrap alone missed
    (cycle-2 finding). A view at ANY position becomes a shallow dict copy of
    the top mapping (identical to what bare ``vars`` yields).

    This strips ONLY the view object; nested DATA is left live (parity with
    the get_var → code-node lane — see module docstring). Identity-preserving:
    a container with no view inside is returned UNCHANGED, so a legitimate
    ``sorted(vars.items)`` result is never re-copied on a hot loop path; only
    the branch that actually carries a view is rebuilt. Depth-capped so a
    cyclic/deep result terminates.
    """
    if isinstance(obj, _ReadOnlyVars):
        return dict(object.__getattribute__(obj, "_vars"))
    if _depth >= _STRIP_MAX_DEPTH:
        return obj
    if isinstance(obj, dict):
        changed = False
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            sv = _strip_views(v, _depth + 1)
            if sv is not v:
                changed = True
            out[k] = sv
        return out if changed else obj
    if isinstance(obj, list):
        stripped = [_strip_views(v, _depth + 1) for v in obj]
        return stripped if any(s is not v for s, v in zip(stripped, obj)) else obj
    if isinstance(obj, tuple):
        stripped = tuple(_strip_views(v, _depth + 1) for v in obj)
        return stripped if any(s is not v for s, v in zip(stripped, obj)) else obj
    if isinstance(obj, (set, frozenset)):
        stripped = [_strip_views(v, _depth + 1) for v in obj]
        if all(s is v for s, v in zip(stripped, obj)):
            return obj
        try:
            return type(obj)(stripped)
        except TypeError:
            # A stripped view became an unhashable dict: a set-of-views is
            # degenerate — normalize to a list of plain values, never crash.
            return stripped
    return obj


def normalize_pin_expressions(raw: Any) -> Dict[str, str]:
    """``node.data.pinExpressions`` → {pin_id: expression}, dropping junk.

    Tolerant on shape (hosts and older documents may carry nulls or blanks)
    but strict on type: only non-empty strings survive. Silent drops here
    are safe — a dropped entry behaves exactly like the old-compiler skew
    path (pin falls back to default), which is the designed degrade.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for pin_id, expr in raw.items():
        if not isinstance(pin_id, str) or not pin_id:
            continue
        if not isinstance(expr, str):
            continue
        text = expr.strip()
        if text:
            out[pin_id] = text
    return out


def compile_pin_expression(
    expression: str,
    *,
    node_label: str,
    pin_id: str,
    library: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> PinExpressionEvaluator:
    """Compile one expression ONCE at flow-build time; return an evaluator.

    The evaluator signature is ``(value, run_vars) -> Any``. Compile errors
    raise immediately (flow build fails loudly naming node+pin — an
    unparseable expression must never wait for run time to be discovered).

    ``library`` is the flow's compiled function library (tier 2,
    ``function_library.py``): declared names become callable inside the
    expression (``build_again(vars.state, 3)``). Library names can never
    shadow the environment's own bindings — the library compiler refuses
    reserved names, and ``value``/``vars`` are bound per evaluation AFTER the
    library merge, so they win by construction.
    """
    where = f"{node_label}.{pin_id}"
    if not RESTRICTED_PYTHON_AVAILABLE:
        # The basic-exec fallback lane (no RestrictedPython installed) is a
        # dev posture; expressions refuse rather than run unguarded eval —
        # the poisoned bare-eval `function` lane is exactly what this module
        # exists to never repeat.
        raise PinExpressionError(
            f"pin expression on {where}: RestrictedPython is required for the expression tier"
        )

    try:
        byte_code = compile_restricted(
            expression, filename=f"<fx:{where}>", mode="eval", policy=_CodeNodePolicy
        )
    except SyntaxError as e:
        raise PinExpressionError(f"pin expression on {where} does not parse: {e}") from e
    except PinExpressionError:
        raise
    except Exception as e:
        # Compile can fail in ways other than SyntaxError — a deeply-nested
        # expression (e.g. ``a + 0 + 0 + ...`` thousands deep) overflows the
        # AST transformer with a RecursionError. The compile boundary is the
        # honesty boundary: EVERY compile failure must fail the build LOUDLY
        # and ATTRIBUTED (node+pin), never crash with a bare stack trace
        # naming RestrictedPython internals.
        raise PinExpressionError(
            f"pin expression on {where} could not be compiled ({type(e).__name__}: {e}) — "
            "it may be too large or deeply nested"
        ) from e
    errors = getattr(byte_code, "errors", None)
    if isinstance(errors, list) and errors:
        raise PinExpressionError(f"pin expression on {where} refused: {'; '.join(str(x) for x in errors)}")

    preview = expression if len(expression) <= 120 else expression[:117] + "..."

    base_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "_getiter_": default_guarded_getiter,
        "_getitem_": default_guarded_getitem,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_getattr_": safer_getattr,
        # Starred/kwargs calls (`max(*vars.nums)`) compile to `_apply_(...)`
        # in eval mode too; without this guard they die with a bare
        # "NameError: name '_apply_' is not defined" naming sandbox internals.
        "_apply_": _apply,
        # Same convenience builtins the code-node sandbox grants, one source.
        **sandbox_helper_globals(),
    }
    if library:
        # Flow function library (tier 2): declared names only, merged at
        # compile time. Reserved-name collisions were refused at library
        # compile; `value`/`vars` are set per evaluation after this dict is
        # copied, so they always win.
        base_globals.update(library)

    def evaluate(value: Any, run_vars: Optional[Mapping[str, Any]]) -> Any:
        env = dict(base_globals)
        env["value"] = value
        env["vars"] = _ReadOnlyVars(run_vars if isinstance(run_vars, Mapping) else {})
        try:
            result = eval(byte_code, env)  # noqa: S307 - RestrictedPython bytecode + guarded globals
        except PinExpressionError:
            raise
        except Exception as e:
            raise PinExpressionError(
                f"pin expression on {where} failed: {type(e).__name__}: {e} — expression: {preview}"
            ) from e
        # The view is an evaluation-time artifact and must never escape into
        # node outputs / persisted run state (not JSON serializable, aliases
        # every run var). Strip it wherever it landed — bare (``vars`` alone
        # means "the whole state as this pin's value") or nested inside a
        # container (``[vars]`` etc.). Nested DATA stays live (get_var parity).
        return _strip_views(result)

    setattr(evaluate, "_expression", expression)
    setattr(evaluate, "_where", where)
    return evaluate


def build_pin_expression_table(
    nodes: Any,
    *,
    library: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> Dict[str, Dict[str, PinExpressionEvaluator]]:
    """Compile every node's pinExpressions at flow build.

    Returns {node_id: {pin_id: evaluator}}. Raises loudly on the first bad
    expression (build-time failure names node+pin — the compile boundary is
    the honesty boundary, same law as connected-unknown-node refusal).
    ``library`` is the flow's compiled function library, shared by every
    evaluator (see :func:`compile_pin_expression`).
    """
    table: Dict[str, Dict[str, PinExpressionEvaluator]] = {}
    for node in nodes:
        data = getattr(node, "data", None)
        if not isinstance(data, dict):
            continue
        exprs = normalize_pin_expressions(data.get("pinExpressions"))
        if not exprs:
            continue
        node_id = getattr(node, "id", None)
        if not isinstance(node_id, str) or not node_id:
            continue
        label = str(data.get("label") or node_id)
        compiled: Dict[str, PinExpressionEvaluator] = {}
        for pin_id, expr in exprs.items():
            compiled[pin_id] = compile_pin_expression(
                expr, node_label=label, pin_id=pin_id, library=library
            )
        table[node_id] = compiled
    return table


def apply_pin_expressions(
    resolved_input: Dict[str, Any],
    evaluators: Optional[Dict[str, PinExpressionEvaluator]],
    run_vars: Optional[Mapping[str, Any]],
) -> None:
    """Apply a node's expressions onto its resolved inputs, in place.

    Called AFTER wires and defaults resolved: each expression sees the pin's
    would-have-been value as ``value`` and replaces it. In-place so both
    resolver sites (pure + exec) share one application step and can never
    drift on precedence.
    """
    if not evaluators:
        return
    for pin_id, evaluate in evaluators.items():
        resolved_input[pin_id] = evaluate(resolved_input.get(pin_id), run_vars)
