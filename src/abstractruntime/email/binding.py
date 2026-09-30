"""Run-scoped email account binding (framework backlog 0992, B1).

One user, one runtime, one mailbox. The host (the gateway) owns the credentials; the runtime
only carries a NON-SECRET binding in the run's vars and asks the host for the account at the
moment a tool needs it:

- `_runtime.email_account = {"account_ref": str, "address": str}`: which account this run acts
  for. Set by the host at its door (`bind_email_account`) after popping client-supplied values
  (`strip_client_email_keys`), and by the automation controller at admission from
  `Runtime.email_binding`. Child runs inherit it (the parent's value wins).
- `Runtime.set_email_context_resolver(fn)`: `fn(binding, *, use) -> EmailContext | None`, held
  in memory on that Runtime only. The tool-call handler and the approval-resume path run each tool
  batch inside `email_run_scope(...)`, so AbstractCore's email tools resolve the account of the
  EXECUTING run, never a process-wide one.
- `use` says who is sending, so the host can apply the right switch: `"agent_tool"` (an agent's
  or a workflow's email tool call: the user's "Agent email tools" choice applies) or `"action"`
  (the runtime's own send-email action, `abstractruntime.email.actions`, which the user authored
  as fixed templates: the account being connected and enabled is enough). The binding itself
  follows the ACCOUNT (connected + enabled), never the agent-tools choice. A resolver written as
  `fn(binding)` (no `use` parameter) keeps working and is called without it.

Credentials never enter run vars, the ledger, tool arguments or results: the `EmailContext`
exists only in memory during the tool call.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

EMAIL_ACCOUNT_KEY = "email_account"
EMAIL_ALLOWED_RECIPIENTS_KEY = "email_allowed_recipients"
CLIENT_EMAIL_KEYS = (EMAIL_ACCOUNT_KEY, EMAIL_ALLOWED_RECIPIENTS_KEY)

# Who is using the account (passed to the host's resolver as `use=`).
EMAIL_USE_AGENT_TOOL = "agent_tool"
EMAIL_USE_ACTION = "action"
EMAIL_USES = (EMAIL_USE_AGENT_TOOL, EMAIL_USE_ACTION)

NOT_BOUND_CAUSE = "No email account is connected for this run."
NOT_BOUND_FIX = "Connect an email account in Settings -> Email, then retry."


@dataclass(frozen=True)
class EmailBinding:
    """Which account a run acts for. Non-secret: safe in run vars, ledgers and logs."""

    account_ref: str
    address: str = ""

    def __post_init__(self) -> None:
        ref = str(self.account_ref or "").strip()
        if not ref:
            raise ValueError("EmailBinding.account_ref must be a non-empty string")
        if any(c in ref for c in "\r\n"):
            raise ValueError("EmailBinding.account_ref must be one line")
        object.__setattr__(self, "account_ref", ref)
        object.__setattr__(self, "address", str(self.address or "").strip())

    def to_dict(self) -> Dict[str, str]:
        return {"account_ref": self.account_ref, "address": self.address}

    @classmethod
    def from_value(cls, value: Any) -> Optional["EmailBinding"]:
        if isinstance(value, EmailBinding):
            return value
        if not isinstance(value, Mapping):
            return None
        try:
            return cls(account_ref=str(value.get("account_ref") or ""), address=str(value.get("address") or ""))
        except ValueError:
            return None


def binding_of(vars_obj: Any) -> Optional[EmailBinding]:
    """The run's binding from its vars (`_runtime.email_account`), or None."""
    if not isinstance(vars_obj, Mapping):
        return None
    rt = vars_obj.get("_runtime")
    if not isinstance(rt, Mapping):
        return None
    return EmailBinding.from_value(rt.get(EMAIL_ACCOUNT_KEY))


def strip_client_email_keys(vars_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Pop `_runtime.email_account` / `_runtime.email_allowed_recipients` from client-supplied vars.

    Hosts call this at their door: a client must never choose the account a run uses or widen
    the recipients it may mail without asking. Returns `vars_obj` (mutated in place).
    """
    if isinstance(vars_obj, dict):
        rt = vars_obj.get("_runtime")
        if isinstance(rt, dict):
            for key in CLIENT_EMAIL_KEYS:
                rt.pop(key, None)
    return vars_obj


def bind_email_account(
    vars_obj: Dict[str, Any],
    *,
    binding: Optional[EmailBinding] = None,
    account_ref: Optional[str] = None,
    address: str = "",
) -> Dict[str, Any]:
    """SET (never setdefault) `_runtime.email_account` on `vars_obj`; None removes it.

    Pass either `binding` or `account_ref` (+ `address`). Returns `vars_obj`.
    """
    if not isinstance(vars_obj, dict):
        raise TypeError("vars must be a dict")
    if binding is None and account_ref:
        binding = EmailBinding(account_ref=account_ref, address=address)
    rt = vars_obj.get("_runtime")
    if not isinstance(rt, dict):
        rt = {}
        vars_obj["_runtime"] = rt
    if binding is None:
        rt.pop(EMAIL_ACCOUNT_KEY, None)
    else:
        rt[EMAIL_ACCOUNT_KEY] = binding.to_dict()
    return vars_obj


# --- execution scope --------------------------------------------------------------------

Resolver = Callable[[EmailBinding], Any]

# (resolver of the executing run's Runtime, the run's binding, the use). Unset outside a tool batch.
_SCOPE: "ContextVar[Optional[Tuple[Optional[Resolver], Optional[EmailBinding], str]]]" = ContextVar(
    "abstractruntime_email_scope", default=None
)


def resolver_accepts_use(resolver: Any) -> bool:
    """True when `resolver` takes a `use` keyword (or `**kwargs`); decided on its signature."""
    import inspect

    try:
        params = inspect.signature(resolver).parameters.values()
    except (TypeError, ValueError):
        return False
    for p in params:
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == "use" and p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return True
    return False


def _call_resolver(resolver: Resolver, binding: EmailBinding, use: str) -> Any:
    if resolver_accepts_use(resolver):
        return resolver(binding, use=use)
    return resolver(binding)
_INSTALL_LOCK = threading.Lock()
_CORE_RESOLVER_INSTALLED = False


def _not_bound_error() -> Exception:
    from abstractcore.comms.email import EmailNotConfigured

    return EmailNotConfigured(NOT_BOUND_CAUSE, NOT_BOUND_FIX)


def resolve_scoped_context() -> Any:
    """AbstractCore's account resolver while a runtime is installed (see `install_core_resolver`).

    Inside a tool batch: the executing run's account through its Runtime's resolver; outside
    one (or for a run without a binding / a runtime without a resolver): None, which the tools
    report as `email_not_configured`. Never the local AbstractCore settings.
    """
    scope = _SCOPE.get()
    if scope is None:
        return None
    resolver, binding, use = scope
    if resolver is None or binding is None:
        return None
    from abstractcore.comms.email import EmailError, EmailSecretUnavailable

    try:
        ctx = _call_resolver(resolver, binding, use)
    except EmailError:
        raise
    except Exception:  # noqa: BLE001 - the host's failure text may carry details; never echo it
        raise EmailSecretUnavailable(
            "The email account of this run could not be loaded.",
            "Open Settings -> Email and Test the account; connect it again if the test fails.",
        ) from None
    return ctx


def install_core_resolver() -> None:
    """Point AbstractCore's email tools at the executing run (idempotent, process-wide).

    Called when a Runtime gets a resolver. From then on the tools in this process never fall
    back to the local AbstractCore settings (a multi-user host must not send from the
    install's own account).
    """
    global _CORE_RESOLVER_INSTALLED
    with _INSTALL_LOCK:
        if _CORE_RESOLVER_INSTALLED:
            return
        from abstractcore.tools.comms_tools import set_email_account_resolver

        set_email_account_resolver(resolve_scoped_context)
        _CORE_RESOLVER_INSTALLED = True


def uninstall_core_resolver() -> None:
    """Remove the process-wide hook (tests; hosts shutting down email entirely)."""
    global _CORE_RESOLVER_INSTALLED
    with _INSTALL_LOCK:
        if not _CORE_RESOLVER_INSTALLED:
            return
        from abstractcore.tools.comms_tools import set_email_account_resolver

        set_email_account_resolver(None)
        _CORE_RESOLVER_INSTALLED = False


def core_resolver_installed() -> bool:
    return _CORE_RESOLVER_INSTALLED


@contextmanager
def email_run_scope(run: Any, *, resolver: Optional[Resolver] = None, use: Optional[str] = None) -> Iterator[None]:
    """Run a tool batch as `run`: its binding, its Runtime's resolver and who is sending.

    `resolver` defaults to `run._runtime_email_resolver` and `use` to `run._runtime_email_use`
    (both set by `Runtime.tick`; `use` is "action" only while the runtime's own send-email
    action workflow runs, decided on the identity of its node function, else "agent_tool").
    The scope is always entered, even for an unbound run, so a nested batch never inherits
    another run's account.
    """
    if resolver is None:
        resolver = getattr(run, "_runtime_email_resolver", None)
    if use is None:
        use = getattr(run, "_runtime_email_use", None) or EMAIL_USE_AGENT_TOOL
    if use not in EMAIL_USES:
        raise ValueError(f"use must be one of {EMAIL_USES}, got {use!r}")
    vars_obj = getattr(run, "vars", None)
    token = _SCOPE.set((resolver if callable(resolver) else None, binding_of(vars_obj), use))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def resolve_email_context(run: Any, *, resolver: Optional[Resolver] = None, use: Optional[str] = None) -> Any:
    """The `EmailContext` of `run` for runtime-owned code that needs it outside a tool batch.

    Raises `EmailNotConfigured` when the run is unbound or its Runtime has no resolver / the
    resolver returns None.
    """
    with email_run_scope(run, resolver=resolver, use=use):
        ctx = resolve_scoped_context()
    if ctx is None:
        raise _not_bound_error()
    return ctx


__all__ = [
    "CLIENT_EMAIL_KEYS",
    "EMAIL_ACCOUNT_KEY",
    "EMAIL_ALLOWED_RECIPIENTS_KEY",
    "EMAIL_USES",
    "EMAIL_USE_ACTION",
    "EMAIL_USE_AGENT_TOOL",
    "EmailBinding",
    "bind_email_account",
    "binding_of",
    "core_resolver_installed",
    "email_run_scope",
    "install_core_resolver",
    "resolve_email_context",
    "resolve_scoped_context",
    "resolver_accepts_use",
    "strip_client_email_keys",
    "uninstall_core_resolver",
]
