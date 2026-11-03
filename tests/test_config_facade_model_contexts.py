"""The model-context facade (AbstractCore ADR 0010 point 8): the gateway reads and sets a local
model's loaded context through AbstractRuntime, never by importing AbstractCore (gateway ADR 0001).

- with Core's `config.model_contexts`, the calls pass through unchanged;
- without it (every released Core up to 2.31.1), they raise `AbstractCoreTooOld` naming what is
  missing, in words, never a silent default.
"""

from __future__ import annotations

import sys
import types

import pytest

from abstractruntime.integrations.abstractcore import config_facade as facade

pytestmark = pytest.mark.basic


def test_the_calls_pass_through_to_core(monkeypatch) -> None:
    seen = {}
    mod = types.ModuleType("abstractcore.config.model_contexts")
    mod.configured_model_context = lambda p, m, *, config_file=None: seen.setdefault("read", (p, m, config_file)) and 32768
    mod.set_model_context = lambda p, m, t, *, config_file=None: seen.setdefault("set", (p, m, t, config_file)) and {"huggingface::org/m": t}
    monkeypatch.setitem(sys.modules, "abstractcore.config.model_contexts", mod)
    assert facade.configured_model_context("huggingface", "org/m", config_file="/c.json") == 32768
    assert facade.set_model_context("huggingface", "org/m", 32768, config_file="/c.json") == {"huggingface::org/m": 32768}
    assert seen == {"read": ("huggingface", "org/m", "/c.json"), "set": ("huggingface", "org/m", 32768, "/c.json")}


@pytest.mark.parametrize("call", [
    lambda: facade.configured_model_context("huggingface", "org/m"),
    lambda: facade.set_model_context("huggingface", "org/m", 32768),
])
def test_an_older_core_says_what_is_missing_in_words(monkeypatch, call) -> None:
    """MUTANT: return None (a silent default) instead of raising -> RED."""
    monkeypatch.setitem(sys.modules, "abstractcore.config.model_contexts", None)
    monkeypatch.setattr(facade, "_installed_abstractcore_version", lambda: "2.31.1")
    with pytest.raises(facade.AbstractCoreTooOld) as info:
        call()
    msg = str(info.value)
    assert "AbstractCore 2.31.1 is installed" in msg and "newer than 2.31.1" in msg and "model_contexts" in msg
    assert info.value.missing == "abstractcore.config.model_contexts"
