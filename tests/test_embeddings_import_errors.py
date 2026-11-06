"""Embedding import failures must keep the original actionable cause."""
import builtins

import pytest

from abstractruntime.integrations.abstractcore.embeddings_client import AbstractCoreEmbeddingsClient


@pytest.mark.parametrize("failure", [
    ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency"),
    ImportError("cannot import name 'EmbeddingManager'"),
    RuntimeError("embedding backend initialization failed"),
])
def test_embedding_import_preserves_original_failure(monkeypatch, failure):
    original = builtins.__import__
    attempted = []

    def importing(name, *args, **kwargs):
        if name.startswith("abstractcore."):
            attempted.append(name)
        if name == "abstractcore.embeddings.manager":
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(type(failure)) as caught:
        AbstractCoreEmbeddingsClient(provider="lmstudio", model="configured-model")
    assert caught.value is failure
    assert attempted == ["abstractcore.embeddings.manager"]
