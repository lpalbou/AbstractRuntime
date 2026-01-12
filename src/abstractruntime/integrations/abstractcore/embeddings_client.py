from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence


@dataclass(frozen=True)
class EmbeddingsResult:
    provider: str
    model: str
    embeddings: List[List[float]]
    dimension: int


class AbstractCoreEmbeddingsClient:
    """Thin adapter around AbstractCore's EmbeddingManager.

    Notes:
    - AbstractCore is an optional dependency of AbstractRuntime; importing this module is an explicit opt-in.
    - The embedding provider/model should be treated as a runtime-instance property (singleton) so all vectors
      live in the same embedding space.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        manager_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Monorepo import hygiene:
        # When running from the repo root, `abstractcore/` (project folder) can be imported as a
        # namespace package and shadow the real python package at `abstractcore/abstractcore/`.
        # In that situation, `abstractcore.embeddings.manager` may not resolve even though the
        # implementation exists. Prefer the public import, but fall back to the concrete path.
        try:
            from abstractcore.embeddings.manager import EmbeddingManager  # type: ignore
        except Exception:  # pragma: no cover
            from abstractcore.abstractcore.embeddings.manager import EmbeddingManager  # type: ignore

        prov = str(provider or "").strip().lower()
        mod = str(model or "").strip()
        if not prov:
            raise ValueError("provider is required for embeddings")
        if not mod:
            raise ValueError("model is required for embeddings")

        self._provider = prov
        self._model = mod
        self._mgr = EmbeddingManager(provider=prov, model=mod, **(manager_kwargs or {}))

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingsResult:
        items = [str(t or "") for t in texts]
        embeddings = self._mgr.embed_batch(items)
        dim = int(self._mgr.get_dimension() or 0)
        return EmbeddingsResult(
            provider=self._provider,
            model=self._model,
            embeddings=embeddings,
            dimension=dim,
        )
