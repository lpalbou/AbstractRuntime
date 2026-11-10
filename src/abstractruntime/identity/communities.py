"""RETIRED (2026-11-10): the runtime's July Louvain over a home's graph, and its keyword labels.

Topics belong to AbstractMemory now (operator 2026-11-10: "AbstractMemory should be the one owning those graph
operations, but it could be called indeed by runtime"; "we only want clean, simple, robust and efficient
abstractions"). Memory forms the entity's `community` records while he sleeps
(`abstractmemory.communities.communities_pass`, run by the runtime's night stage in `identity/topics.py`), each with
an authored title and summary. Memory serves them as payload version 2 (`communities()`), and the entity reads them
with the `topics` tool. What lived here is gone:

- the request-time Louvain (computed on a gateway cache miss, so it had no night, no model and no authored title);
- the keyword "bridges" and the γ sweep;
- the LABELS (the most distinctive member keywords minus a stop list): a regex keyword heuristic naming topics with
  no model, retired under the no-heuristics rule.

The keyword extractor that fed them (`identity/digest.py`) stays this wave: world-model cards and the candidate
miner read its keywords (COMMUNITIES-DESIGN ruling 6).

`memory_communities` remains ONLY so its last caller fails loudly instead of serving the old shape. That caller is
the gateway's `GET /entities/{name}/communities` (routes/entities.py), which must serve memory's payload v2 through
the runtime's memory facade (`integrations.abstractmemory.memory_facade.communities`). The error is an ImportError, the class the route already answers with "501 communities unavailable:
<this sentence>". This module is deleted once no caller imports it.
"""

from __future__ import annotations

from typing import Any, NoReturn

__all__ = ["CommunitiesRetired", "RETIRED", "memory_communities"]

RETIRED = (
    "abstractruntime.identity.communities.memory_communities is retired (2026-11-10): the runtime no longer computes "
    "communities or keyword labels. Serve AbstractMemory's communities() payload (version 2: authored titles and "
    "summaries, formed while the entity sleeps) instead: abstractruntime.integrations.abstractmemory.memory_facade."
    "communities(system, scopes=, owner_id=)."
)


class CommunitiesRetired(ImportError):
    """The runtime's own communities compute is gone (see the module doc). An ImportError: what a caller needs is not
    available from this runtime, and a host that maps ImportError to "unavailable" says so loudly."""


def memory_communities(*_args: Any, **_kwargs: Any) -> NoReturn:
    """Retired: raises CommunitiesRetired with the sentence naming what to serve instead."""
    raise CommunitiesRetired(RETIRED)
