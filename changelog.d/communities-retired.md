### Removed

- **The runtime's own communities compute is retired** (`identity/communities.py`; operator 2026-11-10: memory owns
  the graph operations). The request-time Louvain, the keyword bridges, the γ sweep and the keyword LABELS (a regex
  heuristic naming topics with no model) are gone. Topics are AbstractMemory's community records, formed during the
  sleep with authored titles (the night's topics stage) and served as payload version 2. The keyword extractor in
  `identity/digest.py` stays this wave: world-model cards and the candidate miner read its keywords.
- `memory_communities` remains only to fail loudly. It raises `CommunitiesRetired`, an `ImportError` the gateway route
  already answers with "501 communities unavailable", with a sentence naming what to serve instead. It never returns
  the old shape. The module is deleted once its last caller, the gateway's `GET /entities/{name}/communities`, has
  switched. A test pins that no community algorithm or keyword label remains in the runtime's source.
