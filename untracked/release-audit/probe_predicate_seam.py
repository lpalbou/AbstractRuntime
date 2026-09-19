"""Does the runtime KG boundary make a DATA problem, or only a contract one?

semantics (release#60 ask 1) asks whether the read/query path case-folds.
This writes through the real runtime normalizer and queries the real store.
"""
from abstractruntime.integrations.abstractmemory.effect_handlers import (
    _normalize_predicate_id, _allowed_predicate_ids,
)

print("=== 1. What the WRITE boundary persists ===")
for raw in ["dcterms:isPartOf", "schema:isPartOf", "schema:hasPart", "schema:awareness", "dcterms:hasPart"]:
    out = _normalize_predicate_id(raw)
    print(f"  {raw:24s} -> persisted as {out!r}")

print("\n=== 2. What the QUERY path (line 515, same function) resolves to ===")
for q in ["dcterms:isPartOf", "schema:isPartOf", "dcterms:ispartof", "dcterms:is_part_of"]:
    print(f"  query {q:24s} -> looks for {_normalize_predicate_id(q)!r}")

print("\n=== 3. Is the store match case-sensitive? ===")
try:
    from abstractmemory.triples import TripleQuery
    import abstractmemory, inspect
    from abstractmemory import __version__ as amv
    print("  abstractmemory version:", amv)
except Exception as e:
    print("  import note:", e)

try:
    from abstractmemory.triples.memory_store import InMemoryTripleStore as S
except Exception:
    S = None
    for mod in ("abstractmemory.triples", "abstractmemory.triples.store"):
        try:
            m = __import__(mod, fromlist=["*"])
            for n in dir(m):
                if "InMemory" in n and "Triple" in n:
                    S = getattr(m, n); print("  using store:", mod + "." + n)
        except Exception:
            pass
print("  store class:", S)
