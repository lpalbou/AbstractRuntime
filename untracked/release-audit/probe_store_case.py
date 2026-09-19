"""Is the at-rest predicate split a DATA problem? (semantics release#60 ask 1)"""
from abstractmemory.in_memory_store import InMemoryTripleStore
from abstractmemory.store import TripleQuery, TripleAssertion

st = InMemoryTripleStore()
for i, pred in enumerate(["dcterms:isPartOf", "dcterms:ispartof"]):
    st.add([TripleAssertion(subject="ex:a", predicate=pred, object=f"ex:b{i}",
                            scope="global", owner_id="o1")])
    print("wrote", repr(pred))

print()
for q in ["dcterms:isPartOf", "dcterms:ispartof"]:
    res = st.query(TripleQuery(predicate=q, scope="global", owner_id="o1", limit=50))
    got = sorted({getattr(r, "predicate", None) for r in res})
    print(f"query {q!r:22s} -> {len(res)} row(s), predicates seen: {got}")

res = st.query(TripleQuery(scope="global", owner_id="o1", limit=50))
print(f"\nunfiltered           -> {len(res)} row(s), predicates at rest: {sorted({getattr(r,'predicate',None) for r in res})}")
