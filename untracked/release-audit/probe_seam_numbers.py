"""Two numbers per CONSUMED seam (delegate release#157 ask 2).
(a) references I wrote into their surface
(b) how many I have now OBSERVED resolvable in their LIVE artifact
Resolution is by importing from the neighbour's live source tree.
"""
import re, pathlib, importlib, collections

SRC = pathlib.Path("src/abstractruntime")
pat = re.compile(r"^\s*from\s+(abstractcore|abstractmemory|abstractsemantics)([\w.]*)\s+import\s+([^\n#]+)", re.M)

seams = collections.defaultdict(set)   # pkg -> {(module, symbol)}
for f in SRC.rglob("*.py"):
    for m in pat.finditer(f.read_text(encoding="utf-8", errors="ignore")):
        pkg, sub, names = m.group(1), m.group(2), m.group(3)
        mod = pkg + sub
        for n in names.replace("(", "").replace(")", "").split(","):
            n = n.strip().split(" as ")[0].strip()
            if n and n != "*" and not n.startswith("#"):
                seams[pkg].add((mod, n))

for pkg in ("abstractcore", "abstractmemory", "abstractsemantics"):
    refs = sorted(seams[pkg])
    ok, missing = 0, []
    for mod, sym in refs:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, sym):
                ok += 1
            else:
                missing.append(f"{mod}.{sym}")
        except Exception as e:
            missing.append(f"{mod}.{sym}  [{type(e).__name__}]")
    print(f"\n=== {pkg} ===")
    print(f"  (a) distinct symbol references I wrote : {len(refs)}")
    print(f"  (b) OBSERVED resolvable in live tree   : {ok}")
    print(f"  GAP                                    : {len(refs)-ok}")
    for x in missing[:15]:
        print("     MISSING:", x)
