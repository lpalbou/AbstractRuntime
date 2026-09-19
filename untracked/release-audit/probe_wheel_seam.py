"""Third column (delegate release#220): do my refs resolve in the PUBLISHED WHEEL,
not the neighbour's src/ tree?"""
import re, pathlib, importlib, collections, sys
SRC = pathlib.Path(sys.argv[1])
pat = re.compile(r"^\s*from\s+(abstractmemory|abstractsemantics)([\w.]*)\s+import\s+([^\n#]+)", re.M)
seams = collections.defaultdict(set)
for f in SRC.rglob("*.py"):
    for m in pat.finditer(f.read_text(encoding="utf-8", errors="ignore")):
        pkg, sub, names = m.group(1), m.group(2), m.group(3)
        for n in names.replace("(","").replace(")","").split(","):
            n = n.strip().split(" as ")[0].strip()
            if n and n != "*": seams[pkg].add((pkg+sub, n))
for pkg in sorted(seams):
    refs = sorted(seams[pkg]); ok=0; missing=[]
    for mod, sym in refs:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, sym): ok += 1
            else: missing.append(f"{mod}.{sym}")
        except Exception as e: missing.append(f"{mod}.{sym} [{type(e).__name__}]")
    print(f"{pkg}: written={len(refs)} resolved_in_WHEEL={ok} gap={len(refs)-ok}")
    for x in missing[:10]: print("   MISSING:", x)
