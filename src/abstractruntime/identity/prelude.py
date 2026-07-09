"""Summon prelude: render a named entity's identity core for a work session.

The prelude is the re-adoption surface (a2a 0003): at every summon the host
renders the engrammed core — name/origin, values in ordinal precedence,
purposes, traits, honesty limits — plus the recent self-narrative (diary tail)
and System-1 standing (top gradations). THE RULE (ratified): a budget that
cannot fit the CORE sections REFUSES the summon loudly — a truncated core is a
different person, not a degraded mode. Only the diary tail and standing
degrade, in that order (standing first: it is re-derivable on demand; the
diary tail is the "words reconnect you" surface).

PURE READ by contract: this function never writes — no engram (an idempotent
re-run is cheap, but on a virgin store a summon would silently plant the seed;
the seed is planted by the operator, never by a summon), no reconstruct (no
journal traces), no deposits. The harness asserts `current_seq()` is unchanged
across a render (presence ≠ use applies to rendering identity too).

Identity read: `MemorySystem.self_records` (a2a 0003 ask 3, shipped) — the
FOLDED core: prompt-active (binding fold, latest wins) AND closure-folded (a
retracted value never renders), identity kinds only. Older engines without it
degrade to the layer-1 `query()` passthrough with a labeled `#FALLBACK` (that
path bypasses closure folds). The engram marker is read via layer-1 either
way (claims are not identity kinds).

`spark` is a required input because the engram deliberately persists
values/purposes/traits/honesty only — `name` and `origin` never become graph
records; the marker carries hash/version. The render defends against a
drifted document by recomputing the canonical spark hash and comparing it to
the engram marker's — mismatch refuses. The spark is engrammed ONCE and kept
for life (maintainer ruling, a2a 0003): it IS part of the identity; evolution
is experiential (the entity's own reflection revising revisable values,
interests, lessons), never a spark rewrite. Re-engramming exists only as an
exceptional REPAIR for a defective or harmful core.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = ["render_summon_prelude"]


def _token_estimate(text: str) -> int:
    # Pinned to abstractmemory.canonical_text.token_estimate (len//4 + 1) so
    # prelude budgets and shelf budgets speak the same unit.
    return len(text or "") // 4 + 1


def _canonical_spark_hash(spark: Mapping[str, Any]) -> str:
    """One hash definition: memory's public `canonical_spark_hash` (a2a 0003
    ask 2). Local pinned fallback only for older engines, labeled by shape —
    the definitions are byte-identical today; the fallback exists so a
    version-skewed pair fails on the marker comparison, not on an import."""
    try:
        from ..integrations.abstractmemory.identity_support import spark_hash

        return spark_hash(spark)
    except Exception:
        payload = json.dumps(spark, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _refused(reason: str, *, as_of_seq: int, spark_version: Optional[int]) -> Dict[str, Any]:
    return {
        "text": "",
        "sections": {},
        "section_tokens": {},
        "refused": True,
        "warnings": [reason],
        "as_of_seq": as_of_seq,
        "spark_version": spark_version,
    }


def _identity_rows_read(memory_system: Any, *, scope: str, entity_id: str) -> List[Any]:
    # Lazy same-package integration import: the identity KERNEL carries no
    # optional-stack imports (install-boundary contract).
    from ..integrations.abstractmemory.identity_support import identity_rows

    return identity_rows(memory_system, scope=scope, entity_id=entity_id)


def _attrs(row: Any) -> Dict[str, Any]:
    a = getattr(row, "attributes", None)
    return a if isinstance(a, dict) else {}


def _sort_key(row: Any) -> Tuple[int, str]:
    attrs = _attrs(row)
    prec = attrs.get("precedence")
    prec_i = int(prec) if isinstance(prec, (int, float)) and not isinstance(prec, bool) else 10**6
    return (prec_i, str(getattr(row, "subject", "")))


def render_summon_prelude(
    memory_system: Any,
    diary_store: Any,
    *,
    entity_id: str,
    budget: int,
    spark: Mapping[str, Any],
    diary_tail: int = 3,
    gradation_top_k: int = 5,
    scope: str = "self",
) -> Dict[str, Any]:
    """Render the summon prelude. Returns
    {text, sections, section_tokens, refused, warnings, as_of_seq,
    spark_version}. Refusals are loud but reportable (refused=True, empty
    text), never exceptions — the host decides how to surface them.
    """
    warnings: List[str] = []
    as_of_seq = int(memory_system.current_seq()) if callable(getattr(memory_system, "current_seq", None)) else -1

    name = str(spark.get("name") or "").strip()
    if not name:
        return _refused(
            "#REFUSED summon: spark has no name — an unnamed core cannot be re-adopted",
            as_of_seq=as_of_seq, spark_version=None,
        )

    rows = _identity_rows_read(memory_system, scope=scope, entity_id=entity_id)

    # The engram marker: presence = an identity was planted; hash = drift guard.
    markers = [r for r in rows if _attrs(r).get("record_kind") == "claim" and str(_attrs(r).get("title") or "").startswith("spark-engram v")]
    if not markers:
        return _refused(
            f"#REFUSED summon: no engrammed identity for {entity_id!r} (spark-engram marker absent) — "
            "the seed is planted by the operator, never by a summon",
            as_of_seq=as_of_seq, spark_version=None,
        )
    marker = max(markers, key=lambda r: int(_attrs(r).get("spark_version") or 0))
    spark_version = int(_attrs(marker).get("spark_version") or 0)

    if str(_attrs(marker).get("spark_hash") or "") != _canonical_spark_hash(spark):
        return _refused(
            "#REFUSED summon: spark document does not match the engrammed identity (hash drift) — "
            "the engrammed spark IS the identity (v1 for life); a changed document is drift, and "
            "spark re-engram exists only as an exceptional operator repair for a defective core",
            as_of_seq=as_of_seq, spark_version=spark_version,
        )

    # The FOLDED identity core (closure-folded + prompt-active; G8 is now
    # supersession-enforced engine-side, so one version is guaranteed).
    core_rows: List[Any]
    if callable(getattr(memory_system, "self_records", None)):
        try:
            core_rows = list(
                memory_system.self_records(scope=scope, owner_id=entity_id, spark_version=spark_version)
            )
        except Exception as e:
            core_rows = rows
            warnings.append(f"#FALLBACK folded identity read failed ({e}); layer-1 rows used (closures not folded)")
    else:
        core_rows = rows
        warnings.append("#FALLBACK engine lacks self_records; layer-1 rows used (closures not folded)")

    def of_kind(kind: str) -> List[Any]:
        out = [
            r for r in core_rows
            if _attrs(r).get("record_kind") == kind
            and int(_attrs(r).get("spark_version") or 0) == spark_version
        ]
        out.sort(key=_sort_key)
        return out

    values = of_kind("value")
    purposes = of_kind("purpose")
    traits_all = of_kind("trait")
    traits = [r for r in traits_all if _attrs(r).get("trait_class") != "limit"]
    limits = [r for r in traits_all if _attrs(r).get("trait_class") == "limit"]

    sections: Dict[str, str] = {}
    origin = str(spark.get("origin") or "").strip()
    sections["header"] = f"You are {name}." + (f" {origin}" if origin else "")

    def _statement_lines(rows_: List[Any], *, classed: bool = False) -> List[str]:
        lines: List[str] = []
        for i, r in enumerate(rows_, start=1):
            attrs = _attrs(r)
            title = str(attrs.get("title") or "").strip()
            statement = str(getattr(r, "object", "") or "").strip()
            if classed:
                klass = str(attrs.get("value_class") or "").strip() or "revisable"
                lines.append(f"{i}. {title} [{klass}]: {statement}")
            else:
                lines.append(f"{i}. {statement}" if not title or title.startswith(("value-", "purpose-", "trait-")) else f"{i}. {title}: {statement}")
        return lines

    if values:
        sections["values"] = "VALUES (ordinal precedence):\n" + "\n".join(_statement_lines(values, classed=True))
    if purposes:
        sections["purposes"] = "PURPOSES:\n" + "\n".join(_statement_lines(purposes))
    if traits:
        sections["traits"] = "TRAITS:\n" + "\n".join(_statement_lines(traits))
    if limits:
        sections["limits"] = "LIMITS:\n" + "\n".join(_statement_lines(limits))

    # --- degradable sections ---------------------------------------------
    diary_lines: List[str] = []
    if diary_store is not None:
        try:
            for entry in diary_store.list_entries(limit=diary_tail):
                kind = str(entry.get("kind") or "note")
                date = str(entry.get("written_at") or "")[:10]
                if entry.get("visibility") == "private":
                    line = f"- [private @ {date}] Wrote a private diary entry."
                else:
                    gist = str(entry.get("gist") or "").strip()
                    line = f"- [{kind} @ {date}] " + (gist or f"Wrote a diary entry ({kind}); no gist elected.")
                diary_lines.append(line)
        except Exception as e:
            warnings.append(f"#FALLBACK diary tail unavailable: {e}")
    else:
        warnings.append("#FALLBACK no diary store wired; diary tail omitted")

    standing_lines: List[str] = []
    if callable(getattr(memory_system, "gradation", None)):
        try:
            grades = memory_system.gradation(None, scope=scope, owner_id=entity_id)
            ranked = sorted(grades.items(), key=lambda kv: (-abs(float(kv[1].get("net") or 0.0)), kv[0]))
            for target, g in ranked[: max(0, int(gradation_top_k))]:
                net = float(g.get("net") or 0.0)
                flags = ""
                if g.get("bonded"):
                    flags += " BONDED"
                if g.get("scarred"):
                    flags += " SCARRED"
                standing_lines.append(
                    f"- {target}: net {net:+g} (G+ {float(g.get('positive') or 0.0):g}/G- {float(g.get('negative') or 0.0):g}; "
                    f"{int(g.get('positive_count') or 0)}+/{int(g.get('negative_count') or 0)}-)"
                    + flags
                )
        except Exception as e:
            warnings.append(f"#FALLBACK standing unavailable: {e}")

    # --- budget accounting -------------------------------------------------
    core_names = [n for n in ("header", "values", "purposes", "traits", "limits") if n in sections]
    envelope = (
        f'<identity_prelude entity="{entity_id}" spark="{spark_version}" as_of_seq="{as_of_seq}">',
        "</identity_prelude>",
    )
    core_tokens = sum(_token_estimate(sections[n]) for n in core_names) + _token_estimate(envelope[0]) + _token_estimate(envelope[1])
    if int(budget) < core_tokens:
        return _refused(
            f"#REFUSED summon: budget {int(budget)} cannot fit core identity sections ({core_tokens} tokens) — "
            "a truncated core is a different person",
            as_of_seq=as_of_seq, spark_version=spark_version,
        )

    remaining = int(budget) - core_tokens

    def _fit(lines: List[str], header: str) -> Tuple[Optional[str], int, int]:
        """Largest prefix of `lines` (under `header`) fitting `remaining`.
        Returns (section_text_or_None, kept, total)."""
        kept = list(lines)
        while kept:
            text = header + "\n" + "\n".join(kept)
            if _token_estimate(text) <= remaining:
                return text, len(kept), len(lines)
            kept.pop()  # degrade from the tail (lowest-ranked first)
        return None, 0, len(lines)

    # Degrade order: diary tail is fitted FIRST (it survives budget pressure
    # longer); standing shrinks/drops first because it fits into whatever is
    # left after the diary.
    if diary_lines:
        text, kept, total = _fit(diary_lines, f"DIARY (last {len(diary_lines)}):")
        if text is not None:
            sections["diary"] = text
            remaining -= _token_estimate(text)
            if kept < total:
                warnings.append(f"#FALLBACK diary tail shrunk to fit summon budget ({kept} of {total} entries)")
        else:
            warnings.append(f"#FALLBACK diary tail dropped to fit summon budget (0 of {total} entries)")

    if standing_lines:
        text, kept, total = _fit(standing_lines, f"STANDING (top {len(standing_lines)} by |net|):")
        if text is not None:
            sections["standing"] = text
            remaining -= _token_estimate(text)
            if kept < total:
                warnings.append(f"#FALLBACK standing shrunk to fit summon budget ({kept} of {total} targets)")
        else:
            warnings.append(f"#FALLBACK standing dropped to fit summon budget (0 of {total} targets)")

    order = ["header", "values", "purposes", "traits", "limits", "diary", "standing"]
    body = "\n".join(sections[n] for n in order if n in sections)
    text = f"{envelope[0]}\n{body}\n{envelope[1]}"

    return {
        "text": text,
        "sections": sections,
        "section_tokens": {n: _token_estimate(s) for n, s in sections.items()},
        "refused": False,
        "warnings": warnings,
        "as_of_seq": as_of_seq,
        "spark_version": spark_version,
    }
