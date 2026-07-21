"""Deterministic professional chart renderer (write_chart effect node).

Renders a STRUCTURED chart spec (pure data — callers, including LLMs, never
author code) into a publication-quality PNG (+ PDF sibling) via matplotlib.
This is the in-process replacement for the diagram-render workflow's
write-script-then-execute_command lane, which stalled every unattended run
on a tool-approval prompt (operator ruling 2026-07-20: a deterministic
process must never ask for approval while following its process).

Trust class: write_pdf/write_docx — a fixed renderer over caller data, no
shell, no arbitrary code. Because the render now runs IN-PROCESS (the
subprocess's memory/time isolation is gone), hard resource caps below are
the compensating control, not polish (adversarial review 2026-07-20):
spec size, element counts, figure dimensions, and label lengths are all
clamped, and mathtext/usetex are disabled so `$...$` in labels cannot raise
or invoke TeX.

Failure contract: NEVER raises for render-class failures — returns an
``ok:false`` envelope with a labeled error so workflows keep their honest
ASCII fallbacks (#FALLBACK discipline). Structural misuse of the API
(non-dict spec) is also reported through the envelope for uniformity.

Spec contract (v1, mirrors the diagram-render workflow):
  {"kind":"layered","title":str,"caption":str,
   "layers":[{"label":str,"nodes":[{"id":str,"label":str}]}],
   "edges":[{"from":str,"to":str,"label":str?,"style":"solid"|"dashed"}]}
  {"kind":"line","title":str,"caption":str,"x_label":str,"y_label":str,
   "y_min":number?,
   "series":[{"label":str,"points":[[x,y],...]}]}
"""
from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- hard security/resource caps (in-process render; see module docstring) ---
MAX_SPEC_BYTES = 512 * 1024
MAX_LAYERS = 24
MAX_NODES_PER_LAYER = 40
MAX_TOTAL_NODES = 400
MAX_EDGES = 800
MAX_SERIES = 12
MAX_POINTS_PER_SERIES = 2000
MAX_LABEL_CHARS = 400
# matplotlib refuses images >= 2^16 px per axis; clamp figure inches well
# below that at the fixed dpi so oversized specs fail as ok:false, never as
# an uncaught allocation.
MAX_FIG_INCHES = 40.0
DPI = 200  # fixed — never caller-controllable

LAYER_FILL = ["#dbe9f6", "#e8f0e3", "#fdf0dd", "#efe3f2", "#e7e7e7", "#fce4e4"]
LAYER_EDGE = ["#3d6fa8", "#5a8a4a", "#c98a2d", "#8a5aa8", "#6e6e6e", "#b85454"]
TEXT = "#1a1a2e"
ARROW = "#55606e"
FAMILY = ["Helvetica", "Arial", "DejaVu Sans"]

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class ChartRenderResult:
    ok: bool
    rendered: bool
    png_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class _SpecError(ValueError):
    """Invalid spec — reported as ok:false, never raised to callers."""


def _txt(value: Any, limit: int = MAX_LABEL_CHARS) -> str:
    """Coerce any label to a bounded, control-char-free, mathtext-inert string.

    matplotlib parses ``$...$`` as mathtext by default; an odd number of
    dollar signs raises inside text layout. Escaping every ``$`` keeps
    labels literal regardless of rcParams.
    """
    text = _CTRL_RE.sub("", str(value if value is not None else ""))
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("$", r"\$")


def _wrap(label: Any, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(_txt(label), width=width, break_long_words=False) or [""])


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _validate_spec_size(spec: Dict[str, Any]) -> None:
    try:
        size = len(json.dumps(spec, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception as e:  # unserializable spec = author bug, honest refusal
        raise _SpecError(f"spec is not JSON-serializable: {e}") from e
    if size > MAX_SPEC_BYTES:
        raise _SpecError(f"spec too large ({size} bytes > {MAX_SPEC_BYTES})")


def _render_layered(plt: Any, spec: Dict[str, Any]) -> Any:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    layers = spec.get("layers") or []
    edges = spec.get("edges") or []
    if not isinstance(layers, list) or not layers:
        raise _SpecError("layered spec has no layers")
    if len(layers) > MAX_LAYERS:
        raise _SpecError(f"too many layers ({len(layers)} > {MAX_LAYERS})")
    if not isinstance(edges, list):
        raise _SpecError("layered spec 'edges' must be a list")
    if len(edges) > MAX_EDGES:
        raise _SpecError(f"too many edges ({len(edges)} > {MAX_EDGES})")

    total_nodes = 0
    for layer in layers:
        nodes = (layer.get("nodes") if isinstance(layer, dict) else None) or []
        if len(nodes) > MAX_NODES_PER_LAYER:
            raise _SpecError(f"too many nodes in one layer ({len(nodes)} > {MAX_NODES_PER_LAYER})")
        total_nodes += len(nodes)
    if total_nodes == 0:
        raise _SpecError("layered spec has no nodes")
    if total_nodes > MAX_TOTAL_NODES:
        raise _SpecError(f"too many nodes ({total_nodes} > {MAX_TOTAL_NODES})")

    n_cols = len(layers)
    max_rows = max(len((l.get("nodes") if isinstance(l, dict) else None) or []) for l in layers)

    col_w, row_h = 3.4, 1.35
    box_w, box_h = 2.7, 0.92
    fig_w = min(MAX_FIG_INCHES, max(6.5, n_cols * col_w + 0.8))
    fig_h = min(MAX_FIG_INCHES, max(3.6, max_rows * row_h + 1.9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Header sits just above the tallest column's top box.
    header_y = box_h / 2.0 - row_h / 2.0 + 0.42
    ax.set_xlim(0, n_cols * col_w)
    ax.set_ylim(-(max_rows * row_h) - 0.55, header_y + 0.42)
    ax.axis("off")

    centers: Dict[str, tuple] = {}
    ids_seen: set = set()
    for ci, layer in enumerate(layers):
        layer = layer if isinstance(layer, dict) else {}
        nodes = layer.get("nodes") or []
        x = ci * col_w + col_w / 2.0
        ax.text(x, header_y, _txt(layer.get("label")), ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color=LAYER_EDGE[ci % len(LAYER_EDGE)], fontfamily=FAMILY)
        pad = (max_rows - len(nodes)) * row_h / 2.0
        for ri, node in enumerate(nodes):
            node = node if isinstance(node, dict) else {}
            nid = str(node.get("id") or f"n{ci}_{ri}")
            if nid in ids_seen:
                raise _SpecError(f"duplicate node id: {nid}")
            ids_seen.add(nid)
            y = -(pad + ri * row_h + row_h / 2.0)
            box = FancyBboxPatch((x - box_w / 2.0, y - box_h / 2.0), box_w, box_h,
                                 boxstyle="round,pad=0.06,rounding_size=0.14",
                                 linewidth=1.4,
                                 edgecolor=LAYER_EDGE[ci % len(LAYER_EDGE)],
                                 facecolor=LAYER_FILL[ci % len(LAYER_FILL)])
            ax.add_patch(box)
            ax.text(x, y, _wrap(node.get("label") or nid), ha="center", va="center",
                    fontsize=9.2, color=TEXT, fontfamily=FAMILY, linespacing=1.25)
            centers[nid] = (x, y)

    for e in edges:
        e = e if isinstance(e, dict) else {}
        a = centers.get(str(e.get("from")))
        b = centers.get(str(e.get("to")))
        if not a or not b:
            raise _SpecError(f"edge references unknown node: {e.get('from')} -> {e.get('to')}")
        (cx1, cy1), (cx2, cy2) = a, b
        dx, dy = cx2 - cx1, cy2 - cy1
        # Leave/enter on box boundaries; direction from ORIGINAL centers.
        if abs(dx) >= abs(dy):
            sx = 1.0 if dx >= 0 else -1.0
            x1, y1 = cx1 + sx * box_w / 2.0, cy1
            x2, y2 = cx2 - sx * (box_w / 2.0 + 0.06), cy2
        else:
            sy = 1.0 if dy >= 0 else -1.0
            x1, y1 = cx1, cy1 + sy * box_h / 2.0
            x2, y2 = cx2, cy2 - sy * (box_h / 2.0 + 0.06)
        style = "-" if str(e.get("style") or "solid") != "dashed" else (0, (5, 3))
        rad = 0.14 if (abs(dx) > 0.01 and abs(dy) > 0.01) else 0.0
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.25, color=ARROW, linestyle=style,
                                connectionstyle=f"arc3,rad={rad}", zorder=1)
        ax.add_patch(arrow)
        if e.get("label"):
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(mx, my + 0.14, _txt(e.get("label")), ha="center", va="bottom",
                    fontsize=8.0, color=ARROW, fontstyle="italic", fontfamily=FAMILY,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.85})
    return fig


def _render_line(plt: Any, spec: Dict[str, Any]) -> Any:
    from matplotlib.ticker import MaxNLocator

    series = spec.get("series") or []
    if not isinstance(series, list) or not series:
        raise _SpecError("line spec has no series")
    if len(series) > MAX_SERIES:
        raise _SpecError(f"too many series ({len(series)} > {MAX_SERIES})")

    parsed: List[tuple] = []
    for i, s in enumerate(series):
        s = s if isinstance(s, dict) else {}
        raw_points = s.get("points") or []
        if not isinstance(raw_points, list):
            raise _SpecError(f"series {i + 1} 'points' must be a list")
        if len(raw_points) > MAX_POINTS_PER_SERIES:
            raise _SpecError(
                f"too many points in series {i + 1} ({len(raw_points)} > {MAX_POINTS_PER_SERIES})"
            )
        pts = []
        for p in raw_points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            x, y = _finite(p[0]), _finite(p[1])
            # NaN/Inf are rejected per-point (they would corrupt autoscaling
            # or serialize as invalid JSON upstream), not silently zeroed.
            if x is None or y is None:
                continue
            pts.append((x, y))
        pts.sort(key=lambda p: p[0])
        parsed.append((s, pts))
    if not any(pts for _, pts in parsed):
        raise _SpecError("line spec has no points")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (s, pts) in enumerate(parsed):
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        c = LAYER_EDGE[i % len(LAYER_EDGE)]
        ax.plot(xs, ys, marker="o", markersize=5.5, linewidth=2.0, color=c,
                label=_txt(s.get("label") or f"series {i + 1}"))
        # Per-point value labels only at readable densities.
        if len(pts) <= 40:
            for x, y in pts:
                ax.annotate("%g" % y, (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8.2, color=c, fontfamily=FAMILY)
    ax.set_xlabel(_txt(spec.get("x_label")), fontsize=10, color=TEXT, fontfamily=FAMILY)
    ax.set_ylabel(_txt(spec.get("y_label")), fontsize=10, color=TEXT, fontfamily=FAMILY)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Optional honest y-axis anchor: auto-scaling a trajectory overstates
    # the gain (the Elo figure anchors at the 1200 tournament start).
    y_min = _finite(spec.get("y_min"))
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9aa3ad")
    ax.tick_params(colors="#5a636d", labelsize=9)
    ax.grid(axis="y", linewidth=0.5, color="#e3e6ea")
    ax.set_axisbelow(True)
    if len([1 for _, pts in parsed if pts]) > 1:
        ax.legend(frameon=False, fontsize=9)
    return fig


def render_chart(
    spec: Any,
    png_path: Path,
    *,
    pdf_sibling: bool = True,
) -> ChartRenderResult:
    """Render a chart spec to ``png_path`` (+ a ``.pdf`` sibling by default).

    Path containment is the CALLER's responsibility (the write_chart effect
    handler resolves both paths through the run's workspace scope before
    calling); this function only renders and writes the given paths.
    """
    warnings: List[str] = []
    if not isinstance(spec, dict):
        return ChartRenderResult(
            ok=False, rendered=False,
            error="chart spec must be an object",
            warnings=["#FALLBACK: write_chart received a non-object spec"],
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        # usetex would shell out to TeX (never acceptable in this trust
        # class); assert it off explicitly rather than trusting rcParams.
        matplotlib.rcParams["text.usetex"] = False
        import matplotlib.pyplot as plt
    except Exception as e:
        return ChartRenderResult(
            ok=False, rendered=False,
            error=f"matplotlib unavailable: {e}",
            warnings=["#FALLBACK: matplotlib unavailable — chart not rendered"],
        )

    fig = None
    try:
        _validate_spec_size(spec)
        kind = str(spec.get("kind") or "")
        if kind == "layered":
            fig = _render_layered(plt, spec)
        elif kind == "line":
            fig = _render_line(plt, spec)
        else:
            raise _SpecError(f"unknown kind: {kind!r} (expected layered|line)")

        title = _txt(spec.get("title")).strip()
        caption = _txt(spec.get("caption")).strip()
        if title:
            fig.suptitle(title, fontsize=12.5, fontweight="bold", color=TEXT,
                         fontfamily=FAMILY, y=0.985)
        if caption:
            fig.text(0.5, 0.012, "\n".join(textwrap.wrap(caption, width=110)),
                     ha="center", va="bottom", fontsize=8.6, color="#4a5560",
                     fontstyle="italic", fontfamily=FAMILY)
        fig.tight_layout(rect=(0, 0.05 if caption else 0.01, 1, 0.95 if title else 1.0))

        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(png_path), dpi=DPI, facecolor="white")
        pdf_path: Optional[Path] = None
        if pdf_sibling:
            pdf_path = png_path.with_suffix(".pdf")
            try:
                fig.savefig(str(pdf_path), facecolor="white")
            except Exception as e:
                pdf_path = None
                warnings.append(f"#FALLBACK: PDF sibling not written: {e}")

        # Deterministic gate: never report rendered on prose alone — the PNG
        # must exist with non-trivial bytes (coding-agent delivery lesson).
        if not png_path.is_file() or png_path.stat().st_size < 128:
            return ChartRenderResult(
                ok=False, rendered=False,
                error="render produced no usable PNG bytes",
                warnings=warnings + ["#FALLBACK: PNG missing or trivially small after render"],
            )
        return ChartRenderResult(
            ok=True, rendered=True, png_path=png_path, pdf_path=pdf_path, warnings=warnings,
        )
    except _SpecError as e:
        return ChartRenderResult(
            ok=False, rendered=False, error=f"invalid spec: {e}",
            warnings=warnings + [f"#FALLBACK: chart not rendered — invalid spec: {e}"],
        )
    except Exception as e:
        return ChartRenderResult(
            ok=False, rendered=False, error=f"render failed: {e}",
            warnings=warnings + [f"#FALLBACK: chart render failed: {e}"],
        )
    finally:
        # Figures leak across calls in a long-lived process without close.
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
