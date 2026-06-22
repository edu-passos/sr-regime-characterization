r"""Regenerate the paper's pipeline figure (Fig. 1) from a 'Figure 1 vN.html' design.

The design files are self-contained HTML bundles: the actual figure is an SVG embedded
(JSON-escaped) in the <script type="__bundler/template"> tag. This extracts that SVG,
inlines its CSS custom properties (cairosvg does not resolve var()), and writes:
  figures/fig_pipeline.svg          (editable source of truth)
  figures/fig_pipeline.pdf          (the figure the paper includes)
  figures/fig_pipeline.png          (raster, used in the README)

Usage:  python scripts/render_fig_pipeline.py ["Figure 1 v5.html"]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cairosvg

FIG = Path(__file__).resolve().parent.parent / "figures"

# CSS custom properties from the bundle's :root (cairosvg can't resolve var()).
CSS_VARS = {
    "--band-signal": "#ECF1F5", "--band-symbol": "#FBF5E6", "--band-concept": "#FBEBDC",
    "--ink": "#1a1a1a",
    "--struct": "#4F90C0", "--struct-fill": "#E2ECF4",
    "--behav": "#D88A55", "--behav-fill": "#F6E2D1",
    "--calib": "#8E7CA8", "--calib-fill": "#ECE6F1",
}


def extract_svg(html: str) -> str:
    m = re.search(r'<script type="__bundler/template">\s*(.*?)\s*</script>', html, re.S)
    if not m:
        raise SystemExit("no __bundler/template script found in HTML")
    inner = json.loads(m.group(1))          # decodes / -> / and \n etc.
    svg = re.search(r"<svg\b.*?</svg>", inner, re.S)
    if not svg:
        raise SystemExit("no <svg> found in the bundle template")
    svg = svg.group(0)
    # resolve var(--x); longest keys first so --struct-fill is not shadowed by --struct
    for key in sorted(CSS_VARS, key=len, reverse=True):
        svg = svg.replace(f"var({key})", CSS_VARS[key])
    if "var(--" in svg:
        leftover = sorted(set(re.findall(r"var\((--[\w-]+)\)", svg)))
        raise SystemExit(f"unresolved CSS vars (add to CSS_VARS): {leftover}")
    return svg


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "Figure 1 v5.html"
    html = (FIG / src).read_text()
    svg = extract_svg(html)

    (FIG / "fig_pipeline.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(FIG / "fig_pipeline.pdf"))
    cairosvg.svg2png(bytestring=svg.encode(),
                     write_to=str(FIG / "fig_pipeline.png"), scale=6)
    print(f"regenerated fig_pipeline.{{svg,pdf,png}} from {src!r}")


if __name__ == "__main__":
    main()
