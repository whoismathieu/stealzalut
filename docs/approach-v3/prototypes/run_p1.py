"""P1: look transfer vs current baselines on cases A and B (metrics + sheets)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import harness as h
from look_proto import fit_look

out = Path(sys.argv[1])
src = h.load(h.SOURCE)
view = h.small(src)
VARIANTS = {
    "A": [(0.0, 1.0)],
    "B": [(0.8, 1.0)],
}

for case, ref_path in (("A", h.REF_A), ("B", h.REF_B)):
    ref = h.load(ref_path)
    print(f"== case {case}")
    print(h.fmt("source", h.metrics(view)))
    print(h.fmt("reference", h.metrics(h.small(ref))))
    tiles = [("SOURCE", view), ("REFERENCE", h.small(ref))]
    if case == "A":
        model, params = h.baseline_zones_ai(src, ref)
        img, folds, auto = h.render(model, view, params)
        name = "ACTUEL zones+IA 100%"
    else:
        img, folds, auto = h.render(h.baseline_ambiance(src, ref), view)
        name = "ACTUEL ambiance"
    print(h.fmt(name, h.metrics(img)), f"folds={folds:.4f} auto={auto}")
    tiles.append((name, img))
    for pv, sep in VARIANTS[case]:
        t0 = time.perf_counter()
        look = fit_look(src, ref, protect_vivid=pv, separation=sep)
        t1 = time.perf_counter()
        img, folds, auto = h.render(look, view)
        t2 = time.perf_counter()
        name = f"LOOK viv={pv:.1f} sep={sep:.1f}"
        print(h.fmt(name, h.metrics(img)), f"folds={folds:.4f} fit={t1 - t0:.2f}s bake={t2 - t1:.2f}s")
        tiles.append((name, img))
    h.save(out / f"p1_case_{case}.jpg", h.sheet(tiles, 2, 560))
