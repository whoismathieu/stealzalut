"""P4: reading only part of the reference (zone weights) on case B."""

from __future__ import annotations

import sys
from pathlib import Path

import harness as h
from stealzalut.core.look import fit_look, zone_weights
from stealzalut.core.palette import DEFAULT_ZONE_COUNT, analyze_zones

out = Path(sys.argv[1])
src, ref = h.load(h.SOURCE), h.load(h.REF_B)
view = h.small(src)
palette = analyze_zones(ref, DEFAULT_ZONE_COUNT)
for zone in palette.zones:
    print(f"R{zone.id} {zone.hex} area={zone.area * 100:4.1f}% L={zone.mean[0]:.2f}")

darkest = palette.zones[0].id
largest = max(palette.zones, key=lambda z: z.area).id
tiles = [("REFERENCE", h.small(ref))]
for name, focus in (
    ("LOOK (tout)", {}),
    (f"R{darkest} ignoree (noirs)", {darkest: 0.0}),
    (f"R{largest} ignoree (ciel)", {largest: 0.0}),
    (f"R{largest} privilegiee", {largest: 2.0}),
):
    weights = zone_weights(palette.labels, focus) if focus else None
    img, folds, _ = h.render(fit_look(src, ref, weights), view)
    print(h.fmt(name, h.metrics(img)), f"folds={folds:.4f}")
    tiles.append((name, img))
h.save(out / "p4_focus.jpg", h.sheet(tiles, 5, 360))
