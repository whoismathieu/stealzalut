"""P2: robustness of the look transfer on other source / reference pairs.

The extra pictures are whatever sits in the user's Pictures folder; they are
only here to check that nothing breaks outside the two cases of the brief.
"""

from __future__ import annotations

import sys
from pathlib import Path

import harness as h
from look_proto import fit_look
from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.palette import colour_stats

out = Path(sys.argv[1])
PAIRS = [
    ("IMG_20260810_184343.jpg", "857957197_18632319271014205_4520637185456133031_n.jpg"),
    ("IMG_1660.jpg", "883176110_18139094545636140_2217741488461755966_n.jpg"),
    ("IMG_7232.jpg", "847802485_18642764905026341_5825556147860532178_n.jpg"),
    ("IMG_8856.HEIC", "825352432_18637269925026341_5988942847425001843_n.jpg"),
    ("IMG_6295.DNG", "828648715_18637269904026341_1013748936910041210_n.jpg"),
    ("IMG_20260810_184343.jpg", "811994013_18636844090046035_4466708029834710409_n.jpg"),
]
if len(sys.argv) > 2 and sys.argv[2] == "list":
    names = sorted(p for p in h.PICS.iterdir() if p.suffix.lower() in (".jpg", ".png", ".heic"))
    tiles = [(p.name[:18], h.small(h.load(p))) for p in names]
    h.save(out / "p2_candidates.jpg", h.sheet(tiles, 6, 200))
    sys.exit(0)

for i, (s_name, r_name) in enumerate(PAIRS, start=1):
    src, ref = h.load(h.PICS / s_name), h.load(h.PICS / r_name)
    view = h.small(src)
    mono = colour_stats(ref).is_monochrome
    current = "ambiance" if mono else "zones"
    img_cur, folds_cur, auto = h.render(fit_transfer(src, ref, method=current), view)
    look = fit_look(src, ref, protect_vivid=0.8 if mono else 0.0)
    img_new, folds_new, _ = h.render(look, view)
    print(f"== pair {i}: {s_name} x {r_name} (mono={mono})")
    print(h.fmt("source", h.metrics(view)))
    print(h.fmt("reference", h.metrics(h.small(ref))))
    print(h.fmt(f"ACTUEL {current}", h.metrics(img_cur)), f"folds={folds_cur:.4f} auto={auto}")
    print(h.fmt("LOOK", h.metrics(img_new)), f"folds={folds_new:.4f}")
    tiles = [
        ("SOURCE", view),
        ("REFERENCE", h.small(ref)),
        (f"ACTUEL {current} (auto)", img_cur),
        ("LOOK", img_new),
    ]
    h.save(out / f"p2_pair_{i}.jpg", h.sheet(tiles, 4, 400))
