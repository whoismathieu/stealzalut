"""Experiment: read the reference against average photo statistics, not against the source.

Baseline: fit_look(source = Q, reference = G(P)).
Prior:    fit_look(source = pooled pixels of OTHER pictures, reference = G(P)), applied to Q.
Half:     the pooled "source" is half prior, half Q.

The prior of each lot is built from the other lot only (no leakage).
"""

import sys
from pathlib import Path

import numpy as np

from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.grading import build_grid
from stealzalut.core.lut_apply import apply_lut_colors
from stealzalut.devtools.ground_truth import (
    BENCH_LUT,
    BENCH_SPACE,
    GRADES,
    _into_gamut,
    as_picture,
    load_folder,
    pairs,
    split,
)

SIDE = 256
images = load_folder(Path(sys.argv[1]))
lots = split(images)


def pooled(members, extra=None, share=0.0):
    """A SIDE x SIDE 'picture' whose pixels are drawn evenly from *members* (+ *extra*)."""
    total = SIDE * SIDE
    n_extra = int(total * share) if extra is not None else 0
    per = (total - n_extra) // len(members)
    parts = []
    for m in members:
        flat = m.flat
        parts.append(flat[np.linspace(0, len(flat) - 1, per).astype(int)])
    if n_extra:
        parts.append(extra.flat[np.linspace(0, len(extra.flat) - 1, n_extra).astype(int)])
    lab = np.concatenate(parts)
    lab = np.concatenate([lab, lab[: total - len(lab)]])
    return lab.reshape(SIDE, SIDE, 3)


def run(fit_source_lab, source, reference_lab):
    model = fit_transfer(as_picture(fit_source_lab), as_picture(reference_lab), method="look")
    grid = build_grid(model, BENCH_LUT, None, None, BENCH_SPACE).grid
    encoded = BENCH_SPACE.from_oklab(_into_gamut(source.flat))
    return BENCH_SPACE.to_oklab(apply_lut_colors(encoded, grid, BENCH_LUT))


for lot_index, name in enumerate(("reglage", "controle")):
    members, others = lots[lot_index], lots[1 - lot_index]
    prior = pooled(others)
    scores = {v: {True: [], False: []} for v in ("baseline", "prior", "half")}
    parts = {v: {"L": [], "C": []} for v in scores}
    for s_i, r_i in pairs(len(members), 2):
        source, reference = members[s_i], members[r_i]
        for grade in GRADES:
            truth = grade.grade(source.flat)
            nothing = float(np.sqrt(((source.flat - truth) ** 2).sum(axis=1)).mean())
            ref_lab = grade.grade(reference.lab)
            for variant, fit_lab in (
                ("baseline", source.lab),
                ("prior", prior),
                ("half", pooled(others, source, 0.5)),
            ):
                diff = run(fit_lab, source, ref_lab) - truth
                error = float(np.sqrt((diff**2).sum(axis=1)).mean())
                scores[variant][s_i == r_i].append(error / nothing)
                if s_i != r_i:
                    parts[variant]["L"].append(float(np.abs(diff[:, 0]).mean()))
                    parts[variant]["C"].append(float(np.hypot(diff[:, 1], diff[:, 2]).mean()))
    print(f"## lot {name} ({len(members)} images, prior from the other lot)")
    for variant, by in scores.items():
        print(
            f"{variant:9s} same: mean {np.mean(by[True]):.2f} median {np.median(by[True]):.2f} | "
            f"other: mean {np.mean(by[False]):.2f} median {np.median(by[False]):.2f} "
            f"worst {np.max(by[False]):.1f} | other errL {np.mean(parts[variant]['L']):.3f} "
            f"errC {np.mean(parts[variant]['C']):.3f}"
        )
    sys.stdout.flush()
