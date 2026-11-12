"""Bench of the locality setting of the look method (reach of the displacement).

Fits the look once per case, then applies it with several reaches.  Measured
on the model directly (no LUT), on the analysis colours.

Usage: bench_reach.py FOLDER reglage|controle
"""

import sys
from pathlib import Path

import numpy as np

from stealzalut.core import look
from stealzalut.devtools.ground_truth import GRADES, as_picture, load_folder, pairs, split

REACHES: tuple[float | None, ...] = (None, 0.20, 0.12, 0.08, 0.05, 0.035, 0.02)
RATIOS = (1.0, 2.0, 3.0)

images = load_folder(Path(sys.argv[1]))
lot = sys.argv[2]
members = split(images)[0 if lot == "reglage" else 1]
keys = [(reach, ratio) for reach in REACHES for ratio in (RATIOS if reach else (2.0,))]
acc: dict = {k: {True: [], False: []} for k in keys}
base: dict = {True: [], False: []}

for s_i, r_i in pairs(len(members), 2):
    source, reference = members[s_i], members[r_i]
    same = s_i == r_i
    for grade in GRADES:
        truth = grade.grade(source.flat)
        nothing = truth - source.flat
        n_all = float(np.sqrt((nothing**2).sum(axis=1)).mean())
        if n_all < 0.004:
            continue
        base[same].append(
            (float(np.abs(nothing[:, 0]).mean()), float(np.hypot(*nothing[:, 1:].T).mean()), n_all)
        )
        model = look.fit_look(as_picture(source.lab), as_picture(grade.grade(reference.lab)))
        for reach, ratio in keys:
            look.TONE_REACH_RATIO = ratio
            diff = look.replace(model, reach=reach).apply_lab(source.flat) - truth
            acc[(reach, ratio)][same].append(
                (
                    float(np.abs(diff[:, 0]).mean()),
                    float(np.hypot(*diff[:, 1:].T).mean()),
                    float(np.sqrt((diff**2).sum(axis=1)).mean()),
                )
            )

print(f"## lot {lot}: {len(members)} images")
print("reach  ratio | same: L / colour / residual (mean, median) | other: L / colour / residual (mean, median) / better than nothing")
for key in keys:
    cells = []
    for same in (True, False):
        e, n = np.array(acc[key][same]), np.array(base[same])
        res = e[:, 2] / n[:, 2]
        cell = f"{e[:, 0].mean() / n[:, 0].mean():.2f} / {e[:, 1].mean() / n[:, 1].mean():.2f} / {res.mean():.2f}, {np.median(res):.2f}"
        if not same:
            cell += f" / {100 * np.mean(res < 1.0):.0f} %"
        cells.append(cell)
    reach = "none " if key[0] is None else f"{key[0]:.3f}"
    print(f"{reach}  {key[1]:.0f}     | {cells[0]} | {cells[1]}")
