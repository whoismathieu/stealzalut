"""Experiment: transfer only what depends little on the subject.

Lightness and colour are scored apart (in the look model the output
lightness only depends on the tone option, the colour shift on the colour
option), so every option is measured from one fit per case.

Tone options      : identity | end points only | half of the full match | full match (today)
Colour options    : none | tint of the neutral axis per tone | mean cast per tone | full transport (today)
"""

import sys
from pathlib import Path

import numpy as np

from stealzalut.core.look import fit_look
from stealzalut.devtools.ground_truth import GRADES, as_picture, load_folder, pairs, split

LEVELS = np.linspace(0.0, 1.0, 21)
WIDTH = 0.05
MIN_WEIGHT = 30.0  # pixels (weighted) needed to trust a level
KEEP = 0.4  # share of the pixels nearest to the centre kept at each refinement


def neutral_axis(flat: np.ndarray) -> np.ndarray:
    """Colour (a, b) of the densest, least coloured mass at each lightness level."""
    axis = np.full((len(LEVELS), 2), np.nan)
    for i, level in enumerate(LEVELS):
        w = np.exp(-0.5 * ((flat[:, 0] - level) / WIDTH) ** 2)
        chosen = w > 0.05
        if w[chosen].sum() < MIN_WEIGHT:
            continue
        ab, ww = flat[chosen, 1:], w[chosen]
        centre = np.array([np.median(ab[:, 0]), np.median(ab[:, 1])])
        for _ in range(4):
            dist = np.hypot(*(ab - centre).T)
            near = dist <= np.quantile(dist, KEEP)
            centre = (ab[near] * ww[near, None]).sum(axis=0) / ww[near].sum()
        axis[i] = centre
    valid = ~np.isnan(axis[:, 0])
    if not valid.any():
        return np.zeros_like(axis)
    for c in range(2):
        axis[:, c] = np.interp(LEVELS, LEVELS[valid], axis[valid, c])
    return axis


def at(axis: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(lightness, LEVELS, axis[:, c]) for c in range(2)])


def end_points(l_src: np.ndarray, l_ref: np.ndarray):
    qs = np.quantile(l_src, [0.01, 0.999])
    qr = np.quantile(l_ref, [0.01, 0.999])
    gain = (qr[1] - qr[0]) / max(qs[1] - qs[0], 1e-6)
    return lambda x: np.clip(qr[0] + gain * (x - qs[0]), 0.0, 1.0)


TONES = ("identity", "ends", "half", "full")
COLOURS = ("none", "axis@identity", "axis@ends", "axis@full", "cast", "full")

images = load_folder(Path(sys.argv[1]))
lot_name = sys.argv[2] if len(sys.argv) > 2 else "reglage"
members = split(images)[0 if lot_name == "reglage" else 1]
err_l = {t: {True: [], False: []} for t in TONES}
err_c = {c: {True: [], False: []} for c in COLOURS}
base = {"L": {True: [], False: []}, "C": {True: [], False: []}}

for s_i, r_i in pairs(len(members), 2):
    source, reference = members[s_i], members[r_i]
    same = s_i == r_i
    src = source.flat
    axis_s = neutral_axis(src)
    for grade in GRADES:
        truth = grade.grade(src)
        ref_lab = grade.grade(reference.lab)
        ref = ref_lab.reshape(-1, 3)
        model = fit_look(as_picture(source.lab), as_picture(ref_lab)).with_protect_vivid(0.0)
        axis_r = neutral_axis(ref)
        tone = {
            "identity": src[:, 0],
            "ends": end_points(src[:, 0], ref[:, 0])(src[:, 0]),
            "full": model.tone(src[:, 0]),
        }
        tone["half"] = 0.5 * (tone["identity"] + tone["full"])
        base["L"][same].append(float(np.abs(src[:, 0] - truth[:, 0]).mean()))
        base["C"][same].append(float(np.hypot(*(src[:, 1:] - truth[:, 1:]).T).mean()))
        for name in TONES:
            err_l[name][same].append(float(np.abs(tone[name] - truth[:, 0]).mean()))
        shifts = {
            "none": np.zeros((len(src), 2)),
            "full": model.chroma_shift(src),
            "cast": model.with_separation(0.0).chroma_shift(src),
        }
        for t in ("identity", "ends", "full"):
            shifts[f"axis@{t}"] = at(axis_r, tone[t]) - at(axis_s, src[:, 0])
        for name in COLOURS:
            diff = src[:, 1:] + shifts[name] - truth[:, 1:]
            err_c[name][same].append(float(np.hypot(*diff.T).mean()))

print(f"## lot {lot_name}: {len(members)} images. Mean absolute Oklab error; (ratio to doing nothing)")
for same, title in ((True, "same content"), (False, "other content")):
    nl, nc = np.mean(base["L"][same]), np.mean(base["C"][same])
    print(f"-- {title}: doing nothing  L {nl:.4f}  colour {nc:.4f}")
    for name in TONES:
        v = np.mean(err_l[name][same])
        print(f"   tone   {name:14s} {v:.4f}  ({v / nl:5.2f})")
    for name in COLOURS:
        v = np.mean(err_c[name][same])
        print(f"   colour {name:14s} {v:.4f}  ({v / nc:5.2f})")
