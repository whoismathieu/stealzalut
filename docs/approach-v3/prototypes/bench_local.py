"""Experiment: local registration of colour densities.

Distribution matching moves every source colour until the two histograms
coincide, so a reference full of grass turns a blue sky green.  Here a
colour only looks at reference colours *near it*: the displacement is the
difference between the local centroids of the two densities, weighted by how
much reference mass sits there.  Mass that has no counterpart is left alone
instead of being transported far.
"""

import sys
from pathlib import Path

import numpy as np

from stealzalut.devtools.ground_truth import GRADES, load_folder, pairs, split

SAMPLES = 2500
ITERS = 3


def sample(flat: np.ndarray, n: int = SAMPLES) -> np.ndarray:
    return flat[np.linspace(0, len(flat) - 1, n).astype(int)]


def kernel(points: np.ndarray, centres: np.ndarray, rho: float) -> np.ndarray:
    d2 = ((centres[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-0.5 * d2 / rho**2)


def local_displacement(src: np.ndarray, ref: np.ndarray, query: np.ndarray, rho: float) -> np.ndarray:
    """Displacement (N, 3) of *query* colours registering *src* samples onto *ref* locally."""
    start = src.copy()
    moved = src.copy()
    for _ in range(ITERS):
        k_ref = kernel(ref, moved, rho)
        k_src = kernel(moved, moved, rho)
        w_ref, w_src = k_ref.sum(axis=1), k_src.sum(axis=1)
        m_ref = (k_ref @ ref) / np.maximum(w_ref, 1e-9)[:, None]
        m_src = (k_src @ moved) / w_src[:, None]
        trust = np.minimum(1.0, w_ref / w_src)
        step = (m_ref - m_src) * trust[:, None]
        step = (k_src @ step) / w_src[:, None]  # smooth the field over the source density
        moved = moved + step
    total = moved - start
    k_q = kernel(start, query, rho)
    w_q = k_q.sum(axis=1)
    return (k_q @ total) / np.maximum(w_q, 1e-9)[:, None] * np.minimum(1.0, w_q)[:, None]


images = load_folder(Path(sys.argv[1]))
lot_name = sys.argv[2]
rhos = [float(v) for v in sys.argv[3].split(",")]
members = split(images)[0 if lot_name == "reglage" else 1]
err = {rho: {"L": {True: [], False: []}, "C": {True: [], False: []}} for rho in rhos}
base = {"L": {True: [], False: []}, "C": {True: [], False: []}}
wins = {rho: [] for rho in rhos}

for s_i, r_i in pairs(len(members), 2):
    source, reference = members[s_i], members[r_i]
    same = s_i == r_i
    src = source.flat
    query = sample(src, 6000)
    for grade in GRADES:
        truth = grade.grade(query)
        ref = sample(grade.grade(reference.flat))
        n_l = float(np.abs(query[:, 0] - truth[:, 0]).mean())
        n_c = float(np.hypot(*(query[:, 1:] - truth[:, 1:]).T).mean())
        base["L"][same].append(n_l)
        base["C"][same].append(n_c)
        for rho in rhos:
            out = query + local_displacement(sample(src), ref, query, rho)
            e_l = float(np.abs(out[:, 0] - truth[:, 0]).mean())
            e_c = float(np.hypot(*(out[:, 1:] - truth[:, 1:]).T).mean())
            err[rho]["L"][same].append(e_l)
            err[rho]["C"][same].append(e_c)
            if not same:
                total_n = float(np.sqrt(((query - truth) ** 2).sum(axis=1)).mean())
                total_e = float(np.sqrt(((out - truth) ** 2).sum(axis=1)).mean())
                wins[rho].append(total_e < total_n)

print(f"## lot {lot_name}: {len(members)} images. Mean absolute Oklab error (ratio to doing nothing)")
for same, title in ((True, "same content"), (False, "other content")):
    nl, nc = np.mean(base["L"][same]), np.mean(base["C"][same])
    print(f"-- {title}: doing nothing  L {nl:.4f}  colour {nc:.4f}")
    for rho in rhos:
        vl, vc = np.mean(err[rho]["L"][same]), np.mean(err[rho]["C"][same])
        extra = "" if same else f" | better than nothing in {100 * np.mean(wins[rho]):.0f} % of cases"
        print(f"   rho {rho:.3f}:  L {vl:.4f} ({vl / nl:4.2f})   colour {vc:.4f} ({vc / nc:4.2f}){extra}")
