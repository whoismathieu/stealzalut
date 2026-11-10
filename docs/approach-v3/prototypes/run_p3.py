"""P3: what would spatial masks add?  (hypotheses H2 and H3 of the brief)

Crude but honest mask: a vertical gradient.  The top half of the source takes
the look of the top half of the reference, the bottom half the bottom's.

* COMPOSITE: the two LUTs blended with the mask (needs a mask in the host).
* H2: one single LUT regressed on the composite (mean target per LUT cell).
* LOOK: the global look LUT (no spatial information at all).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import harness as h
from look_proto import fit_look
from stealzalut.core.colorspace import srgb_to_oklab
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.lut_engine import generate_neutral_grid

out = Path(sys.argv[1])
src = h.load(h.SOURCE)
view = h.small(src)
SIZE = 33


def halves(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = img.shape[0] // 2
    return img[:mid], img[mid:]


def lab_of(img: np.ndarray) -> np.ndarray:
    return srgb_to_oklab(img.reshape(-1, 3) / 255.0)


def regress_lut(pixels: np.ndarray, targets: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Mean target per LUT node (trilinear splat); *fallback* where no pixel falls."""
    lut_num = np.zeros((SIZE, SIZE, SIZE, 3))
    lut_cnt = np.zeros((SIZE, SIZE, SIZE))
    pos = pixels.reshape(-1, 3) / 255.0 * (SIZE - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, SIZE - 2)
    f = pos - i0
    tgt = targets.reshape(-1, 3) / 255.0
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = (
                    (f[:, 2] if db else 1 - f[:, 2])
                    * (f[:, 1] if dg else 1 - f[:, 1])
                    * (f[:, 0] if dr else 1 - f[:, 0])
                )
                idx = (i0[:, 2] + db, i0[:, 1] + dg, i0[:, 0] + dr)
                np.add.at(lut_cnt, idx, w)
                np.add.at(lut_num, idx, w[:, None] * tgt)
    fb = fallback.reshape(SIZE, SIZE, SIZE, 3)
    prior = 0.5
    grid = (lut_num + prior * fb) / (lut_cnt + prior)[..., None]
    return grid.reshape(-1, 3)


for case, ref_path, pv in (("A", h.REF_A, 0.0), ("B", h.REF_B, 0.8)):
    ref = h.load(ref_path)
    s_top, s_bot = halves(src)
    r_top, r_bot = halves(ref)
    glob = fit_look(src, ref, protect_vivid=pv)
    img_glob, _, _ = h.render(glob, view)
    img_top, _, _ = h.render(fit_look(s_top, r_top, protect_vivid=pv), view)
    img_bot, _, _ = h.render(fit_look(s_bot, r_bot, protect_vivid=pv), view)
    rows = np.linspace(0.0, 1.0, view.shape[0])
    mask = np.clip((rows - 0.35) / 0.30, 0.0, 1.0)[:, None, None]
    composite = (img_top * (1 - mask) + img_bot * mask + 0.5).astype(np.uint8)
    base_grid = glob.apply(generate_neutral_grid(SIZE))
    img_h2 = apply_lut(view, regress_lut(view, composite, base_grid), SIZE)

    def rms(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((lab_of(a) - lab_of(b)) ** 2, axis=1))))

    print(f"== case {case} (RMS Oklab distance, 0.02 is clearly visible)")
    print(f"composite vs LOOK global : {rms(composite, img_glob):.4f}")
    print(f"composite vs H2 single LUT: {rms(composite, img_h2):.4f}  <- what one .cube cannot do")
    print(f"H2 single LUT vs LOOK     : {rms(img_h2, img_glob):.4f}  <- what H2 buys over LOOK")
    print(h.fmt("LOOK", h.metrics(img_glob)))
    print(h.fmt("H2", h.metrics(img_h2)))
    print(h.fmt("COMPOSITE", h.metrics(composite)))
    tiles = [
        ("REFERENCE", h.small(ref)),
        ("LOOK (1 LUT)", img_glob),
        ("H2 (1 LUT ajustee sur masque)", img_h2),
        ("COMPOSITE (2 LUT + masque)", composite),
    ]
    h.save(out / f"p3_case_{case}.jpg", h.sheet(tiles, 4, 400))
