"""Where does the look model lose chroma?  model direct / no gamut / LUT 33 / LUT 64."""

from __future__ import annotations

import numpy as np

import harness as h
from look_proto import fit_look
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab

src = h.load(h.SOURCE)
view = h.small(src)
flat = view.reshape(-1, 3) / 255.0


def stats(lab: np.ndarray) -> str:
    c = np.hypot(lab[:, 1], lab[:, 2])
    return f"C={c.mean():.4f} vivid={np.mean(c > 0.06) * 100:5.1f}% col={np.mean(c >= 0.05) * 100:5.1f}%"


for case, ref_path in (("A", h.REF_A), ("B", h.REF_B)):
    ref = h.load(ref_path)
    print("== case", case)
    print("reference        ", stats(srgb_to_oklab(h.small(ref).reshape(-1, 3) / 255.0)))
    for sigma in (1.5, 0.75):
        model = fit_look(src, ref, sigma_fine=sigma)
        lab = srgb_to_oklab(flat)
        raw = lab.copy()
        raw[:, 0] = np.interp(lab[:, 0], model.tone_x, model.tone_y)
        raw[:, 1:] += model.chroma_shift(lab)
        print(f"sigma={sigma} field    ", stats(raw))
        print(f"sigma={sigma} +gamut   ", stats(srgb_to_oklab(model.apply(flat))))
        for size in (33, 64):
            img, _, _ = h.render(model, view, size=size)
            print(f"sigma={sigma} LUT {size}   ", stats(srgb_to_oklab(img.reshape(-1, 3) / 255.0)))
        _ = oklab_to_oklch
