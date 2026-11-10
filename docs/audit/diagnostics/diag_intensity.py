"""What limits the intensity of the Tokyo render? One lever at a time."""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.recipe import apply_global, apply_recipe, full_dose, parse_recipe
from stealzalut.core.zone_transfer import ZoneBakeParams, fit_zone_transfer
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(PICS / "IMG_6295 - Copie.DNG"))
ref = bgr_to_rgb(load_image_bgr(PICS / "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg"))
model = fit_zone_transfer(src, ref, 8)
print("analysis", model.analysis_id)
recipe = parse_recipe((S / "ai_answer.txt").read_text(encoding="utf-8"))
table = apply_recipe(recipe, model.table, model.reference.ids)
params, tau = apply_global(recipe, GradingParams(), 1.0)
ftable, fparams = full_dose(table, params)
small = create_thumbnail(src, 640)


def metrics(img: np.ndarray) -> str:
    lab = srgb_to_oklab(img.reshape(-1, 3) / 255.0)
    lch = oklab_to_oklch(lab)
    dark = lab[:, 0] < 0.35
    return (
        f"L moy={lab[:, 0].mean():.3f}  C moy={lch[:, 1].mean():.3f}  "
        f"ombres: a={lab[dark, 1].mean():+.3f} C={lch[dark, 1].mean():.3f} ({dark.mean()*100:.0f}% px)  "
        f"px C>0.06={np.mean(lch[:, 1] > 0.06)*100:.0f}%"
    )


print(f"{'SOURCE':34s} {metrics(small)}")
print(f"{'REFERENCE':34s} {metrics(create_thumbnail(ref, 640))}")

variants = [
    ("1 dosee 100% (lum .35, soft)", ftable, fparams, tau),
    ("2 + luminance 1.0", ftable, replace(fparams, luminance_strength=1.0), tau),
    ("3 + transitions medium (tau 1)", ftable, replace(fparams, luminance_strength=1.0), 1.0),
    ("4 + transitions sharp (tau .5)", ftable, replace(fparams, luminance_strength=1.0), 0.5),
    ("5 lum .35 mais tau .5", ftable, fparams, 0.5),
    ("6 auto, reglages par defaut", model.table, GradingParams(), 1.0),
]
images = []
for name, t, p, ta in variants:
    m = model.with_table(t).with_params(ZoneBakeParams(tau=ta))
    r = build_grid(m, 33, p)
    img = apply_lut(small, r.grid, 33)
    images.append((name, img))
    print(f"{name:34s} {metrics(img)}  folds={r.fold_fraction:.4f} smooth={r.auto_smoothed}")

h = small.shape[0]
refs = create_thumbnail(ref, 640)
ref_fit = cv2.resize(refs, (round(refs.shape[1] * h / refs.shape[0]), h))
tiles = [("REFERENCE", ref_fit), ("SOURCE", small)] + [images[0], images[3]]
out = []
for name, im in tiles:
    im = im.copy()
    cv2.rectangle(im, (0, 0), (12 + 11 * len(name), 26), (0, 0, 0), -1)
    cv2.putText(im, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out.append(im)
gap = np.full((h, 6, 3), 20, np.uint8)
top = np.hstack([out[0], gap, out[1]])
bottom = np.hstack([out[2], gap, out[3]])
w = max(top.shape[1], bottom.shape[1])
pad = lambda a: np.hstack([a, np.full((h, w - a.shape[1], 3), 20, np.uint8)])
sheet = np.vstack([pad(top), np.full((6, w, 3), 20, np.uint8), pad(bottom)])
cv2.imwrite(str(S / "tokyo_intensity.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
