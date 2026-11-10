"""Tokyo case after the fix: recipe as imported vs. dosed at 100 %."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
from stealzalut.core.colorspace import srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.recipe import apply_global, apply_recipe, colour_dose, full_dose, parse_recipe
from stealzalut.core.zone_transfer import ZoneBakeParams, fit_zone_transfer
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(PICS / "IIIIIIIMG_6295 - Copie.DNG"))
ref = bgr_to_rgb(load_image_bgr(PICS / "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg"))
model = fit_zone_transfer(src, ref, 8)
recipe = parse_recipe((S / "ai_answer.txt").read_text(encoding="utf-8"))
table = apply_recipe(recipe, model.table, model.reference.ids)
params, tau = apply_global(recipe, GradingParams(), 1.0)
areas = {z.id: z.area for z in model.source.zones}
print(f"dose recette : {colour_dose(table, params, areas) * 100:.0f} %")
full_table, full_params = full_dose(table, params)
small = create_thumbnail(src, 640)


def shadows_a(img):
    lab = srgb_to_oklab(img.reshape(-1, 3) / 255.0)
    return lab[lab[:, 0] < 0.35, 1].mean()


tiles = [("SOURCE", small)]
for name, t, p in (("RECETTE TELLE QUELLE", table, params), ("DOSEE A 100 %", full_table, full_params)):
    m = model.with_table(t).with_params(ZoneBakeParams(tau=tau))
    r = build_grid(m, 33, p)
    img = apply_lut(small, r.grid, 33)
    print(f"{name}: ombres a={shadows_a(img):+.3f} folds={r.fold_fraction:.4f} lissage force={r.auto_smoothed}")
    tiles.append((name, img))
ref_small = create_thumbnail(ref, 640)
print(f"source ombres a={shadows_a(small):+.3f} reference ombres a={shadows_a(ref_small):+.3f}")

h = small.shape[0]
ref_fit = cv2.resize(ref_small, (round(ref_small.shape[1] * h / ref_small.shape[0]), h))
row = []
for name, im in [("REFERENCE", ref_fit)] + tiles:
    im = im.copy()
    cv2.rectangle(im, (0, 0), (12 + 11 * len(name), 26), (0, 0, 0), -1)
    cv2.putText(im, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    row.append(im)
top = np.hstack([row[0], np.full((h, 6, 3), 20, np.uint8), row[1]])
bottom = np.hstack([row[2], np.full((h, 6, 3), 20, np.uint8), row[3]])
width = max(top.shape[1], bottom.shape[1])
pad = lambda a: np.hstack([a, np.full((h, width - a.shape[1], 3), 20, np.uint8)])
sheet = np.vstack([pad(top), np.full((6, width, 3), 20, np.uint8), pad(bottom)])
cv2.imwrite(str(S / "tokyo_after.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
