import sys
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
import stealzalut.core.zone_transfer as zt
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.finishing import Finishing
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.recipe import apply_global, apply_recipe, full_dose, parse_recipe
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr
S = Path(sys.argv[1]); PICS = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(PICS / "IMG_6295 - Copie.DNG"))
ref = bgr_to_rgb(load_image_bgr(PICS / "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg"))
model = zt.fit_zone_transfer(src, ref, 8)
recipe = parse_recipe((S / "ai_answer.txt").read_text(encoding="utf-8"))
table = apply_recipe(recipe, model.table, model.reference.ids)
params, _ = apply_global(recipe, GradingParams(), 1.0)
ft, fp = full_dose(table, params)
fp = replace(fp, luminance_strength=1.0)
m = model.with_table(ft).with_params(zt.ZoneBakeParams(tau=1.0))
small = create_thumbnail(src, 640)
def met(img):
    lab = srgb_to_oklab(img.reshape(-1, 3) / 255.0); lch = oklab_to_oklch(lab)
    return f"L={lab[:,0].mean():.3f} C={lch[:,1].mean():.3f} C>0.06={np.mean(lch[:,1]>0.06)*100:.0f}%"
print(f"{'REFERENCE':32s}", met(create_thumbnail(ref, 640)))
out = {}
for name, p, f in [
    ("A dosee 100% + luminance 100%", fp, None),
    ("B + intensite couleur 130%", replace(fp, chroma_strength=1.3), None),
    ("C + vibrance +50%", fp, Finishing(vibrance=0.5)),
    ("D + vibrance +50% sat +15%", fp, Finishing(vibrance=0.5, saturation=0.15)),
]:
    img = apply_lut(small, build_grid(m, 33, p, f).grid, 33); out[name] = img
    print(f"{name:32s}", met(img))
h = small.shape[0]; refs = create_thumbnail(ref, 640)
ref_fit = cv2.resize(refs, (round(refs.shape[1]*h/refs.shape[0]), h))
tiles = [("REFERENCE", ref_fit), ("A dosee+luminance 100%", out["A dosee 100% + luminance 100%"]), ("D + vibrance 50 sat 15", out["D + vibrance +50% sat +15%"])]
row = []
for n, im in tiles:
    im = im.copy(); cv2.rectangle(im, (0,0), (12+11*len(n), 26), (0,0,0), -1)
    cv2.putText(im, n, (6,19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA); row.append(im); row.append(np.full((h,6,3),20,np.uint8))
cv2.imwrite(str(S/"tokyo_boost.jpg"), cv2.cvtColor(np.hstack(row[:-1]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
