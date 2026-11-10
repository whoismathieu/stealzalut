"""Purple monochrome reference: why does the render become a uniform filter?"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
import stealzalut.core.zone_transfer as zt
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.utils.image_io import bgr_to_rgb, create_thumbnail, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(PICS / "IMG_6295.DNG"))
ref = bgr_to_rgb(load_image_bgr(PICS / "841121141_18631647988026044_4768591964563047386_n.jpg"))
np.save(S / "purple_src.npy", create_thumbnail(src, 640))
np.save(S / "purple_ref.npy", create_thumbnail(ref, 640))
pack = Path(r"C:\Users\mathi\Downloads\IMG_6295_x_841121141_18631647988026044_4768591964563047386_n_stealzalut_pack")
analysis = json.loads((pack / "analysis.json").read_text(encoding="utf-8"))
k = analysis["zone_count"]
model = zt.fit_zone_transfer(src, ref, k)
print("K", k, "id", model.analysis_id, "pack id", analysis["analysis_id"])
print("Zones de la reference (hex, aire, chroma, teinte):")
for z in model.reference.zones:
    l, c, h = oklab_to_oklch(z.mean[None, :])[0]
    print(f"  R{z.id} {z.hex} {z.area*100:4.0f}%  L={l:.2f} C={c:.3f} h={h:5.1f}")


def hue_stats(img):
    lab = srgb_to_oklab(img.reshape(-1, 3) / 255.0)
    lch = oklab_to_oklch(lab)
    col = lch[:, 1] > 0.05
    h = np.radians(lch[col, 2])
    w = lch[col, 1]
    r = np.hypot((w * np.cos(h)).sum(), (w * np.sin(h)).sum()) / w.sum()
    return f"L={lab[:,0].mean():.3f} C={lch[:,1].mean():.3f} colores={col.mean()*100:.0f}% concentration teinte R={r:.2f}"


small = np.load(S / "purple_src.npy")
print("SOURCE    ", hue_stats(small))
print("REFERENCE ", hue_stats(np.load(S / "purple_ref.npy")))
img = apply_lut(small, build_grid(model, 33, GradingParams()).grid, 33)
print("AUTO zones", hue_stats(img))
np.save(S / "purple_auto.npy", img)
